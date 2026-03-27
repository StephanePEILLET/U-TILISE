"""Utility to generate binary parcel masks from a GPKG file for each evaluation patch.

Given a GeoPackage containing parcel geometries and the patch metadata (CRS, transform, size),
this module rasterizes the parcels that intersect each patch into a binary mask (1 = parcel pixel).

Uses fiona + shapely + pyproj directly (no geopandas dependency).
"""

import json
from pathlib import Path

import fiona
import numpy as np
import torch
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely import STRtree
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid


class ParcelMaskGenerator:
    """Generates binary raster masks indicating parcel pixels for each evaluation patch.

    Loads a GPKG file containing parcel geometries, builds a spatial index,
    and for each patch (identified by mgrs25 + window), produces a 256x256 binary mask.
    Results are cached to avoid recomputation.
    """

    def __init__(self, gpkg_path: str, patches_json_path: str):
        self.gpkg_path = Path(gpkg_path)
        self.patches_json_path = Path(patches_json_path)

        # Load patch metadata from JSON
        with open(self.patches_json_path, encoding="utf-8") as f:
            raw = json.load(f)
        self._build_patch_meta_index(raw)

        # Load parcel geometries (EPSG:2154) using fiona
        print(f"Loading parcel geometries from {self.gpkg_path}...")
        self.parcels_geoms: list = []
        with fiona.open(str(self.gpkg_path)) as src:
            self.parcels_src_crs = CRS.from_user_input(src.crs)
            for feat in src:
                geom = shape(feat["geometry"])
                if geom is not None and not geom.is_empty:
                    self.parcels_geoms.append(geom)
        print(f"  Loaded {len(self.parcels_geoms)} parcels in CRS {self.parcels_src_crs}")

        # Build spatial index on source geometries
        self._src_strtree = STRtree(self.parcels_geoms)

        # Cache reprojected geometries + STRtree per target CRS
        self._reprojected: dict[str, tuple[list, STRtree]] = {}

        # Cache for generated masks: key = (mgrs25, window_str)
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _build_patch_meta_index(self, raw: dict):
        """Build a lookup: (mgrs25, window_str) -> {crs, transform, height, width}."""
        self._patch_meta = {}
        n_patches = len(raw["patch"])
        for i in range(n_patches):
            idx = str(i)
            mgrs25 = raw["mgrs25"][idx]
            window = raw["window"][idx]  # list [col_off, row_off, h, w]
            window_str = "_".join(map(str, window))
            meta = raw["meta"][idx]
            self._patch_meta[mgrs25, window_str] = {
                "crs": meta["crs"],
                "transform": meta["transform"],  # [a, b, c, d, e, f, 0, 0, 1]
                "height": window[2],
                "width": window[3],
            }

    def _get_parcels_in_crs(self, target_crs_wkt: str) -> tuple[list, STRtree]:
        """Get parcels reprojected to the target CRS, with caching per CRS."""
        try:
            crs_obj = CRS.from_wkt(target_crs_wkt)
            crs_key = str(crs_obj.to_epsg()) if crs_obj.to_epsg() else target_crs_wkt[:80]
        except Exception:
            crs_key = target_crs_wkt[:80]

        if crs_key not in self._reprojected:
            print(f"  Reprojecting parcels to CRS {crs_key}...")
            target_crs = CRS.from_wkt(target_crs_wkt)
            transformer = Transformer.from_crs(self.parcels_src_crs, target_crs, always_xy=True)
            proj_fn = transformer.transform
            reprojected = []
            for geom in self.parcels_geoms:
                try:
                    rg = shapely_transform(proj_fn, geom)
                    if rg is not None and not rg.is_empty:
                        reprojected.append(rg)
                except Exception:
                    continue
            tree = STRtree(reprojected)
            self._reprojected[crs_key] = (reprojected, tree)
            print(f"  Reprojected {len(reprojected)} parcels.")
        return self._reprojected[crs_key]

    def _patch_bbox_and_transform(self, meta: dict) -> tuple:
        """Compute the patch bounding box and affine transform from metadata."""
        t = meta["transform"]
        transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        h, w = meta["height"], meta["width"]
        left, top = t[2], t[5]
        right = left + w * t[0]
        bottom = top + h * t[4]  # t[4] is negative
        patch_bbox = box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        return transform, h, w, patch_bbox

    def _rasterize_parcels(self, parcels: list, strtree: STRtree, patch_bbox, transform, h: int, w: int) -> np.ndarray:
        """Rasterize parcel geometries intersecting the patch bbox."""
        candidate_idxs = strtree.query(patch_bbox, predicate="intersects")

        if len(candidate_idxs) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        shapes = []
        for idx in candidate_idxs:
            geom = parcels[idx]
            try:
                clipped = geom.intersection(patch_bbox)
            except Exception:
                clipped = make_valid(geom).intersection(patch_bbox)
            if clipped is not None and not clipped.is_empty:
                shapes.append((clipped, 1))

        if not shapes:
            return np.zeros((h, w), dtype=np.uint8)

        return rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)

    def get_mask(self, mgrs25: str, window_str: str) -> torch.Tensor:
        """Generate or retrieve a cached binary parcel mask for a given patch.

        Args:
            mgrs25: Tile identifier (e.g., '31TGJ_row-2_col-2').
            window_str: Window string (e.g., '0_0_256_256').

        Returns:
            torch.Tensor of shape (1, 1, H, W) with 1 for parcel pixels, 0 otherwise.
        """
        cache_key = (mgrs25, window_str)
        if cache_key in self._cache:
            return self._cache[cache_key]

        meta = self._patch_meta.get(cache_key)
        if meta is None:
            raise KeyError(
                f"No metadata found for patch ({mgrs25}, {window_str}). "
                f"Available keys sample: {list(self._patch_meta.keys())[:5]}"
            )

        transform, h, w, patch_bbox = self._patch_bbox_and_transform(meta)
        parcels, strtree = self._get_parcels_in_crs(meta["crs"])
        mask = self._rasterize_parcels(parcels, strtree, patch_bbox, transform, h, w)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        self._cache[cache_key] = mask_tensor
        return mask_tensor
