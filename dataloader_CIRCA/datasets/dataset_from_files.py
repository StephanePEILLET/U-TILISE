import ast
import datetime as dt
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rasterio
import torch
from numpy import ndarray
from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.tools.mask_generation import masks_init_filling
from dataloader_CIRCA.tools.positional_encoding import get_position_for_positional_encoding

ChannelType = Literal["all", "bgr-nir"]


class Dataset_from_files(Dataset):
    """
    A custom PyTorch Dataset designed to manage Sentinel-1 and Sentinel-2 data specifically for cloud
    reconstruction tasks.
    """

    def __init__(
        self,
        data_optique: str | Path,
        data_radar: str | Path,
        mgrsc: str | None,
        image_size: tuple = (256, 256),
        overlap: int | None = 0,
        load_dataset: str | None = None,
        shuffle: bool = False,
        use_sar: bool = "mix_closest",
        no_filter: bool = False,
        channels: ChannelType = "all",
        pe_strategy: str = "day-within-sequence",
        fill_value: float = 1.0,
        mask_type: str = "orignal_masks",  # orignal_masks, fully_masked
        data_masks: str | Path | None = None,
        keep_all_dates: bool = False,  # Mode inférence : ne pas filtrer les dates nuageuses
    ):
        """
        Initializes the dataset.

        Parameters:
        - data_optique (Union[str, Path]): Path to the optical data directory.
        - data_radar (Union[str, Path]): Path to the radar data directory.
        - image_size (int, optional): Size of the patches to extract. Defaults to 256.
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        - shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        - use_sar (bool, optional): Whether to include SAR data in the dataset. Defaults to True.
        """
        self.data_optique = Path(data_optique)
        self.data_radar = Path(data_radar)
        self.mgrsc = mgrsc
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.overlap = overlap
        self.load_dataset = load_dataset
        self.shuffle = shuffle
        self.zones_dataset, self.dates_dict = None, None
        self.patches_dataset = None
        self.use_sar = use_sar
        self.no_filter = no_filter
        self.num_channels, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels(channels)
        self.s2_tile = None
        self.s1_asc_tile = None
        self.s1_desc_tile = None
        self.pe_strategy = pe_strategy
        self.fill_value = fill_value
        self.mask_type = mask_type
        self.data_masks = Path(data_masks) if data_masks is not None else None
        self.keep_all_dates = keep_all_dates
        self.setup()

    def __len__(self) -> int:
        """
        Returns the number of patches in the dataset.

        Returns:
        - int: Number of patches.
        """
        return len(self.mgrsc_dataset)

    def setup(self, load_dataset: str | None = None) -> None:
        """
        Sets up the dataset by either loading from a pre-saved file or processing the data.

        Parameters:
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        """
        if self.load_dataset is not None or load_dataset is not None:
            self.load_exported_data(load_dataset or self.load_dataset)
        else:
            self.setup_zones()
            self.setup_patches()

        if self.shuffle:
            self.patches_dataset = self.patches_dataset.sample(frac=1).reset_index(drop=True)

        self.setup_mgrsc_parameters()
        self.setup_mgrsc_files()
        if self.use_sar:
            self.setup_s1_dates()

    def setup_channels(self, channels: ChannelType) -> tuple[int, torch.Tensor, torch.Tensor, list[int]]:
        """
        Configure channel settings based on the specified channel mode.

        Args:
            channels: Channel configuration ('all' or 'bgr-nir')

        Returns:
            Tuple containing:
            - Number of channels
            - RGB channel indices tensor
            - NIR channel index tensor
            - List of Sentinel-2 channel indices

        Raises:
            ValueError: If invalid channel configuration is specified
        """
        if channels == "all":
            num_channels = 10
            c_index_rgb = torch.tensor([2, 1, 0], dtype=torch.long)
            c_index_nir = torch.tensor([6], dtype=torch.long)
            s2_channels = list(range(10))
        elif channels == "bgr-nir":
            num_channels = 4
            c_index_rgb = torch.tensor([2, 1, 0], dtype=torch.long)
            c_index_nir = torch.tensor([6], dtype=torch.long)
            s2_channels = [0, 1, 2, 6]
        else:
            raise ValueError(f"Channels {channels} not recognized. Use 'all' or 'bgr-nir'.")

        if self.use_sar:
            if self.use_sar == "asc+desc":
                num_channels += 8
            elif self.use_sar == "asc" or self.use_sar == "desc" or self.use_sar == "mix_closest":
                num_channels += 4
            else:
                raise ValueError(
                    f"SAR pairing {self.use_sar} not recognized. Use 'asc+desc', 'asc', 'desc' or 'mix_closest'."
                )
        return num_channels, c_index_rgb, c_index_nir, s2_channels

    def setup_mgrsc_parameters(self) -> None:
        """
        Sets up parameters specific to a given MGRS-C area.

        Parameters:
        - mgrsc (Optional[str]): The MGRS-C area identifier.
        """
        if self.mgrsc is None:
            raise ValueError("MGRS-C area must be specified for this operation.")
        # Implement any specific setup required for the given MGRS-C area
        self.mgrsc_dataset = (
            self.patches_dataset[self.patches_dataset["mgrs25"] == self.mgrsc].copy().reset_index(drop=True)
        )
        self.dates_s2 = np.array([self.str2date(date) for date in self.dates_dict[self.mgrsc]["S2"]])
        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        self.days = get_position_for_positional_encoding(self.dates_s2, "day-within-sequence")
        # Get positions for positional encoding
        self.position_days = get_position_for_positional_encoding(self.dates_s2, self.pe_strategy)

    def setup_mgrsc_files(self) -> None:
        # listage des fichiers à utiliser pour itérer sur les patchs contenus dans la zone MGRS-C
        files = set([el for sublist in self.mgrsc_dataset["files"].tolist() for el in sublist])
        if self.use_sar:
            assert len(files) == 3, "Expected exactly 3 files (S2, S1_ASC, S1_DESC) for the MGRS-C area."
        else:
            assert len(files) == 1, "Expected exactly 1 file (S2) for the MGRS-C area."
        self.s2_file = [f for f in files if "ASC" not in f and "DESC" not in f][0]
        if self.use_sar:
            self.s1_asc_file = [f for f in files if "ASC" in f][0]
            self.s1_desc_file = [f for f in files if "DESC" in f][0]
        # Setup mask file if data_masks directory is provided
        self.mask_file = None
        if self.data_masks is not None:
            mgrs_name = self.mgrsc_dataset.iloc[0]["mgrs"]
            mgrs25_name = self.mgrsc_dataset.iloc[0]["mgrs25"]
            mask_zone = self.data_masks / mgrs_name / ("MGRS25-" + mgrs25_name)
            mask_tifs = sorted(mask_zone.rglob("*.tif"))
            if mask_tifs:
                self.mask_file = mask_tifs[0].as_posix()
        with rasterio.open(self.s2_file) as src:
            self.s2_meta = src.meta.copy()
        if self.use_sar:
            with rasterio.open(self.s1_asc_file) as src:
                self.s1_asc_meta = src.meta.copy()
            with rasterio.open(self.s1_desc_file) as src:
                self.s1_desc_meta = src.meta.copy()

    def setup_s1_dates(self) -> None:
        self.dates_s1_asc = self.dates_dict[self.mgrsc]["S1"]["ASC"]
        self.dates_s1_desc = self.dates_dict[self.mgrsc]["S1"]["DESC"]
        self.closest_dates_matches = SentinelDataProcessor.get_pairedS1_closest_matches(
            dates_S2=self.dates_s2,
            dates_S1_asc=self.dates_s1_asc,
            dates_S1_desc=self.dates_s1_desc,
        )

    def load_exported_data(self, path_data: str | Path) -> None:
        """
        Loads the dataset from a pre-saved CSV file.

        Parameters:
        - path_data (Union[str, Path]): Path to the CSV file.
        """
        self.patches_dataset = pd.read_csv(path_data)
        cols_to_convert = [
            "window",
            "files",
            "dates_S2",
            "dates_S1_ASC",
            "dates_S1_DESC",
        ]
        self.patches_dataset[cols_to_convert] = self.patches_dataset[cols_to_convert].map(ast.literal_eval)

        if (
            self.patches_dataset.loc[0, "window"][2] != self.image_size[0]
            and self.patches_dataset.loc[0, "window"][3] != self.image_size[1]
        ):
            print(
                """
                WARNING : Patch size load in the .csv file is not corresponding to the patch size requested 
                during the dataset initialization. New setup of the dataset will be done.."
                """
            )
            self.setup()

        # Recreate the dates dictionary after loading
        self.dates_dict = {
            mgrs25: {
                "S2": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]["dates_S2"].values[0],
                "S1": {
                    "ASC": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]["dates_S1_ASC"].values[0],
                    "DESC": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]["dates_S1_DESC"].values[0],
                },
            }
            for mgrs25 in self.patches_dataset["mgrs25"].unique()
        }

    def setup_patches(self) -> None:
        """
        Sets up the patches by processing the data and creating a DataFrame.
        """
        self.patches_dataset = pd.DataFrame(columns=["patch", "window", "mgrs", "mgrs25", "files"]).astype(object)

        for _, row in tqdm(self.zones_dataset.iterrows(), total=len(self.zones_dataset), leave=False):
            for window in row.windows:
                windows_str = "_".join(map(str, window))
                patch_df = pd.DataFrame(
                    {
                        "patch": f"patches_{row.mgrs25}_window_{windows_str}",
                        "window": [window],
                        "mgrs": row.mgrs,
                        "mgrs25": row.mgrs25,
                        "files": [row.files],  # 0: S2, 1: S1_ASC, 2: S1_DESC
                        "dates_S2": [row.dates_S2],
                        "dates_S1_ASC": [row.dates_S1_ASC],
                        "dates_S1_DESC": [row.dates_S1_DESC],
                    }
                )
                self.patches_dataset = pd.concat([self.patches_dataset, patch_df], ignore_index=True)

    def setup_zones(self) -> None:
        """
        Sets up the zones by processing the data and creating a DataFrame.
        """
        self.zones_dataset = pd.DataFrame(columns=["mgrs", "mgrs25", "files", "windows"]).astype(object)
        self.dates_dict = {}

        for mgrs in tqdm(list(self.data_optique.iterdir()), leave=False, desc="mgrs"):
            for mgrs25 in tqdm(list(mgrs.iterdir()), leave=False, desc="mgrs25"):
                self.process_mgrs25_zone(mgrs, mgrs25)

    def process_mgrs25_zone(self, mgrs: Path, mgrs25: Path) -> None:
        """
        Processes a single MGRS25 zone.

        Parameters:
        - mgrs (Path): Path to the MGRS zone.
        - mgrs25 (Path): Path to the MGRS25 zone.
        """
        mgrs_name = mgrs.stem
        mgrs25_name = mgrs25.stem[7:]
        self.dates_dict[mgrs25_name] = {}

        # Process Sentinel-2 data
        list_tifs_optique = sorted(mgrs25.rglob("*.tif"))
        list_jsons_optique = sorted(mgrs25.rglob("*.json"))
        dates_S2 = json.load(open(list_jsons_optique[0]))
        self.dates_dict[mgrs25_name]["S2"] = dates_S2

        # Process Sentinel-1 data
        mgrs25_radar = self.data_radar / mgrs_name / ("MGRS25-" + mgrs25_name)
        assert mgrs25_radar.exists()

        list_tifs_radar = sorted(mgrs25_radar.rglob("*.tif"))
        list_jsons_radar = sorted(mgrs25_radar.rglob("*.json"))

        tif_files = {"S2": list_tifs_optique[0]}
        for file in list_tifs_radar:
            if file.stem.endswith("ASC"):
                tif_files["S1_ASC"] = file
            else:
                tif_files["S1_DESC"] = file

        list_windows = SentinelDataProcessor.split_raster_into_windows(
            path_raster=tif_files["S2"],
            image_size=self.image_size,
            overlap=self.overlap,
        )

        self.dates_dict[mgrs25_name]["S1"] = {
            file.stem.split("_")[-1]: json.load(open(file)) for file in list_jsons_radar
        }

        data = {
            "mgrs": mgrs_name,
            "mgrs25": mgrs25_name,  # Remove the 'MGRS25-' prefix
            "files": [sorted(f.as_posix() for f in tif_files.values())],
            "windows": [list_windows],
            "dates_S2": [dates_S2],
            "dates_S1_ASC": [self.dates_dict[mgrs25_name]["S1"]["ASC"]],
            "dates_S1_DESC": [self.dates_dict[mgrs25_name]["S1"]["DESC"]],
        }
        df_temp = pd.DataFrame(data).astype(object)
        self.zones_dataset = pd.concat([self.zones_dataset, df_temp], ignore_index=True)

    def export_dataset(self, outpath: str | Path = "tiles_windows.csv") -> None:
        """
        Exports the dataset to a CSV file.

        Parameters:
        - outpath (Union[str, Path], optional): Path to save the CSV file. Defaults to "datasetCIRCAUnCRtainTS.csv".
        """
        self.patches_dataset.to_csv(outpath, index=False)

    def str2date(self, date_string: str) -> dt.date:
        """
        Convert a date string in format 'YYYYMMDD' to datetime object.

        Args:
            date_string: Date string in format 'YYYYMMDD'

        Returns:
            Corresponding datetime.date object

        Example:
            >>> str2date("20200101")
            datetime.date(2020, 1, 1)
        """
        return dt.datetime.strptime(date_string, "%Y%m%d")

    def decode_dates(self, dates: ndarray[np.bytes_]) -> list[str]:
        """
        Decode byte strings in date array to UTF-8 strings.

        Args:
            dates: Array of date byte strings

        Returns:
            List of decoded date strings
        """
        return [el.decode("utf-8") for el in dates]

    def get_item_from_mgrsc(
        self,
        mgrsc: str,
        x: int,
        y: int,
        width: int,
        height: int,
        t_sampled: list | None = None,
        dates_to_mask: list[int] | None = None,
    ) -> dict[str, np.ndarray | str | list[str]]:
        """
        Retrieves cropped data for a given MGRSC zone using pixel coordinates.

        Parameters:
        - mgrsc (str): MGRS25 zone identifier (e.g. '31TCJ_25_100').
        - x (int): Column offset (pixel) for the crop window.
        - y (int): Row offset (pixel) for the crop window.
        - width (int): Width of the crop window in pixels.
        - height (int): Height of the crop window in pixels.
        - t_sampled (list | None): Optional list of temporal indices to sample.
        - dates_to_mask (list[int] | None): Optional list of temporal indices (0-based,
          within the loaded sequence) to fully mask. When provided, overrides mask_type:
          original cloud masks are applied, plus the specified dates are entirely replaced
          by fill_value. The model must reconstruct these dates.

        Returns:
        - Dict with the same structure as __getitem__.
        """
        mgrs_parts = mgrsc.split("_")
        mgrs_name = mgrs_parts[0]
        mgrs25_name = mgrsc

        mgrs25_optique_dir = self.data_optique / mgrs_name / ("MGRS25-" + mgrs25_name)
        mgrs25_radar_dir = self.data_radar / mgrs_name / ("MGRS25-" + mgrs25_name)

        if not mgrs25_optique_dir.exists():
            raise FileNotFoundError(f"Optical data directory not found: {mgrs25_optique_dir}")

        list_tifs_optique = sorted(mgrs25_optique_dir.rglob("*.tif"))
        list_jsons_optique = sorted(mgrs25_optique_dir.rglob("*.json"))
        s2_file = list_tifs_optique[0]
        dates_S2 = json.load(open(list_jsons_optique[0]))

        s1_asc_file, s1_desc_file = None, None
        dates_S1_asc, dates_S1_desc = None, None
        if self.use_sar:
            if not mgrs25_radar_dir.exists():
                raise FileNotFoundError(f"Radar data directory not found: {mgrs25_radar_dir}")
            list_tifs_radar = sorted(mgrs25_radar_dir.rglob("*.tif"))
            list_jsons_radar = sorted(mgrs25_radar_dir.rglob("*.json"))
            for file in list_tifs_radar:
                if file.stem.endswith("ASC"):
                    s1_asc_file = file
                else:
                    s1_desc_file = file
            for file in list_jsons_radar:
                orbit = file.stem.split("_")[-1]
                dates = json.load(open(file))
                if orbit == "ASC":
                    dates_S1_asc = dates
                else:
                    dates_S1_desc = dates

        crop_window = (x, y, width, height)

        S2_N_CHANNELS = 12
        S1_N_CHANNELS = 4

        if t_sampled is not None:
            bands_s2 = []
            for t in t_sampled:
                bands_s2.extend([t * S2_N_CHANNELS + c + 1 for c in range(S2_N_CHANNELS)])
            with rasterio.open(s2_file) as src_S2:
                patch_S2_array = src_S2.read(bands_s2, window=Window(*crop_window))
            patch_S2_array = patch_S2_array.reshape(
                len(t_sampled), S2_N_CHANNELS, patch_S2_array.shape[-2], patch_S2_array.shape[-1]
            )
        else:
            with rasterio.open(s2_file) as src_S2:
                patch_S2_array = src_S2.read(window=Window(*crop_window))
            T = patch_S2_array.shape[0] // S2_N_CHANNELS
            patch_S2_array = patch_S2_array.reshape(
                T, S2_N_CHANNELS, patch_S2_array.shape[-2], patch_S2_array.shape[-1]
            )

        patch_S2_array = torch.from_numpy(patch_S2_array.astype(np.float32))

        data_s2 = patch_S2_array[:, 0:10, ...]
        data_s2 = SentinelDataProcessor.process_MS(data_s2)

        if self.mask_file is not None:
            S2_N_CHANNELS_MASK = 12  # Assuming mask file has the same number of channels per time step as S2 data
            if t_sampled is not None:
                bands_mask = []
                for t in t_sampled:
                    bands_mask.extend([t * S2_N_CHANNELS_MASK + c + 1 for c in range(10, 12)])
                with rasterio.open(self.mask_file) as src_mask:
                    mask_array = src_mask.read(bands_mask, window=Window(*crop_window))
                mask_array = mask_array.reshape(len(t_sampled), 2, mask_array.shape[-2], mask_array.shape[-1])
            else:
                mask_bands = []
                T_mask = patch_S2_array.shape[0]
                for t in range(T_mask):
                    mask_bands.extend([t * S2_N_CHANNELS_MASK + c + 1 for c in range(10, 12)])
                with rasterio.open(self.mask_file) as src_mask:
                    mask_array = src_mask.read(mask_bands, window=Window(*crop_window))
                mask_array = mask_array.reshape(T_mask, 2, mask_array.shape[-2], mask_array.shape[-1])
            original_masks = torch.from_numpy(mask_array.astype(np.float32))
        else:
            original_masks = patch_S2_array[:, 10:, ...]
        # Est-ce que ce n'est pas 1 plutôt que 0 pour les masques de nuages ? A vérifier dans les données
        cloud_probs = original_masks[:, 0, ...].clone().unsqueeze(axis=1)
        missing_data_mask = torch.all(data_s2 == 0, dim=1, keepdim=True)
        cloud_probs[missing_data_mask] = 1.0
        cloud_masks = (cloud_probs > 0).float()

        dates_s2_list = [self.str2date(d) for d in dates_S2]
        dates_s2_np = np.array(dates_s2_list)
        days = get_position_for_positional_encoding(dates_s2_np, "day-within-sequence")
        position_days = get_position_for_positional_encoding(dates_s2_np, self.pe_strategy)

        data_s1 = None
        dates_s1_sampled = None
        closest_matches = None
        if self.use_sar:
            closest_matches = SentinelDataProcessor.get_pairedS1_closest_matches(
                dates_S2=dates_S2,
                dates_S1_asc=dates_S1_asc,
                dates_S1_desc=dates_S1_desc,
            )

            s1_tile, s1_dates = [], []
            indices_s2 = t_sampled if t_sampled is not None else range(len(closest_matches))

            asc_bands, desc_bands = [], []
            asc_indices_map, desc_indices_map = {}, {}

            for index_s2 in indices_s2:
                _, date_s1, index_s1, orbit_type = closest_matches[index_s2]
                if orbit_type == "ASC":
                    if index_s1 not in asc_indices_map:
                        asc_indices_map[index_s1] = True
                        asc_bands.extend([index_s1 * S1_N_CHANNELS + c + 1 for c in range(S1_N_CHANNELS)])
                elif index_s1 not in desc_indices_map:
                    desc_indices_map[index_s1] = True
                    desc_bands.extend([index_s1 * S1_N_CHANNELS + c + 1 for c in range(S1_N_CHANNELS)])

            asc_bands = sorted(set(asc_bands))
            desc_bands = sorted(set(desc_bands))

            s1_tile_asc_dict = {}
            if len(asc_bands) > 0:
                with rasterio.open(s1_asc_file) as src_s1:
                    asc_data = src_s1.read(asc_bands, window=Window(*crop_window))
                    for i, band in enumerate(asc_bands):
                        idx = (band - 1) // S1_N_CHANNELS
                        c = (band - 1) % S1_N_CHANNELS
                        if idx not in s1_tile_asc_dict:
                            s1_tile_asc_dict[idx] = np.zeros(
                                (S1_N_CHANNELS, asc_data.shape[-2], asc_data.shape[-1]), dtype=asc_data.dtype
                            )
                        s1_tile_asc_dict[idx][c] = asc_data[i]

            s1_tile_desc_dict = {}
            if len(desc_bands) > 0:
                with rasterio.open(s1_desc_file) as src_s1:
                    desc_data = src_s1.read(desc_bands, window=Window(*crop_window))
                    for i, band in enumerate(desc_bands):
                        idx = (band - 1) // S1_N_CHANNELS
                        c = (band - 1) % S1_N_CHANNELS
                        if idx not in s1_tile_desc_dict:
                            s1_tile_desc_dict[idx] = np.zeros(
                                (S1_N_CHANNELS, desc_data.shape[-2], desc_data.shape[-1]), dtype=desc_data.dtype
                            )
                        s1_tile_desc_dict[idx][c] = desc_data[i]

            for index_s2 in indices_s2:
                _, date_s1, index_s1, orbit_type = closest_matches[index_s2]
                s1_band = s1_tile_asc_dict[index_s1] if orbit_type == "ASC" else s1_tile_desc_dict[index_s1]
                s1_tile.append(s1_band)
                s1_dates.append(date_s1)

            s1_tile = np.stack(s1_tile, axis=0)
            s1_tile = torch.from_numpy(s1_tile.astype(np.float32))
            dates_s1_sampled = np.array([self.str2date(date) for date in s1_dates])
            data_s1 = SentinelDataProcessor.process_SAR(s1_tile)

        idx_kept = None

        if dates_to_mask is not None:
            # Manual date masking: filter out cloudy dates, keep only clean ones,
            # then fully mask the user-specified dates (like fully_masked mode but
            # with manually chosen dates instead of pre-computed mask files).
            cloud_probs_manual = original_masks[:, 0, ...].clone().unsqueeze(axis=1)
            snow_probs_manual = original_masks[:, 1, ...].clone().unsqueeze(axis=1)
            masks_to_filter = np.concatenate([snow_probs_manual.numpy(), cloud_probs_manual.numpy()], axis=1)
            masks_to_filter = masks_to_filter.transpose(0, 2, 3, 1)
            idx_clean = SentinelDataProcessor.filter_dates(masks_to_filter)

            # Keep clean dates + ensure dates_to_mask are included
            idx_kept = np.sort(np.union1d(idx_clean, dates_to_mask))

            data_s2 = data_s2[idx_kept]
            original_masks = original_masks[idx_kept]
            cloud_masks = cloud_masks[idx_kept]
            if self.use_sar:
                data_s1 = data_s1[idx_kept]
                dates_s1_sampled = dates_s1_sampled[idx_kept]

            # Clean dates: no masking. Only dates_to_mask are fully masked.
            images_masked = data_s2.clone()
            masks = torch.zeros(data_s2.shape[0], 1, data_s2.shape[2], data_s2.shape[3])
            is_to_mask = np.isin(idx_kept, dates_to_mask)
            for i in np.where(is_to_mask)[0]:
                images_masked[i] = self.fill_value
                masks[i] = self.fill_value
        elif self.mask_type == "fully_masked":
            cloud_probs_fm = original_masks[:, 0, ...].clone().unsqueeze(axis=1)
            snow_probs = original_masks[:, 1, ...].clone().unsqueeze(axis=1)

            if self.mask_file is not None:
                SYNTHETIC_THRESHOLD = 100
                is_synthetic = cloud_probs_fm.squeeze(1).mean(dim=(1, 2)) > SYNTHETIC_THRESHOLD
                idx_synthetic = np.where(is_synthetic.numpy())[0]

                cloud_orig = cloud_probs_fm.clone()
                snow_orig = snow_probs.clone()
                cloud_orig[is_synthetic] -= 150
                snow_orig[is_synthetic] -= 150
                cloud_orig.clamp_(min=0)
                snow_orig.clamp_(min=0)

                if not self.keep_all_dates:
                    masks_to_filter = np.concatenate([snow_orig.numpy(), cloud_orig.numpy()], axis=1)
                    masks_to_filter = masks_to_filter.transpose(0, 2, 3, 1)
                    idx_clean = SentinelDataProcessor.filter_dates(masks_to_filter)
                    idx_kept = np.sort(np.union1d(idx_clean, idx_synthetic))

                    data_s2 = data_s2[idx_kept]
                    original_masks = original_masks[idx_kept]
                    cloud_masks = cloud_masks[idx_kept]
                    if self.use_sar:
                        data_s1 = data_s1[idx_kept]
                        dates_s1_sampled = dates_s1_sampled[idx_kept]

                    is_synth_in_kept = np.isin(idx_kept, idx_synthetic)
                    images_masked = data_s2.clone()
                    masks = torch.zeros(data_s2.shape[0], 1, data_s2.shape[2], data_s2.shape[3])
                    for i in np.where(is_synth_in_kept)[0]:
                        images_masked[i] = self.fill_value
                        masks[i] = self.fill_value
                else:
                    images_masked, masks = masks_init_filling(
                        seq=data_s2.clone(),
                        masks=cloud_masks.clone(),
                        fill_type="fill_value",
                        fill_value=self.fill_value,
                        dilate_cloud_masks=False,
                    )
                    for i in idx_synthetic:
                        images_masked[i] = self.fill_value
                        masks[i] = self.fill_value
            else:
                images_masked, masks = masks_init_filling(
                    seq=data_s2.clone(),
                    masks=cloud_masks.clone(),
                    fill_type="fill_value",
                    fill_value=self.fill_value,
                    dilate_cloud_masks=False,
                )
        else:
            images_masked, masks = masks_init_filling(
                seq=data_s2.clone(),
                masks=cloud_masks.clone(),
                fill_type="fill_value",
                fill_value=self.fill_value,
                dilate_cloud_masks=False,
            )

        frames_input = torch.cat((images_masked, data_s1), dim=1) if self.use_sar else images_masked

        if self.use_sar and masks is not None:
            frames_input = frames_input.masked_fill(masks == 1.0, self.fill_value)

        frames_target = data_s2.clone()
        masks_valid_obs = torch.ones(frames_input.shape[0], dtype=torch.uint8)

        if idx_kept is not None:
            t_effective = idx_kept if t_sampled is None else np.array([t_sampled[i] for i in idx_kept])
        elif t_sampled is not None:
            t_effective = np.array(t_sampled)
        else:
            t_effective = None

        if t_effective is not None:
            out = {
                "x": frames_input,
                "y": frames_target,
                "masks": masks,
                "masks_valid_obs": masks_valid_obs,
                "position_days": position_days[t_effective],
                "days": days[t_effective] - days[t_effective][0],
                "sample_index": 0,
                "c_index_rgb": self.c_index_rgb,
                "c_index_nir": self.c_index_nir,
                "S2_dates": [dates_s2_np[i].strftime("%Y-%m-%d") for i in t_effective],
                "original_masks": original_masks,
                "cloud_mask": cloud_masks,
                "window": crop_window,
            }
            if self.use_sar:
                out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in dates_s1_sampled]
        else:
            out = {
                "x": frames_input,
                "y": frames_target,
                "masks": masks,
                "masks_valid_obs": masks_valid_obs,
                "position_days": position_days,
                "days": days - days[0],
                "sample_index": 0,
                "c_index_rgb": self.c_index_rgb,
                "c_index_nir": self.c_index_nir,
                "S2_dates": [date.strftime("%Y-%m-%d") for date in dates_s2_np],
                "original_masks": original_masks,
                "cloud_mask": cloud_masks,
                "window": crop_window,
            }
            if self.use_sar:
                out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in dates_s1_sampled]
        return out

    def __getitem__(self, item: int, t_sampled: list | None = None) -> dict[str, np.ndarray | str | list[str]]:
        """
        Retrieves an item from the dataset.

        Parameters:
        - item (int): Index of the item to retrieve.

        Returns:
        - Dict[str, Union[np.ndarray, str, List[str]]]: A dictionary containing the data, name, masks, and dates.
        """
        patch_data = self.mgrsc_dataset.iloc[item]

        # Determine the number of channels per chunk from SentinelDataProcessor constants or fallback
        S2_N_CHANNELS = 12
        S1_N_CHANNELS = 4

        # Extraction données S2
        if t_sampled is not None:
            bands_s2 = []
            for t in t_sampled:
                bands_s2.extend([t * S2_N_CHANNELS + c + 1 for c in range(S2_N_CHANNELS)])
            with rasterio.open(self.s2_file) as src_S2:
                patch_S2_array = src_S2.read(bands_s2, window=Window(*patch_data.window))
            patch_S2_array = patch_S2_array.reshape(len(t_sampled), S2_N_CHANNELS, patch_S2_array.shape[-2], patch_S2_array.shape[-1])
        else:
            with rasterio.open(self.s2_file) as src_S2:
                patch_S2_array = src_S2.read(window=Window(*patch_data.window))
            T = patch_S2_array.shape[0] // S2_N_CHANNELS
            patch_S2_array = patch_S2_array.reshape(T, S2_N_CHANNELS, patch_S2_array.shape[-2], patch_S2_array.shape[-1])

        patch_S2_array = torch.from_numpy(patch_S2_array.astype(np.float32))

        # Extraction données S2 (10 bandes spectrales)
        data_s2 = patch_S2_array[:, 0:10, ...]
        data_s2 = SentinelDataProcessor.process_MS(data_s2)

        # Récupération des masques : depuis data_masks si fourni, sinon depuis le raster S2 optique
        if self.mask_file is not None:
            S2_N_CHANNELS_MASK = 12
            if t_sampled is not None:
                bands_mask = []
                for t in t_sampled:
                    bands_mask.extend([t * S2_N_CHANNELS_MASK + c + 1 for c in range(10, 12)])
                with rasterio.open(self.mask_file) as src_mask:
                    mask_array = src_mask.read(bands_mask, window=Window(*patch_data.window))
                mask_array = mask_array.reshape(len(t_sampled), 2, mask_array.shape[-2], mask_array.shape[-1])
            else:
                mask_bands = []
                T_mask = patch_S2_array.shape[0]
                for t in range(T_mask):
                    mask_bands.extend([t * S2_N_CHANNELS_MASK + c + 1 for c in range(10, 12)])
                with rasterio.open(self.mask_file) as src_mask:
                    mask_array = src_mask.read(mask_bands, window=Window(*patch_data.window))
                mask_array = mask_array.reshape(T_mask, 2, mask_array.shape[-2], mask_array.shape[-1])
            original_masks = torch.from_numpy(mask_array.astype(np.float32))
        else:
            original_masks = patch_S2_array[:, 10:, ...]
        cloud_probs = original_masks[:, 0, ...].clone().unsqueeze(axis=1)

        # FIX: Identifier les pixels sans données (No Data = 0 sur tous les canaux S2) et les ajouter au masque de nuages / à reconstruire
        missing_data_mask = torch.all(data_s2 == 0, dim=1, keepdim=True)
        cloud_probs[missing_data_mask] = 1.0

        cloud_masks = (cloud_probs > 0).float()  # Binarization of cloud masks

        if self.use_sar:
            s1_tile, s1_dates = [], []
            indices_s2 = t_sampled if t_sampled is not None else range(len(self.closest_dates_matches))

            asc_bands, desc_bands = [], []
            asc_indices_map, desc_indices_map = {}, {}

            for index_s2 in indices_s2:
                _, date_s1, index_s1, orbit_type = self.closest_dates_matches[index_s2]
                if orbit_type == "ASC":
                    if index_s1 not in asc_indices_map:
                        asc_indices_map[index_s1] = True
                        asc_bands.extend([index_s1 * S1_N_CHANNELS + c + 1 for c in range(S1_N_CHANNELS)])
                elif index_s1 not in desc_indices_map:
                    desc_indices_map[index_s1] = True
                    desc_bands.extend([index_s1 * S1_N_CHANNELS + c + 1 for c in range(S1_N_CHANNELS)])

            asc_bands = sorted(set(asc_bands))
            desc_bands = sorted(set(desc_bands))

            s1_tile_asc_dict = {}
            if len(asc_bands) > 0:
                with rasterio.open(self.s1_asc_file) as src_s1:
                    asc_data = src_s1.read(asc_bands, window=Window(*patch_data.window))
                    for i, band in enumerate(asc_bands):
                        idx = (band - 1) // S1_N_CHANNELS
                        c = (band - 1) % S1_N_CHANNELS
                        if idx not in s1_tile_asc_dict:
                            s1_tile_asc_dict[idx] = np.zeros((S1_N_CHANNELS, asc_data.shape[-2], asc_data.shape[-1]), dtype=asc_data.dtype)
                        s1_tile_asc_dict[idx][c] = asc_data[i]

            s1_tile_desc_dict = {}
            if len(desc_bands) > 0:
                with rasterio.open(self.s1_desc_file) as src_s1:
                    desc_data = src_s1.read(desc_bands, window=Window(*patch_data.window))
                    for i, band in enumerate(desc_bands):
                        idx = (band - 1) // S1_N_CHANNELS
                        c = (band - 1) % S1_N_CHANNELS
                        if idx not in s1_tile_desc_dict:
                            s1_tile_desc_dict[idx] = np.zeros((S1_N_CHANNELS, desc_data.shape[-2], desc_data.shape[-1]), dtype=desc_data.dtype)
                        s1_tile_desc_dict[idx][c] = desc_data[i]

            for index_s2 in indices_s2:
                _, date_s1, index_s1, orbit_type = self.closest_dates_matches[index_s2]
                s1_band = s1_tile_asc_dict[index_s1] if orbit_type == "ASC" else s1_tile_desc_dict[index_s1]
                s1_tile.append(s1_band)
                s1_dates.append(date_s1)

            s1_tile = np.stack(s1_tile, axis=0)
            s1_tile = torch.from_numpy(s1_tile.astype(np.float32))
            dates_s1_sampled = np.array([self.str2date(date) for date in s1_dates])
            data_s1 = SentinelDataProcessor.process_SAR(s1_tile)

        # Variable pour tracker la sous-sélection temporelle (masques synthétiques)
        idx_kept = None

        if self.mask_type == "fully_masked":
            cloud_probs = original_masks[:, 0, ...].clone().unsqueeze(axis=1)
            snow_probs = original_masks[:, 1, ...].clone().unsqueeze(axis=1)

            if self.mask_file is not None:
                # === Masques synthétiques : convention +150 ===
                # Bande cloud : 0-100 = proba originale, 150-250 = date masquée synthétiquement
                # On doit :
                #   1. Identifier les dates synthétiques (+150)
                #   2. Retrouver les probas originales (retirer +150) pour filter_dates
                #   3. Exclure les dates réellement nuageuses (> 5% couverture) [sauf keep_all_dates]
                #   4. Garder les dates clean (contexte) + synthétiques (à reconstruire)
                SYNTHETIC_THRESHOLD = 100

                # 1. Identifier dates synthétiquement masquées (moyenne cloud > 100 par date)
                is_synthetic = cloud_probs.squeeze(1).mean(dim=(1, 2)) > SYNTHETIC_THRESHOLD  # [T]
                idx_synthetic = np.where(is_synthetic.numpy())[0]

                # 2. Retrouver les probabilités originales en retirant le +150
                cloud_orig = cloud_probs.clone()
                snow_orig = snow_probs.clone()
                cloud_orig[is_synthetic] -= 150
                snow_orig[is_synthetic] -= 150
                cloud_orig.clamp_(min=0)
                snow_orig.clamp_(min=0)

                if not self.keep_all_dates:
                    # === Mode évaluation : filtrer les dates réellement nuageuses ===
                    # 3. filter_dates sur probas originales → indices des dates clean
                    masks_to_filter = np.concatenate([snow_orig.numpy(), cloud_orig.numpy()], axis=1)
                    masks_to_filter = masks_to_filter.transpose(0, 2, 3, 1)  # T x H x W x 2
                    idx_clean = SentinelDataProcessor.filter_dates(masks_to_filter)

                    # 4. Garder = dates clean (contexte) ∪ dates synthétiques (à reconstruire)
                    #    Exclues = dates réellement nuageuses (ni clean ni synthétiques)
                    idx_kept = np.sort(np.union1d(idx_clean, idx_synthetic))

                    # 5. Sous-sélection de toutes les données temporelles
                    data_s2 = data_s2[idx_kept]
                    original_masks = original_masks[idx_kept]
                    cloud_masks = cloud_masks[idx_kept]
                    if self.use_sar:
                        data_s1 = data_s1[idx_kept]
                        dates_s1_sampled = dates_s1_sampled[idx_kept]

                    # 6. Parmi les dates gardées, masquer les synthétiques en input
                    is_synth_in_kept = np.isin(idx_kept, idx_synthetic)
                    images_masked = data_s2.clone()
                    masks = torch.zeros(data_s2.shape[0], 1, data_s2.shape[2], data_s2.shape[3])
                    for i in np.where(is_synth_in_kept)[0]:
                        images_masked[i] = self.fill_value
                        masks[i] = self.fill_value
                else:
                    # === Mode inférence (keep_all_dates) : garder TOUTES les dates ===
                    # On ne filtre pas les dates réellement nuageuses pour que T_output == T_original.
                    # Masquer les dates synthétiques + les dates nuageuses originales en input.
                    images_masked, masks = masks_init_filling(
                        seq=data_s2.clone(),
                        masks=cloud_masks.clone(),
                        fill_type="fill_value",
                        fill_value=self.fill_value,
                        dilate_cloud_masks=False,
                    )
                    # En plus du masquage par nuages originaux, masquer entièrement les dates synthétiques
                    for i in idx_synthetic:
                        images_masked[i] = self.fill_value
                        masks[i] = self.fill_value
            else:
                # Mode produit (pas de données externes) : pas de filtrage,
                # on garde toutes les dates y compris nuageuses.
                # Les masques de nuages originaux sont appliqués en input (pixels nuageux → fill_value).
                images_masked, masks = masks_init_filling(
                    seq=data_s2.clone(),
                    masks=cloud_masks.clone(),
                    fill_type="fill_value",
                    fill_value=self.fill_value,
                    dilate_cloud_masks=False,
                )
        else:
            # Ajout des masques de nuages originaux dans l'input
            # Image time series with overlaid cloud masks filled with value `fill_value`
            images_masked, masks = masks_init_filling(
                seq=data_s2.clone(),
                masks=cloud_masks.clone(),
                fill_type="fill_value",
                fill_value=self.fill_value,
                dilate_cloud_masks=False,
            )

        frames_input = torch.cat((images_masked, data_s1), dim=1) if self.use_sar else images_masked

        # LEGACY (ALL_SAR) : masque aussi les canaux SAR quand des nuages sont présents.
        # Ce comportement reproduit l'erreur du dataloader d'entraînement du modèle ALL_SAR.
        # Pour les nouveaux modèles (comportement correct), le SAR ne doit PAS être masqué.
        # TODO: ajouter un flag mask_sar pour contrôler ce comportement (cf. UTILISE_adapter.py)
        if self.use_sar and masks is not None:
            frames_input = frames_input.masked_fill(masks == 1.0, self.fill_value)

        frames_target = data_s2.clone()

        masks_valid_obs = torch.ones(frames_input.shape[0], dtype=torch.uint8)

        # Déterminer les indices effectifs pour l'output
        # idx_kept : sous-sélection issue des masques synthétiques (peut être None)
        # t_sampled : sous-sélection externe (peut être None)
        if idx_kept is not None:
            # Masques synthétiques : on a sous-sélectionné les dates
            # t_effective contient les indices dans la série originale (self.dates_s2, etc.)
            t_effective = idx_kept if t_sampled is None else np.array([t_sampled[i] for i in idx_kept])
        elif t_sampled is not None:
            t_effective = np.array(t_sampled)
        else:
            t_effective = None

        if t_effective is not None:
            out = {
                "x": frames_input,
                "y": frames_target,
                "masks": masks,
                "masks_valid_obs": masks_valid_obs,
                "position_days": self.position_days[t_effective],
                "days": self.days[t_effective] - self.days[t_effective][0],
                "sample_index": item,
                "c_index_rgb": self.c_index_rgb,
                "c_index_nir": self.c_index_nir,
                "S2_dates": [self.dates_s2[i].strftime("%Y-%m-%d") for i in t_effective],
                "original_masks": original_masks,
                "cloud_mask": cloud_masks,
                "window": patch_data["window"],
            }
            if self.use_sar:
                out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in dates_s1_sampled]
        else:
            out = {
                "x": frames_input,
                "y": frames_target,
                "masks": masks,
                "masks_valid_obs": masks_valid_obs,
                "position_days": self.position_days,
                "days": self.days - self.days[0],
                "sample_index": item,
                "c_index_rgb": self.c_index_rgb,
                "c_index_nir": self.c_index_nir,
                "S2_dates": [date.strftime("%Y-%m-%d") for date in self.dates_s2],
                "original_masks": original_masks,
                "cloud_mask": cloud_masks,
                "window": patch_data["window"],
            }
            if self.use_sar:
                out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in dates_s1_sampled]

        if self.keep_all_dates is False:
            out["t_effective"] = t_effective  # Indices des dates gardées (clean + synthétiques) dans la série temporelle filtrée
            out["idx_kept"] = idx_kept  # Indices des dates gardées (clean + synthétiques) dans la série originale
            out["full_s2"] = patch_S2_array  # Raw S2 data (T, 12, h, w) — NOT normalized, for direct writing back to raster
        return out
