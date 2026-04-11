"""
Version parallèle de metriques_claude.py.

Architecture :
  - ProcessPoolExecutor : chaque worker traite une tuile complète (lecture raster +
    filtrage numpy) et retourne les tableaux filtrés prêts pour l'agrégation.
  - Process principal : appelle agg.update() de façon séquentielle sur les résultats
    (les accumulateurs CloudRemovalDatasetMetrics ne sont pas thread/process-safe).

Pourquoi ProcessPoolExecutor plutôt que ThreadPoolExecutor :
  - Les opérations numpy (reshape, where, indexing) ne libèrent pas systématiquement
    le GIL → les threads seraient limités au GIL.
  - rasterio libère bien le GIL pour les appels GDAL, mais l'overhead numpy justifie
    quand même le multiprocessing.
  - Pas de problème de sérialisation : Path et numpy arrays sont picklables.
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from cr_torchmetrics import CloudRemovalDatasetMetrics
from torch import Tensor
from tqdm import tqdm

PATCH_SIZE = 256
INDICES_4 = np.array([0, 1, 2, 7])

store_dai = Path("/mnt/stores/store_dai")
# store_dai = Path("/mnt/stores/store-DAI") # jzellou
path_input = store_dai / "tmp/speillet/inferences/v3_combined/consecutive_fully_masked/2026-03-26_16-17"
# path_input = store_dai / "tmp/speillet/inferences/v3_combined/random_fully_masked/2026-03-26_16-17"
gt_path_dir = store_dai / "projets/pac/3str/EXP_2/Data_Raster/test_v3/consecutif"
# gt_path_dir = store_dai / "projets/pac/3str/EXP_2/Data_Raster/test_v3/aleatoire"


def _filter_patch(tile_inference_raw: np.ndarray, tile_gt_raw: np.ndarray):
    """
    Filtre un patch brut pour ne garder que les dates synthétiquement masquées et
    originalement sans nuages.

    Retourne (tile_gt, tile_inference, mask_agg) en float32, ou None si aucune
    date valide n'est trouvée.
    """
    h, w = tile_inference_raw.shape[-2], tile_inference_raw.shape[-1]
    tile_inference = tile_inference_raw.reshape((-1, 12, h, w))
    tile_gt = tile_gt_raw.reshape((-1, 12, h, w))

    clouds_image_gt = tile_gt[:, 10]

    indices_faux_nuages = np.unique(np.where(clouds_image_gt > 100)[0])
    if len(indices_faux_nuages) == 0:
        return None

    tile_inference = tile_inference[indices_faux_nuages]
    tile_gt = tile_gt[indices_faux_nuages]
    clouds_image_gt = clouds_image_gt[indices_faux_nuages]

    maximum = np.max(clouds_image_gt, axis=(1, 2))
    indices_vrai_faux_nuages = np.where(maximum == 150)[0]
    if len(indices_vrai_faux_nuages) == 0:
        return None

    tile_inference = tile_inference[indices_vrai_faux_nuages, :10].astype(np.float32)
    tile_gt = tile_gt[indices_vrai_faux_nuages, :10]
    tile_gt = np.where(tile_gt == 0, 0, tile_gt - 1000).astype(np.float32)

    tile_gt = np.expand_dims(tile_gt, 0)          # (1, T, 10, H, W)
    tile_inference = np.expand_dims(tile_inference, 0)  # (1, T, 10, H, W)
    mask_agg = np.where(tile_gt != 0, 0, 1).astype(np.float32)
    mask_agg = mask_agg[:, :, 0, :, :]            # (1, T, H, W)

    return tile_gt, tile_inference, mask_agg


def process_tile(tif_file: str, path_input: Path, gt_path_dir: Path) -> list:
    """
    Worker : lit tous les patches d'une tuile, filtre, retourne la liste des
    (tile_gt, tile_inference, mask_agg) valides.

    Fonction au niveau module pour être picklable par ProcessPoolExecutor.
    """
    tile = tif_file.split("_")[2]
    subtile = "MGRS25-" + "_".join(tif_file.replace(".tif", "").split("_")[2:5])
    gt_file = tif_file.replace("pred_mgrsc", "bands_stacked")
    gt_path = gt_path_dir / tile / subtile / gt_file

    results = []
    with rasterio.open(path_input / tif_file) as src_inferences, \
         rasterio.open(gt_path) as src_gt:

        n_cols = math.ceil(src_inferences.width / PATCH_SIZE)
        n_rows = math.ceil(src_inferences.height / PATCH_SIZE)

        for i in range(n_cols):
            for j in range(n_rows):
                w = min(PATCH_SIZE, src_inferences.width - i * PATCH_SIZE)
                h = min(PATCH_SIZE, src_inferences.height - j * PATCH_SIZE)
                window = rasterio.windows.Window(i * PATCH_SIZE, j * PATCH_SIZE, w, h)
                try:
                    patch_inference = src_inferences.read(window=window)
                except Exception:
                    time.sleep(30)
                    patch_inference = src_inferences.read(window=window)
                patch_gt = src_gt.read(window=window)

                filtered = _filter_patch(patch_inference, patch_gt)
                if filtered is not None:
                    results.append(filtered)

    return results


def main():
    tif_files = [f for f in os.listdir(path_input) if f.endswith(".tif")]

    aggs = [CloudRemovalDatasetMetrics() for _ in range(10)]
    agg_global = CloudRemovalDatasetMetrics()
    agg_global_4 = CloudRemovalDatasetMetrics()

    # Nombre de workers : tous les cœurs disponibles, plafonné à 8 pour limiter
    # la pression mémoire (chaque worker charge une tuile entière).
    n_workers = min(os.cpu_count() or 1, 8)
    print(f"[parallel] {len(tif_files)} tuiles, {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_tile, f, path_input, gt_path_dir): f
            for f in tif_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tuiles"):
            tif_file = futures[future]
            try:
                patches = future.result()
            except Exception as e:
                print(f"  [ERREUR] {tif_file} : {e}")
                continue

            # Agrégation dans le process principal (accumulateurs non thread-safe)
            for tile_gt, tile_inference, mask_agg in patches:
                gt_t = Tensor(tile_gt)
                inf_t = Tensor(tile_inference)
                mask_t = Tensor(mask_agg)

                agg_global.update(target=gt_t, masks=mask_t, predicted=inf_t)
                agg_global_4.update(
                    target=gt_t[:, :, INDICES_4],
                    masks=mask_t,
                    predicted=inf_t[:, :, INDICES_4],
                )
                for i in range(10):
                    aggs[i].update(
                        target=gt_t[:, :, i:i+1],
                        masks=mask_t,
                        predicted=inf_t[:, :, i:i+1],
                    )

    # ── Affichage et sauvegarde des métriques ────────────────────────────────
    dictionnaire = {
        "bande": [],
        "mae": [],
        "mse": [],
        "psnr": [],
        "r2": [],
        "rmse": [],
        "sam": [],
        "ssim": [],
    }

    print("\n--------------global---------------")
    results = agg_global.compute()
    dictionnaire["bande"].append("all")
    for key in sorted(results.keys()):
        if "observed" in key:
            print(f"{key} : {results[key]}")
            dictionnaire[key.split("_")[0]].append(results[key])

    print("\n--------------global_4---------------")
    results = agg_global_4.compute()
    dictionnaire["bande"].append("all_4")
    for key in sorted(results.keys()):
        if "observed" in key:
            print(f"{key} : {results[key]}")
            dictionnaire[key.split("_")[0]].append(results[key])

    for i in range(10):
        print(f"\n--------------band{i}---------------")
        dictionnaire["bande"].append(i)
        results = aggs[i].compute()
        for key in sorted(results.keys()):
            if "observed" in key:
                print(f"{key} : {results[key]}")
                dictionnaire[key.split("_")[0]].append(results[key])

    pd.DataFrame(dictionnaire).to_csv("resultats.csv")
    print("\nRésultats sauvegardés dans resultats.csv")


if __name__ == "__main__":
    main()
