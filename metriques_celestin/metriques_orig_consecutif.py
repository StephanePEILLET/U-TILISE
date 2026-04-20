import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from cr_torchmetrics import CloudRemovalDatasetMetrics
from torch import Tensor
from tqdm import tqdm

# inferences = "pred_mgrsc_30TXR_row-3_col-3.tif"
# gt = "bands_stacked_30TXR_row-3_col-3.tif"


# src_inferences = rasterio.open(inferences)
# src_gt = rasterio.open(gt)

# Paths LNV16
# store_dai = Path("/mnt/stores/store_dai")
store_dai = Path("/mnt/stores/store-DAI")
path_input = store_dai / "tmp/speillet/inferences/v3_combined/consecutive_fully_masked/2026-03-26_16-17"
gt_path_dir = store_dai / "projets/pac/3str/EXP_2/Data_Raster/test_v3/consecutif"

# path_input = Path("/media/store-dai/tmp/speillet/inferences/v3_combined/consecutive_fully_masked/2026-03-26_16-17")
# #path_input = Path("/media/store-dai/tmp/speillet/inferences/v3_combined/random_fully_masked/2026-03-26_16-17")
# gt_path_dir = Path("/media/store-dai/projets/pac/3str/EXP_2/Data_Raster/test_v3/consecutif")
# #gt_path_dir = Path("/media/store-dai/projets/pac/3str/EXP_2/Data_Raster/test_v3/aleatoire")
tif_files = [i for i in os.listdir(path_input) if i[-4:] == ".tif"]


def save_image(image: np.ndarray, path, transform=None, encoding=np.uint16, crs=None) -> None:
    dictionnaire = {
            'interleave': 'Band',
            'tiled': True
        }
    with rasterio.open(
        path, "w",
        driver="GTiff",
        transform=transform,
        dtype=encoding,
        count=image.shape[0],
        width=image.shape[2],
        height=image.shape[1],
        crs=crs,
        **dictionnaire) as dst:
        dst.write(image)


def run_tile(tile_inference, tile_gt):
    tile_inference = tile_inference.reshape((-1, 12, 256, 256))
    tile_gt = tile_gt.reshape((-1, 12, 256, 256))

    # On récupère seulement les bandes qui correspondent au masque de nuage de la vérité terrain
    clouds_image_gt = tile_gt[:, 10]

    # On récupère les lieux où il y a un faux nuage
    indices_faux_nuages = np.unique(np.where(clouds_image_gt > 100)[0])

    tile_inference = tile_inference[indices_faux_nuages]
    tile_gt = tile_gt[indices_faux_nuages]

    # On récupère les indices où il y a les vrais faux nuages (dates qui ne sont pas déjà nuageuses)
    clouds_image_gt = clouds_image_gt[indices_faux_nuages]
    maximum = np.max(clouds_image_gt, axis=(1, 2))

    indices_vrai_faux_nuages = np.where(maximum == 150)[0]

    tile_inference = tile_inference[indices_vrai_faux_nuages, :10]
    tile_gt = tile_gt[indices_vrai_faux_nuages, :10]
    tile_gt = np.where(tile_gt == 0, 0, tile_gt - 1000)

    tile_gt = np.expand_dims(tile_gt, 0)
    tile_inference = np.expand_dims(tile_inference, 0)
    mask_agg = np.where(tile_gt != 0, 0, 1)
    mask_agg = mask_agg[:, :, 0, :, :]

    agg_global.update(target=Tensor(tile_gt), masks=Tensor(mask_agg), predicted=Tensor(tile_inference))
    indices_4 = np.array([0, 1, 2, 7])
    agg_global_4.update(target=Tensor(tile_gt[:, :, indices_4]), masks=Tensor(mask_agg), predicted=Tensor(tile_inference[:, :, indices_4]))

    for i in range(10):
        tile_gt_i = np.expand_dims(tile_gt[:, :, i], 2)
        tile_inference_i = np.expand_dims(tile_inference[:, :, i], 2)
        aggs[i].update(target=Tensor(tile_gt_i), masks=Tensor(mask_agg), predicted=Tensor(tile_inference_i))


aggs = [CloudRemovalDatasetMetrics() for i in range(10)]
agg_global = CloudRemovalDatasetMetrics()
agg_global_4 = CloudRemovalDatasetMetrics()
for tif_file in tqdm(tif_files):

    tile = tif_file.split("_")[2]
    subtile = "MGRS25-" + "_".join(tif_file.replace(".tif", "").split("_")[2:5])
    gt_file = tif_file.replace("pred_mgrsc", "bands_stacked")
    gt_path = gt_path_dir / tile / subtile / gt_file

    src_inferences = rasterio.open(path_input / tif_file)

    src_gt = rasterio.open(gt_path)
    for i in range(8):
        for j in range(8):
            window = rasterio.windows.Window(i * 256, j * 256, 256, 256)
            try:
                tile_inference = src_inferences.read(window=window)
            except:
                time.sleep(30)
                tile_inference = src_inferences.read(window=window)
            tile_gt = src_gt.read(window=window)

            # save_image(tile_inference, "tile_inference.tif")
            # save_image(tile_gt, "tile_gt.tif")

            run_tile(tile_inference, tile_gt)


dictionnaire = {
    "bande": [],
    "mae": [],
    "mse": [],
    "psnr": [],
    "r2": [],
    "rmse": [],
    "sam": [],
    "ssim": []
}

print("")
print("--------------global---------------")
results = agg_global.compute()
dictionnaire["bande"].append("all")
for key in sorted(list(results.keys())):
    if "observed" in key:
        print(f"{key} : {results[key]}")
        dictionnaire[key.split("_")[0]].append(results[key])

print("")
print("--------------global---------------")
results = agg_global_4.compute()
dictionnaire["bande"].append("all_4")
for key in sorted(list(results.keys())):
    if "observed" in key:
        print(f"{key} : {results[key]}")
        dictionnaire[key.split("_")[0]].append(results[key])


for i in range(10):
    print("")
    print(f"--------------band{i}---------------")
    dictionnaire["bande"].append(i)
    results = aggs[i].compute()
    for key in sorted(list(results.keys())):
        if "observed" in key:
            print(f"{key} : {results[key]}")
            dictionnaire[key.split("_")[0]].append(results[key])


pd.DataFrame(dictionnaire).to_csv("resultats_orig_consecutif.csv")
