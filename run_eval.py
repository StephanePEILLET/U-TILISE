"""Point d'entrée pour l'évaluation du modèle U-TILISE.

Charge un checkpoint entraîné, applique l'imputation par fenêtre glissante
sur le jeu de test, calcule les métriques (MAE, RMSE, PSNR, SSIM, SAM, R²)
et sauvegarde les résultats en JSON.

Usage:
    python run_eval.py <config_eval.yaml> utilise
"""

import argparse
import os
import sys
import time
from pathlib import Path

import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import config_utils
from src.arguments import eval_parser
from src.data.processing.parcel_mask import ParcelMaskGenerator
from src.data.processing.transforms import SentinelDataProcessor, TypeConverter
from src.data_utils import get_dataset
from src.eval_tools import Imputation

THRESHOLD = 0.5
MAX_PIXEL_INTENSITY_USED_FOR_REVERSE = 10_000
GDAL_OPTIONS = {
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "SPARSE_MODE": False,
}


class Evaluator:
    def __init__(
        self,
        config: DictConfig,
        args: argparse.Namespace,
    ):

        self.args = args
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args_metrics = {
            "masked_metrics": True,
            "sam_units": "deg",
            "eval_occluded_observed": True,
            "mae": True,
            "rmse": True,
            "mse": False,
            "ssim": True,
            "psnr": True,
            "sam": True,
        }

        from src.metrics.aggregation import CloudRemovalDatasetMetrics
        from src.metrics.cloud_removal import CloudRemovalMetrics

        list_available_metrics = [l.value for l in CloudRemovalMetrics.MetricType]
        # metrics = (
        #     [k for k in self.args.metrics if k in list_available_metrics]
        #     if ("metrics" in self.args and self.args.metrics is not None)
        #     else list_available_metrics
        # )
        metrics = ["mae", "mse", "rmse", "psnr", "ssim", "r2", "sam"]
        self.compute_metrics = CloudRemovalDatasetMetrics(
            metrics=metrics,
            eval_occluded_observed=True,
            clean_gt_cloudy_pixels=True,
            max_pixel_intensity=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE,
        )

        self.cr_metrics = CloudRemovalMetrics(
            metrics=metrics,
            eval_occluded_observed=True,
            clean_gt_cloudy_pixels=True,
            compute_per_band=True,
            max_pixel_intensity=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE,
        )

        _ = torch.set_grad_enabled(False)

        phase = "test"

        # Get the data loader
        if phase == "test" and self.config.mask.mask_type not in [
            "real_clouds",
            "consecutive_fully_masked",
            "random_fully_masked",
        ]:
            print(
                "During an evaluation, the argument mask_type must be set to “real_clouds“ \
                in order to be able to make an inference on the cloud masks passed as input \
                without adding additional synthetic cloud masks. \
                3STR => In order to use synthetic mask already included in the orignal cloud masks with \
                random or consecutive occlusions set the argument mask_type to “consecutive_fully_masked“\
                or “consecutive_fully_masked“."
            )

        self.dset = get_dataset(self.config, phase=phase)
        print(f"Dataset length: {len(self.dset)}")
        subset = self.config.data.get("subset", False)
        if subset and isinstance(self.config.data.subset, bool):
            subset = 1

        # # FORCER UN SUBSET POUR LE TEST
        # subset = 200
        # print(f"INFO: Forcing evaluation on a subset of {subset} samples for testing.")

        from src import data_utils

        self.dataloader = data_utils.get_dataloader(
            self.dset,
            self.config,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            subset=subset,
        )

        if self.config.misc.get("device", False):
            device = torch.device(self.config.misc.device)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Evaluation will be performed on device: {device}\n")

        # MAX_SAMPLES_ON_GPU = 14
        # Get the imputation model
        self.imputation = Imputation(
            train_config_path=self.args.train_config_path,
            checkpoint=self.args.checkpoint,
            num_channels=self.dset.num_channels,
            device=device,
            blend_mode=self.config.data.get("blend_mode", "switch"),
            center_only_n_keep=self.config.data.get("center_only_n_keep", 2),
        )
        # Parcel mask generator (optional)
        parcel_gpkg = self.config.data.get("parcel_gpkg", None)
        if parcel_gpkg is not None:
            load_transforms = self.config.data.get("load_transforms", None)
            if load_transforms is None:
                raise ValueError(
                    "load_transforms must be specified in the data config when using parcel_gpkg."
                )
            self.parcel_mask_gen = ParcelMaskGenerator(
                gpkg_path=parcel_gpkg,
                patches_json_path=load_transforms,
            )
            print(f"Parcel mask generator loaded from {parcel_gpkg}")
        else:
            self.parcel_mask_gen = None

        # Case with return of predictions
        if "return_predictions" in self.args:
            self.return_predictions = self.args.return_predictions
        else:
            self.return_predictions = False

        if "predictions_save_path" in self.args:
            self.predictions_save_path = Path(self.args.predictions_save_path)
            if self.return_predictions and not self.predictions_save_path.exists():
                self.predictions_save_path.mkdir(parents=True, exist_ok=True)
                print(f"Prediction save path created: {self.predictions_save_path}")
        else:
            self.predictions_save_path = None
            if self.return_predictions:
                raise ValueError(
                    "The argument 'predictions_save_path' must be specified if 'return_predictions' is set to True."
                )

    def write_predictions(
        self,
        batch,
        y_pred,
    ):
        """
        Write the predictions to disk with rasterio (georeferenced tiff).
        """
        output_type = "uint16"
        row = self.dset.patches_dataset[
            (self.dset.patches_dataset["mgrs25"] == batch["info"]["mgrs25"][0])
            & (self.dset.patches_dataset["window"] == batch["info"]["window"][0])
        ]
        y_pred = y_pred.squeeze(0).detach().numpy()
        y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2], y_pred.shape[3])

        meta = row.meta.values[0].copy()
        meta = {
            "driver": "GTiff",
            "dtype": output_type,
            "count": y_pred.shape[0],
            "width": y_pred.shape[1],
            "height": y_pred.shape[2],
        }
        out_filename = self.predictions_save_path / f"pred_{row.mgrs25.values[0]}_window_{row.window.values[0]}.tif"
        print(f"Writing predictions to {out_filename}")
        with rasterio.open(out_filename, "w", **meta, **GDAL_OPTIONS) as src:
            converter = TypeConverter()
            pred = converter.from_type("float32").to_type(output_type).convert(y_pred, threshold=THRESHOLD)
            src.write(pred)

    def evaluate(self):
        with torch.no_grad():  # Envelopper la boucle
            for i, batch in enumerate(tqdm(self.dataloader, leave=False)):
                # print(("Info batch:", batch["info"]))
                batch, y_pred = self.imputation.impute_sample(batch)
                # if self.return_predictions:
                #     self.write_predictions(batch, y_pred)
                # metrics_dict = self.cr_metrics(
                #     target=batch["y"],
                #     masks=batch["masks"],
                #     predicted=y_pred,
                #     cloud_masks=batch.get("cloud_mask", None),
                # )
                # Evaluation

                # Reverse normalization
                denorm_pred = SentinelDataProcessor.reverse_process_MS(
                    y_pred, intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE
                )
                denorm_target = SentinelDataProcessor.reverse_process_MS(
                    batch["y"], intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE
                )

                cloud_masks = batch.get("cloud_mask", None)

                # Exclude black (no-data) pixels from metrics:
                # pixels where all bands == 0 in the target are marked as cloudy
                nodata_mask = (denorm_target == 0).all(dim=2, keepdim=True).float()  # (B, T, 1, H, W)
                if cloud_masks is not None:
                    cloud_masks = torch.clamp(cloud_masks + nodata_mask, 0, 1)
                else:
                    cloud_masks = nodata_mask

                # Apply parcel mask: exclude non-parcel pixels from metrics
                if self.parcel_mask_gen is not None:
                    mgrs25 = batch["info"]["mgrs25"][0]  # batch_size=1
                    window_str = batch["info"]["window"][0]
                    parcel_mask = self.parcel_mask_gen.get_mask(mgrs25, window_str)
                    # parcel_mask: (1, 1, H, W), 1=parcel, 0=non-parcel
                    # Mark non-parcel pixels as cloudy so they are excluded
                    non_parcel = (1 - parcel_mask)  # (1, 1, H, W), 1=non-parcel
                    # Expand to match temporal dimension (1, T, 1, H, W)
                    T = denorm_target.shape[1]
                    non_parcel = non_parcel.unsqueeze(1).expand(-1, T, -1, -1, -1)
                    if cloud_masks is not None:
                        cloud_masks = torch.clamp(cloud_masks + non_parcel, 0, 1)
                    else:
                        cloud_masks = non_parcel

                self.compute_metrics.update(
                    target=denorm_target,
                    masks=batch["masks"],
                    predicted=denorm_pred,
                    cloud_masks=cloud_masks,
                )

            return self.compute_metrics.compute()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        eval_parser.print_help()
        sys.exit(1)

    args = eval_parser.parse_args()

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Cannot find the configuration file: {args.config_file}\n")

    config = config_utils.read_config_with_defaults(args.config_file, run_mode="test")

    # Extraire les metadonnees de test_data depuis la config d'evaluation
    test_data_section = config.pop("test_data", OmegaConf.create())
    train_config_path = test_data_section.get("test_config", None)
    checkpoint = test_data_section.get("checkpoint", args.checkpoint)

    if train_config_path is None:
        raise ValueError("test_data.test_config (chemin vers la config d'entrainement) est requis.\n")
    if not os.path.isfile(train_config_path):
        raise FileNotFoundError(f"Cannot find the training configuration file: {train_config_path}\n")

    train_config = config_utils.read_config_with_defaults(train_config_path, run_mode="test")

    # L'architecture du modele vient de la config d'entrainement.
    # On merge: default.yaml < train_config < eval_config
    # Ce qui garantit que utilise.encoder_widths, etc. viennent du train.
    config = OmegaConf.merge(train_config, config)
    config.misc.run_mode = "test"
    config.data.max_seq_length = None

    args.train_config_path = train_config_path
    args.checkpoint = checkpoint

    evaluator = Evaluator(config, args)
    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print(f"Evaluation completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")
    print("Statistics:\n===========")
    print(stats)

    import json
    from pathlib import Path

    if config.output.get("save_dir", False):
        save_dir = Path(config.output.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if stats is not None:
            with open((save_dir / "test_stats.json"), "w") as f:
                json.dump(stats, f)
        config_dump_path = save_dir / "config_eval.yaml"
        OmegaConf.save(config, config_dump_path)
        print(f"Config d'evaluation sauvegardee : {config_dump_path}")
