import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from prodict import Prodict
from rasterio import Affine
from tqdm import tqdm

from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from lib import config_utils
from lib.arguments import eval_parser
from lib.data_utils import get_dataset
from lib.eval_tools import Imputation

THRESHOLD = 0.5
MAX_PIXEL_INTENSITY_USED_FOR_REVERSE = 10_000
GDAL_OPTIONS = {
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "SPARSE_MODE": False,
}


class TypeConverter:

    def __init__(self):
        self._from = "float32"
        self._to = "uint8"

    def from_type(self, img_type):
        self._from = img_type
        return self

    def to_type(self, img_type):
        self._to = img_type
        return self

    def convert(self, img, threshold=0.5):
        if self._from == "float32":
            if self._to == "float32":
                return img
            elif self._to == "uint8":
                if img.max() > 1:
                    info = np.idebug(img.dtype)  # Get the information of the incoming image type
                    img = img.astype(np.float32) / info.max  # normalize the data to 0 - 1
                img = 255 * img  # scale by 255
                return img.astype(np.uint8)
            elif self._to == "uint16":
                if img.max() > 1:
                    info = np.idebug(img.dtype)  # Get the information of the incoming image type
                    img = img.astype(np.float32) / info.max  # normalize the data to 0 - 1
                img = np.iinfo(np.uint16).max * img  # scale by 65535
                return img.astype(np.uint16)
            elif self._to == "bit":
                img = img > threshold
                return img.astype(np.uint8)
            else:
                return img


class Evaluator:
    def __init__(
        self,
        args: argparse.Namespace,
        args_test_data: DictConfig,
    ):

        self.args = args
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

        from dataloader_CIRCA.datasets.cr_metrics_nina import CloudRemovalMetrics
        from dataloader_CIRCA.datasets.cr_torchmetrics import CloudRemovalDatasetMetrics

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
            max_pixel_intensity=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE,
        )

        # self.compute_metrics = EvalMetrics(self.args_metrics)

        _ = torch.set_grad_enabled(False)

        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f"Cannot find the configuration file used during training: {args.config_file}\n")

        # Read config file used during training
        self.config = config_utils.read_config(args.config_file)

        # if "test_data" in self.config:
        #     args_test_data = self.config.test_data

        # Merge generic data settings (used during training) with test-specific data settings
        if self.config.data.get("hdf5_file", False):
            for key in ["hdf5_file", "hdf5_file_read"]:
                if key in args_test_data:
                    args_test_data.pop(key)
            args_test_data.hdf5_file = self.config.data.hdf5_file
        # Manage old config settings
        if "include_S1" in args_test_data:
            if args_test_data.include_S1 is True:
                self.config.data.use_sar = "mix_closest"
            else:
                self.config.data.use_sar = False
            args_test_data.pop("include_S1")
        self.config.data.update(args_test_data)

        if self.config.data.dataset != "circa":
            self.config.data.preprocessed = True

        # Evaluate the entire image sequence (dans le cas de l'evaluation)
        self.config.data.max_seq_length = None

        if args_test_data.get("mode", False) and args_test_data.mode is not None:
            phase = args_test_data.mode
        else:
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

        from lib import data_utils

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
            config_file_train=self.args.config_file,
            method=self.args.method,
            mode=args.mode,
            checkpoint=self.args.checkpoint,
            config_file_test=self.args.test_data.test_config,
            # temporal_window=MAX_SAMPLES_ON_GPU,
            num_channels=self.dset.num_channels,
            device=device,
        )
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

                self.compute_metrics.update(
                    target=denorm_target,
                    masks=batch["masks"],
                    predicted=denorm_pred,
                    cloud_masks=batch.get("cloud_mask", None),
                )

            return self.compute_metrics.compute()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        eval_parser.print_help()
        sys.exit(1)

    args = eval_parser.parse_args()

    config = config_utils.read_config(args.config_file)
    if "test_data" in config:
        temp = OmegaConf.create()
        temp.config_file = args.config_file
        temp.method = args.method
        temp.test_data = config.test_data
        if "mode" in temp.test_data:
            temp.mode = config.test_data.mode
        if "checkpoint" in temp.test_data:
            temp.checkpoint = config.test_data.checkpoint
            del temp.test_data.checkpoint
        if "return_predictions" in temp.test_data:
            temp.return_predictions = config.test_data.return_predictions
            del temp.test_data.return_predictions
        if "predictions_save_path" in temp.test_data:
            temp.predictions_save_path = config.test_data.predictions_save_path
            del temp.test_data.predictions_save_path
        args = temp

    # Extract settings w.r.t. test data
    if args.test_data.test_config is not None:
        if not os.path.isfile(args.test_data.test_config):
            raise FileNotFoundError(f"Cannot find the test configuration file: {args.test_data.test_config}\n")
        test_config = config_utils.read_config(args.test_data.test_config)
        args_test_data = test_config.data
    else:
        args_test_data = OmegaConf.create()

    if args.test_data.hdf5_file is not None:
        # if not os.path.isfile(os.path.join(args_test_data.root, args.test_data.hdf5_file)):
        #     raise FileNotFoundError(
        #         f"Cannot find the data file: {os.path.join(args_test_data.root, args.test_data.hdf5_file)}\n"
        #     )
        args_test_data.hdf5_file = args.test_data.hdf5_file
    if args.test_data.split is not None:
        args_test_data.split = args.test_data.split
    if args.test_data.mode is not None:
        args_test_data.mode = args.test_data.mode

    evaluator = Evaluator(args, args_test_data)
    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print(f"Evaluation completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")
    print("Statistics:\n===========")
    print(stats)

    import json
    from pathlib import Path

    main_config = config_utils.read_config(args.config_file)
    if main_config.get("output", False) and main_config.output.get("save_dir", False):
        if stats is not None:
            with open((Path(main_config.output.save_dir) / "test_stats.json"), "w") as f:
                json.dump(stats, f)
