import argparse
import os
import sys
import time

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from prodict import Prodict
from tqdm import tqdm

from lib import config_utils
from lib.arguments import eval_parser
from lib.data_utils import get_dataset
from lib.eval_tools import Imputation


class Evaluator:
    def __init__(self, args: argparse.Namespace, args_test_data: DictConfig):
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
        from dataloader_CIRCA.datasets.cr_metrics import CloudRemovalMetrics
        from dataloader_CIRCA.datasets.cr_torchmetrics import CloudRemovalDatasetMetrics

        list_available_metrics = [l.value for l in CloudRemovalMetrics.MetricType]
        metrics = (
            [k for k in self.args.metrics if k in list_available_metrics]
            if ("metrics" in self.args and self.args.metrics is not None)
            else list_available_metrics
        )

        self.compute_metrics = CloudRemovalDatasetMetrics(
            metrics=metrics,
            eval_occluded_observed=True,
            # device=self.device,
        )

        # self.compute_metrics = EvalMetrics(self.args_metrics)

        _ = torch.set_grad_enabled(False)

        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f"Cannot find the configuration file used during training: {args.config_file}\n")

        # Read config file used during training
        self.config = config_utils.read_config(args.config_file)

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

        if args_test_data.mode is not None:
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

        dset = get_dataset(self.config, phase=phase)
        print(f"Dataset length: {len(dset)}")
        subset = self.config.data.get("subset", False)
        if subset and isinstance(self.config.data.subset, bool):
            subset = 1

        # # FORCER UN SUBSET POUR LE TEST
        # subset = 200
        # print(f"INFO: Forcing evaluation on a subset of {subset} samples for testing.")

        from lib import data_utils

        self.dataloader = data_utils.get_dataloader(
            dset,
            self.config,
            batch_size=1,
            shuffle=False,
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
            device=device,
        )

    def evaluate(self):
        with torch.no_grad():  # Envelopper la boucle
            for i, batch in enumerate(tqdm(self.dataloader, leave=False)):
                _, y_pred = self.imputation.impute_sample(
                    batch,
                    # t_start=None,
                    # t_end=None,
                    # return_all=False,
                )
                # Evaluation
                self.compute_metrics.update(
                    target=batch["y"],
                    masks=batch["masks"],
                    predicted=y_pred,
                    cloud_masks=batch.get("cloud_mask", None),
                )
            return self.compute_metrics.compute()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        eval_parser.print_help()
        sys.exit(1)

    args = eval_parser.parse_args()

    # Extract settings w.r.t. test data
    if args.test_data.test_config is not None:
        if not os.path.isfile(args.test_data.test_config):
            raise FileNotFoundError(f"Cannot find the test configuration file: {args.test_data.test_config}\n")
        test_config = config_utils.read_config(args.test_data.test_config)
        args_test_data = test_config.data
    else:
        args_test_data = OmegaConf.create()

    if args.test_data.hdf5_file is not None:
        if not os.path.isfile(os.path.join(args_test_data.root, args.test_data.hdf5_file)):
            raise FileNotFoundError(
                f"Cannot find the data file: {os.path.join(args_test_data.root, args.test_data.hdf5_file)}\n"
            )
        args_test_data.hdf5_file = args.test_data.hdf5_file
    if args.test_data.hdf5_file_read is not None:
        args_test_data.hdf5_file = args.test_data.hdf5_file_read
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
