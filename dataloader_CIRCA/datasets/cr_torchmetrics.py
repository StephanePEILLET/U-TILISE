import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
import math
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics.aggregation import MeanMetric

from dataloader_CIRCA.datasets.cr_metrics import CloudRemovalMetrics


class CloudRemovalDatasetMetrics:
    """Aggregate cloud removal metrics over multiple samples using torchmetrics' MeanMetric.

    Usage:
        agg = CloudRemovalDatasetMetrics(metrics=["mae", "rmse", "psnr"])  # or None for all
        for batch in dataloader:
            preds = model(...)
            agg.update(target=batch["y"], masks=batch["masks"], predicted=preds, cloud_masks=batch.get("cloud_mask"))
        results = agg.compute()  # dict metric_name -> mean value
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = True,
        sam_units: str = "rad",
        window_size: int = 5,
        skip_nan: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """Args:
        metrics: List of metric names or None for all.
        eval_occluded_observed: Forwarded to CloudRemovalMetrics.
        clean_gt_cloudy_pixels: Forwarded.
        sam_units: "rad" | "deg".
        window_size: SSIM window size.
        skip_nan: If True, ignore NaN values when updating aggregates.
        device: Optional device for internal MeanMetric tensors.
        """
        self.sample_metrics = CloudRemovalMetrics(
            metrics=metrics,
            eval_occluded_observed=eval_occluded_observed,
            clean_gt_cloudy_pixels=clean_gt_cloudy_pixels,
            sam_units=sam_units,
            window_size=window_size,
        )
        self.skip_nan = skip_nan
        self.device = device if device is not None else torch.device("cpu")
        self._aggregators: Dict[str, MeanMetric] = {}

    def _get_or_create_aggregator(self, name: str) -> MeanMetric:
        if name not in self._aggregators:
            self._aggregators[name] = MeanMetric().to(self.device)
        return self._aggregators[name]

    @torch.no_grad()
    def update(
        self,
        *,
        target: Tensor,
        masks: Tensor,
        predicted: Tensor,
        cloud_masks: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics for one sample/batch and update running means.

        Returns the per-sample metrics dict (not averaged)."""
        metrics_dict = self.sample_metrics(target=target, masks=masks, predicted=predicted, cloud_masks=cloud_masks)
        for k, v in metrics_dict.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                if self.skip_nan:
                    continue
            value = torch.tensor(v, dtype=torch.float32, device=self.device)
            if torch.isnan(value) or torch.isinf(value):
                if self.skip_nan:
                    continue
            self._get_or_create_aggregator(k).update(value)
        return metrics_dict

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """Return the mean value for each metric accumulated so far."""
        return {k: agg.compute().item() for k, agg in self._aggregators.items()}

    @torch.no_grad()
    def reset(self) -> None:
        for agg in self._aggregators.values():
            agg.reset()

    def to(self, device: torch.device):  # convenience
        self.device = device
        for agg in self._aggregators.values():
            agg.to(device)
        return self

    # Optional: make the class iterable-friendly with state_dict / load_state_dict
    def state_dict(self) -> Dict[str, Dict[str, Tensor]]:
        return {k: agg.state_dict() for k, agg in self._aggregators.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, Tensor]]):
        for k, sd in state.items():
            agg = self._get_or_create_aggregator(k)
            agg.load_state_dict(sd)


# if __name__ == "__main__":
#     # Minimal integration test for multi-batch aggregation.
#     import warnings
#     from pathlib import Path

#     import torch

#     from dataloader_CIRCA.datasets import CIRCA_ADAPTED2UTILISE_Dataset
#     from lib import config_utils
#     from lib import data_utils
#     from lib.eval_tools import impute_sequence
#     from lib.models.utilise import UTILISE

#     warnings.filterwarnings("ignore", category=FutureWarning)

#     # Config -----------------------------------------------------------------
#     MAX_BATCHES = 5  # limit number of batches iterated for the demo
#     BATCH_SIZE = 1
#     T_START, T_END = 0, 5  # optional temporal crop

#     filter_settings = {
#         "type": "cloud-free",
#         "min_length": 5,
#         "return_valid_obs_only": True,
#     }

#     mask_kwargs = {
#         "mask_type": "random_clouds",
#         "ratio_masked_frames": 0.5,
#         "ratio_fully_masked_frames": 0.0,
#         "fixed_masking_ratio": False,
#         "non_masked_frames": [0],
#         "intersect_real_cloud_masks": False,
#         "dilate_cloud_masks": False,
#         "fill_type": "fill_value",
#         "fill_value": 1,
#         "p_filter": 0.1,
#     }

#     params_dataset = {
#         "phase": "test",
#         "hdf5_file": "/DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5",
#         "shuffle": False,
#         "use_sar": "mix_closest",
#         "channels": "all",
#         "filter_settings": filter_settings,
#         "max_seq_length": 30,
#         "render_occluded_above_p": None,
#         "mask_kwargs": mask_kwargs,
#         "pe_strategy": "day-within-sequence",
#         "augment": False,
#         "process_data": True,
#         "seed": 42,
#         "crop_settings": None,
#         "return_cloud_mask": True,
#     }

#     dset = CIRCA_ADAPTED2UTILISE_Dataset(**params_dataset)
#     dataloader = torch.utils.data.DataLoader(
#         dataset=dset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=None,
#         pin_memory=False,
#         drop_last=False,
#     )

#     # Model ------------------------------------------------------------------
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     temporal_window = dset.max_seq_length
#     num_channels = dset.num_channels
#     path_trainings_results = Path("/DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_SAR_120_epochs_2025-07-11_16-56")
#     path_ckpt = path_trainings_results / "checkpoints" / "Model_best.pth"
#     path_config_training = path_trainings_results / "config.yaml"
#     assert path_ckpt.exists(), f"Missing checkpoint at {path_ckpt}"
#     assert path_config_training.exists(), f"Missing config at {path_config_training}"

#     config_training = config_utils.read_config(path_config_training)
#     config_training.utilise.input_dim = num_channels
#     config_training.utilise.output_dim = 10  # model output dims (depends on SAR usage)
#     model = UTILISE(**config_training.utilise)
#     checkpoint = torch.load(path_ckpt, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device).eval()
#     del checkpoint

#     def infer_one_batch(batch, model, temporal_window, device, t_start=None, t_end=None):
#         if t_start is not None and t_end is not None:
#             batch["x"] = batch["x"][:, t_start:t_end, ...]
#             for key in ["y", "masks", "cloud_mask", "masks_valid_obs"]:
#                 if key in batch:
#                     batch[key] = batch[key][:, t_start:t_end, ...]
#             for key in ["days", "position_days"]:
#                 if key in batch:
#                     batch[key] = batch[key][:, t_start:t_end]
#         batch = data_utils.to_device(batch, device)
#         y_pred = impute_sequence(model, batch, temporal_window, return_att=False)
#         batch = data_utils.to_device(batch, "cpu")
#         y_pred = y_pred.cpu()
#         return batch, y_pred

#     # Aggregation ------------------------------------------------------------
#     dataset_metrics = CloudRemovalDatasetMetrics(metrics=None)  # None -> all metrics

#     for i, batch in enumerate(dataloader):
#         if i >= 20:
#             break
#         batch_processed, y_pred = infer_one_batch(
#             batch=batch,
#             model=model,
#             temporal_window=temporal_window,
#             device=device,
#             t_start=T_START,
#             t_end=T_END,
#         )
#         per_batch = dataset_metrics.update(
#             target=batch_processed["y"],
#             masks=batch_processed["masks"],
#             predicted=y_pred,
#             cloud_masks=batch_processed.get("cloud_mask"),
#         )
#         if i == 0:
#             print("Per-metric values on first batch:")
#             for k, v in per_batch.items():
#                 if v is None or (isinstance(v, float) and np.isnan(v)):
#                     print(f"  {k}: {v}")
#                 else:
#                     print(f"  {k}: {v:.4f}")

#     agg_results = dataset_metrics.compute()
#     print("\nAggregated mean metrics over", min(MAX_BATCHES, i + 1), "batches:")
#     for k, v in agg_results.items():
#         print(f"  {k}: {v:.4f}")
