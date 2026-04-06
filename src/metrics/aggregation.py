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

from src.metrics.cloud_removal import CloudRemovalMetrics


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
        max_pixel_intensity: int = 1,
        sam_units: str = "rad",
        window_size: int = 5,
        skip_nan: bool = True,
    ) -> None:
        """Args:
        metrics: List of metric names or None for all.
        eval_occluded_observed: Forwarded to CloudRemovalMetrics.
        clean_gt_cloudy_pixels: Forwarded.
        sam_units: "rad" | "deg".
        window_size: SSIM window size.
        skip_nan: If True, ignore NaN values when updating aggregates.
        """
        self.sample_metrics = CloudRemovalMetrics(
            metrics=metrics,
            eval_occluded_observed=eval_occluded_observed,
            clean_gt_cloudy_pixels=clean_gt_cloudy_pixels,
            sam_units=sam_units,
            window_size=window_size,
            max_pixel_intensity=max_pixel_intensity,
        )
        self.skip_nan = skip_nan
        self._aggregators: Dict[str, MeanMetric] = {}

    def _get_or_create_aggregator(self, name: str) -> MeanMetric:
        if name not in self._aggregators:
            self._aggregators[name] = MeanMetric()
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
        predicted = predicted.detach()
        target = target.detach()
        masks = masks.detach()
        if cloud_masks is not None:
            cloud_masks = cloud_masks.detach()

        metrics_dict = self.sample_metrics(target=target, masks=masks, predicted=predicted, cloud_masks=cloud_masks)

        for k, v in metrics_dict.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                if self.skip_nan:
                    continue
            value = torch.tensor(v, dtype=torch.float32)
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

    # Optional: make the class iterable-friendly with state_dict / load_state_dict
    def state_dict(self) -> Dict[str, Dict[str, Tensor]]:
        return {k: agg.state_dict() for k, agg in self._aggregators.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, Tensor]]):
        for k, sd in state.items():
            agg = self._get_or_create_aggregator(k)
            agg.load_state_dict(sd)
