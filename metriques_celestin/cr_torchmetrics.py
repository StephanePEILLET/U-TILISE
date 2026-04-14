import math

import torch
from cr_metrics_per_bands import CloudRemovalMetrics
from torch import Tensor
from torchmetrics.aggregation import MeanMetric


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
        metrics: list[str] | None = None,
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = True,
        sam_units: str = "rad",
        window_size: int = 5,
        skip_nan: bool = True,
        device: torch.device | None = None,
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
        self._aggregators: dict[str, MeanMetric] = {}

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
        cloud_masks: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute metrics for one sample/batch and update running means.

        Returns the per-sample metrics dict (not averaged)."""
        predicted = predicted.detach()
        target = target.detach()
        masks = masks.detach()
        if cloud_masks is not None:
            cloud_masks = cloud_masks.detach()

        metrics_dict = self.sample_metrics(target=target, masks=masks, predicted=predicted, cloud_masks=cloud_masks)

        for k, v in metrics_dict.items():
            if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) and self.skip_nan:
                continue
            value = torch.tensor(v, dtype=torch.float32, device=self.device)
            if (torch.isnan(value) or torch.isinf(value)) and self.skip_nan:
                continue
            self._get_or_create_aggregator(k).update(value)
        return metrics_dict

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
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
    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {k: agg.state_dict() for k, agg in self._aggregators.items()}

    def load_state_dict(self, state: dict[str, dict[str, Tensor]]):
        for k, sd in state.items():
            agg = self._get_or_create_aggregator(k)
            agg.load_state_dict(sd)
