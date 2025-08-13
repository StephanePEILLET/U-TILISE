from enum import Enum
from typing import Dict
from typing import List

import numpy as np
from torch import Tensor
from torch.nn import ModuleDict
from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric


class MetricType(Enum):
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    PSNR = "psnr"
    SSIM = "ssim"
    R2 = "r2"
    SAM = "sam"


class DatasetCloudRemovalMetrics(Metric):
    """
    A torchmetrics.Metric wrapper for CloudRemovalMetrics to compute metrics over a whole dataset.

    This class accumulates metrics from individual batches and computes the mean at the end
    by delegating the aggregation to `torchmetrics.aggregation.MeanMetric`.
    """

    def __init__(
        self,
        metrics: List[MetricType] = list(MetricType),
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = True,
        sam_units: str = "rad",
        window_size: int = 5,
        **kwargs,
    ):
        """
        Initializes the DatasetLevelCloudRemovalMetrics class.

        Args:
            metrics (List[MetricType]): A list of metrics to compute.
            eval_occluded_observed (bool): If True, evaluates metrics separately for occluded and observed pixels.
            clean_gt_cloudy_pixels (bool): If True, excludes ground truth cloudy pixels from the evaluation.
            sam_units (str): The units for the SAM metric, either "rad" or "deg".
            window_size (int): The window size for the SSIM metric calculation.
            **kwargs: Additional arguments for torchmetrics.Metric.
        """
        super().__init__(**kwargs)
        self.cloud_removal_metrics = CloudRemovalMetrics(
            metrics=metrics,
            eval_occluded_observed=eval_occluded_observed,
            clean_gt_cloudy_pixels=clean_gt_cloudy_pixels,
            sam_units=sam_units,
            window_size=window_size,
        )

        # Generate the list of all possible metric names
        self.metric_names = self._get_all_possible_metric_names(metrics, eval_occluded_observed)

        # Use a ModuleDict to hold a MeanMetric for each metric type.
        # This is the recommended way to compose metrics.
        self.mean_metrics = ModuleDict({name: MeanMetric() for name in self.metric_names})

    def _get_all_possible_metric_names(self, metrics: List[MetricType], eval_occluded_observed: bool) -> List[str]:
        """Helper to generate all metric names based on configuration."""
        names = []
        suffixes = [""]
        if eval_occluded_observed:
            suffixes.extend(["_occluded_input_pixels", "_observed_input_pixels"])

        for metric_type in metrics:
            for suffix in suffixes:
                names.append(f"{metric_type.value}{suffix}")
        return names

    def update(self, target: Tensor, masks: Tensor, predicted: Tensor, cloud_masks: Tensor = None) -> None:
        """
        Update state with metrics from a single batch.

        Args:
            target (Tensor): The ground truth tensor (B x T x C x H x W).
            masks (Tensor): The input mask tensor (B x T x 1 x H x W).
            predicted (Tensor): The model's prediction tensor (B x T x C x H x W).
            cloud_masks (Tensor, optional): The ground truth cloud mask (B x T x 1 x H x W).
        """
        batch_metrics = self.cloud_removal_metrics(target, masks, predicted, cloud_masks)

        for name, value in batch_metrics.items():
            if name in self.mean_metrics and value is not None and not np.isnan(value):
                self.mean_metrics[name].update(value)

    def compute(self) -> Dict[str, float]:
        """
        Compute the mean of all metrics over all batches.

        Returns:
            Dict[str, float]: A dictionary of the mean of each metric.
        """
        # Compute the final mean for each metric, handling cases where a metric was never updated.
        return {
            name: metric.compute().item() if metric.count > 0 else np.nan for name, metric in self.mean_metrics.items()
        }
