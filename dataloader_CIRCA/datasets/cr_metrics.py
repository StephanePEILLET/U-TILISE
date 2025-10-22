import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
import math
from enum import Enum
from functools import partial
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
import scipy

# import sklearn
import torch
import torchgeometry as tgm
from sklearn.metrics import r2_score
from torch import Tensor


class CloudRemovalMetrics:
    """
    Computes the metrics used to monitor the training progress or for evaluation.
    """

    class MetricType(Enum):
        MAE = "mae"
        MSE = "mse"
        RMSE = "rmse"
        PSNR = "psnr"
        SSIM = "ssim"
        R2 = "r2"
        SAM = "sam"

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = False,
        sam_units: str = "rad",
        window_size: int = 5,
    ) -> None:
        """
        Initializes the CloudRemovalMetrics class.

        Args:
            metrics (List[str] | None): List of metric names to compute (mae, mse, rmse, psnr, ssim, r2, sam).
                                        If None, all available metrics are used.
            eval_occluded_observed (bool): If True, also computes metrics separately for occluded/observed pixels.
            clean_gt_cloudy_pixels (bool): If True, excludes cloudy pixels from the ground truth during evaluation.
            sam_units (str): Units for the SAM metric ("rad" or "deg").
            window_size (int): Window size for SSIM computation.
        """
        # True to evaluate the metrics over all pixels and separately for occluded and observed input pixels;
        # False to evaluate the metrics over all pixels only
        self.eval_occluded_observed = eval_occluded_observed
        self.clean_gt_cloudy_pixels = clean_gt_cloudy_pixels
        self.sam_units = sam_units
        self.window_size = window_size
        # Initialize metric functions
        self.metric_fns: Dict[CloudRemovalMetrics.MetricType, callable] = {}
        metrics_enum = self._parse_metrics(metrics)
        self._init_metric_functions(metrics_enum)

    @staticmethod
    def _parse_metrics(metrics: Optional[List[str]]) -> List["CloudRemovalMetrics.MetricType"]:
        """Parse and validate the list of metric names into MetricType objects.

        Args:
            metrics: List of metric names (case-insensitive) or None.

        Returns:
            Ordered list (without duplicates) of MetricType values.

        Raises:
            ValueError: If one or more metric names are invalid.
        """
        if metrics is None:
            return list(CloudRemovalMetrics.MetricType)
        allowed = {m.value: m for m in CloudRemovalMetrics.MetricType}
        seen = set()
        ordered: List[CloudRemovalMetrics.MetricType] = []
        invalid: List[str] = []
        for m in metrics:
            key = m.lower()
            if key not in allowed:
                if key not in invalid:  # collect each invalid only once
                    invalid.append(key)
                continue
            enum_val = allowed[key]
            if enum_val not in seen:
                seen.add(enum_val)
                ordered.append(enum_val)
        if invalid:
            raise ValueError(
                f"Unknown metric name(s): {invalid}. Valid metrics are: {sorted(allowed.keys())}"  # noqa: E501
            )
        return ordered

    def _init_metric_functions(self, metrics: List["CloudRemovalMetrics.MetricType"]) -> None:
        """
        Initializes the metric functions based on the provided list of metric types.

        Args:
            metrics (List[MetricType]): The list of metrics to initialize.
        """
        metric_set = set(metrics)
        MetricType = CloudRemovalMetrics.MetricType  # local alias for brevity

        if MetricType.MAE in metric_set:
            self.metric_fns[MetricType.MAE] = lambda p, t: torch.mean(torch.abs(p - t))

        if MetricType.MSE in metric_set:
            self.metric_fns[MetricType.MSE] = lambda p, t: torch.mean(torch.square(p - t))

        if MetricType.RMSE in metric_set:
            self.metric_fns[MetricType.RMSE] = lambda p, t: torch.sqrt(self.metric_fns[MetricType.MSE](p, t))

        if MetricType.SSIM in metric_set:
            self.metric_fns[MetricType.SSIM] = tgm.losses.SSIM(self.window_size, reduction="mean")

        if MetricType.PSNR in metric_set:
            self.metric_fns[MetricType.PSNR] = lambda p, t: 20 * torch.log10(1 / self.metric_fns[MetricType.RMSE](p, t))

        if MetricType.SAM in metric_set:
            self.metric_fns[MetricType.SAM] = partial(self._compute_sam, units=self.sam_units)

        if MetricType.R2 in metric_set:
            self.metric_fns[MetricType.R2] = lambda p, t: r2_score(
                y_true=t.flatten(), y_pred=p.flatten(), sample_weight=None, multioutput=None
            )

    @staticmethod
    def _compute_sam(predicted: Tensor, target: Tensor, units: Literal["deg", "rad"] = "rad") -> Tensor:
        """
        Computes the spectral angle mapper (SAM) averaged over all time steps and batch samples.

        Args:
            predicted:   torch.Tensor,  (n_frames x C x H x W).
            target:      torch.Tensor,  (n_frames x C x H x W).

        Returns:
            sam_value:   torch.Tensor, (1, ), mean spectral angle [rad].
        """
        dot_product = (predicted * target).sum(dim=1)
        predicted_norm = predicted.norm(dim=1)
        target_norm = target.norm(dim=1)
        # Compute the SAM score for all pixels with vector norm > 0
        flag = torch.logical_and(predicted_norm != 0.0, target_norm != 0.0)
        if torch.any(flag):
            spectral_angles = torch.clamp(dot_product[flag] / (predicted_norm[flag] * target_norm[flag]), -1, 1).acos()
            sam_score = torch.mean(spectral_angles)
            if units == "deg":
                sam_score *= 180 / math.pi
            return sam_score
        else:
            return None

    def _compute_pixelwise_metric(
        self, metric_name: str, metric_fn, predicted: Tensor, target: Tensor, masks: Tensor
    ) -> Dict[str, float]:
        """
        Computes a single pixel-wise metric and its variants for occluded and observed regions.

        Args:
            metric_name (str): The name of the metric (e.g., 'mae', 'rmse').
            metric_fn (callable): The function to compute the metric.
            predicted (Tensor): The predicted tensor of shape (N, C), where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C), where 1 indicates occlusion.

        Returns:
            Dict[str, float]: A dictionary containing the computed metric for all pixels,
                              and optionally for occluded and observed pixels.
        """
        metrics = {}
        try:
            metrics[metric_name] = metric_fn(predicted, target)
        except ValueError:
            metrics[metric_name] = np.nan

        if self.eval_occluded_observed:
            occluded_mask = (masks == 1.0).any(dim=1)
            if occluded_mask.any():
                try:
                    metrics[f"{metric_name}_occluded_input_pixels"] = metric_fn(
                        predicted[occluded_mask], target[occluded_mask]
                    )
                except ValueError:
                    metrics[f"{metric_name}_occluded_input_pixels"] = np.nan
            else:
                metrics[f"{metric_name}_occluded_input_pixels"] = np.nan

            observed_mask = (masks == 0.0).all(dim=1)
            if observed_mask.any():
                try:
                    metrics[f"{metric_name}_observed_input_pixels"] = metric_fn(
                        predicted[observed_mask], target[observed_mask]
                    )
                except ValueError:
                    metrics[f"{metric_name}_observed_input_pixels"] = np.nan
            else:
                metrics[f"{metric_name}_observed_input_pixels"] = np.nan

        return metrics

    # @staticmethod
    # def r2_score_torch(y_pred: Tensor, y_target: Tensor, epsilon: float = 1e-6) -> Tensor:
    #     """
    #     Computes the R² (coefficient of determination) score using PyTorch.

    #     The R² score is a statistical measure of how well the regression predictions
    #     approximate the real data points. An R² of 1 indicates that the predictions
    #     perfectly fit the data.

    #     The formula is: R² = 1 - (SS_res / SS_tot)
    #     where:
    #     - SS_res (Residual Sum of Squares) is the sum of squared differences: Σ(y_target - y_pred)²
    #     - SS_tot (Total Sum of Squares) is the sum of squared differences from the mean: Σ(y_target - mean(y_target))²

    #     Args:
    #         y_pred (Tensor): The tensor containing the predicted values from the model.
    #         y_target (Tensor): The tensor containing the ground truth (target) values.
    #                         Must have the same shape as y_pred.
    #         epsilon (float): A small value added to the denominator for numerical
    #                         stability to prevent division by zero.

    #     Returns:
    #         Tensor: A scalar tensor containing the R² score.
    #     """
    #     if y_pred.shape != y_target.shape:
    #         raise ValueError(
    #             f"Input tensor shapes must be the same. " f"Got y_pred={y_pred.shape} and y_target={y_target.shape}"
    #         )
    #     # R² is not defined for single-sample inputs
    #     if y_pred.shape[0] <= 1:
    #         return None

    #     ss_res = torch.sum((y_target - y_pred) ** 2)
    #     ss_tot = torch.sum((y_target - torch.mean(y_target)) ** 2)

    #     return 1 - ss_res / (ss_tot + epsilon)

    def _compute_imagewise_metrics(self, predicted: Tensor, target: Tensor, masks: Tensor) -> Dict[str, float]:
        """
        Computes image-wise metrics like SSIM.

        Args:
            predicted (Tensor): The predicted tensor of shape (B*T, C, H, W).
            target (Tensor): The target tensor of shape (B*T, C, H, W).
            masks (Tensor): The mask tensor of shape (B*T, C, H, W).

        Returns:
            Dict[str, float]: A dictionary containing the computed image-wise metrics.
        """
        metrics = {}
        if CloudRemovalMetrics.MetricType.SSIM in self.metric_fns:
            dssim = self.metric_fns[CloudRemovalMetrics.MetricType.SSIM](predicted, target)
            metrics["ssim"] = 1 - 2 * dssim

            if self.eval_occluded_observed:
                occ_images = (masks == 1.0).any(dim=-1).any(dim=-1).any(dim=-1)
                if occ_images.any():
                    metrics["ssim_images_occluded_input_pixels"] = 1 - 2 * self.metric_fns[
                        CloudRemovalMetrics.MetricType.SSIM
                    ](predicted[occ_images], target[occ_images])
                else:
                    metrics["ssim_images_occluded_input_pixels"] = np.nan

                obs_images = ~occ_images
                if obs_images.any():
                    metrics["ssim_images_observed_input_pixels"] = 1 - 2 * self.metric_fns[
                        CloudRemovalMetrics.MetricType.SSIM
                    ](predicted[obs_images], target[obs_images])
                else:
                    metrics["ssim_images_observed_input_pixels"] = np.nan
        return metrics

    def _compute_channelwise_metrics(self, predicted: Tensor, target: Tensor, masks: Tensor) -> Dict[str, float]:
        """
        Computes channel-wise metrics like SAM for all, occluded, and observed pixels.

        Args:
            predicted (Tensor): The predicted tensor of shape (N, C), where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C).

        Returns:
            Dict[str, float]: A dictionary containing the computed channel-wise metrics.
        """
        metrics = {}
        if CloudRemovalMetrics.MetricType.SAM in self.metric_fns:
            sam_score = self.metric_fns[CloudRemovalMetrics.MetricType.SAM](predicted, target)
            if sam_score is not None:
                metrics["sam"] = sam_score

            if self.eval_occluded_observed:
                occluded_mask = (masks == 1.0).any(dim=1)
                if occluded_mask.any():
                    sam_occ = self.metric_fns[CloudRemovalMetrics.MetricType.SAM](
                        predicted[occluded_mask], target[occluded_mask]
                    )
                    if sam_occ is not None:
                        metrics["sam_occluded_input_pixels"] = sam_occ
                else:
                    metrics["sam_occluded_input_pixels"] = np.nan

                observed_mask = (masks == 0.0).all(dim=1)
                if observed_mask.any():
                    sam_obs = self.metric_fns[CloudRemovalMetrics.MetricType.SAM](
                        predicted[observed_mask], target[observed_mask]
                    )
                    if sam_obs is not None:
                        metrics["sam_observed_input_pixels"] = sam_obs
                else:
                    metrics["sam_observed_input_pixels"] = np.nan
        return metrics

    def _compute_pixelwise_metrics(self, predicted: Tensor, target: Tensor, masks: Tensor) -> Dict[str, float]:
        """
        Computes all requested pixel-wise metrics (MAE, MSE, RMSE, PSNR, R2).

        Args:
            predicted (Tensor): The predicted tensor of shape (N, C), where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C).

        Returns:
            Dict[str, float]: A dictionary containing all computed pixel-wise metrics.
        """
        metrics = {}
        pixel_metrics_to_compute = {
            CloudRemovalMetrics.MetricType.MAE,
            CloudRemovalMetrics.MetricType.MSE,
            CloudRemovalMetrics.MetricType.RMSE,
            CloudRemovalMetrics.MetricType.PSNR,
            CloudRemovalMetrics.MetricType.R2,
        }
        for metric_type in pixel_metrics_to_compute:
            if metric_type in self.metric_fns:
                new_metrics = self._compute_pixelwise_metric(
                    metric_name=metric_type.value,
                    metric_fn=self.metric_fns[metric_type],
                    predicted=predicted,
                    target=target,
                    masks=masks,
                )
                metrics.update(new_metrics)

        return metrics

    def _prepare_pixelwise_tensors(
        self,
        predicted_img: Tensor,
        target_img: Tensor,
        masks_img: Tensor,
        cloud_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepares and reshapes tensors for pixel-wise metric computation.

        This involves flattening the image-like tensors into pixel-lists and optionally
        filtering out pixels that are marked as cloudy in the ground truth.

        Args:
            predicted_img (Tensor): The predicted tensor of shape (B*T, C, H, W).
            target_img (Tensor): The target tensor of shape (B*T, C, H, W).
            masks_img (Tensor): The mask tensor of shape (B*T, C, H, W).
            cloud_masks (Tensor): The ground truth cloud mask tensor (B x T x 1 x H x W).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the prepared pixel-wise tensors:
                                           (predicted_pix, target_pix, masks_pix).
        """
        n_frames, C, H, W = predicted_img.shape

        # Optionally filter out cloudy pixels from the ground truth
        if cloud_masks is not None and self.clean_gt_cloudy_pixels:
            cloud_masks = cloud_masks.view(n_frames, 1, H, W)
            # Create a boolean flag to select only non-cloudy pixels from the ground truth
            flag = (cloud_masks.permute(0, 2, 3, 1).reshape(n_frames * H * W) == 0.0).squeeze()
            predicted_pix = predicted_img.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
            target_pix = target_img.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
            masks_pix = masks_img.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
        else:
            # If not filtering, flatten all tensors to a pixel-wise representation
            n_pixels = n_frames * H * W
            predicted_pix = predicted_img.permute(0, 2, 3, 1).reshape(n_pixels, C)
            target_pix = target_img.permute(0, 2, 3, 1).reshape(n_pixels, C)
            masks_pix = masks_img.permute(0, 2, 3, 1).reshape(n_pixels, C)

        return predicted_pix, target_pix, masks_pix

    def __call__(
        self, target: Tensor, masks: Tensor, predicted: Tensor, cloud_masks: Tensor = None
    ) -> Dict[str, float]:
        """
        Computes the specified cloud removal metrics.

        Args:
            target (Tensor): The ground truth tensor (B x T x C x H x W).
            masks (Tensor): The input mask tensor (B x T x 1 x H x W).
            predicted (Tensor): The model's prediction tensor (B x T x C x H x W).
            cloud_masks (Tensor, optional): The ground truth cloud mask (B x T x 1 x H x W).

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        # Detach all input tensors from the computation graph to prevent memory leaks
        predicted = predicted.detach()
        target = target.detach()
        masks = masks.detach()
        if cloud_masks is not None:
            cloud_masks = cloud_masks.detach()

        predicted = predicted.to(torch.float32)
        target = target.to(torch.float32)
        masks = masks.to(torch.float32)
        metrics = {}
        B, T, C, H, W = predicted.shape
        n_frames = B * T

        # always compute SSIM (image-wise) before any cloud filtering
        predicted_img_orig = predicted.view(n_frames, C, H, W)
        target_img_orig = target.view(n_frames, C, H, W)
        masks_img_orig = masks.view(n_frames, 1, H, W).expand(target_img_orig.shape)
        metrics.update(self._compute_imagewise_metrics(predicted_img_orig, target_img_orig, masks_img_orig))

        # Step 2: apply cloud filtering for pixel-wise metrics
        if self.clean_gt_cloudy_pixels and cloud_masks is not None:
            clear_pixels_mask = 1 - cloud_masks
            predicted = predicted * clear_pixels_mask
            target = target * clear_pixels_mask
            masks = masks * clear_pixels_mask

        # Step 3: prepare tensors and compute remaining metrics
        predicted_img = predicted.view(n_frames, C, H, W)
        target_img = target.view(n_frames, C, H, W)
        masks_img = masks.view(n_frames, 1, H, W).expand(target_img.shape)
        predicted_pix, target_pix, masks_pix = self._prepare_pixelwise_tensors(
            predicted_img, target_img, masks_img, cloud_masks
        )
        metrics.update(self._compute_pixelwise_metrics(predicted_pix, target_pix, masks_pix))
        metrics.update(self._compute_channelwise_metrics(predicted_pix, target_pix, masks_pix))
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif value is None:
                metrics[key] = np.nan
            else:
                metrics[key] = value
        return metrics


import math
from typing import Dict

from rich.console import Console
from rich.table import Table


def display_metrics(results: Dict[str, float]):
    """
    Displays result metrics in a formatted table using rich.
    The table structure adapts based on whether detailed metrics are provided.

    Args:
        results (Dict[str, float]): The dictionary containing metric names and their values.
    """
    console = Console()

    # Check if detailed (occluded/observed) metrics are present in the results
    has_detailed_metrics = any(
        "_occluded_input_pixels" in key or "_observed_input_pixels" in key for key in results.keys()
    )

    table = Table(
        title="[bold bright_blue]Cloud Reconstruction Metrics[/bold bright_blue]",
        show_header=True,
        header_style="bold magenta",
    )

    # Define table columns based on the content of the results dictionary
    table.add_column("Metric", style="cyan", no_wrap=True, justify="right")
    if has_detailed_metrics:
        table.add_column("Overall", style="white", justify="center")
        table.add_column("Occluded Pixels", style="yellow", justify="center")
        table.add_column("Observed Pixels", style="green", justify="center")
    else:
        table.add_column("Value", style="white", justify="center")

    # Get a sorted list of base metrics for consistent display order
    base_metrics = sorted([k for k in results.keys() if not ("_occluded" in k or "_observed" in k)])

    # Helper function to format numbers into strings, handling NaN values
    def format_value(v: float) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "[dim]N/A[/dim]"
        return f"{v:.4f}"

    for metric in base_metrics:
        # If detailed metrics exist, populate all columns
        if has_detailed_metrics:
            # Handle the special case for 'ssim' key naming
            occluded_key = (
                f"{metric}_images_occluded_input_pixels" if metric == "ssim" else f"{metric}_occluded_input_pixels"
            )
            observed_key = (
                f"{metric}_images_observed_input_pixels" if metric == "ssim" else f"{metric}_observed_input_pixels"
            )

            # Get values, which will be None if the key is missing
            val_global = results.get(metric)
            val_occluded = results.get(occluded_key)
            val_observed = results.get(observed_key)

            table.add_row(
                metric.upper(),
                format_value(val_global),
                format_value(val_occluded),
                format_value(val_observed),
            )
        # Otherwise, populate the simple table
        else:
            val_global = results.get(metric)
            table.add_row(metric.upper(), format_value(val_global))

    console.print(table)


# if __name__ == "__main__":
#     # Ajout calcul de métriques
#     from pathlib import Path
#     from typing import Any

#     from omegaconf import OmegaConf

#     from dataloader_CIRCA.datasets.cr_metrics import CloudRemovalMetrics
#     from dataloader_CIRCA.datasets.cr_metrics import display_metrics
#     from lib import config_utils
#     from lib import data_utils
#     from lib.eval_tools import Imputation
#     from run_train import setup_logging

#     def get_sample(dataloader: torch.utils.data.dataloader.DataLoader, sample_index: int) -> Dict[str, Any]:

#         batch = dataloader.dataset.__getitem__(sample_index)

#         # Introduce the batch dimension (required for the forward pass)
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.unsqueeze(0)
#             elif isinstance(v, int):
#                 batch[k] = [v]

#         return batch

#     # Setup configuration
#     config_file = Path("./configs/config_run_train.yaml")
#     default_config_path = Path("./configs/default.yaml")
#     cfg_custom = config_utils.read_config(config_file)
#     cfg_default = config_utils.read_config(default_config_path)
#     config = OmegaConf.merge(cfg_default, cfg_custom)

#     path_inference_config_file = Path("/home/SPeillet/cloud_reconstruction/U-TILISE/configs/config_run_eval.yaml")

#     # Faire différents changments dans les fichiers de configs pour les vérifications
#     config.data.hdf5_file = "/DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5"
#     config.mask.mask_type = "random_fully_masked"
#     config.data.max_seq_length = None
#     config.training_settings.batch_size = 10
#     config.mask.ratio_masked_frames = 0.8
#     config.mask.intersect_real_cloud_masks = False
#     config.mask.dilate_cloud_masks = False

#     # Setup logging
#     logger = setup_logging(config)

#     # Génération des dataset
#     phase = "test"
#     dset = data_utils.get_dataset(config, phase=phase, logger=logger)

#     # Modèle ALL_BANDS avec SAR
#     path_ckpt_config_file = Path(
#         "/DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_SAR_120_epochs_2025-07-11_16-56/config.yaml"
#     )
#     path_ckpt_pth = Path(
#         "/DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_SAR_120_epochs_2025-07-11_16-56/checkpoints/Model_best.pth"
#     )

#     training_config_file = config_utils.read_config(path_ckpt_config_file)
#     inference_config_file = config_utils.read_config(path_inference_config_file)

#     inference_imputation = Imputation(
#         config_file_train=path_inference_config_file,  # Config permettant de faire la configuration de l'inference
#         method="utilise",
#         checkpoint=path_ckpt_pth,
#         config_file_test=path_ckpt_config_file,  # Fichier ayant servi à l'entrainement du modèle (update params model)
#     )

#     # Setup data loader
#     subset = config.data.get("subset", False)
#     if subset and isinstance(config.data.subset, bool):
#         subset = 10

#     generator = torch.Generator()
#     generator.manual_seed(config.misc.get("random_seed", 42))

#     data_loader = data_utils.get_dataloader(
#         dset,
#         config,
#         drop_last=False,
#         shuffle=True,
#         generator=generator,
#         subset=subset,
#     )

#     for i in range(len(dset)):

#         sample = get_sample(data_loader, i)
#         batch, y_pred, att = inference_imputation.impute_sample(
#             batch=sample,
#             t_start=0,
#             t_end=5,
#             return_att=True,
#         )

#         cr_metrics = CloudRemovalMetrics(
#             eval_occluded_observed=True,
#             clean_gt_cloudy_pixels=False,
#         )

#         results = cr_metrics(
#             target=batch["y"],
#             predicted=y_pred,
#             masks=batch["masks"],
#             cloud_masks=batch["cloud_mask"],
#         )

#         display_metrics(results)
#         print("Sample info: ", sample["info"])
