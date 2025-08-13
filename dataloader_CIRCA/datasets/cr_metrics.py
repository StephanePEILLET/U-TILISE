import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
import math
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal

import numpy as np
import sklearn
import torch
import torch.utils
import torch.utils.data
import torchgeometry as tgm
from torch import Tensor

from dataloader_CIRCA.datasets import CIRCA_ADAPTED2UTILISE_Dataset

warnings.filterwarnings("ignore", category=FutureWarning)


class MetricType(Enum):
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    PSNR = "psnr"
    SSIM = "ssim"
    R2 = "r2"
    SAM = "sam"


class CloudRemovalMetrics:
    """
    Computes the metrics used to monitor the training progress or for evaluation.
    """

    def __init__(
        self,
        metrics: List[MetricType] = list(MetricType),
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = True,
        sam_units: str = "rad",
        window_size: int = 5,
    ) -> None:
        """
        Initializes the CloudRemovalMetrics class.

        Args:
            metrics (List[MetricType]): A list of metrics to compute.
            eval_occluded_observed (bool): If True, evaluates metrics separately for occluded and observed pixels.
            clean_gt_cloudy_pixels (bool): If True, excludes ground truth cloudy pixels from the evaluation.
            sam_units (str): The units for the SAM metric, either "rad" or "deg".
            window_size (int): The window size for the SSIM metric calculation.
        """
        # True to evaluate the metrics over all pixels and separately for occluded and observed input pixels;
        # False to evaluate the metrics over all pixels only
        self.eval_occluded_observed = eval_occluded_observed
        self.clean_gt_cloudy_pixels = clean_gt_cloudy_pixels
        self.sam_units = sam_units
        self.window_size = window_size
        # Initialize metric functions
        self.metric_fns: Dict[MetricType, callable] = {}
        self._init_metric_functions(metrics)

    def _init_metric_functions(self, metrics: List[MetricType]) -> None:
        """
        Initializes the metric functions based on the provided list of metric types.

        Args:
            metrics (List[MetricType]): The list of metrics to initialize.
        """
        metric_set = set(metrics)

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
            self.metric_fns[MetricType.R2] = lambda p, t: sklearn.metrics.r2_score(
                y_true=t.detach().flatten(), y_pred=p.detach().flatten()
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
        if MetricType.SSIM in self.metric_fns:
            dssim = self.metric_fns[MetricType.SSIM](predicted, target)
            metrics["ssim"] = 1 - 2 * dssim

            if self.eval_occluded_observed:
                occ_images = (masks == 1.0).any(dim=-1).any(dim=-1).any(dim=-1)
                if occ_images.any():
                    metrics["ssim_images_occluded_input_pixels"] = 1 - 2 * self.metric_fns[MetricType.SSIM](
                        predicted[occ_images], target[occ_images]
                    )
                else:
                    metrics["ssim_images_occluded_input_pixels"] = np.nan

                obs_images = ~occ_images
                if obs_images.any():
                    metrics["ssim_images_observed_input_pixels"] = 1 - 2 * self.metric_fns[MetricType.SSIM](
                        predicted[obs_images], target[obs_images]
                    )
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
        if MetricType.SAM in self.metric_fns:
            sam_score = self.metric_fns[MetricType.SAM](predicted, target)
            if sam_score is not None:
                metrics["sam"] = sam_score

            if self.eval_occluded_observed:
                occluded_mask = (masks == 1.0).any(dim=1)
                if occluded_mask.any():
                    sam_occ = self.metric_fns[MetricType.SAM](predicted[occluded_mask], target[occluded_mask])
                    if sam_occ is not None:
                        metrics["sam_occluded_input_pixels"] = sam_occ
                else:
                    metrics["sam_occluded_input_pixels"] = np.nan

                observed_mask = (masks == 0.0).all(dim=1)
                if observed_mask.any():
                    sam_obs = self.metric_fns[MetricType.SAM](predicted[observed_mask], target[observed_mask])
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
            MetricType.MAE,
            MetricType.MSE,
            MetricType.RMSE,
            MetricType.PSNR,
            MetricType.R2,
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
        Computes all configured metrics for the given prediction and target.

        Args:
            target (Tensor): The ground truth tensor (B x T x C x H x W).
            masks (Tensor): The input mask tensor (B x T x 1 x H x W), where 1 indicates a masked (occluded)
                          input pixel and 0 indicates an observed one.
            predicted (Tensor): The model's prediction tensor (B x T x C x H x W).
            cloud_masks (Tensor, optional): The ground truth cloud mask (B x T x 1 x H x W), where 1 indicates
                                           a cloudy pixel in the target. Defaults to None.

        Returns:
            Dict[str, float]: A dictionary of computed metrics, with metric names as keys and their float values.
                              Returns `np.nan` for metrics that could not be computed.
        """
        metrics = {}

        # 1. Prepare tensors for image-wise metrics (shape: B*T, C, H, W)
        B, T, C, H, W = predicted.shape
        n_frames = B * T
        predicted_img = predicted.view(n_frames, C, H, W)
        target_img = target.view(n_frames, C, H, W)
        masks_img = masks.view(n_frames, 1, H, W).expand(target_img.shape)

        # 2. Compute image-wise metrics (e.g., SSIM)
        metrics.update(self._compute_imagewise_metrics(predicted_img, target_img, masks_img))

        # 3. Prepare tensors for pixel-wise metrics
        predicted_pix, target_pix, masks_pix = self._prepare_pixelwise_tensors(
            predicted_img, target_img, masks_img, cloud_masks
        )

        # 4. Compute all pixel-wise metrics
        metrics.update(self._compute_pixelwise_metrics(predicted_pix, target_pix, masks_pix))
        metrics.update(self._compute_channelwise_metrics(predicted_pix, target_pix, masks_pix))

        # 5. Finalize metrics: convert tensors to float and handle None values
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif value is None:
                metrics[key] = np.nan
            else:
                metrics[key] = value

        return metrics


if __name__ == "__main__":
    from lib import config_utils
    from lib import data_utils
    from lib.data_utils import pad_collate
    from lib.data_utils import seed_worker
    from lib.eval_tools import impute_sequence
    from lib.models.utilise import UTILISE

    # Définition du dataset
    SUBSET_LENGTH = 20
    batch_size_inference = 1

    filter_settings = {
        "type": "cloud-free",  # Strategy for removing observations with data gaps.
        # ['cloud-free', 'cloud-free_consecutive']
        "min_length": 5,  # Minimum sequence length.
        "return_valid_obs_only": True,  # True to return the cloud-filtered sequences, False otherwise.
        # "max_t_sampling": 10,            # Maximum temporal sampling frequency in days.
    }

    mask_kwargs = {
        "mask_type": "random_clouds",  # Mask the input time series with randomly sampled cloud masks or the actual cloud masks. ['random_clouds', 'real_clouds']
        "ratio_masked_frames": 0.5,  # Ratio of partially/fully masked images per image time series (upper bound).
        "ratio_fully_masked_frames": 0.0,  # Ratio of fully masked images per image time series (upper bound).
        "fixed_masking_ratio": False,  # True to vary the masking ratio across different image time series, False otherwise.
        "non_masked_frames": [
            0
        ],  # list of int, time steps to be excluded from masking. E.g., [0] never masks the first frame in a sequence.
        "intersect_real_cloud_masks": False,  # True to intersect randomly sampled cloud masks with the actual cloud masks, False otherwise.
        "dilate_cloud_masks": False,  # True to dilate the cloud masks before masking, False otherwise.
        "fill_type": "fill_value",  # Strategy for initializing masked pixels. ['fill_value', 'white_noise', 'mean']
        "fill_value": 1,  # Pixel value of masked pixels. Used if fill_type == 'fill_value'.
        "p_filter": 0.1,
    }

    params_dataset = {
        "phase": "test",
        "hdf5_file": "/DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5",
        "shuffle": False,
        "use_sar": "mix_closest",
        "channels": "all",
        # U-TILISE specific parameters
        "filter_settings": filter_settings,
        "max_seq_length": 30,
        "render_occluded_above_p": None,  # Set to None to keep original cloud masks. Minimum cloud cover to fully mask an input image (0.9 demo config)
        "mask_kwargs": mask_kwargs,
        "pe_strategy": "day-within-sequence",
        "augment": False,
        "process_data": True,
        "seed": 42,
        # Récupération de vieux arguments du repo
        "crop_settings": None,
        "return_cloud_mask": True,
    }

    dset = CIRCA_ADAPTED2UTILISE_Dataset(**params_dataset)

    # dset = torch.utils.data.Subset(dset, range(SUBSET_LENGTH))
    dataloader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=batch_size_inference,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
    )

    temporal_window = dset.max_seq_length
    num_channels = dset.num_channels
    device = torch.device("cuda:0")
    path_trainings_results = Path("/DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_SAR_120_epochs_2025-07-11_16-56")
    path_ckpt = path_trainings_results / "checkpoints" / "Model_best.pth"
    path_config_training = path_trainings_results / "config.yaml"
    assert path_ckpt.exists()
    assert path_config_training.exists()
    config_training = config_utils.read_config(path_config_training)
    config_training.utilise.input_dim = num_channels
    config_training.utilise.output_dim = 10  # num_channels - 4 if use_sar
    model = UTILISE(**config_training.utilise)
    checkpoint = torch.load(path_ckpt)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    del checkpoint

    def infer_one_batch(batch, model, temporal_window, device, t_start=None, t_end=None):
        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch["x"] = batch["x"][:, t_start:t_end, ...]

            for key in ["y", "masks", "cloud_mask", "masks_valid_obs"]:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]

            for key in ["days", "position_days"]:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        batch = data_utils.to_device(batch, device)
        y_pred = impute_sequence(model, batch, temporal_window, return_att=False)
        batch = data_utils.to_device(batch, "cpu")
        y_pred = y_pred.cpu()
        return batch, y_pred

    batch = next(iter(dataloader))

    batch_processed, y_pred = infer_one_batch(
        batch=batch,
        model=model,
        temporal_window=temporal_window,
        device=device,
        t_start=0,
        t_end=10,
    )

    compute_metrics = CloudRemovalMetrics()
    metrics_on_sample = compute_metrics(
        target=batch["y"],
        masks=batch["masks"],
        predicted=y_pred,
        cloud_masks=batch["cloud_mask"],
    )
    print("**Metrics on the sample:**")
    for k, v in metrics_on_sample.items():
        if v is None or np.isnan(v):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
