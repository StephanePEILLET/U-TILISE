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
        eval_occluded_observed: bool = True,  #  to evaluate the metrics over all pixels and separately for occluded and observed input pixels
        clean_gt_cloudy_pixels: bool = False,  # whether to keep the ground truth cloudy pixels in the evaluation,
        sam_units: str = "rad",  # "deg" or "rad"
        window_size: int = 5,  # For SSIM
    ):
        # True to evaluate the metrics over all pixels and separately for occluded and observed input pixels;
        # False to evaluate the metrics over all pixels only
        self.eval_occluded_observed = eval_occluded_observed
        self.clean_gt_cloudy_pixels = clean_gt_cloudy_pixels
        self.sam_units = sam_units
        self.window_size = window_size
        # Initialize metric functions
        self.metric_fns = {}
        self._init_metric_functions(metrics)

    def _init_metric_functions(self, metrics: List[MetricType]) -> None:
        """Initialize the requested metric functions."""
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

    def __call__(
        self, target: Tensor, masks: Tensor, predicted: Tensor, cloud_masks: Tensor = None
    ) -> Dict[str, float]:
        """
        Args:
            target:       torch.Tensor, (B x T x C x W x H); target sequence.
            masks:        torch.Tensor, (B x T x 1 x W x H); a pixel value of 0 indicates an
                          observed (non-masked) input pixel and a pixel value of 1 a masked input
                          pixel.
            cloud_mask:   torch.Tensor, (B x T x 1 x W x H), 0 indicates a non-occluded target pixel
                          and 1 an occluded target pixel.
            predicted:       torch.Tensor, (B x T x C x W x H); predicted sequence.
        """
        # Initialize metrics
        metrics = dict()

        # Concatenate batch and time dimension
        B, T, C, H, W = predicted.shape
        n_frames = B * T
        predicted = predicted.view(n_frames, C, H, W)
        target = target.view(n_frames, C, H, W)
        masks = masks.view(n_frames, 1, H, W).expand(target.shape)

        # Structural similarity index (SSIM) evaluated over all images (prend l'image entière et pas des pixels)
        if MetricType.SSIM in self.metric_fns:
            dssim = self.metric_fns[MetricType.SSIM](
                predicted, target
            )  # outputs (1 - SSIM)/2; structural dissimilarity
            metrics["ssim"] = 1 - 2 * dssim

            # Structural similarity index (SSIM) evaluated over all images with data gaps
            if self.eval_occluded_observed:
                occ_images = (masks == 1.0).any(dim=-1).any(dim=-1).any(dim=-1)
                metrics["ssim_images_occluded_input_pixels"] = 1 - 2 * self.metric_fns[MetricType.SSIM](
                    predicted[occ_images], target[occ_images]
                )
                metrics["ssim_images_observed_input_pixels"] = 1 - 2 * self.metric_fns[MetricType.SSIM](
                    predicted[~occ_images], target[~occ_images]
                )
        # Si les gt masques nuages sont fournis, on n'utilise que les pixels non flagué comme étant des nuages
        # afin de pouvoir évaluer les métriques seulement sur les pixels non nuagueux qui seront masqués ou non synthétiquement.
        if cloud_masks is not None and self.clean_gt_cloudy_pixels:
            cloud_masks = cloud_masks.view(n_frames, 1, H, W)
            # Evaluate non-occluded target pixels only
            flag = cloud_masks.permute(0, 2, 3, 1).reshape(n_frames * H * W) == 0.0
            # Tensor shapes: (n_frames * H * W, C)
            predicted = predicted.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
            target = target.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
            masks = masks.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
        else:
            # Reshape tensors to a list of pixels, preserving the channel dimension: (N*H*W, C)
            n_pixels = n_frames * H * W
            predicted = predicted.permute(0, 2, 3, 1).reshape(n_pixels, C)
            target = target.permute(0, 2, 3, 1).reshape(n_pixels, C)
            masks = masks.permute(0, 2, 3, 1).reshape(n_pixels, C)

        # MAE (mean absolute error) evaluated over all pixels in the input sequence
        if MetricType.MAE in self.metric_fns:
            metrics[f"mae"] = self.metric_fns[MetricType.MAE](predicted, target)
            if self.eval_occluded_observed:
                metrics[f"mae_occluded_input_pixels"] = self.metric_fns[MetricType.MAE](
                    predicted[masks == 1.0], target[masks == 1.0]
                )
                metrics[f"mae_observed_input_pixels"] = self.metric_fns[MetricType.MAE](
                    predicted[masks == 0.0], target[masks == 0.0]
                )

        # Root mean squared error (RMSE)
        if MetricType.RMSE in self.metric_fns:
            metrics[f"rmse"] = self.metric_fns[MetricType.RMSE](predicted, target)
            if self.eval_occluded_observed:
                metrics[f"rmse_occluded_input_pixels"] = self.metric_fns[MetricType.RMSE](
                    predicted[masks == 1.0], target[masks == 1.0]
                )
                metrics[f"rmse_observed_input_pixels"] = self.metric_fns[MetricType.RMSE](
                    predicted[masks == 0.0], target[masks == 0.0]
                )

        # Mean squared error (MSE)
        if MetricType.MSE in self.metric_fns:
            metrics[f"mse"] = self.metric_fns[MetricType.MSE](predicted, target)
            if self.eval_occluded_observed:
                metrics[f"mse_occluded_input_pixels"] = self.metric_fns[MetricType.MSE](
                    predicted[masks == 1.0], target[masks == 1.0]
                )
                metrics[f"mse_observed_input_pixels"] = self.metric_fns[MetricType.MSE](
                    predicted[masks == 0.0], target[masks == 0.0]
                )

        # PSNR
        if MetricType.PSNR in self.metric_fns:
            metrics[f"psnr"] = self.metric_fns[MetricType.PSNR](predicted, target)
            if self.eval_occluded_observed:
                metrics[f"psnr_occluded_input_pixels"] = self.metric_fns[MetricType.PSNR](
                    predicted[masks == 1.0], target[masks == 1.0]
                )
                metrics[f"psnr_observed_input_pixels"] = self.metric_fns[MetricType.PSNR](
                    predicted[masks == 0.0], target[masks == 0.0]
                )

        # R2
        if MetricType.R2 in self.metric_fns:
            metrics[f"r2"] = self.metric_fns[MetricType.R2](predicted, target)
            if self.eval_occluded_observed:
                metrics[f"r2_occluded_input_pixels"] = self.metric_fns[MetricType.R2](
                    predicted[masks == 1.0], target[masks == 1.0]
                )
                metrics[f"r2_observed_input_pixels"] = self.metric_fns[MetricType.R2](
                    predicted[masks == 0.0], target[masks == 0.0]
                )

        # SAM
        if MetricType.SAM in self.metric_fns:
            # First, compute SAM on the full images
            sam_score = self.metric_fns[MetricType.SAM](predicted, target)
            if sam_score is not None:
                metrics[f"sam"] = sam_score

            # Then, compute SAM on pixel subsets if required
            if self.eval_occluded_observed:
                # --- Occluded pixels SAM ---
                # Select pixels where at least one channel is masked
                occluded_mask = (masks == 1.0).any(dim=1)
                if occluded_mask.any():
                    pred_occ = predicted[occluded_mask]
                    targ_occ = target[occluded_mask]
                    # Reshape to (1, C, N_occ_pixels, 1) to be compatible with compute_sam
                    pred_occ_reshaped = pred_occ.permute(1, 0).unsqueeze(0).unsqueeze(3)
                    targ_occ_reshaped = targ_occ.permute(1, 0).unsqueeze(0).unsqueeze(3)
                    sam_occ = self.metric_fns[MetricType.SAM](pred_occ_reshaped, targ_occ_reshaped)
                    if sam_occ is not None:
                        metrics[f"sam_occluded_input_pixels"] = sam_occ

                # --- Observed pixels SAM ---
                # Select pixels where no channel is masked
                observed_mask = (masks == 0.0).all(dim=1)
                if observed_mask.any():
                    pred_obs = predicted[observed_mask]
                    targ_obs = target[observed_mask]
                    # Reshape to (1, C, N_obs_pixels, 1) to be compatible with compute_sam
                    pred_obs_reshaped = pred_obs.permute(1, 0).unsqueeze(0).unsqueeze(3)
                    targ_obs_reshaped = targ_obs.permute(1, 0).unsqueeze(0).unsqueeze(3)
                    sam_obs = self.metric_fns[MetricType.SAM](pred_obs_reshaped, targ_obs_reshaped)
                    if sam_obs is not None:
                        metrics[f"sam_observed_input_pixels"] = sam_obs

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = value

        return metrics


if __name__ == "__main__":
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

    from lib import config_utils
    from lib.data_utils import pad_collate
    from lib.data_utils import seed_worker

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

    from lib import data_utils
    from lib.eval_tools import impute_sequence
    from lib.models.utilise import UTILISE

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
        if v is None:
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
