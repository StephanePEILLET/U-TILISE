from typing import Any

import torch
import torchgeometry as tgm
from prodict import Prodict
from torch import Tensor, nn

from lib.data_utils import extract_sample


class TrainLoss:
    """
    Loss computation.

    Args:
        args: dict, parameters that define the type of loss functions and their relative weighting.
    """

    def __init__(self, args: dict):
        self.args = args

        # L1 reconstruction loss
        self.recon_l1 = nn.L1Loss(reduction="mean")

        # SSIM loss: (1 - SSIM)/2
        self.ssim = tgm.losses.SSIM(5, reduction="mean")

        # Weights
        self.weights = Prodict()
        self.weights.l1_loss = self.args.get("l1_loss_w", 1.0)
        self.weights.l1_loss_occluded_input_pixels = self.args.get(
            "l1_loss_occluded_input_pixels_w", 1.0
        )
        self.weights.l1_loss_observed_input_pixels = self.args.get(
            "l1_loss_observed_input_pixels_w", 1.0
        )
        self.weights.ssim_loss = self.args.get("ssim_loss_w", 1.0)
        self.weights.masked_l1_loss = self.args.get("masked_l1_loss_w", 1.0)
        self.weights.ndvi_loss = self.args.get("ndvi_loss_w", 0.15)
        self.weights.temporal_r2_loss = self.args.get("temporal_r2_loss_w", 0.1)

        # NDVI band indices (Sentinel-2 10-band ordering: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12)
        self.red_idx = self.args.get("ndvi_red_idx", 2)   # B4 (Red)
        self.nir_idx = self.args.get("ndvi_nir_idx", 6)   # B8 (NIR)

    def __call__(
        self, batch: dict[str, Any], predicted: Tensor
    ) -> tuple[dict[str, float], Tensor]:
        """
        Args:
            batch:           dict, batch sample; the following information will be extracted from the batch using
                             the functionality extract_sample():
                                target:       torch.Tensor, (B x T x C x W x H); target sequence.
                                masks:        torch.Tensor, (B x T x 1 x W x H); a pixel value of 0 indicates an
                                              observed (non-masked) input pixel and a pixel value of 1 a masked input
                                              pixel.
                                mask_valid:   torch.Tensor, (B, T); 0 indicates a non-valid time step (zero-padded) and
                                              1 a valid time step.
                                cloud_mask:   torch.Tensor, (B x T x 1 x W x H), 0 indicates a non-occluded target pixel
                                              and 1 an occluded target pixel.
            predicted:       torch.Tensor, (B x T x C x W x H); predicted sequence.
        """

        _, target, masks, mask_valid, cloud_mask, _, _ = extract_sample(batch)

        # Initialize losses
        loss_dict = Prodict()

        # Temporal R² loss (needs B, T, C, H, W shape — compute before flattening)
        if self.args.get("temporal_r2_loss", False) and self.weights.temporal_r2_loss > 0:
            loss_dict.temporal_r2_loss = self._temporal_r2_loss(
                predicted, target, mask_valid
            )

        # Concatenate batch and time dimension
        b, t, c, h, w = predicted.shape
        predicted = predicted.view(b * t, c, h, w)
        target = target.view(b * t, c, h, w)
        masks = masks.view(b * t, 1, h, w).expand(target.shape)

        # Evaluate valid and non-padded frames only
        if mask_valid is not None:
            mask_valid = mask_valid.view(b * t).bool()
            predicted = predicted[mask_valid, ...]
            target = target[mask_valid, ...]
            masks = masks[mask_valid, ...]
            if cloud_mask is not None:
                cloud_mask = cloud_mask.view(b * t, 1, h, w)[mask_valid, ...].expand(
                    target.shape
                )

        # L1 reconstruction loss w.r.t. all pixels
        if self.args.get("l1_loss", False) and self.weights.l1_loss:
            loss_dict.l1_loss = self.recon_l1(predicted, target)

        # L1 reconstruction loss w.r.t. masked input pixels
        if (
            self.args.get("l1_loss_occluded_input_pixels", False)
            and self.weights.l1_loss_occluded_input_pixels > 0
        ):
            loss_dict.l1_loss_occluded_input_pixels = self.recon_l1(
                predicted[masks == 1.0], target[masks == 1.0]
            )

        # L1 reconstruction loss w.r.t. observed input pixels (unmasked input pixels)
        if (
            self.args.get("l1_loss_observed_input_pixels", False)
            and self.weights.l1_loss_observed_input_pixels > 0
        ):
            loss_dict.l1_loss_observed_input_pixels = self.recon_l1(
                predicted[masks == 0.0], target[masks == 0.0]
            )

        # SSIM loss
        if self.args.get("ssim_loss", False) and self.weights.ssim_loss > 0:
            loss_dict.ssim_loss = self.ssim(predicted, target)

        # L1 reconstruction loss w.r.t. all pixels associated with a valid ground truth reflectance
        if self.args.get("masked_l1_loss", False) and self.weights.masked_l1_loss:
            loss_dict.masked_l1_loss = self.recon_l1(
                predicted[cloud_mask == 0.0], target[cloud_mask == 0.0]
            )

        # NDVI loss: L1 between predicted and target NDVI
        if self.args.get("ndvi_loss", False) and self.weights.ndvi_loss > 0:
            loss_dict.ndvi_loss = self._ndvi_loss(predicted, target)

        # Compute the total loss as the sum of (weighted) individual losses
        total_loss = torch.zeros(1, requires_grad=True).to(target.device)
        for key in loss_dict:
            total_loss += loss_dict[key] * self.weights[key]
            loss_dict[key] = loss_dict[key].detach().item()

        loss_dict.total_loss = total_loss.detach().item()

        return loss_dict, total_loss

    def _ndvi_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        """Compute L1 loss between predicted and target NDVI maps.

        NDVI = (NIR - Red) / (NIR + Red), with a small epsilon for numerical stability.
        Data is expected in [0, 1] (post process_MS normalization).
        """
        eps = 1e-6
        pred_ndvi = (predicted[:, self.nir_idx] - predicted[:, self.red_idx]) / \
                    (predicted[:, self.nir_idx] + predicted[:, self.red_idx] + eps)
        tgt_ndvi = (target[:, self.nir_idx] - target[:, self.red_idx]) / \
                   (target[:, self.nir_idx] + target[:, self.red_idx] + eps)
        return nn.functional.l1_loss(pred_ndvi, tgt_ndvi)

    def _temporal_r2_loss(self, predicted: Tensor, target: Tensor,
                          mask_valid: Tensor | None) -> Tensor:
        """Loss = 1 - R² computed per pixel along the temporal dimension.

        R² measures how well predicted temporal profiles match the target.
        Maximizing R² ≡ minimizing (1 - R²).

        Args:
            predicted: (B, T, C, H, W) predicted sequence.
            target:    (B, T, C, H, W) target sequence.
            mask_valid: (B, T) or None; 1 = valid time step, 0 = padded.
        """
        # mask_valid: (B, T) → (B, T, 1, 1, 1) to broadcast over C, H, W
        if mask_valid is not None:
            mv = mask_valid.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1, 1)
            n_valid = mv.sum(dim=1, keepdim=True).clamp(min=2)  # (B, 1, 1, 1, 1)
            # Masked mean along T
            tgt_mean = (target * mv).sum(dim=1, keepdim=True) / n_valid
            ss_tot = ((target - tgt_mean) ** 2 * mv).sum(dim=1)
            ss_res = ((target - predicted) ** 2 * mv).sum(dim=1)
        else:
            tgt_mean = target.mean(dim=1, keepdim=True)
            ss_tot = ((target - tgt_mean) ** 2).sum(dim=1)
            ss_res = ((target - predicted) ** 2).sum(dim=1)

        # Avoid division by zero for constant pixels (ss_tot ~ 0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        # Clamp R² to [-1, 1] to bound the loss in [0, 2]
        r2 = r2.clamp(min=-1.0, max=1.0)
        return (1.0 - r2).mean()
