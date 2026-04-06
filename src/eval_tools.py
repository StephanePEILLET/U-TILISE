"""Outils d'imputation pour l'évaluation et l'inférence U-TILISE.

Contient la classe Imputation qui gère :
- Le chargement du modèle à partir d'un checkpoint
- L'inférence par fenêtre glissante (sliding window)
- La fusion des prédictions (blend) : switch, center, center_only, iterative
- La visualisation des résultats et masques d'attention
"""

import math
import os
from typing import Any

import matplotlib
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch import Tensor, nn

from src import config_utils, data_utils, utils, visutils
from src.visutils import COLORMAPS


class Imputation:
    def __init__(
        self,
        config_file_train: str | None,
        checkpoint: str | None = None,
        config_file_test: str | None = None,
        temporal_window: int | None = None,
        device: torch.device | None = None,
        num_channels: int = 10,
        blend_mode: str = "switch",
        center_only_n_keep: int = 2,
    ):
        self.checkpoint = checkpoint
        self.config_file_train = config_file_train
        self.blend_mode = blend_mode
        self.center_only_n_keep = center_only_n_keep

        if self.checkpoint is None:
            raise ValueError("No checkpoint specified.\n")

        if self.config_file_train is None:
            raise ValueError("No training configuration file specified.\n")

        if not os.path.isfile(self.config_file_train):
            raise FileNotFoundError(
                f"Cannot find the configuration file used during training: {self.config_file_train}\n"
            )

        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"Cannot find the model weights: {self.checkpoint}\n")

        # Read the configuration file used during training
        self.config = config_utils.read_config(self.config_file_train)

        if config_file_test is not None:
            test_config = config_utils.read_config(config_file_test)
            # Training config provides base (architecture, method, etc.),
            # eval config overrides what it specifies (data, mask, etc.)
            self.config = OmegaConf.merge(test_config, self.config)
        # Extract the temporal window size and the number of channels used during training
        if temporal_window is not None:
            self.temporal_window = temporal_window
        else:
            self.temporal_window = self.config.data.max_seq_length
        self.num_channels = num_channels

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _ = torch.set_grad_enabled(False)

        # Get the model
        self.model, _ = utils.get_model(self.config, self.num_channels)
        self._resume()
        self.model.to(self.device).eval()

    def impute_sample(
        self,
        batch: dict[str, Any],
        t_start: int | None = None,
        t_end: int | None = None,
        return_att: bool | None = False,
    ) -> tuple[dict[str, Any], Tensor, Tensor] | tuple[dict[str, Any], Tensor]:

        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch["x"] = batch["x"][:, t_start:t_end, ...]

            for key in ["y", "masks", "cloud_mask", "masks_valid_obs"]:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]

            for key in ["days", "position_days"]:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        # Impute the given satellite image time series
        batch = data_utils.to_device(batch, self.device)
        if return_att:
            y_pred, att = impute_sequence(
                self.model, batch, self.temporal_window,
                return_att=True, blend_mode=self.blend_mode,
                center_only_n_keep=self.center_only_n_keep,
            )
            if att is not None:
                att = att.cpu()
        else:
            y_pred = impute_sequence(
                self.model, batch, self.temporal_window,
                return_att=False, blend_mode=self.blend_mode,
                center_only_n_keep=self.center_only_n_keep,
            )
        batch = data_utils.to_device(batch, "cpu")
        y_pred = y_pred.cpu()

        if return_att:
            return batch, y_pred, att
        return batch, y_pred

    def _resume(self) -> None:
        checkpoint = torch.load(self.checkpoint, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint '{self.checkpoint}' loaded.")
        print(f"Chosen epoch: {checkpoint['epoch']}\n")
        del checkpoint


def _center_weights(window_size: int, device: torch.device) -> Tensor:
    """Triangular weights peaking at the center of the window.

    For window_size=5: [1, 2, 3, 2, 1] (normalized so they sum to 1).
    """
    half = window_size / 2.0
    w = torch.arange(window_size, dtype=torch.float32, device=device)
    w = 1.0 + torch.min(w, torch.tensor(window_size, device=device) - 1.0 - w)
    return w / w.sum()


def _impute_center(model, batch: dict[str, Any], temporal_window: int) -> Tensor:
    """Run center-weighted blending and return the full-length prediction."""
    x = batch["x"]
    positions = batch["position_days"]
    B, T, _, H, W = x.shape
    cloud_coverage = torch.mean(batch["masks"], dim=(0, 2, 3, 4))

    y_accum: Tensor | None = None
    w_accum: Tensor | None = None

    t_start = 0
    t_end = temporal_window
    t_max = T
    reached_end = False

    while not reached_end:
        y_pred_chunk = model(
            x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end]
        )
        if y_accum is None:
            C = y_pred_chunk.shape[2]
            y_accum = torch.zeros((B, T, C, H, W), device=x.device)
            w_accum = torch.zeros((1, T, 1, 1, 1), device=x.device)

        chunk_len = t_end - t_start
        weights = _center_weights(chunk_len, x.device).view(1, chunk_len, 1, 1, 1)
        y_accum[:, t_start:t_end] += y_pred_chunk * weights
        w_accum[:, t_start:t_end] += weights

        if t_end == t_max:
            reached_end = True
        else:
            t_start, t_end = move_temporal_window_next(
                t_start, t_max, temporal_window, cloud_coverage
            )

    return y_accum / w_accum.clamp(min=1e-8)


def _compute_iterative_passes(masks: Tensor, temporal_window: int) -> int:
    """Determine how many iterative passes are needed.

    Analyses the mask tensor to find the longest run of consecutive fully-masked
    dates, then returns ``ceil(longest_gap / (temporal_window // 2))``, capped
    at 6. A single pass is always returned when there is no gap longer than
    half the temporal window.

    Args:
        masks: (B, T, 1, H, W) with 1 = masked.
        temporal_window: sliding-window size used during inference.

    Returns:
        Number of passes (>= 1).
    """
    # A date is "fully masked" when its spatial average >= 0.99
    coverage = masks[0, :, 0].mean(dim=(-2, -1))  # (T,)
    is_masked = (coverage >= 0.99)

    # Find the longest consecutive run of True
    longest = 0
    current = 0
    for m in is_masked.tolist():
        if m:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    half_win = temporal_window // 2
    if longest <= half_win:
        return 1
    n_passes = math.ceil(longest / half_win)
    return min(n_passes, 6)


def impute_sequence(
    model,
    batch: dict[str, Any],
    temporal_window: int,
    return_att: bool = False,
    blend_mode: str = "switch",
    center_only_n_keep: int = 2,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Sliding-window imputation of satellite image time series.

    Args:
        blend_mode: ``"switch"`` (default) uses the original hard-switch at the
            frame with minimum prediction error between overlapping windows.
            ``"center"`` uses center-weighted blending where predictions from the
            middle of each window contribute more than those at the edges.
            ``"center_only"`` keeps only the ``center_only_n_keep`` central frames
            from each window (default 2). The stride equals ``n_keep``, so many
            more inference passes are required, but every date is predicted from
            the most central position possible — ideal for long cloudy stretches.
            ``"iterative"`` runs multiple center-weighted passes; after each pass
            the predictions are injected back into masked S2 channels so the
            next pass sees them as observed data. The number of passes is
            computed dynamically from the longest consecutive masked gap.
        center_only_n_keep: number of central frames to keep per window when
            ``blend_mode="center_only"`` (default 2).

    Assumption: `batch` consists of a single sample.
    """

    x = batch["x"]
    positions = batch["position_days"]
    y_pred: Tensor
    att: Tensor

    if temporal_window is None or x.shape[1] <= temporal_window:
        # Process the entire sequence in one go
        if return_att:
            y_pred, att = model(x, batch_positions=positions, return_att=True)
        else:
            y_pred = model(x, batch_positions=positions)
    elif blend_mode == "center_only":
        # ── Center-only: keep only n_keep central frames per window ──────
        if return_att:
            att = None
        B, T, _, H, W = x.shape
        n_keep = min(center_only_n_keep, temporal_window)
        stride = n_keep  # advance by exactly n_keep frames each pass

        y_pred_init = False
        t_start = 0

        while t_start + temporal_window <= T:
            t_end = t_start + temporal_window

            y_pred_chunk = model(
                x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end]
            )

            if not y_pred_init:
                C = y_pred_chunk.shape[2]
                y_pred = torch.zeros((B, T, C, H, W), device=x.device)
                y_pred_init = True

            # Central n_keep indices within the chunk
            mid = temporal_window // 2
            half_keep = n_keep // 2
            c_start = mid - half_keep
            c_end = c_start + n_keep

            # Map back to global indices
            g_start = t_start + c_start
            g_end = t_start + c_end
            y_pred[:, g_start:g_end] = y_pred_chunk[:, c_start:c_end]

            t_start += stride

        # ── Handle remaining frames at the edges ─────────────────────────
        # Left edge: frames [0, c_start_of_first_window) from the first pass
        first_c_start = (temporal_window // 2) - (n_keep // 2)
        if first_c_start > 0:
            y_pred_chunk = model(
                x[:, :temporal_window], batch_positions=positions[:, :temporal_window]
            )
            y_pred[:, :first_c_start] = y_pred_chunk[:, :first_c_start]

        # Right edge: use the last possible window covering remaining frames
        last_g_end = t_start + (temporal_window // 2) - (n_keep // 2) + n_keep - stride
        # More simply: find which frames at the end are still zero
        if t_start < T:
            # Need to cover frames from (last written g_end) to T
            t_start_last = T - temporal_window
            t_end_last = T
            y_pred_chunk = model(
                x[:, t_start_last:t_end_last],
                batch_positions=positions[:, t_start_last:t_end_last],
            )
            # Fill only the frames not yet written
            already_written_up_to = t_start + (temporal_window // 2) - (n_keep // 2)
            fill_from = max(already_written_up_to, t_start_last)
            y_pred[:, fill_from:T] = y_pred_chunk[:, (fill_from - t_start_last):]
    elif blend_mode == "iterative":
        # ── Iterative multi-pass reconstruction ──────────────────────────
        # Dynamically compute how many passes are needed based on the
        # longest consecutive masked gap.
        if return_att:
            att = None

        n_passes = _compute_iterative_passes(batch["masks"], temporal_window)
        n_s2 = 10  # number of S2 channels in x

        # Work on clones so the original batch is not modified
        x_iter = batch["x"].clone()
        masks_iter = batch["masks"].clone()

        for p in range(n_passes):
            iter_batch = {**batch, "x": x_iter, "masks": masks_iter}
            y_pred = _impute_center(model, iter_batch, temporal_window)

            if p < n_passes - 1:
                # Inject predictions into masked S2 channels for the next pass
                mask_broad = masks_iter.expand_as(x_iter[:, :, :n_s2])
                x_iter[:, :, :n_s2] = torch.where(
                    mask_broad.bool(), y_pred, x_iter[:, :, :n_s2]
                )
                # Clear the mask so the model sees these as "observed"
                masks_iter = torch.zeros_like(masks_iter)
    elif blend_mode == "center":
        # ── Center-weighted blending ─────────────────────────────────────
        if return_att:
            att = None
        y_pred = _impute_center(model, batch, temporal_window)
    else:
        # ── Original hard-switch blending ────────────────────────────────
        if return_att:
            att = None

        t_start = 0
        t_end = temporal_window
        t_max = x.shape[1]
        cloud_coverage = torch.mean(batch["masks"], dim=(0, 2, 3, 4))
        reached_end = False

        while not reached_end:
            y_pred_chunk = model(x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end])

            if t_start == 0:
                # Initialize the full-length output sequence
                B, T, _, H, W = x.shape
                C = y_pred_chunk.shape[2]
                y_pred = torch.zeros((B, T, C, H, W), device=x.device)

                y_pred[:, t_start:t_end] = y_pred_chunk

                # Move the temporal window
                t_start_old = t_start
                t_end_old = t_end
                t_start, t_end = move_temporal_window_next(t_start, t_max, temporal_window, cloud_coverage)
            else:
                # Find the indices of those frames that have been processed by both the previous and the current
                # temporal window
                t_candidates = (
                    torch.Tensor(
                        list(
                            set(torch.arange(t_start_old, t_end_old).tolist())
                            & set(torch.arange(t_start, t_end).tolist())
                        )
                    )
                    .long()
                    .to(x.device)
                )

                # Find the frame for which the difference between the previous and the current prediction is
                # the lowest:
                # use this frame to switch from the previous imputation results to the current imputation results
                error = torch.mean(
                    torch.abs(y_pred[:, t_candidates] - y_pred_chunk[:, t_candidates - t_start]),
                    dim=(0, 2, 3, 4),
                )
                t_switch = error.argmin().item() + t_start
                y_pred[:, t_switch:t_end] = y_pred_chunk[:, (t_switch - t_start) :]

                if t_end == t_max:
                    reached_end = True
                else:
                    # Move the temporal window
                    t_start_old = t_start
                    t_end_old = t_end
                    t_start, t_end = move_temporal_window_next(t_start_old, t_max, temporal_window, cloud_coverage)

    if return_att:
        return y_pred, att
    return y_pred


def move_temporal_window_end(t_max: int, temporal_window: int) -> tuple[int, int]:
    """
    Moves the temporal window for evaluation such that the last frame of the temporal window coincides with the
    last frame of the image sequence.

    Args:
        t_max:              int, sequence length of the image sequence
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    t_start = t_max - temporal_window
    t_end = t_max

    return t_start, t_end


def move_temporal_window_next(
    t_start: int, t_max: int, temporal_window: int, cloud_coverage: Tensor
) -> tuple[int, int]:
    """
    Moves the temporal window for evaluation by half of the temporal window size (= stride).
    If the first frame within the new temporal window is cloudy (cloud coverage above 10%), the temporal window is
    shifted by at most half the stride (backward or forward) such that the first frame is as least cloudy as
    possible.

    Args:
        t_start:            int, frame index, start of the subsequence for processing
        t_max:              int, frame index, t_max - 1 is the last frame of the subsequence for processing
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing
        cloud_coverage:     torch.Tensor, (T,), cloud coverage [-] per frame

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    stride = temporal_window // 2
    t_start += stride

    if t_start + temporal_window > t_max:
        # Reduce the stride such that the end of the temporal window coincides with the end of the entire sequence
        t_start, t_end = move_temporal_window_end(t_max, temporal_window)
    # Check if the start of the next temporal window is mostly cloud-free
    elif cloud_coverage[t_start] <= 0.1:
        # Keep the default stride and ensure that the temporal window does not exceed the sequence length
        t_end = t_start + temporal_window
        if t_end > t_max:
            t_start, t_end = move_temporal_window_end(t_max, temporal_window)
    else:
        # Find the least cloudy frame within [t_start + stride - dt, t_start + stride + dt]
        dt = math.ceil(stride / 2)
        left = max(0, t_start - dt)
        right = min(t_start + dt + 1, t_max)

        # Frame(s) with the lowest cloud coverage within [t_start + stride - dt, t_start + stride + dt]
        t_candidates = (cloud_coverage[left:right] == cloud_coverage[left:right].min()).nonzero(as_tuple=True)[0] + left

        # Take the frame closest to the standard stride
        t_start = t_candidates[torch.abs(t_candidates - t_start).argmin()].item()

        # Ensure that the temporal window does not exceed the sequence length
        t_end = t_start + temporal_window
        if t_end > t_max:
            t_start, t_end = move_temporal_window_end(t_max, temporal_window)

    return t_start, t_end


def upsample_att_maps(att: Tensor, target_shape: tuple[int, int]) -> Tensor:
    """Upsamples the attention masks `att` to the spatial resolution `target_shape`."""

    n_heads, b, t_out, t_in, h, w = att.shape
    attn = att.view(n_heads * b * t_out, t_in, h, w)

    attn = nn.Upsample(size=target_shape, mode="bilinear", align_corners=False)(attn)

    return attn.view(n_heads, b, t_out, t_in, *target_shape)


def visualize_att_for_one_head_across_time(
    seq: Tensor,
    att: Tensor,
    head: int,
    batch: int = 0,
    upsample_att: bool = True,
    indices_rgb: list[int] | list[float] | Tensor | None = None,
    brightness_factor: float = 1,
    fontsize: int = 10,
    scale_individually: bool = False,
) -> matplotlib.figure.Figure:
    """
    Visualizes the attention masks learned by the `head`.th attention head across time.

    Args:
        seq:                    torch.Tensor, B x T x C x H x W, satellite image time series.
        att:                    torch.Tensor, n_head x B x T x T x h x w, attention masks.
        head:                   int, index of the attention head to be visualized.
        batch:                  int, batch index to visualize.
        upsample_att:           bool, True to upsample the attention masks to the spatial resolution of the satellite
                                image time series; False to keep the native spatial resolution of the attention masks.
        indices_rgb:            list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:      float, brightness factor applied to all images in the sequence.
        figsize:                (float, float), figure size.
        fontsize:               int, font size.
        scale_individually:     bool, True to scale the attention masks for each time step individually; False to
                                maintain a common scale across all attention masks and time.

    Returns:
        matplotlib.pyplot.
    """

    indices_rgb = [0, 1, 2] if indices_rgb is None else indices_rgb

    if upsample_att:
        target_shape = seq.shape[-2:]
        att = upsample_att_maps(att, target_shape)

    seq_length = seq.shape[1]
    figsize = (7, 1 + seq_length)
    fig, axes = plt.subplots(nrows=seq_length + 1, ncols=1, figsize=figsize)

    # Plot satellite image time series
    grid = visutils.gallery(seq[batch, :, indices_rgb, :, :], brightness_factor=brightness_factor)
    axes[0].imshow(grid, COLORMAPS["rgb"])
    axes[0].set_title("Input sequence", fontsize=fontsize)

    if scale_individually:
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 1

    # Plot attention mask for attention head `head` across all time steps
    for t in range(seq_length):
        grid = visutils.gallery(att[head, batch, t, :, :, :].unsqueeze(1), brightness_factor=1)
        axes[t + 1].imshow(grid, COLORMAPS["att"], vmin=vmin, vmax=vmax)
        axes[t + 1].set_title(f"Attention mask, head {head}, target frame {t}", fontsize=fontsize)

    for ax in axes.ravel():
        ax.set_axis_off()
    plt.tight_layout()

    return fig


def visualize_att_for_target_t_across_heads(
    seq: Tensor,
    att: Tensor,
    t_target: int,
    batch: int = 0,
    upsample_att: bool = True,
    indices_rgb: list[int] | list[float] | Tensor | None = None,
    brightness_factor: float = 1,
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 200,
    fontsize: int = 10,
    scale_individually: bool = False,
    highlight_t_target: bool = True,
) -> matplotlib.figure.Figure:
    """
    Visualizes the attention masks of all attention heads w.r.t. to the time step `t_target`.

    Args:
        seq:                    torch.Tensor, B x T x C x H x W, satellite image time series.
        att:                    torch.Tensor, n_head x B x T x T x h x w, attention masks.
        t_target:               int, time step (temporal coordinate) to visualize.
        batch:                  int, batch index to visualize.
        upsample_att:           bool, True to upsample the attention masks to the spatial resolution of the satellite
                                image time series; False to keep the native spatial resolution of the attention masks.
        indices_rgb:            list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:      float, brightness factor applied to all images in the sequence.
        figsize:                (float, float), figure size.
        dpi:                    int, dpi of the figure.
        fontsize:               int, font size.
        scale_individually:     bool, True to scale the attention masks for each time step individually; False to
                                maintain a common scale across all attention masks and time.
        highlight_t_target:     bool, True to highlight the target time step by drawing a red frame around the
                                respective image in the time series.

    Returns:
        matplotlib.pyplot.
    """

    indices_rgb = [0, 1, 2] if indices_rgb is None else indices_rgb

    if upsample_att:
        target_shape = seq.shape[-2:]
        att = upsample_att_maps(att, target_shape)

    n_heads = att.shape[0]
    fig, axes = plt.subplots(nrows=n_heads + 1, ncols=1, figsize=figsize, dpi=dpi)

    # Plot input sequence
    grid = visutils.gallery(seq[batch, :, indices_rgb, :, :], brightness_factor=brightness_factor)

    if highlight_t_target:
        # Create a red frame to highlight the target frame
        border_thickness = 2
        H, W = seq.shape[-2:]
        H += 2 * border_thickness
        W += 2 * border_thickness
        frame_color = torch.Tensor([1, 0, 0]).type(grid.dtype)

        if t_target < 0:
            t_target = seq.shape[1] - abs(t_target)

        grid[0 : (2 * border_thickness + 1), t_target * W : (t_target + 1) * W, :] = frame_color
        grid[-2 * border_thickness :, t_target * W : (t_target + 1) * W, :3] = frame_color
        grid[:, t_target * W - border_thickness : t_target * W + border_thickness, :] = frame_color
        grid[
            :,
            ((t_target + 1) * W - border_thickness) : (t_target + 1) * W + border_thickness,
            :,
        ] = frame_color

        if t_target == seq.shape[1] - 1:
            grid[
                :,
                ((t_target + 1) * W - 2 * border_thickness) : (t_target + 1) * W + border_thickness,
                :,
            ] = frame_color
        elif t_target == 0:
            grid[:, 0 : 2 * border_thickness, :] = frame_color

    axes[0].imshow(grid, COLORMAPS["rgb"])
    axes[0].set_title("Input sequence", fontsize=fontsize)

    if scale_individually:
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 1

    # Plot attention masks per head for frame `t_target`
    for head in range(n_heads):
        grid = visutils.gallery(att[head, batch, t_target, :, :, :].unsqueeze(1), brightness_factor=1)
        axes[head + 1].imshow(grid, COLORMAPS["att"], vmin=vmin, vmax=vmax)
        axes[head + 1].set_title(f"Attention mask, head {head}, target frame {t_target}", fontsize=fontsize)

    for ax in axes.ravel():
        ax.set_axis_off()
    plt.tight_layout()

    return fig
