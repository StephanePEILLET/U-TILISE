import datetime as dt
import math
import os
import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from torch import Tensor
from torchvision import transforms


def sampling_consecutive_frames(
    idx_valid_input_frames: np.ndarray,
    ratio_masked_frames: float = None,
    fixed_masking_ratio: bool = False,
) -> np.ndarray:
    """
    Samples a sequence of `num_input_frames` consecutive frames from a total of `num_total_frames` frames.
    Args:
        num_total_frames:    int, total number of frames available for sampling.
        ratio_masked_frames: float or None, ratio of frames to be masked.
        fixed_masking_ratio: bool, True to enforce the same ratio of masked frames across image time sequences,

    Returns:
        np.ndarray, indices of the sampled frames.
    """
    num_total_frames = len(idx_valid_input_frames)
    if not fixed_masking_ratio:
        # Vary the sampling ratio by adjusting the number of frames available for masking
        # (at least one frame has to be masked)
        num_total_frames = random.randint(1, num_total_frames)
    if ratio_masked_frames is None:
        num_input_frames = (num_total_frames // 4) + 1
    else:
        num_input_frames = np.ceil(ratio_masked_frames * num_total_frames)

    if num_input_frames > num_total_frames:
        raise ValueError(f"Cannot sample {num_input_frames} frames from a total of {num_total_frames} frames.")

    start_frame = random.randint(0, num_total_frames - num_input_frames)
    if start_frame + num_input_frames > num_total_frames:
        indices_masked = np.arange(num_total_frames - num_input_frames, num_total_frames)
    elif start_frame + num_input_frames <= num_total_frames:
        indices_masked = np.arange(start_frame, start_frame + num_input_frames)
    elif start_frame + num_input_frames == num_total_frames:
        indices_masked = np.arange(start_frame, num_total_frames)
    else:
        raise ValueError("Something went wrong when sampling consecutive frames.")

    return {
        "indices_masked": indices_masked,
        "indices_fully_masked": indices_masked,
    }


def sample_indices_masked_frames(
    idx_valid_input_frames: np.ndarray,
    ratio_masked_frames: float = 0.5,
    ratio_fully_masked_frames: float = 0.0,
    non_masked_frames: Optional[List[int]] = None,
    fixed_masking_ratio: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generates a sequence of `masks` to synthetically mask an image time series. masks[t1, 0, y1, x1] == 1 will mask the
    spatio-temporal location (t1, y1, x1), whereas masks[t2, 0, y2, x2] == 0 will retain the observed reflectance at
    the spatio-temporal location (t2, y2,x2) (w.r.t. all spectral channels).

    Args:
        idx_valid_input_frames:      np.ndarray, indices of those frames that are available for masking.
        ratio_masked_frames:         float, ratio of (partially or fully) masked frames.
        ratio_fully_masked_frames:   float, ratio of fully masked frames.
        non_masked_frames:           list of int, indices of those frames that should be excluded from masking
                                     (e.g., first frame).
        fixed_masking_ratio:         bool, True to enforce the same ratio of masked frames across image time sequences,
                                     False to vary the ratio of masked frames across image time sequences.
                                     For varying sampling ratios: `ratio_masked_frames` and `ratio_fully_masked_frames`
                                     define upper bounds.

    Returns:
        dict, defines two mutually exclusive sets of frame indices sampled from `idx_valid_input_frames`:
            'indices_masked':        np.ndarray, indices of (partially) masked frames.
            'indices_fully_masked':  np.ndarray, indices of fully masked frames.
    """

    assert ratio_fully_masked_frames <= ratio_masked_frames, (
        "Masking parameter `ratio_fully_masked_frames` needs to " "be smaller or equal to `ratio_masked_frames.`"
    )

    # Upper bound: Maximum number of masked input frames (partially or fully masked)
    num_total = len(idx_valid_input_frames)

    if not fixed_masking_ratio:
        # Vary the sampling ratio by adjusting the number of frames available for masking
        # (at least one frame has to be masked)
        num_total = random.randint(1, num_total)

    # Number of masked frames (partially or fully masked)
    num_masked = math.ceil(ratio_masked_frames * num_total)

    # Number of fully masked frames
    num_fully_masked = math.ceil(ratio_fully_masked_frames * num_total)

    # Randomly select the indices of those frames that will be masked (partially or fully)
    if non_masked_frames is not None:
        non_masked_frames = np.asarray(non_masked_frames)
        if np.any(non_masked_frames < 0):
            # Account for negative indices
            indices_pos = non_masked_frames[non_masked_frames >= 0]
            indices_neg = idx_valid_input_frames[non_masked_frames[non_masked_frames < 0]]
            non_masked_frames = np.concatenate((indices_pos, indices_neg), axis=0)
        else:
            non_masked_frames = idx_valid_input_frames[non_masked_frames]
        list_frames = np.setdiff1d(idx_valid_input_frames, non_masked_frames)
        indices_masked = np.random.choice(list_frames, min(num_masked, list_frames.size), replace=False)
    else:
        indices_masked = np.random.choice(idx_valid_input_frames, num_masked, replace=False)

    # Randomly selected the frame indices of the fully masked frames
    indices_fully_masked = np.random.choice(indices_masked, num_fully_masked, replace=False)

    return {
        "indices_masked": indices_masked,
        "indices_fully_masked": indices_fully_masked,
    }
