from typing import Union

import numpy as np
import torch


def compute_tensor_histogram(
    tensor: Union[torch.Tensor, np.ndarray],
    n_bins: int = 100,
    range_min: int = np.iinfo(np.int16).min,
    range_max: int = np.iinfo(np.int16).max
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of values in a tensor (T x C x H x W) within specified range.
    
    Args:
        tensor: Input tensor of shape (T x C x H x W)
        n_bins: Number of bins for the histogram
        range_min: Minimum value of the range (default: np.int16 min)
        range_max: Maximum value of the range (default: np.int16 max)
        
    Returns:
        tuple containing:
            - hist: Array of histogram values
            - bin_edges: Array of bin edges including the rightmost edge
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Flatten the tensor to 1D array
    flat_tensor = tensor.reshape(-1)

    # Compute histogram
    hist, bin_edges = np.histogram(
        flat_tensor,
        bins=n_bins,
        range=(range_min, range_max)
    )

    return hist, bin_edges


def set_n_bins():
    n_bins = np.iinfo(np.int16).max - np.iinfo(np.int16).min + 1
    range_min = np.iinfo(np.int16).min
    range_max = np.iinfo(np.int16).max
    return n_bins, range_min, range_max


if __name__ == "__main__":
    n_bins, range_min, range_max = set_n_bins()
    hist, bin_edges = compute_tensor_histogram(
        np.random.randint(-32768, 32767, (10, 10)),
        n_bins,
        range_min,
        range_max,
    )
    print(hist.shape)
    print(bin_edges.shape)
