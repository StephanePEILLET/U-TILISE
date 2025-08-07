import datetime
import itertools
from pathlib import Path
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch import Tensor

# Constants for the number of channels in Sentinel-2 and Sentinel-1 data
S2_N_CHANNELS = 12
S1_N_CHANNELS = 4
INDEX_CLOUD_BAND = 10


class SentinelDataProcessor:
    """
    Utility class for processing Sentinel data.
    """

    @staticmethod
    def read_MS(path_raster: str, window: Window) -> np.ndarray:
        """
        Reads and processes multispectral (MS) data from a raster file.

        Parameters:
        - path_raster (str): Path to the raster file.
        - window (rasterio.windows.Window): Window defining the region of interest.

        Returns:
        - np.ndarray: Processed MS data with shape (T, C, H, W), where:
            - T: Number of time steps.
            - H: Height of the patch.
            - W: Width of the patch.
            - C: Number of spectral channels.
        """
        with rasterio.open(path_raster) as src_S2:
            patch_S2_array = src_S2.read(window=window)
            patch_S2_array = SentinelDataProcessor.reshape_sentinel(patch_S2_array, chunk_size=S2_N_CHANNELS)
            return patch_S2_array

    @staticmethod
    def read_SAR(path_raster: str, window: Window) -> np.ndarray:
        """
        Reads and processes SAR data from a raster file.

        Parameters:
        - path_raster (str): Path to the raster file.
        - window (rasterio.windows.Window): Window defining the region of interest.

        Returns:
        - np.ndarray: Processed SAR data with shape (T, H, W, C), where:
            - T: Number of time steps.
            - H: Height of the patch.
            - W: Width of the patch.
            - C: Number of SAR channels.
        """
        with rasterio.open(path_raster) as src_S1:
            patch_S1_array = src_S1.read(window=window)
            patch_S1_array = SentinelDataProcessor.reshape_sentinel(
                patch_S1_array, chunk_size=S1_N_CHANNELS
            )  # (T * C, H, W) => (T, C, H, W)
            return patch_S1_array

    @staticmethod
    def read_raster_per_dates(
        path_raster: str,
        window: Window = None,
        indexes_dates=None,
        type_bands: str = None,
    ) -> np.ndarray:

        bands_span = {"s2": S2_N_CHANNELS, "s1": S1_N_CHANNELS, "s2_bands": 10}
        span = bands_span[type_bands]

        with rasterio.open(path_raster) as src_S2:
            if indexes_dates is None:
                raster_array = src_S2.read(window=window)
            else:
                indexes_bands = []
                for index_date in indexes_dates:
                    indexes_bands += [index_date + x for x in np.arange(1, span + 1)]
                raster_array = src_S2.read(indexes_bands, window=window)
            return SentinelDataProcessor.reshape_sentinel(raster_array, chunk_size=span)

    @staticmethod
    def read_mask_prob(
        path_raster: str,
        window: Window = None,
        type_mask: Literal["cloud", "snow"] = "cloud",
        mask_band_index: int = None,
    ) -> np.ndarray:
        if mask_band_index is None:
            index_band = {"cloud": 10, "snow": 11}
            mask_band_index = index_band[type_mask]

        with rasterio.open(path_raster) as src_S2:
            len_ts_stacked = src_S2.count // S2_N_CHANNELS
            cloud_bands = [(x * S2_N_CHANNELS + mask_band_index) + 1 for x in np.arange(len_ts_stacked)]
            if window is None:
                cloud_prob = src_S2.read(cloud_bands).transpose(1, 2, 0)
            else:
                cloud_prob = src_S2.read(cloud_bands, window=window).transpose(1, 2, 0)
            return np.expand_dims(cloud_prob, axis=2)  # H x W X 1 X T

    @staticmethod
    def read_cloud_mask(path_raster: str, window: Window, cloud_band_index: int = 10) -> np.ndarray:
        cloud_prob = SentinelDataProcessor.read_mask_prob(
            path_raster=path_raster,
            window=window,
            type_mask="cloud",
        )
        return (cloud_prob != 0).astype(int)  # H x W X 1 X T

    @staticmethod
    def reshape_sentinel(arr: np.ndarray, chunk_size: int = 10) -> np.ndarray:
        """
        Reshapes a temporally stacked Sentinel array into chunks.

        Parameters:
        - arr (np.ndarray): Input array with temporal data.
        - chunk_size (int, optional): Number of time steps per chunk. Defaults to 10.

        Returns:
        - np.ndarray: Reshaped array with shape (n_chunks, chunk_size, height, width).
        """
        first_dim_size = arr.shape[0] // chunk_size
        return arr.reshape((first_dim_size, chunk_size, *arr.shape[1:]))

    @staticmethod
    def get_img_windows_list(
        img_shape: Tuple[int, int], tile_size: Tuple[int, int], overlap: int = 0
    ) -> List[Tuple[int, int, int, int]]:
        """
        Compute patches windows from an image with overlap on all sides.
        Return a list of coordinates for each window. All patches are entirely within the image.

        Parameters:
        - img_shape (Tuple[int, int]): Size of the input image (height, width).
        - tile_size (Tuple[int, int]): Size of the output patches .
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.

        Returns:
        - List[Tuple[int, int, int, int]]: List of coordinates (col_off, row_off, width, height).
        """
        height, width = img_shape
        stride_col = tile_size[0] - overlap  # Calculate stride based on overlap
        stride_row = tile_size[1] - overlap

        # Calculate the starting points for rows and columns
        col_steps = list(range(0, width - tile_size[0] + 1, stride_col))
        row_steps = list(range(0, height - tile_size[1] + 1, stride_row))

        # Ensure the last patch covers the edge of the image
        if (width - tile_size[0]) % stride_col != 0:
            col_steps.append(width - tile_size[0])
        if (height - tile_size[1]) % stride_row != 0:
            row_steps.append(height - tile_size[1])

        # Generate all combinations of row and column steps
        windows_list = [(col, row, tile_size[0], tile_size[1]) for col, row in itertools.product(col_steps, row_steps)]

        return windows_list

    @staticmethod
    def split_raster_into_windows(
        path_raster: Union[str, Path],
        image_size: Tuple[int, int],
        overlap: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Splits a raster into windows of a specified patch size.

        Parameters:
        - path_raster (Union[str, Path]): Path to the raster file.
        - image_size (int): Size of the patches to create.
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.

        Returns:
        - List[Tuple[int, int, int, int]]: List of window coordinates.
        """
        with rasterio.open(path_raster) as dataset:
            return SentinelDataProcessor.get_img_windows_list(
                img_shape=(dataset.height, dataset.width),
                tile_size=image_size,
                overlap=overlap,
            )

    @staticmethod
    def get_datetime(date: str) -> datetime.datetime:
        """
        Converts a date string in 'YYYYMMDD' format to a datetime object.

        Parameters:
        - date (str): Date string in 'YYYYMMDD' format.

        Returns:
        - datetime.datetime: Corresponding datetime object.
        """
        return datetime.datetime.strptime(date, "%Y%m%d")

    @staticmethod
    def cloud_mask_correction(input_mask: np.ndarray, threshold: int = 50) -> np.ndarray:
        """
        Corrects cloud masks by identifying and removing inconsistent pixels based on temporal statistics.

        Parameters:
        - input_mask (np.ndarray): Input cloud masks with shape (T, H, W), where:
            - T: Number of time steps.
            - H: Height of the mask.
            - W: Width of the mask.
        - threshold (int, optional): Percentile threshold for identifying inconsistent pixels. Defaults to 60.

        Returns:
        - np.ndarray: Corrected cloud masks with the same shape as input_mask.
        """

        def compute_persistence_mask(array: np.ndarray) -> np.ndarray:
            """
            Binarizes the input array (non-zero values become 1) and sums along the time axis.

            Parameters:
            - array (np.ndarray): Input array with shape (T, H, W).

            Returns:
            - np.ndarray: Summed array with shape (H, W).
            """
            binary_array = (array != 0).astype(int)  # Binarisation
            return binary_array.sum(axis=0)

        cloud_masks = input_mask.copy()
        persistence_mask = compute_persistence_mask(cloud_masks)  # Stationnarité temporelle
        pixel_threshold = np.percentile(persistence_mask, threshold)
        error_indexes = np.where(persistence_mask > pixel_threshold)
        cloud_masks[:, error_indexes[0], error_indexes[1]] = 0
        return np.stack(cloud_masks, axis=0)

    @staticmethod
    def filter_dates(
        masks: np.ndarray,
        max_fraction_covered: float = 0.05,
    ) -> np.ndarray:
        """
        Filters dates based on cloud and snow coverage.

        Parameters:
        - masks (np.ndarray): Array containing cloud and snow masks. T * H * W * 2
        - max_cloud_value (int, optional): Maximum allowed cloud value. Defaults to 10.
        - max_snow_value (int, optional): Maximum allowed snow value. Defaults to 10.
        - max_fraction_covered (float, optional): Maximum fraction of the image covered by clouds or snow. Defaults to 0.05.

        Returns:
        - np.ndarray: Indices of the selected dates.
        """
        MAX_CLOUD_VALUE = MAX_SNOW_VALUE = 1
        T, H, W, _ = masks.shape
        select = (masks[:, :, :, 0] <= MAX_SNOW_VALUE) & (masks[:, :, :, 1] <= MAX_CLOUD_VALUE)
        num_pix = H * W
        threshold = (1 - max_fraction_covered) * num_pix
        selected_days = np.sum(select, axis=(1, 2)) >= threshold
        # Boucle while permettant de diminuer le seuillage de sélection tant que l'on obtient un nombre de dates filtrées trop bas.
        while np.sum(selected_days) <= 0.1 * T and max_fraction_covered < 1.0:
            max_fraction_covered += 0.05
            threshold = (1 - max_fraction_covered) * num_pix
            selected_days = np.sum(select, axis=(1, 2)) >= threshold
        return np.where(selected_days)[0]

    @staticmethod
    def extract_and_transform_S2(
        S2_array: np.ndarray,
        dates: List[str],
        S2_channels_selected: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts and transforms Sentinel-2 data by filtering cloudy dates and correcting cloud masks.

        Parameters:
        - S2_array (np.ndarray): Input Sentinel-2 data with shape (T, H, W, C), where:
            - T: Number of time steps.
            - H: Height of the patch.
            - W: Width of the patch.
            - C: Number of spectral bands and masks.
        - dates (List[str]): List of dates corresponding to the time steps in S2_array.
        - S2_channels_selected (List[str]): List of bands index to extract.
        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - patch_S2_data: Filtered Sentinel-2 data with shape (T_filtered, H, W, 10).
            - dates_filtered: Filtered dates corresponding to the remaining time steps.
            - cloud_masks_corrected: Corrected cloud masks with shape (T_filtered, H, W).
        """
        # Filtrage des dates S2 nuageuses
        masks = S2_array[:, :, :, -2:]
        snow_masks, cloud_masks = masks[:, :, :, 0], masks[:, :, :, 1]

        if S2_channels_selected is None:
            patch_S2_data = S2_array[:, :, :, 0:10]
        else:
            patch_S2_data = S2_array[:, :, :, S2_channels_selected]

        cloud_masks_corrected = SentinelDataProcessor.cloud_mask_correction(cloud_masks)
        index_S2_curated = SentinelDataProcessor.filter_dates(np.stack([snow_masks, cloud_masks_corrected], axis=-1))
        return (
            patch_S2_data[index_S2_curated],
            np.asarray(dates)[index_S2_curated],
            cloud_masks_corrected[index_S2_curated],
        )

    @staticmethod
    def get_pairedS1(
        dates_S2: List[str],
        dates_S1_asc: List[str],
        dates_S1_desc: List[str],
    ) -> Tuple[List[str], List[int], str]:
        """
        Pairs Sentinel-1 data with Sentinel-2 data based on the closest dates.

        Parameters:
        - dates_S2 (List[str]): List of Sentinel-2 dates.
        - dates_S1_asc (List[str]): List of Sentinel-1 ascendant dates.
        - dates_S1_desc (List[str]): List of Sentinel-1 descendant dates.

        Returns:
        - Tuple[List[str], List[int], str]: A tuple containing the curated dates, indices, and the orbit type of the radar file.
        """
        deltas_S1_ASC, deltas_S1_DESC = [], []
        index_S1_ASC, index_S1_DESC = [], []
        dates_S1_ASC_curated, dates_S1_DESC_curated = [], []

        for date_S2 in dates_S2:
            deltas_asc = [
                (SentinelDataProcessor.get_datetime(date_S2) - SentinelDataProcessor.get_datetime(date_S1)).days
                for date_S1 in dates_S1_asc
            ]
            deltas_desc = [
                (SentinelDataProcessor.get_datetime(date_S2) - SentinelDataProcessor.get_datetime(date_S1)).days
                for date_S1 in dates_S1_desc
            ]
            deltas_S1_ASC.append(np.min(np.abs(deltas_asc)))
            deltas_S1_DESC.append(np.min(np.abs(deltas_desc)))

            idx_min_asc = np.argmin(np.abs(deltas_asc))
            index_S1_ASC.append(idx_min_asc)
            dates_S1_ASC_curated.append(dates_S1_asc[idx_min_asc])

            idx_min_desc = np.argmin(np.abs(deltas_desc))
            index_S1_DESC.append(idx_min_desc)
            dates_S1_DESC_curated.append(dates_S1_desc[idx_min_desc])

        if sum(deltas_S1_ASC) <= sum(deltas_S1_DESC):
            return dates_S1_ASC_curated, index_S1_ASC, "ASC"
        else:
            return dates_S1_DESC_curated, index_S1_DESC, "DESC"

    @staticmethod
    def process_MS(img: Tensor) -> Tensor:
        """Modified from: https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py#L33"""
        # Intensity clipping to a global unified MS intensity range
        intensity_min, intensity_max = 0, 10000
        img = torch.clamp(img, min=intensity_min, max=intensity_max)
        # Project to [0,1], preserve global intensities (across patches)
        img = SentinelDataProcessor.rescale(img, intensity_min, intensity_max)
        return img

    @staticmethod
    def process_SAR(img: Tensor) -> Tensor:
        """Source: https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py#L44"""

        # Intensity clipping to a global unified SAR dB range
        dB_min, dB_max = -25, 0
        img = torch.clamp(img, min=dB_min, max=dB_max)

        # Project to [0,1], preserve global intensities (across patches)
        img = SentinelDataProcessor.rescale(img, dB_min, dB_max)
        return img

    @staticmethod
    def rescale(img: Tensor, old_min: float, old_max: float) -> Tensor:
        """Source: https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py#L28"""
        old_range = old_max - old_min
        img = (img - old_min) / old_range
        return img
