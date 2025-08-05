import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
import ast
import json
import math
import random
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from rasterio.windows import Window
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import dataloader_CIRCA.tools.positional_encoding as encodings
import dataloader_CIRCA.tools.torch_transforms as torch_transforms
from dataloader_CIRCA.datasets import CircaPatchDataSet
from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.tools.mask_generation import (
    masks_init_filling,
    overlay_seq_with_clouds,
)
from dataloader_CIRCA.tools.positional_encoding import (
    get_position_for_positional_encoding,
    str2date,
)
from dataloader_CIRCA.tools.sampling import sample_indices_masked_frames

CHANNEL_CONFIG = ["bgr", "bgr-nir", "all", "bgr-mask", "bgr-nir-mask", "all-mask"]


class UTILISE_Flash(CircaPatchDataSet):

    def __init__(
        self,
        data_optique: Union[str, Path],
        data_radar: Union[str, Path],
        hdf5_file: Union[str, Path] = None,
        image_size: Tuple = (256, 256),
        overlap: Optional[int] = 0,
        load_dataset: Optional[str] = None,
        shuffle: bool = False,
        use_SAR: bool = True,
        filter_settings: dict = None,
        max_seq_length: Optional[int] = None,
        render_occluded_above_p: Optional[float] = None,
        mask_kwargs: Optional[Dict | DictConfig] = None,
        pe_strategy: str = "day-within-sequence",
        augment: Optional[bool] = False,
        crop_settings: Optional[DictConfig] = None,
        return_cloud_mask: bool = True,
        channels: Optional[str] = "all",
    ):

        super().__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            image_size=image_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_SAR=use_SAR,
        )
        self.setup_load(hdf5_file)
        self.setup_channels()
        self.setup_filter_settings(filter_settings, max_seq_length)
        self.render_occluded_above_p = (
            render_occluded_above_p  # Fully occlude images with high cloud cover
        )
        self.setup_mask_kwargs(mask_kwargs)
        self.pe_strategy = pe_strategy
        self.augment = augment
        self.channels = channels

    def setup_load(self, hdf5_file):
        if hdf5_file is not None:
            self.hdf5_file = hdf5_file
            self.f = h5py.File(self.hdf5_file, "r", libver="latest", swmr=True)
            self.load_data = self.load_data_from_hdf5
            self.__len__ = lambda: len(self.f["ROIs"])
        else:
            self.load_data = self.load_data_from_files

    def setup_channels(self):
        self.num_channels = 10
        self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
        self.c_index_nir = torch.Tensor([6]).long()
        self.s2_channels = list(np.arange(10))
        if self.use_SAR:
            self.num_channels += 4

    def setup_filter_settings(
        self,
        filter_settings: Optional[DictConfig] = None,
        max_seq_length: Optional[int] = None,
    ):
        if filter_settings is None:
            self.filter_settings = {
                "type": None,
                "min_length": 5,
                "return_valid_obs_only": False,
                "max_t_sampling": None,
            }

        if isinstance(filter_settings, dict):
            self.filter_settings = OmegaConf.create(filter_settings)
        else:
            self.filter_settings = filter_settings

        if self.filter_settings.get("type", None) is not None:
            self.variable_seq_length = filter_settings.return_valid_obs_only
        else:
            self.variable_seq_length = False

        # Definition en dur de certaines variables
        self.filter_settings.max_num_consec_invalid = self.filter_settings.get(
            "max_num_consec_invalid", None
        )
        self.filter_settings.min_length = self.filter_settings.get("min_length", 0)
        self.filter_settings.max_t_sampling = self.filter_settings.get(
            "max_t_sampling", None
        )
        self.max_seq_length = max_seq_length
        self.seq_length = 30 if self.max_seq_length is None else self.max_seq_length

    def setup_mask_kwargs(self, mask_kwargs: Optional[DictConfig] = None):
        # Parameters used for creating synthetic data gaps
        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)
        if mask_kwargs is not None:
            mask_kwargs.mask_type = mask_kwargs.get("mask_type", "random_clouds")
            mask_kwargs.ratio_masked_frames = mask_kwargs.get(
                "ratio_masked_frames", 0.5
            )
            mask_kwargs.ratio_fully_masked_frames = mask_kwargs.get(
                "ratio_fully_masked_frames", 0.0
            )
            mask_kwargs.non_masked_frames = mask_kwargs.get("non_masked_frames", [])
            self.fill_type = mask_kwargs.get("fill_type", "fill_value")
            self.fill_value = mask_kwargs.get("fill_value", 1)
            self.fixed_masking_ratio = mask_kwargs.get("fixed_masking_ratio", False)
            self.intersect_real_cloud_masks = mask_kwargs.get(
                "intersect_real_cloud_masks", False
            )
            self.dilate_cloud_masks = mask_kwargs.get("dilate_cloud_masks", False)
        self.mask_kwargs = mask_kwargs

    def close(self):
        if hasattr(self, "f"):
            self.f.close()

    def decode_dates(self, dates):
        return np.asanyarray([el.decode("utf-8") for el in dates])

    def load_data_from_hdf5(self, item: int):
        patch = self.f["ROIs"][str(item)]
        return {
            "S1": {
                "S1": patch["S1/S1"][:],
                "S1_dates": self.decode_dates(patch["S1/S1_dates"][:].T[0]),
            },
            "S2": {
                "S2": patch["S2/S2"][:],
                "S2_dates": self.decode_dates(patch["S2/S2_dates"][:].T[0]),
                "cloud_mask": patch["S2/cloud_mask"][:],
                "cloud_prob": patch["S2/cloud_prob"][:],
            },
            "idx_cloudy_frames": patch["idx_cloudy_frames"][:].T[0],
            "idx_good_frames": patch["idx_good_frames"][:].T[0],
            "idx_impaired_frames": patch["idx_impaired_frames"][:].T[0],
            "valid_obs": patch["valid_obs"][:].T[0],
        }

    def load_data_from_files(self, item: int):
        patch_data = self.patches_dataset.iloc[item]
        patch_window = Window(*patch_data.window)
        S2_array = SentinelDataProcessor.read_MS(patch_data.files[0], patch_window)

        patch_S2_array = S2_array[:, :, :, 0:10]
        masks = S2_array[:, :, :, -2:]
        dates_S2 = self.dates_dict[patch_data.mgrs25]["S2"]

        snow_masks, cloud_prob = masks[:, :, :, 0], masks[:, :, :, 1]
        cloud_prob_corrected = SentinelDataProcessor.cloud_mask_correction(cloud_prob)
        TRESHOLD = 1
        cloud_mask = (cloud_prob_corrected > TRESHOLD).astype(np.float32)
        idx_good_frames = SentinelDataProcessor.filter_dates(
            np.stack([snow_masks, cloud_prob_corrected], axis=-1)
        )
        idx_cloudy_frames = np.asarray(
            [d for d in range(len(dates_S2)) if d not in idx_good_frames]
        )
        dates_S1, index_S1, orbit_type = SentinelDataProcessor.get_pairedS1(
            dates_S2,
            self.dates_dict[patch_data.mgrs25]["S1"]["ASC"],
            self.dates_dict[patch_data.mgrs25]["S1"]["DESC"],
        )
        path_S1 = patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
        patch_S1_data = SentinelDataProcessor.read_SAR(path_S1, patch_window)
        bands_S1 = [patch_S1_data[t_index] for t_index in index_S1]
        patch_S1_array = np.stack(bands_S1, axis=0)

        return {
            "S1": {
                "S1": patch_S1_array,
                "S1_dates": dates_S1,
            },
            "S2": {
                "S2": patch_S2_array,
                "S2_dates": dates_S2,
                "cloud_mask": cloud_mask,
                "cloud_prob": cloud_prob_corrected,
            },
            "idx_cloudy_frames": idx_cloudy_frames,
            "idx_good_frames": idx_good_frames,
            "idx_impaired_frames": idx_cloudy_frames,
            "valid_obs": np.asarray(
                [1 if i in idx_good_frames else 0 for i in range(len(dates_S2))]
            ),
        }

    def _longest_consecutive_seq_within_sampling_frequency(
        self, sample: dict
    ) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images, where the temporal sampling between
        consecutive cloud-free images does not exceed `self.filter_settings.max_t_sampling` days.

        Args:
            sample:     Dict.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """
        # Extract the acquisition dates of the cloud-free images within the sequence
        # s2_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['S2/S2_dates']]
        s2_dates = sample["S2"]["S2_dates"]
        t_cloudfree = sample["idx_good_frames"]
        s2_dates = [s2_dates[t] for t in t_cloudfree]

        # Count number of consecutive cloud-free images with temporal sampling of at most
        # `self.filter_settings.max_t_sampling:`
        subseq = {"start": 0, "end": 0, "len": 0}
        count = 1
        start = 0

        for i in range(len(s2_dates) - 1):
            if (
                s2_dates[i + 1] - s2_dates[i]
            ).days <= self.filter_settings.max_t_sampling:
                end = i + 1
                count += 1
                if count > subseq["len"]:
                    subseq["start"] = t_cloudfree[start]
                    subseq["end"] = t_cloudfree[end]
                    subseq["len"] = count
            else:
                start = i + 1
                count = 1
        return subseq

    @staticmethod
    def _longest_consecutive_seq(sample: dict) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images.

        Args:
            sample:      Dict.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """

        idx_frames = sample["idx_good_frames"]
        # Count number of consecutive cloud-free images
        subseq = {"start": 0, "end": 0, "len": 0}
        count = 1
        start = 0
        for i in range(len(idx_frames) - 1):
            if idx_frames[i] + 1 == idx_frames[i + 1]:
                end = i + 1
                count += 1
                if count > subseq["len"]:
                    subseq["start"] = idx_frames[start]
                    subseq["end"] = idx_frames[end]
                    subseq["len"] = count
            else:
                start = i + 1
                count = 1
        return subseq

    def _subsample_sequence(
        self, sample: dict, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
           images is at most `self.filter_settings.max_t_sampling` days,
        3) trims the sequence to a maximum temporal length.

        Args:
            sample:           h5py group.
            seq_length:       int, temporal length of the sample.

        Returns:
            t_sampled:        torch.Tensor, length T.
            masks_valid_obs:  torch.Tensor, (T, ).
        """
        # Generate a mask to exclude invalid frames:
        # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
        if self.filter_settings.type == "cloud-free":
            # Indices of available and cloud-free images
            masks_valid_obs = torch.from_numpy(sample["valid_obs"][:])

            if self.filter_settings.max_t_sampling is not None:
                subseq = self._longest_consecutive_seq_within_sampling_frequency(sample)
                masks_valid_obs[: subseq["start"]] = 0
                masks_valid_obs[subseq["end"] + 1 :] = 0

        elif self.filter_settings.type == "cloud-free_consecutive":
            subseq = self._longest_consecutive_seq(sample)
            masks_valid_obs = torch.from_numpy(sample["valid_obs"][:])
            masks_valid_obs[: subseq["start"]] = 0
            masks_valid_obs[subseq["end"] + 1 :] = 0
        else:
            masks_valid_obs = torch.ones(
                seq_length,
            )

        if self.filter_settings.get("return_valid_obs_only", True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(
                np.arange(0, len(t_sampled) - self.max_seq_length + 1)
            )
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled, masks_valid_obs[t_sampled]

    def _mask_images_with_cloud_coverage_above_p(
        self, cloud_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Marks all pixels of an image as occluded if its cloud coverage exceeds `self.render_occluded_above_p` [-].
        Args:
            cloud_mask: torch.Tensor, (T x 1 x H x W), time series of cloud masks.
        Returns:
            cloud_mask: torch.Tensor, (T x 1 x H x W), updated time series of cloud masks.
        """
        coverage = torch.mean(cloud_mask, dim=(1, 2, 3))
        cloud_mask[coverage > self.render_occluded_above_p, :, :, :] = 1
        return cloud_mask

    def format_data(self, sample: dict):
        """
        Passage de T* H * W * C en T * C * H * W.
        """
        sample["S1"]["S1"] = sample["S1"]["S1"].transpose(0, 3, 1, 2)
        sample["S1"]["S1_dates"] = np.array(
            [str2date(date) for date in sample["S1"]["S1_dates"]]
        )
        sample["S2"]["S2"] = sample["S2"]["S2"].transpose(0, 3, 1, 2)
        sample["S2"]["cloud_mask"] = np.expand_dims(sample["S2"]["cloud_mask"], axis=1)
        sample["S2"]["cloud_prob"] = np.expand_dims(sample["S2"]["cloud_prob"], axis=1)
        sample["S2"]["S2_dates"] = np.array(
            [str2date(date) for date in sample["S2"]["S2_dates"]]
        )
        return sample

    # def augment_data(self, s2, cloud_mask):
    #     data = torch.cat([s2, cloud_mask], dim=1)
    #     data = self.transform(data)
    #     s2, cloud_mask = data[:, :10, :, :],  data[:, 10:, :, :]
    #     return s2, cloud_mask

    def __getitem__(
        self,
        item: int,
        t_sampled: Optional[Tensor] = None,
        t_masked: Optional[Dict[str, np.ndarray]] = None,
    ):
        patch = self.load_data(item)
        patch = self.format_data(patch)
        # Load the entire S2 satellite image time series, T x C x H x W
        s2 = torch.from_numpy(patch["S2"]["S2"].astype(np.float32))
        s2 = SentinelDataProcessor.process_MS(s2)
        cloud_mask = torch.from_numpy(
            patch["S2"]["cloud_mask"]
        )  # .squeeze(1) # Load the cloud masks, T x 1 x H x W => T x H x W

        # Temporally subsample/trim the sequence
        if t_sampled is None:
            t_sampled, masks_valid_obs = self._subsample_sequence(
                patch, seq_length=s2.shape[0]
            )
        else:
            masks_valid_obs = torch.ones(
                len(t_sampled),
            )

        frames_input, frames_target = s2[t_sampled].clone(), s2[t_sampled].clone()
        s2_dates = patch["S2"]["S2_dates"][
            t_sampled
        ]  # Extract the acquisition dates of the temporally trimmed S2 image sequence

        if self.use_SAR:
            s1 = torch.from_numpy(patch["S1"]["S1"].astype(np.float32))[t_sampled]
            s1 = SentinelDataProcessor.process_SAR(s1)
            s1_dates = patch["S1"]["S1_dates"][t_sampled]

            # Concatenate the (masked) S2 bands and the unmasked S1 bands
            frames_input = torch.cat((frames_input, s1), dim=1)

        cloud_mask = cloud_mask[t_sampled]  # T x C x H x W

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.0:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        # Generate masks
        if self.mask_kwargs is not None:
            t_masked, frames_input, masks = self._generate_masks(
                item,
                patch,
                frames_input,
                cloud_mask,
                t_masked,
            )
        else:
            masks = torch.zeros(
                (frames_input.shape[0], 1, *frames_input.shape[-2:])
            )  # T x C x H x W

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = get_position_for_positional_encoding(s2_dates, "day-within-sequence")
        # Get positions for positional encoding
        position_days = get_position_for_positional_encoding(s2_dates, self.pe_strategy)

        # Assemble output
        out = {
            "x": frames_input,  # (synthetically masked) S2 satellite image time series, (T x C x H x W), optionally including S1 bands
            "y": frames_target,  # observed/target satellite image time series, (T x C x H x W)
            "masks": masks,  # masks applied to `x`, (T x 1 x H x W); pixel with value 1 is masked, 0 otherwise
            "masks_valid_obs": masks_valid_obs,  # flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid
            "position_days": position_days,
            "days": days,  # temporal sampling, number of days since the first observation in the sequence, (T, )
            "sample_index": item,
            "filepath": self.patches_dataset.iloc[item].files,
            "c_index_rgb": self.c_index_rgb,
            "c_index_nir": self.c_index_nir,
            "S2_dates": [date.strftime("%Y-%m-%d") for date in s2_dates],
            "cloud_prob": torch.from_numpy(
                patch["S2"]["cloud_prob"].astype(np.float32)
            ),
            "cloud_mask": cloud_mask,
        }
        if self.use_SAR:
            out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in s1_dates]
        return out

    ######################################################################################
    ######################################################################################
    ######################################################################################

    ### FONCTION POUR LA GENERATION DE MASKS ###
    def _generate_masks(
        self,
        id_obs: int,
        sample: dict,
        frames_input: Tensor,
        cloud_mask_input: Tensor,
        t_masked: Dict[str, np.ndarray] | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Tensor, Tensor]:
        """
        Uses a sequence of masks (randomly generated or actual cloud mask sequence) to synthetically generate data gaps
        in the given satellite image time series.

        Args:
            sample:             Dict.
            frames_input:       torch.Tensor, (T x C x H x W), temporally trimmed (subsampled) input image time series.
            cloud_mask_input:   torch.Tensor, (T x 1 x H x W), cloud masks associated with `frames_input`.
            t_masked:           dict, optional, defines two mutually exclusive sets of frame indices:
                                    'indices_masked':        np.ndarray, indices of the (partially) masked frames.
                                    'indices_fully_masked':  np.ndarray, indices of fully masked frames.

        Returns:
            t_masked:           dict, defines two mutually exclusive sets of frame indices:
                                    'indices_masked':        np.ndarray, indices of (partially) masked frames.
                                    'indices_fully_masked':  np.ndarray, indices of fully masked frames.
            frames_input:       torch.Tensor, (T x C x H x W), randomly masked image time series `frames_input`.
            masks:              torch.Tensor, (T x C x H x W), corresponding sequence of masks.
        """

        if self.mask_kwargs.mask_type == "random_clouds":

            if t_masked is None:
                # Indices of the frames to be masked w.r.t. the temporally trimmed sequence
                t_masked = sample_indices_masked_frames(
                    idx_valid_input_frames=np.arange(0, frames_input.shape[0]),
                    ratio_masked_frames=self.mask_kwargs.ratio_masked_frames,
                    ratio_fully_masked_frames=self.mask_kwargs.ratio_fully_masked_frames,
                    non_masked_frames=self.mask_kwargs.non_masked_frames,
                    fixed_masking_ratio=self.fixed_masking_ratio,
                )

            # Randomly sample cloud masks
            sampled_clouds = self._sample_cloud_masks_from_tiles(
                id_obs, sample, len(t_masked["indices_masked"])
            )

            # Generate a sequence of masks
            masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))
            masks[t_masked["indices_masked"], :, :, :] = sampled_clouds

            # Intersect the randomly generated sequence of cloud masks with the actual cloud masks of the sequence
            if self.intersect_real_cloud_masks:
                masks = self._intersect_masks(masks, cloud_mask_input)

            # Apply masking
            frames_input, masks = overlay_seq_with_clouds(
                frames_input,
                masks,
                t_masked=None,
                fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks,
            )

        elif self.mask_kwargs.mask_type == "real_clouds":
            # Use the real cloud masks for masking
            frames_input, masks = masks_init_filling(
                frames_input,
                cloud_mask_input,
                None,
                fill_type="fill_value",
                fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks,
            )
        else:
            raise NotImplementedError

        return t_masked, frames_input, masks

    def _intersect_masks(
        self, masks: torch.Tensor, cloud_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Intersects a randomly generated sequence of cloud masks `masks` with the actual cloud mask sequence of the
        image time series to be masked.

        Args:
            masks:      torch.Tensor, (T x 1 x H x W), sequence of randomly sampled cloud masks.
            cloud_mask: torch.Tensor, (T x 1 x H x W), actual time series of cloud masks.

        Returns:
            masks:      torch.Tensor, (T x 1 x H x W), intersection of `masks` with `cloud_mask`.
        """

        assert masks[0].shape == cloud_mask[0].shape, (
            "Cannot intersect two sequences of masks with unequal temporal " "shape."
        )
        assert masks[-2:].shape == cloud_mask[-2:].shape, (
            "Cannot intersect two sequences of masks with unequal " "spatial shape."
        )
        assert masks[1].shape == cloud_mask[1].shape, (
            "Cannot intersect two sequences of masks with unequal " "spectral shape."
        )

        masks[torch.logical_or(masks > 0.0, cloud_mask == 1)] = 1
        if self.render_occluded_above_p and self.render_occluded_above_p > 0.0:
            masks = self._mask_images_with_cloud_coverage_above_p(masks)
        return masks

    def _mask_images_with_cloud_coverage_above_p(
        self, cloud_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Marks all pixels of an image as occluded if its cloud coverage exceeds `self.render_occluded_above_p` [-].

        Args:
            cloud_mask: torch.Tensor, (T x 1 x H x W), time series of cloud masks.

        Returns:
            cloud_mask: torch.Tensor, (T x 1 x H x W), updated time series of cloud masks.
        """

        coverage = torch.mean(cloud_mask, dim=(1, 2, 3))
        cloud_mask[coverage > self.render_occluded_above_p, :, :, :] = 1
        return cloud_mask

    def _sample_cloud_masks_from_tiles(
        self, id_sample, sample: dict, n: int, p: float = 0.1
    ) -> torch.Tensor:
        """
        Randomly samples `n` cloud masks from a given tile.

        Args:
            sample:  Dict.
            n:       int, number of cloud masks to be sampled.
            p:       float, minimum cloud coverage [-] of the sampled cloud masks.

        Returns:
            cloud_mask:  torch.Tensor, n x 1 x H x W, sampled cloud masks.
        """
        # Retrieve information about the tile from which the sample originates
        sample_info = self.patches_dataset.iloc[id_sample]
        # Extract all samples that originate from the same tile as the given input sample
        samples = self.patches_dataset[
            self.patches_dataset["mgrs25"] == sample_info.mgrs25
        ]
        # Randomly sample `n` cloud masks with cloud coverage of >= p
        cloud_mask = []
        while len(cloud_mask) < n:
            # Extract the cloud masks of a randomly drawn image time series, H x W x 1 x T
            selected_idx = random.choice(samples.index)
            if hasattr(self, "f") and str(selected_idx) in self.f["ROIs"]:
                seq = self.f[f"ROIs/{str(selected_idx)}/S2/cloud_mask"][:]
                seq = torch.from_numpy(
                    np.expand_dims(seq, axis=1).transpose(2, 3, 1, 0)
                )  # T * H * W =>  T * 1 * H * W  => H x W x 1 x T
            else:
                # Extraction à la volée des masks dans les fichiers plats
                cloud_mask_data = SentinelDataProcessor.read_cloud_mask(
                    path_raster=self.patches_dataset.iloc[selected_idx].files[0],
                    window=Window(*self.patches_dataset.iloc[selected_idx].window),
                )
                seq = torch.from_numpy(cloud_mask_data).float()

            # Compute cloud coverage per frame
            coverage = torch.mean(seq, dim=(0, 1, 2))
            indices = torch.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[:, :, :, np.random.choice(indices)])

        # n x 1 x H x W
        cloud_mask = torch.stack(cloud_mask, dim=3).permute(3, 2, 0, 1)

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.0:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask

    def _subsample_sequence(
        self, sample: Dict, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) removes invalid time steps (i.e., no observation, black image),
        3) trims the sequence to a maximum temporal length.

        Args:
            sample:           h5py group.
            seq_length:       int, temporal length of the sample.

        Returns:
            t_sampled:        torch.Tensor, length T.
            masks_valid_obs:  torch.Tensor, (T, ).
        """

        # Generate a mask to exclude invalid frames:
        # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
        if self.filter_settings.type == "cloud-free":
            # Indices of available and cloud-free images
            masks_valid_obs = torch.from_numpy(sample["valid_obs"][:])

        elif self.filter_settings.type == "cloud-free_consecutive":
            subseq = self._longest_consecutive_seq(sample)
            masks_valid_obs = torch.from_numpy(sample["valid_obs"][:])
            masks_valid_obs[: subseq["start"]] = 0
            masks_valid_obs[subseq["end"] + 1 :] = 0
        else:
            masks_valid_obs = torch.ones(
                seq_length,
            )

        if self.filter_settings.get("return_valid_obs_only", True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(
                np.arange(0, len(t_sampled) - self.max_seq_length + 1)
            )
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled, masks_valid_obs[t_sampled]

    @staticmethod
    def _longest_consecutive_seq(sample: Dict) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images.

        Args:
            sample:      Dict.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """
        idx_frames = sample["idx_good_frames"][:]

        # Count number of consecutive cloud-free images
        subseq = {"start": 0, "end": 0, "len": 0}
        count = 1
        start = 0

        for i in range(len(idx_frames) - 1):
            if idx_frames[i] + 1 == idx_frames[i + 1]:
                end = i + 1
                count += 1
                if count > subseq["len"]:
                    subseq["start"] = idx_frames[start]
                    subseq["end"] = idx_frames[end]
                    subseq["len"] = count
            else:
                start = i + 1
                count = 1
        return subseq


######################################################################################
######################################################################################
######################################################################################


if __name__ == "__main__":

    store_dai = Path("/home/SPeillet/Partage/store-dai")
    path_dataset_circa = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_circa / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_circa / "Data_Raster" / "radar_dataset_v4"
    image_size = (256, 256)
    overlap = 0

    mask_kwargs = {
        "mask_type": "random_clouds",  # Strategy for synthetic data gap generation. ['random_clouds', 'real_clouds']
        "ratio_masked_frames": 0.5,  # Ratio of partially/fully masked images in a satellite image time series (upper bound).
        "ratio_fully_masked_frames": 0.0,  # Ratio of fully masked images in a satellite image time series (upper bound).
        "fixed_masking_ratio": True,  # False de base, True to vary the masking ratio across satellite image time series, False otherwise.
        "non_masked_frames": [
            0
        ],  # list of int, time steps to be excluded from masking.
        "intersect_real_cloud_masks": False,  # True to intersect randomly sampled cloud masks with the actual cloud mask sequence, False otherwise.
        "dilate_cloud_masks": False,  # True to dilate the cloud masks before masking, False otherwise.
        "fill_type": "fill_value",  # Strategy for initializing masked pixels. ['fill_value', 'white_noise', 'mean']
        "fill_value": 1,  # Pixel value of masked pixels. Used if fill_type == 'fill_value'.
    }

    ds = UTILISE_Flash(
        data_optique=data_optique,
        data_radar=data_radar,
        # hdf5_file=(store_dai / "tmp/speillet/debug_circa.hdf5"),
        load_dataset="datasetCIRCAUnCRtainTS.csv",
        image_size=image_size,
        overlap=overlap,
        shuffle=False,
        use_SAR=True,
        mask_kwargs=mask_kwargs,
        # channels="all",
        # return_cloud_prob=False,
        # render_occluded_above_p=None,
        # augment=False,
        # seq_length=30,
        # pe_strategy='day-of-year',
    )

    # ds.setup()
    # ds.export_dataset()
    sample = next(iter(ds))
    print(sample.keys())

    ds.close()
