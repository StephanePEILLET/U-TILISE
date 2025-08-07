import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import albumentations as A
import h5py
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import Tensor
from tqdm.auto import tqdm

from dataloader_CIRCA.datasets import CIRCA_from_HDF5
from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.tools.mask_generation import masks_init_filling
from dataloader_CIRCA.tools.mask_generation import overlay_seq_with_clouds
from dataloader_CIRCA.tools.positional_encoding import get_pairwise_representative_dates
from dataloader_CIRCA.tools.positional_encoding import get_position_for_positional_encoding  # NOQA
from dataloader_CIRCA.tools.positional_encoding import str2date
from dataloader_CIRCA.tools.sampling import sample_indices_masked_frames

MAX_SEQ_LENGTH = 30
SEED = 42

import datetime as dt
from typing import Dict
from typing import List
from typing import Literal

DateArray = np.ndarray[dt.date]
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
SampleDict = Dict[str, Union[np.ndarray, Dict[str, np.ndarray], List[str]]]
PhaseType = Literal["train", "val", "test", "train+val", "all"]
ChannelType = Literal["all", "bgr-nir"]


class CIRCA_ADAPTED2UTILISE_Dataset(CIRCA_from_HDF5):
    """
    Dataset qui exporte / ou importe les données CIRCA dans / depuis un fichier HDF5.
    """

    def __init__(
        self,
        # CIRCA_from_HDF5 parameters
        phase: PhaseType = "all",
        hdf5_file: Optional[Union[str, Path]] = None,
        shuffle: bool = False,
        use_sar: bool = True,
        channels: ChannelType = "all",
        # U-TILISE specific parameters
        filter_settings: dict = None,
        max_seq_length: Optional[int] = MAX_SEQ_LENGTH,
        render_occluded_above_p: Optional[float] = None,
        mask_kwargs: Optional[dict | DictConfig] = None,
        pe_strategy: str = "day-within-sequence",
        augment: Optional[bool] = False,
        process_data: Optional[bool] = True,
        stats: Optional[DictConfig] = None,
        seed: int = SEED,
        # Récupération de vieux arguments du repo
        crop_settings: Optional[DictConfig] = None,
        return_cloud_mask: bool = True,
    ):
        # Initialize the random seed for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        super().__init__(
            phase=phase,
            hdf5_file=hdf5_file,
            shuffle=shuffle,
            use_sar=use_sar,
            channels=channels,
        )
        self.crop_settings = crop_settings
        self.return_cloud_mask = return_cloud_mask
        self.transform = None
        self.process_data = process_data

        if stats is not None and isinstance(stats, DictConfig):
            self.stats = stats
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=self.stats["means"],
                        std=self.stats["stds"],
                        max_pixel_value=255.0,
                    ),
                    A.pytorch.transforms.ToTensorV2(),
                ]
            )

        self.render_occluded_above_p = render_occluded_above_p  # Fully occlude images with high cloud cover
        self.pe_strategy = pe_strategy
        self.augment = augment

        (
            self.filter_settings,
            self.variable_seq_length,
            self.seq_length,
            self.max_seq_length,
        ) = self.setup_filter_settings(
            filter_settings=filter_settings,
            max_seq_length=max_seq_length,
        )
        (
            self.mask_kwargs,
            self.fill_type,
            self.fill_value,
            self.fixed_masking_ratio,
            self.intersect_real_cloud_masks,
            self.dilate_cloud_masks,
        ) = self.setup_mask_kwargs(mask_kwargs)

    def setup_filter_settings(
        self,
        filter_settings: Optional[DictConfig] = None,
        max_seq_length: Optional[int] = None,
    ):
        if filter_settings is None:
            filter_settings = {
                "type": None,
                "min_length": 5,
                "return_valid_obs_only": False,
                "max_t_sampling": None,
                "p_filter": 0.1,
            }

        if isinstance(filter_settings, dict):
            filter_settings = OmegaConf.create(filter_settings)

        if filter_settings.get("type", None):
            variable_seq_length = filter_settings.return_valid_obs_only
        else:
            variable_seq_length = False

        # Definition en dur de certaines variables
        filter_settings.max_num_consec_invalid = filter_settings.get("max_num_consec_invalid", None)
        filter_settings.min_length = filter_settings.get("min_length", 0)
        filter_settings.max_t_sampling = filter_settings.get("max_t_sampling", None)
        seq_length = MAX_SEQ_LENGTH if max_seq_length is None else max_seq_length
        return filter_settings, variable_seq_length, seq_length, max_seq_length

    def setup_mask_kwargs(self, mask_kwargs: Optional[DictConfig] = None):
        # Parameters used for creating synthetic data gaps

        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)

        if mask_kwargs is None:
            mask_kwargs = {
                # Mask the input time series with randomly sampled cloud masks or the actual cloud masks.
                # ['random_clouds', 'real_clouds']
                "mask_type": "random_clouds",
                # Ratio of partially/fully masked images per image time series (upper bound).
                "ratio_masked_frames": 0.5,
                # Ratio of fully masked images per image time series (upper bound).
                "ratio_fully_masked_frames": 0.0,
                # True to vary the masking ratio across different image time series, False otherwise.
                "fixed_masking_ratio": False,
                # list of int, time steps to be excluded from masking. E.g., [0] never masks the first frame in a seq.
                "non_masked_frames": [0],
                # True to intersect randomly sampled cloud masks with the actual cloud masks, False otherwise.
                "intersect_real_cloud_masks": False,
                "dilate_cloud_masks": False,  # True to dilate the cloud masks before masking, False otherwise.
                # Strategy for initializing masked pixels. ['fill_value', 'white_noise', 'mean']
                "fill_type": "fill_value",
                # Pixel value of masked pixels. Used if fill_type == 'fill_value'.
                "fill_value": 1,
            }
            mask_kwargs = OmegaConf.create(mask_kwargs)
            mask_kwargs.mask_type = mask_kwargs.get("mask_type", "random_clouds")
            mask_kwargs.ratio_masked_frames = mask_kwargs.get("ratio_masked_frames", 0.5)
            mask_kwargs.ratio_fully_masked_frames = mask_kwargs.get("ratio_fully_masked_frames", 0.0)
            mask_kwargs.non_masked_frames = mask_kwargs.get("non_masked_frames", [])

        fill_type = mask_kwargs.get("fill_type", "fill_value")
        fill_value = mask_kwargs.get("fill_value", 1)
        fixed_masking_ratio = mask_kwargs.get("fixed_masking_ratio", False)
        intersect_real_cloud_masks = mask_kwargs.get("intersect_real_cloud_masks", False)
        dilate_cloud_masks = mask_kwargs.get("dilate_cloud_masks", False)
        return (
            mask_kwargs,
            fill_type,
            fill_value,
            fixed_masking_ratio,
            intersect_real_cloud_masks,
            dilate_cloud_masks,
        )

    def _longest_consecutive_seq_within_sampling_frequency(
        self, s2_dates, idx_good_frames, max_t_sampling
    ) -> dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images, where the temporal sampling between
        consecutive cloud-free images does not exceed `self.filter_settings.max_t_sampling` days.

        Args:
            sample:     dict.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """
        # Extract the acquisition dates of the cloud-free images within the sequence
        # s2_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['S2/S2_dates']]
        t_cloudfree = idx_good_frames
        s2_dates = [str2date(s2_dates[t]) for t in t_cloudfree]

        # Count number of consecutive cloud-free images with temporal sampling of at most
        # `self.filter_settings.max_t_sampling:`
        subseq = {"start": 0, "end": 0, "len": 0}
        count = 1
        start = 0

        for i in range(len(s2_dates) - 1):
            if (s2_dates[i + 1] - s2_dates[i]).days <= max_t_sampling:
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
    def _longest_consecutive_seq(idx_frames: list) -> dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images.

        Args:
            sample:      list.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """
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

    def subsample_sequence(self, masks_valid_obs: torch.Tensor) -> torch.Tensor:
        """
        Trims the sequence to a maximum temporal length.
        """
        if self.filter_settings.get("return_valid_obs_only", True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = self.rng.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled, masks_valid_obs

    def _mask_images_with_cloud_coverage_above_p(self, cloud_mask: torch.Tensor) -> torch.Tensor:
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

    def __getitem__(
        self,
        item: int,
        t_sampled: Optional[torch.Tensor] = None,
        t_masked: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset.

        Args:
            item: int, index of the sample to be returned.

        Returns:
            sample: dict, a dictionary containing the following key-value pairs:
                'x':                  torch.Tensor, (T x C x H x W), (synthetically masked) S2 time series.
                'y':                  torch.Tensor, (T x C x H x W), observed/target satellite image time series.
                'masks':              torch.Tensor, (T x 1 x H x W), masks applied to `x`.
                'masks_valid_obs':    torch.Tensor, (T, ), flag to indicate valid time steps.
                'position_days':      torch.Tensor, (T, ), positions for positional encoding.
                'days':               torch.Tensor, (T, ), temporal sampling.
                'sample_index':       int, index of the sample in the dataset.
                'filepath':           list of str, file paths of the sample.
                'c_index_rgb':        torch.Tensor, indices of RGB bands in `x`.
                'c_index_nir':        torch.Tensor, indices of NIR bands in `x`.
                'S2_dates':           list of str, dates of the S2 images in `x`.
                'cloud_prob':         torch.Tensor, cloud probabilities associated with `x`.
                'cloud_mask':         torch.Tensor, cloud mask associated with `x`.
        """
        patch_data = self.etl_item(item=item)

        # Select the correct channels
        if self.num_channels != patch_data["S2"]["S2"].shape[1]:
            patch_data["S2"]["S2"] = patch_data["S2"]["S2"][:, self.s2_channels, :, :]

        if t_sampled is None:
            t_sampled, masks_valid_obs = self.subsample_sequence(patch_data["valid_obs"])
        masks_valid_obs = patch_data["valid_obs"][t_sampled]

        frames_input, frames_target = (
            patch_data["S2"]["S2"][t_sampled].clone(),
            patch_data["S2"]["S2"][t_sampled].clone(),
        )

        if self.process_data:
            frames_input = SentinelDataProcessor.process_MS(frames_input)
            frames_target = SentinelDataProcessor.process_MS(frames_target)

        s2_dates = np.asarray(patch_data["S2"]["S2_dates"])[t_sampled]

        if self.use_sar:
            if self.use_sar == "asc+desc":
                s1_asc = patch_data["S1"]["S1_asc"][t_sampled]
                s1_asc_dates = patch_data["S1"]["S1_dates_asc"][t_sampled]
                s1_desc = patch_data["S1"]["S1_desc"][t_sampled]
                s1_desc_dates = patch_data["S1"]["S1_dates_desc"][t_sampled]

                s1 = torch.cat((s1_asc, s1_desc), dim=1)
                s1_dates = get_pairwise_representative_dates(asc_dates=s1_asc_dates, desc_dates=s1_desc_dates)
            else:
                s1 = patch_data["S1"]["S1"][t_sampled]
                s1_dates = patch_data["S1"]["S1_dates"][t_sampled]

            if self.process_data:
                s1 = SentinelDataProcessor.process_SAR(s1)

            # Concatenate the (masked) S2 bands and the unmasked S1 bands
            frames_input = torch.cat((frames_input, s1), dim=1)
        cloud_mask = patch_data["S2"]["cloud_mask"][t_sampled]  # T x C x H x W

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.0:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        # Generate masks
        if self.mask_kwargs is not None:
            t_masked, frames_input, masks = self._generate_masks(
                item,
                patch_data,
                frames_input,
                cloud_mask,
                t_masked,
            )
        else:
            masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))  # T x C x H x W

        # Cast des dates en datetime.datetime à datetime.date si besoin
        if isinstance(s2_dates[0], dt.datetime):
            s2_dates = [date.date() for date in s2_dates]
        if self.use_sar and isinstance(s1_dates[0], dt.datetime):
            s1_dates = [date.date() for date in s1_dates]

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = get_position_for_positional_encoding(s2_dates, "day-within-sequence")
        # Get positions for positional encoding
        position_days = get_position_for_positional_encoding(s2_dates, self.pe_strategy)
        # Assemble output
        out = {
            "x": frames_input,  # (synthetically masked) S2 TS, (T x C x H x W), optionally including S1.
            "y": frames_target,  # observed/target satellite image time series, (T x C x H x W)
            "masks": masks,  # masks applied to `x`, (T x 1 x H x W); pixel with value 1 is masked, 0 otherwise
            "masks_valid_obs": masks_valid_obs,  # flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid
            "position_days": position_days,
            "days": days,  # temporal sampling, number of days since the first observation in the sequence, (T, )
            "sample_index": item,
            "c_index_rgb": self.c_index_rgb,
            "c_index_nir": self.c_index_nir,
            "S2_dates": [date.strftime("%Y-%m-%d") for date in s2_dates],
            "cloud_prob": patch_data["S2"]["cloud_prob"],
            "cloud_mask": cloud_mask,
        }
        if self.use_sar:
            out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in s1_dates]
        return out

    # FONCTION POUR LA GENERATION DE MASKS
    def _generate_masks(
        self,
        id_obs: int,
        sample: dict,
        frames_input: Tensor,
        cloud_mask_input: Tensor,
        t_masked: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], Tensor, Tensor]:
        """
        Uses a sequence of masks (randomly generated or actual cloud mask sequence) to synthetically generate data gaps
        in the given satellite image time series.

        Args:
            sample:             dict.
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
                id_sample=id_obs,
                sample=sample,
                n=len(t_masked["indices_masked"]),
                p=self.mask_kwargs.p_filter,
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

    def _intersect_masks(self, masks: torch.Tensor, cloud_mask: torch.Tensor) -> torch.Tensor:
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

    def _mask_images_with_cloud_coverage_above_p(self, cloud_mask: torch.Tensor) -> torch.Tensor:
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

    def _sample_cloud_masks_from_tiles(self, id_sample, sample: dict, n: int, p: float = 0.1) -> torch.Tensor:
        """
        Randomly samples `n` cloud masks from a given tile.

        Args:
            sample:  dict.
            n:       int, number of cloud masks to be sampled.
            p:       float, minimum cloud coverage [-] of the sampled cloud masks.

        Returns:
            cloud_mask:  torch.Tensor, n x 1 x H x W, sampled cloud masks.
        """
        # Retrieve information about the tile from which the sample originates
        sample_info = self.patches_dataset.iloc[id_sample]
        # Extract all samples that originate from the same tile as the given input sample
        samples = self.patches_dataset[self.patches_dataset["mgrs25"] == sample_info.mgrs25]
        # Randomly sample `n` cloud masks with cloud coverage of >= p
        cloud_mask = []
        while len(cloud_mask) < n:
            # Extract the cloud masks of a randomly drawn image time series, T * 1 * H * W (ancien code H x W x 1 x T)
            selected_idx = self.rng.choice(samples.index)
            seletect_row = self.patches_dataset.iloc[selected_idx]
            seq = self.hdf5_file[f"{seletect_row.mgrs}/{seletect_row.mgrs25}/{seletect_row.window}/S2/cloud_mask"][:]
            seq = torch.from_numpy(np.expand_dims(seq, axis=1)).type(torch.float32)  # H x W x T => T * 1 * H * W

            # Compute cloud coverage per frame
            coverage = torch.mean(seq, dim=(1, 2, 3))
            indices = torch.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[self.rng.choice(indices)])

        # n x 1 x H x W
        cloud_mask = torch.stack(cloud_mask, dim=0)  # on stack sur C du T *C * H * W

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.0:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask

    # def _subsample_sequence(self, idx_good_frames: np.ndarray, seq_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
    #     `self.max_seq_length`):
    #     1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
    #     2) removes invalid time steps (i.e., no observation, black image),
    #     3) trims the sequence to a maximum temporal length.

    #     Args:
    #         sample:           h5py group.
    #         seq_length:       int, temporal length of the sample.

    #     Returns:
    #         t_sampled:        torch.Tensor, length T.
    #         masks_valid_obs:  torch.Tensor, (T, ).
    #     """
    #     # Generate a mask to exclude invalid frames:
    #     # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
    #     if self.filter_settings.type == "cloud-free":
    #         # Indices of available and cloud-free images
    #         masks_valid_obs = torch.from_numpy(idx_good_frames)

    #     elif self.filter_settings.type == "cloud-free_consecutive":
    #         subseq = self._longest_consecutive_seq(idx_good_frames)
    #         masks_valid_obs = torch.from_numpy(idx_good_frames)
    #         masks_valid_obs[: subseq["start"]] = 0
    #         masks_valid_obs[subseq["end"] + 1 :] = 0
    #     else:
    #         masks_valid_obs = torch.ones(
    #             seq_length,
    #         )

    #     if self.filter_settings.get("return_valid_obs_only", True):
    #         t_sampled = masks_valid_obs.nonzero().view(-1)
    #     else:
    #         t_sampled = torch.arange(0, len(masks_valid_obs))

    #     if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
    #         # Randomly select `self.max_seq_length` consecutive frames
    #         t_start = self.rng.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
    #         t_end = t_start + self.max_seq_length
    #         t_sampled = t_sampled[t_start:t_end]

    #     return t_sampled, masks_valid_obs[t_sampled]

    # @staticmethod
    # def _longest_consecutive_seq(idx_frames: torch.Tensor) -> dict[str, int]:
    #     """
    #     Determines the longest subsequence of consecutive cloud-free images.

    #     Args:
    #         idx_frames:      torch.Tensor.

    #     Returns:
    #         subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
    #                      pairs:
    #                         'start':  int, index of the first image of the subsequence.
    #                         'end':    int, index of the last image of the subsequence.
    #                         'len':    int, temporal length of the subsequence.
    #     """

    #     # Count number of consecutive cloud-free images
    #     subseq = {"start": 0, "end": 0, "len": 0}
    #     count = 1
    #     start = 0

    #     for i in range(len(idx_frames) - 1):
    #         if idx_frames[i] + 1 == idx_frames[i + 1]:
    #             end = i + 1
    #             count += 1
    #             if count > subseq["len"]:
    #                 subseq["start"] = idx_frames[start]
    #                 subseq["end"] = idx_frames[end]
    #                 subseq["len"] = count
    #         else:
    #             start = i + 1
    #             count = 1
    #     return subseq

    # def _filter_consecutive_sequence(
    #     self,
    #     dates,
    #     idx_good_frames: list,
    #     seq_length: int,
    #     filter_type: Optional[str] = None,
    #     max_t_sampling: Optional[int] = None,
    # ) -> torch.Tensor:
    #     """
    #     Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
    #     `self.max_seq_length`):
    #     1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
    #     2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
    #        images is at most `self.filter_settings.max_t_sampling` days,
    #     3) trims the sequence to a maximum temporal length.

    #     Args:
    #         idx_good_frames:           list.
    #         seq_length:       int, temporal length of the sample.

    #     Returns:
    #         t_sampled:        torch.Tensor, length T.
    #         masks_valid_obs:  torch.Tensor, (T, ).
    #     """
    #     # Indices of available and cloud-free images
    #     if isinstance(idx_good_frames, torch.Tensor):
    #         masks_valid_obs = idx_good_frames.clone()
    #     else:
    #         masks_valid_obs = torch.from_numpy(idx_good_frames.copy())

    #     # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
    #     if filter_type == "cloud-free":
    #         # Generate a mask to exclude invalid frames:
    #         if max_t_sampling is not None:
    #             subseq = self._longest_consecutive_seq_within_sampling_frequency(dates, masks_valid_obs, max_t_sampling)
    #             masks_valid_obs[: subseq["start"]] = 0
    #             masks_valid_obs[subseq["end"] + 1 :] = 0
    #     elif filter_type == "cloud-free_consecutive":
    #         subseq = self._longest_consecutive_seq(masks_valid_obs)
    #         masks_valid_obs[: subseq["start"]] = 0
    #         masks_valid_obs[subseq["end"] + 1 :] = 0
    #     else:
    #         masks_valid_obs = torch.ones(
    #             seq_length,
    #         )
    #     return masks_valid_obs


######################################################################################
######################################################################################
######################################################################################

if __name__ == "__main__":
    # path_dataset_circa = Path("/DATA_10TB/data_rpg/circa/hdf5")
    # hdf5_file = path_dataset_circa / "circa_cloud_removal_asc_desc.hdf5"

    # Example usage
    path_dataset_circa = Path("/DATA_10TB/data_rpg/circa/hdf5")
    # hdf5_file = path_dataset_circa / "new_circa_ligth.hdf5"
    hdf5_file = path_dataset_circa / "merged_archives.hdf5"

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

    dataset = CIRCA_ADAPTED2UTILISE_Dataset(
        # CIRCA_from_HDF5 parameters
        phase="all",
        hdf5_file=hdf5_file,
        shuffle=False,
        use_sar="asc+desc",
        channels="all",
        # U-TILISE specific parameters
        pe_strategy="day-within-sequence",
        filter_settings=filter_settings,
        mask_kwargs=mask_kwargs,
        max_seq_length=10,
        process_data=True,
        render_occluded_above_p=None,
    )
    sample = next(iter(dataset))
    print(sample.keys())
