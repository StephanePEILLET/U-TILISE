from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from typing import Tuple
import ast
import json
from typing import Dict, List, Optional, Union
import math
import numpy as np
import pandas as pd
import rasterio
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.datasets import CircaPatchDataSet
import h5py
from dataloader_CIRCA.tools.positional_encoding import get_position_for_positional_encoding
import random
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from dataloader_CIRCA.tools.mask_generation import overlay_seq_with_clouds, masks_init_filling
from dataloader_CIRCA.tools.sampling import sample_indices_masked_frames
from dataloader_CIRCA.tools.positional_encoding import str2date

MAX_SEQ_LENGTH = 30
MIN_SEQ_LENGTH = 5

class UTILISE_Dataset_HDF5_Handler(CircaPatchDataSet):
    """
        Dataset qui exporte / ou importe les données CIRCA dans / depuis un fichier HDF5.
    """
    def __init__(
        self,
        data_optique: Union[str, Path]= None,
        data_radar: Union[str, Path]= None,
        image_size: int = 256,
        hdf5_file_output: Optional[Union[str, Path]] = None,
        hdf5_file_read: Optional[Union[str, Path]] = None,
        overlap: Optional[int] = 0,
        load_dataset: Optional[str] = None,
        shuffle: bool = False,
        use_sar: bool = True,
        filter_settings: dict = None,
        min_seq_length: Optional[int] = MIN_SEQ_LENGTH,
        max_seq_length: Optional[int] = None,
        render_occluded_above_p: Optional[float] = None,
        mask_kwargs: Optional[Dict | DictConfig] = None,
        pe_strategy: str = 'day-within-sequence',
        augment: Optional[bool] = False,
        crop_settings: Optional[DictConfig] = None,
        return_cloud_mask: bool = True,
        channels: Optional[str] = 'all',
        sampling_random: Optional[float] = None,
    ):
        if hdf5_file_read is None:
            super().__init__(
                data_optique=data_optique,
                data_radar=data_radar,
                image_size=image_size,
                overlap=overlap,
                load_dataset=load_dataset,
                shuffle=shuffle,
                use_sar=use_sar,
            )
            self.hdf5_file_output = hdf5_file_output
            self.sampling_random = sampling_random
            self.min_seq_length = min_seq_length
        else:
            self.hdf5_file, self.patches_dataset = self.setup_hdf5_file(hdf5_file_read)
            self.use_sar = use_sar
            self.render_occluded_above_p = render_occluded_above_p    # Fully occlude images with high cloud cover
            # TODO Potentiellement stocker dans le hdf5 les hparams sur le filtrage les channels et les masks 
            self.pe_strategy = pe_strategy
            self.augment = augment
            self.channels = channels
            self.num_channels, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels()
            self.filter_settings, self.variable_seq_length, self.seq_length, self.max_seq_length = self.setup_filter_settings(
                filter_settings, max_seq_length)
            (
                self.mask_kwargs,
                self.fill_type,
                self.fill_value,
                self.fixed_masking_ratio,
                self.intersect_real_cloud_masks,
                self.dilate_cloud_masks
            ) = self.setup_mask_kwargs(mask_kwargs)

    def setup_hdf5_file(self, path_file):
        if Path(path_file).exists():
            f = h5py.File(path_file, 'r', libver='latest', swmr=True)
            patches_dataset = self.list_files_in_hdf5(f)
        else:
            raise FileNotFoundError(f"HDF5 file {path_file} does not exist.")
        return f, patches_dataset

    def list_files_in_hdf5(self, hdf5_file: Union[str, Path]) -> pd.DataFrame:
        patches_dataset = pd.DataFrame(columns=['mgrs', 'mgrs25', 'window'])
        for mgrs_level, mgrsc_list in hdf5_file.items():
            for mgrsc_level, mgrsc_data in mgrsc_list.items():
                for window in mgrsc_data.keys():
                    data = {'mgrs': [mgrs_level], 'mgrs25': [mgrsc_level], 'window': [window]}
                    patches_dataset = pd.concat([patches_dataset, pd.DataFrame(data)], ignore_index=True)
        return patches_dataset

    def setup_channels(self):
        num_channels = 10
        c_index_rgb = torch.Tensor([2, 1, 0]).long()
        c_index_nir = torch.Tensor([6]).long()
        s2_channels = list(np.arange(10))
        if self.use_sar:
            num_channels += 4
        return num_channels, c_index_rgb, c_index_nir, s2_channels

    def setup_filter_settings(self, filter_settings: Optional[DictConfig] = None, max_seq_length: Optional[int] = None):
        if filter_settings is None:
            filter_settings = {
                'type': None,
                'min_length': 5,
                'return_valid_obs_only': False,
                'max_t_sampling': None,
                'p_filter': 0.1,
            }

        if isinstance(filter_settings, dict):
            filter_settings = OmegaConf.create(filter_settings)

        if  filter_settings.get('type', None):
            variable_seq_length = filter_settings.return_valid_obs_only
        else:
            variable_seq_length = False

        # Definition en dur de certaines variables
        filter_settings.max_num_consec_invalid = filter_settings.get('max_num_consec_invalid', None)
        filter_settings.min_length = filter_settings.get('min_length', 0)
        filter_settings.max_t_sampling = filter_settings.get('max_t_sampling', None)
        seq_length = MAX_SEQ_LENGTH if max_seq_length is None else max_seq_length
        return filter_settings, variable_seq_length, seq_length, max_seq_length

    def setup_mask_kwargs(self, mask_kwargs: Optional[DictConfig] = None):
        # Parameters used for creating synthetic data gaps

        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)

        if mask_kwargs is None:
            mask_kwargs = {
                "mask_type": "random_clouds",             # Mask the input time series with randomly sampled cloud masks or the actual cloud masks. ['random_clouds', 'real_clouds']
                "ratio_masked_frames": 0.5,               # Ratio of partially/fully masked images per image time series (upper bound).
                "ratio_fully_masked_frames": 0.0,         # Ratio of fully masked images per image time series (upper bound).
                "fixed_masking_ratio": False,             # True to vary the masking ratio across different image time series, False otherwise.
                "non_masked_frames": [0],                 # list of int, time steps to be excluded from masking. E.g., [0] never masks the first frame in a sequence.
                "intersect_real_cloud_masks": False,      # True to intersect randomly sampled cloud masks with the actual cloud masks, False otherwise.
                "dilate_cloud_masks": False,              # True to dilate the cloud masks before masking, False otherwise.
                "fill_type": "fill_value",                # Strategy for initializing masked pixels. ['fill_value', 'white_noise', 'mean']
                "fill_value": 1,                          # Pixel value of masked pixels. Used if fill_type == 'fill_value'.
            }
            mask_kwargs = OmegaConf.create(mask_kwargs)
            mask_kwargs.mask_type = mask_kwargs.get('mask_type', 'random_clouds')
            mask_kwargs.ratio_masked_frames = mask_kwargs.get('ratio_masked_frames', 0.5)
            mask_kwargs.ratio_fully_masked_frames = mask_kwargs.get('ratio_fully_masked_frames', 0.0)
            mask_kwargs.non_masked_frames = mask_kwargs.get('non_masked_frames', [])

        fill_type = mask_kwargs.get('fill_type', 'fill_value')
        fill_value = mask_kwargs.get('fill_value', 1)
        fixed_masking_ratio = mask_kwargs.get('fixed_masking_ratio', False)
        intersect_real_cloud_masks = mask_kwargs.get('intersect_real_cloud_masks', False)
        dilate_cloud_masks = mask_kwargs.get('dilate_cloud_masks', False)
        return mask_kwargs, fill_type, fill_value, fixed_masking_ratio, intersect_real_cloud_masks, dilate_cloud_masks

    def _longest_consecutive_seq_within_sampling_frequency(self, s2_dates, idx_good_frames, max_t_sampling) -> dict[str, int]:
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
        t_cloudfree = idx_good_frames
        s2_dates = [str2date(s2_dates[t]) for t in t_cloudfree]

        # Count number of consecutive cloud-free images with temporal sampling of at most
        # `self.filter_settings.max_t_sampling:`
        subseq = {'start': 0, 'end': 0, 'len': 0}
        count = 1
        start = 0

        for i in range(len(s2_dates) - 1):
            if (s2_dates[i+1] - s2_dates[i]).days <= max_t_sampling:
                end = i + 1
                count += 1
                if count > subseq['len']:
                    subseq['start'] = t_cloudfree[start]
                    subseq['end'] = t_cloudfree[end]
                    subseq['len'] = count
            else:
                start = i + 1
                count = 1
        return subseq

    @staticmethod
    def _longest_consecutive_seq(idx_frames: list) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images.

        Args:
            sample:      List.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """
        # Count number of consecutive cloud-free images
        subseq = {'start': 0, 'end': 0, 'len': 0}
        count = 1
        start = 0
        for i in range(len(idx_frames) - 1):
            if idx_frames[i] + 1 == idx_frames[i + 1]:
                end = i + 1
                count += 1
                if count > subseq['len']:
                    subseq['start'] = idx_frames[start]
                    subseq['end'] = idx_frames[end]
                    subseq['len'] = count
            else:
                start = i + 1
                count = 1
        return subseq

    def _filter_consecutive_sequence(
            self,
            dates,
            idx_good_frames: list,
            seq_length: int,
            filter_type: Optional[str] = None,
            max_t_sampling: Optional[int] = None,
        ) -> torch.Tensor:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
           images is at most `self.filter_settings.max_t_sampling` days,
        3) trims the sequence to a maximum temporal length.

        Args:
            idx_good_frames:           list.
            seq_length:       int, temporal length of the sample.

        Returns:
            t_sampled:        torch.Tensor, length T.
            masks_valid_obs:  torch.Tensor, (T, ).
        """
        # Indices of available and cloud-free images
        if isinstance(idx_good_frames, torch.Tensor):
            masks_valid_obs = idx_good_frames.clone()
        else:
            masks_valid_obs = torch.from_numpy(idx_good_frames.copy())

        # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
        if filter_type == 'cloud-free':
            # Generate a mask to exclude invalid frames:
            if max_t_sampling is not None:
                subseq = self._longest_consecutive_seq_within_sampling_frequency(dates, masks_valid_obs, max_t_sampling)
                masks_valid_obs[:subseq['start']] = 0
                masks_valid_obs[subseq['end'] + 1:] = 0
        elif filter_type == 'cloud-free_consecutive':
            subseq = self._longest_consecutive_seq(masks_valid_obs)
            masks_valid_obs[:subseq['start']] = 0
            masks_valid_obs[subseq['end'] + 1:] = 0
        else:
            masks_valid_obs = torch.ones(seq_length, )
        return masks_valid_obs

    def subsample_sequence(self, masks_valid_obs: torch.Tensor) -> torch.Tensor:
        """
            Trims the sequence to a maximum temporal length.
        """
        if self.filter_settings.get('return_valid_obs_only', True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
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

    def load_items_to_hdf5(self):
        with h5py.File(self.hdf5_file_output, 'w') as hf:
            for mgrs_id in tqdm(self.patches_dataset["mgrs"].unique(), desc="MGRS IDs"):
                mgrs_group = hf.create_group(mgrs_id)
                mgrs_dataset = self.patches_dataset[self.patches_dataset["mgrs"] == mgrs_id]
                for mgrs25_id in tqdm(mgrs_dataset["mgrs25"].unique(), desc="MGRS25 IDs"):
                    mgrs25_group = mgrs_group.create_group(mgrs25_id)
                    mgrs25_dataset = mgrs_dataset[mgrs_dataset["mgrs25"] == mgrs25_id]
                    # ETL des données S2 pour la zone mgrs25 concernée
                    # 1.Chargement des masks nuages / neiges concernant la zone MGRSC
                    mgrs25_files = mgrs25_dataset.iloc[0].files # First sample contains all the files of the mgrsc area
                    dates_s2 = self.dates_dict[mgrs25_id]["S2"]
                    dates_s1_asc = self.dates_dict[mgrs25_id]["S1"]["ASC"]
                    dates_s1_desc = self.dates_dict[mgrs25_id]["S1"]["DESC"]
                    # 2. Récupération des masks nuage et neige
                    cloud_probs = SentinelDataProcessor.read_mask_prob(path_raster=mgrs25_files[0], type_mask="cloud") # H x W X 1 X T
                    snow_probs = SentinelDataProcessor.read_mask_prob(path_raster=mgrs25_files[0], type_mask="snow") # H x W X 1 X T
                    # 3. Correction du mask nuage et binarisation
                    cloud_probs = cloud_probs.squeeze(axis=2).transpose((2, 0, 1))   #  H x W X 1 X T => T, H, W
                    snow_probs = snow_probs.squeeze(axis=2).transpose((2, 0, 1))     #  H x W X 1 X T => T, H, W
                    cloud_probs_corrected = SentinelDataProcessor.cloud_mask_correction(cloud_probs) # Attends du (T, H, W)

                    index_to_drop, mgrs25_data = [], {}
                    for row_index, row in tqdm(mgrs25_dataset.iterrows(), total=mgrs25_dataset.shape[0], desc="Windows Processing"):
                        # Découpage des données selon la fenêtre
                        window = row.window
                        x, y, width, height = window[0], window[1], window[2], window[3]
                        # 4. Filtrage à la fenêtre des masks nuage et neige
                        snow_probs_window = snow_probs[:, x:x+width, y:y+height] # T * H * W
                        cloud_probs_window = cloud_probs_corrected[:, x:x+width, y:y+height] # T * H * W
                        idx_good_frames = SentinelDataProcessor.filter_dates(np.stack([snow_probs_window, cloud_probs_window], axis=-1))  # T * H * W * 2
                        idx_cloudy_frames = np.asarray([d for d in range(len(dates_s2)) if d not in idx_good_frames])
                        # 5. Recherche de la plus longue série de dates non-nuageuses
                        masks_valid_obs = self._filter_consecutive_sequence(
                            dates=dates_s2,
                            idx_good_frames=idx_good_frames,
                            seq_length=len(dates_s2),
                            filter_type=self.filter_settings.get('type', None),
                            max_t_sampling=self.filter_settings.get('max_t_sampling', None),
                        )
                        # dates_s2_valid = [dates_s2[t] for t in masks_valid_obs.nonzero().view(-1)]
                        dates_s2_valid = [dates_s2[t] for t in masks_valid_obs]
                        # 6. En fonction de la tailles des séries de dates non-nuageuses, garder ou extraire la TS / patch du dataset
                        if self.min_seq_length is not None and len(dates_s2_valid) < self.min_seq_length:
                            index_to_drop.append(row_index)
                        else:
                            mgrs25_data[row_index] = {
                                'idx_good_frames': idx_good_frames.tolist(),
                                'idx_cloudy_frames': idx_cloudy_frames.tolist(),
                                'masks_valid_obs': masks_valid_obs.tolist(),
                                'dates_s2_valid': dates_s2_valid,
                            }
                    # Gestion de la df du mgrs25, enrichissement feature / drop index
                    if self.sampling_random is not None:
                        len_mgrs25 = mgrs25_dataset.shape[0] # Récupération du nombre de sample par mgrs25 avantsuppression certaines observations
                        n_sampling = np.floor(len_mgrs25 * self.sampling_random).astype(np.int16)

                    mgrs25_dataset = mgrs25_dataset.drop(index=index_to_drop)
                    for label in ['idx_cloudy_frames', 'idx_good_frames', 'masks_valid_obs', 'dates_s2_valid']:
                        mgrs25_dataset[label] = [v[label] for v in mgrs25_data.values()]

                    if self.sampling_random is not None:
                        mgrs25_dataset = mgrs25_dataset.sample(n=n_sampling)

                    # Analyse des bandes utilisés dans la TS de la zone MGRS25
                    bands_used = mgrs25_dataset['masks_valid_obs'].explode().unique()
                    dates_s2_used = mgrs25_dataset['dates_s2_valid'].explode().unique()
                    # Récupération des données S1 associées aux dates s2 valides prises
                    dates_s1, index_s1, orbit_type = SentinelDataProcessor.get_pairedS1(dates_s2_used, dates_s1_asc, dates_s1_desc)
                    # Récupération des images S2 associées aux dates valides  # Refaire la fonction d'extraction
                    mgrsc_s2 = SentinelDataProcessor.read_raster_per_dates(
                        path_raster=mgrs25_files[0],
                        indexes_dates=bands_used,
                        type_bands="s2_bands",
                    )
                    path_s1 = mgrs25_files[1] if orbit_type == "ASC" else mgrs25_files[2]
                    mgrsc_s1 = SentinelDataProcessor.read_raster_per_dates(
                        path_raster=path_s1,
                        indexes_dates=index_s1,
                        type_bands="s1",
                    )
                    for row_index, row in tqdm(mgrs25_dataset.iterrows(), total=mgrs25_dataset.shape[0], desc="Windows loading"):
                        # Découpage des données selon la fenêtre
                        window = row.window
                        x, y, width, height = window[0], window[1], window[2], window[3]
                        windows_str = "_".join(map(str, window))
                        # 4. Filtrage à la fenêtre des masks nuage et neige
                        cloud_probs_window = cloud_probs_corrected[:, x:x+width, y:y+height]
                        snow_probs_window = snow_probs[:, x:x+width, y:y+height]
                        cloud_masks_window = (cloud_probs_window != 0).astype(int)
                        s2 = mgrsc_s2[:, :, x:x+width, y:y+height]
                        s1 = mgrsc_s1[:, :, x:x+width, y:y+height]
                        # Pre-process data MS / SAR
                        # s2 = SentinelDataProcessor.process_MS(torch.from_numpy(s2).type(torch.float32))
                        # s1 = SentinelDataProcessor.process_SAR(torch.from_numpy(s1).type(torch.float32))
                        sample = {
                            "S1": {
                                "S1": s1,
                                "S1_dates": dates_s1,
                            },
                            "S2": {
                                "S2": s2, # Bandes correspondant aux dates correctes de la TS
                                "S2_dates": dates_s2_valid, # Dates correctes de la TS
                                "cloud_mask": cloud_masks_window,  # Mask entier de la TS
                                "cloud_prob": cloud_probs_window.astype(np.float32), # Probs cloud entier de la TS
                            },
                            "idx_cloudy_frames": np.asarray(mgrs25_dataset.loc[row_index, 'idx_cloudy_frames']),
                            "idx_good_frames": np.asarray(mgrs25_dataset.loc[row_index, 'idx_good_frames']),
                            "idx_impaired_frames": np.asarray(mgrs25_dataset.loc[row_index, 'idx_cloudy_frames']),
                            "valid_obs": np.asarray(mgrs25_dataset.loc[row_index, 'masks_valid_obs']),
                        }
                        window_group = mgrs25_group.create_group(windows_str)
                        for key, value in sample.items():
                            if isinstance(value, dict):
                                window_subgroup = window_group.create_group(key)
                                for meta_key, meta_value in value.items():
                                    if isinstance(meta_value, np.ndarray):
                                        window_subgroup.create_dataset(
                                            meta_key,
                                            data=meta_value,
                                            compression='gzip',
                                            compression_opts=9,
                                        )
                                    else:
                                        window_subgroup.create_dataset(meta_key, data=meta_value)
                            else:
                                window_group.create_dataset(key, data=value)

    def decode_dates(self, dates):
        return np.asanyarray([el.decode('utf-8') for el in dates])

    def format_item(self, sample: dict):
        """
            Passage de T * C * H * W au bon format pour le modèle.
        """
        return {
            "S1": {
                "S1": torch.from_numpy(sample['S1']['S1'].astype(np.float32)), # T * C * H * W
                "S1_dates": np.array([str2date(date) for date in sample['S1']['S1_dates']]),
            },
            "S2": {
                "S2": torch.from_numpy(sample['S2']['S2'].astype(np.float32)),  # T * C * H * W
                "S2_dates": np.array([str2date(date) for date in sample['S2']['S2_dates']]),
                "cloud_mask":torch.from_numpy(np.expand_dims(sample['S2']['cloud_mask'], axis=1)),
                "cloud_prob": torch.from_numpy(np.expand_dims(sample['S2']['cloud_prob'], axis=1)),
                },
            "idx_cloudy_frames":  torch.from_numpy(sample['idx_cloudy_frames']),
            "idx_good_frames": torch.from_numpy(sample['idx_good_frames']),
            "idx_impaired_frames":  torch.from_numpy(sample['idx_impaired_frames']),
            "valid_obs": torch.from_numpy(sample['valid_obs']),
        }


    def etl_item(self, item: int) -> Dict[str, Union[np.ndarray, List[str]]]:
        row = self.patches_dataset.iloc[item]
        patch = self.hdf5_file[f"{row.mgrs}/{row.mgrs25}/{row.window}"]
        sample = {
            "S1": {
                "S1": patch['S1/S1'][:], # T * C * H * W
                "S1_dates": self.decode_dates(patch['S1/S1_dates'][:]),
            },
            "S2": {
                "S2": patch['S2/S2'][:], # T * C * H * W
                "S2_dates": self.decode_dates(patch['S2/S2_dates'][:]),
                "cloud_mask": patch['S2/cloud_mask'][:], # T * C * H * W
                "cloud_prob": patch['S2/cloud_prob'][:], # T * C * H * W
                },
            "idx_cloudy_frames": patch['idx_cloudy_frames'][:],
            "idx_good_frames": patch['idx_good_frames'][:],
            "idx_impaired_frames": patch['idx_impaired_frames'][:],
            "valid_obs": patch['valid_obs'][:],
        }
        return self.format_item(sample)


    def __getitem__(self, item: int, t_sampled: Optional[torch.Tensor] = None,
            t_masked: Optional[torch.Tensor] = None,) -> dict[str, torch.Tensor]:
        """
        Returns a sample from the dataset.

        Args:
            item: int, index of the sample to be returned.

        Returns:
            sample: dict, a dictionary containing the following key-value pairs:
                'x':                  torch.Tensor, (T x C x H x W), (synthetically masked) S2 satellite image time series.
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

        if t_sampled is None:
            t_sampled, masks_valid_obs = self.subsample_sequence(patch_data['valid_obs'])
        masks_valid_obs = patch_data['valid_obs'][t_sampled]

        frames_input, frames_target = patch_data["S2"]['S2'][t_sampled].clone(), patch_data["S2"]['S2'][t_sampled].clone()
        s2_dates = patch_data["S2"]['S2_dates'][t_sampled]
        if self.use_sar:
            s1 = patch_data['S1']['S1'][t_sampled]
            s1_dates = patch_data['S1']['S1_dates'][t_sampled]
            # Concatenate the (masked) S2 bands and the unmasked S1 bands
            frames_input = torch.cat((frames_input, s1), dim=1)
        cloud_mask = patch_data['S2']["cloud_mask"][t_sampled]  # T x C x H x W

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
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

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = get_position_for_positional_encoding(s2_dates, 'day-within-sequence')
        # Get positions for positional encoding
        position_days = get_position_for_positional_encoding(s2_dates, self.pe_strategy)
        # Assemble output
        out = {
            'x': frames_input,  # (synthetically masked) S2 satellite image time series, (T x C x H x W), optionally including S1 bands
            'y': frames_target,  # observed/target satellite image time series, (T x C x H x W)
            'masks': masks,  # masks applied to `x`, (T x 1 x H x W); pixel with value 1 is masked, 0 otherwise
            'masks_valid_obs': masks_valid_obs,  # flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid
            'position_days': position_days,
            'days': days,    # temporal sampling, number of days since the first observation in the sequence, (T, )
            'sample_index': item,
            # 'filepath': self.patches_dataset.iloc[item].files,
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir,
            'S2_dates': [date.strftime('%Y-%m-%d') for date in s2_dates],
            'cloud_prob': patch_data['S2']['cloud_prob'][t_sampled],
            'cloud_mask': cloud_mask,
        }
        if self.use_sar:
            out["S1_dates"] = [date.strftime('%Y-%m-%d') for date in s1_dates]
        return out


    ### FONCTION POUR LA GENERATION DE MASKS ###
    def _generate_masks(
            self,
            id_obs: int,
            sample: dict,
            frames_input: Tensor,
            cloud_mask_input: Tensor,
            t_masked: Dict[str, np.ndarray] | None = None
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

        if self.mask_kwargs.mask_type == 'random_clouds':

            if t_masked is None:
                    # Indices of the frames to be masked w.r.t. the temporally trimmed sequence
                    t_masked = sample_indices_masked_frames(
                        idx_valid_input_frames=np.arange(0, frames_input.shape[0]),
                        ratio_masked_frames=self.mask_kwargs.ratio_masked_frames,
                        ratio_fully_masked_frames=self.mask_kwargs.ratio_fully_masked_frames,
                        non_masked_frames=self.mask_kwargs.non_masked_frames,
                        fixed_masking_ratio=self.fixed_masking_ratio
                    )

            # Randomly sample cloud masks
            sampled_clouds = self._sample_cloud_masks_from_tiles(
                id_sample=id_obs,
                sample=sample,
                n=len(t_masked['indices_masked']),
                p=self.mask_kwargs.p_filter,
            )

            # Generate a sequence of masks
            masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))
            masks[t_masked['indices_masked'], :, :, :] = sampled_clouds

            # Intersect the randomly generated sequence of cloud masks with the actual cloud masks of the sequence
            if self.intersect_real_cloud_masks:
                masks = self._intersect_masks(masks, cloud_mask_input)

            # Apply masking
            frames_input, masks = overlay_seq_with_clouds(
                frames_input, masks, t_masked=None, fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks,
            )

        elif self.mask_kwargs.mask_type == 'real_clouds':
            # Use the real cloud masks for masking
            frames_input, masks = masks_init_filling(
                frames_input, cloud_mask_input, None, fill_type='fill_value', fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks
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
        assert masks[0].shape == cloud_mask[0].shape, 'Cannot intersect two sequences of masks with unequal temporal ' \
                                                      'shape.'
        assert masks[-2:].shape == cloud_mask[-2:].shape, 'Cannot intersect two sequences of masks with unequal ' \
                                                      'spatial shape.'
        assert masks[1].shape == cloud_mask[1].shape, 'Cannot intersect two sequences of masks with unequal ' \
                                                      'spectral shape.'

        masks[torch.logical_or(masks > 0., cloud_mask == 1)] = 1
        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
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
            sample:  Dict.
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
            selected_idx = random.choice(samples.index)
            seletect_row = self.patches_dataset.iloc[selected_idx]
            seq = self.hdf5_file[f"{seletect_row.mgrs}/{seletect_row.mgrs25}/{seletect_row.window}/S2/cloud_mask"][:]
            seq = torch.from_numpy(np.expand_dims(seq, axis=1)).type(torch.float32)  # H x W x T => T * 1 * H * W

            # Compute cloud coverage per frame
            coverage = torch.mean(seq, dim=(1, 2, 3))
            indices = torch.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[np.random.choice(indices)])

        # n x 1 x H x W
        cloud_mask = torch.stack(cloud_mask, dim=0) # on stack sur C du T *C * H * W

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask


    def _subsample_sequence(self, idx_good_frames: np.ndarray, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if self.filter_settings.type == 'cloud-free':
            # Indices of available and cloud-free images
            masks_valid_obs = torch.from_numpy(idx_good_frames)

        elif self.filter_settings.type == 'cloud-free_consecutive':
            subseq = self._longest_consecutive_seq(idx_good_frames)
            masks_valid_obs = torch.from_numpy(idx_good_frames)
            masks_valid_obs[:subseq['start']] = 0
            masks_valid_obs[subseq['end'] + 1:] = 0
        else:
            masks_valid_obs = torch.ones(seq_length, )

        if self.filter_settings.get('return_valid_obs_only', True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled, masks_valid_obs[t_sampled]

    @staticmethod
    def _longest_consecutive_seq(idx_frames: torch.Tensor) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images.

        Args:
            idx_frames:      torch.Tensor.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """

        # Count number of consecutive cloud-free images
        subseq = {'start': 0, 'end': 0, 'len': 0}
        count = 1
        start = 0

        for i in range(len(idx_frames) - 1):
            if idx_frames[i] + 1 == idx_frames[i + 1]:
                end = i + 1
                count += 1
                if count > subseq['len']:
                    subseq['start'] = idx_frames[start]
                    subseq['end'] = idx_frames[end]
                    subseq['len'] = count
            else:
                start = i + 1
                count = 1
        return subseq

######################################################################################
######################################################################################
######################################################################################

if __name__ == "__main__":
    # store_dai = Path("/home/SPeillet/Partage/store-dai")
    # path_dataset_circa = store_dai / "projets/pac/3str/EXP_2"
    # output_file =  store_dai / "tmp/speillet" / "circa_ligth_0.5.hdf5"

    path_dataset_circa = Path("/home/SPeillet/Downloads/data")
    data_optique = path_dataset_circa / "optique_dataset"
    data_radar = path_dataset_circa / "radar_dataset_v4"
    image_size = [256, 256]
    overlap = 0
    SAMPLING_SUBSET = 0.5
    output_file =  path_dataset_circa / "toy_circa_ligth_0.5.hdf5"

    filter_settings = {
        "type": "cloud-free",           # Strategy for removing observations with data gaps. ['cloud-free', 'cloud-free_consecutive']
        "min_length": 10,                # Minimum sequence length.
        "return_valid_obs_only": True,  # True to return the cloud-filtered sequences, False otherwise.
        # "max_t_sampling": 10,            # Maximum temporal sampling frequency in days.
    }

    mask_kwargs = {
        "mask_type": "random_clouds",              # Mask the input time series with randomly sampled cloud masks or the actual cloud masks. ['random_clouds', 'real_clouds']
        "ratio_masked_frames": 0.5,                # Ratio of partially/fully masked images per image time series (upper bound).
        "ratio_fully_masked_frames": 0.0,          # Ratio of fully masked images per image time series (upper bound).
        "fixed_masking_ratio": False,              # True to vary the masking ratio across different image time series, False otherwise.
        "non_masked_frames": [0],                  # list of int, time steps to be excluded from masking. E.g., [0] never masks the first frame in a sequence.
        "intersect_real_cloud_masks": False,       # True to intersect randomly sampled cloud masks with the actual cloud masks, False otherwise.
        "dilate_cloud_masks": False,               # True to dilate the cloud masks before masking, False otherwise.
        "fill_type": "fill_value",                 # Strategy for initializing masked pixels. ['fill_value', 'white_noise', 'mean']
        "fill_value": 1,                           # Pixel value of masked pixels. Used if fill_type == 'fill_value'.
        "p_filter": 0.1,
    }

    ## Si export des données vers un fichier hdf5
    # dataset = UTILISE_Dataset_HDF5_Handler(
    #     hdf5_file_output=output_file,
    #     data_optique=data_optique,
    #     data_radar=data_radar,
    #     image_size=image_size,
    #     overlap=overlap,
    #     filter_settings=filter_settings,
    #     mask_kwargs=mask_kwargs,
    #     sampling_random=SAMPLING_SUBSET,
    # )
    # # dataset.load_items_to_hdf5()

    # Import des données depuis un fichier hdf5
    dataset = UTILISE_Dataset_HDF5_Handler(
        hdf5_file_read=output_file,
        filter_settings=filter_settings,
        mask_kwargs=mask_kwargs,
        max_seq_length=10,
    )
    sample = next(iter(dataset))
    print(sample.keys())

    # print("Conversion du dataset PyTorch en HDF5...")
    # if SUBSET:
    #     dataset = Subset(dataset, np.arange(SUBSIZE))
    #     pytorch_dict_2_hdf5(dataset, output_file, num_workers=8)
    # elif not output_file.exists():
    #     pytorch_dict_2_hdf5(dataset, output_file, num_workers=8)
    # print(f"Dataset converti et enregistré dans {output_file}")
