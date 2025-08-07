import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from typing import Optional
from typing import Union

import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import json

import h5py
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dataloader_CIRCA.datasets import CIRCA_from_files
from dataloader_CIRCA.datasets.CIRCA_constants import MGRSC_SPLITS
from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.tools.positional_encoding import get_position_for_positional_encoding  # NOQA
from dataloader_CIRCA.utils.pairing_S2_S1 import appariement_S1_to_S2

MAX_SEQ_LENGTH = 30
MIN_SEQ_LENGTH = 5
SEED = 42


class CIRCA_HDF5_Maker(CIRCA_from_files):
    """
    Dataset qui exporte  les données CIRCA dans un fichier HDF5.
    """

    def __init__(
        self,
        data_optique: Union[str, Path] = None,
        data_radar: Union[str, Path] = None,
        image_size: int = (256, 256),
        hdf5_folder: Optional[Union[str, Path]] = None,
        overlap: Optional[int] = 0,
        load_dataset: Optional[str] = None,
        shuffle: bool = False,
        use_sar: bool = True,
        filter_settings: dict = None,
        min_seq_length: Optional[int] = MIN_SEQ_LENGTH,
        max_seq_length: Optional[int] = None,
        render_occluded_above_p: Optional[float] = None,
        mask_kwargs: Optional[dict | DictConfig] = None,
        pe_strategy: str = "day-within-sequence",
        channels: Optional[str] = "all",
    ):
        self.rng = np.random.default_rng(seed=SEED)

        super().__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            image_size=image_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_sar=use_sar,
        )

        self.hdf5_folder = hdf5_folder
        self.min_seq_length = min_seq_length
        self.render_occluded_above_p = render_occluded_above_p  # Fully occlude images with high cloud cover
        self.pe_strategy = pe_strategy
        _, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels(channels)
        (
            self.filter_settings,
            self.variable_seq_length,
            self.seq_length,
            self.max_seq_length,
        ) = self.setup_filter_settings(filter_settings, max_seq_length)
        (
            self.mask_kwargs,
            self.fill_type,
            self.fill_value,
            self.fixed_masking_ratio,
            self.intersect_real_cloud_masks,
            self.dilate_cloud_masks,
        ) = self.setup_mask_kwargs(mask_kwargs)

    def setup_channels(self, channels: str):
        if channels == "all":
            num_channels = 10
            c_index_rgb = torch.Tensor([2, 1, 0]).long()
            c_index_nir = torch.Tensor([6]).long()
            s2_channels = list(np.arange(10))
        elif channels == "bgr-nir":
            num_channels = 4
            c_index_rgb = torch.Tensor([2, 1, 0]).long()
            c_index_nir = torch.Tensor([6]).long()
            s2_channels = [0, 1, 2, 6]
        else:
            raise ValueError(f"Channels {channels} not recognized. Use 'all' or 'bgr-nir'.")
        if self.use_sar:
            num_channels += 4
        return num_channels, c_index_rgb, c_index_nir, s2_channels

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

    def load_items_to_hdf5(self):
        """
        Load the CIRCA dataset into an HDF5 file.
        """

        for mgrs_id in tqdm(self.patches_dataset["mgrs"].unique(), desc="MGRS IDs"):
            mgrs_dataset = self.patches_dataset[self.patches_dataset["mgrs"] == mgrs_id]
            for mgrs25_id in tqdm(mgrs_dataset["mgrs25"].unique(), desc="MGRS25 IDs"):
                hdf5_file = self.hdf5_folder / f"{mgrs_id}.hdf5"
                if not hdf5_file.exists():
                    with h5py.File(hdf5_file, "w") as hf:
                        print(f"mgrs25_id: {mgrs25_id}")
                        print(f"Creating new HDF5 file: {hdf5_file}")
                        mgrs_group = hf.create_group(mgrs_id)
                        mgrs25_group = mgrs_group.create_group(mgrs25_id)
                        in_test_set = True if mgrs25_id in MGRSC_SPLITS["test"] else False

                        mgrs25_dataset = mgrs_dataset[mgrs_dataset["mgrs25"] == mgrs25_id]

                        # ETL des données S2 pour la zone mgrs25 concernée
                        # 1.Chargement des masks nuages / neiges concernant la zone mgrs25
                        mgrs25_files = mgrs25_dataset.iloc[
                            0
                        ].files  # First sample contains all the files of the mgrs25 area
                        dates_s2 = self.dates_dict[mgrs25_id]["S2"]

                        dates_s1_asc = self.dates_dict[mgrs25_id]["S1"]["ASC"]
                        dates_s1_desc = self.dates_dict[mgrs25_id]["S1"]["DESC"]

                        # 2. Récupération des masks nuage et neige
                        cloud_probs = SentinelDataProcessor.read_mask_prob(
                            path_raster=mgrs25_files[0], type_mask="cloud"
                        )  # H x W X 1 X T
                        snow_probs = SentinelDataProcessor.read_mask_prob(
                            path_raster=mgrs25_files[0], type_mask="snow"
                        )  # H x W X 1 X T

                        # 3. Correction du mask nuage (et binarisation ?)
                        cloud_probs = cloud_probs.squeeze(axis=2).transpose((2, 0, 1))  # H x W X 1 X T => T, H, W
                        snow_probs = snow_probs.squeeze(axis=2).transpose((2, 0, 1))  # H x W X 1 X T => T, H, W

                        if not in_test_set:
                            cloud_probs = SentinelDataProcessor.cloud_mask_correction(
                                cloud_probs
                            )  # Attends du (T, H, W)
                        else:
                            # Détermination des index des dates masquées synthétiquement
                            folder_test = Path("/mnt/stores/store-dai/projets/pac/3str/EXP_2/Data_Raster/test_v2")
                            folder_aleatoire = folder_test / "aleatoire"
                            folder_consecutif = folder_test / "consecutif"

                            assert folder_aleatoire.exists(), f"Random folder {folder_aleatoire} does not exist."
                            assert folder_consecutif.exists(), f"Consecutive folder {folder_consecutif} does not exist."

                            path_file_aleatoire = (
                                folder_aleatoire / mgrs_id / ("MGRS25-" + mgrs25_id) / Path(mgrs25_files[0]).name
                            )
                            path_file_consecutif = (
                                folder_consecutif / mgrs_id / ("MGRS25-" + mgrs25_id) / Path(mgrs25_files[0]).name
                            )

                            cloud_probs_aleatoire = SentinelDataProcessor.read_mask_prob(
                                path_raster=path_file_aleatoire, type_mask="cloud"
                            )  # H x W X 1 X T
                            cloud_probs_consecutif = SentinelDataProcessor.read_mask_prob(
                                path_raster=path_file_consecutif, type_mask="cloud"
                            )  # H x W X 1 X T

                            cloud_probs_aleatoire = cloud_probs_aleatoire.squeeze(axis=2).transpose(
                                (2, 0, 1)
                            )  # H x W X 1 X T => T, H, W
                            cloud_probs_consecutif = cloud_probs_consecutif.squeeze(axis=2).transpose(
                                (2, 0, 1)
                            )  # H x W X 1 X T => T, H, W

                        index_to_drop: list = []
                        mgrs25_data: dict = {}
                        for row_index, row in tqdm(
                            mgrs25_dataset.iterrows(), total=mgrs25_dataset.shape[0], desc="Windows Processing"
                        ):
                            # Découpage des données selon la fenêtre
                            window = row.window
                            x, y, width, height = window[0], window[1], window[2], window[3]
                            # 4. Filtrage à la fenêtre des masks nuage et neige
                            snow_probs_window = snow_probs[:, x : x + width, y : y + height]  # T * H * W
                            cloud_probs_window = cloud_probs[:, x : x + width, y : y + height]  # T * H * W

                            idx_good_frames = SentinelDataProcessor.filter_dates(
                                np.stack([snow_probs_window, cloud_probs_window], axis=-1)
                            )  # T * H * W * 2
                            idx_cloudy_frames = np.asarray(
                                [d for d in range(len(dates_s2)) if d not in idx_good_frames]
                            )
                            dates_s2_valid = [dates_s2[t] for t in idx_good_frames]

                            # 6. En fonction de la tailles des séries de dates non-nuageuses
                            # garder ou extraire la TS / patch du dataset
                            if self.min_seq_length is not None and len(dates_s2_valid) < self.min_seq_length:
                                index_to_drop.append(row_index)
                            else:
                                mgrs25_data[row_index] = {
                                    "idx_good_frames": idx_good_frames.tolist(),
                                    "idx_cloudy_frames": idx_cloudy_frames.tolist(),
                                    "masks_valid_obs": idx_good_frames.tolist(),
                                    "dates_s2_valid": dates_s2_valid,
                                }

                        # Après collect des index des dates nuages / non nuageuses, ajout à la dataframe mgrs25
                        mgrs25_dataset = mgrs25_dataset.drop(index=index_to_drop)
                        for label in [
                            "idx_cloudy_frames",
                            "idx_good_frames",
                            "masks_valid_obs",
                            "dates_s2_valid",
                        ]:
                            mgrs25_dataset[label] = [v[label] for v in mgrs25_data.values()]

                        mgrs25_s2 = SentinelDataProcessor.read_raster_per_dates(
                            path_raster=mgrs25_files[0], type_bands="s2"
                        )  # T x C x H x W
                        mgrs25_s2 = mgrs25_s2[:, self.s2_channels, :, :]  # Sélection des canaux S2

                        (
                            list_index_prelevement_asc,  # juste pour extraction données asc
                            list_index_prelevement_desc,  # juste pour extraction données desc
                            dates_s1_asc_collected,  # à mettre dans le hdf5
                            dates_s1_desc_collected,  # à mettre dans le hdf5
                            dict_appariement,
                        ) = appariement_S1_to_S2(
                            S2_dates=dates_s2,
                            S1_dates_asc=dates_s1_asc,
                            S1_dates_desc=dates_s1_desc,
                        )

                        path_s1_asc, path_s1_desc = mgrs25_files[1], mgrs25_files[2]
                        mgrs25_s1_asc = SentinelDataProcessor.read_raster_per_dates(
                            path_raster=path_s1_asc,
                            indexes_dates=list_index_prelevement_asc,
                            type_bands="s1",
                        )  # T x C x H x W

                        mgrs25_s1_desc = SentinelDataProcessor.read_raster_per_dates(
                            path_raster=path_s1_desc,
                            indexes_dates=list_index_prelevement_desc,
                            type_bands="s1",
                        )  # T x C x H x W

                        # Vérification de la cohérence des données
                        # assert len(dates_s2) == len(dates_s1), "Number of S2 dates must match the number of S1 images."

                        for row_index, row in tqdm(
                            mgrs25_dataset.iterrows(), total=mgrs25_dataset.shape[0], desc="Windows loading"
                        ):
                            # Découpage des données selon la fenêtre
                            window = row.window
                            x, y, width, height = window[0], window[1], window[2], window[3]
                            windows_str = "_".join(map(str, window))

                            # 4. Filtrage à la fenêtre des masks nuage et neige
                            cloud_probs_window = cloud_probs[:, x : x + width, y : y + height]
                            snow_probs_window = snow_probs[:, x : x + width, y : y + height]
                            cloud_masks_window = (cloud_probs_window != 0).astype(int)

                            idx_selected = np.asarray(mgrs25_dataset.loc[row_index, "idx_good_frames"])
                            s2_dates_non_cloudy = mgrs25_dataset.loc[row_index, "dates_s2_valid"]

                            if not in_test_set:
                                # Collecte des dates S1 correspondantes aux dates S2 non nuageuses
                                s2_dates = s2_dates_non_cloudy
                                s2 = mgrs25_s2[idx_selected, :, x : x + width, y : y + height]

                                s1_asc_non_cloudy_indexes, s1_desc_non_cloudy_indexes = [], []
                                for s2_date in s2_dates_non_cloudy:
                                    # ASC
                                    s1_date_asc, _, i_asc = dict_appariement["asc"][s2_date]
                                    assert i_asc == dates_s1_asc_collected.index(s1_date_asc)
                                    s1_asc_non_cloudy_indexes.append(i_asc)
                                    # DESC
                                    s1_date_desc, _, i_desc = dict_appariement["desc"][s2_date]
                                    assert i_desc == dates_s1_desc_collected.index(s1_date_desc)
                                    s1_desc_non_cloudy_indexes.append(i_desc)

                                s1_asc = mgrs25_s1_asc[s1_asc_non_cloudy_indexes, :, x : x + width, y : y + height]
                                s1_dates_asc = np.asarray(dates_s1_asc_collected)[s1_asc_non_cloudy_indexes].tolist()

                                s1_desc = mgrs25_s1_desc[s1_desc_non_cloudy_indexes, :, x : x + width, y : y + height]
                                s1_dates_desc = np.asarray(dates_s1_desc_collected)[s1_desc_non_cloudy_indexes].tolist()

                            else:
                                # In test set, we only keep the S2 data and the cloud masks
                                s2 = mgrs25_s2[:, :, x : x + width, y : y + height]
                                s2_dates = np.asarray(dates_s2).tolist()
                                s1_asc = mgrs25_s1_asc[:, :, x : x + width, y : y + height]
                                s1_dates_asc = np.asarray(dates_s1_asc_collected).tolist()
                                s1_desc = mgrs25_s1_desc[:, :, x : x + width, y : y + height]
                                s1_dates_desc = np.asarray(dates_s1_desc_collected).tolist()

                                cloud_probs_window_aleatoire = cloud_probs_aleatoire[:, x : x + width, y : y + height]
                                cloud_probs_window_consecutif = cloud_probs_consecutif[:, x : x + width, y : y + height]

                                index_syn_aleatoire = np.asarray(
                                    [t for t in range(len(s2_dates)) if cloud_probs_window_aleatoire[t].mean() > 150]
                                )

                                index_syn_consecutif = np.asarray(
                                    [t for t in range(len(s2_dates)) if cloud_probs_window_consecutif[t].mean() > 150]
                                )

                            if s2.shape[2] != self.image_size[0] or s2.shape[3] != self.image_size[1]:
                                continue
                            if s1_asc.shape[2] != self.image_size[0] or s1_asc.shape[3] != self.image_size[1]:
                                continue
                            if s1_desc.shape[2] != self.image_size[0] or s1_desc.shape[3] != self.image_size[1]:
                                continue
                            if (
                                cloud_probs_window.shape[1] != self.image_size[0]
                                or cloud_probs_window.shape[2] != self.image_size[1]
                            ):
                                continue

                            sample = {
                                "S1": {
                                    "S1_asc": s1_asc,  # Bandes S1 asc correspondantes aux dates S2 valides
                                    "S1_desc": s1_desc,  # Bandes S1 desc correspondantes
                                    "S1_dates_asc": s1_dates_asc,  # Dates S1 asc correspondantes aux bandes
                                    "S1_dates_desc": s1_dates_desc,  # Dates S1 desc correspondantes aux bandes
                                    "S2_S1_pairing": dict_appariement,  # Appariement S2-S1
                                },
                                "S2": {
                                    "S2": s2,  # Bandes correspondant aux dates correctes de la TS
                                    "S2_dates": s2_dates,  # Dates correctes de la TS
                                    "cloud_mask": cloud_masks_window,  # Mask entier de la TS
                                    "cloud_prob": cloud_probs_window.astype(np.float32),  # Probs cloud entier de la TS
                                },
                                "idx_cloudy_frames": np.asarray(mgrs25_dataset.loc[row_index, "idx_cloudy_frames"]),
                                "idx_good_frames": np.asarray(mgrs25_dataset.loc[row_index, "idx_good_frames"]),
                                "idx_impaired_frames": np.asarray(mgrs25_dataset.loc[row_index, "idx_cloudy_frames"]),
                                "valid_obs": np.asarray(mgrs25_dataset.loc[row_index, "masks_valid_obs"]),
                            }

                            if in_test_set:
                                sample.update(
                                    {
                                        "idx_syn_aleatoire": index_syn_aleatoire,
                                        "idx_syn_consecutif": index_syn_consecutif,
                                    }
                                )

                            window_group = mgrs25_group.create_group(windows_str)
                            for key, value in sample.items():
                                if isinstance(value, dict):
                                    window_subgroup = window_group.create_group(key)
                                    for meta_key, meta_value in value.items():
                                        if isinstance(meta_value, np.ndarray):
                                            window_subgroup.create_dataset(
                                                meta_key,
                                                data=meta_value,
                                                compression="gzip",
                                                compression_opts=9,
                                            )
                                        elif isinstance(meta_value, dict):
                                            window_subgroup.create_dataset(meta_key, data=json.dumps(meta_value))
                                        else:
                                            window_subgroup.create_dataset(meta_key, data=meta_value)
                                else:
                                    window_group.create_dataset(key, data=value)


######################################################################################
######################################################################################
######################################################################################

if __name__ == "__main__":
    print("Entering CIRCA HDF5 Maker...")
    # path_dataset_circa_sample = Path("/home/SPeillet/cloud_reconstruction/data/circa/CIRCA_MGRS25_SAMPLE")
    # data_optique = path_dataset_circa_sample / "optique_dataset"
    # data_radar = path_dataset_circa_sample / "radar_dataset_v4"
    # image_size = [256, 256]
    # overlap = 0
    # hdf5_file = Path("/home/SPeillet/cloud_reconstruction/data/circa/hdf5") / "new_circa_ligth.hdf5"

    path_dataset_circa = Path("/mnt/stores/store-dai/projets/pac/3str/EXP_2/Data_Raster")
    data_optique = path_dataset_circa / "optique_dataset"
    data_radar = path_dataset_circa / "radar_dataset_v4"
    image_size = [256, 256]
    overlap = 0
    # hdf5_file = Path("/home/SPeillet/cloud_reconstruction/data/circa/hdf5") / "CIRCA_CR.hdf5"
    hdf5_folder = Path("/home/SPeillet/cloud_reconstruction/data/circa/hdf5/archives_MGRSC")
    filter_settings = {
        "type": "cloud-free",  # Strategy for removing observations with data gaps.
        # ['cloud-free', 'cloud-free_consecutive']
        "min_length": 10,  # Minimum sequence length.
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

    # Si export des données vers un fichier hdf5
    dataset = CIRCA_HDF5_Maker(
        hdf5_folder=hdf5_folder,
        data_optique=data_optique,
        data_radar=data_radar,
        image_size=image_size,
        overlap=overlap,
        filter_settings=filter_settings,
        mask_kwargs=mask_kwargs,
    )
    dataset.load_items_to_hdf5()

    # Import des données depuis un fichier hdf5
    # dataset = CIRCA_HDF5_Dataset(
    #     hdf5_file_read=output_file,
    #     filter_settings=filter_settings,
    #     mask_kwargs=mask_kwargs,
    #     max_seq_length=10,
    # )
    # sample = next(iter(dataset))
    # print(sample.keys())
