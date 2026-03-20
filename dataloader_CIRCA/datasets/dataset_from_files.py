import ast
import datetime as dt
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rasterio
import torch
from numpy import ndarray
from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.tools.mask_generation import masks_init_filling
from dataloader_CIRCA.tools.positional_encoding import get_position_for_positional_encoding

ChannelType = Literal["all", "bgr-nir"]


class Dataset_from_files(Dataset):
    """
    A custom PyTorch Dataset designed to manage Sentinel-1 and Sentinel-2 data specifically for cloud
    reconstruction tasks.
    """

    def __init__(
        self,
        data_optique: str | Path,
        data_radar: str | Path,
        mgrsc: str | None,
        image_size: tuple = (256, 256),
        overlap: int | None = 0,
        load_dataset: str | None = None,
        shuffle: bool = False,
        use_sar: bool = "mix_closest",
        no_filter: bool = False,
        channels: ChannelType = "all",
        pe_strategy: str = "day-within-sequence",
        fill_value: float = 1.0,
        mask_type: str = "orignal_masks",  # orignal_masks, fully_masked
    ):
        """
        Initializes the dataset.

        Parameters:
        - data_optique (Union[str, Path]): Path to the optical data directory.
        - data_radar (Union[str, Path]): Path to the radar data directory.
        - image_size (int, optional): Size of the patches to extract. Defaults to 256.
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        - shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        - use_sar (bool, optional): Whether to include SAR data in the dataset. Defaults to True.
        """
        self.data_optique = Path(data_optique)
        self.data_radar = Path(data_radar)
        self.mgrsc = mgrsc
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.overlap = overlap
        self.load_dataset = load_dataset
        self.shuffle = shuffle
        self.zones_dataset, self.dates_dict = None, None
        self.patches_dataset = None
        self.use_sar = use_sar
        self.no_filter = no_filter
        self.num_channels, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels(channels)
        self.s2_tile = None
        self.s1_asc_tile = None
        self.s1_desc_tile = None
        self.pe_strategy = pe_strategy
        self.fill_value = fill_value
        self.mask_type = mask_type
        self.setup()

    def __len__(self) -> int:
        """
        Returns the number of patches in the dataset.

        Returns:
        - int: Number of patches.
        """
        return len(self.mgrsc_dataset)

    def setup(self, load_dataset: str | None = None) -> None:
        """
        Sets up the dataset by either loading from a pre-saved file or processing the data.

        Parameters:
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        """
        if self.load_dataset is not None or load_dataset is not None:
            self.load_exported_data(load_dataset or self.load_dataset)
        else:
            self.setup_zones()
            self.setup_patches()

        if self.shuffle:
            self.patches_dataset = self.patches_dataset.sample(frac=1).reset_index(drop=True)

        self.setup_mgrsc_parameters()
        self.setup_mgrsc_files()
        if self.use_sar:
            self.setup_s1_dates()

    def setup_channels(self, channels: ChannelType) -> tuple[int, torch.Tensor, torch.Tensor, list[int]]:
        """
        Configure channel settings based on the specified channel mode.

        Args:
            channels: Channel configuration ('all' or 'bgr-nir')

        Returns:
            Tuple containing:
            - Number of channels
            - RGB channel indices tensor
            - NIR channel index tensor
            - List of Sentinel-2 channel indices

        Raises:
            ValueError: If invalid channel configuration is specified
        """
        if channels == "all":
            num_channels = 10
            c_index_rgb = torch.tensor([2, 1, 0], dtype=torch.long)
            c_index_nir = torch.tensor([6], dtype=torch.long)
            s2_channels = list(range(10))
        elif channels == "bgr-nir":
            num_channels = 4
            c_index_rgb = torch.tensor([2, 1, 0], dtype=torch.long)
            c_index_nir = torch.tensor([6], dtype=torch.long)
            s2_channels = [0, 1, 2, 6]
        else:
            raise ValueError(f"Channels {channels} not recognized. Use 'all' or 'bgr-nir'.")

        if self.use_sar:
            if self.use_sar == "asc+desc":
                num_channels += 8
            elif self.use_sar == "asc" or self.use_sar == "desc" or self.use_sar == "mix_closest":
                num_channels += 4
            else:
                raise ValueError(
                    f"SAR pairing {self.use_sar} not recognized. Use 'asc+desc', 'asc', 'desc' or 'mix_closest'."
                )
        return num_channels, c_index_rgb, c_index_nir, s2_channels

    def setup_mgrsc_parameters(self) -> None:
        """
        Sets up parameters specific to a given MGRS-C area.

        Parameters:
        - mgrsc (Optional[str]): The MGRS-C area identifier.
        """
        if self.mgrsc is None:
            raise ValueError("MGRS-C area must be specified for this operation.")
        # Implement any specific setup required for the given MGRS-C area
        self.mgrsc_dataset = (
            self.patches_dataset[self.patches_dataset["mgrs25"] == self.mgrsc].copy().reset_index(drop=True)
        )
        self.dates_s2 = np.array([self.str2date(date) for date in self.dates_dict[self.mgrsc]["S2"]])
        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        self.days = get_position_for_positional_encoding(self.dates_s2, "day-within-sequence")
        # Get positions for positional encoding
        self.position_days = get_position_for_positional_encoding(self.dates_s2, self.pe_strategy)

    def setup_mgrsc_files(self) -> None:
        # listage des fichiers à utiliser pour itérer sur les patchs contenus dans la zone MGRS-C
        files = set([el for sublist in self.mgrsc_dataset["files"].tolist() for el in sublist])
        if self.use_sar:
            assert len(files) == 3, "Expected exactly 3 files (S2, S1_ASC, S1_DESC) for the MGRS-C area."
        else:
            assert len(files) == 1, "Expected exactly 1 file (S2) for the MGRS-C area."
        self.s2_file = [f for f in files if "ASC" not in f and "DESC" not in f][0]
        if self.use_sar:
            self.s1_asc_file = [f for f in files if "ASC" in f][0]
            self.s1_desc_file = [f for f in files if "DESC" in f][0]
        with rasterio.open(self.s2_file) as src:
            self.s2_meta = src.meta.copy()
        if self.use_sar:
            with rasterio.open(self.s1_asc_file) as src:
                self.s1_asc_meta = src.meta.copy()
            with rasterio.open(self.s1_desc_file) as src:
                self.s1_desc_meta = src.meta.copy()

    def setup_s1_dates(self) -> None:
        self.dates_s1_asc = self.dates_dict[self.mgrsc]["S1"]["ASC"]
        self.dates_s1_desc = self.dates_dict[self.mgrsc]["S1"]["DESC"]
        self.closest_dates_matches = SentinelDataProcessor.get_pairedS1_closest_matches(
            dates_S2=self.dates_s2,
            dates_S1_asc=self.dates_s1_asc,
            dates_S1_desc=self.dates_s1_desc,
        )

    def load_exported_data(self, path_data: str | Path) -> None:
        """
        Loads the dataset from a pre-saved CSV file.

        Parameters:
        - path_data (Union[str, Path]): Path to the CSV file.
        """
        self.patches_dataset = pd.read_csv(path_data)
        cols_to_convert = [
            "window",
            "files",
            "dates_S2",
            "dates_S1_ASC",
            "dates_S1_DESC",
        ]
        self.patches_dataset[cols_to_convert] = self.patches_dataset[cols_to_convert].map(ast.literal_eval)

        if (
            self.patches_dataset.loc[0, "window"][2] != self.image_size[0]
            and self.patches_dataset.loc[0, "window"][3] != self.image_size[1]
        ):
            print(
                """
                WARNING : Patch size load in the .csv file is not corresponding to the patch size requested 
                during the dataset initialization. New setup of the dataset will be done.."
                """
            )
            self.setup()

        # Recreate the dates dictionary after loading
        self.dates_dict = {
            mgrs25: {
                "S2": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]["dates_S2"].values[0],
                "S1": {
                    "ASC": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]["dates_S1_ASC"].values[0],
                    "DESC": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]["dates_S1_DESC"].values[0],
                },
            }
            for mgrs25 in self.patches_dataset["mgrs25"].unique()
        }

    def setup_patches(self) -> None:
        """
        Sets up the patches by processing the data and creating a DataFrame.
        """
        self.patches_dataset = pd.DataFrame(columns=["patch", "window", "mgrs", "mgrs25", "files"]).astype(object)

        for _, row in tqdm(self.zones_dataset.iterrows(), total=len(self.zones_dataset), leave=False):
            for window in row.windows:
                windows_str = "_".join(map(str, window))
                patch_df = pd.DataFrame(
                    {
                        "patch": f"patches_{row.mgrs25}_window_{windows_str}",
                        "window": [window],
                        "mgrs": row.mgrs,
                        "mgrs25": row.mgrs25,
                        "files": [row.files],  # 0: S2, 1: S1_ASC, 2: S1_DESC
                        "dates_S2": [row.dates_S2],
                        "dates_S1_ASC": [row.dates_S1_ASC],
                        "dates_S1_DESC": [row.dates_S1_DESC],
                    }
                )
                self.patches_dataset = pd.concat([self.patches_dataset, patch_df], ignore_index=True)

    def setup_zones(self) -> None:
        """
        Sets up the zones by processing the data and creating a DataFrame.
        """
        self.zones_dataset = pd.DataFrame(columns=["mgrs", "mgrs25", "files", "windows"]).astype(object)
        self.dates_dict = {}

        for mgrs in tqdm(list(self.data_optique.iterdir()), leave=False, desc="mgrs"):
            for mgrs25 in tqdm(list(mgrs.iterdir()), leave=False, desc="mgrs25"):
                self.process_mgrs25_zone(mgrs, mgrs25)

    def process_mgrs25_zone(self, mgrs: Path, mgrs25: Path) -> None:
        """
        Processes a single MGRS25 zone.

        Parameters:
        - mgrs (Path): Path to the MGRS zone.
        - mgrs25 (Path): Path to the MGRS25 zone.
        """
        mgrs_name = mgrs.stem
        mgrs25_name = mgrs25.stem[7:]
        self.dates_dict[mgrs25_name] = {}

        # Process Sentinel-2 data
        list_tifs_optique = sorted(mgrs25.rglob("*.tif"))
        list_jsons_optique = sorted(mgrs25.rglob("*.json"))
        dates_S2 = json.load(open(list_jsons_optique[0]))
        self.dates_dict[mgrs25_name]["S2"] = dates_S2

        # Process Sentinel-1 data
        mgrs25_radar = self.data_radar / mgrs_name / ("MGRS25-" + mgrs25_name)
        assert mgrs25_radar.exists()

        list_tifs_radar = sorted(mgrs25_radar.rglob("*.tif"))
        list_jsons_radar = sorted(mgrs25_radar.rglob("*.json"))

        tif_files = {"S2": list_tifs_optique[0]}
        for file in list_tifs_radar:
            if file.stem.endswith("ASC"):
                tif_files["S1_ASC"] = file
            else:
                tif_files["S1_DESC"] = file

        list_windows = SentinelDataProcessor.split_raster_into_windows(
            path_raster=tif_files["S2"],
            image_size=self.image_size,
            overlap=self.overlap,
        )

        self.dates_dict[mgrs25_name]["S1"] = {
            file.stem.split("_")[-1]: json.load(open(file)) for file in list_jsons_radar
        }

        data = {
            "mgrs": mgrs_name,
            "mgrs25": mgrs25_name,  # Remove the 'MGRS25-' prefix
            "files": [sorted(f.as_posix() for f in tif_files.values())],
            "windows": [list_windows],
            "dates_S2": [dates_S2],
            "dates_S1_ASC": [self.dates_dict[mgrs25_name]["S1"]["ASC"]],
            "dates_S1_DESC": [self.dates_dict[mgrs25_name]["S1"]["DESC"]],
        }
        df_temp = pd.DataFrame(data).astype(object)
        self.zones_dataset = pd.concat([self.zones_dataset, df_temp], ignore_index=True)

    def export_dataset(self, outpath: str | Path = "tiles_windows.csv") -> None:
        """
        Exports the dataset to a CSV file.

        Parameters:
        - outpath (Union[str, Path], optional): Path to save the CSV file. Defaults to "datasetCIRCAUnCRtainTS.csv".
        """
        self.patches_dataset.to_csv(outpath, index=False)

    def str2date(self, date_string: str) -> dt.date:
        """
        Convert a date string in format 'YYYYMMDD' to datetime object.

        Args:
            date_string: Date string in format 'YYYYMMDD'

        Returns:
            Corresponding datetime.date object

        Example:
            >>> str2date("20200101")
            datetime.date(2020, 1, 1)
        """
        return dt.datetime.strptime(date_string, "%Y%m%d")

    def decode_dates(self, dates: ndarray[np.bytes_]) -> list[str]:
        """
        Decode byte strings in date array to UTF-8 strings.

        Args:
            dates: Array of date byte strings

        Returns:
            List of decoded date strings
        """
        return [el.decode("utf-8") for el in dates]

    def __getitem__(self, item: int, t_sampled: list | None = None) -> dict[str, np.ndarray | str | list[str]]:
        """
        Retrieves an item from the dataset.

        Parameters:
        - item (int): Index of the item to retrieve.

        Returns:
        - Dict[str, Union[np.ndarray, str, List[str]]]: A dictionary containing the data, name, masks, and dates.
        """
        patch_data = self.mgrsc_dataset.iloc[item]

        # Determine the number of channels per chunk from SentinelDataProcessor constants or fallback
        S2_N_CHANNELS = 12
        S1_N_CHANNELS = 4

        # Extraction données S2
        if t_sampled is not None:
            bands_s2 = []
            for t in t_sampled:
                bands_s2.extend([t * S2_N_CHANNELS + c + 1 for c in range(S2_N_CHANNELS)])
            with rasterio.open(self.s2_file) as src_S2:
                patch_S2_array = src_S2.read(bands_s2, window=Window(*patch_data.window))
            patch_S2_array = patch_S2_array.reshape(len(t_sampled), S2_N_CHANNELS, patch_S2_array.shape[-2], patch_S2_array.shape[-1])
        else:
            with rasterio.open(self.s2_file) as src_S2:
                patch_S2_array = src_S2.read(window=Window(*patch_data.window))
            T = patch_S2_array.shape[0] // S2_N_CHANNELS
            patch_S2_array = patch_S2_array.reshape(T, S2_N_CHANNELS, patch_S2_array.shape[-2], patch_S2_array.shape[-1])

        patch_S2_array = torch.from_numpy(patch_S2_array.astype(np.float32))

        # patch_S2_array = self.s2_tile[:, :, y : y + h, x : x + w]  # Extraction données S2
        # Pas de filtrage sur les dates sur les données s2 -> filtering has already been applied if t_sampled!
        data_s2 = patch_S2_array[:, 0:10, ...]
        patch_S2_array[:, 10, ...] = patch_S2_array[:, 10, ...]  # cloud mask synthetic data

        data_s2 = SentinelDataProcessor.process_MS(data_s2)
        # faire une récupération des données synthétiques
        original_masks = patch_S2_array[:, 10:, ...]

        cloud_probs = original_masks[:, 0, ...].clone().unsqueeze(axis=1)

        # FIX: Identifier les pixels sans données (No Data = 0 sur tous les canaux S2) et les ajouter au masque de nuages / à reconstruire
        missing_data_mask = torch.all(data_s2 == 0, dim=1, keepdim=True)
        cloud_probs[missing_data_mask] = 1.0

        cloud_masks = (cloud_probs > 0).float()  # Binarization of cloud masks

        if self.use_sar:
            s1_tile, s1_dates = [], []
            indices_s2 = t_sampled if t_sampled is not None else range(len(self.closest_dates_matches))

            asc_bands, desc_bands = [], []
            asc_indices_map, desc_indices_map = {}, {}

            for index_s2 in indices_s2:
                _, date_s1, index_s1, orbit_type = self.closest_dates_matches[index_s2]
                if orbit_type == "ASC":
                    if index_s1 not in asc_indices_map:
                        asc_indices_map[index_s1] = True
                        asc_bands.extend([index_s1 * S1_N_CHANNELS + c + 1 for c in range(S1_N_CHANNELS)])
                elif index_s1 not in desc_indices_map:
                    desc_indices_map[index_s1] = True
                    desc_bands.extend([index_s1 * S1_N_CHANNELS + c + 1 for c in range(S1_N_CHANNELS)])

            asc_bands = sorted(set(asc_bands))
            desc_bands = sorted(set(desc_bands))

            s1_tile_asc_dict = {}
            if len(asc_bands) > 0:
                with rasterio.open(self.s1_asc_file) as src_s1:
                    asc_data = src_s1.read(asc_bands, window=Window(*patch_data.window))
                    for i, band in enumerate(asc_bands):
                        idx = (band - 1) // S1_N_CHANNELS
                        c = (band - 1) % S1_N_CHANNELS
                        if idx not in s1_tile_asc_dict:
                            s1_tile_asc_dict[idx] = np.zeros((S1_N_CHANNELS, asc_data.shape[-2], asc_data.shape[-1]), dtype=asc_data.dtype)
                        s1_tile_asc_dict[idx][c] = asc_data[i]

            s1_tile_desc_dict = {}
            if len(desc_bands) > 0:
                with rasterio.open(self.s1_desc_file) as src_s1:
                    desc_data = src_s1.read(desc_bands, window=Window(*patch_data.window))
                    for i, band in enumerate(desc_bands):
                        idx = (band - 1) // S1_N_CHANNELS
                        c = (band - 1) % S1_N_CHANNELS
                        if idx not in s1_tile_desc_dict:
                            s1_tile_desc_dict[idx] = np.zeros((S1_N_CHANNELS, desc_data.shape[-2], desc_data.shape[-1]), dtype=desc_data.dtype)
                        s1_tile_desc_dict[idx][c] = desc_data[i]

            for index_s2 in indices_s2:
                _, date_s1, index_s1, orbit_type = self.closest_dates_matches[index_s2]
                s1_band = s1_tile_asc_dict[index_s1] if orbit_type == "ASC" else s1_tile_desc_dict[index_s1]
                s1_tile.append(s1_band)
                s1_dates.append(date_s1)

            s1_tile = np.stack(s1_tile, axis=0)
            s1_tile = torch.from_numpy(s1_tile.astype(np.float32))
            dates_s1_sampled = np.array([self.str2date(date) for date in s1_dates])
            data_s1 = SentinelDataProcessor.process_SAR(s1_tile)
        if self.mask_type == "fully_masked":
            # Couverture complète
            cloud_probs = original_masks[:, 0, ...].clone().unsqueeze(axis=1)
            snow_probs = original_masks[:, 1, ...].clone().unsqueeze(axis=1)
            masks_to_filter = np.concatenate([snow_probs.numpy(), cloud_probs.numpy()], axis=1)
            masks_to_filter = masks_to_filter.transpose(0, 2, 3, 1)  # T * H * W * 2
            # Filter dates according to cloud masks
            idx_good_frames = SentinelDataProcessor.filter_dates(masks_to_filter)  # T * H * W * 2
            if t_sampled is not None:
                idx_cloudy_frames = np.asarray([d for d in range(len(t_sampled)) if d not in idx_good_frames])
            else:
                idx_cloudy_frames = np.asarray([d for d in range(len(self.dates_s2)) if d not in idx_good_frames])
            images_masked = data_s2.clone()
            masks = torch.zeros_like(cloud_masks)  # Dummy masks (not used in fully masked mode)
            for idx in idx_cloudy_frames:
                images_masked[idx] = self.fill_value  # Set cloudy frames to fill value
                masks[idx] = self.fill_value  # Mark these frames as cloudy in the masks (if needed for analysis)
        else:
            # Ajout des masques de nuages originaux dans l'input
            # Image time series with overlaid cloud masks filled with value `fill_value`
            images_masked, masks = masks_init_filling(
                seq=data_s2.clone(),
                masks=cloud_masks.clone(),
                fill_type="fill_value",
                fill_value=self.fill_value,
                dilate_cloud_masks=False,
            )

        frames_input = torch.cat((images_masked, data_s1), dim=1) if self.use_sar else images_masked

        # CRITICAL FIX: Ensure SAR channels are also masked to self.fill_value when clouds are present.
        # During training (in UTILISE_adapter), all 12 channels are masked.
        if self.use_sar and masks is not None:
            frames_input = frames_input.masked_fill(masks == 1.0, self.fill_value)

        frames_target = data_s2.clone()

        masks_valid_obs = torch.ones(frames_input.shape[0], dtype=torch.uint8)

        if t_sampled is not None:
            out = {
                "x": frames_input,  # already filtered
                "y": frames_target,
                "masks": masks,
                "masks_valid_obs": masks_valid_obs,
                "position_days": self.position_days[t_sampled],
                "days": self.days[t_sampled] - self.days[t_sampled][0],
                "sample_index": item,
                "c_index_rgb": self.c_index_rgb,
                "c_index_nir": self.c_index_nir,
                "S2_dates": [self.dates_s2[i].strftime("%Y-%m-%d") for i in t_sampled],
                "original_masks": original_masks,
                "cloud_mask": cloud_masks,
                "window": patch_data["window"],
            }
            if self.use_sar:
                out["S1_dates"] = [dates_s1_sampled[i].strftime("%Y-%m-%d") for i in range(len(t_sampled))]
        else:
            out = {
                "x": frames_input,
                "y": frames_target,
                "masks": masks,
                "masks_valid_obs": masks_valid_obs,
                "position_days": self.position_days,
                "days": self.days - self.days[0],
                "sample_index": item,
                "c_index_rgb": self.c_index_rgb,
                "c_index_nir": self.c_index_nir,
                "S2_dates": [date.strftime("%Y-%m-%d") for date in self.dates_s2],
                "original_masks": original_masks,
                "cloud_mask": cloud_masks,
                "window": patch_data["window"],
            }
            if self.use_sar:
                out["S1_dates"] = [date.strftime("%Y-%m-%d") for date in dates_s1_sampled]
        return out
