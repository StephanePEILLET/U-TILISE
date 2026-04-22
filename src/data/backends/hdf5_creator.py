"""
hdf5_creator.py — Module consolidé pour la création de fichiers HDF5 à partir des données CIRCA.

Ce module regroupe toute la chaîne de création HDF5 :
1. Lecture et indexation des fichiers TIF Sentinel-1/2 (classe FileScanner)
2. Appariement temporel des dates S1 ASC/DESC vers les dates S2 (fonctions d'appariement)
3. Écriture des données dans des fichiers HDF5 par zone MGRS (classe HDF5Maker)
4. Fusion des fichiers HDF5 individuels en un seul fichier (fonction merge_hdf5_files)

Pipeline typique :
    1. Instancier HDF5Maker avec les chemins vers les données raster
    2. Appeler load_items_to_hdf5() pour écrire un fichier HDF5 par zone MGRS
    3. Appeler merge_hdf5_files() pour fusionner tous les fichiers en un seul

Note sur les splits train/val vs test :
    - Train/Val : on ne stocke que les dates cloud-free (S2, S1, dates) → fichier plus petit
    - Test : on stocke TOUTES les dates (S2, S1, dates, cloud_mask, cloud_prob)
      car l'inférence en mode produit nécessite toutes les dates.
    - Pour train/val, valid_obs est un vecteur binaire de 1 (toutes les frames stockées sont valides).
    - Pour test, valid_obs contient les indices des frames non-nuageuses dans la série complète.
"""

import ast
import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data.backends.constants import GEOGRAPHIC_SPLITS
from src.data.processing.transforms import SentinelDataProcessor
from src.data.processing.positional import get_position_for_positional_encoding  # NOQA

torch.multiprocessing.set_sharing_strategy("file_system")


# =============================================================================
# Constantes
# =============================================================================

MAX_SEQ_LENGTH = 30
MIN_SEQ_LENGTH = 5
SEED = 42


# =============================================================================
# Section 1 : Lecture et indexation des fichiers TIF (FileScanner)
# =============================================================================


class FileScanner(Dataset):
    """
    Dataset PyTorch pour la gestion des données Sentinel-1 et Sentinel-2
    à partir de fichiers TIF pour les tâches de reconstruction sans nuages.

    Cette classe scanne les répertoires de données optiques (S2) et radar (S1),
    découpe les rasters en patches de taille fixe, et permet d'itérer sur
    les patches pour l'entraînement ou l'export HDF5.
    """

    def __init__(
        self,
        data_optique: str | Path,
        data_radar: str | Path,
        image_size: tuple = (256, 256),
        overlap: int | None = 0,
        load_dataset: str | None = None,
        shuffle: bool = False,
        use_sar: bool = True,
        no_filter: bool = False,
    ):
        """
        Initialise le dataset FileScanner.

        Parameters:
        - data_optique (str | Path): Chemin vers le répertoire racine des données optiques
            Sentinel-2. Structure attendue : data_optique/<MGRS>/<MGRS25-XXXXX>/*.tif.
            Chaque fichier TIF contient les bandes S2 empilées (T dates × C bandes × H × W).
        - data_radar (str | Path): Chemin vers le répertoire racine des données radar
            Sentinel-1. Structure attendue : data_radar/<MGRS>/<MGRS25-XXXXX>/*.tif.
            Les fichiers sont suffixés par _ASC ou _DESC selon l'orbite.
        - image_size (tuple, optional): Taille des patches spatiaux à extraire, au format
            (hauteur, largeur). Si un int est passé, il est converti en (int, int).
            Defaults to (256, 256).
        - overlap (int | None, optional): Chevauchement entre patches adjacents en pixels.
            0 = pas de chevauchement. Defaults to 0.
        - load_dataset (str | None, optional): Chemin vers un fichier CSV pré-exporté
            (via export_dataset()). Si fourni, le scan des répertoires TIF est sauté.
            Defaults to None.
        - shuffle (bool, optional): Si True, l'ordre des patches est mélangé après le setup.
            Defaults to False.
        - use_sar (bool, optional): Si True, les données Sentinel-1 (SAR) sont incluses.
            Si False, seules les données S2 (optiques) sont retournées par __getitem__.
            Defaults to True.
        - no_filter (bool, optional): Si True, le filtrage des dates nuageuses est désactivé.
            Toutes les dates S2 sont retournées sans filtrage (utile pour le mode test).
            Defaults to False.
        """
        self.data_optique = Path(data_optique)
        self.data_radar = Path(data_radar)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.overlap = overlap
        self.load_dataset = load_dataset
        self.shuffle = shuffle
        self.zones_dataset, self.dates_dict = None, None
        self.patches_dataset = None
        self.use_sar = use_sar
        self.no_filter = no_filter
        self.setup()

    def __len__(self) -> int:
        """
        Retourne le nombre total de patches dans le dataset.

        Returns:
        - int: Nombre de lignes dans self.patches_dataset, i.e. le nombre de patches
            spatiaux découpés à partir de toutes les zones MGRS25.
        """
        return len(self.patches_dataset)

    def setup(self, load_dataset: str | None = None) -> None:
        """
        Configure le dataset depuis un CSV pré-existant ou en scannant les répertoires TIF.

        Parameters:
        - load_dataset (str | None, optional): Chemin optionnel vers un fichier CSV de
            dataset pré-exporté. Si fourni, override self.load_dataset. Defaults to None.

        Returns:
        - None: Met à jour self.patches_dataset et self.dates_dict en place.
        """
        if self.load_dataset is not None or load_dataset is not None:
            self.load_exported_data(load_dataset or self.load_dataset)
        else:
            self.setup_zones()
            self.setup_patches()

        if self.shuffle:
            self.patches_dataset = self.patches_dataset.sample(frac=1).reset_index(drop=True)

    def load_exported_data(self, path_data: str | Path) -> None:
        """
        Charge le dataset depuis un fichier CSV pré-exporté (via export_dataset).
        Les colonnes 'window', 'files', 'dates_S2', 'dates_S1_ASC', 'dates_S1_DESC'
        sont désérialisées via ast.literal_eval (passage str → list/tuple).

        Parameters:
        - path_data (str | Path): Chemin vers le fichier CSV contenant les métadonnées.
            Colonnes attendues : patch, window, mgrs, mgrs25, files, dates_S2,
            dates_S1_ASC, dates_S1_DESC.

        Returns:
        - None: Met à jour self.patches_dataset (pd.DataFrame) et self.dates_dict (dict).
            Reconstruit dates_dict à partir des colonnes du CSV.
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
                "WARNING : La taille des patches dans le CSV ne correspond pas à la taille demandée. "
                "Reconfiguration du dataset..."
            )
            self.setup()

        # Reconstruction du dictionnaire de dates après chargement
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
        Crée la DataFrame des patches à partir des zones identifiées dans self.zones_dataset.
        Pour chaque zone MGRS25 et chaque fenêtre spatiale (Window), crée une entrée
        dans self.patches_dataset.

        Returns:
        - None: Construit self.patches_dataset (pd.DataFrame) avec les colonnes :
            - 'patch' (str): identifiant unique "patches_{mgrs25}_window_{x}_{y}_{w}_{h}"
            - 'window' (list): [col_off, row_off, width, height], coordonnées du patch
            - 'mgrs' (str): identifiant de la tuile MGRS (ex: "31TDH")
            - 'mgrs25' (str): identifiant de la sous-tuile MGRS25 (ex: "31TDH_1")
            - 'files' (list[str]): chemins [S2, S1_ASC, S1_DESC]
            - 'dates_S2' (list[str]): dates S2 au format YYYYMMDD
            - 'dates_S1_ASC' (list[str]): dates S1 orbite ascendante
            - 'dates_S1_DESC' (list[str]): dates S1 orbite descendante
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
        Scanne les répertoires de données pour identifier toutes les zones MGRS/MGRS25.
        Parcourt la hiérarchie data_optique/<MGRS>/<MGRS25-*>/ et appelle
        process_mgrs25_zone() pour chaque sous-répertoire.

        Returns:
        - None: Construit self.zones_dataset (pd.DataFrame) et self.dates_dict (dict).
            self.zones_dataset contient les colonnes :
            - 'mgrs' (str): identifiant de la tuile MGRS
            - 'mgrs25' (str): identifiant de la sous-tuile MGRS25
            - 'files' (list[str]): chemins absolus [S2.tif, S1_ASC.tif, S1_DESC.tif]
            - 'windows' (list[Window]): liste des fenêtres spatiales découpées
            - 'dates_S2', 'dates_S1_ASC', 'dates_S1_DESC' (list[str]): dates par capteur
        """
        self.zones_dataset = pd.DataFrame(columns=["mgrs", "mgrs25", "files", "windows"]).astype(object)
        self.dates_dict = {}

        for mgrs in tqdm(list(self.data_optique.iterdir()), leave=False, desc="mgrs"):
            for mgrs25 in tqdm(list(mgrs.iterdir()), leave=False, desc="mgrs25"):
                self.process_mgrs25_zone(mgrs, mgrs25)

    def process_mgrs25_zone(self, mgrs: Path, mgrs25: Path) -> None:
        """
        Traite une zone MGRS25 : lit les fichiers TIF S2 et S1, découpe en fenêtres spatiales.

        Parameters:
        - mgrs (Path): Chemin vers le répertoire de la tuile MGRS (ex: /data/S2/31TDH/).
            Le stem du Path donne l'identifiant MGRS (ex: "31TDH").
        - mgrs25 (Path): Chemin vers le sous-répertoire MGRS25
            (ex: /data/S2/31TDH/MGRS25-31TDH_1/). Le stem donne "MGRS25-31TDH_1",
            d'où on extrait "31TDH_1".

        Returns:
        - None: Ajoute une ligne à self.zones_dataset et alimente self.dates_dict
            pour la zone mgrs25_name avec les dates S2 et S1 (ASC/DESC).
        """
        mgrs_name = mgrs.stem
        mgrs25_name = mgrs25.stem[7:]
        self.dates_dict[mgrs25_name] = {}

        # Données Sentinel-2
        list_tifs_optique = sorted(mgrs25.rglob("*.tif"))
        list_jsons_optique = sorted(mgrs25.rglob("*.json"))
        dates_S2 = json.load(open(list_jsons_optique[0]))
        self.dates_dict[mgrs25_name]["S2"] = dates_S2

        # Données Sentinel-1
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
            "mgrs25": mgrs25_name,
            "files": [sorted(f.as_posix() for f in tif_files.values())],
            "windows": [list_windows],
            "dates_S2": [dates_S2],
            "dates_S1_ASC": [self.dates_dict[mgrs25_name]["S1"]["ASC"]],
            "dates_S1_DESC": [self.dates_dict[mgrs25_name]["S1"]["DESC"]],
        }
        df_temp = pd.DataFrame(data).astype(object)
        self.zones_dataset = pd.concat([self.zones_dataset, df_temp], ignore_index=True)

    def export_dataset(self, outpath: str | Path = "datasetCIRCAUnCRtainTS.csv") -> None:
        """
        Exporte le dataset vers un fichier CSV pour réutilisation rapide.

        Parameters:
        - outpath (str | Path, optional): Chemin du fichier CSV de sortie.
            Sera écrasé s'il existe déjà. Defaults to "datasetCIRCAUnCRtainTS.csv".

        Returns:
        - None: Écrit le fichier CSV contenant self.patches_dataset.
            Les colonnes complexes (listes) sont sérialisées en chaînes via pandas.
        """
        self.patches_dataset.to_csv(outpath, index=False)

    def get_patches_per_mgrs(self, mgrs: str) -> pd.DataFrame:
        """
        Retourne tous les patches appartenant à une tuile MGRS donnée.

        Parameters:
        - mgrs (str): Identifiant de la tuile MGRS (ex: "31TDH").

        Returns:
        - pd.DataFrame: Sous-ensemble de self.patches_dataset filtré sur la colonne 'mgrs'.
            Mêmes colonnes que self.patches_dataset (patch, window, mgrs, mgrs25, files, dates_*).
        """
        return self.patches_dataset[self.patches_dataset["mgrs"] == mgrs]

    def get_patches_per_mgrs25(self, mgrs25: str) -> pd.DataFrame:
        """
        Retourne tous les patches appartenant à une sous-tuile MGRS25 donnée.

        Parameters:
        - mgrs25 (str): Identifiant de la sous-tuile MGRS25 (ex: "31TDH_1").

        Returns:
        - pd.DataFrame: Sous-ensemble de self.patches_dataset filtré sur la colonne 'mgrs25'.
            Mêmes colonnes que self.patches_dataset.
        """
        return self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]

    def get_patch_item(self, patch_name: str) -> int | None:
        """
        Retourne l'index d'un patch à partir de son nom.

        Parameters:
        - patch_name (str): Identifiant complet du patch
            (ex: "patches_31TDH_1_window_0_0_256_256").

        Returns:
        - int | None: Index du patch dans self.patches_dataset si trouvé, sinon None.
        """
        matching_patches = self.patches_dataset[self.patches_dataset["patch"] == patch_name]
        if not matching_patches.empty:
            return matching_patches.index.values[0]
        return None

    def get_random_patch_by_mgrs(self, mgrs: str) -> dict[str, np.ndarray | str | list]:
        """
        Retourne un patch aléatoire pour une tuile MGRS donnée.

        Parameters:
        - mgrs (str): Identifiant de la tuile MGRS (ex: "31TDH").

        Returns:
        - dict[str, np.ndarray | str | list]: Dictionnaire retourné par __getitem__
            contenant les clés :
            - 'name' (str): identifiant du patch
            - 'data_S2' (np.ndarray): shape (T_valid, H, W, C_s2), données optiques S2
            - 'dates_S2' (list[str]): dates S2 au format YYYYMMDD
            - 'masks' (np.ndarray): masques nuages
            - 'data_S1' (np.ndarray): shape (T_valid, H, W, C_s1), données SAR S1 (si use_sar=True)
            - 'dates_S1' (list[str]): dates S1 correspondantes (si use_sar=True)
        """
        df_mgrs = self.get_patches_per_mgrs(mgrs=mgrs)
        return self.__getitem__(np.random.choice(df_mgrs.index.values))

    def enrich_patches_df_with_transforms(self):
        """
        Enrichit la DataFrame des patches avec les informations de géoréférencement.
        Pour chaque patch, lit le fichier TIF S2 avec rasterio et extrait les
        métadonnées de géotransformation (CRS, transform, bounds) pour la fenêtre
        spatiale correspondante.

        Returns:
        - None: Ajoute la colonne 'meta' à self.patches_dataset. Chaque entrée est un
            dictionnaire contenant les informations de géoréférencement de la fenêtre
            (CRS, transform affine, bounds, etc.).

        Raises:
        - ValueError: Si self.patches_dataset n'est pas initialisé.
        """
        if self.patches_dataset is None:
            raise ValueError("Le dataset n'est pas initialisé. Lancer setup() d'abord.")

        self.patches_dataset["meta"] = None
        for idx, row in tqdm(
            self.patches_dataset.iterrows(), total=len(self.patches_dataset), desc="Enrichissement des patches"
        ):
            meta = SentinelDataProcessor.get_window_info(row.files[0], Window(*row.window))
            self.patches_dataset.at[idx, "meta"] = meta

    def __getitem__(self, item: int) -> dict[str, np.ndarray | str | list[str]]:
        """
        Retourne un échantillon du dataset (patch S2 + S1 optionnel) pour un index donné.
        Lit le raster S2 et optionnellement S1 pour la fenêtre spatiale correspondante,
        filtre les dates nuageuses (sauf si no_filter=True), et apparie les dates S1
        aux dates S2 valides.

        Parameters:
        - item (int): Index du patch dans self.patches_dataset (0-indexed).

        Returns:
        - dict: Dictionnaire contenant les clés suivantes :
            - 'name' (str): Identifiant unique du patch
                (ex: "patches_31TDH_1_window_0_0_256_256").
            - 'data_S2' (np.ndarray): shape (T_valid, H, W, C_s2), données optiques
                Sentinel-2 filtrées (uniquement dates cloud-free).
                C_s2 = 10 bandes (B2-B8A, B11, B12) si no_filter=False,
                ou 10 bandes brutes si no_filter=True.
            - 'dates_S2' (list[str] | np.ndarray): Dates S2 au format YYYYMMDD
                correspondant aux frames dans data_S2.
                Type np.ndarray si no_filter=True, list[str] sinon.
            - 'masks' (np.ndarray): Masques nuages binaires (1 = nuage, 0 = clair).
                Si no_filter=True : shape (T_total, H, W), channel 10 du raster S2.
                Sinon : masques extraits par extract_and_transform_S2.
            - 'data_S1' (np.ndarray): shape (T_valid, H, W, C_s1), uniquement si
                use_sar=True. Données radar Sentinel-1 (VV, VH) appariées aux dates
                S2 valides. C_s1 = 2 canaux (VV, VH).
            - 'dates_S1' (list[str]): uniquement si use_sar=True. Dates S1 au format
                YYYYMMDD sélectionnées via appariement temporel (date S1 la plus proche
                de chaque date S2).
        """
        patch_data = self.patches_dataset.iloc[item]
        patch_window = Window(*patch_data.window)

        patch_S2_array = SentinelDataProcessor.read_MS(patch_data.files[0], patch_window)

        if self.no_filter:
            patch_S2_curated = patch_S2_array[:, :, :, 0:10]
            dates_S2_curated = np.asarray(self.dates_dict[patch_data.mgrs25]["S2"])
            cloud_masks = patch_S2_array[:, :, :, 10]
        else:
            (
                patch_S2_curated,
                dates_S2_curated,
                cloud_masks,
            ) = SentinelDataProcessor.extract_and_transform_S2(
                patch_S2_array,
                self.dates_dict[patch_data.mgrs25]["S2"],
            )

        sample = {
            "name": patch_data.patch,
            "data_S2": patch_S2_curated,
            "dates_S2": dates_S2_curated,
            "masks": cloud_masks,
        }

        if self.use_sar:
            (
                dates_S1_curated,
                index_S1_curated,
                orbit_type,
            ) = SentinelDataProcessor.get_pairedS1(
                dates_S2_curated,
                self.dates_dict[patch_data.mgrs25]["S1"]["ASC"],
                self.dates_dict[patch_data.mgrs25]["S1"]["DESC"],
            )

            path_S1 = patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
            patch_S1_array = SentinelDataProcessor.read_SAR(path_S1, patch_window)
            bands_S1 = [patch_S1_array[t_index] for t_index in index_S1_curated]
            patch_S1_curated = np.stack(bands_S1, axis=0)
            sample.update(
                {
                    "data_S1": patch_S1_curated,
                    "dates_S1": dates_S1_curated,
                }
            )
        return sample


# =============================================================================
# Section 2 : Appariement temporel S1 ↔ S2
# =============================================================================


def _get_datetime(date: str) -> datetime.datetime:
    """
    Convertit une date au format string YYYYMMDD en objet datetime.datetime.

    Parameters:
    - date (str): Date au format "YYYYMMDD" (ex: "20220901" pour le 1er septembre 2022).
        Doit contenir exactement 8 caractères numériques.

    Returns:
    - datetime.datetime: Objet datetime correspondant (ex: datetime(2022, 9, 1, 0, 0)).
    """
    return datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]))


def _convert_dates(dates: list) -> np.ndarray:
    """
    Convertit une liste de dates YYYYMMDD en jours relatifs depuis le 2022-09-01.
    La date de référence (epoch) est fixée au 1er septembre 2022.

    Parameters:
    - dates (list[str]): Liste de dates au format "YYYYMMDD"
        (ex: ["20220901", "20220911", "20221001"]).

    Returns:
    - np.ndarray: Array 1D d'entiers (shape: (N,)), où N = len(dates).
        Chaque entier = (date - 2022-09-01).days.
        Exemple : ["20220901", "20220911"] → array([0, 10]).
    """
    d0 = datetime.datetime(2022, 9, 1)
    return np.array([(_get_datetime(date) - d0).days for date in dates])


def appariement_S1_to_S2(
    S2_dates: list,
    S1_dates_asc: list,
    S1_dates_desc: list,
) -> tuple:
    """
    Appariement temporel des dates Sentinel-1 (ASC et DESC) vers les dates Sentinel-2.
    Pour chaque date S2, on trouve la date S1 ASC et DESC la plus proche temporellement,
    ainsi que la date S1 la plus proche toutes orbites confondues.

    Parameters:
    - S2_dates (list[str]): Liste des dates Sentinel-2 au format "YYYYMMDD".
        Exemple : ["20220901", "20220911", "20221001"].
        Correspond aux dates où des images optiques sont disponibles.
    - S1_dates_asc (list[str]): Liste des dates Sentinel-1 en orbite ascendante (ASC)
        au format "YYYYMMDD". Exemple : ["20220902", "20220914", "20220926"].
        Peut contenir plus ou moins de dates que S2_dates.
    - S1_dates_desc (list[str]): Liste des dates Sentinel-1 en orbite descendante (DESC)
        au format "YYYYMMDD". Exemple : ["20220908", "20220920", "20221002"].
        Peut contenir plus ou moins de dates que S2_dates.

    Returns:
    - tuple: Tuple de 5 éléments :
        - list_index_prelevement_asc (list[int]): Indices des images S1 ASC à extraire
            du raster S1_ASC. Correspondent aux positions dans S1_dates_asc.
            Exemple : [0, 2, 4] → extraire les images aux positions 0, 2, 4.
        - list_index_prelevement_desc (list[int]): Indices des images S1 DESC à extraire
            du raster S1_DESC. Même logique que list_index_prelevement_asc.
        - dates_S1_ASC_collected (list[str]): Dates S1 ASC effectivement sélectionnées
            (sous-ensemble de S1_dates_asc).
            Exemple : ["20220902", "20220926"].
        - dates_S1_DESC_collected (list[str]): Dates S1 DESC effectivement sélectionnées
            (sous-ensemble de S1_dates_desc).
        - dict_appariement (dict[str, dict]): Dictionnaire d'appariement complet :
            - dict_appariement["asc"] (dict[str, tuple]):
                Clé = date S2, Valeur = (date_S1_ASC_proche, "ASC", index_dans_ASC_collected)
            - dict_appariement["desc"] (dict[str, tuple]):
                Clé = date S2, Valeur = (date_S1_DESC_proche, "DESC", index_dans_DESC_collected)
            - dict_appariement["mix_closest"] (dict[str, tuple]):
                Clé = date S2, Valeur = (date_S1_proche, orbit_type, index)
                Date S1 la plus proche toutes orbites confondues.
    """
    S1_dates_asc_int = _convert_dates(S1_dates_asc)
    S1_dates_desc_int = _convert_dates(S1_dates_desc)
    S2_dates_int = _convert_dates(S2_dates)
    S1_dates_int = np.concatenate([S1_dates_asc_int, S1_dates_desc_int])
    indices = np.argsort(S1_dates_int)
    S1_dates_int = S1_dates_int[indices]
    S1_dates_tmp = S1_dates_asc + S1_dates_desc
    S1_dates = np.array([S1_dates_tmp[indices[i]] for i in range(indices.shape[0])])

    # Appariement ASC : pour chaque date S2, trouver la date S1 ASC la plus proche
    dict_appariement_S2_S1_asc = {}
    list_index_prelevement_asc = []
    distance_S2_S1_asc = np.abs(np.expand_dims(S2_dates_int, 1) - np.expand_dims(S1_dates_asc_int, 0))
    id_min_S2_S1_asc = np.argmin(distance_S2_S1_asc, axis=1)

    for i in range(len(S2_dates)):
        date_min = S1_dates_asc[id_min_S2_S1_asc[i]]
        orbit_type = "ASC"
        index_prelevement_asc = S1_dates_asc.index(date_min)
        if index_prelevement_asc not in list_index_prelevement_asc:
            list_index_prelevement_asc.append(index_prelevement_asc)
        index_in_new_data_asc = list_index_prelevement_asc.index(index_prelevement_asc)
        dict_appariement_S2_S1_asc[S2_dates[i]] = (date_min, orbit_type, index_in_new_data_asc)

    # Appariement DESC : idem pour les orbites descendantes
    dict_appariement_S2_S1_desc = {}
    list_index_prelevement_desc = []
    distance_S2_S1_desc = np.abs(np.expand_dims(S2_dates_int, 1) - np.expand_dims(S1_dates_desc_int, 0))
    id_min_S2_S1_desc = np.argmin(distance_S2_S1_desc, axis=1)
    for i in range(len(S2_dates)):
        date_min = S1_dates_desc[id_min_S2_S1_desc[i]]
        orbit_type = "DESC"
        index_prelevement_desc = S1_dates_desc.index(date_min)
        if index_prelevement_desc not in list_index_prelevement_desc:
            list_index_prelevement_desc.append(index_prelevement_desc)
        index_in_new_data_desc = list_index_prelevement_desc.index(index_prelevement_desc)
        dict_appariement_S2_S1_desc[S2_dates[i]] = (date_min, orbit_type, index_in_new_data_desc)

    dates_S1_ASC_collected = np.asarray(S1_dates_asc)[list_index_prelevement_asc].tolist()
    dates_S1_DESC_collected = np.asarray(S1_dates_desc)[list_index_prelevement_desc].tolist()

    # Appariement mix : pour chaque date S2, trouver la date S1 la plus proche (ASC ou DESC)
    distance_S2_S1_all = np.abs(np.expand_dims(S2_dates_int, 1) - np.expand_dims(S1_dates_int, 0))
    id_min_S2_S1_all = np.argmin(distance_S2_S1_all, axis=1)
    dict_appariement_S2_S1_all = {}
    for i in range(len(S2_dates)):
        date_min = S1_dates[id_min_S2_S1_all[i]]
        if date_min in S1_dates_asc:
            orbit_type = "ASC"
            index = dates_S1_ASC_collected.index(date_min)
        else:
            orbit_type = "DESC"
            index = dates_S1_DESC_collected.index(date_min)
        dict_appariement_S2_S1_all[S2_dates[i]] = (date_min, orbit_type, index)

    dict_appariement = {
        "asc": dict_appariement_S2_S1_asc,
        "desc": dict_appariement_S2_S1_desc,
        "mix_closest": dict_appariement_S2_S1_all,
    }

    assert len(list_index_prelevement_asc) == len(dates_S1_ASC_collected)
    assert len(list_index_prelevement_desc) == len(dates_S1_DESC_collected)

    return (
        list_index_prelevement_asc,
        list_index_prelevement_desc,
        dates_S1_ASC_collected,
        dates_S1_DESC_collected,
        dict_appariement,
    )


# =============================================================================
# Section 3 : Écriture HDF5 (HDF5Maker)
# =============================================================================


class HDF5Maker(FileScanner):
    """
    Classe qui exporte les données CIRCA dans des fichiers HDF5.

    Hérite de FileScanner pour le scan des données, puis écrit les patches
    dans des fichiers HDF5 organisés par zone MGRS.

    Architecture HDF5 :
        MGRS_id/
            MGRS25_id/
                window_x_y_w_h/
                    S1/  (S1_asc, S1_desc, dates, appariement)
                    S2/  (bandes, dates, cloud_mask, cloud_prob)
                    valid_obs  (masque binaire pour train/val, indices pour test)
                    idx_good_frames, idx_cloudy_frames, idx_impaired_frames

    Différences train/val vs test :
        - Train/Val : S2 contient uniquement les frames cloud-free, valid_obs = vecteur de 1
        - Test : S2 contient TOUTES les frames, valid_obs = indices des frames cloud-free
    """

    def __init__(
        self,
        data_optique: str | Path = None,
        data_radar: str | Path = None,
        image_size: int = (256, 256),
        hdf5_folder: str | Path | None = None,
        overlap: int | None = 0,
        load_dataset: str | None = None,
        shuffle: bool = False,
        use_sar: bool = True,
        min_seq_length: int | None = MIN_SEQ_LENGTH,
        max_seq_length: int | None = None,
        render_occluded_above_p: float | None = None,
        mask_kwargs: dict | DictConfig | None = None,
        pe_strategy: str = "day-within-sequence",
        channels: str | None = "all",
        # Rétro-compatibilité : filter_settings est accepté mais ignoré
        filter_settings: dict = None,
        folder_test: Path = Path("/mnt/stores/store_dai/projets/pac/3str/EXP_2/Data_Raster/test_v3"),
    ):
        """
        Initialise le HDF5Maker pour exporter les données CIRCA au format HDF5.

        Parameters:
        - data_optique (str | Path, optional): Chemin vers le répertoire racine des
            données optiques Sentinel-2. Structure : data_optique/<MGRS>/<MGRS25-*>/*.tif.
            Si None et load_dataset est fourni, le scan est sauté. Defaults to None.
        - data_radar (str | Path, optional): Chemin vers le répertoire racine des données
            radar Sentinel-1. Structure : data_radar/<MGRS>/<MGRS25-*>/*_ASC.tif, *_DESC.tif.
            Defaults to None.
        - image_size (int | tuple, optional): Taille des patches spatiaux
            (hauteur, largeur) en pixels. Defaults to (256, 256).
        - hdf5_folder (str | Path | None, optional): Répertoire de sortie pour les fichiers
            HDF5 créés. Un fichier HDF5 par tuile MGRS sera créé dans ce dossier.
            Defaults to None.
        - overlap (int | None, optional): Chevauchement en pixels entre patches adjacents.
            Defaults to 0.
        - load_dataset (str | None, optional): Chemin vers un CSV pré-exporté pour éviter
            le scan des répertoires. Defaults to None.
        - shuffle (bool, optional): Si True, mélange l'ordre des patches. Defaults to False.
        - use_sar (bool, optional): Si True, inclut les données Sentinel-1 dans l'export HDF5.
            Defaults to True.
        - min_seq_length (int | None, optional): Nombre minimum de dates valides (non nuageuses)
            requis par patch. Les patches avec moins de dates valides sont ignorés.
            Defaults to MIN_SEQ_LENGTH (5).
        - max_seq_length (int | None, optional): Nombre maximum de dates par séquence.
            Si None, utilise MAX_SEQ_LENGTH (30). Defaults to None.
        - render_occluded_above_p (float | None, optional): Seuil de probabilité au-dessus
            duquel une frame est considérée comme masquée/occluse. Defaults to None.
        - mask_kwargs (dict | DictConfig | None, optional): Configuration du masquage
            synthétique pour l'entraînement. Clés possibles :
            - 'mask_type' (str): type de masque (ex: "random_clouds")
            - 'ratio_masked_frames' (float): proportion de frames à masquer [0, 1]
            - 'ratio_fully_masked_frames' (float): proportion de frames entièrement masquées
            - 'fixed_masking_ratio' (bool): si True le ratio est fixe
            - 'non_masked_frames' (list[int]): indices des frames à ne jamais masquer
            - 'intersect_real_cloud_masks' (bool): intersecter avec les vrais masques nuages
            - 'dilate_cloud_masks' (bool): dilater les masques nuages existants
            - 'fill_type' (str): type de remplissage ("fill_value" ou autre)
            - 'fill_value' (int): valeur de remplissage pour les pixels masqués
            Si None, les valeurs par défaut sont utilisées. Defaults to None.
        - pe_strategy (str, optional): Stratégie d'encodage positionnel temporel.
            - "day-within-sequence" : jour relatif dans la séquence
            - "day-since-beginning" : jours écoulés depuis le début de l'année
            Defaults to "day-within-sequence".
        - channels (str | None, optional): Canaux S2 à exporter :
            - "all" : 10 bandes (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
            - "bgr-nir" : 4 bandes (B2, B3, B4, B8)
            Defaults to "all".
        - filter_settings (dict, optional): DEPRÉCIÉ — Ancien paramètre de filtrage,
            ignoré avec un avertissement. Utiliser mask_kwargs à la place. Defaults to None.
        - folder_test (Path, optional): Chemin vers le répertoire des données de test
            contenant les masques nuages synthétiques (dossiers "aleatoire" et "consecutif").
            Uniquement utilisé pour le split test.
            Defaults to Path("/mnt/stores/store_dai/projets/pac/3str/EXP_2/Data_Raster/test_v3").
        """
        self.rng = np.random.default_rng(seed=SEED)

        if filter_settings is not None:
            import warnings
            warnings.warn(
                "HDF5DatasetCreator: 'filter_settings' est déprécié et ignoré. "
                "Le filtrage des dates nuageuses est effectué directement via "
                "SentinelDataProcessor.filter_dates() lors de la création du HDF5. "
                "Les champs type, min_length, return_valid_obs_only, etc. n'ont pas d'effet ici.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            image_size=image_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_sar=use_sar,
        )

        self.hdf5_folder = Path(hdf5_folder) if isinstance(hdf5_folder, str) else hdf5_folder
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length = MAX_SEQ_LENGTH if max_seq_length is None else max_seq_length
        self.render_occluded_above_p = render_occluded_above_p
        self.pe_strategy = pe_strategy
        _, self.c_index_rgb, self.c_index_nir, self.s2_channels = self._setup_channels(channels)
        self.folder_test = folder_test
        (
            self.mask_kwargs,
            self.fill_type,
            self.fill_value,
            self.fixed_masking_ratio,
            self.intersect_real_cloud_masks,
            self.dilate_cloud_masks,
        ) = self._setup_mask_kwargs(mask_kwargs)

    def _setup_channels(self, channels: str):
        """
        Configure les canaux S2 à utiliser selon le mode sélectionné.

        Parameters:
        - channels (str): Mode de sélection des canaux :
            - "all" : utilise les 10 bandes S2 (B2-B8A, B11, B12)
            - "bgr-nir" : utilise uniquement 4 bandes (B2 bleu, B3 vert, B4 rouge, B8 NIR)

        Returns:
        - tuple: Tuple de 4 éléments :
            - num_channels (int): Nombre total de canaux (S2 + S1 si use_sar).
                Exemples : 14 (10 S2 + 4 S1) pour "all" avec SAR, 10 (10 S2) sans SAR,
                8 (4 S2 + 4 S1) pour "bgr-nir" avec SAR.
            - c_index_rgb (torch.Tensor): shape (3,), indices des canaux RGB [R,G,B] = [2,1,0].
            - c_index_nir (torch.Tensor): shape (1,), indice du canal NIR (= 6, correspondant à B8).
            - s2_channels (list[int]): Indices des canaux S2 à conserver.
                "all" → [0,1,2,...,9], "bgr-nir" → [0,1,2,6].

        Raises:
        - ValueError: Si channels n'est ni "all" ni "bgr-nir".
        """
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
            raise ValueError(f"Canaux '{channels}' non reconnus. Utiliser 'all' ou 'bgr-nir'.")
        if self.use_sar:
            num_channels += 4
        return num_channels, c_index_rgb, c_index_nir, s2_channels

    def _setup_mask_kwargs(self, mask_kwargs: DictConfig | None = None):
        """
        Configure les paramètres de masquage synthétique avec valeurs par défaut.

        Parameters:
        - mask_kwargs (DictConfig | dict | None, optional): Configuration du masquage
            synthétique. Si dict, converti en OmegaConf DictConfig. Si None, les valeurs
            par défaut sont utilisées :
            - mask_type : "random_clouds"
            - ratio_masked_frames : 0.5
            - ratio_fully_masked_frames : 0.0
            - fixed_masking_ratio : False
            - non_masked_frames : []
            - intersect_real_cloud_masks : False
            - dilate_cloud_masks : False
            - fill_type : "fill_value"
            - fill_value : 1
            Defaults to None.

        Returns:
        - tuple: Tuple de 6 éléments :
            - mask_kwargs (DictConfig): Configuration complète du masquage (OmegaConf DictConfig).
            - fill_type (str): Méthode de remplissage des pixels masqués ("fill_value").
            - fill_value (int): Valeur numérique de remplissage (1 = pixels masqués mis à 1).
            - fixed_masking_ratio (bool): Si True, le ratio de masquage est fixe.
            - intersect_real_cloud_masks (bool): Si True, intersecte avec les vrais masques nuages.
            - dilate_cloud_masks (bool): Si True, dilate les masques nuages existants.
        """
        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)

        if mask_kwargs is None:
            mask_kwargs = OmegaConf.create(
                {
                    "mask_type": "random_clouds",
                    "ratio_masked_frames": 0.5,
                    "ratio_fully_masked_frames": 0.0,
                    "fixed_masking_ratio": False,
                    "non_masked_frames": [0],
                    "intersect_real_cloud_masks": False,
                    "dilate_cloud_masks": False,
                    "fill_type": "fill_value",
                    "fill_value": 1,
                }
            )
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
        Charge les données CIRCA et les écrit dans des fichiers HDF5 par zone MGRS.

        Pipeline de traitement pour chaque zone MGRS25 :
            1. Lecture des probabilités de nuages et neige depuis les rasters S2
            2. Correction des masques nuages (train/val uniquement)
            3. Filtrage des dates par patch : identification des frames valides (cloud-free)
            4. Suppression des patches avec trop peu de dates valides (< min_seq_length)
            5. Lecture des données S2 et S1 pour la zone entière
            6. Appariement temporel S1 ↔ S2 via appariement_S1_to_S2()
            7. Pour chaque patch : extraction, découpage par fenêtre, écriture HDF5

        Différences selon le split :
            - Train/Val : seules les dates cloud-free sont stockées.
              valid_obs = vecteur binaire de 1 (toutes les frames stockées sont valides).
            - Test : toutes les dates sont stockées avec masques nuages synthétiques.
              valid_obs = indices des frames cloud-free dans la série complète.

        Returns:
        - None: Crée des fichiers HDF5 dans self.hdf5_folder, un par tuile MGRS.

        Structure HDF5 créée pour chaque patch (window_group) :
            MGRS_id/
                MGRS25_id/
                    window_{x}_{y}_{w}_{h}/
                        S1/
                            S1_asc (np.ndarray): shape (T_s1, C_s1, H, W), données S1 ASC.
                                T_s1 = nombre de dates S1 ASC collectées, C_s1 = 2 (VV, VH).
                            S1_desc (np.ndarray): shape (T_s1, C_s1, H, W), données S1 DESC.
                            S1_dates_asc (list[str]): dates S1 ASC au format YYYYMMDD.
                            S1_dates_desc (list[str]): dates S1 DESC au format YYYYMMDD.
                            S2_S1_pairing (str): JSON, dictionnaire d'appariement S2↔S1.
                        S2/
                            S2 (np.ndarray): shape (T_s2, C_s2, H, W), bandes S2.
                                Train/Val : T_s2 = nombre de dates cloud-free uniquement.
                                Test : T_s2 = nombre total de dates.
                                C_s2 = 10 ("all") ou 4 ("bgr-nir").
                            S2_dates (list[str]): dates S2 au format YYYYMMDD.
                            cloud_mask (np.ndarray): shape (T, H, W), masques nuages binaires (0/1).
                            cloud_prob (np.ndarray): shape (T, H, W), probabilités nuages (float32).
                        valid_obs (np.ndarray):
                            Train/Val : shape (N_valid,), vecteur de 1 ([1,1,...,1]).
                            Test : shape (N_valid,), indices des frames cloud-free (ex: [0,2,5,7]).
                        idx_good_frames (np.ndarray): shape (N_valid,),
                            indices des frames non nuageuses dans la série temporelle complète.
                        idx_cloudy_frames (np.ndarray): shape (N_cloudy,),
                            indices des frames nuageuses dans la série temporelle complète.
                        idx_impaired_frames (np.ndarray): shape (N_cloudy,),
                            identique à idx_cloudy_frames (redondance historique).
                        [Test uniquement :]
                        idx_syn_aleatoire (np.ndarray): indices des frames avec masque synthétique
                            aléatoire.
                        idx_syn_consecutif (np.ndarray): indices des frames avec masque synthétique
                            consécutif.
        """
        for mgrs_id in tqdm(self.patches_dataset["mgrs"].unique(), desc="MGRS IDs"):
            mgrs_dataset = self.patches_dataset[self.patches_dataset["mgrs"] == mgrs_id]
            hdf5_file = self.hdf5_folder / f"{mgrs_id}.hdf5"
            if not hdf5_file.exists():
                print(f"Création d'un nouveau fichier HDF5 : {hdf5_file}")
            for mgrs25_id in tqdm(mgrs_dataset["mgrs25"].unique(), desc="MGRS25 IDs"):
                print(f"Traitement de la zone MGRS25 : {mgrs25_id}")
                with h5py.File(hdf5_file, "a") as hf:
                    if mgrs_id in hf and mgrs25_id in hf[mgrs_id]:
                        print(f"La zone {mgrs25_id} existe déjà dans {hdf5_file}. Passage...")
                        continue
                    elif mgrs_id not in hf:
                        print(f"Création du groupe MGRS {mgrs_id} dans {hdf5_file}...")
                        mgrs_group = hf.create_group(mgrs_id)
                    else:
                        print(f"Réutilisation du groupe MGRS existant pour {mgrs_id}...")
                        mgrs_group = hf[mgrs_id]

                    print(f"Chargement des données MGRS25 {mgrs25_id} dans {hdf5_file}...")
                    mgrs25_group = mgrs_group.create_group(mgrs25_id)
                    in_test_set = mgrs25_id in GEOGRAPHIC_SPLITS["test"]

                    mgrs25_dataset = mgrs_dataset[mgrs_dataset["mgrs25"] == mgrs25_id]

                    # --- ETL des données S2 pour la zone MGRS25 ---
                    # 1. Chargement des masques nuages/neige
                    mgrs25_files = mgrs25_dataset.iloc[0].files
                    dates_s2 = self.dates_dict[mgrs25_id]["S2"]
                    dates_s1_asc = self.dates_dict[mgrs25_id]["S1"]["ASC"]
                    dates_s1_desc = self.dates_dict[mgrs25_id]["S1"]["DESC"]

                    # 2. Lecture des probabilités de nuages et de neige
                    cloud_probs = SentinelDataProcessor.read_mask_prob(
                        path_raster=mgrs25_files[0], type_mask="cloud"
                    )  # H x W x 1 x T
                    snow_probs = SentinelDataProcessor.read_mask_prob(
                        path_raster=mgrs25_files[0], type_mask="snow"
                    )  # H x W x 1 x T

                    # 3. Transposition en (T, H, W) et correction des masques
                    cloud_probs = cloud_probs.squeeze(axis=2).transpose((2, 0, 1))
                    snow_probs = snow_probs.squeeze(axis=2).transpose((2, 0, 1))

                    if not in_test_set:
                        # Pour train/val : correction du masque nuage
                        cloud_probs = SentinelDataProcessor.cloud_mask_correction(cloud_probs)
                    else:
                        # Pour test : lecture des masques nuages synthétiques (aléatoire et consécutif)
                        # folder_test = Path("/mnt/stores/store_dai/projets/pac/3str/EXP_2/Data_Raster/test_v3")
                        folder_aleatoire = self.folder_test / "aleatoire"
                        folder_consecutif = self.folder_test / "consecutif"

                        assert folder_aleatoire.exists(), f"Dossier aléatoire {folder_aleatoire} introuvable."
                        assert folder_consecutif.exists(), f"Dossier consécutif {folder_consecutif} introuvable."

                        path_file_aleatoire = (
                            folder_aleatoire / mgrs_id / ("MGRS25-" + mgrs25_id) / Path(mgrs25_files[0]).name
                        )
                        path_file_consecutif = (
                            folder_consecutif / mgrs_id / ("MGRS25-" + mgrs25_id) / Path(mgrs25_files[0]).name
                        )

                        cloud_probs_aleatoire = SentinelDataProcessor.read_mask_prob(
                            path_raster=path_file_aleatoire, type_mask="cloud"
                        )
                        cloud_probs_consecutif = SentinelDataProcessor.read_mask_prob(
                            path_raster=path_file_consecutif, type_mask="cloud"
                        )

                        cloud_probs_aleatoire = cloud_probs_aleatoire.squeeze(axis=2).transpose((2, 0, 1))
                        cloud_probs_consecutif = cloud_probs_consecutif.squeeze(axis=2).transpose((2, 0, 1))

                    # --- Filtrage des dates par fenêtre ---
                    index_to_drop: list = []
                    mgrs25_data: dict = {}
                    for row_index, row in tqdm(
                        mgrs25_dataset.iterrows(), total=mgrs25_dataset.shape[0], desc="Filtrage fenêtres"
                    ):
                        window = row.window
                        x, y, width, height = window[0], window[1], window[2], window[3]

                        # 4. Filtrage des masques nuage et neige par fenêtre
                        snow_probs_window = snow_probs[:, x : x + width, y : y + height]
                        cloud_probs_window = cloud_probs[:, x : x + width, y : y + height]

                        idx_good_frames = SentinelDataProcessor.filter_dates(
                            np.stack([snow_probs_window, cloud_probs_window], axis=-1)
                        )
                        idx_cloudy_frames = np.asarray([d for d in range(len(dates_s2)) if d not in idx_good_frames])
                        dates_s2_valid = [dates_s2[t] for t in idx_good_frames]

                        # 5. Supprimer les patches avec trop peu de dates valides
                        if self.min_seq_length is not None and len(dates_s2_valid) < self.min_seq_length:
                            index_to_drop.append(row_index)
                        else:
                            mgrs25_data[row_index] = {
                                "idx_good_frames": idx_good_frames.tolist(),
                                "idx_cloudy_frames": idx_cloudy_frames.tolist(),
                                "dates_s2_valid": dates_s2_valid,
                            }

                    # Mise à jour de la DataFrame avec les résultats du filtrage
                    mgrs25_dataset = mgrs25_dataset.drop(index=index_to_drop)
                    for label in ["idx_cloudy_frames", "idx_good_frames", "dates_s2_valid"]:
                        mgrs25_dataset[label] = [v[label] for v in mgrs25_data.values()]

                    # --- Lecture des données S2 et S1 pour la zone entière ---
                    mgrs25_s2 = SentinelDataProcessor.read_raster_per_dates(
                        path_raster=mgrs25_files[0], type_bands="s2"
                    )  # T x C x H x W
                    mgrs25_s2 = mgrs25_s2[:, self.s2_channels, :, :]

                    # Appariement S1 ↔ S2
                    (
                        list_index_prelevement_asc,
                        list_index_prelevement_desc,
                        dates_s1_asc_collected,
                        dates_s1_desc_collected,
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
                    )
                    mgrs25_s1_desc = SentinelDataProcessor.read_raster_per_dates(
                        path_raster=path_s1_desc,
                        indexes_dates=list_index_prelevement_desc,
                        type_bands="s1",
                    )

                    # --- Écriture des données par fenêtre dans le HDF5 ---
                    for row_index, row in tqdm(
                        mgrs25_dataset.iterrows(), total=mgrs25_dataset.shape[0], desc="Écriture HDF5"
                    ):
                        window = row.window
                        x, y, width, height = window[0], window[1], window[2], window[3]
                        windows_str = "_".join(map(str, window))

                        cloud_probs_window = cloud_probs[:, x : x + width, y : y + height]
                        snow_probs_window = snow_probs[:, x : x + width, y : y + height]
                        cloud_masks_window = (cloud_probs_window != 0).astype(int)

                        idx_selected = np.asarray(mgrs25_dataset.loc[row_index, "idx_good_frames"])
                        s2_dates_non_cloudy = mgrs25_dataset.loc[row_index, "dates_s2_valid"]

                        if not in_test_set:
                            # --- Train/Val : ne stocker que les dates cloud-free ---
                            s2_dates = s2_dates_non_cloudy
                            s2 = mgrs25_s2[idx_selected, :, x : x + width, y : y + height]

                            # Collecte des dates S1 correspondantes aux dates S2 non nuageuses
                            s1_asc_non_cloudy_indexes, s1_desc_non_cloudy_indexes = [], []
                            for s2_date in s2_dates_non_cloudy:
                                s1_date_asc, _, i_asc = dict_appariement["asc"][s2_date]
                                assert i_asc == dates_s1_asc_collected.index(s1_date_asc)
                                s1_asc_non_cloudy_indexes.append(i_asc)

                                s1_date_desc, _, i_desc = dict_appariement["desc"][s2_date]
                                assert i_desc == dates_s1_desc_collected.index(s1_date_desc)
                                s1_desc_non_cloudy_indexes.append(i_desc)

                            s1_asc = mgrs25_s1_asc[s1_asc_non_cloudy_indexes, :, x : x + width, y : y + height]
                            s1_dates_asc = np.asarray(dates_s1_asc_collected)[s1_asc_non_cloudy_indexes].tolist()

                            s1_desc = mgrs25_s1_desc[s1_desc_non_cloudy_indexes, :, x : x + width, y : y + height]
                            s1_dates_desc = np.asarray(dates_s1_desc_collected)[s1_desc_non_cloudy_indexes].tolist()

                            # CORRECTION BUG valid_obs : vecteur binaire au lieu d'indices
                            # Pour train/val, toutes les frames stockées sont valides,
                            # donc valid_obs = [1, 1, ..., 1] de longueur N_valid.
                            # L'ancien code stockait idx_good_frames (ex: [0, 2, 4, 7, 9])
                            # ce qui causait un bug dans subsample_sequence() où .nonzero()
                            # excluait la frame d'indice 0 (valeur=0 traitée comme False).
                            valid_obs = np.ones(len(s2_dates_non_cloudy), dtype=int)
                        else:
                            # --- Test : stocker TOUTES les dates ---
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

                            # Pour test, valid_obs = indices des frames cloud-free
                            valid_obs = idx_selected

                        # Vérification des dimensions spatiales
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
                                "S1_asc": s1_asc,
                                "S1_desc": s1_desc,
                                "S1_dates_asc": s1_dates_asc,
                                "S1_dates_desc": s1_dates_desc,
                                "S2_S1_pairing": dict_appariement,
                            },
                            "S2": {
                                "S2": s2,
                                "S2_dates": s2_dates,
                                "cloud_mask": cloud_masks_window,
                                "cloud_prob": cloud_probs_window.astype(np.float32),
                            },
                            "idx_cloudy_frames": np.asarray(mgrs25_dataset.loc[row_index, "idx_cloudy_frames"]),
                            "idx_good_frames": np.asarray(mgrs25_dataset.loc[row_index, "idx_good_frames"]),
                            "idx_impaired_frames": np.asarray(mgrs25_dataset.loc[row_index, "idx_cloudy_frames"]),
                            "valid_obs": valid_obs,
                        }

                        if in_test_set:
                            sample.update(
                                {
                                    "idx_syn_aleatoire": index_syn_aleatoire,
                                    "idx_syn_consecutif": index_syn_consecutif,
                                }
                            )

                        # Écriture du sample dans le HDF5
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


# =============================================================================
# Section 4 : Fusion des fichiers HDF5
# =============================================================================


def merge_hdf5_files(source_dir: str, output_file: str) -> None:
    """
    Fusionne les fichiers HDF5 individuels (un par zone MGRS) en un seul fichier HDF5.
    Parcourt tous les fichiers .hdf5/.h5 du répertoire source, lit les groupes MGRS
    de premier niveau et copie récursivement les sous-groupes MGRS25 dans le fichier
    de sortie unique. Les doublons (même MGRS25 dans le même groupe MGRS) sont ignorés.

    Parameters:
    - source_dir (str): Chemin vers le répertoire contenant les fichiers HDF5 individuels
        à fusionner. Chaque fichier est typiquement nommé par son MGRS (ex: "31TDH.hdf5").
        Le répertoire doit exister.
    - output_file (str): Chemin vers le fichier HDF5 fusionné en sortie
        (ex: "/data/CIRCA_CR_merged.hdf5"). Le répertoire parent est créé automatiquement.
        Si le fichier existe déjà, il sera écrasé.

    Returns:
    - None: Écrit le fichier HDF5 fusionné à l'emplacement output_file. Structure créée :
        MGRS_id_1/
            MGRS25_id_1/
                window_x_y_w_h/ (données S1, S2, valid_obs, etc.)
            MGRS25_id_2/
                ...
        MGRS_id_2/
            ...
    """
    if not os.path.isdir(source_dir):
        print(f"Erreur : Le répertoire source n'existe pas : {source_dir}")
        return

    try:
        hdf5_files = sorted([f for f in os.listdir(source_dir) if f.endswith((".hdf5", ".h5"))])
    except FileNotFoundError:
        print(f"Erreur : Impossible d'accéder au répertoire : {source_dir}")
        return

    if not hdf5_files:
        print(f"Aucun fichier .hdf5 trouvé dans {source_dir}")
        return

    print(f"{len(hdf5_files)} fichiers HDF5 trouvés. Début de la fusion dans {output_file}...")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with h5py.File(output_file, "w") as h5_out:
        for filename in tqdm(hdf5_files, desc="Fusion des fichiers"):
            file_path = os.path.join(source_dir, filename)
            try:
                with h5py.File(file_path, "r") as h5_in:
                    for mgrs_id in h5_in.keys():
                        dest_mgrs_group = h5_out.require_group(mgrs_id)
                        source_mgrs_group = h5_in[mgrs_id]

                        for mgrsc_id in source_mgrs_group.keys():
                            if mgrsc_id in dest_mgrs_group:
                                print(
                                    f"Avertissement : La zone MGRSC '{mgrsc_id}' existe déjà "
                                    f"dans le groupe MGRS '{mgrs_id}'. Ignoré. (Fichier: {filename})"
                                )
                                continue
                            source_mgrs_group.copy(mgrsc_id, dest_mgrs_group)
            except Exception as e:
                print(f"\nErreur lors du traitement du fichier {filename}: {e}")

    print(f"\nFusion terminée. Le fichier de sortie est : {output_file}")


# =============================================================================
# Point d'entrée — Exemple d'utilisation
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Création et fusion de fichiers HDF5 CIRCA")
    parser.add_argument("--mode", choices=["create", "merge"], required=True, help="Mode d'exécution")
    parser.add_argument("--data-optique", type=str, help="Chemin vers les données optiques S2")
    parser.add_argument("--data-radar", type=str, help="Chemin vers les données radar S1 v4")
    parser.add_argument("--hdf5-folder", type=str, help="Répertoire de sortie des fichiers HDF5 individuels")
    parser.add_argument("--source-dir", type=str, help="Répertoire source pour la fusion")
    parser.add_argument("--output-file", type=str, help="Fichier HDF5 fusionné en sortie")
    args = parser.parse_args()

    if args.mode == "create":
        # Exemple : python hdf5_creator.py --mode create \
        #   --data-optique /path/to/optique_dataset \
        #   --data-radar /path/to/radar_dataset_v4 \
        #   --hdf5-folder /path/to/hdf5/archives_MGRSC

        mask_kwargs = {
            "mask_type": "random_clouds",
            "ratio_masked_frames": 0.5,
            "ratio_fully_masked_frames": 0.0,
            "fixed_masking_ratio": False,
            "non_masked_frames": [0],
            "intersect_real_cloud_masks": False,
            "dilate_cloud_masks": False,
            "fill_type": "fill_value",
            "fill_value": 1,
            "p_filter": 0.1,
        }

        dataset = HDF5Maker(
            hdf5_folder=Path(args.hdf5_folder),
            data_optique=Path(args.data_optique),
            data_radar=Path(args.data_radar),
            image_size=[256, 256],
            overlap=0,
            min_seq_length=10,
            mask_kwargs=mask_kwargs,
        )
        dataset.load_items_to_hdf5()

    elif args.mode == "merge":
        # Exemple : python hdf5_creator.py --mode merge \
        #   --source-dir /path/to/hdf5/archives_MGRSC \
        #   --output-file /path/to/CIRCA_CR_merged.hdf5
        merge_hdf5_files(args.source_dir, args.output_file)
