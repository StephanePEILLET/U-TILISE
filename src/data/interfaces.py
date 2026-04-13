"""
Interfaces et types communs pour le module de données.

Définit le contrat que TOUS les backends doivent respecter pour
fonctionner avec SentinelDataset.
"""

import datetime as dt
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor


PhaseType = Literal["train", "val", "test", "train+val", "all"]
"""Phase du dataset : train / validation / test / combinaison"""

ChannelType = Literal["all", "bgr-nir"]
"""Ensemble de bandes à charger : toutes les bandes ou seulement BGR + NIR"""

DateArray = np.ndarray[dt.date]
"""Array numpy contenant des objets datetime.date"""

SampleDict = dict[str, np.ndarray | dict[str, np.ndarray] | list[str]]
"""Format d'échantillon brut retourné par les backends"""

TensorDict = dict[str, torch.Tensor | dict[str, torch.Tensor]]
"""Format d'échantillon traité retourné par SentinelDataset"""


@runtime_checkable
class SentinelBackend(Protocol):
    """
    🔌 Interface commune pour TOUS les backends de données.
    
    Tout backend qui implémente cette interface fonctionnera **automatiquement**
    avec SentinelDataset sans aucune modification.
    
    📜 Règles d'implémentation d'un nouveau backend :
    1. ❌ Ne faites AUCUN traitement métier dans le backend
    2. ❌ Ne générez PAS de masques synthétiques
    3. ❌ Ne faites PAS d'échantillonnage temporel
    4. ✅ Retournez seulement les données brutes telles qu'elles sont stockées
    5. ✅ Tous les prétraitements doivent être dans SentinelDataset
    6. ✅ Respectez exactement le format de retour de __getitem__
    
    Backends existants :
    - ✅ `HDF5Backend` : lit depuis un fichier HDF5 pré-calculé
    - ✅ `FilesBackend` : lit directement depuis les fichiers TIF brutes
    """

    def __getitem__(self, item: int) -> SampleDict:
        """Retourne un échantillon brut du backend"""
        ...

    def __len__(self) -> int:
        """Retourne le nombre total d'échantillons"""
        ...

    @property
    def c_index_rgb(self) -> Tensor:
        """Indices des canaux RGB dans les tenseurs S2"""
        ...

    @property
    def c_index_nir(self) -> Tensor:
        """Indice du canal NIR dans les tenseurs S2"""
        ...

    @property
    def num_channels(self) -> int:
        """Nombre total de canaux dans le jeu de données"""
        ...

    @property
    def phase(self) -> PhaseType:
        """Phase du dataset (train/val/test)"""
        ...
