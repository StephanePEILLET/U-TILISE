"""
Module de données unifié pour U-TILISE.

Ce module encapsule toute la logique de chargement et de traitement des données
Sentinel, indépendamment de la source (HDF5 ou fichiers brutes).

Utilisation principale :
```python
# Depuis HDF5
dataset = SatelliteDataset.from_hdf5(
    phase="train",
    hdf5_file="data/circa.hdf5",
    use_sar="asc"
)

# Depuis fichiers brutes
from src.data.backends.files_backend import Dataset_from_files
backend = Dataset_from_files(
    data_optique="./s2_data",
    data_radar="./s1_data",
    mgrsc="31TCJ"
)
dataset = SatelliteDataset(backend=backend)
```
"""

from src.data.interfaces import (
    SentinelBackend,
    PhaseType,
    ChannelType,
    SampleDict,
    TensorDict,
)
from src.data.sentinel_dataset import SentinelDataset

__all__ = [
    "SentinelBackend",
    "SentinelDataset",
    "PhaseType",
    "ChannelType",
    "SampleDict",
    "TensorDict",
]
