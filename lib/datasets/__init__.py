from dataloader_CIRCA.datasets.UTILISE_ligth import CIRCA_HDF5_Dataset

from .EarthNet2021Dataset import EarthNet2021Dataset
from .SEN12MSCRTSDataset import SEN12MSCRTSDataset

DATASETS = {
    "earthnet2021": EarthNet2021Dataset,
    "sen12mscrts": SEN12MSCRTSDataset,
    "circa": CIRCA_HDF5_Dataset,
}
