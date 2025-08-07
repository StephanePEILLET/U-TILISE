from dataloader_CIRCA.datasets import CIRCA_ADAPTED2UTILISE_Dataset

from .EarthNet2021Dataset import EarthNet2021Dataset
from .SEN12MSCRTSDataset import SEN12MSCRTSDataset

DATASETS = {
    "earthnet2021": EarthNet2021Dataset,
    "sen12mscrts": SEN12MSCRTSDataset,
    "circa": CIRCA_ADAPTED2UTILISE_Dataset,
}
