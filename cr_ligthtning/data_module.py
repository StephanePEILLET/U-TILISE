from functools import partial
from typing import Optional

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from dataloader_CIRCA.datasets.UTILISE_adapter import CIRCA_ADAPTED2UTILISE_Dataset as Dataset  # NOQA
from lib.data_utils import pad_collate
from lib.utils import without_keys

RANDOM_SEED = 42


class CRDataModule(LightningDataModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.subset = config.data.get("subset", False)
        self.fit_batch_size, self.infer_batch_size = config.training_settings.batch_size, 1
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if config.data.filter_settings.get("return_valid_obs_only", False) and config.training_settings.batch_size > 1:
            self.variable_seq_length = True
            self.collate_fn = partial(pad_collate, pad_value=config.method.pad_value)
        else:
            self.variable_seq_length = False
            self.collate_fn = None
        self.fit_generator = torch.Generator()
        self.fit_generator.manual_seed(config.misc.get("random_seed", RANDOM_SEED))
        self.num_channels, self.seq_length, self.image_size = None, None, None
        # Paramètre permettant de faire de l'inférence complète lors des evaluations
        if self.config.data.get("eval_full_inference", False):
            self.config_full_inference = OmegaConf.create(self.config)
            self.config_full_inference.data.max_seq_length = None
            self.config_full_inference.training_settings.batch_size = 1

    def get_dataset(self, config, phase: str):
        """Returns a torch.utils.data.Dataset instance."""
        dset = Dataset(
            hdf5_file=config.data.hdf5_file,
            phase=phase,
            **without_keys(config.data, ["dataset", "hdf5_file", "subset", "eval_full_inference"]),
            mask_kwargs=config.mask,
            augment=False,
        )
        if self.num_channels is None and self.seq_length is None and self.image_size is None:
            self.num_channels = dset.num_channels
            self.seq_length = dset.seq_length
            self.image_size = dset.image_size
        return dset

    def get_dataloader(
        self,
        config,
        dset: torch.utils.data.Dataset,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None,
        pin_memory: Optional[bool] = False,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.utils.data.dataloader.DataLoader:
        """Returns a torch.utils.data.DataLoader instance."""
        loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=(batch_size if batch_size is not None else config.training_settings.batch_size),
            shuffle=(shuffle if shuffle is not None else False),
            num_workers=config.misc.num_workers,
            collate_fn=(None if self.config.data.get("eval_full_inference", False) else self.collate_fn),
            pin_memory=config.misc.get("pin_memory", pin_memory),
            drop_last=drop_last,
            generator=(None if self.config.data.get("eval_full_inference", False) else generator),
        )
        return loader

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            if not self.train_dataset and not self.val_dataset:
                self.train_dataset = self.get_dataset(self.config, phase="train")
                if self.config.data.get("eval_full_inference", False):
                    self.val_dataset = self.get_dataset(self.config_full_inference, phase="val")
                else:
                    self.val_dataset = self.get_dataset(self.config, phase="val")
            if self.subset is True:
                self.train_dataset = Subset(self.train_dataset, range(0, 20))
                self.val_dataset = Subset(self.val_dataset, range(0, 10))

        if stage == "test":
            if not self.test_dataset:
                if self.config.data.get("eval_full_inference", False):
                    self.test_dataset = self.get_dataset(self.config_full_inference, phase="test")
                else:
                    self.test_dataset = self.get_dataset(self.config, phase="test")
            if self.subset is True:
                self.test_dataset = Subset(self.test_dataset, range(0, 10))

    def train_dataloader(self):
        return self.get_dataloader(
            self.config,
            dset=self.train_dataset,
            shuffle=True,
            batch_size=self.fit_batch_size,
            pin_memory=self.config.misc.get("pin_memory", False),
            generator=self.fit_generator,
            drop_last=True,
        )

    def val_dataloader(self):
        return self.get_dataloader(
            self.config,
            dset=self.val_dataset,
            shuffle=False,
            batch_size=self.fit_batch_size,
            pin_memory=self.config.misc.get("pin_memory", False),
            drop_last=False,
            generator=self.fit_generator,
        )

    def test_dataloader(self):
        return self.get_dataloader(
            self.config,
            dset=self.test_dataset,
            shuffle=True,
            batch_size=self.infer_batch_size,
            pin_memory=self.config.misc.get("pin_memory", False),
            drop_last=False,
        )
