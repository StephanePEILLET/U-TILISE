import os
import sys
import torch
from matplotlib import pyplot as plt
from typing import Any, Dict

from lib import config_utils
from lib import data_utils
from lib import visutils
from lib.eval_tools import (
    Imputation,
    visualize_att_for_one_head_across_time,
    visualize_att_for_target_t_across_heads,
)


def get_dataloader_testdata(
    config_file_train: str,
    config_file_test: str,
    run_mode: str = 'test'
) -> torch.utils.data.dataloader.DataLoader:
    if not os.path.isfile(config_file_train):
        raise FileNotFoundError(f'Cannot find the configuration file used during training: {config_file_train}\n')

    if not os.path.isfile(config_file_test):
        raise FileNotFoundError(f'Cannot find the test configuration file: {config_file_test}\n')

    # Read the configuration file used during training
    config = config_utils.read_config(config_file_train)

    # Merge generic data settings (used during training) with test-specific data settings
    config_testdata = config_utils.read_config(config_file_test)
    config.data.update(config_testdata.data)
    if 'mask' in config_testdata:
        config.mask.update(config_testdata.mask)
    config.misc.run_mode = run_mode

    # Get the data loader
    dset = data_utils.get_dataset(config, phase=run_mode)
    dataloader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
    )
    return dataloader


def get_sample(
    dataloader: torch.utils.data.dataloader.DataLoader,
    sample_index: int
) -> Dict[str, Any]:
    
    batch = dataloader.dataset.__getitem__(sample_index)
    # Introduce the batch dimension (required for the forward pass)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        elif isinstance(v, int):
            batch[k] = [v]

    return batch


if __name__ == "__main__":
    # Default data and model settings (i.e., settings used during training)
    # config_file_train = 'configs/demo.yaml'
    config_file_train = 'configs/demo.yaml'
    # Test-specific data settings
    config_file_test = 'configs/config_sen12mscrts_test.yaml'

    # Model weights
    checkpoint = 'checkpoints/utilise_sen12mscrts_w_s1.pth'

    train_config = config_utils.read_config(config_file_train)
    test_config = config_utils.read_config(config_file_test)
    dataloader = get_dataloader_testdata(config_file_train, config_file_test)
    config_utils.print_config(train_config)

    batch = get_sample(dataloader, sample_index=269)

    print(batch.keys())


