import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar

from cr_ligthtning.json_logger import HistorySaver
from cr_ligthtning.tensorboard import MetricsAdder

NUM_CKPT_SAVED = 3
PROGRESS = 1
EARLY_STOPPING_CONFIG = {
    "patience": 30,
    "monitor": "val_loss",
    "mode": "min",
    "min_delta": 0.00,
}


def configure_callbacks(config) -> List[pl.Callback]:
    """Configure training callbacks."""
    folder_checkpoints = Path(config.output.experiment_folder) / "checkpoints"
    folder_checkpoints.mkdir(parents=True, exist_ok=True)

    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=folder_checkpoints,
        filename="checkpoint-{epoch:02d}-{" + "val_loss" + ":.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=EARLY_STOPPING_CONFIG["monitor"],
        min_delta=EARLY_STOPPING_CONFIG["min_delta"],
        patience=EARLY_STOPPING_CONFIG["patience"],
        mode=EARLY_STOPPING_CONFIG["mode"],
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    return [
        checkpoint_loss_callback,
        early_stop_callback,
        lr_monitor_callback,
        TQDMProgressBar(),
        MetricsAdder(),
        HistorySaver(),
    ]
