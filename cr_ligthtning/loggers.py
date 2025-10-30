from pathlib import Path

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger

from cr_ligthtning.json_logger import JSONLogger


def configure_loggers(config):
    """Configure experiment loggers."""
    csv_logger = CSVLogger(
        save_dir=config.output.experiment_folder,
        name="csv_logs",
    )

    json_logger = JSONLogger(
        save_dir=config.output.experiment_folder,
        name="json_logs",
    )

    train_logger = TensorBoardLogger(
        save_dir=config.output.experiment_folder,
        name="tensorboard_logs",
        default_hp_metric=False,
        version="Train",
        filename_suffix="_train",
    )

    valid_logger = TensorBoardLogger(
        save_dir=config.output.experiment_folder,
        name="tensorboard_logs",
        default_hp_metric=False,
        version="Validation",
        filename_suffix="_val",
    )

    test_logger = TensorBoardLogger(
        save_dir=config.output.experiment_folder,
        name="tensorboard_logs",
        default_hp_metric=False,
        version="Test",
        filename_suffix="_test",
    )

    return [csv_logger, json_logger, train_logger, valid_logger, test_logger]
