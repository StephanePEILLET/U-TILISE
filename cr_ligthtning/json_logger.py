"""
JSON logger
----------

JSON logger for basic experiment logging that does not require opening ports

"""

import json
import logging
import os
from argparse import Namespace
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.logger import _convert_params
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_warn

log = logging.getLogger(__name__)

INDENT = 2


class ExperimentWriter:
    r"""
    Experiment writer for JSONLogger.

    Currently supports to log hyperparameters and metrics in YAML and JSON
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_HPARAMS_FILE = "hparams.yaml"
    NAME_METRICS_FILE = "metrics.json"

    def __init__(self, log_dir: str) -> None:
        self.hparams = {}
        self.metrics = {}

        self.log_dir = log_dir
        if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            elif isinstance(value, np.ndarray):
                return value.tolist()
            return value

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}

        # TODO: have to correct a lag on the step/epoch where the LR monitor values are stored.
        # Change the method for obtaining the step value (in LR case) to make it work also when step != epoch.
        if "LR Scheduler" in metrics_dict.keys() and len(self.metrics) > 0:
            step = len(self.metrics) - 1
            metrics.update(self.metrics[step])
        else:
            step = len(self.metrics)

        self.metrics.update({int(step): metrics})

    def save(self) -> None:
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.metrics:
            return

        with open(self.metrics_file_path, "w") as f:
            json.dump(self.metrics, f, indent=INDENT)


class JSONLogger(Logger):
    r"""
    Log to local file system in yaml and JSON format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._experiment = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is ``None`` or the empty string, no experiment subdirectory is used and the
        checkpoint will be saved in "save_dir/version_dir"
        """
        if not self.name:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        """The current directory where logs are saved.

        Returns:
            The path to current directory where logs are saved.
        """
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics, step)
        if step is not None and (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name

    @property
    def version(self) -> int:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self._save_dir, self.name)

        if not os.path.isdir(root_dir):
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


from pytorch_lightning import Trainer
from pytorch_lightning.loggers.logger import DummyLogger


class HistorySaver(pl.Callback):
    def __init__(self):
        super().__init__()
        self.idx_json_loggers = None
        self.phase_dict = {"val": 0, "test": 1}

    def get_json_logger(self, trainer: Trainer, phase: str) -> JSONLogger:
        """
        Safely get JSONlogger from Trainer attributes according to the current phase.
        """
        if self.idx_json_loggers is None:
            self.idx_json_loggers = []

            if isinstance(trainer.logger, JSONLogger):
                self.idx_json_loggers = [0]

            elif isinstance(trainer.loggers, list):
                for idx, logger in enumerate(trainer.loggers):
                    if isinstance(logger, JSONLogger):
                        self.idx_json_loggers.append(idx)

        if self.idx_json_loggers is not None:
            if len(self.idx_json_loggers) == 1:
                return trainer.loggers[self.idx_json_loggers[0]]
            else:
                phase_idx = self.phase_dict[phase]
                logger_idx = self.idx_json_loggers[phase_idx]
                return trainer.loggers[logger_idx]
        else:
            raise ValueError(
                "HistogramAdder callback is not use properly.",
            )

    def get_metrics_to_log(self, pl_module, phase):
        metrics, loss = None, None
        if phase not in ["train", "val", "test"]:
            raise ValueError(
                "The possible phases are train, val, test or predict.",
            )
        elif phase == "train":
            metrics, loss = pl_module.train_epoch_metrics, pl_module.train_epoch_loss
        elif phase == "val":
            metrics, loss = pl_module.val_epoch_metrics, pl_module.val_epoch_loss
        elif phase == "test":
            metrics, loss = pl_module.test_epoch_metrics, pl_module.test_epoch_loss

        if metrics is not None and loss is not None:
            metric_collection = {
                key: (value.cpu().numpy() if isinstance(value, torch.Tensor) else value)
                for key, value in metrics.items()
            }
            metric_collection["loss"] = loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss
            return metric_collection
        else:
            return None

    def logs_to_json(self, trainer, pl_module, phase):
        if not isinstance(trainer.logger, DummyLogger):
            logger = self.get_json_logger(trainer=trainer, phase=phase)
            metric_collection = self.get_metrics_to_log(pl_module=pl_module, phase=phase)
            if metric_collection is not None:
                logger.experiment.log_metrics(metric_collection, pl_module.current_epoch)
                logger.experiment.save()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.logs_to_json(trainer, pl_module, phase="val")

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.logs_to_json(trainer, pl_module, phase="test")
