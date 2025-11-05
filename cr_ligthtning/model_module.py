from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchmetrics import MeanMetric

from dataloader_CIRCA.datasets.cr_torchmetrics import CloudRemovalDatasetMetrics
from lib.eval_tools import impute_sequence


class CR_module(pl.LightningModule):

    def __init__(self, config, model, criterion, optimizer, scheduler):
        super().__init__()
        self.config = config
        self.temporal_window = config.data.get("max_seq_length", None)
        # fenêtre de temps utilisée lors de l'inférence complète pour découper
        # la séquence en sous-séquences et passer au modèle des sequences de la taille même
        # tant que celle utilisée lors de l'entraînement.
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None
            self.train_metrics = CloudRemovalDatasetMetrics(eval_occluded_observed=True, clean_gt_cloudy_pixels=True)
            self.val_metrics = CloudRemovalDatasetMetrics(eval_occluded_observed=True, clean_gt_cloudy_pixels=True)
            self.train_loss = MeanMetric(nan_strategy="ignore")
            self.val_loss = MeanMetric(nan_strategy="ignore")
        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            self.val_metrics = CloudRemovalDatasetMetrics(eval_occluded_observed=True, clean_gt_cloudy_pixels=True)
            self.val_loss = MeanMetric(nan_strategy="ignore")
        elif stage == "test":
            self.test_epoch_loss, self.test_epoch_metrics = None, None
            self.test_metrics = CloudRemovalDatasetMetrics(eval_occluded_observed=True, clean_gt_cloudy_pixels=True)
            self.test_loss = MeanMetric(nan_strategy="ignore")

    def inference_one_batch(self, batch):
        y_pred = self.model(batch["x"], batch_positions=batch["position_days"])
        # Compute losses and evaluation metrics
        _, loss = self.criterion(batch, y_pred)
        return loss, y_pred, batch

    def impute_sample(
        self,
        batch: Dict[str, Any],
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        return_att: Optional[bool] = False,
    ) -> Tuple[Dict[str, Any], Tensor, Tensor] | Tuple[Dict[str, Any], Tensor]:

        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch["x"] = batch["x"][:, t_start:t_end, ...]
            for key in ["y", "masks", "cloud_mask", "masks_valid_obs"]:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]
            for key in ["days", "position_days"]:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        # Impute the given satellite image time series
        if return_att:
            y_pred, att = impute_sequence(self.model, batch, self.temporal_window, return_att=True)
            if att is not None:
                att = att.detach().cpu()
        else:
            y_pred = impute_sequence(self.model, batch, self.temporal_window, return_att=False)

        if return_att:
            return batch, y_pred, att
        return batch, y_pred

    def inference_full_sequence(self, batch):
        batch, y_pred = self.impute_sample(
            batch,
            t_start=None,
            t_end=None,
            return_att=False,
        )
        _, loss = self.criterion(batch, y_pred)
        return loss, y_pred, batch

    def training_step(self, batch, batch_idx):
        loss, y_pred, batch = self.inference_one_batch(batch)
        return {
            "loss": loss,
            "y_pred": y_pred,
            "batch": batch,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        loss, y_pred, batch = outputs["loss"], outputs["y_pred"], outputs["batch"]
        self.train_loss.update(loss)
        self.train_metrics.update(
            target=batch["y"],
            masks=batch["masks"],
            predicted=y_pred,
            cloud_masks=batch.get("cloud_mask", None),
        )
        return loss

    def on_train_epoch_end(self):
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()
        self.log("train_loss", self.train_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, y_pred, batch = self.inference_full_sequence(batch)
        return {
            "loss": loss,
            "y_pred": y_pred,
            "batch": batch,
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx) -> None:
        loss, y_pred, batch = outputs["loss"], outputs["y_pred"], outputs["batch"]
        self.val_loss.update(loss)
        self.val_metrics.update(
            target=batch["y"],
            masks=batch["masks"],
            predicted=y_pred,
            cloud_masks=batch.get("cloud_mask", None),
        )
        return loss

    def on_validation_epoch_end(self):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log("val_loss", self.val_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log(
            "mae",
            self.val_epoch_metrics["mae_occluded_input_pixels"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.val_loss.reset()
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, y_pred, batch = self.inference_full_sequence(batch)
        self.val_loss.update(loss)
        self.val_metrics.update(
            target=batch["y"],
            masks=batch["masks"],
            predicted=y_pred,
            cloud_masks=batch.get("cloud_mask", None),
        )
        return loss

    def on_test_epoch_end(self):
        self.test_epoch_loss = self.test_loss.compute()
        self.test_epoch_metrics = self.test_metrics.compute()
        self.log("test_loss", self.test_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.test_loss.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        lr_scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",
            "monitor": "val_loss",
            "frequency": 1,
            "strict": True,
            "name": "LR Scheduler",
        }
        config = {
            "optimizer": self.optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
        return config
