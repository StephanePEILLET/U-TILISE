"""Boucle d'entraînement U-TILISE.

Gère le cycle train/val par epoch avec :
- Accumulation de gradients (gradient_accumulation_steps)
- Sauvegarde du meilleur modèle et checkpoints réguliers
- Logging TensorBoard
- Évaluation automatique en fin d'entraînement (3 masquages)
"""

import logging
import os
import time
from typing import Any

import numpy as np
import prodict
import torch
import torchvision.utils
from omegaconf import DictConfig, ListConfig
from prodict import Prodict
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import logger, visutils
from src.data_utils import compute_false_color, extract_sample, to_device
from src.logger import AverageMeter
from src.loss import TrainLoss
from src.metrics.cloud_removal import CloudRemovalMetrics

OBJECTIVE = {
    "l1": "min",
    "l1_occluded_input_pixels": "min",
    "l1_observed_input_pixels": "min",
    "masked_l1_loss": "min",
    "mae": "min",
    "masked_mae": "min",
    "mse": "min",
    "masked_mse": "min",
    "rmse": "min",
    "masked_rmse": "min",
    "psnr": "max",
    "sam": "min",
    "ssim": "max",
    "total_loss": "min",
    "mad": "max",
    "ols": "max",
    "emd": "max",
    "ens": "max",
}


def seconds_to_dd_hh_mm_ss(seconds_elapsed: int) -> tuple[int, int, int, int]:
    days = seconds_elapsed // (24 * 3600)
    seconds_remainder = seconds_elapsed % (24 * 3600)
    hours = seconds_remainder // 3600
    seconds_remainder %= 3600
    minutes = seconds_remainder // 60
    seconds_remainder %= 60
    seconds = seconds_remainder

    return days, hours, minutes, seconds


class Trainer:
    def __init__(
        self,
        args: DictConfig,
        train_dset: torch.utils.data.Dataset,
        val_dset: torch.utils.data.Dataset,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        model,
        optimizer,
        scheduler,
        device: torch.device = None,
    ):
        self.args = args
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = {"train": train_dset, "val": val_dset}
        self.dataloader = {"train": train_loader, "val": val_loader}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.to(self.device)
        self.args.accum_iter = self.args.get("accum_iter", 1)  # accumulate gradients for `accum_iter` iterations

        self.compute_losses = TrainLoss(self.args.loss)
        # Transform metrics
        list_available_metrics = [l.value for l in CloudRemovalMetrics.MetricType]
        self.compute_metrics = CloudRemovalMetrics(
            metrics=[k for k in self.args.metrics if k in list_available_metrics],
            eval_occluded_observed=False,
            # device=self.device,
        )

        # Losses: Initialize statistics
        self.train_stats = self._stats_meter(stats_type="loss")
        self.val_stats = self._stats_meter(stats_type="loss")

        # Losses: Initialize metrics
        self.train_metrics = self._stats_meter(stats_type="metrics")
        self.val_metrics = self._stats_meter(stats_type="metrics")

        self.best_loss = np.inf
        self.epoch_best_loss = np.nan
        self.early_stop_counter = 0

        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

        self.args.path_model_best = os.path.join(self.args.checkpoint_dir, "Model_best.pth")
        self.args.path_model_last = os.path.join(self.args.checkpoint_dir, "Model_last.pth")
        self.logger = logger.prepare_logger(
            "train_logger",
            level=logging.INFO,
            log_to_console=True,
            log_file=os.path.join(args.save_dir, "training.log"),
        )

        # Set up TensorBoard
        os.makedirs(os.path.join(self.args.save_dir, "tb"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, "tb"))

        # Resume training
        if self.args.resume and self.args.pretrained_path:
            self._resume(path=self.args.pretrained_path)
        else:
            self.logger.info("\nTraining from scratch.\n")
            self.epoch = 0
            self.iter = 0

    def _get_lr(self, group: int = 0) -> float:
        return self.optimizer.param_groups[group]["lr"]

    def _resume(self, path: str) -> None:
        """
        Resumes training.

        Args:
            path:  str, path of the pretrained model weights.
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}\n")

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.args.get("load_scheduler_state_dict", True) and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Extract the last training epoch
        self.epoch = checkpoint["epoch"] + 1
        self.iter = checkpoint["iter"]
        self.args.num_epochs += self.epoch

        # Best validation loss so far
        self.best_loss = checkpoint["best_loss"]
        self.epoch_best_loss = checkpoint["epoch"]
        self.early_stop_counter = checkpoint.get("early_stop_counter", 0)

        self.logger.info("\n\nRestoring the pretrained model from epoch %d.", self.epoch - 1)
        self.logger.info("Successfully loaded pretrained model weights from %s.\n", path)
        self.logger.info("Current best loss %.4f\n", self.best_loss)

    def _save_checkpoint(self, filepath: str) -> None:
        state = {
            "epoch": self.epoch,
            "iter": self.iter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "best_epoch": self.epoch_best_loss,
            "early_stop_counter": self.early_stop_counter,
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def _log_stats_meter(self, phase: str) -> None:
        if phase == "train":
            for k, v in self.train_stats.items():
                self.writer.add_scalar("train_losses/" + k, v.avg, self.iter)
            for k, v in self.train_metrics.items():
                self.writer.add_scalar("train_metrics/" + k, v.avg, self.iter)
        else:
            for k, v in self.val_stats.items():
                self.writer.add_scalar("val_losses/" + k, v.avg, self.iter)
            for k, v in self.val_metrics.items():
                self.writer.add_scalar("val_metrics/" + k, v.avg, self.iter)

        # Write validation stats and metrics to the log file
        if phase == "val":
            self.logger.info(
                f"val:\tEpoch: {self.epoch}\t"
                + "".join([f"{k}: {v.avg:.5f}\t" for k, v in self.val_stats.items()])
                + "".join([f"{k}: {v.avg:.5f}\t" for k, v in self.val_metrics.items()])
            )

    def _log_iter_epoch(self) -> None:
        self.writer.add_scalar("epoch", self.epoch, self.iter)

    def _log_learning_rate(self) -> None:
        self.writer.add_scalar("log_lr", np.log10(self._get_lr()), self.epoch)

    def _stats_dict(self, stats_type: str) -> prodict.Prodict:
        stats = Prodict()

        if stats_type == "metrics":
            masked_metrics = self.args.metrics.masked_metrics
            for key, value in self.args.metrics.items():
                if key == "masked_metrics":
                    pass
                elif value:
                    if masked_metrics and key != "ssim":
                        stats[f"{key}"] = np.inf
                    else:
                        stats[key] = np.inf

        elif stats_type == "loss":
            for key, value in self.args.loss.items():
                # Exclude weight keys
                if value and isinstance(value, bool):
                    stats[key] = np.inf
            stats.total_loss = np.inf

        return stats

    def _update_meter(self, meter: prodict.Prodict, key: str, value: float) -> None:
        if key not in meter:
            meter[key] = AverageMeter()
        meter[key].update(value)

    def _stats_meter(self, stats_type: str) -> prodict.Prodict:
        meters = Prodict()
        stats = self._stats_dict(stats_type)
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def _visualize_sample(self, sample_index: int | None = None) -> None:
        if sample_index is None:
            # Get one random batch
            batch = next(iter(self.dataloader["val"]))
        else:
            # Get specific sample and introduce batch dimension
            batch = self.dataset["val"].__getitem__(sample_index)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0)
                elif isinstance(v, int):
                    batch[k] = [v]

        batch = to_device(batch, self.device)
        x, y, _, mask_valid, _, indices_rgb, index_nir = extract_sample(batch)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x, batch_positions=batch["position_days"])

        # Visualize the valid frames of the first sample in the batch, T x C x H x W
        valid = mask_valid[0] if mask_valid is not None else torch.ones((x.shape[1],))
        x = x[0, valid == 1].cpu()
        y_pred = y_pred[0, valid == 1].cpu()
        y = y[0, valid == 1].cpu()
        ncols = x.shape[0] if x.shape[0] <= 15 else 10
        indices_rgb = indices_rgb.cpu()

        title = "examples_val" if sample_index is None else f"sample_{sample_index}"

        # True color RGB grid: input / prediction / observed
        grid_rgb = torchvision.utils.make_grid(
            [
                visutils.gallery(x[:, indices_rgb, :, :], ncols=ncols).permute(2, 0, 1),
                visutils.gallery(y_pred[:, indices_rgb, :, :], ncols=ncols).permute(2, 0, 1),
                visutils.gallery(y[:, indices_rgb, :, :], ncols=ncols).permute(2, 0, 1),
            ],
            nrow=1,
        )
        self.writer.add_image(f"{title}_true_color_RGB", grid_rgb, global_step=self.iter)

        # False color NIR-R-G grid (if NIR band available)
        if not np.isnan(index_nir):
            grid_fc = torchvision.utils.make_grid(
                [
                    visutils.gallery(
                        compute_false_color(x, index_rgb=indices_rgb, index_nir=index_nir),
                        ncols=ncols,
                    ).permute(2, 0, 1),
                    visutils.gallery(
                        compute_false_color(y_pred, index_rgb=indices_rgb, index_nir=index_nir),
                        ncols=ncols,
                    ).permute(2, 0, 1),
                    visutils.gallery(
                        compute_false_color(y, index_rgb=indices_rgb, index_nir=index_nir),
                        ncols=ncols,
                    ).permute(2, 0, 1),
                ],
                nrow=1,
            )
            self.writer.add_image(f"{title}_false_color_NIRRG", grid_fc, global_step=self.iter)

    def train(self) -> None:
        # Log validation metrics before training starts (to log initial improvement)
        if not (self.args.resume and self.args.pretrained_path):
            self.validate_epoch()
            self._log_stats_meter(phase="val")

        self.logger.info("\nStart training...\n")
        start_time = time.time()

        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_description("Epoch")
            tnr.set_postfix(
                epoch=self.epoch,
                training_loss=np.nan,
                validation_loss=np.nan,
                best_validation_loss=self.best_loss,
            )
            for _ in tnr:
                if self.scheduler is not None:
                    self._log_learning_rate()

                # -------------------------------- TRAINING -------------------------------- #
                self.train_epoch(tnr)

                # -------------------------------- VALIDATION -------------------------------- #
                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate_epoch(tnr)
                    self._log_stats_meter(phase="val")

                    # Save the best model
                    if self.val_stats.total_loss.avg < self.best_loss:
                        self.best_loss = self.val_stats.total_loss.avg
                        self.epoch_best_loss = self.epoch
                        self.early_stop_counter = 0
                        self._save_checkpoint(self.args.path_model_best)
                    else:
                        self.early_stop_counter += 1

                    # Early stopping
                    early_stop_patience = self.args.get("early_stop_patience", 0)
                    if early_stop_patience > 0 and self.early_stop_counter >= early_stop_patience:
                        self.logger.info(
                            f"\nEarly stopping triggered after {self.early_stop_counter} epochs "
                            f"without improvement. Best val loss: {self.best_loss:.4f} "
                            f"at epoch {self.epoch_best_loss}."
                        )
                        break

                    # Plot inference
                    if (self.epoch + 1) % self.args.plot_every_n_epochs == 0:
                        self._visualize_sample()  # Plot a random validation sample
                        if self.args.get("plot_val_sample", None) is not None:
                            # Plot specific validation sample(s)
                            if isinstance(self.args.plot_val_sample, int):
                                self._visualize_sample(sample_index=self.args.plot_val_sample)
                            elif isinstance(self.args.plot_val_sample, (list, ListConfig)):
                                for idx in self.args.plot_val_sample:
                                    self._visualize_sample(sample_index=idx)

                # After the epoch if finished, update the learning rate scheduler
                if self.scheduler is not None:

                    if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(self.val_stats.total_loss.avg)
                    else:
                        self.scheduler.step()

                # Save the model at the selected interval
                if (self.epoch + 1) % self.args.checkpoint_every_n_epochs == 0:
                    name = "Model_after_" + str(self.epoch + 1) + "_epochs.pth"
                    self._save_checkpoint(os.path.join(self.args.checkpoint_dir, name))

                self.epoch += 1

        time_elapsed = int(time.time() - start_time)
        self.logger.info(
            "\n\nTraining finished!\nTraining time: %dd %dh %dm %ds" % seconds_to_dd_hh_mm_ss(time_elapsed)
        )
        self.logger.info("\nBest model at epoch: %d", self.epoch_best_loss)
        self.logger.info(f"Validation loss of the best model: {self.best_loss:.4f}")

        # Save the last model
        self._save_checkpoint(self.args.path_model_last)

        self.writer.close()

    def train_epoch(self, tnr=None) -> None:
        # Initialize stats meter
        self.train_stats = self._stats_meter(stats_type="loss")
        self.train_metrics = self._stats_meter(stats_type="metrics")
        self.model.train()

        # Clear gradients
        # self.model.zero_grad(set_to_none=True)
        for param in self.model.parameters():
            param.grad = None

        with tqdm(self.dataloader["train"], leave=False) as tnr_train:
            tnr_train.set_description("Training")
            tnr_train.set_postfix(
                epoch=self.epoch,
                training_loss=-np.inf,
                **{k: v.avg for (k, v) in self.train_metrics.items()},
            )

            for i, batch in enumerate(tnr_train):
                self._log_iter_epoch()  # Itération à l'epoch
                loss_dict, metrics, loss = self.inference_one_batch(batch, phase="train")
                # Update to stats_meter
                # self.train_stats.update(**loss_dict)
                # self.train_metrics.update(**metrics)
                for key, value in loss_dict.items():
                    self.train_stats[key].update(value)
                for key, value in metrics.items():
                    self._update_meter(self.train_metrics, key, value)

                loss = loss / self.args.accum_iter
                loss.backward()

                if ((i + 1) % self.args.accum_iter == 0) or (i + 1 == len(self.dataloader["train"])):
                    # Gradient clipping
                    if getattr(self.args, "gradient_clip_norm", False) and self.args.gradient_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.gradient_clip_norm,
                        )

                    elif getattr(self.args, "gradient_clip_value", False) and self.args.gradient_clip_value > 0.0:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clip_value)

                    self.optimizer.step()

                    # Clear gradients
                    for param in self.model.parameters():
                        param.grad = None

                if (i + 1) % min(self.args.logstep_train, len(self.dataloader["train"])) == 0:
                    self._log_stats_meter(phase="train")

                    tnr_train.set_postfix(
                        epoch=self.epoch,
                        training_loss=self.train_stats.total_loss.avg,
                        **{k: v.avg for (k, v) in self.train_metrics.items()},
                    )
                    if tnr is not None:
                        tnr.set_postfix(
                            epoch=self.epoch,
                            training_loss=self.train_stats.total_loss.avg,
                            validation_loss=self.val_stats.total_loss.avg,
                            best_validation_loss=self.best_loss,
                        )

                    # Reset stats and metrics
                    for key in self.train_stats:
                        self.train_stats[key].reset()
                    for key in self.train_metrics:
                        self.train_metrics[key].reset()
                self.iter += 1

    def validate_epoch(self, tnr=None) -> None:
        # Initialize stats meter
        self.val_stats = self._stats_meter(stats_type="loss")
        self.val_metrics = self._stats_meter(stats_type="metrics")
        self.model.eval()

        with tqdm(self.dataloader["val"], leave=False) as tnr_val:
            tnr_val.set_description("Validation")
            tnr_val.set_postfix(epoch=self.epoch)

            for batch_idx, batch in enumerate(tnr_val):
                loss_dict, metrics = self.inference_one_batch(batch, phase="val")

                # Update to stats_meter
                for key, value in loss_dict.items():
                    self.val_stats[key].update(value)
                for key, value in metrics.items():
                    self._update_meter(self.val_metrics, key, value)

        if tnr is not None:
            tnr.set_postfix(
                epoch=self.epoch,
                training_loss=self.train_stats.total_loss.avg,
                validation_loss=self.val_stats.total_loss.avg,
                best_validation_loss=self.best_loss,
            )

    def inference_one_batch(
        self,
        batch: dict[str, Any],
        phase: str,
    ) -> tuple[dict[str, float], dict[str, float], Tensor] | tuple[dict[str, float], dict[str, float]]:
        """
        Perform inference on a single batch of data.
        Args:
            batch (Dict[str, Any]): Input batch containing 'x' (input data) and
                'position_days' (temporal position information).
            phase (str): Training phase, must be one of "train", "val", or "test".
        Returns:
            Tuple containing:
            - For training phase: (loss_dict, metrics, loss) where:
                - loss_dict (Dict[str, float]): Dictionary of computed losses
                - metrics (Dict[str, float]): Dictionary of evaluation metrics
                - loss (Tensor): Combined loss tensor for backpropagation
            - For validation/test phases: (loss_dict, metrics) where:
                - loss_dict (Dict[str, float]): Dictionary of computed losses
                - metrics (Dict[str, float]): Dictionary of evaluation metrics
        Raises:
            AssertionError: If phase is not one of "train", "val", or "test".
        Note:
            - For training phase, gradients are computed and loss tensor is returned
            - For validation/test phases, inference is performed with torch.no_grad()
            - Input batch is automatically moved to the appropriate device
        """
        assert phase in ["train", "val", "test"]

        batch = to_device(batch, self.device)

        if phase == "train":
            # with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # casts operations to mixed precision
            y_pred = self.model(batch["x"], batch_positions=batch["position_days"])

            # Compute losses and evaluation metrics
            loss_dict, loss = self.compute_losses(batch, y_pred)
            metrics = self.compute_metrics(
                target=batch["y"],
                masks=batch["masks"],
                predicted=y_pred,
                cloud_masks=batch.get("cloud_mask", None),
            )
            # Diff en loss_dict
            return loss_dict, metrics, loss

        # if phase == "val" or "test"
        # with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # casts operations to mixed precision
        with torch.no_grad():
            y_pred = self.model(batch["x"], batch_positions=batch["position_days"])

            # Compute  losses and evaluation metrics
            loss_dict, _ = self.compute_losses(batch, y_pred)
            metrics = self.compute_metrics(
                target=batch["y"],
                masks=batch["masks"],
                predicted=y_pred,
                cloud_masks=batch.get("cloud_mask", None),
            )

            return loss_dict, metrics
