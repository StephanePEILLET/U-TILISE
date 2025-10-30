import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import make_grid


class TensorboardCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.idx_loggers = None
        self.phase_dict = {"train": 0, "val": 1, "test": 2}

    def get_tensorboard_logger(self, trainer: Trainer, phase: str) -> TensorBoardLogger:
        """
        Safely get TensorBoardLogger from Trainer attributes according to the current phase.
        """

        if phase not in self.phase_dict.keys():
            raise ValueError(
                "The possible phases are train, val, test or predict.",
            )

        if self.idx_loggers is None:
            self.idx_loggers = []

            if isinstance(trainer.logger, TensorBoardLogger):
                self.idx_loggers = 0

            elif isinstance(trainer.loggers, list):
                for idx, logger in enumerate(trainer.loggers):
                    if isinstance(logger, TensorBoardLogger):
                        self.idx_loggers.append(idx)
            else:
                raise ValueError(
                    "ERROR: the callback TensorboardCallback won't work if there is any logger\
                        of type TensorBoardLogger."
                )

        if self.idx_loggers:
            if self.idx_loggers == 0:
                return trainer.loggers
            else:
                phase_idx = self.phase_dict[phase]
                logger_idx = self.idx_loggers[phase_idx]
                return trainer.loggers[logger_idx]


class MetricsAdder(TensorboardCallback):

    @rank_zero_only
    def add_metrics(self, trainer, pl_module, metric_collection, loss, phase):
        if not isinstance(trainer.logger, DummyLogger) and metric_collection is not None and loss is not None:
            # Get logger for the current phase
            logger = self.get_tensorboard_logger(trainer=trainer, phase=phase)
            # Add the loss value to the experiment
            logger.experiment.add_scalar("Loss", loss, global_step=pl_module.current_epoch)
            # Add every value computed to the experiment
            for key_metric in metric_collection.keys():
                logger.experiment.add_scalar(
                    key_metric,
                    metric_collection[key_metric],
                    global_step=pl_module.current_epoch,
                )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.add_metrics(
            trainer=trainer,
            pl_module=pl_module,
            metric_collection=pl_module.train_epoch_metrics,
            loss=pl_module.train_epoch_loss,
            phase="train",
        )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_metrics(
            trainer=trainer,
            pl_module=pl_module,
            metric_collection=pl_module.val_epoch_metrics,
            loss=pl_module.val_epoch_loss,
            phase="val",
        )

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.add_metrics(
            trainer=trainer,
            pl_module=pl_module,
            metric_collection=pl_module.test_epoch_metrics,
            loss=pl_module.test_epoch_loss,
            phase="test",
        )


class HistogramAdder(TensorboardCallback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def add_histogram(self, trainer, pl_module, phase):
        if not isinstance(trainer.logger, DummyLogger):
            logger = self.get_tensorboard_logger(trainer=trainer, phase=phase)
            for name, params in pl_module.named_parameters():
                logger.experiment.add_histogram(name, params, pl_module.current_epoch)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.add_histogram(trainer=trainer, pl_module=pl_module, phase="train")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        self.add_histogram(trainer=trainer, pl_module=pl_module, phase="val")

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        self.add_histogram(trainer=trainer, pl_module=pl_module, phase="test")


# class PredictionsAdder(TensorboardCallback):

# ALPHA = 0.4
# OCSGE_LUT = [
#  (219,  14, 154),
#  (114, 113, 112),
#  (248,  12,   0),
#  ( 61, 230, 235),
#  (169, 113,   1),
#  ( 21,  83, 174),
#  (255, 255, 255),
#  (138, 179, 160),
#  ( 70, 228, 131),
#  ( 25,  74,  38),
#  (243, 166,  13),
#  (102,   0, 130),
#  (255, 243,  13),
#  (228, 223, 124),
#   (128, 0, 255),
#  (64, 128, 128),
#  (223, 0, 223),
#  (128, 128, 192),
#  (  0,   0,   0),
# ]

#     def __init__(
#         self,
#         train_samples=None,
#         val_samples=None,
#         test_samples=None,
#         display_bands=[1, 2, 3],
#     ):

#         super().__init__()
#         self.tensorboard_logger_idx = None
#         self.train_samples = train_samples
#         self.val_samples = val_samples
#         self.test_samples = test_samples
#         self.sample_dataset = None
#         self.display_bands = [idx_band - 1 for idx_band in display_bands]

#     @rank_zero_only
#     def add_predictions(self, trainer, pl_module, phase):
#         if not isinstance(trainer.logger, DummyLogger):
#             trainer.datamodule.create_samples(phase=phase)
#             samples = trainer.datamodule.samples[phase]
#             images, targets = samples["image"].to(device=pl_module.device), samples[
#                 "mask"
#             ].to(device=pl_module.device)

#             with torch.no_grad():
#                 logits = pl_module.forward(images)
#                 proba = torch.softmax(logits, dim=1)
#                 preds = torch.argmax(proba, dim=1)

#             # images = images.cpu().type(torch.uint8)
#             images = images.cpu().numpy()
#             targets = targets.cpu()
#             preds = preds.cpu()
#             grids = []

#             inv_tfm = trainer.datamodule.inv_transforms[phase]
#             inv_images = np.stack([inv_tfm(image=image)["image"] for image in images])
#             images = torch.tensor(inv_images).type(torch.uint8)
#             images = torch.stack(
#                 [images[:, band_i, :, :] for band_i in self.display_bands], 1
#             )

#             for image, target, pred in zip(images, targets, preds):
#                 pred_bands = torch.zeros_like(target)
#                 for class_i in np.arange(trainer.datamodule.num_classes):
#                     pred_bands[class_i, :, :] = pred == class_i
#                 pred_bands = (
#                     pred_bands == 1
#                 )  # draw_segmentation_masks function needs masks as bool tensors
#                 target = target == 1
#                 pred_overlay = draw_segmentation_masks(
#                     image, masks=pred_bands, colors=OCSGE_LUT, alpha=ALPHA
#                 )
#                 target_overlay = draw_segmentation_masks(
#                     image, masks=target, colors=OCSGE_LUT, alpha=ALPHA
#                 )
#                 grids.append(make_grid([image, target_overlay, pred_overlay]))
#             image_grid = torch.cat(grids, 1)

#             logger = self.get_tensorboard_logger(trainer=trainer, phase=phase)
#             logger.experiment.add_image(
#                 "Images - Masks - Predictions", image_grid, pl_module.current_epoch
#             )

#     @rank_zero_only
#     def on_train_epoch_end(self, trainer, pl_module):
#         self.add_predictions(trainer=trainer, pl_module=pl_module, phase="train")

#     @rank_zero_only
#     def on_validation_epoch_end(self, trainer, pl_module):
#         self.add_predictions(trainer=trainer, pl_module=pl_module, phase="val")

#     @rank_zero_only
#     def on_test_epoch_end(self, trainer, pl_module):
#         self.add_predictions(trainer=trainer, pl_module=pl_module, phase="test")
