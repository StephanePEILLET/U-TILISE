import argparse
import logging
import os
import sys
import warnings
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from lib import config_utils
from lib import data_utils
from lib import utils
from lib.formatter import RawFormatter
from lib.logger import prepare_logger

# Ignore spécifiquement ce warning de PyTorch
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

# Constants
MIN_ARGS_COUNT = 2
PROGRAM_TITLE = "U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series (Training)"

parser = ArgumentParser(
    description=PROGRAM_TITLE,
    formatter_class=RawFormatter,
)
parser.add_argument(
    "config_file",
    type=str,
    help="yaml configuration file to augment/overwrite the settings in configs/default.yaml",
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="Path to the directory where models and logs should be saved",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Use Weights & Biases instead of TensorBoard",
)
parser.add_argument("--wandb_project", type=str, default="utilise", help="Wandb project name")


def setup_configuration(args: argparse.Namespace) -> OmegaConf:
    """Setup and merge configuration files."""
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"ERROR: Cannot find the yaml configuration file: {args.config_file}")

    # Import the user configuration file
    cfg_custom = config_utils.read_config(args.config_file)
    if not cfg_custom:
        sys.exit(1)

    # Augment/overwrite the default parameter settings with the runtime arguments given by the user
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "configs", "default.yaml")
    cfg_default = config_utils.read_config(default_config_path)
    config = OmegaConf.merge(cfg_default, cfg_custom)
    config.output.output_directory = args.save_dir

    if args.wandb:
        config.wandb = OmegaConf.create()
        config.wandb.project = args.wandb_project

    return config


def setup_logging(config: OmegaConf) -> logging.Logger:
    """Setup logging configuration."""
    # Create the output directory
    config.output.experiment_folder = utils.create_output_directory(config)

    # Set up the logger
    log_file = os.path.join(config.output.experiment_folder, "run.log") if config.output.experiment_folder else None
    return prepare_logger("root_logger", level=logging.INFO, log_to_console=True, log_file=log_file)


def setup_datasets(
    config: OmegaConf,
    logger: logging.Logger,
):
    train_dset = data_utils.get_dataset(config, phase="train", logger=logger)
    val_dset = data_utils.get_dataset(config, phase="val", logger=logger)
    return train_dset, val_dset


def setup_data_loaders(
    train_dset: torch.utils.data.Dataset,
    val_dset: torch.utils.data.Dataset,
    config: OmegaConf,
    logger: logging.Logger,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Initialize and return train and validation data loaders."""
    logger.info("\nInitialize data loader (training set)...")

    subset = config.data.get("subset", False)
    if subset and isinstance(config.data.subset, bool):
        subset = 10

    train_loader = data_utils.get_dataloader(
        train_dset,
        config,
        drop_last=True,
        subset=subset,
    )

    logger.info("Initialize data loader (validation set)...\n")
    val_loader = data_utils.get_dataloader(
        val_dset,
        config,
        drop_last=False,
        subset=subset,
    )

    if subset:
        logger.info("Number of training samples: %d", subset)
        logger.info("Number of validation samples: %d", subset)
    else:
        logger.info("Number of training samples: %d", train_dset.__len__())
        logger.info("Number of validation samples: %d", val_dset.__len__())
    logger.info("Variable sequence lengths: %r\n", train_dset.variable_seq_length)

    return train_loader, val_loader


def setup_output_directories(config: OmegaConf, logger: logging.Logger) -> None:
    """Setup output directories and save configuration files."""
    logger.info("\nPrepare output folders and files\n--------------------------------\n")

    # Save the path of the checkpoint directory
    config.output.checkpoint_dir = os.path.join(config.output.experiment_folder, "checkpoints")
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    logger.info("Model weights will be stored in: %s\n", config.output.checkpoint_dir)

    # Write the runtime configuration to file
    config_file = os.path.join(config.output.experiment_folder, "config.yaml")
    config_utils.write_config(config, config_file)


def setup_model(config: OmegaConf, train_dset: torch.utils.data.Dataset, logger: logging.Logger):
    """Setup and configure the model."""
    logger.info("\nModel Architecture\n------------------\n")
    logger.info("Architecture: %s", config.method.model_type)

    input_dim = train_dset.num_channels
    model, args_model = utils.get_model(config, input_dim, logger)
    logger.info("Number of trainable parameters: %d\n", utils.count_model_parameters(model))

    # Log model parameters to file
    config_file = os.path.join(config.output.experiment_folder, "model_config.yaml")
    config_utils.write_config(OmegaConf.create({config.method.model_type: args_model}), config_file)

    # Write model architecture to txt file
    if config.output.plot_model_txt:
        file = os.path.join(config.output.experiment_folder, "model_parameters.txt")
        logger.info("Writing model architecture to file: %s\n", file)
        utils.write_model_structure_to_file(
            file,
            model,
            config.training_settings.batch_size,
            train_dset.seq_length,
            input_dim,
            train_dset.image_size,
        )
    return model, args_model


def setup_training_components(config: OmegaConf, model, logger: logging.Logger):
    """Setup optimizer and scheduler for training."""
    optimizer = utils.get_optimizer(config, model, logger)
    scheduler = utils.get_scheduler(config, optimizer, logger)
    return optimizer, scheduler


def log_system_info(logger: logging.Logger) -> None:
    """Log system and environment information."""
    logger.info("\nPrepare training\n----------------\n")
    logger.info("Python version: %s", sys.version)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA version: %s\n", torch.version.cuda)


def main(args: argparse.Namespace) -> None:
    """Main training function."""
    print(f"\n{PROGRAM_TITLE}\n{'=' * len(PROGRAM_TITLE)}\n")

    # Setup configuration
    config = setup_configuration(args)

    # Setup logging
    logger = setup_logging(config)

    # Print runtime arguments to the console
    logger.info("Configuration file: %s", args.config_file)
    logger.info("\nSettings\n--------\n")
    config_utils.print_config(config, logger=logger)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # Setup datasets
    train_dset, val_dset = setup_datasets(config, logger)

    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(train_dset, val_dset, config, logger)

    # Setup output directories
    setup_output_directories(config, logger)

    # Setup model
    model, _ = setup_model(config, train_dset, logger)

    # Log system information
    log_system_info(logger)

    # Setup training components
    optimizer, scheduler = setup_training_components(config, model, logger)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # Initialize the trainer and start training
    trainer = utils.get_trainer(config, train_dset, val_dset, train_loader, val_loader, model, optimizer, scheduler)
    trainer.train()


if __name__ == "__main__":
    if len(sys.argv) < MIN_ARGS_COUNT:
        parser.print_help()
        sys.exit(1)

    main(parser.parse_args())
