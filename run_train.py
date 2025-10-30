import argparse
import os
import sys
import warnings
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from cr_ligthtning.callbacks import configure_callbacks
from cr_ligthtning.data_module import CRDataModule
from cr_ligthtning.display_configs import PROGRAM_TITLE
from cr_ligthtning.display_configs import print_config_rich as print_config
from cr_ligthtning.display_configs import print_title
from cr_ligthtning.loggers import configure_loggers
from cr_ligthtning.model_module import CR_module
from lib import config_utils
from lib import utils
from lib.formatter import RawFormatter
from lib.loss import TrainLoss

# Ignore spécifiquement ce warning de PyTorch
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)

# Constants
SEED = 42
MIN_ARGS_COUNT = 2

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
    config.output.experiment_folder = utils.create_output_directory(config)
    return config


def setup_output_directories(config: OmegaConf) -> None:
    """Setup output directories and save configuration files."""
    # Save the path of the checkpoint directory
    config.output.checkpoint_dir = os.path.join(config.output.experiment_folder, "checkpoints")
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    # Write the runtime configuration to file
    config_file = os.path.join(config.output.experiment_folder, "config.yaml")
    config_utils.write_config(config, config_file)


def setup_model(
    config: OmegaConf,
    input_dim: int,
    seq_length: int,
    image_size: int,
) -> tuple[torch.nn.Module, dict]:
    """Setup and configure the model."""
    model, args_model = utils.get_model(config, input_dim)
    # Log model parameters to file
    config_file = os.path.join(config.output.experiment_folder, "model_config.yaml")
    config_utils.write_config(OmegaConf.create({config.method.model_type: args_model}), config_file)
    # Write model architecture to txt file
    if config.output.plot_model_txt:
        file = os.path.join(config.output.experiment_folder, "model_parameters.txt")
        utils.write_model_structure_to_file(
            file,
            model,
            config.training_settings.batch_size,
            seq_length,
            input_dim,
            image_size,
        )
    return model, args_model


def main(args: argparse.Namespace) -> None:
    """Main training function."""

    print_title()
    config = setup_configuration(args)
    setup_output_directories(config)
    print_config(config)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    data_module = CRDataModule(config=config)
    data_module.setup(stage="fit")

    # Setup model
    model, _ = setup_model(
        config=config,
        input_dim=data_module.num_channels,
        seq_length=data_module.seq_length,
        image_size=data_module.image_size,
    )

    # Setup training components
    optimizer = utils.get_optimizer(config, model=model)
    scheduler = utils.get_scheduler(config, optimizer)

    criterion = TrainLoss(config.loss)

    cr_module = CR_module(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        num_nodes=1,
        callbacks=configure_callbacks(config),
        logger=configure_loggers(config),
        max_epochs=config.training_settings.num_epochs,
        strategy="auto",
        enable_progress_bar=True,
    )

    trainer.fit(
        cr_module,
        datamodule=data_module,
    )

    trainer.test(
        cr_module,
        datamodule=data_module,
    )


if __name__ == "__main__":
    if len(sys.argv) < MIN_ARGS_COUNT:
        parser.print_help()
        sys.exit(1)

    main(parser.parse_args())
