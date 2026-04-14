from nestargs import NestedArgumentParser

from src.formatter import RawFormatter

eval_parser = NestedArgumentParser(
    description="U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series (Evaluation/Inference)",
    formatter_class=RawFormatter,
)

eval_parser.add_argument(
    "config_file",
    metavar="config-file",
    type=str,
    help="Evaluation or inference configuration file (YAML). Must contain a test_data section with test_config pointing to the training config.",
)

eval_parser.add_argument(
    "--checkpoint",
    type=str,
    required=False,
    help="Model checkpoint for U-TILISE",
)

# Optional arguments to overwrite the data settings in the config file
eval_parser.add_argument(
    "--test-data.test-config",
    type=str,
    required=False,
    help="YAML configuration file, test-specific settings",
)
eval_parser.add_argument(
    "--test-data.data-dir",
    type=str,
    required=False,
    help="Root directory of the dataset",
)
eval_parser.add_argument(
    "--test-data.hdf5-file",
    type=str,
    required=False,
    help="HDF5 test dataset, path relative to <data-dir>",
)
eval_parser.add_argument(
    "--test-data.hdf5-file-read",
    type=str,
    required=False,
    help="HDF5 test dataset, path relative to <data-dir>",
)

eval_parser.add_argument("--test-data.split", type=str, required=False, help="Data split")
eval_parser.add_argument(
    "--test-data.mode",
    type=str,
    required=False,
    help="Phase du dataset (test, val, train)",
)
