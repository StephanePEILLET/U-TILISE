"""
Code taken from the U-TILISE repo made by Corinne Stucker (https://github.com/prs-eth/U-TILISE/blob/main/lib/config_utils.py)
"""

import logging
import os
import sys
from typing import Optional

import prodict
import yaml
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from prodict import Prodict
from rich import get_console
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.tree import Tree

PROGRAM_TITLE = "U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series (Training)"


class _PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


_PrettySafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _PrettySafeLoader.construct_python_tuple)


def resolve_tuple(*args):
    """Resolves a value in a config file with a structure like ${tuple:1,2} to a tuple (1, 2)."""
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)


def print_config(config: DictConfig | prodict.Prodict | str, logger: Optional[logging.Logger] = None) -> None:
    """
    Prints a yaml configuration file to the console.

    Args:
        config:      string or dict, path of the yaml file or previously imported configuration file.
        logger:      logger instance.
    """

    if isinstance(config, Prodict):
        config = config.to_dict(is_recursive=True)
    elif isinstance(config, str):
        config = read_config(config)
    elif isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)

    if logger:
        logger.info(yaml.dump(config, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True))
    else:
        yaml.dump(config, sys.stdout, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True)


def read_config(file: str) -> DictConfig:
    """
    Reads a yaml configuration file.

    Args:
        file:  str, path of the yaml configuration file.

    Returns:
        dict (DictConfig), imported yaml file.
    """

    # Load configuration from file
    if not os.path.exists(file):
        raise FileNotFoundError(f"ERROR: Cannot find the file {file}\n")
    try:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=_PrettySafeLoader)
    except yaml.YAMLError as e:
        raise RuntimeError(f"ERROR: Cannot load the file {file}\n") from e

    return OmegaConf.create(config)


def write_config(data: DictConfig | prodict.Prodict, outfile: str) -> None:
    """
    Writes the dictionary data to a yaml file.

    Args:
        data:     dict (DictConfig), data to be stored as a yaml file.
        outfile:  str, path of the output file.
    """

    with open(outfile, "w", encoding="utf-8") as f:
        if isinstance(data, Prodict):
            yaml.dump(
                data.to_dict(is_recursive=True),
                f,
                indent=4,
                default_flow_style=None,
                sort_keys=False,
                allow_unicode=True,
            )
        elif isinstance(data, DictConfig):
            OmegaConf.save(config=data, f=f)
        else:
            yaml.dump(data, f, indent=4, default_flow_style=None, sort_keys=False, allow_unicode=True)


def print_config_rich(config: DictConfig) -> None:
    """Print content of given config using Rich library and its tree structure.
    Args: config: Config to print to console using a Rich tree.
    """

    def contains_dict(value) -> bool:
        """Check if a value contains a dictionary (recursively for lists)."""
        if isinstance(value, (dict, DictConfig)):
            return True
        if isinstance(value, (list, ListConfig)):
            return any(contains_dict(item) for item in value)
        return False

    def walk_config(tree: Tree, config: DictConfig):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, (dict, DictConfig)):
                # Dictionnaire : créer une branche et continuer récursivement
                branch = tree.add(str(group_name), style=Style(color="pink1", bold=True))
                walk_config(branch, group_option)
            elif isinstance(group_option, (list, ListConfig)):
                if not group_option:
                    # Liste vide : affichage inline
                    tree.add(f"{group_name}: []", style=Style(color="pink1", bold=True))
                elif contains_dict(group_option):
                    # Liste contenant des dictionnaires : affichage avec retour à la ligne
                    branch = tree.add(f"{group_name}:", style=Style(color="pink1", bold=True))
                    for idx, item in enumerate(group_option):
                        if isinstance(item, (dict, DictConfig)):
                            subbranch = branch.add(f"[{idx}]", style=Style(color="cyan", bold=True))
                            walk_config(subbranch, item)
                        else:
                            branch.add(f"[{idx}]: {item}", style=Style(color="pink1"))
                else:
                    # Liste simple : affichage inline
                    tree.add(f"{str(group_name)}: {group_option}", style=Style(color="pink1", bold=True))
            else:
                # Valeur simple
                if group_name == "_target_":
                    tree.add(f"{str(group_name)}: {group_option}", style=Style(color="white", italic=True, bold=True))
                else:
                    tree.add(f"{str(group_name)}: {group_option}", style=Style(color="pink1", bold=True))

    tree = Tree(
        ":rainbow: Configuration Tree :sun_behind_cloud:",
        style=Style(color="white", bold=True, encircle=True),
        guide_style=Style(color="sky_blue1", bold=True),
        expanded=True,
        highlight=True,
    )
    walk_config(tree, config)
    get_console().print(tree)


def print_title() -> None:
    """Print program title to the console."""
    console = Console()
    console.print(
        Panel.fit(
            PROGRAM_TITLE,
            style="bold magenta",
            border_style="bright_blue",
        ),
        justify="center",
    )
