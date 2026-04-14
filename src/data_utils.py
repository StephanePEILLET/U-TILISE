import collections.abc
import logging
import random
import re
import warnings
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.data import SentinelDataset

warnings.filterwarnings("ignore", category=UserWarning)

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def to_device(sample: dict[str, Any], device: torch.device = torch.device("cuda")) -> dict[str, Any]:
    sample_out = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sample_out[key] = val.to(device)
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.to(device))
                else:
                    new_val.append(e)
            sample_out[key] = new_val
        else:
            sample_out[key] = val

    return sample_out


def extract_sample(
    sample: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float | int]:
    inputs = sample["x"]
    target = sample["y"]
    masks = sample["masks"]
    mask_valid = sample["masks_valid_obs"] if "masks_valid_obs" in sample else None
    cloud_mask = sample["cloud_mask"] if "cloud_mask" in sample else None
    indices_rgb = sample.get("c_index_rgb", torch.Tensor([2, 1, 0]))
    index_nir = sample.get("c_index_nir", torch.Tensor([np.nan]))

    if isinstance(indices_rgb[0], torch.Tensor):
        indices_rgb = indices_rgb[0]
    if not isinstance(index_nir, (float, int)):
        index_nir = index_nir[0].item()

    return inputs, target, masks, mask_valid, cloud_mask, indices_rgb, index_nir


def pad_tensor(x: Tensor, l: int, pad_value: int | float = 0) -> Tensor:
    """
    Source: https://github.com/VSainteuf/utae-paps/blob/main/src/utils.py
    """

    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch: list[Any], pad_value: int | float = 0) -> Any:
    """
    Modified version of: https://github.com/VSainteuf/utae-paps/blob/main/src/utils.py
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            # out = elem.new(storage)
            out = elem.new(storage).resize_(len(batch), *list(batch[0].size()))
        return torch.stack(batch, 0, out=out)
    if elem_type.__module__ == "numpy" and elem_type.__name__ not in {"str_", "string_"}:
        if elem_type.__name__ in ("ndarray", "memmap"):
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(f"Format not managed : {elem.dtype}")
            return pad_collate([torch.as_tensor(b) for b in batch])
        if elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    if isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    if isinstance(elem, int):
        return torch.tensor(batch)
    if isinstance(elem, str):
        return batch
    if isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError(f"Format not managed : {elem_type}")


def seed_worker(worker_id):
    """
    Initialise la graine aléatoire pour un worker du DataLoader.
    Assure que chaque worker a une séquence de nombres aléatoires différente.
    """
    # On récupère la graine de base (si définie avec torch.manual_seed) et on la rend unique
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_dataloader(
    dset: torch.utils.data.Dataset,
    config: DictConfig,
    drop_last: bool = False,
    subset: bool | int | None = False,
    shuffle: bool | None = None,
    batch_size: int | None = None,
    pin_memory: bool | None = False,
    generator: torch.Generator | None = None,
) -> torch.utils.data.dataloader.DataLoader:
    """Returns a torch.utils.data.DataLoader instance."""

    variable_seq_length = getattr(dset, "variable_seq_length", False) and config.training_settings.batch_size > 1
    # shuffle = config["misc"]["run_mode"] != "test"

    if subset:
        dset = torch.utils.data.Subset(dset, range(subset))

    if variable_seq_length:
        collate_fn = partial(pad_collate, pad_value=config.get("pad_value", 0))
    else:
        collate_fn = None

    loader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=(batch_size if batch_size is not None else config.training_settings.batch_size),
        shuffle=(shuffle if shuffle is not None else False),
        num_workers=config.misc.get("num_workers", 4) if hasattr(config.misc, 'get') else 4,
        collate_fn=collate_fn,
        pin_memory=config.misc.get("pin_memory", pin_memory),
        drop_last=drop_last,
        # generator=generator,
        # worker_init_fn=seed_worker,
    )
    return loader


def get_dataset(config: DictConfig, phase: str, logger: logging.Logger | None = None) -> Dataset:
    """
    Instancie et retourne le dataset configure selon les parametres.
    
    Détecte AUTOMATIQUEMENT le backend à utiliser :
    - Si `data_optique` ET `data_radar` sont présents → utilise `FilesBackend` (fichiers TIF directs)
    - Sinon → utilise `HDF5Backend` (comportement historique)
    
    Fonctionne de la même façon pour training, évaluation et inférence.
    Rétrocompatibilité 100% avec toutes les anciennes configurations.
    
    Args:
        config: Configuration complète du run
        phase: Phase du dataset (train/val/test/train+val)
        logger: Logger optionnel pour les messages
    """
    assert config["misc"]["run_mode"] in ["train", "val", "test"]
    assert phase in ["train", "val", "train+val", "test"]

    augment = config.data.get("augment", phase == "train")

    # Rétro-compatibilité : si l'ancienne clé filter_settings existe dans la config,
    # on extrait return_valid_obs_only et on passe filter_settings au constructeur
    # qui émettra un DeprecationWarning.
    filter_settings = config.data.get("filter_settings")
    if filter_settings is not None:
        return_valid_obs_only = filter_settings.get("return_valid_obs_only", True)
    else:
        return_valid_obs_only = config.data.get("return_valid_obs_only", True)

    # Paramètres spécifiques U-TILISE (communs à TOUS les backends)
    dataset_kwargs = {
        "return_valid_obs_only": return_valid_obs_only,
        "max_seq_length": config.data.get("max_seq_length"),
        "render_occluded_above_p": config.data.get("render_occluded_above_p"),
        "pe_strategy": config.data.get("pe_strategy", "day-of-year"),
        "process_data": config.data.get("process_data", True),
        "mask_sar": config.data.get("mask_sar", False),
        "return_windows": config.data.get("return_windows", False),
        "mask_kwargs": config.mask,
        "augment": augment,
    }
    # Retirer les None pour laisser les défauts du constructeur s'appliquer
    dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if v is not None}

    # =========================================================================
    # Detection automatique du backend
    # =========================================================================
    data_optique = config.data.get("data_optique")
    data_radar = config.data.get("data_radar")

    # Cas 1 : Utilisateur veut charger depuis des fichiers TIF directs
    if data_optique is not None and data_radar is not None:
        if logger:
            logger.info(f"Backend detecte : Fichiers brutes (pas de HDF5)")
            logger.info(f"   Optique : {data_optique}")
            logger.info(f"   Radar : {data_radar}")
        
        from src.data.backends.files_backend import Dataset_from_files
        
        # Paramètres spécifiques au backend fichiers
        backend_kwargs = {
            "data_optique": data_optique,
            "data_radar": data_radar,
            "mgrsc": config.data.get("mgrsc"),
            "image_size": config.data.get("image_size", (256, 256)),
            "overlap": config.data.get("overlap", 0),
            "load_dataset": config.data.get("load_dataset"),
            "shuffle": config.data.get("shuffle", False),
            "use_sar": config.data.get("use_sar", "mix_closest"),
            "channels": config.data.get("channels", "all"),
            "pe_strategy": config.data.get("pe_strategy", "day-within-sequence"),
            "fill_value": config.data.get("fill_value", 1.0),
            "mask_type": config.data.get("mask_type", "original_masks"),
            "data_masks": config.data.get("data_masks"),
            "keep_all_dates": config.data.get("keep_all_dates", False),
        }
        backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}
        
        # Initialiser le backend fichiers
        backend = Dataset_from_files(**backend_kwargs)
        
        # Initialiser l'adaptateur générique
        return SentinelDataset(backend=backend, **dataset_kwargs)
    
    # Cas 2 : Comportement historique → backend HDF5
    else:
        if logger:
            logger.info(f"Backend detecte : Fichier HDF5")
        
        # Résoudre le fichier HDF5 (peut être un dict par phase ou un chemin unique)
        hdf5_file = config.data.get("hdf5_file")
        if isinstance(hdf5_file, DictConfig):
            hdf5_file = hdf5_file[phase]
        
        if logger and hdf5_file:
            logger.info(f"   Fichier : {hdf5_file}")
        
        # Utiliser le constructeur rétro-compatible
        return SentinelDataset.from_hdf5(
            hdf5_file=hdf5_file,
            phase=phase,
            channels=config.data.get("channels", "all"),
            use_sar=config.data.get("use_sar", "mix_closest"),
            load_transforms=config.data.get("load_transforms"),
            shuffle=config.data.get("shuffle", False),
            **dataset_kwargs
        )


def compute_false_color(x: Tensor, index_rgb: Tensor | list[int], index_nir: int | float) -> Tensor:
    """
    Returns the false color composite (NIR, R, G) for every time step of the input sequence or
    for the single input image.

    Args:
        x:           torch.Tensor, image time series, T x C x H x W (sequence) or C x H x W (single image).
        index_rgb:   list of int, indices of the RGB channels.
        index_nir:   int, index of the NIR channel.

    Returns:         torch.Tensor, false color composite (NIR, R, G), T x 3 x H x W (sequence) or
                     3 x H x W (single image).
    """

    if x.dim() == 4:
        return torch.stack(
            (x[:, index_nir, ...], x[:, index_rgb[0], ...], x[:, index_rgb[1], ...]),
            dim=1,
        )

    return torch.stack((x[index_nir, ...], x[index_rgb[0], ...], x[index_rgb[1], ...]), dim=1)
