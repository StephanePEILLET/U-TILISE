"""Inférence U-TILISE sur des tuiles Sentinel-2 complètes.

Charge les données à plat (GeoTIFF) via dataset_from_files, applique le modèle
par patches 256×256 avec fenêtre glissante, puis fusionne les résultats en
GeoTIFF géoréférencés.

Usage:
    python run_inference.py <config_infer.yaml> utilise
"""

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.backends.files_backend import Dataset_from_files
from src.data.processing.transforms import SentinelDataProcessor
from src.data.processing.transforms import TypeConverter
from src import config_utils
from src.arguments import eval_parser
from src.eval_tools import Imputation

MAX_PIXEL_INTENSITY_USED_FOR_REVERSE = 10_000


def _worker_init_fn(worker_id):
    """Propagate file_system sharing strategy to spawned DataLoader workers.

    With 'spawn' start method, workers don't inherit the parent's sharing
    strategy and default to file_descriptor (POSIX shm → /dev/shm).
    Also redirect tempfile to disk-backed storage (not tmpfs).
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Ensure temp files go to the same disk-backed dir as the parent
    _tmp = os.environ.get("TMPDIR", None)
    if _tmp:
        tempfile.tempdir = _tmp


GDAL_OPTIONS = {
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "sparse_ok": True,  # Was "SPARSE_MODE" (invalid), recommended True for large files
    "bigtiff": "YES",  # Force BigTIFF to avoid issues with file size limits and re-writes
    "num_threads": "1",  # Restricted to 1 to prevent issues with locking/multiprocessing on clusters
    "interleave": "pixel",  # CRITICAL: PIXEL interleave prevents massive seeking when writing multi-band tiles
}


def _handle_folders(config: DictConfig):
    """
    Handle folders paths and output    # 1. Commit et push les changements locaux
    git add run_inference.py dataloader/datasets/dataset_from_files.py
    git commit -m "fix: keep_all_dates=True pour inférence tuiles + validation bandes + use_sar + shared memory"
    git push origin jzay

    # 2. Sur jzay, faire un git pull puis supprimer les fichiers corrompus et relancer directories based on the provided configuration.
    """
    # Recupération chemins depuis la patie test_data de la config
    data_optique = Path(config.test_data.get("data_optique", None))
    assert data_optique is not None, "Le chemin vers les données optiques doit être spécifié dans la configuration de test."
    data_radar = Path(config.test_data.get("data_radar", None))
    assert data_radar is not None, "Le chemin vers les données radar doit être spécifié dans la configuration de test."
    # Répertoire optionnel contenant les masques synthétiques (aléatoire/consécutif)
    data_masks_val = config.test_data.get("data_masks", None)
    data_masks = Path(data_masks_val) if data_masks_val is not None else None
    # Répertoires de sortie
    output_folder = Path(config.output.save_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    name_experiment = Path(config.test_data.get("test_config", "default")).parent.name
    output_folder_inferences = output_folder / name_experiment
    output_folder_inferences.mkdir(parents=True, exist_ok=True)
    return Path(data_optique), Path(data_radar), data_masks, Path(output_folder_inferences)


def _prepare_patch_for_writing(y_pred, batch, converter, output_type):
    """Effectue la dénormalisation, la concaténation et le formatage du patch."""
    # Reverse normalization
    denorm_pred = SentinelDataProcessor.reverse_process_MS(
        y_pred, intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE
    )

    # 1. Get prediction (T, 10, h, w) on CPU
    pred_patch = denorm_pred.squeeze(axis=0).cpu().numpy()

    # 2. Get original bands 11 & 12 (T, 2, h, w) from the batch
    original_bands = batch["original_masks"].squeeze(axis=0).cpu().numpy()

    # 3. Concatenate (T, 12, h, w)
    full_patch = np.concatenate([pred_patch, original_bands], axis=1)

    # 4. Reshape to flattened channels (T*12, h, w)
    full_patch = full_patch.reshape(
        full_patch.shape[0] * full_patch.shape[1], full_patch.shape[2], full_patch.shape[3]
    )

    # 5. Convert to output type (e.g., uint16)
    return converter.from_type("float32").to_type(output_type).convert(full_patch)


def inference_one_tile(
    args: argparse.Namespace,
    mgrs25: str,
    config: DictConfig,
    image_size: list,
    pin_memory: bool,
    num_workers: int,
    overlap: int = 0,
):
    """
    Perform inference on a single MGRS-C tile using the provided imputation model and configuration.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_optique, data_radar, data_masks, output_folder_inferences = _handle_folders(config)
    out_filename = output_folder_inferences / f"pred_mgrsc_{mgrs25}.tif"

    if out_filename.exists():
        # Vérifier la taille du fichier : si < 10 KB, c'est probablement un fichier corrompu d'un run précédent
        file_size_kb = out_filename.stat().st_size / 1024
        if file_size_kb > 10:
            print(f"Predictions for MGRS-C area {mgrs25} already exist ({file_size_kb:.0f} KB). Skipping...")
            return
        else:
            print(f"Predictions for MGRS-C area {mgrs25} exist but are likely corrupted ({file_size_kb:.1f} KB). Re-processing...")
            out_filename.unlink()

    mask_type_val = config.mask.get("mask_type", "original_masks")
    if (mask_type_val == "random_fully_masked" or mask_type_val == "consecutive_fully_masked"):
        data_masks_val = config.test_data.get("data_masks", None)
        if data_masks_val is None:
            raise ValueError(f"Mask type {mask_type_val} requires a data_masks directory in the configuration.")
        mask_type = "fully_masked"
        keep_all_dates = False
    else:
        mask_type = "original_masks"
        keep_all_dates = True

    ds = Dataset_from_files(
        mgrsc=mgrs25,
        data_optique=data_optique,
        data_radar=data_radar,
        image_size=image_size,
        overlap=overlap,
        fill_value=config.mask.get("fill_value", 1),
        mask_type=mask_type,
        data_masks=data_masks,
        use_sar=config.data.get("use_sar", "mix_closest"),
        keep_all_dates=keep_all_dates,
    )

    # ds.keep_all_dates = True
    meta = ds.s2_meta.copy()
    output_type = meta["dtype"]
    print(f"Output dtype: {output_type}")
    mgrs25_dataloader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers, worker_init_fn=_worker_init_fn)

    # Get the imputation model
    imputation = Imputation(
        train_config_path=args.train_config_path,
        checkpoint=args.checkpoint,
        num_channels=ds.num_channels,
        device=device,
    )
    converter = TypeConverter()
    expected_bands = meta["count"]

    write_errors = 0
    patches_written = 0

    # Test de la modification afin de pouvoir passer les inférences en interleave à la place de pixel (pour faciliter la lecture dans QGIS et éviter les problèmes de lecture des bandes dans certains logiciels SIG)
    dictionnaire = {
        'interleave': 'Band',
        'tiled': True
    }
    GDAL_OPTIONS.update(dictionnaire)

    with rasterio.open(out_filename, "w", **meta, **GDAL_OPTIONS) as dst:
        with torch.no_grad():
            for batch_in in tqdm(mgrs25_dataloader, leave=False, total=len(ds), desc="Patches"):
                batch, y_pred = imputation.impute_sample(batch_in)

                # Unpacking direct de la fenêtre pour gagner des variables
                x, y, w, h = batch["window"][0].item(), batch["window"][1].item(), batch["window"][2].item(), batch["window"][3].item()

                if keep_all_dates:
                    # Appel de la fonction utilitaire
                    final_patch = _prepare_patch_for_writing(y_pred, batch, converter, output_type)
                    # En mode inférence, le nombre de bandes doit être constant et égal à T*12 (T dates, 12 bandes par date)
                else:
                    full_s2 = batch["full_s2"].squeeze(axis=0).cpu().numpy()  # (T_all, 12, h, w) raw values
                    full_s2_data, s2_masks = full_s2[:, :10, ...], full_s2[:, 10:, ...]  # Séparer les données S2 (raw) des masques (raw)
                    idx_kept = batch["idx_kept"].squeeze(axis=0).cpu().numpy()  # (T_kept,)
                    assert len(idx_kept) == y_pred.shape[1], f"Mismatch between number of kept dates ({len(idx_kept)}) and model output time dimension ({y_pred.shape[1]})."

                    denorm_pred = SentinelDataProcessor.reverse_process_MS(
                        y_pred, intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE
                    )
                    # Get prediction (T_kept, 10, h, w) on CPU — denormalized to [0, 10000]
                    pred_patch = denorm_pred.squeeze(axis=0).cpu().numpy()

                    # Restore the original temporal order: predictions for kept dates,
                    # raw S2 data (same scale) for dropped dates (really cloudy)
                    full_s2_data[idx_kept, ...] = pred_patch

                    # Concatenate (T_all, 12, h, w) — raw mask values preserved
                    full_patch = np.concatenate([full_s2_data, s2_masks], axis=1)

                    # 4. Reshape to flattened channels (T*12, h, w)
                    final_patch = full_patch.reshape(
                        full_patch.shape[0] * full_patch.shape[1], full_patch.shape[2], full_patch.shape[3]
                    )
                    # 5. Convert to output type (e.g., uint16)
                    final_patch = converter.from_type("float32").to_type(output_type).convert(final_patch)

                # Validation du nombre de bandes AVANT écriture
                if final_patch.shape[0] != expected_bands:
                    write_errors += 1
                    if write_errors == 1:
                        print(
                            f"\n[WARNING] BAND COUNT MISMATCH pour {mgrs25} : patch a {final_patch.shape[0]} bandes "
                            f"mais le fichier attend {expected_bands}. "
                            f"Vérifiez que keep_all_dates=True est actif (T doit être constant par patch)."
                        )
                    continue

                try:
                    dst.write(final_patch, window=Window(x, y, w, h))
                    patches_written += 1
                except Exception as e:
                    write_errors += 1
                    if write_errors <= 3:
                        print(f"Error writing patch at x={x}, y={y}: {e}")

    if write_errors > 0:
        print(f"\n[ECHOEC] pour {mgrs25} : {write_errors} patchs en erreur, {patches_written} ecrits.")
        if patches_written == 0:
            print(f"  Fichier vide supprimé : {out_filename.as_posix()}")
            out_filename.unlink(missing_ok=True)
        print("-----------------------------------------------------")
    else:
        print(f"\n[OK] Predictions for MGRS-C area {mgrs25} saved successfully ({patches_written} patches).")
        print(f"  File path: {out_filename.as_posix()}")
        print("-----------------------------------------------------")

    del mgrs25_dataloader
    del ds
    gc.collect()


def main(
    args: argparse.Namespace,
    config: DictConfig,
):
    """
    Flux des configurations :
    - config : configuration finale fusionnee (default.yaml + train_config + inference_config)
    - args.train_config_path : chemin vers la config d'entrainement (pour Imputation)
    - args.checkpoint : chemin vers le checkpoint du modele
    """
    _ = torch.set_grad_enabled(False)

    if "misc" not in config:
        config.misc = OmegaConf.create({"num_workers": 0, "pin_memory": False})
    if "output" not in config:
        raise ValueError(
            "La section 'output' avec 'save_dir' doit etre specifiee dans la configuration d'inference.\n"
            "Ajoutez :\n  output:\n    save_dir: /chemin/vers/repertoire/sortie"
        )

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    mp.set_sharing_strategy('file_system')
    for scratch_var in ("JOBSCRATCH", "SCRATCH", "SLURM_TMPDIR"):
        scratch_dir = os.environ.get(scratch_var)
        if scratch_dir and os.path.isdir(scratch_dir):
            os.environ["TMPDIR"] = scratch_dir
            tempfile.tempdir = scratch_dir
            print(f"[shared memory fix] TMPDIR redirige vers ${scratch_var}={scratch_dir}")
            break
    else:
        print("[shared memory fix] Aucun repertoire scratch trouve, TMPDIR inchange.")

    image_size = [256, 256]
    overlap = 0

    os.environ["GDAL_PAM_ENABLED"] = "NO"
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["GDAL_CACHEMAX"] = "512"

    try:
        num_workers = min(config.misc.num_workers, 2)
    except (AttributeError, Exception):
        num_workers = 2
    pin_memory = False if num_workers > 0 else torch.cuda.is_available()

    test_tiles_file = Path(config.test_data.get("test_tiles", None))
    assert test_tiles_file is not None, "Le chemin vers le fichier JSON contenant les MGRS-C a evaluer doit etre specifie dans la configuration de test."
    with open(test_tiles_file, encoding="utf-8") as f:
        test_tiles = json.load(f)

    for mgrs25 in tqdm(test_tiles, desc="MGRS-C areas"):
        inference_one_tile(
            args=args,
            mgrs25=mgrs25,
            image_size=image_size,
            config=config,
            pin_memory=pin_memory,
            num_workers=num_workers,
            overlap=overlap,
        )
    print("Evaluation completed.")

    _, _, _, output_folder_inferences = _handle_folders(config)
    config_dump_path = output_folder_inferences / "config_inference.yaml"
    OmegaConf.save(config, config_dump_path)
    print(f"Config d'inference sauvegardee : {config_dump_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        eval_parser.print_help()
        sys.exit(1)

    args = eval_parser.parse_args()

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Cannot find the configuration file: {args.config_file}\n")

    config = config_utils.read_config_with_defaults(args.config_file, run_mode="test")

    test_data_section = config.get("test_data", OmegaConf.create())
    train_config_path = test_data_section.get("test_config", None)
    checkpoint = test_data_section.get("checkpoint", args.checkpoint)

    if train_config_path is None:
        raise ValueError("test_data.test_config (chemin vers la config d'entrainement) est requis.\n")
    if not os.path.isfile(train_config_path):
        raise FileNotFoundError(f"Cannot find the training configuration file: {train_config_path}\n")

    train_config = config_utils.read_config_with_defaults(train_config_path, run_mode="test")

    config = OmegaConf.merge(train_config, config)
    config.misc.run_mode = "test"
    config.data.max_seq_length = None

    args.train_config_path = train_config_path
    args.checkpoint = checkpoint
    args.test_data = test_data_section

    main(args, config)
