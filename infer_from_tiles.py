import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader_CIRCA.datasets.dataset_from_files import Dataset_from_files
from dataloader_CIRCA.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA.tools.type_converter import TypeConverter
from lib import config_utils
from lib.arguments import eval_parser
from lib.eval_tools import Imputation

MAX_PIXEL_INTENSITY_USED_FOR_REVERSE = 10_000
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
    git add infer_from_tiles.py dataloader_CIRCA/datasets/dataset_from_files.py
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
    name_experiment = Path(config.test_data.test_config).parent.name
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

    mask_type = config.test_data.get("mask_type", "orignal_masks")

    ds = Dataset_from_files(
        mgrsc=mgrs25,
        data_optique=data_optique,
        data_radar=data_radar,
        image_size=image_size,
        overlap=overlap,
        fill_value=config.mask.fill_value,
        mask_type=mask_type,
        data_masks=data_masks,
        use_sar=config.data.get("use_sar", "mix_closest"),
    )
    # En mode inférence, ne pas filtrer les dates nuageuses (sinon T varie par patch
    # et ne correspond plus au nombre de bandes du fichier de sortie)
    ds.keep_all_dates = True
    meta = ds.s2_meta.copy()
    output_type = meta["dtype"]
    mgrs25_dataloader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    # Get the imputation model
    imputation = Imputation(
        config_file_train=args.config_file,
        method=args.method,
        mode=args.mode,
        checkpoint=args.checkpoint,
        config_file_test=args.test_data.test_config,
        # temporal_window=MAX_SAMPLES_ON_GPU,
        num_channels=ds.num_channels,
        device=device,
    )
    converter = TypeConverter()
    expected_bands = meta["count"]

    write_errors = 0
    patches_written = 0

    with rasterio.open(out_filename, "w", **meta, **GDAL_OPTIONS) as dst:
        with torch.no_grad():
            for batch_in in tqdm(mgrs25_dataloader, leave=False, total=len(ds), desc="Patches"):
                batch, y_pred = imputation.impute_sample(batch_in)

                # Unpacking direct de la fenêtre pour gagner des variables
                x, y, w, h = batch["window"][0].item(), batch["window"][1].item(), batch["window"][2].item(), batch["window"][3].item()

                # Appel de la fonction utilitaire
                final_patch = _prepare_patch_for_writing(y_pred, batch, converter, output_type)

                # Validation du nombre de bandes AVANT écriture
                if final_patch.shape[0] != expected_bands:
                    write_errors += 1
                    if write_errors == 1:
                        print(
                            f"\n⚠ BAND COUNT MISMATCH pour {mgrs25} : patch a {final_patch.shape[0]} bandes "
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
        print(f"\n✗ ÉCHEC pour {mgrs25} : {write_errors} patchs en erreur, {patches_written} écrits.")
        if patches_written == 0:
            print(f"  Fichier vide supprimé : {out_filename.as_posix()}")
            out_filename.unlink(missing_ok=True)
        print("-----------------------------------------------------")
    else:
        print(f"\n✓ Predictions for MGRS-C area {mgrs25} saved successfully ({patches_written} patches).")
        print(f"  File path: {out_filename.as_posix()}")
        print("-----------------------------------------------------")

    del mgrs25_dataloader
    del ds
    gc.collect()


def main(
    args: argparse.Namespace,
    args_test_data: DictConfig,
):
    """
    Flux des configurations :
    - args.config_file     : chemin vers le fichier de config d'inférence (config_run_infer_from_tiles.yaml)
                             Il est aussi passé comme 'config_file_train' à Imputation (convention héritée de run_eval).
    - args.test_data       : section test_data de la config d'inférence (data_optique, data_radar, test_tiles, etc.)
    - args.test_data.test_config : chemin vers la config d'entraînement du modèle.
    - args_test_data       : section 'data' de la config d'entraînement (channels, use_sar, etc.)
    """
    _ = torch.set_grad_enabled(False)
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Cannot find the configuration file used during training: {args.config_file}\n")
    # Read config file (inference config) — contient test_data, mask, output, misc, etc.
    config = config_utils.read_config(args.config_file)
    # Manage old config settings
    if "include_S1" in args_test_data:
        if args_test_data.include_S1 is True:
            config.data.use_sar = "mix_closest"
        else:
            config.data.use_sar = False
        args_test_data.pop("include_S1")
    # Merge les paramètres 'data' de la config d'entraînement dans la config d'inférence
    config.data.update(args_test_data)
    # Evaluate the entire image sequence (dans le cas de l'evaluation)
    config.data.max_seq_length = None

    # --- Valeurs par défaut pour les sections optionnelles ---
    if "misc" not in config:
        config.misc = OmegaConf.create({"num_workers": 0, "pin_memory": False})
    if "output" not in config:
        raise ValueError(
            "La section 'output' avec 'save_dir' doit être spécifiée dans la configuration d'inférence.\n"
            "Ajoutez :\n  output:\n    save_dir: /chemin/vers/repertoire/sortie"
        )

    # 3. Optimisation pour multiprocessing (num_workers > 0)
    # Rasterio/GDAL est thread-safe mais peut avoir des problèmes avec fork()
    # "spawn" est plus sûr mais plus lent au démarrage.
    # Pour Unix "fork" est plus standard mais peut causer des verrous sur les fichiers ouverts par GDAL
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ==============================================================================
    # FIX POUR L'ERREUR "Bus error / out of shared memory" SUR SLURM
    # ==============================================================================
    # Force PyTorch à utiliser le système de fichiers plutôt que /dev/shm
    # pour le transfert des tenseurs entre les workers du DataLoader
    mp.set_sharing_strategy('file_system')
    # slurm ne permet pas l'utilisation de /dev/shm pour les workers du DataLoader, ce qui peut entraîner des erreurs de mémoire partagée. En utilisant 'file_system', PyTorch utilisera des fichiers temporaires pour le partage de données, ce qui est plus compatible avec les environnements SLURM.
    # ==============================================================================

    image_size = [256, 256]
    overlap = 0

    # CONFIGURATION CRITIQUE pour les workers sur stockage réseau :
    # 1. Empêche GDAL d'essayer d'écrire des fichiers de métadonnées (.aux.xml)
    os.environ["GDAL_PAM_ENABLED"] = "NO"
    # 2. Empêche GDAL de scanner tout le dossier à chaque ouverture
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    # 3. Limit GDAL Cache to avoid OOM on write or heavy flushing issues
    os.environ["GDAL_CACHEMAX"] = "512"  # 512 MB

    # AMÉLIORATION : Plus de workers pour charger les données en parallèle pendant le calcul GPU
    num_workers = config.misc.num_workers  # Essayez 4 ou 8 selon votre CPU
    # Sécuriser GDAL pour les environnements multithread/multiprocess
    # Removing VSI_CACHE as it might cause issues with high-throughput writing or network drives ("dirty block" errors)
    # os.environ["VSI_CACHE"] = "TRUE"
    # os.environ["VSI_CACHE_SIZE"] = "100000000"  # 100MB

    # Désactiver pin_memory si multiprocessing complexe cause des problèmes
    # ou si la RAM est limite
    pin_memory = False if num_workers > 0 else torch.cuda.is_available()

    test_tiles_file = Path(config.test_data.get("test_tiles", None))  # JSON file containing the list of MGRS-C tiles to evaluate on
    assert test_tiles_file is not None, "Le chemin vers le fichier JSON contenant les MGRS-C à évaluer doit être spécifié dans la configuration de test."
    with open(test_tiles_file, encoding="utf-8") as f:
        test_tiles = json.load(f)

    # load_dataset = config.output.get("tiles_window_file", None)
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        eval_parser.print_help()
        sys.exit(1)

    args = eval_parser.parse_args()

    config = config_utils.read_config(args.config_file)
    if "test_data" in config:
        temp = OmegaConf.create()
        temp.config_file = args.config_file
        temp.method = args.method
        temp.test_data = config.test_data
        if "mode" in temp.test_data:
            temp.mode = config.test_data.mode
        if "checkpoint" in temp.test_data:
            temp.checkpoint = config.test_data.checkpoint
            del temp.test_data.checkpoint
        args = temp

    # Extract settings w.r.t. test data
    if args.test_data.test_config is not None:
        if not os.path.isfile(args.test_data.test_config):
            raise FileNotFoundError(f"Cannot find the test configuration file: {args.test_data.test_config}\n")
        test_config = config_utils.read_config(args.test_data.test_config)
        args_test_data = test_config.data
    else:
        args_test_data = OmegaConf.create()

    if args.test_data.split is not None:
        args_test_data.split = args.test_data.split
    if args.test_data.mode is not None:
        args_test_data.mode = args.test_data.mode

    main(args, args_test_data)
    # EOF
