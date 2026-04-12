import argparse
import gc
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
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

MAX_WRITE_RETRIES = 3
WRITE_RETRY_DELAY_S = 2
GC_INTERVAL = 50

# Mutable container for the SLURM stop flag (avoids global statement).
# Set to True by signal handlers so the tile loop exits cleanly.
_stop_flag: dict[str, bool] = {"requested": False}


def _handle_stop_signal(signum, _frame) -> None:
    sig_name = signal.Signals(signum).name
    print(f"\n[SIGNAL] Received {sig_name}. Will stop cleanly after the current tile.", flush=True)
    _stop_flag["requested"] = True


GDAL_OPTIONS = {
    "compress": "LZW",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "bigtiff": "YES",
    "num_threads": "1",
    "interleave": "band",
}


def _worker_init_fn(worker_id):
    torch.multiprocessing.set_sharing_strategy('file_system')
    _tmp = os.environ.get("TMPDIR", None)
    if _tmp:
        tempfile.tempdir = _tmp


def _validate_tif(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "File does not exist"
    if path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"
    try:
        with rasterio.open(path) as src:
            _ = src.profile
            src.read(1, window=Window(0, 0, min(256, src.width), min(256, src.height)))
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _safe_unlink(path: Path, label: str = "") -> None:
    try:
        if path.exists():
            path.unlink()
            tag = f"[{label}]" if label else ""
            print(f"  {tag} Deleted: {path.name}")
    except Exception as e:
        print(f"  Failed to delete {path.name}: {e}")


def _write_patch_with_retry(dst, final_patch: np.ndarray, window: Window) -> bool:
    for attempt in range(1, MAX_WRITE_RETRIES + 1):
        try:
            dst.write(final_patch, window=window)
            return True
        except Exception:
            if attempt < MAX_WRITE_RETRIES:
                time.sleep(WRITE_RETRY_DELAY_S)
    return False


def _prepare_patch_keep_all(y_pred, batch, converter, output_type):
    denorm_pred = SentinelDataProcessor.reverse_process_MS(
        y_pred, intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE
    )
    pred_patch = denorm_pred.squeeze(axis=0).cpu().numpy()  # (T, 10, H, W) in [0, 10000]
    # Re-apply nodata mask: pixels where all bands == 0 in the target must stay 0
    # (partial-nodata patches, e.g. tile edges, would otherwise get model predictions)
    target = batch["y"].squeeze(0).cpu().numpy()           # (T, 10, H, W) in [0, 1]
    nodata = (target == 0).all(axis=1, keepdims=True)      # (T, 1, H, W)
    pred_patch = np.where(nodata, 0.0, pred_patch)
    original_bands = batch["original_masks"].squeeze(axis=0).cpu().numpy()
    full_patch = np.concatenate([pred_patch, original_bands], axis=1)
    full_patch = full_patch.reshape(
        full_patch.shape[0] * full_patch.shape[1], full_patch.shape[2], full_patch.shape[3]
    )
    return converter.from_type("float32").to_type(output_type).convert(full_patch)


def _prepare_patch_filtered(y_pred, batch, converter, output_type):
    full_s2 = batch["full_s2"].squeeze(axis=0).cpu().numpy()
    full_s2_data, s2_masks = full_s2[:, :10, ...], full_s2[:, 10:, ...]
    idx_kept = batch["idx_kept"].squeeze(axis=0).cpu().numpy()
    assert len(idx_kept) == y_pred.shape[1], (
        f"Mismatch: {len(idx_kept)} kept dates vs {y_pred.shape[1]} output."
    )
    denorm_pred = SentinelDataProcessor.reverse_process_MS(
        y_pred, intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE
    )
    pred_patch = denorm_pred.squeeze(axis=0).cpu().numpy()  # (T_kept, 10, H, W) in [0, 10000]
    # Re-apply nodata mask: full_s2_data contains raw values (0 = nodata) for kept dates
    nodata = (full_s2_data[idx_kept] == 0).all(axis=1, keepdims=True)  # (T_kept, 1, H, W)
    pred_patch = np.where(nodata, 0.0, pred_patch)
    full_s2_data[idx_kept, ...] = pred_patch
    full_patch = np.concatenate([full_s2_data, s2_masks], axis=1)
    full_patch = full_patch.reshape(
        full_patch.shape[0] * full_patch.shape[1],
        full_patch.shape[2],
        full_patch.shape[3],
    )
    return converter.from_type("float32").to_type(output_type).convert(full_patch)


def _save_manifest(output_dir: Path, results: dict) -> None:
    manifest_path = output_dir / "inference_manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(results)
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    tmp.replace(manifest_path)


def inference_one_tile(
    args: argparse.Namespace,
    mgrs25: str,
    config: DictConfig,
    image_size: list,
    pin_memory: bool,
    num_workers: int,
    overlap: int,
    data_optique: Path,
    data_radar: Path,
    data_masks: Path | None,
    output_folder_inferences: Path,
    imputation: "Imputation | None" = None,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_filename = output_folder_inferences / f"pred_mgrsc_{mgrs25}.tif"
    tmp_filename = output_folder_inferences / f".tmp_pred_mgrsc_{mgrs25}.tif"

    if out_filename.exists():
        ok, msg = _validate_tif(out_filename)
        if ok:
            print(f"[{mgrs25}] Already exists and valid. Skipping.")
            return "skipped", 0, 0, 0
        print(f"[{mgrs25}] Existing file corrupted ({msg}). Re-processing...")
        _safe_unlink(out_filename, "cleanup")

    _safe_unlink(tmp_filename, "stale")

    if (config.mask.mask_type == "random_fully_masked" or config.mask.mask_type == "consecutive_fully_masked"):
        if config.test_data.data_masks is None:
            raise ValueError(f"Mask type {config.mask.mask_type} requires a data_masks directory.")
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
        fill_value=config.mask.fill_value,
        mask_type=mask_type,
        data_masks=data_masks,
        use_sar=config.data.get("use_sar", "mix_closest"),
        keep_all_dates=keep_all_dates,
    )

    meta = ds.s2_meta.copy()
    output_type = meta["dtype"]
    mgrs25_dataloader = DataLoader(
        ds, batch_size=1, shuffle=False, pin_memory=pin_memory,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
        prefetch_factor=1 if num_workers > 0 else None,
        persistent_workers=False,
        timeout=180 if num_workers > 0 else 0,
    )

    if imputation is None:
        imputation = Imputation(
            config_file_train=args.config_file,
            method=args.method,
            mode=args.mode,
            checkpoint=args.checkpoint,
            config_file_test=args.test_data.test_config,
            num_channels=ds.num_channels,
            device=device,
        )
    converter = TypeConverter()
    expected_bands = meta["count"]

    patches_written = 0
    patches_skipped_nodata = 0
    patches_failed = 0
    success = False

    try:
        with rasterio.open(tmp_filename, "w", **meta, **GDAL_OPTIONS) as dst:
            with torch.no_grad():
                for patch_idx, batch_in in enumerate(tqdm(
                    mgrs25_dataloader, leave=False, total=len(ds),
                    desc=f"Patches [{mgrs25}]",
                )):
                    x, y, w, h = (
                        batch_in["window"][0].item(),
                        batch_in["window"][1].item(),
                        batch_in["window"][2].item(),
                        batch_in["window"][3].item(),
                    )

                    # Check nodata on the original (un-masked) S2 target rather than
                    # the masked input: a nodata patch with some cloud-masked pixels
                    # would have fill_value=1 in batch_in["x"], giving a non-zero sum
                    # even though the underlying data is entirely nodata.
                    is_nodata = batch_in["y"][:, :, :10].abs().sum().item() == 0
                    if is_nodata:
                        patches_skipped_nodata += 1
                        del batch_in
                        continue

                    try:
                        batch, y_pred = imputation.impute_sample(batch_in)
                    except RuntimeError as e:
                        print(f"  [GPU ERROR] {mgrs25} patch ({x},{y}): {e}")
                        torch.cuda.empty_cache()
                        patches_failed += 1
                        del batch_in
                        continue

                    try:
                        if keep_all_dates:
                            final_patch = _prepare_patch_keep_all(y_pred, batch, converter, output_type)
                        else:
                            final_patch = _prepare_patch_filtered(y_pred, batch, converter, output_type)
                    except Exception as e:
                        print(f"  [POST-PROCESS ERROR] {mgrs25}: {e}")
                        patches_failed += 1
                        del batch_in, batch, y_pred
                        continue

                    if final_patch.shape[0] != expected_bands:
                        print(
                            f"  [BAND MISMATCH] {mgrs25}: got {final_patch.shape[0]} "
                            f"expected {expected_bands}."
                        )
                        patches_failed += 1
                        del batch_in, batch, y_pred, final_patch
                        continue

                    if _write_patch_with_retry(dst, final_patch, Window(x, y, w, h)):
                        patches_written += 1
                    else:
                        print(f"  [WRITE FAIL] {mgrs25} patch ({x},{y}) after {MAX_WRITE_RETRIES} retries.")
                        patches_failed += 1

                    del batch, batch_in, final_patch, y_pred

                    if patch_idx > 0 and patch_idx % GC_INTERVAL == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            alloc_gb = torch.cuda.memory_allocated() / 1e9
                            reserv_gb = torch.cuda.memory_reserved() / 1e9
                            print(
                                f"  [GPU MEM @{patch_idx}] "
                                f"allocated={alloc_gb:.1f}GB reserved={reserv_gb:.1f}GB",
                                flush=True,
                            )

        if patches_written == 0:
            if patches_skipped_nodata > 0 and patches_failed == 0:
                print(
                    f"[{mgrs25}] WARNING: all {patches_skipped_nodata} patches were nodata "
                    f"— output file contains no predictions."
                )
            elif patches_failed > 0:
                print(
                    f"[{mgrs25}] WARNING: 0 patches written — "
                    f"{patches_failed} failed, {patches_skipped_nodata} nodata-skipped."
                )
            else:
                print(f"[{mgrs25}] WARNING: 0 patches written — dataset appears empty.")

        ok, msg = _validate_tif(tmp_filename)
        if not ok:
            print(f"[{mgrs25}] Post-write validation FAILED: {msg}")
            _safe_unlink(tmp_filename, "invalid")
            return "failed", patches_written, patches_skipped_nodata, patches_failed

        try:
            tmp_filename.replace(out_filename)
            success = True
        except OSError as e:
            print(f"[{mgrs25}] Atomic rename failed ({e}), falling back to copy.")
            try:
                shutil.copy2(str(tmp_filename), str(out_filename))
                ok2, msg2 = _validate_tif(out_filename)
                if ok2:
                    _safe_unlink(tmp_filename, "rename-fallback")
                    success = True
                else:
                    print(f"[{mgrs25}] Copied file invalid: {msg2}")
                    _safe_unlink(out_filename, "bad-copy")
                    _safe_unlink(tmp_filename, "bad-copy")
            except Exception as e2:
                print(f"[{mgrs25}] Fallback copy failed: {e2}")
                _safe_unlink(tmp_filename, "fallback-fail")

    except Exception as e:
        print(f"[{mgrs25}] UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        _safe_unlink(tmp_filename, "crash")
    finally:
        del mgrs25_dataloader
        del ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status = "ok" if success else "failed"
    if success:
        parts = [f"{patches_written} written"]
        if patches_skipped_nodata > 0:
            parts.append(f"{patches_skipped_nodata} nodata-skipped")
        if patches_failed > 0:
            parts.append(f"{patches_failed} failed")
            print(f"[{mgrs25}] COMPLETED WITH WARNINGS: {', '.join(parts)}.")
        else:
            print(f"[{mgrs25}] OK — {', '.join(parts)}.")
        print(f"  -> {out_filename.as_posix()}")
    else:
        print(f"[{mgrs25}] FAILED — output file not produced.")
    print("-" * 60)

    return status, patches_written, patches_skipped_nodata, patches_failed


def main(
    args: argparse.Namespace,
    args_test_data: DictConfig,
):
    _ = torch.set_grad_enabled(False)
    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Cannot find the configuration file used during training: {args.config_file}\n")
    config = config_utils.read_config(args.config_file)

    if "include_S1" in args_test_data:
        if args_test_data.include_S1 is True:
            config.data.use_sar = "mix_closest"
        else:
            config.data.use_sar = False
        args_test_data.pop("include_S1")

    config.data.update(args_test_data)
    config.data.max_seq_length = None

    if "misc" not in config:
        config.misc = OmegaConf.create({"num_workers": 0, "pin_memory": False})
    if "output" not in config:
        raise ValueError(
            "La section 'output' avec 'save_dir' doit être spécifiée dans la configuration d'inférence.\n"
            "Ajoutez :\n  output:\n    save_dir: /chemin/vers/repertoire/sortie"
        )

    # Register SLURM signal handlers: SIGUSR1 is sent by Jean Zay ~60s before the
    # time limit; SIGTERM is sent on preemption. Both set _stop_flag so the loop
    # finishes the current tile cleanly before exiting.
    signal.signal(signal.SIGUSR1, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

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
            print(f"[shared memory fix] TMPDIR -> ${scratch_var}={scratch_dir}")
            break
    else:
        print("[shared memory fix] No scratch dir found, TMPDIR unchanged.")

    image_size = config.misc.get("image_size", [256, 256])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    overlap = config.misc.get("overlap", 0)

    os.environ["GDAL_PAM_ENABLED"] = "NO"
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["GDAL_CACHEMAX"] = "512"

    num_workers = min(config.misc.num_workers, 2)
    print(f"[config] image_size={image_size}, overlap={overlap}, num_workers={num_workers}")
    pin_memory = False  # Pinned memory is non-swappable; not worth it for batch_size=1 inference

    data_optique = Path(config.test_data.get("data_optique", None))
    assert data_optique is not None
    data_radar = Path(config.test_data.get("data_radar", None))
    assert data_radar is not None
    data_masks_val = config.test_data.get("data_masks", None)
    data_masks = Path(data_masks_val) if data_masks_val is not None else None
    output_folder = Path(config.output.save_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    name_experiment = Path(config.test_data.test_config).parent.name
    output_folder_inferences = output_folder / name_experiment
    output_folder_inferences.mkdir(parents=True, exist_ok=True)

    test_tiles_file = Path(config.test_data.get("test_tiles", None))
    assert test_tiles_file is not None
    with open(test_tiles_file, encoding="utf-8") as f:
        test_tiles = json.load(f)

    total = {"ok": 0, "failed": 0, "skipped": 0, "patches_written": 0, "patches_skipped_nodata": 0, "patches_failed": 0}

    # Compute num_channels from config (must match the trained model)
    use_sar = config.data.get("use_sar", False)
    channels = config.data.get("channels", "all")
    num_channels = 10 if channels == "all" else 4
    if use_sar:
        if use_sar == "asc+desc":
            num_channels += 8
        elif use_sar in ("asc", "desc", "mix_closest"):
            num_channels += 4

    # Load model once for all tiles
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shared_imputation = Imputation(
        config_file_train=args.config_file,
        method=args.method,
        mode=args.mode,
        checkpoint=args.checkpoint,
        config_file_test=args.test_data.test_config,
        num_channels=num_channels,
        device=device,
    )

    for mgrs25 in tqdm(test_tiles, desc="MGRS-C areas"):
        if _stop_flag["requested"]:
            print(f"[STOP] Signal received — stopping before tile {mgrs25}.", flush=True)
            break

        status, pw, psn, pf = inference_one_tile(
            args=args,
            mgrs25=mgrs25,
            image_size=image_size,
            config=config,
            pin_memory=pin_memory,
            num_workers=num_workers,
            overlap=overlap,
            data_optique=data_optique,
            data_radar=data_radar,
            data_masks=data_masks,
            output_folder_inferences=output_folder_inferences,
            imputation=shared_imputation,
        )
        total[status] += 1
        total["patches_written"] += pw
        total["patches_skipped_nodata"] += psn
        total["patches_failed"] += pf

        _save_manifest(output_folder_inferences, {
            mgrs25: {
                "status": status,
                "patches_written": pw,
                "patches_skipped_nodata": psn,
                "patches_failed": pf,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        })

    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print(f"  Tiles OK: {total['ok']} | Failed: {total['failed']} | Skipped (existing): {total['skipped']}")
    print(f"  Patches written: {total['patches_written']} | "
          f"Nodata-skipped: {total['patches_skipped_nodata']} | "
          f"Failed: {total['patches_failed']}")
    print("=" * 60)

    config_dump_path = output_folder_inferences / "config_inference.yaml"
    OmegaConf.save(config, config_dump_path)
    print(f"Config saved: {config_dump_path}")


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
