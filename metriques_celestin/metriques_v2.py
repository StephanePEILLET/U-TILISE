"""
Calcul des métriques de reconstruction nuageuse sur des tuiles inférées.

Architecture :
  - ProcessPoolExecutor : chaque worker lit une tuile entière, filtre les dates
    synthétiques, calcule des statistiques partielles (PixelStats) et les retourne.
  - Process principal : fusionne les PixelStats partielles (addition des
    accumulateurs) puis en déduit les métriques finales exactes.

Avantages vs scripts existants :
  - Accumulation pixel-par-pixel (micro-moyenne exacte, non macro-moyenne de patches).
  - PSNR correct : données normalisées dans [0, 1] avant calcul.
  - R² global sur tous les pixels, pas moyenné par patch.
  - Parallélisme : un worker par tuile (I/O + numpy), sans GIL.

Convention d'encodage du masque nuage dans les rasters GT :
  - cloud ∈ [0, 100]   → probabilité nuageuse originale
  - cloud ∈ [150, 250] → date masquée synthétiquement (+150 ajouté à l'original)
  Détection synthétique : any(cloud > 100) par date.
  Date originalement claire : max(cloud) == 150 (original = 0 partout).
"""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Rend le projet importable depuis ce sous-répertoire
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import rasterio
import torch
import torchgeometry as tgm
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_PIXEL_VALUE = 10_000
N_BANDS = 10
INDICES_4 = [0, 1, 2, 7]  # B2, B3, B4, B8A
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Seuils liés à l'encodage synthétique des masques nuages
SYNTHETIC_CLOUD_THRESHOLD = 100   # cloud > 100 → date masquée synthétiquement
SYNTHETIC_CLOUD_OFFSET = 150      # valeur ajoutée à l'original lors du masquage
# Seuil numérique pour éviter la division par zéro
_EPS = 1e-12

store_dai = Path("/mnt/stores/store_dai")
path_input = store_dai / "tmp/speillet/inferences/v3_combined/consecutive_fully_masked/2026-03-26_16-17"
gt_path_dir = store_dai / "projets/pac/3str/EXP_2/Data_Raster/test_v3/consecutif"


# ── Accumulateur de statistiques pixel par pixel ──────────────────────────────

class PixelStats:
    """
    Accumule les statistiques nécessaires au calcul exact de :
    MAE, MSE, RMSE, PSNR, R², SAM (rad), SSIM.

    Toutes les mises à jour sont O(1) en mémoire.
    Entièrement picklable : peut être retourné par un worker ProcessPoolExecutor.
    """

    __slots__ = ("_sam_n", "_sam_sum", "_ssim_wn", "_ssim_wsum",
                 "_sum_ae", "_sum_se", "_sum_t", "_sum_t2", "n")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n: int = 0
        self._sum_ae: float = 0.0
        self._sum_se: float = 0.0
        self._sum_t: float = 0.0
        self._sum_t2: float = 0.0
        self._sam_sum: float = 0.0
        self._sam_n: int = 0
        self._ssim_wsum: float = 0.0
        self._ssim_wn: int = 0

    def merge(self, other: "PixelStats") -> None:
        """Fusionne les statistiques d'un autre accumulateur dans celui-ci."""
        self.n += other.n
        self._sum_ae += other._sum_ae
        self._sum_se += other._sum_se
        self._sum_t += other._sum_t
        self._sum_t2 += other._sum_t2
        self._sam_sum += other._sam_sum
        self._sam_n += other._sam_n
        self._ssim_wsum += other._ssim_wsum
        self._ssim_wn += other._ssim_wn

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt : tableaux numpy quelconques (aplatis), float32 dans [0, 1]."""
        p = pred.ravel().astype(np.float64)
        t = gt.ravel().astype(np.float64)
        self.n += len(p)
        self._sum_ae += float(np.abs(p - t).sum())
        self._sum_se += float(((p - t) ** 2).sum())
        self._sum_t += float(t.sum())
        self._sum_t2 += float((t ** 2).sum())

    def update_sam(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt : (N, C) float32 dans [0, 1]. Ignore les pixels de norme nulle."""
        dot = (pred * gt).sum(axis=1)
        norm_p = np.linalg.norm(pred, axis=1)
        norm_t = np.linalg.norm(gt, axis=1)
        valid = (norm_p > 0) & (norm_t > 0)
        if valid.any():
            cos = np.clip(dot[valid] / (norm_p[valid] * norm_t[valid]), -1.0, 1.0)
            self._sam_sum += float(np.arccos(cos).sum())
            self._sam_n += int(valid.sum())

    def update_ssim(self, ssim_val: float, n_images: int) -> None:
        self._ssim_wsum += ssim_val * n_images
        self._ssim_wn += n_images

    def compute(self) -> dict:
        if self.n == 0:
            return {k: float("nan") for k in ["mae", "mse", "rmse", "psnr", "r2", "sam", "ssim"]}
        mae = self._sum_ae / self.n
        mse = self._sum_se / self.n
        rmse = math.sqrt(mse)
        psnr = 20.0 * math.log10(1.0 / rmse) if rmse > _EPS else float("inf")
        mean_t = self._sum_t / self.n
        ss_tot = self._sum_t2 - self.n * mean_t ** 2
        r2 = float(1.0 - self._sum_se / ss_tot) if ss_tot > _EPS else float("nan")
        sam = self._sam_sum / self._sam_n if self._sam_n > 0 else float("nan")
        ssim = self._ssim_wsum / self._ssim_wn if self._ssim_wn > 0 else float("nan")
        return {"mae": mae, "mse": mse, "rmse": rmse, "psnr": psnr, "r2": r2, "sam": sam, "ssim": ssim}


# ── Calcul SSIM (instancié une fois par process via un conteneur mutable) ─────

_ssim_cache: dict = {}


def _get_ssim_fn() -> "tgm.losses.SSIM":
    """Instancie le module SSIM une seule fois par process (lazy singleton)."""
    if "fn" not in _ssim_cache:
        _ssim_cache["fn"] = tgm.losses.SSIM(window_size=5, reduction="mean")
    return _ssim_cache["fn"]


@torch.no_grad()
def _compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt : (T, C, H, W) float32 dans [0, 1]."""
    fn = _get_ssim_fn()
    dssim = fn(torch.from_numpy(pred), torch.from_numpy(gt))
    return float(1.0 - 2.0 * dssim)


# ── Filtrage et normalisation d'une tuile brute ───────────────────────────────

def extract_valid_data(
    raw_inference: np.ndarray,
    raw_gt: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """
    Filtre les données brutes d'une tuile pour ne conserver que les dates
    synthétiquement masquées et originalement sans nuages.

    raw_inference, raw_gt : (T*12, H, W) uint16

    Retourne (pred, gt, valid_mask) ou None si aucune date valide.
      pred       : (T_valid, 10, H, W) float32 dans [0, 1]
      gt         : (T_valid, 10, H, W) float32 dans [0, 1]
      valid_mask : (T_valid, H, W) bool — True pour les pixels non-nodata
    """
    h, w = raw_inference.shape[-2], raw_inference.shape[-1]
    inference = raw_inference.reshape(-1, 12, h, w)
    gt_full = raw_gt.reshape(-1, 12, h, w)

    cloud_gt = gt_full[:, 10]  # (T, H, W)

    # 1. Dates synthétiquement masquées (valeur cloud > SYNTHETIC_CLOUD_THRESHOLD)
    idx_masked = np.unique(np.where(cloud_gt > SYNTHETIC_CLOUD_THRESHOLD)[0])
    if len(idx_masked) == 0:
        return None

    inference = inference[idx_masked]
    gt_full = gt_full[idx_masked]
    cloud_gt = cloud_gt[idx_masked]

    # 2. Parmi celles-là, garder uniquement les dates originalement sans nuages
    #    max == SYNTHETIC_CLOUD_OFFSET → original = offset - offset = 0
    max_cloud = np.max(cloud_gt, axis=(1, 2))
    idx_real = np.where(max_cloud == SYNTHETIC_CLOUD_OFFSET)[0]
    if len(idx_real) == 0:
        return None

    pred = inference[idx_real, :10].astype(np.float32) / MAX_PIXEL_VALUE

    gt_raw_bands = gt_full[idx_real, :10]
    # Valeur brute 0 = nodata ; sinon réflectance = (brut - 1000) / 10000
    gt = np.where(
        gt_raw_bands == 0, 0.0, (gt_raw_bands - 1000.0) / MAX_PIXEL_VALUE
    ).astype(np.float32)

    valid_mask = gt_raw_bands[:, 0, :, :] != 0  # (T_valid, H, W)

    return pred, gt, valid_mask


# ── Accumulation des stats à partir de données filtrées ──────────────────────

def _accumulate(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
) -> "tuple[PixelStats, PixelStats, list[PixelStats]]":
    """
    Calcule les PixelStats à partir de tableaux déjà filtrés.

    pred, gt    : (T, 10, H, W) float32 dans [0, 1]
    valid_mask  : (T, H, W) bool
    """
    s_global = PixelStats()
    s_global_4 = PixelStats()
    s_per_band: list[PixelStats] = [PixelStats() for _ in range(N_BANDS)]

    _, C, _, _ = pred.shape
    pred_valid = pred.transpose(0, 2, 3, 1).reshape(-1, C)[valid_mask.ravel()]
    gt_valid = gt.transpose(0, 2, 3, 1).reshape(-1, C)[valid_mask.ravel()]

    if len(pred_valid) == 0:
        return s_global, s_global_4, s_per_band

    s_global.update(pred_valid, gt_valid)
    s_global.update_sam(pred_valid, gt_valid)

    s_global_4.update(pred_valid[:, INDICES_4], gt_valid[:, INDICES_4])
    s_global_4.update_sam(pred_valid[:, INDICES_4], gt_valid[:, INDICES_4])

    for i in range(C):
        s_per_band[i].update(pred_valid[:, i], gt_valid[:, i])

    has_valid = valid_mask.any(axis=(1, 2))
    if has_valid.any():
        _accumulate_ssim(pred, gt, has_valid, s_global, s_global_4, s_per_band, C)

    return s_global, s_global_4, s_per_band


def _accumulate_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    has_valid: np.ndarray,
    s_global: PixelStats,
    s_global_4: PixelStats,
    s_per_band: list[PixelStats],
    n_bands: int,
) -> None:
    n_t = int(has_valid.sum())
    pv, gv = pred[has_valid], gt[has_valid]
    s_global.update_ssim(_compute_ssim(pv, gv), n_t)
    s_global_4.update_ssim(_compute_ssim(pv[:, INDICES_4], gv[:, INDICES_4]), n_t)
    for i in range(n_bands):
        s_per_band[i].update_ssim(_compute_ssim(pv[:, i:i + 1], gv[:, i:i + 1]), n_t)


# ── Worker picklable pour ProcessPoolExecutor ─────────────────────────────────

def _process_tile_worker(
    tif_file: str,
    path_input_str: str,
    gt_path_dir_str: str,
) -> "tuple[PixelStats, PixelStats, list[PixelStats]]":
    """
    Lit une tuile entière, filtre les dates synthétiques et retourne les
    PixelStats partielles. Fonction au niveau module → picklable.
    """
    gt_path_dir_ = Path(gt_path_dir_str)
    tile = tif_file.split("_")[2]
    subtile = "MGRS25-" + "_".join(tif_file.replace(".tif", "").split("_")[2:5])
    gt_path = gt_path_dir_ / tile / subtile / tif_file.replace("pred_mgrsc", "bands_stacked")

    with (
        rasterio.open(Path(path_input_str) / tif_file) as src_inf,
        rasterio.open(gt_path) as src_gt,
    ):
        raw_inf = src_inf.read()
        raw_gt = src_gt.read()

    result = extract_valid_data(raw_inf, raw_gt)
    if result is None:
        return PixelStats(), PixelStats(), [PixelStats() for _ in range(N_BANDS)]

    return _accumulate(*result)


# ── Affichage et sauvegarde des résultats ────────────────────────────────────

_METRICS_ORDER = ["mae", "rmse", "psnr", "ssim", "sam", "r2", "mse"]
_METRIC_FMT: dict[str, str] = {
    "mae": ".4f", "mse": ".6f", "rmse": ".4f",
    "psnr": ".2f", "ssim": ".4f", "sam": ".4f", "r2": ".4f",
}


def _fmt(val: float, metric: str) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "[dim]N/A[/dim]"
    return f"{val:{_METRIC_FMT.get(metric, '.4f')}}"


def _display_all_metrics(
    global_metrics: dict,
    global_4_metrics: dict,
    per_band_metrics: list[dict],
) -> None:
    """Affiche un tableau récapitulatif unique : lignes = sections, colonnes = métriques."""
    console = Console()

    table = Table(
        title="[bold bright_blue]Cloud Reconstruction Metrics[/bold bright_blue]",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )

    table.add_column("Scope", style="cyan", no_wrap=True, justify="right")
    for m in _METRICS_ORDER:
        table.add_column(m.upper(), justify="center", style="white")

    def add_row(label: str, metrics: dict, style: str = "") -> None:
        values = [_fmt(metrics.get(m), m) for m in _METRICS_ORDER]
        table.add_row(label, *values, style=style)

    add_row("All bands", global_metrics, style="bold")
    add_row("4 bands (B2/B3/B4/B8A)", global_4_metrics, style="bold")
    table.add_section()
    for name, m in zip(BAND_NAMES, per_band_metrics, strict=True):
        add_row(name, m)

    console.print(table)


def results_to_dataframe(
    global_metrics: dict,
    global_4_metrics: dict,
    per_band_metrics: list[dict],
) -> pd.DataFrame:
    rows = []
    for label, m in [("all", global_metrics), ("all_4", global_4_metrics)]:
        rows.append({"bande": label, **m})
    for name, m in zip(BAND_NAMES, per_band_metrics, strict=True):
        rows.append({"bande": name, **m})
    return pd.DataFrame(rows)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def _run_parallel(tif_files: list[str]) -> "tuple[PixelStats, PixelStats, list[PixelStats]]":
    """Lance les workers et fusionne les résultats."""
    n_workers = min(os.cpu_count() or 1, 8)
    print(f"Tuiles : {len(tif_files)} | Workers : {n_workers}")

    agg_g = PixelStats()
    agg_g4 = PixelStats()
    agg_pb: list[PixelStats] = [PixelStats() for _ in range(N_BANDS)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_tile_worker, f, str(path_input), str(gt_path_dir)): f
            for f in tif_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tuiles"):
            tif_file = futures[future]
            try:
                s_g, s_g4, s_pb = future.result()
            except FileNotFoundError as exc:
                print(f"  [ABSENT] {tif_file} : {exc}")
                continue
            except Exception as exc:
                print(f"  [ERREUR] {tif_file} : {exc}")
                continue
            agg_g.merge(s_g)
            agg_g4.merge(s_g4)
            for i in range(N_BANDS):
                agg_pb[i].merge(s_pb[i])

    return agg_g, agg_g4, agg_pb


def _print_and_save(
    stats_global: PixelStats,
    stats_global_4: PixelStats,
    stats_per_band: list[PixelStats],
) -> None:
    global_metrics = stats_global.compute()
    global_4_metrics = stats_global_4.compute()
    per_band_metrics = [s.compute() for s in stats_per_band]

    _display_all_metrics(global_metrics, global_4_metrics, per_band_metrics)

    out_csv = Path("resultats_v2.csv")
    results_to_dataframe(global_metrics, global_4_metrics, per_band_metrics).to_csv(out_csv, index=False)
    Console().print(f"\nRésultats sauvegardés dans [green]{out_csv.resolve()}[/green]")


def main() -> None:
    tif_files = [f for f in os.listdir(path_input) if f.endswith(".tif")]
    agg_g, agg_g4, agg_pb = _run_parallel(tif_files)
    _print_and_save(agg_g, agg_g4, agg_pb)


if __name__ == "__main__":
    main()
