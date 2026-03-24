#!/usr/bin/env python3
"""
Agrège les résultats de métriques de tous les checkpoints et modes de masquage
en un rapport lisible avec tableaux.

Deux modes de lecture :
  1. Depuis les fichiers de logs SLURM (par défaut) : parse le dict Python affiché
     après "Statistics:" dans chaque .out
  2. Depuis les fichiers test_stats.json si --json-root est fourni

Usage:
    python aggregate_metrics.py [--logs-dir /chemin/vers/logs] [--output rapport.md]
    python aggregate_metrics.py --json-root /chemin/vers/metrics [--output rapport.md]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_LOGS_DIR = Path("/mnt/stores/store_dai/tmp/speillet/logs")

# Correspondance :  (nom_affichage, dossier_metrics, {mask_mode: job-name du log})
EXPERIMENTS: list[tuple[str, str, dict[str, str]]] = [
    (
        "**ALL_SAR_120_epochs** (mix_closest)",
        "ALL_SAR_120_epochs",
        {
            "random_fully_masked": "metrics_allsar_rfm",
            "consecutive_fully_masked": "metrics_allsar_cfm",
        },
    ),
    (
        "asc+desc, random_clouds, DA",
        "all_bands_sar_asc_desc_random_clouds_da",
        {
            "random_fully_masked": "metrics_asc_desc_rc_da_rfm",
            "consecutive_fully_masked": "metrics_asc_desc_rc_da_cfm",
        },
    ),
    (
        "asc+desc, random_clouds",
        "all_bands_sar_asc_desc_random_clouds",
        {
            "random_fully_masked": "metrics_asc_desc_rc_rfm",
            "consecutive_fully_masked": "metrics_asc_desc_rc_cfm",
        },
    ),
    (
        "asc+desc, random_fully_masked, DA",
        "all_bands_sar_asc_desc_random_fully_masked_da",
        {
            "random_fully_masked": "metrics_asc_desc_rfm_da_rfm",
            "consecutive_fully_masked": "metrics_asc_desc_rfm_da_cfm",
        },
    ),
    (
        "mix_closest, random_fully_masked",
        "all_bands_sar_closest_mix_random_fully_masked",
        {
            "random_fully_masked": "metrics_mix_rfm_rfm",
            "consecutive_fully_masked": "metrics_mix_rfm_cfm",
        },
    ),
    (
        "mix_closest, random_fully_masked, DA",
        "all_bands_sar_closest_mix_random_fully_masked_da",
        {
            "random_fully_masked": "metrics_mix_rfm_da_rfm",
            "consecutive_fully_masked": "metrics_mix_rfm_da_cfm",
        },
    ),
    (
        "coherence_only",
        "coherence_only",
        {
            "random_fully_masked": "metrics_coh_only_rfm",
            "consecutive_fully_masked": "metrics_coh_only_cfm",
        },
    ),
]

MASK_MODES = ["random_fully_masked", "consecutive_fully_masked"]

# Métriques principales à afficher (globales)
MAIN_METRICS = ["mae", "rmse", "psnr", "ssim", "sam", "r2"]

BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


# ─── Chargement des stats ──────────────────────────────────────────────────────


def _find_latest_log(logs_dir: Path, job_prefix: str) -> Path | None:
    """Trouve le fichier .out le plus récent correspondant à un job-name.

    Les fichiers sont nommés : <job-name>-<slurm_job_id>.out
    On prend celui avec le plus grand job_id (= le plus récent).
    """
    pattern = re.compile(rf"^{re.escape(job_prefix)}-(\d+)\.out$")
    candidates: list[tuple[int, Path]] = []
    for f in logs_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_stats_from_log(log_path: Path) -> dict | None:
    """Extrait le dict de stats depuis un fichier de log SLURM.

    Cherche le bloc après 'Statistics:' puis parse le dict Python.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")

    idx = text.rfind("Statistics:")
    if idx == -1:
        return None

    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None

    # Trouver le '}' correspondant au '{' de début
    depth = 0
    brace_end = -1
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break
    if brace_end == -1:
        return None

    dict_str = text[brace_start : brace_end + 1]
    try:
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return None


def load_stats_from_logs(logs_dir: Path, job_prefix: str) -> dict | None:
    """Charge les stats depuis le log le plus récent pour un job donné."""
    log_path = _find_latest_log(logs_dir, job_prefix)
    if log_path is None:
        return None
    return _parse_stats_from_log(log_path)


def load_stats_from_json(json_root: Path, experiment_dir: str, mask_mode: str) -> dict | None:
    """Charge un fichier test_stats.json s'il existe."""
    path = json_root / experiment_dir / mask_mode / "test_stats.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Formatage ──────────────────────────────────────────────────────────────────


def fmt(val: float | None, metric_name: str) -> str:
    """Formate une valeur selon le type de métrique."""
    if val is None:
        return "—"
    if "ssim" in metric_name or "sam" in metric_name or "r2" in metric_name:
        return f"{val:.4f}"
    if "psnr" in metric_name:
        return f"{val:.2f}"
    if "mae" in metric_name or "rmse" in metric_name:
        return f"{val:.1f}"
    if "mse" in metric_name:
        return f"{val:.0f}"
    return f"{val:.4f}"


def _stat_key(metric: str, suffix: str, band: int | None = None) -> str:
    """Construit la clé de stat en tenant compte du préfixe 'images' pour SSIM.

    SSIM utilise 'ssim_images_occluded_input_pixels' alors que les autres
    métriques utilisent '{metric}_occluded_input_pixels'.
    """
    base = f"{metric}_images{suffix}" if "ssim" in metric and suffix else f"{metric}{suffix}"
    if band is not None:
        return f"{base}_band_{band}"
    return base


# ─── Sections du rapport ────────────────────────────────────────────────────────

StatsStore = dict[tuple[str, str], dict | None]  # (exp_dir, mask_mode) -> stats


def _load_all_stats(logs_dir: Path | None, json_root: Path | None) -> StatsStore:
    """Charge toutes les stats et les retourne dans un dict indexé."""
    store: StatsStore = {}
    for _display_name, exp_dir, log_jobs in EXPERIMENTS:
        for mask_mode in MASK_MODES:
            key = (exp_dir, mask_mode)
            stats = None
            if logs_dir is not None:
                job_prefix = log_jobs.get(mask_mode)
                if job_prefix:
                    stats = load_stats_from_logs(logs_dir, job_prefix)
            if stats is None and json_root is not None:
                stats = load_stats_from_json(json_root, exp_dir, mask_mode)
            store[key] = stats
    return store


def _section_global(lines: list[str], store: StatsStore) -> None:
    """Section 1 : tableau récapitulatif global."""
    lines.append("## 1. Récapitulatif Global\n")
    for mask_mode in MASK_MODES:
        mode_label = mask_mode.replace("_", " ").title()
        lines.append(f"### Masquage : {mode_label}\n")
        header = "| Modèle | " + " | ".join(m.upper() for m in MAIN_METRICS) + " |"
        sep = "|:---|" + "|".join(":---:" for _ in MAIN_METRICS) + "|"
        lines.extend([header, sep])
        for display_name, exp_dir, _jobs in EXPERIMENTS:
            stats = store.get((exp_dir, mask_mode))
            vals = [fmt(stats.get(m), m) for m in MAIN_METRICS] if stats else ["—"] * len(MAIN_METRICS)
            lines.append(f"| {display_name} | {' | '.join(vals)} |")
        lines.append("")


def _section_occluded_only(lines: list[str], store: StatsStore) -> None:
    """Section 2 : métriques sur les pixels reconstruits (occluded) uniquement."""
    lines.append("## 2. Métriques sur pixels reconstruits (Occluded)\n")
    sfx = "_occluded_input_pixels"
    for mask_mode in MASK_MODES:
        mode_label = mask_mode.replace("_", " ").title()
        lines.append(f"### Masquage : {mode_label}\n")
        header = "| Modèle | " + " | ".join(m.upper() for m in MAIN_METRICS) + " |"
        sep = "|:---|" + "|".join(":---:" for _ in MAIN_METRICS) + "|"
        lines.extend([header, sep])
        for display_name, exp_dir, _jobs in EXPERIMENTS:
            stats = store.get((exp_dir, mask_mode))
            if stats is None:
                vals = ["—"] * len(MAIN_METRICS)
            else:
                vals = [fmt(stats.get(_stat_key(m, sfx)), m) for m in MAIN_METRICS]
            lines.append(f"| {display_name} | {' | '.join(vals)} |")
        lines.append("")


def _section_occluded_observed(lines: list[str], store: StatsStore) -> None:
    """Section 3 : métriques occluded vs observed."""
    lines.append("## 3. Occluded vs Observed (détail)\n")
    for mask_mode in MASK_MODES:
        mode_label = mask_mode.replace("_", " ").title()
        lines.append(f"### Masquage : {mode_label}\n")
        header = "| Modèle | Type | " + " | ".join(m.upper() for m in MAIN_METRICS) + " |"
        sep = "|:---|:---|" + "|".join(":---:" for _ in MAIN_METRICS) + "|"
        lines.extend([header, sep])
        for display_name, exp_dir, _jobs in EXPERIMENTS:
            stats = store.get((exp_dir, mask_mode))
            for sub_label, suffix in [("Occluded", "_occluded_input_pixels"), ("Observed", "_observed_input_pixels")]:
                if stats is None:
                    vals = ["—"] * len(MAIN_METRICS)
                else:
                    vals = [fmt(stats.get(_stat_key(m, suffix)), m) for m in MAIN_METRICS]
                lines.append(f"| {display_name} | {sub_label} | {' | '.join(vals)} |")
        lines.append("")


def _section_per_band(lines: list[str], store: StatsStore) -> None:
    """Section 3 : métriques par bande spectrale."""
    lines.append("## 4. Métriques par bande spectrale\n")
    for metric in ["ssim", "psnr", "mae"]:
        lines.append(f"### {metric.upper()} par bande\n")
        for mask_mode in MASK_MODES:
            mode_label = mask_mode.replace("_", " ").title()
            lines.append(f"#### Masquage : {mode_label}\n")
            header = "| Modèle | " + " | ".join(BAND_NAMES) + " |"
            sep = "|:---|" + "|".join(":---:" for _ in BAND_NAMES) + "|"
            lines.extend([header, sep])
            for display_name, exp_dir, _jobs in EXPERIMENTS:
                stats = store.get((exp_dir, mask_mode))
                if stats is None:
                    vals = ["—"] * len(BAND_NAMES)
                else:
                    vals = [fmt(stats.get(f"{metric}_band_{i}"), metric) for i in range(10)]
                lines.append(f"| {display_name} | {' | '.join(vals)} |")
            lines.append("")


def _section_availability(lines: list[str], store: StatsStore) -> None:
    """Section 4 : résumé des résultats trouvés / manquants."""
    lines.append("## 5. Disponibilité des résultats\n")
    header = "| Modèle | " + " | ".join(m.replace("_", " ").title() for m in MASK_MODES) + " |"
    sep = "|:---|" + "|".join(":---:" for _ in MASK_MODES) + "|"
    lines.extend([header, sep])
    for display_name, exp_dir, _jobs in EXPERIMENTS:
        statuses = ["✅" if store.get((exp_dir, mm)) is not None else "❌" for mm in MASK_MODES]
        lines.append(f"| {display_name} | {' | '.join(statuses)} |")
    lines.append("")


# ─── Génération du rapport ──────────────────────────────────────────────────────


def generate_report(store: StatsStore, source_label: str) -> str:
    """Génère le rapport complet en Markdown."""
    lines: list[str] = [
        "# Rapport de Métriques — Cloud Reconstruction U-TILISE\n",
        f"**Source des résultats :** {source_label}\n",
    ]
    _section_global(lines, store)
    _section_occluded_only(lines, store)
    _section_occluded_observed(lines, store)
    _section_per_band(lines, store)
    _section_availability(lines, store)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrège les métriques d'évaluation cloud reconstruction")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Répertoire contenant les fichiers de logs SLURM (.out)",
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=None,
        help="Répertoire racine contenant les test_stats.json (fallback si log absent)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rapport_metriques.md"),
        help="Fichier de sortie du rapport (Markdown)",
    )
    args = parser.parse_args()

    # Vérifications
    if not args.logs_dir.exists() and args.json_root is None:
        parser.error(f"Le répertoire de logs {args.logs_dir} n'existe pas et aucun --json-root fourni.")

    logs_dir = args.logs_dir if args.logs_dir.exists() else None
    json_root = args.json_root if args.json_root and args.json_root.exists() else None

    sources = []
    if logs_dir:
        sources.append(f"logs SLURM dans `{logs_dir}`")
    if json_root:
        sources.append(f"JSON dans `{json_root}`")
    source_label = " + ".join(sources)

    # Chargement
    store = _load_all_stats(logs_dir, json_root)

    n_found = sum(1 for v in store.values() if v is not None)
    n_total = len(store)
    print(f"Résultats trouvés : {n_found}/{n_total}")

    # Génération du rapport
    report = generate_report(store, source_label)

    # Afficher sur stdout
    print(report)

    # Sauvegarder
    args.output.write_text(report, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Rapport sauvegardé dans : {args.output.resolve()}")


if __name__ == "__main__":
    main()
