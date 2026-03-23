#!/usr/bin/env python3
"""
Agrège les résultats de métriques de tous les checkpoints et modes de masquage
en un rapport lisible avec tableaux.

Usage:
    python aggregate_metrics.py [--metrics-root /chemin/vers/metrics] [--output rapport_metriques.md]

Par défaut, cherche dans /mnt/common/hdd/home/SPeillet/outputs/U-TILISE/metrics/
"""

import argparse
import json
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_METRICS_ROOT = Path("/mnt/common/hdd/home/SPeillet/outputs/U-TILISE/metrics")

# Checkpoints à scanner : (nom_affichage, sous-dossier dans metrics/)
EXPERIMENTS = [
    ("asc+desc, random_clouds, DA", "all_bands_sar_asc_desc_random_clouds_da"),
    ("asc+desc, random_clouds", "all_bands_sar_asc_desc_random_clouds"),
    ("asc+desc, random_fully_masked, DA", "all_bands_sar_asc_desc_random_fully_masked_da"),
    ("mix_closest, random_fully_masked", "all_bands_sar_closest_mix_random_fully_masked"),
    ("mix_closest, random_fully_masked, DA", "all_bands_sar_closest_mix_random_fully_masked_da"),
    ("coherence_only", "coherence_only"),
]

MASK_MODES = ["random_fully_masked", "consecutive_fully_masked"]

# Métriques principales à afficher (globales)
MAIN_METRICS = ["mae", "rmse", "psnr", "ssim", "sam", "r2"]

# Suffixes pour les sous-catégories
SUFFIXES = {
    "global": "",
    "occluded": "_occluded_input_pixels",
    "observed": "_observed_input_pixels",
}

BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def load_stats(metrics_root: Path, experiment_dir: str, mask_mode: str) -> dict | None:
    """Charge un fichier test_stats.json s'il existe."""
    path = metrics_root / experiment_dir / mask_mode / "test_stats.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(val, metric_name: str) -> str:
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


def _section_global(lines: list[str], metrics_root: Path) -> None:
    """Section 1 : tableau récapitulatif global."""
    lines.append("## 1. Récapitulatif Global\n")
    for mask_mode in MASK_MODES:
        mode_label = mask_mode.replace("_", " ").title()
        lines.append(f"### Masquage : {mode_label}\n")
        header = "| Modèle | " + " | ".join(m.upper() for m in MAIN_METRICS) + " |"
        sep = "|:---|" + "|".join(":---:" for _ in MAIN_METRICS) + "|"
        lines.extend([header, sep])
        for display_name, exp_dir in EXPERIMENTS:
            stats = load_stats(metrics_root, exp_dir, mask_mode)
            vals = [fmt(stats.get(m), m) for m in MAIN_METRICS] if stats else ["—"] * len(MAIN_METRICS)
            lines.append(f"| {display_name} | {' | '.join(vals)} |")
        lines.append("")


def _section_occluded_observed(lines: list[str], metrics_root: Path) -> None:
    """Section 2 : métriques occluded vs observed."""
    lines.append("## 2. Occluded vs Observed\n")
    for mask_mode in MASK_MODES:
        mode_label = mask_mode.replace("_", " ").title()
        lines.append(f"### Masquage : {mode_label}\n")
        header = "| Modèle | Type |" + " | ".join(m.upper() for m in MAIN_METRICS) + " |"
        sep = "|:---|:---|" + "|".join(":---:" for _ in MAIN_METRICS) + "|"
        lines.extend([header, sep])
        for display_name, exp_dir in EXPERIMENTS:
            stats = load_stats(metrics_root, exp_dir, mask_mode)
            for sub_label, suffix in [("Occluded", "_occluded_input_pixels"), ("Observed", "_observed_input_pixels")]:
                if stats is None:
                    vals = ["—"] * len(MAIN_METRICS)
                else:
                    vals = [fmt(stats.get(f"{m}{suffix}"), m) for m in MAIN_METRICS]
                lines.append(f"| {display_name} | {sub_label} | {' | '.join(vals)} |")
        lines.append("")


def _section_per_band(lines: list[str], metrics_root: Path) -> None:
    """Section 3 : métriques par bande spectrale."""
    lines.append("## 3. Métriques par bande spectrale\n")
    for metric in ["ssim", "psnr", "mae"]:
        lines.append(f"### {metric.upper()} par bande\n")
        for mask_mode in MASK_MODES:
            mode_label = mask_mode.replace("_", " ").title()
            lines.append(f"#### Masquage : {mode_label}\n")
            header = "| Modèle | " + " | ".join(BAND_NAMES) + " |"
            sep = "|:---|" + "|".join(":---:" for _ in BAND_NAMES) + "|"
            lines.extend([header, sep])
            for display_name, exp_dir in EXPERIMENTS:
                stats = load_stats(metrics_root, exp_dir, mask_mode)
                if stats is None:
                    vals = ["—"] * len(BAND_NAMES)
                else:
                    vals = [fmt(stats.get(f"{metric}_band_{i}"), metric) for i in range(10)]
                lines.append(f"| {display_name} | {' | '.join(vals)} |")
            lines.append("")


def _section_availability(lines: list[str], metrics_root: Path) -> None:
    """Section 4 : résumé des fichiers trouvés / manquants."""
    lines.append("## 4. Disponibilité des résultats\n")
    header = "| Modèle | " + " | ".join(m.replace("_", " ").title() for m in MASK_MODES) + " |"
    sep = "|:---|" + "|".join(":---:" for _ in MASK_MODES) + "|"
    lines.extend([header, sep])
    for display_name, exp_dir in EXPERIMENTS:
        statuses = [
            "✅" if (metrics_root / exp_dir / mm / "test_stats.json").exists() else "❌"
            for mm in MASK_MODES
        ]
        lines.append(f"| {display_name} | {' | '.join(statuses)} |")
    lines.append("")


def generate_report(metrics_root: Path) -> str:
    """Génère le rapport complet en Markdown."""
    lines: list[str] = [
        "# Rapport de Métriques — Cloud Reconstruction U-TILISE\n",
        f"**Source des résultats :** `{metrics_root}`\n",
    ]
    _section_global(lines, metrics_root)
    _section_occluded_observed(lines, metrics_root)
    _section_per_band(lines, metrics_root)
    _section_availability(lines, metrics_root)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Agrège les métriques d'évaluation cloud reconstruction")
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help="Répertoire racine contenant les sous-dossiers de métriques",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rapport_metriques.md"),
        help="Fichier de sortie du rapport (Markdown)",
    )
    args = parser.parse_args()

    report = generate_report(args.metrics_root)

    # Afficher sur stdout
    print(report)

    # Sauvegarder
    args.output.write_text(report, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Rapport sauvegardé dans : {args.output.resolve()}")


if __name__ == "__main__":
    main()
