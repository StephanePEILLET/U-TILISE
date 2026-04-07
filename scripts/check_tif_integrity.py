"""
Vérifie l'intégrité de tous les fichiers .tif dans un répertoire d'inférences.

Structure attendue :
    <base_path>/<mode>/<run_timestamp>/*.tif

Modes supportés : mode_produit, mode_consecutif, mode_aleatoire

Usage :
    python scripts/check_tif_integrity.py <base_path> [--workers N] [--verbose]

Exemple :
    python scripts/check_tif_integrity.py /mnt/stores/store_dai/tmp/speillet/inferences/v3_combined
"""

import argparse
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from tqdm import tqdm


def check_single_tif(filepath: Path) -> dict:
    """Vérifie l'intégrité d'un fichier TIFF unique.

    Retourne un dict avec le statut et les erreurs éventuelles.
    """
    result = {
        "filepath": filepath,
        "status": "ok",
        "errors": [],
        "warnings": [],
        "info": {},
    }

    try:
        with rasterio.open(filepath) as ds:
            result["info"]["width"] = ds.width
            result["info"]["height"] = ds.height
            result["info"]["count"] = ds.count
            result["info"]["dtype"] = str(ds.dtypes[0]) if ds.dtypes else None
            result["info"]["crs"] = str(ds.crs) if ds.crs else None

            if ds.width == 0 or ds.height == 0:
                result["status"] = "error"
                result["errors"].append(f"Dimensions nulles : {ds.width}x{ds.height}")

            if ds.count == 0:
                result["status"] = "error"
                result["errors"].append("Aucune bande trouvée")

            for band_idx in range(1, ds.count + 1):
                try:
                    band_tags = ds.tags(band_idx)
                    if band_tags:
                        pass
                except Exception as e:
                    result["warnings"].append(f"Bande {band_idx} : erreur lecture tags - {e}")

            try:
                sample_h = min(64, ds.height)
                sample_w = min(64, ds.width)
                data_sample = ds.read(1, window=((0, sample_h), (0, sample_w)))
                if data_sample is None:
                    result["status"] = "error"
                    result["errors"].append("Bande 1 : read() retourne None")
                elif data_sample.size > 0:
                    sample_min = float(data_sample.min())
                    sample_max = float(data_sample.max())
                    result["info"]["band1_sample_min"] = sample_min
                    result["info"]["band1_sample_max"] = sample_max
                    if sample_min == 0 and sample_max == 0:
                        result["warnings"].append("Bande 1 : échantillon uniformément à 0 (données potentiellement vides)")
            except Exception as e:
                result["status"] = "error"
                result["errors"].append(f"Bande 1 : erreur lecture échantillon - {e}")
                return result

            last_band = ds.count
            if last_band > 1:
                try:
                    ds.read(last_band, window=((0, min(1, ds.height)), (0, min(1, ds.width))))
                except Exception as e:
                    result["status"] = "error"
                    result["errors"].append(f"Dernière bande ({last_band}) : erreur lecture - {e}")

            try:
                full_profile = ds.profile
                if full_profile.get("nodata") is not None:
                    result["info"]["nodata"] = full_profile["nodata"]
                result["info"]["driver"] = full_profile.get("driver", "unknown")
                result["info"]["compress"] = full_profile.get("compress", None)
            except Exception as e:
                result["warnings"].append(f"Profil : {e}")

    except rasterio.errors.RasterioIOError as e:
        result["status"] = "error"
        result["errors"].append(f"RasterioIOError : {e}")
    except OSError as e:
        result["status"] = "error"
        result["errors"].append(f"OSError : {e}")
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"{type(e).__name__} : {e}")
        result["traceback"] = traceback.format_exc()

    return result


def scan_directory(base_path: Path, workers: int = 1) -> list[dict]:
    """Scanne récursivement le répertoire et vérifie tous les .tif trouvés."""
    tif_files = sorted(base_path.rglob("*.tif"))

    if not tif_files:
        print(f"Aucun fichier .tif trouvé dans {base_path}")
        return []

    print(f"Trouvé {len(tif_files)} fichier(s) .tif dans {base_path}")
    print(f"Vérification avec {workers} worker(s)...\n")

    results = []
    if workers <= 1:
        for fp in tqdm(tif_files, desc="Vérification"):
            results.append(check_single_tif(fp))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(check_single_tif, fp): fp for fp in tif_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Vérification"):
                results.append(future.result())

    return results


def group_by_mode(results: list[dict], base_path: Path) -> dict[str, list[dict]]:
    """Regroupe les résultats par mode (sous-répertoire direct de base_path)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        rel = r["filepath"].relative_to(base_path)
        mode = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        grouped[mode].append(r)
    return dict(grouped)


def print_report(results: list[dict], base_path: Path, verbose: bool = False) -> bool:
    """Affiche le rapport de vérification. Retourne True s'il y a des erreurs."""
    grouped = group_by_mode(results, base_path)

    has_errors = False
    total_errors = 0
    total_warnings = 0

    print("=" * 80)
    print("RAPPORT DE VÉRIFICATION D'INTÉGRITÉ TIFF")
    print(f"Répertoire : {base_path}")
    print(f"Fichiers analysés : {len(results)}")
    print("=" * 80)

    for mode, mode_results in sorted(grouped.items()):
        errors_in_mode = [r for r in mode_results if r["status"] == "error"]
        warnings_in_mode = [r for r in mode_results if r["warnings"]]

        print(f"\n--- {mode} ({len(mode_results)} fichiers) ---")

        if not errors_in_mode and not warnings_in_mode:
            print(f"  ✓ Tous les fichiers sont intègres")
            continue

        if errors_in_mode:
            has_errors = True
            total_errors += len(errors_in_mode)
            print(f"\n  ✗ {len(errors_in_mode)} fichier(s) avec des ERREURS :")
            for r in errors_in_mode:
                print(f"\n    Fichier : {r['filepath'].relative_to(base_path)}")
                for err in r["errors"]:
                    print(f"      - {err}")
                if verbose and "traceback" in r:
                    for line in r["traceback"].strip().split("\n"):
                        print(f"        {line}")

        if warnings_in_mode:
            total_warnings += len(warnings_in_mode)
            print(f"\n  ⚠ {len(warnings_in_mode)} fichier(s) avec des AVERTISSEMENTS :")
            for r in warnings_in_mode:
                print(f"\n    Fichier : {r['filepath'].relative_to(base_path)}")
                for w in r["warnings"]:
                    print(f"      - {w}")

        if verbose:
            ok_in_mode = [r for r in mode_results if r["status"] == "ok"]
            if ok_in_mode:
                print(f"\n  ✓ {len(ok_in_mode)} fichier(s) intègres :")
                for r in ok_in_mode:
                    info = r["info"]
                    dims = f"{info.get('width', '?')}x{info.get('height', '?')}"
                    bands = info.get("count", "?")
                    print(f"    {r['filepath'].relative_to(base_path)}  "
                          f"{dims} | {bands} bandes | {info.get('dtype', '?')}")

    print("\n" + "=" * 80)
    print(f"BILAN : {total_errors} erreur(s), {total_warnings} avertissement(s) "
          f"sur {len(results)} fichier(s)")
    if has_errors:
        print("⚠ Des fichiers sont corrompus ou illisibles.")
    else:
        print("✓ Aucune erreur critique détectée.")
    print("=" * 80)

    return has_errors


def main():
    parser = argparse.ArgumentParser(
        description="Vérifie l'intégrité des fichiers .tif d'inférence."
    )
    parser.add_argument(
        "base_path",
        type=Path,
        help="Répertoire racine des inférences (ex: .../v3_combined)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Nombre de workers parallèles (défaut: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les détails des fichiers intègres aussi",
    )
    args = parser.parse_args()

    if not args.base_path.exists():
        print(f"Erreur : le répertoire {args.base_path} n'existe pas.")
        sys.exit(1)

    results = scan_directory(args.base_path, workers=args.workers)
    if not results:
        sys.exit(0)

    has_errors = print_report(results, args.base_path, verbose=args.verbose)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
