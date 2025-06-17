#!/usr/bin/env python3
"""
Script complet pour explorer l'architecture d'un fichier HDF5 avec Rich
Combine l'exploration générale et les détails des patches S1/S2
"""

from pathlib import Path
from typing import Any, Optional

import h5py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Constantes
BYTES_TO_MB = 1024**2
BYTES_TO_GB = 1024**3


def format_size(size_bytes: int) -> str:
    """Formate une taille en bytes vers une unité lisible"""
    if size_bytes >= BYTES_TO_GB:
        return f"{size_bytes / BYTES_TO_GB:.2f} GB"
    else:
        return f"{size_bytes / BYTES_TO_MB:.2f} MB"


def add_dataset_to_tree(tree: Tree, name: str, obj: h5py.Dataset) -> None:
    """Ajoute un dataset au tree Rich"""
    size_info = format_size(obj.nbytes) if obj.size > 0 else "0 B"

    dataset_text = Text()
    dataset_text.append("📊 ", style="blue")
    dataset_text.append(name.split('/')[-1], style="bold cyan")
    dataset_text.append(f" [{obj.dtype}]", style="dim")

    dataset_node = tree.add(dataset_text)
    dataset_node.add(f"📏 Shape: {obj.shape}")
    dataset_node.add(f"📦 Size: {obj.size:,} elements")
    dataset_node.add(f"💾 Memory: {size_info}")


def add_group_to_tree(tree: Tree, name: str, obj: h5py.Group) -> Tree:
    """Ajoute un groupe au tree Rich et retourne le sous-arbre"""
    group_text = Text()
    group_text.append("📁 ", style="yellow")
    group_text.append(name.split('/')[-1] or "Root", style="bold yellow")
    group_text.append(f" ({len(obj.keys())} items)", style="dim")

    return tree.add(group_text)


def build_file_tree(f: h5py.File, console: Console, max_depth: int = 5) -> Tree:
    """Construit un arbre de la structure du fichier HDF5"""
    tree = Tree("🗂️  Structure du fichier HDF5", style="bold magenta")

    def add_items_to_tree(group: h5py.Group, parent_tree: Tree,
                          path: str = "", depth: int = 0) -> None:
        if depth >= max_depth:
            parent_tree.add(f"[dim]... ({len(group.keys())} items cachés)[/dim]")
            return

        items = list(group.items())

        # Limiter le nombre d'items affichés par niveau
        max_items_map = {0: 2, 1: 3, 2: 2, 3: 4, 4: 2}
        max_items_per_level = max_items_map.get(depth, 2)

        if len(items) > max_items_per_level:
            items = items[:max_items_per_level]
            show_truncated = True
        else:
            show_truncated = False

        for key, obj in items:
            current_path = f"{path}/{key}" if path else key

            if isinstance(obj, h5py.Dataset):
                add_dataset_to_tree(parent_tree, current_path, obj)
            elif isinstance(obj, h5py.Group):
                group_tree = add_group_to_tree(parent_tree, current_path, obj)
                add_items_to_tree(obj, group_tree, current_path, depth + 1)

        if show_truncated:
            remaining = len(list(group.items())) - max_items_per_level
            parent_tree.add(f"[dim]... et {remaining} autres items[/dim]")

    add_items_to_tree(f, tree)
    return tree


def collect_statistics(f: h5py.File, console: Console) -> tuple[int, int, dict, dict]:
    """Collecte les statistiques du fichier HDF5"""
    total_datasets = 0
    total_size = 0
    dataset_types = {}
    max_shape_dims = {}

    def accumulate_stats(name: str, obj: Any) -> None:
        nonlocal total_datasets, total_size, dataset_types, max_shape_dims
        if isinstance(obj, h5py.Dataset):
            total_datasets += 1
            total_size += obj.nbytes

            # Compteur par type de données
            dtype_str = str(obj.dtype)
            dataset_types[dtype_str] = dataset_types.get(dtype_str, 0) + 1

            # Dimensions maximales
            if obj.shape:
                ndims = len(obj.shape)
                if ndims not in max_shape_dims or obj.size > max_shape_dims[ndims][1]:
                    max_shape_dims[ndims] = (obj.shape, obj.size, name)

    with console.status("[bold green]Collecte des statistiques...", spinner="dots"):
        f.visititems(accumulate_stats)

    return total_datasets, total_size, dataset_types, max_shape_dims


def create_statistics_table(total_datasets: int, total_size: int,
                            dataset_types: dict, max_shape_dims: dict) -> Table:
    """Crée un tableau des statistiques"""
    table = Table(title="📊 Statistiques du fichier HDF5",
                  show_header=True, header_style="bold magenta")
    table.add_column("Métrique", style="cyan", no_wrap=True)
    table.add_column("Valeur", style="green")

    # Statistiques générales
    table.add_row("📈 Nombre total de datasets", f"{total_datasets:,}")
    table.add_row("💾 Taille totale", format_size(total_size))
    table.add_row("", "")  # Ligne vide

    # Types de données
    table.add_row("📊 Types de données", "")
    for dtype, count in sorted(dataset_types.items()):
        table.add_row(f"  └─ {dtype}", f"{count} datasets")

    if max_shape_dims:
        table.add_row("", "")  # Ligne vide
        table.add_row("📏 Plus gros datasets", "")
        for ndims, (shape, size, name) in sorted(max_shape_dims.items()):
            table.add_row(f"  └─ {ndims}D", f"{shape} ({size:,} éléments)")
            table.add_row(f"     📂 {name.split('/')[-1]}", "")

    return table


def find_first_patch(f: h5py.File) -> tuple[Optional[str], Optional[h5py.Group]]:
    """Trouve le premier patch disponible"""
    for zone_key in list(f.keys())[:1]:
        zone = f[zone_key]
        for mgrs_key in list(zone.keys())[:1]:
            mgrs = zone[mgrs_key]
            for patch_key in list(mgrs.keys())[:1]:
                patch_path = f"{zone_key}/{mgrs_key}/{patch_key}"
                return patch_path, mgrs[patch_key]
    return None, None


def create_patch_table(patch_path: str, patch: h5py.Group) -> Table:
    """Crée un tableau détaillé pour un patch"""
    patch_name = patch_path.split('/')[-1]
    table = Table(
        title=f"📊 Exemple de patch: {patch_name}",
        show_header=True,
        header_style="bold magenta",
        title_style="bold cyan"
    )
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Shape", style="yellow", width=20)
    table.add_column("Type", style="blue", width=10)
    table.add_column("Taille", style="red", width=10)

    # Datasets S1
    if 'S1' in patch:
        s1_group = patch['S1']
        for i, s1_key in enumerate(list(s1_group.keys())[:1]):  # Premier dataset
            s1_dataset = s1_group[s1_key]
            if isinstance(s1_dataset, h5py.Dataset):
                size_info = format_size(s1_dataset.nbytes) if s1_dataset.size > 0 else "0 B"
                table.add_row(
                    f"🛰️ S1_{s1_key}",
                    str(s1_dataset.shape),
                    str(s1_dataset.dtype),
                    size_info
                )

    # Datasets S2
    if 'S2' in patch:
        s2_group = patch['S2']
        for i, s2_key in enumerate(list(s2_group.keys())[:2]):  # Deux premiers
            s2_dataset = s2_group[s2_key]
            if isinstance(s2_dataset, h5py.Dataset):
                size_info = format_size(s2_dataset.nbytes) if s2_dataset.size > 0 else "0 B"
                table.add_row(
                    f"🌍 S2_{s2_key}",
                    str(s2_dataset.shape),
                    str(s2_dataset.dtype),
                    size_info
                )

    # Métadonnées
    if 'idx_good_frames' in patch:
        meta_dataset = patch['idx_good_frames']
        if isinstance(meta_dataset, h5py.Dataset):
            table.add_row(
                "📋 idx_good_frames",
                str(meta_dataset.shape),
                str(meta_dataset.dtype),
                "< 1MB"
            )

    return table


def get_frame_statistics(patch: h5py.Group) -> str:
    """Calcule les statistiques des frames"""
    good_frames = len(patch['idx_good_frames']) if 'idx_good_frames' in patch else 0
    cloudy_frames = len(patch['idx_cloudy_frames']) if 'idx_cloudy_frames' in patch else 0
    total_frames = good_frames + cloudy_frames
    cloud_ratio = (cloudy_frames / total_frames * 100) if total_frames > 0 else 0

    return f"{good_frames} bonnes | {cloudy_frames} nuageuses ({cloud_ratio:.0f}% nuages)"


def create_summary_table() -> Table:
    """Crée le tableau de résumé des données satellitaires"""
    table = Table(
        title="🛰️ Types de données par satellite",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Satellite", style="cyan", no_wrap=True)
    table.add_column("Datasets typiques", style="green")
    table.add_column("Dimensions attendues", style="yellow")
    table.add_column("Usage", style="blue")

    table.add_row(
        "🛰️ Sentinel-1",
        "S1, S1_dates",
        "(T, 4, H, W)",
        "Données SAR multi-temporelles"
    )
    table.add_row(
        "🌍 Sentinel-2",
        "S2, S2_dates, cloud_mask, cloud_prob",
        "(T, 10, H, W)",
        "Données optiques + masques nuages"
    )

    return table


def display_file_info(console: Console, file_path: str) -> None:
    """Affiche les informations de base du fichier"""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        console.print("❌ [red]Le fichier n'existe pas[/red]")
        return

    file_size = file_path_obj.stat().st_size
    console.print(f"📁 Taille du fichier: [bold green]{format_size(file_size)}[/bold green]")
    console.print()


def display_statistics_section(f: h5py.File, console: Console) -> None:
    """Affiche la section des statistiques"""
    stats = collect_statistics(f, console)
    total_datasets, total_size, dataset_types, max_shape_dims = stats
    stats_table = create_statistics_table(
        total_datasets, total_size, dataset_types, max_shape_dims
    )
    console.print(stats_table)
    console.print()


def display_tree_section(f: h5py.File, console: Console) -> None:
    """Affiche la section de l'arborescence"""
    console.print("🌳 [bold]Structure détaillée (niveaux 1-5):[/bold]")
    with console.status("[bold green]Construction de l'arbre...", spinner="dots"):
        tree = build_file_tree(f, console)
    console.print(tree)


def display_patch_section(f: h5py.File, console: Console) -> None:
    """Affiche la section des détails de patch"""
    patch_path, patch_found = find_first_patch(f)
    if patch_found:
        patch_table = create_patch_table(patch_path, patch_found)
        console.print(patch_table)

        # Statistiques du patch
        frame_stats = get_frame_statistics(patch_found)
        console.print(f"[dim]📈 {frame_stats}[/dim]")

        # Résumé des types de données S1/S2
        console.print("\n📋 [bold]Résumé des données satellitaires:[/bold]")
        summary_table = create_summary_table()
        console.print(summary_table)
    else:
        console.print("❌ [red]Aucun patch trouvé[/red]")


def explore_hdf5_complete(file_path: str, show_tree: bool = True,
                          show_stats: bool = True, show_patch: bool = True) -> None:
    """Explore complètement la structure d'un fichier HDF5 avec Rich"""
    console = Console()

    # Header avec style
    console.print(Panel.fit(
        f"🔍 Exploration complète du fichier HDF5\n[bold cyan]{file_path}[/bold cyan]",
        style="bold blue",
        border_style="blue"
    ))

    try:
        display_file_info(console, file_path)

        with h5py.File(file_path, 'r') as f:
            console.print("✅ [green]Fichier HDF5 ouvert avec succès![/green]")

            # Clés racines
            root_keys = list(f.keys())
            console.print(f"📋 Clés racines: [yellow]{root_keys}[/yellow]")
            console.print()

            # Sections modulaires
            if show_stats:
                display_statistics_section(f, console)

            if show_tree:
                display_tree_section(f, console)

            if show_patch:
                display_patch_section(f, console)

    except PermissionError:
        console.print("❌ [red]Permissions insuffisantes pour lire le fichier[/red]")
    except OSError as e:
        console.print(f"❌ [red]Erreur système: {str(e)}[/red]")
    except Exception as e:
        console.print(f"❌ [red]Erreur inattendue: {str(e)}[/red]")


if __name__ == "__main__":
    file_path = "/home/SPeillet/Downloads/data/toy_circa_ligth_0.5.hdf5"
    
    # Exploration complète
    explore_hdf5_complete(file_path, show_tree=True, show_stats=True, show_patch=True)
    
    # Exemples d'utilisation alternative :
    # explore_hdf5_complete(file_path, show_tree=False, show_stats=True, show_patch=False)
    # explore_hdf5_complete(file_path, show_tree=True, show_stats=False, show_patch=True)
