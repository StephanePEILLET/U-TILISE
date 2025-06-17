#!/usr/bin/env python3
"""
Generic script to explore HDF5 file architecture with Rich
Displays structure, statistics, and data group details

This tool provides comprehensive analysis of any HDF5 file including:
- Global statistics (datasets count, total size, data types)
- Interactive hierarchical tree structure
- Detailed analysis of representative data groups
- Modern colorful interface with tables, trees, and progress bars
- Modular options for controlling displayed sections

Usage:
    python explore_hdf5_complete.py [file.hdf5]
    python explore_hdf5_complete.py  # uses example file if available

Examples:
    python explore_hdf5_complete.py data.hdf5
    python explore_hdf5_complete.py  # auto-search for *.hdf5 files

Author: Generic HDF5 Explorer
Compatible with: Any HDF5 file (scientific, ML, geospatial, etc.)
"""

import sys
from pathlib import Path
from typing import Any, Optional

import h5py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Constants
BYTES_TO_MB = 1024**2
BYTES_TO_GB = 1024**3
MAX_DISPLAYED_ITEMS = 10  # Display limit to avoid overload
MAX_SEARCH_DEPTH = 3      # Maximum depth for group search
MAX_SEARCH_ITEMS = 3      # Max items to examine per level

# Icon selection constants
ONE_DIMENSION = 1
TWO_DIMENSIONS = 2
THREE_OR_MORE_DIMENSIONS = 3


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable units.

    Converts raw byte values to appropriate units (MB/GB) for better readability.
    Uses binary units (1024-based) for accurate storage representation.

    Args:
        size_bytes (int): Size in bytes to format

    Returns:
        str: Formatted size string with units (e.g., "1.23 GB", "456.78 MB")
    """
    if size_bytes >= BYTES_TO_GB:
        return f"{size_bytes / BYTES_TO_GB:.2f} GB"
    else:
        return f"{size_bytes / BYTES_TO_MB:.2f} MB"


def add_dataset_to_tree(tree: Tree, name: str, obj: h5py.Dataset) -> None:
    """Add a dataset to the Rich tree structure.

    Creates a visually formatted tree node representing an HDF5 dataset with
    metadata including shape, size, data type, and memory usage. Uses icons
    and colors for better visual distinction.

    Args:
        tree (Tree): The Rich Tree object to add the dataset node to
        name (str): Full path name of the dataset within the HDF5 file
        obj (h5py.Dataset): The HDF5 dataset object containing the data

    Returns:
        None: Modifies the tree object in-place by adding a new dataset node
    """
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
    """Add a group to the Rich tree structure and return the sub-tree.

    Creates a formatted tree node representing an HDF5 group with visual
    styling including folder icon, group name, and item count. Returns
    the created sub-tree for further population with child elements.

    Args:
        tree (Tree): The Rich Tree object to add the group node to
        name (str): Full path name of the group within the HDF5 file
        obj (h5py.Group): The HDF5 group object containing sub-items

    Returns:
        Tree: The newly created sub-tree node for this group
    """
    group_text = Text()
    group_text.append("📁 ", style="yellow")
    group_text.append(name.split('/')[-1] or "Root", style="bold yellow")
    group_text.append(f" ({len(obj.keys())} items)", style="dim")

    return tree.add(group_text)


def build_file_tree(f: h5py.File, console: Console, max_depth: int = 5) -> Tree:
    """Build a tree representation of the HDF5 file structure.

    Creates a hierarchical Rich Tree showing the complete organization of
    groups and datasets within an HDF5 file. Includes depth limiting and
    item count restrictions to prevent overwhelming output for large files.

    Args:
        f (h5py.File): The opened HDF5 file object to explore
        console (Console): Rich Console object for status updates during tree building
        max_depth (int, optional): Maximum depth to traverse in the hierarchy. Defaults to 5.

    Returns:
        Tree: Rich Tree object representing the complete file structure

    Note:
        Uses MAX_DISPLAYED_ITEMS constant to limit items shown per level.
        Shows truncation indicators when content is limited for readability.
    """
    tree = Tree("🗂️  HDF5 File Structure", style="bold magenta")

    def add_items_to_tree(group: h5py.Group, parent_tree: Tree,
                          path: str = "", depth: int = 0) -> None:
        if depth >= max_depth:
            parent_tree.add(f"[dim]... ({len(group.keys())} items hidden)[/dim]")
            return

        items = list(group.items())

        # Limit number of items displayed per level
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
            parent_tree.add(f"[dim]... and {remaining} other items[/dim]")

    add_items_to_tree(f, tree)
    return tree


def collect_statistics(f: h5py.File, console: Console) -> tuple[int, int, dict, dict]:
    """Collect comprehensive statistics from the HDF5 file.

    Traverses the entire file structure to gather dataset statistics including
    total count, size, data types distribution, and maximum shape dimensions.
    Shows progress indicator during the traversal process.

    Args:
        f (h5py.File): The opened HDF5 file object to analyze
        console (Console): Rich Console object for displaying progress status

    Returns:
        tuple[int, int, dict, dict]: A 4-tuple containing:
            - total_datasets (int): Total number of datasets found
            - total_size (int): Combined size of all datasets in bytes
            - dataset_types (dict): Count of datasets per data type
            - max_shape_dims (dict): Largest dataset per dimensionality

    Note:
        Uses visititems() for efficient recursive traversal of the file.
        Progress is shown with a spinning indicator during processing.
    """
    total_datasets = 0
    total_size = 0
    dataset_types = {}
    max_shape_dims = {}

    def accumulate_stats(name: str, obj: Any) -> None:
        nonlocal total_datasets, total_size, dataset_types, max_shape_dims
        if isinstance(obj, h5py.Dataset):
            total_datasets += 1
            total_size += obj.nbytes

            # Counter by data type
            dtype_str = str(obj.dtype)
            dataset_types[dtype_str] = dataset_types.get(dtype_str, 0) + 1

            # Maximum dimensions
            if obj.shape:
                ndims = len(obj.shape)
                if ndims not in max_shape_dims or obj.size > max_shape_dims[ndims][1]:
                    max_shape_dims[ndims] = (obj.shape, obj.size, name)

    with console.status("[bold green]Collecting statistics...", spinner="dots"):
        f.visititems(accumulate_stats)

    return total_datasets, total_size, dataset_types, max_shape_dims


def create_statistics_table(total_datasets: int, total_size: int,
                            dataset_types: dict, max_shape_dims: dict) -> Table:
    """Create a formatted statistics table for HDF5 file analysis.

    Generates a comprehensive Rich Table displaying file statistics including
    dataset counts, total size, data type distribution, and largest datasets
    organized by dimensionality.

    Args:
        total_datasets (int): Total number of datasets in the file
        total_size (int): Combined size of all datasets in bytes
        dataset_types (dict): Dictionary mapping data types to their counts
        max_shape_dims (dict): Dictionary mapping dimensionality to largest dataset info

    Returns:
        Table: Rich Table object with formatted statistics ready for display

    Note:
        Includes sections for general stats, data types breakdown, and largest datasets.
        Uses format_size() for human-readable size representation.
    """
    table = Table(title="📊 HDF5 File Statistics",
                  show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # General statistics
    table.add_row("📈 Total datasets", f"{total_datasets:,}")
    table.add_row("💾 Total size", format_size(total_size))
    table.add_row("", "")  # Empty line

    # Data types
    table.add_row("📊 Data types", "")
    for dtype, count in sorted(dataset_types.items()):
        table.add_row(f"  └─ {dtype}", f"{count} datasets")

    if max_shape_dims:
        table.add_row("", "")  # Empty line
        table.add_row("📏 Largest datasets", "")
        for ndims, (shape, size, name) in sorted(max_shape_dims.items()):
            table.add_row(f"  └─ {ndims}D", f"{shape} ({size:,} elements)")
            table.add_row(f"     📂 {name.split('/')[-1]}", "")

    return table


def find_first_data_group(f: h5py.File) -> tuple[Optional[str], Optional[h5py.Group]]:
    """Find the first available data group with hierarchical structure.

    Recursively searches through the HDF5 file hierarchy to locate the first
    group that contains datasets, providing a representative sample for
    detailed analysis. Uses depth and item limits to prevent excessive traversal.

    Args:
        f (h5py.File): The opened HDF5 file object to search through

    Returns:
        tuple[Optional[str], Optional[h5py.Group]]: A 2-tuple containing:
            - group_path (str | None): Full path to the found group, or None if not found
            - group (h5py.Group | None): The group object itself, or None if not found

    Note:
        Stops at first group containing datasets rather than searching entire tree.
        Uses MAX_SEARCH_DEPTH and MAX_SEARCH_ITEMS constants for performance.
    """
    def find_deepest_group(group, path="", max_depth=MAX_SEARCH_DEPTH):
        """Recursively find the deepest group containing datasets"""
        if max_depth <= 0:
            return None, None

        for key, obj in list(group.items())[:MAX_SEARCH_ITEMS]:
            current_path = f"{path}/{key}" if path else key

            if isinstance(obj, h5py.Group):
                # If this group contains datasets, return it
                has_datasets = any(isinstance(item, h5py.Dataset) for item in obj.values())
                if has_datasets:
                    return current_path, obj

                # Otherwise continue recursively
                result_path, result_group = find_deepest_group(obj, current_path, max_depth - 1)
                if result_group is not None:
                    return result_path, result_group

        return None, None

    return find_deepest_group(f)


def create_data_group_table(group_path: str, group: h5py.Group) -> Table:
    """Create a detailed table for a data group.

    Generates a comprehensive Rich Table showing all datasets and subgroups
    within a specific HDF5 group, including shape, data type, and size information.
    Limits output to prevent overwhelming display for large groups.

    Args:
        group_path (str): Full path to the group within the HDF5 file
        group (h5py.Group): The HDF5 group object to analyze

    Returns:
        Table: Rich Table object containing detailed group contents

    Note:
        Shows icons based on dimensionality (📈 for 1D, 🗂️ for 2D, 📊 for 3D+).
        Limits display to MAX_DISPLAYED_ITEMS to prevent output overflow.
        Includes truncation indicator when group contains more items.
    """
    group_name = group_path.split('/')[-1]
    table = Table(
        title=f"📊 Example group: {group_name}",
        show_header=True,
        header_style="bold magenta",
        title_style="bold cyan"
    )
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Shape", style="yellow", width=20)
    table.add_column("Type", style="blue", width=10)
    table.add_column("Size", style="red", width=10)

    # Display all datasets in the group (limited to avoid overload)
    dataset_count = 0
    for key, obj in group.items():
        if isinstance(obj, h5py.Dataset) and dataset_count < MAX_DISPLAYED_ITEMS:
            size_info = format_size(obj.nbytes) if obj.size > 0 else "0 B"
            # Icon based on dimensionality
            if len(obj.shape) == ONE_DIMENSION:
                icon = "📈"
            elif len(obj.shape) == TWO_DIMENSIONS:
                icon = "🗂️"
            elif len(obj.shape) >= THREE_OR_MORE_DIMENSIONS:
                icon = "📊"
            else:
                icon = "📄"

            table.add_row(
                f"{icon} {key}",
                str(obj.shape),
                str(obj.dtype),
                size_info
            )
            dataset_count += 1
        elif isinstance(obj, h5py.Group) and dataset_count < MAX_DISPLAYED_ITEMS:
            # Display subgroups too
            table.add_row(
                f"📁 {key}",
                f"Group ({len(obj.keys())} items)",
                "group",
                "-"
            )
            dataset_count += 1

    if dataset_count == MAX_DISPLAYED_ITEMS and len(group.keys()) > MAX_DISPLAYED_ITEMS:
        remaining = len(group.keys()) - MAX_DISPLAYED_ITEMS
        table.add_row("...", f"+ {remaining} others", "...", "...")

    return table


def get_group_statistics(group: h5py.Group) -> str:
    """Calculate general statistics for a group.

    Computes and formats summary statistics for an HDF5 group including
    counts of datasets and subgroups contained within it.

    Args:
        group (h5py.Group): The HDF5 group object to analyze

    Returns:
        str: Formatted statistics string describing group contents

    Note:
        Returns "Empty group" for groups with no contents.
        Statistics are formatted for human readability.
    """
    dataset_count = sum(1 for obj in group.values() if isinstance(obj, h5py.Dataset))
    subgroup_count = sum(1 for obj in group.values() if isinstance(obj, h5py.Group))

    stats = []
    if dataset_count > 0:
        stats.append(f"{dataset_count} datasets")
    if subgroup_count > 0:
        stats.append(f"{subgroup_count} subgroups")

    return " | ".join(stats) if stats else "Empty group"


def create_summary_table() -> Table:
    """Create a generic summary table of data types"""
    table = Table(
        title="📋 Data Type Summary",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Typical Usage", style="blue")

    table.add_row(
        "📊 Multidimensional datasets",
        "Data arrays (3D, 4D+)",
        "Temporal data, images, matrices"
    )
    table.add_row(
        "📈 1D/2D datasets",
        "Vectors and matrices",
        "Time series, metadata"
    )
    table.add_row(
        "📁 Hierarchical groups",
        "Data organization",
        "Logical structure, categorization"
    )

    return table


def display_file_info(console: Console, file_path: str) -> None:
    """Display basic file information"""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        console.print("❌ [red]File does not exist[/red]")
        return

    file_size = file_path_obj.stat().st_size
    console.print(f"📁 File size: [bold green]{format_size(file_size)}[/bold green]")
    console.print()


def display_statistics_section(f: h5py.File, console: Console) -> None:
    """Display the statistics section"""
    stats = collect_statistics(f, console)
    total_datasets, total_size, dataset_types, max_shape_dims = stats
    stats_table = create_statistics_table(
        total_datasets, total_size, dataset_types, max_shape_dims
    )
    console.print(stats_table)
    console.print()


def display_tree_section(f: h5py.File, console: Console) -> None:
    """Display the tree section"""
    console.print("🌳 [bold]Detailed structure (levels 1-5):[/bold]")
    with console.status("[bold green]Building tree...", spinner="dots"):
        tree = build_file_tree(f, console)
    console.print(tree)


def display_data_group_section(f: h5py.File, console: Console) -> None:
    """Display the data group details section"""
    group_path, group_found = find_first_data_group(f)
    if group_found:
        group_table = create_data_group_table(group_path, group_found)
        console.print(group_table)

        # Group statistics
        group_stats = get_group_statistics(group_found)
        console.print(f"[dim]📈 {group_stats}[/dim]")

        # Generic summary of data types
        console.print("\n📋 [bold]Data type summary:[/bold]")
        summary_table = create_summary_table()
        console.print(summary_table)
    else:
        console.print("❌ [red]No data group found[/red]")


def explore_hdf5_complete(file_path: str, show_tree: bool = True,
                          show_stats: bool = True, show_data_group: bool = True) -> None:
    """Complete exploration of HDF5 file structure with Rich interface.

    Main function providing comprehensive analysis of any HDF5 file including
    statistics, hierarchical tree view, and detailed data group inspection.
    Uses modular approach allowing selective display of different sections.

    Args:
        file_path (str): Path to the HDF5 file to analyze
        show_tree (bool, optional): Whether to display hierarchical tree structure. Defaults to True.
        show_stats (bool, optional): Whether to display file statistics. Defaults to True.
        show_data_group (bool, optional): Whether to show detailed data group analysis. Defaults to True.

    Returns:
        None: Outputs analysis directly to console using Rich formatting

    Raises:
        PermissionError: When insufficient permissions to read the file
        OSError: For system-level file access errors
        Exception: For unexpected errors during analysis

    Note:
        Provides error handling for common file access issues.
        Uses Rich panels, tables, and trees for professional output formatting.
    """
    console = Console()

    # Header with style
    console.print(Panel.fit(
        f"🔍 Complete HDF5 File Exploration\n[bold cyan]{file_path}[/bold cyan]",
        style="bold blue",
        border_style="blue"
    ))

    try:
        display_file_info(console, file_path)

        with h5py.File(file_path, 'r') as f:
            console.print("✅ [green]HDF5 file opened successfully![/green]")

            # Root keys
            root_keys = list(f.keys())
            console.print(f"📋 Root keys: [yellow]{root_keys}[/yellow]")
            console.print()

            # Modular sections
            if show_stats:
                display_statistics_section(f, console)

            if show_tree:
                display_tree_section(f, console)

            if show_data_group:
                display_data_group_section(f, console)

    except PermissionError:
        console.print("❌ [red]Insufficient permissions to read file[/red]")
    except OSError as e:
        console.print(f"❌ [red]System error: {str(e)}[/red]")
    except Exception as e:
        console.print(f"❌ [red]Unexpected error: {str(e)}[/red]")


def get_file_path() -> str:
    """Determine the HDF5 file to analyze from arguments or examples.

    Automatically resolves the target HDF5 file by checking command line arguments
    first, then searching common directories for example files. Provides helpful
    error messages when no suitable file is found.

    Returns:
        str: Path to the HDF5 file to analyze

    Raises:
        SystemExit: When file specified in arguments doesn't exist or no files found

    Note:
        Searches directories: ./data/, ./examples/, ./, ../data/, ./test_data/
        Exits with error message if no HDF5 files are found anywhere.
    """
    if len(sys.argv) > 1:
        # Use file provided as argument
        file_path = sys.argv[1]
        if not Path(file_path).exists():
            print(f"❌ Error: File '{file_path}' does not exist")
            sys.exit(1)
        return file_path

    # Search for example HDF5 files in common directories
    example_paths = [
        "./data/*.hdf5",
        "./examples/*.hdf5",
        "*.hdf5",
        "../data/*.hdf5",
        "./test_data/*.hdf5"
    ]

    from glob import glob
    for pattern in example_paths:
        files = glob(pattern)
        if files:
            return files[0]  # First file found

    # If no file found, ask user
    print("❌ No HDF5 file specified")
    print("Usage: python explore_hdf5_complete.py <file.hdf5>")
    print("   or: place a .hdf5 file in the current directory")
    sys.exit(1)


if __name__ == "__main__":
    # Determine file to analyze
    file_path = get_file_path()

    print(f"🔍 Analyzing file: {file_path}")

    # Complete exploration
    explore_hdf5_complete(file_path, show_tree=True, show_stats=True, show_data_group=True)

    # Alternative usage examples:
    # explore_hdf5_complete(file_path, show_tree=False, show_stats=True, show_data_group=False)
    # explore_hdf5_complete(file_path, show_tree=True, show_stats=False, show_data_group=True)
