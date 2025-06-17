#!/usr/bin/env python3
"""
Script to compute comprehensive statistics on S1/S2 data from HDF5 files
Uses the compute_stats.py module for histogram analysis

This script analyzes S1/S2 tensor data with:
- Basic statistics (mean, std, min, max, percentiles)
- Full-range histogram analysis for uint16 data
- Band-wise statistics for multispectral data
- Temporal statistics across time series
- Rich console output with tables and progress bars

Usage:
    python analyze_s2_stats.py <file.hdf5> --key-path S2/S2
    python analyze_s2_stats.py <file.hdf5> --key-path S1/S1

Author: S1/S2 Statistics Analyzer
Compatible with: CIRCA/UTILISE HDF5 datasets
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# Import Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

NUM_DIMS = 4


# Local implementation of compute_stats functions to avoid import issues
def compute_tensor_histogram(
    tensor: np.ndarray,
    n_bins: int = 100,
    range_min: int = -32768,
    range_max: int = 32767,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of values in a tensor within specified range."""
    # Flatten the tensor
    flat_tensor = tensor.reshape(-1)

    # Compute histogram
    hist, bin_edges = np.histogram(
        flat_tensor,
        bins=n_bins,
        range=(range_min, range_max)
    )
    return hist, bin_edges


def set_n_bins():
    """Set the number of bins for histogram computation based on int16 range."""
    n_bins = 65536  # Full range for uint16 data
    range_min = np.iinfo(np.int16).min
    range_max = np.iinfo(np.int16).max
    return n_bins, range_min, range_max


console = Console()

# Constants for efficient processing
EXPECTED_S2_BANDS = 13  # Sentinel-2 has 13 bands
HISTOGRAM_BINS = 65536  # Full range for uint16 data
SAMPLE_SIZE_THRESHOLD = 50_000_000  # 50M elements threshold for sampling

# Band labels for display
BAND_LABELS_DICT = {
    "S2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B11", "B12"],
    "S1": ["sigma VV", "sigma VH", "coherence VV", "coherence VH"]
}
CURRENT_BAND_LABELS = []


def get_band_label(band_idx: int) -> str:
    """Return the human-readable label for a given band index.

    Uses the globally-set CURRENT_BAND_LABELS.
    Falls back to a formatted index if the label list is incomplete.
    """
    if 0 <= band_idx < len(CURRENT_BAND_LABELS):
        return CURRENT_BAND_LABELS[band_idx]
    return f"Band {band_idx:02d}"


def accumulate_band_histograms(datasets: list[tuple[str, h5py.Dataset]]) -> tuple[np.ndarray, int]:
    """
    Accumulate histograms across all S2 datasets for each band.
    
    Args:
        datasets: List of (path, dataset) tuples

    Returns:
        Tuple of (accumulated_histograms, total_pixels) where:
        - accumulated_histograms: array of shape (n_bands, n_bins)
        - total_pixels: total number of pixels processed per band
    """

    # Get optimal histogram parameters
    n_bins, range_min, range_max = set_n_bins()

    # Initialize accumulated histograms for each band
    accumulated_histograms = None
    total_pixels = 0
    n_bands = None

    for _, dataset in track(
        datasets, description="Processing datasets and accumulating histograms..."
    ):
        # Load data
        data = dataset[:]

        # Verify 4D structure (T, C, H, W)
        if len(data.shape) != NUM_DIMS:
            continue

        T, C, H, W = data.shape

        # Initialize histograms on first dataset
        if accumulated_histograms is None:
            n_bands = C
            accumulated_histograms = np.zeros((n_bands, n_bins), dtype=np.int64)

        # Process each band separately
        for band_idx in range(n_bands):
            band_data = data[:, band_idx, :, :].reshape(-1)  # Flatten T×H×W

            # Compute histogram for this band
            hist, _ = compute_tensor_histogram(
                band_data,
                n_bins=n_bins,
                range_min=range_min,
                range_max=range_max
            )

            # Accumulate histogram
            accumulated_histograms[band_idx] += hist

        # Update total pixel count (per band)
        total_pixels += T * H * W

    return accumulated_histograms, total_pixels


def compute_global_band_statistics(accumulated_histograms: np.ndarray, total_pixels: int) -> dict[str, Any]:
    """
    Compute global mean and std for each band from accumulated histograms.
    
    Args:
        accumulated_histograms: Array of shape (n_bands, n_bins)
        total_pixels: Total number of pixels per band
        
    Returns:
        Dictionary with global statistics per band
    """
    n_bands, n_bins = accumulated_histograms.shape

    # Get bin centers
    _, range_min, range_max = set_n_bins()
    bin_edges = np.linspace(range_min, range_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    band_stats = []

    for band_idx in range(n_bands):
        hist = accumulated_histograms[band_idx]

        # Normalize histogram to get probability distribution
        total_counts = np.sum(hist)
        if total_counts == 0:
            console.print(f"⚠️ Band {band_idx}: No data found")
            continue

        prob_dist = hist / total_counts

        # Compute mean from histogram
        mean = np.sum(bin_centers * prob_dist)

        # Compute variance and std from histogram
        variance = np.sum(((bin_centers - mean) ** 2) * prob_dist)
        std = np.sqrt(variance)

        # Additional statistics
        non_zero_bins = np.count_nonzero(hist)
        coverage = non_zero_bins / n_bins * 100

        # Find actual min/max values (non-zero bins)
        non_zero_indices = np.where(hist > 0)[0]
        if len(non_zero_indices) > 0:
            actual_min = bin_centers[non_zero_indices[0]]
            actual_max = bin_centers[non_zero_indices[-1]]
        else:
            actual_min = actual_max = 0

        band_stats.append({
            'band_index': band_idx,
            'mean': float(mean),
            'std': float(std),
            'variance': float(variance),
            'actual_min': float(actual_min),
            'actual_max': float(actual_max),
            'total_counts': int(total_counts),
            'non_zero_bins': int(non_zero_bins),
            'coverage_percent': float(coverage)
        })

    return {
        'band_statistics': band_stats,
        'total_pixels_per_band': total_pixels,
        'n_bands': n_bands,
        'histogram_bins': n_bins
    }


def create_global_stats_table(global_stats: dict[str, Any]) -> Table:
    """Create a Rich table for global band statistics."""
    table = Table(title="🌍 Global Band Statistics (All Datasets Combined)",
                  show_header=True,
                  header_style="bold magenta")
    table.add_column("Band", style="yellow", no_wrap=True, width=6)
    table.add_column("Mean (μ)", style="green", width=10)
    table.add_column("Std (σ)", style="blue", width=10)
    table.add_column("Min", style="red", width=8)
    table.add_column("Max", style="red", width=8)
    table.add_column("Coverage", style="cyan", width=10)
    table.add_column("Pixels", style="white", width=12)

    band_stats = global_stats['band_statistics']
    total_pixels = global_stats['total_pixels_per_band']

    for stats in band_stats:
        # Fix potential -0 display issue for min values
        min_val = stats['actual_min']
        if abs(min_val) < 0.5:  # If very close to zero
            min_display = "0"
        else:
            min_display = f"{min_val:6.0f}"

        table.add_row(
            get_band_label(stats['band_index']),
            f"{stats['mean']:8.2f}",
            f"{stats['std']:8.2f}",
            min_display,
            f"{stats['actual_max']:6.0f}",
            f"{stats['coverage_percent']:6.1f}%",
            f"{total_pixels:,}"
        )

    return table


def find_datasets(hdf5_file: h5py.File, key_path: str) -> list[tuple[str, h5py.Dataset]]:
    """Find all datasets in the HDF5 file matching a key path.
    
    Recursively searches for datasets with a specific path suffix.
    
    Args:
        hdf5_file: Opened HDF5 file object
        key_path: The suffix of the dataset path to find (e.g., 'S2/S2' or 'S1/S1')
        
    Returns:
        List of tuples containing (full_path, dataset_object) for each S2 dataset
    """
    datasets = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.endswith(f'/{key_path}'):
            datasets.append((name, obj))

    hdf5_file.visititems(visitor)
    return datasets


def compute_basic_statistics(data: np.ndarray) -> dict[str, float]:
    """Compute basic statistical measures for tensor data.
    
    Args:
        data: Input numpy array of any shape
        
    Returns:
        Dictionary containing statistical measures
    """
    flat_data = data.reshape(-1)

    return {
        'count': len(flat_data),
        'mean': float(np.mean(flat_data)),
        'std': float(np.std(flat_data)),
        'min': float(np.min(flat_data)),
        'max': float(np.max(flat_data)),
        'median': float(np.median(flat_data)),
        'q25': float(np.percentile(flat_data, 25)),
        'q75': float(np.percentile(flat_data, 75)),
        'non_zero_count': int(np.count_nonzero(flat_data)),
        'zero_count': int(np.sum(flat_data == 0))
    }


def compute_band_statistics(data: np.ndarray) -> list[dict[str, float]]:
    """Compute statistics for each spectral band separately.
    
    Assumes data shape is (time, bands, height, width).
    
    Args:
        data: 4D numpy array with shape (T, B, H, W)
        
    Returns:
        List of statistics dictionaries, one per band
    """
    if len(data.shape) != NUM_DIMS:
        console.print(f"⚠️ Expected 4D data, got {len(data.shape)}D. Skipping band analysis.")
        return []

    _, n_bands, _, _ = data.shape
    band_stats = []

    for band_idx in range(n_bands):
        band_data = data[:, band_idx, :, :]
        stats = compute_basic_statistics(band_data)
        stats['band_index'] = band_idx
        band_stats.append(stats)

    return band_stats


def compute_temporal_statistics(data: np.ndarray) -> dict[str, Any]:
    """Compute statistics across the temporal dimension.
    
    Args:
        data: 4D numpy array with shape (T, B, H, W)
        
    Returns:
        Dictionary with temporal statistics
    """
    if len(data.shape) != NUM_DIMS:
        return {'error': 'Expected 4D data for temporal analysis'}

    time_steps, n_bands, height, width = data.shape

    # Compute mean and std across spatial dimensions for each timestep
    temporal_means = np.mean(data, axis=(2, 3))  # Shape: (T, B)
    temporal_stds = np.std(data, axis=(2, 3))    # Shape: (T, B)

    return {
        'time_steps': time_steps,
        'n_bands': n_bands,
        'spatial_shape': (height, width),
        'mean_across_time': {
            'mean': float(np.mean(temporal_means)),
            'std': float(np.std(temporal_means)),
            'min': float(np.min(temporal_means)),
            'max': float(np.max(temporal_means))
        },
        'std_across_time': {
            'mean': float(np.mean(temporal_stds)),
            'std': float(np.std(temporal_stds)),
            'min': float(np.min(temporal_stds)),
            'max': float(np.max(temporal_stds))
        }
    }


def create_basic_stats_table(stats: dict[str, float], title: str) -> Table:
    """Create a Rich table for basic statistics."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("📊 Count", f"{stats['count']:,}")
    table.add_row("📈 Mean", f"{stats['mean']:.2f}")
    table.add_row("📉 Std Dev", f"{stats['std']:.2f}")
    table.add_row("⬇️ Min", f"{stats['min']:.0f}")
    table.add_row("⬆️ Max", f"{stats['max']:.0f}")
    table.add_row("🎯 Median", f"{stats['median']:.2f}")
    table.add_row("📊 Q25", f"{stats['q25']:.2f}")
    table.add_row("📊 Q75", f"{stats['q75']:.2f}")
    table.add_row("✅ Non-zero", f"{stats['non_zero_count']:,}")
    table.add_row("⭕ Zero", f"{stats['zero_count']:,}")

    return table


def create_band_stats_table(band_stats: list[dict[str, float]]) -> Table:
    """Create a Rich table for band-wise statistics."""
    table = Table(title="📊 Band-wise Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Band", style="yellow", no_wrap=True)
    table.add_column("Mean", style="green", width=10)
    table.add_column("Std", style="blue", width=10)
    table.add_column("Min", style="red", width=10)
    table.add_column("Max", style="red", width=10)
    table.add_column("Non-zero %", style="cyan", width=12)

    for stats in band_stats:
        non_zero_pct = (stats['non_zero_count'] / stats['count']) * 100
        table.add_row(
            get_band_label(stats['band_index']),
            f"{stats['mean']:7.1f}",
            f"{stats['std']:7.1f}",
            f"{stats['min']:7.0f}",
            f"{stats['max']:7.0f}",
            f"{non_zero_pct:8.1f}%"
        )

    return table


def create_temporal_stats_table(temporal_stats: dict[str, Any]) -> Table:
    """Create a Rich table for temporal statistics."""
    table = Table(title="⏰ Temporal Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    if 'error' in temporal_stats:
        table.add_row("❌ Error", temporal_stats['error'])
        return table

    table.add_row("🕐 Time steps", f"{temporal_stats['time_steps']}")
    table.add_row("📊 Bands", f"{temporal_stats['n_bands']}")
    table.add_row("📐 Spatial shape", f"{temporal_stats['spatial_shape']}")

    table.add_row("", "")  # Separator
    table.add_row("📈 Mean across time", "")
    mean_stats = temporal_stats['mean_across_time']
    table.add_row("  └─ Mean", f"{mean_stats['mean']:.2f}")
    table.add_row("  └─ Std", f"{mean_stats['std']:.2f}")
    table.add_row("  └─ Range", f"{mean_stats['min']:.2f} - {mean_stats['max']:.2f}")

    table.add_row("", "")  # Separator
    table.add_row("📉 Std across time", "")
    std_stats = temporal_stats['std_across_time']
    table.add_row("  └─ Mean", f"{std_stats['mean']:.2f}")
    table.add_row("  └─ Std", f"{std_stats['std']:.2f}")
    table.add_row("  └─ Range", f"{std_stats['min']:.2f} - {std_stats['max']:.2f}")

    return table


def analyze_s2_dataset(dataset_path: str, dataset: h5py.Dataset) -> None:
    """Perform comprehensive analysis of a single S2 dataset."""
    console.print(f"\n🔍 [bold]Analyzing S2 dataset:[/bold] [cyan]{dataset_path}[/cyan]")

    # Display dataset info
    console.print(f"📏 Shape: {dataset.shape}")
    console.print(f"🏷️ Data type: {dataset.dtype}")
    console.print(f"💾 Size: {dataset.size:,} elements")

    # Load data
    console.print("📥 Loading data...")
    data = dataset[:]

    # Basic statistics
    console.print("📊 Computing basic statistics...")
    basic_stats = compute_basic_statistics(data)
    stats_table = create_basic_stats_table(basic_stats, f"📊 Basic Statistics - {dataset_path.split('/')[-3]}")
    console.print(stats_table)

    # Band-wise statistics (if 4D data)
    if len(data.shape) == NUM_DIMS:
        console.print("🎨 Computing band-wise statistics...")
        band_stats = compute_band_statistics(data)
        if band_stats:
            band_table = create_band_stats_table(band_stats)
            console.print(band_table)

    # Temporal statistics (if 4D data)
    if len(data.shape) == NUM_DIMS:
        console.print("⏰ Computing temporal statistics...")
        temporal_stats = compute_temporal_statistics(data)
        temporal_table = create_temporal_stats_table(temporal_stats)
        console.print(temporal_table)

    # Histogram analysis for uint16 data
    if dataset.dtype == np.uint16:
        console.print("📈 Computing histogram analysis...")
        try:
            # Sample data for histogram if too large
            if data.size > 10_000_000:  # 10M elements
                console.print("⚠️ Large dataset detected. Sampling for histogram analysis...")
                sample_size = 1_000_000
                flat_data = data.reshape(-1)
                indices = np.random.choice(len(flat_data), sample_size, replace=False)
                sample_data = flat_data[indices]
            else:
                sample_data = data

            # Get optimal parameters for uint16 histogram
            n_bins, range_min, range_max = set_n_bins()

            hist, bins = compute_tensor_histogram(
                sample_data,
                n_bins=n_bins,
                range_min=range_min,
                range_max=range_max
            )

            # Display histogram statistics
            non_zero_bins = np.count_nonzero(hist)
            total_bins = len(hist)

            histogram_table = Table(title="📈 Histogram Analysis", show_header=True, header_style="bold magenta")
            histogram_table.add_column("Metric", style="cyan")
            histogram_table.add_column("Value", style="green")

            histogram_table.add_row("📊 Total bins", f"{total_bins:,}")
            histogram_table.add_row("✅ Non-empty bins", f"{non_zero_bins:,}")
            histogram_table.add_row("📈 Coverage", f"{(non_zero_bins/total_bins)*100:.2f}%")

            if non_zero_bins > 0:
                non_zero_indices = np.where(hist > 0)[0]
                min_val = int(bins[non_zero_indices[0]])
                max_val = int(bins[non_zero_indices[-1]])
                histogram_table.add_row("⬇️ Min value", f"{min_val}")
                histogram_table.add_row("⬆️ Max value", f"{max_val}")
                histogram_table.add_row("📏 Value range", f"{max_val - min_val}")

            console.print(histogram_table)

        except Exception as e:
            console.print(f"❌ Error computing histogram: {e}")


def create_global_stats_dataframe(global_stats: dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame from global band statistics.
    
    Args:
        global_stats: Dictionary with global statistics per band
        
    Returns:
        DataFrame with band statistics
    """
    band_stats = global_stats['band_statistics']

    # Convert to DataFrame
    df = pd.DataFrame(band_stats)

    # Add formatted band column
    df['Band_ID'] = df['band_index'].apply(lambda x: get_band_label(int(x)))

    # Reorder columns for better presentation
    column_order = [
        'Band_ID', 'band_index', 'mean', 'std', 'variance',
        'actual_min', 'actual_max', 'total_counts', 'non_zero_bins', 'coverage_percent'
    ]
    df = df[column_order]

    return df


def export_statistics_to_json(global_stats: dict[str, Any], output_path: str) -> None:
    """Export statistics to JSON file via DataFrame.
    
    Args:
        global_stats: Dictionary with global statistics
        output_path: Path for the output JSON file
    """
    # Create DataFrame
    df = create_global_stats_dataframe(global_stats)

    # Prepare complete statistics dictionary
    export_data = {
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_datasets_processed': len(global_stats.get('dataset_paths', [])),
            'n_bands': global_stats['n_bands'],
            'total_pixels_per_band': global_stats['total_pixels_per_band'],
            'histogram_bins': global_stats['histogram_bins']
        },
        'band_statistics': df.to_dict('records'),
        'summary': {
            'mean_across_bands': float(df['mean'].mean()),
            'std_across_bands': float(df['std'].mean()),
            'min_value_global': float(df['actual_min'].min()),
            'max_value_global': float(df['actual_max'].max()),
            'average_coverage': float(df['coverage_percent'].mean())
        }
    }

    # Export to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    console.print(f"📄 Statistics exported to: [bold green]{output_path}[/bold green]")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze S1/S2 tensor statistics from HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze S2 data (default)
  python analyze_s2_stats.py /path/to/file.hdf5

  # Analyze S1 data
  python analyze_s2_stats.py /path/to/file.hdf5 --key-path S1/S1
  
  # Analyze both S1 and S2 data
  python analyze_s2_stats.py /path/to/file.hdf5 --key-path S1/S1 S2/S2

  # Specify a base output file name (key will be appended)
  python analyze_s2_stats.py /path/to/file.hdf5 -k S1/S1 S2/S2 -o /path/to/output/stats.json
"""
    )

    parser.add_argument(
        'hdf5_file',
        type=str,
        help='Path to the HDF5 file to analyze'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Base output JSON file path. The key path will be appended to avoid overwrites.'
    )

    parser.add_argument(
        '--key-path', '-k',
        type=str,
        nargs='+',
        default=['S2/S2', 'S1/S1'],
        help='One or more path-like keys for datasets to find (e.g., "S1/S1" "S2/S2")'
    )

    return parser.parse_args()


def get_args() -> argparse.Namespace:
    """Get and validate command line arguments.
    
    Returns:
        Parsed and validated arguments namespace
    """
    args = parse_arguments()

    # Validate input file
    input_path = Path(args.hdf5_file)
    if not input_path.exists():
        console.print(f"❌ [red]File '{input_path}' does not exist[/red]")
        sys.exit(1)

    if input_path.suffix.lower() not in ['.hdf5', '.h5']:
        console.print(f"⚠️ [yellow]Warning: File '{input_path}' does not have .hdf5/.h5 extension[/yellow]")

    return args


def analyze_key_path(key_path: str, hdf5_file: h5py.File, args: argparse.Namespace):
    """Run analysis for a single key path within the HDF5 file."""
    console.print(Panel(f"Analyzing key: [bold yellow]{key_path}[/bold yellow]", expand=False, border_style="blue"))

    # Set current band labels based on key path
    data_type = key_path.split('/')[0]
    global CURRENT_BAND_LABELS
    CURRENT_BAND_LABELS = BAND_LABELS_DICT.get(data_type, [])

    # Determine output path
    input_path = Path(args.hdf5_file)
    key_suffix = key_path.replace('/', '_')
    if args.output:
        # If a base output is given, append key to it
        output_path_obj = Path(args.output)
        output_path = output_path_obj.parent / f"{output_path_obj.stem}_{key_suffix}{output_path_obj.suffix}"
    else:
        # Auto-generate output filename
        output_path = input_path.parent / f"{input_path.stem}_{key_suffix}_statistics.json"

    console.print(f"📄 Output will be saved to: [bold cyan]{output_path}[/bold cyan]")

    # Find all datasets
    console.print(f"🔍 Searching for '{key_path}' datasets...")
    datasets = find_datasets(hdf5_file, key_path)

    if not datasets:
        console.print(f"❌ [red]No '{key_path}' datasets found in the file.[/red]")
        return

    console.print(f"✅ Found {len(datasets)} '{key_path}' dataset(s)")

    # Accumulate histograms across all datasets
    accumulated_histograms, total_pixels = accumulate_band_histograms(datasets)

    if accumulated_histograms is None:
        console.print(f"❌ [red]No valid datasets found for '{key_path}' for histogram accumulation.[/red]")
        return

    # Compute global statistics from accumulated histograms
    global_stats = compute_global_band_statistics(accumulated_histograms, total_pixels)
    global_stats['dataset_paths'] = [path for path, _ in datasets]

    # Display global statistics table
    console.print("\n")
    global_table = create_global_stats_table(global_stats)
    console.print(global_table)

    # Export statistics to JSON
    console.print("📄 Exporting statistics to JSON...")
    export_statistics_to_json(global_stats, str(output_path))

    # Display summary
    console.print(f"\n📋 [bold]Summary for {key_path}:[/bold]")
    console.print(f"   📊 Processed datasets: {len(datasets)}")
    console.print(f"   🎨 Spectral bands: {global_stats['n_bands']}")
    console.print(f"   🔢 Total pixels per band: {global_stats['total_pixels_per_band']:,}")
    console.print(f"   📈 Histogram bins: {global_stats['histogram_bins']:,}")


def main():
    """Main analysis function."""
    console.print(Panel.fit(
        "🛰️  Tensor Statistics Analyzer\n[bold cyan]Global mean/std computation across all datasets[/bold cyan]",
        style="bold blue",
        border_style="blue"
    ))

    args = get_args()
    input_file_path = args.hdf5_file
    console.print(f"📁 Analyzing file: [bold green]{input_file_path}[/bold green]")

    try:
        with h5py.File(input_file_path, 'r') as f:
            for key_path in args.key_path:
                try:
                    analyze_key_path(key_path, f, args)
                except Exception as e:
                    console.print(f"❌ [red]An unexpected error occurred while processing '{key_path}': {e}[/red]")

        console.print("\n✅ [bold green]Analysis complete for all specified keys![/bold green]")

    except Exception as e:
        console.print(f"❌ [red]Error opening file: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
