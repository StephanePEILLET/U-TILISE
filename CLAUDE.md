# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

U-TILISE is a deep learning model for **cloud removal in optical satellite image time series**. It reconstructs cloud-masked Sentinel-2 (optical) images using spatial and temporal context, optionally enhanced with Sentinel-1 (SAR/radar) data.

## Setup

```bash
conda env create -f envs/cr.yml
conda activate cr

cp .env.example .env
# Edit .env with local data paths (HDF5 files, raster tiles, output dirs)
```

Configuration uses OmegaConf: YAML configs are merged with `configs/default.yaml`, and environment variables are interpolated via `${oc.env:VAR_NAME}` from the `.env` file.

## Main Commands

**Training:**
```bash
python run_train.py configs/<config>.yaml --save_dir /path/to/output
```

**Evaluation:**
```bash
python run_eval.py configs/<config>.yaml utilise \
  --checkpoint /path/to/model.pth \
  --test-data.data-dir ./data/ \
  --test-data.hdf5-file <file>.hdf5
```

**Inference on geospatial raster tiles:**
```bash
python infer_from_tiles.py configs/config_run_infer_from_tiles_aleatoire.yaml utilise
# or via batch script:
bash launch_infer_masks.sh [aleatoire|consecutif|all]
```

**Regenerate metrics summary report:**
```bash
python aggregate_metrics.py
```

There is no automated test suite. Validation is done through evaluation metrics during and after training.

## Architecture

The model has three main stages:

1. **Spatial Encoder** (`lib/models/utilise.py`): Per-frame CNN with strided convolutions that produces spatial feature maps for each timestep independently.

2. **Temporal Module** (`lib/models/ltae_transformer.py`): Lightweight Temporal Attention Encoder (LTAE) — a transformer-based module that aggregates features across the time dimension using positional encodings (day-of-year).

3. **Spatial Decoder** (`lib/models/utilise.py`): Upsampling convolutions with skip connections (U-Net style) that reconstruct the clean multi-spectral output image.

### Key Components

| File | Role |
|------|------|
| `run_train.py` | Training orchestration |
| `run_eval.py` | Evaluation loop |
| `infer_from_tiles.py` | Large-scale inference on GeoTIFF tiles |
| `lib/trainer.py` | Training step/epoch logic |
| `lib/loss.py` | Loss functions (L1, SSIM, NDVI, temporal R²) |
| `lib/data_utils.py` | Data loading and preprocessing utilities |
| `lib/eval_tools.py` | Pixel-level metrics (MAE, RMSE, PSNR, SSIM, SAM, R²) |
| `dataloader_CIRCA/` | Extended data loaders for cloud reconstruction with HDF5 support |
| `configs/` | YAML configs for training/eval |
| `configs/jzay/` | SLURM cluster configs |

### Data Pipeline

- **Format**: HDF5 files with temporally trimmed sequences and synthetic cloud gaps
- **Cloud masking**: Simulated via random or consecutive frame masking during training
- **Multi-modal**: Optional Sentinel-1 SAR channels (ascending/descending orbit modes)
- **Evaluation**: Metrics computed separately on *occluded* (reconstructed) and *observed* (unmasked) pixels

### Loss Functions

Current best-performing models (v4) use a combination of:
- L1 (on observed pixels)
- L1_occluded (on cloud-masked pixels)
- SSIM
- NDVI loss (vegetation index consistency)
- Temporal R² loss (inter-frame temporal coherence)

## Experiment Tracking

- **Weights & Biases** (`wandb`): Primary training visualization and metric tracking
- Each experiment saves its merged `config.yaml` alongside model checkpoints for reproducibility
- Metrics summary across all trained models: `rapport_metriques.md`

## Important Notes

- Comments and config names mix French and English (French for local context, English for docstrings/APIs)
- Inference (`infer_from_tiles.py`) handles SLURM `SIGUSR1` signals for graceful preemption and cleans up temporary files on failure
- Recent memory optimizations (branches `jzay`, recent commits) address OOM issues during inference on large tiles — avoid reverting those changes
- The `dataloader_CIRCA/` directory extends the original `lib/datasets/` loaders and is where most active dataset development happens
