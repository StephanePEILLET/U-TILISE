#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate cr

python ./run_eval.py ./configs/config_run_eval.yaml utilise \
--checkpoint /DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_BANDS_120_epochs_2025-07-11_18-05/checkpoints/Model_best.pth \
--test-data.test-config /DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_BANDS_120_epochs_2025-07-11_18-05/config.yaml \
--test-data.hdf5-file-read /DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5 \
--test-data.split test

python ./run_eval.py ./configs/config_run_eval.yaml utilise \
--checkpoint /DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_SAR_120_epochs_2025-07-11_16-56/checkpoints/Model_best.pth \
--test-data.test-config /DATA_10TB/data_rpg/outputs/U-TILISE/results/ALL_SAR_120_epochs_2025-07-11_16-56/config.yaml \
--test-data.hdf5-file-read /DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5 \
--test-data.split test

python ./run_eval.py ./configs/config_run_eval.yaml utilise \
--checkpoint /DATA_10TB/data_rpg/outputs/U-TILISE/results/BGR_NIR_SAR_120_epochs_2025-07-11_16-56/checkpoints/Model_best.pth \
--test-data.test-config /DATA_10TB/data_rpg/outputs/U-TILISE/results/BGR_NIR_SAR_120_epochs_2025-07-11_16-56/config.yaml \
--test-data.hdf5-file-read /DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5 \
--test-data.split test

python ./run_eval.py ./configs/config_run_eval.yaml utilise \
--checkpoint /DATA_10TB/data_rpg/outputs/U-TILISE/results/BGR_NIR_120_epochs_2025-07-11_16-56/checkpoints/Model_best.pth \
--test-data.test-config /DATA_10TB/data_rpg/outputs/U-TILISE/results/BGR_NIR_120_epochs_2025-07-11_16-56/config.yaml \
--test-data.hdf5-file-read /DATA_10TB/data_rpg/circa/hdf5/CIRCA_CR_merged.hdf5 \
--test-data.split test