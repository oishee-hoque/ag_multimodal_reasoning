#!/bin/bash
#SBATCH --job-name=irrigation
#SBATCH --output=slurm/%u-%j.out
#SBATCH --error=slurm/%u-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=760GB
#SBATCH --time=6:00:00
#SBATCH --account=nssac_students

# Activate your conda env
module load miniforge
source activate /sfs/weka/scratch/gza5dr/IrrigationType_experiments/mutli_reasoning/conda_env

# Run from scratch (faster I/O than /home)
cd /sfs/weka/scratch/gza5dr/IrrigationType_experiments/mutli_reasoning/ag_multimodal_reasoning
# Redirect wandb to scratch (avoid home quota)
export WANDB_DIR=/sfs/weka/scratch/gza5dr/wandb_tmp
export WANDB_CACHE_DIR=/sfs/weka/scratch/gza5dr/wandb_cache
export WANDB_DATA_DIR=/sfs/weka/scratch/gza5dr/wandb_data


# python scripts/train.py model=deeplabv3plus_swin_tiny data=rgb_single experiment_name=A4_dlv3_50_rgb_effe_dice_norm


# python scripts/train.py model=deeplabv3plus_resnet50 data=rgb_single experiment_name= DLR_50_rgb_effe_dice_norm
# python scripts/train.py  model=deeplabv3plus_resnet50 data=rgb_noise_reduced data.noise_strategy=ndvi_relabel data.low_threshold=0.15 experiment_name=DLR_50_noise_rgb_effe_dice
# python scripts/train.py  model=deeplabv3plus_resnet50 data=spectral_single data.noise_strategy=ndvi_relabel data.low_threshold=0.15 experiment_name=DLR_50_spectral_effe_dice
python scripts/train.py  model=deeplabv3plus_resnet50 data=temporal data.noise_strategy=ndvi_relabel data.low_threshold=0.15 experiment_name=DLR_50_temporal_effe_dice


