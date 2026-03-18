#!/bin/bash
#SBATCH --job-name=irrigation
#SBATCH --output=%u-%j.out
#SBATCH --error=%u-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --account=gza5dr

# Activate your conda env
module load miniforge
source activate /sfs/weka/scratch/gza5dr/IrrigationType_experiments/mutli_reasoning/conda_env

# Run from scratch (faster I/O than /home)
cd /sfs/weka/scratch/gza5dr/IrrigationType_experiments/mutli_reasoning/ag_multimodal_reasoning

python scripts/train.py +experiment=step1_rgb_baseline
