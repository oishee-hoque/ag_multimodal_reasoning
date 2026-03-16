#!/bin/bash
#SBATCH --job-name=irrigation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

# Load modules (adjust for your HPC)
module load anaconda3
module load cuda/12.1

# Activate environment
conda activate irrigation

# Create log directory
mkdir -p logs/slurm

# Run training with all arguments passed through
python "$@"
