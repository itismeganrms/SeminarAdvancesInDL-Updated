#!/bin/bash

#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/cache/training/new_geo/output%A.out

# Load required modules
module load ALICE/legacy
module load Miniconda2/4.7.10
module load CUDA/11.3.1

# Activate Conda environment

base_dir="/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

find "$base_dir" -depth -type d -name "BraTS20_Training_*" | while read dir; do
    parent=$(dirname "$dir")
    newname=$(basename "$dir" | sed 's/^BraTS20_Training_//')
    mv "$dir" "$parent/$newname"
done
