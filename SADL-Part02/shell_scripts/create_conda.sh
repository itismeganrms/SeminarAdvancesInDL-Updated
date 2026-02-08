#!/bin/bash

#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=try_env
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

# Load required modules
module load ALICE/legacy
module load Miniconda2/4.7.10
module load CUDA/11.3.1

# Activate Conda environment
source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

# Check if torch is available
python3 -c "import torch; print('GPU available?', torch.cuda.is_available())"
