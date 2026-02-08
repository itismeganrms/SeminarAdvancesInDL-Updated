#!/bin/bash -l

#SBATCH --job-name=swinunet-sweep
#SBATCH --time=6-23:59:59
#SBATCH --partition=gpu-mig-40g
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/swinunet/new/sweep-output-%A.out
#SBATCH --gres=gpu:1

# Load required modules

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SADL-Part02/ALICE_Code/self_models/

python swinunetr_sweep.py 