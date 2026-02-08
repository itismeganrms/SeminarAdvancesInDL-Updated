#!/bin/bash -l

#SBATCH --job-name=eoformer_gt
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/eoformer/final/output%A.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

# Load required modules

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/self_models/

python optimized_eoformer_gt.py