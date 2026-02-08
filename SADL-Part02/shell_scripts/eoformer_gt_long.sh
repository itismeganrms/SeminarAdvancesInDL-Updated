#!/bin/bash -l

#SBATCH --job-name=eoformer_gt
#SBATCH --time=4-01:00:00
#SBATCH --partition=gpu-a100-80g
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/eoformer/new/gt_long-%A.out
#SBATCH --gres=gpu:1

# Load required modules
echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SADL-Part02/ALICE_Code/self_models/

python optimized_eoformer_gt_copy.py