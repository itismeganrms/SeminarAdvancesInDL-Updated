#!/bin/bash -l

#SBATCH --job-name=unetpp
#SBATCH --time=5-10:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/unetpp/gt/output%A.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

# Navigate to project directory
cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/self_models/unetpp3d

python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/self_models/unetpp3d/unetpp3d_run_with_gt_copy.py \
  --data_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --max_epochs 30 \
  --batch_size 1  \
  --fold 3 

echo "Script executed"
