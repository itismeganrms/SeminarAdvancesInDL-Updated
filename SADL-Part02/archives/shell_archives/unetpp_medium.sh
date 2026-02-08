#!/bin/bash -l

#SBATCH --job-name=unetpp
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-medium
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/unet3d/output%A.out
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

python unetpp3d_run.py \
  --data_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --train_csv /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/train_ids_seed_34.csv \
  --val_csv /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/val_ids_seed_34.csv \
  --max_epochs 5 \
  --batch_size 1 \
  --fold 3

echo "Script executed"
