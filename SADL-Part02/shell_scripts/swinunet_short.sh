#!/bin/bash

#SBATCH --job-name=swinunetr
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/swinunet/new/output-%A.out

# Load required modules

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

# Activate Conda environment
source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SADL-Part02/ALICE_Code/SwinUNETR/BRATS21

# python main.py --json_list=/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets/cv_split_5folds_brats_56789.pkl --data_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --max_epochs=300 --val_every=5 --noamp --distributed \
# --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
python simple_unet_run.py --data_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --max_epochs 2 --batch_size 1 --fold 2