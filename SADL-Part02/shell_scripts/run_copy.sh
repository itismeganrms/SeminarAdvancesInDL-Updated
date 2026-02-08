#!/bin/bash

#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/cache/mona_outputs/output%A.out

# Load required modules
module load ALICE/legacy
module load Miniconda2/4.7.10
module load CUDA/11.3.1

# Activate Conda environment
source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

# pip uninstall mlpipeline -y

## Install the package locally

#Splitting the Dataset - This has been done
# cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/
# python SiNGR-tumor/mlpipeline/utils/split_brats.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets --seed 56789 
# python SiNGR-tumor/mlpipeline/utils/split_lggflair.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/ --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets --seed 56789 

#SiNG transform
cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# pip install -e .

# cd mlpipeline/utils
# python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/utils/geodesic_transform_own.py --label_name "fast_sgc_margin" --dataset "BRATS" --root_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --gt_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS

# Starting a trial run to see model
# python -m  mlpipeline.train.run /
#         experiment=brats_uncertainty_sem_seg    /
#         model.params.cfg.arch=SwinUNETR

# python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg model.params.cfg.arch=SwinUNETR
python -m mlpipeline.train.run \
  experiment=brats_uncertainty_sem_seg \
  model.params.cfg.arch=SwinUNETR \
  paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/outputs/ \
  data.num_workers=2 \
  data.batch_size=1 \
  data.training_samples=4 \
  data.valid_samples=4

# python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg \        
#         model.params.cfg.arch=SwinUNETR
# python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/train/run.py