#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=env1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=output_%j.out

# Load required modules
module load ALICE/legacy
module load Miniconda3/4.9.2
module load CUDA/11.7.0


# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

# pip uninstall mlpipeline -y

## Install the package locally

#Splitting the Dataset - This has been done
cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/
python SiNGR-tumor/mlpipeline/utils/split_brats.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/ --seed 42 
python SiNGR-tumor/mlpipeline/utils/split_lggflair.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/ --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/ --seed 42 

#SiNG transform
cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# pip install -e .

# cd mlpipeline/utils
# python SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/utils/geodesic_transform_own.py --label_name "fast_sgc_margin" --dataset "BRATS" --root_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --gt_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS

# Starting a trial run to see model
# python -m  mlpipeline.train.run /
#         experiment=brats_uncertainty_sem_seg    /
#         model.params.cfg.arch=SwinUNETR

# python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg model.params.cfg.arch=SwinUNETR
# python -m mlpipeline.train.run \
  # experiment=brats_uncertainty_sem_seg \
  # model.params.cfg.arch=SwinUNETR \
  # paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/outputs_moj
python -m mlpipeline.train.run \
  experiment=brats_uncertainty_sem_seg \
  model.params.cfg.arch=SwinUNETR \
  paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/outputs_moj \
  data.num_workers=2 \
  data.batch_size=1 \
  data.training_samples=4 \
  data.valid_samples=4
  # data.label_name=fast_sgc_margin

# python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg \        
#         model.params.cfg.arch=SwinUNETR
# python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/train/run.py