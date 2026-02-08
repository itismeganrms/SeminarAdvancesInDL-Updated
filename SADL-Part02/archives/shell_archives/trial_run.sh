#!/bin/bash -l

#SBATCH --job-name=swinunetrun
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/data1/courses/2024-2025/4343SADL6/Tumor_Group/slurm/swinunet/final/output%A.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

# Load required modules

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init


# Activate Conda environment
source activate /data1/courses/2024-2025/4343SADL6/Tumor_Group/conda_envs/sdl_course

# cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# # pip install -e .
# cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/self_models

python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/simple_unet_run.py --data_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData --max_epochs 100 --batch_size 1 --fold 2

# # pip uninstall mlpipeline -y

# ## Install the package locally

# #Splitting the Dataset - This has been done
# # cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/
# # python SiNGR-tumor/mlpipeline/utils/split_brats.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets --seed 56789 
# # python SiNGR-tumor/mlpipeline/utils/split_lggflair.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/ --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets --seed 56789 

# #SiNG transform
# cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# # pip install -e .

# cd mlpipeline/utils
# # python gt_final_pt2.py
# # python geodesic_transform_own.py
# python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/utils/gt_final.py \
# --dataset "BraTS" \
# --mode "fast" \
# --input_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData \
# --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT/Validation

# # Starting a trial run to see model
# # python -m  mlpipeline.train.run /
# #         experiment=brats_uncertainty_sem_seg    /
# #         model.params.cfg.arch=SwinUNETR

# # python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg model.params.cfg.arch=SwinUNETR
# # python -m mlpipeline.train.run \
# #   experiment=brats_uncertainty_sem_seg \
# #   model.params.cfg.arch=NestedFormer \
# #   paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/outputs/ \
# #   data.num_workers=1 \
# #   data.batch_size=1 \
# #   data.training_samples=4 \
# #   data.valid_samples=4

# # # Starting a trial run to see model
# # # python -m  mlpipeline.train.run /
# # #         experiment=brats_uncertainty_sem_seg    /
# # #         model.params.cfg.arch=SwinUNETR

# # # python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg model.params.cfg.arch=SwinUNETR
# # python -m mlpipeline.train.run \
# #   experiment=brats_uncertainty_sem_seg \
# #   model.params.cfg.arch=NestedFormer \
# #   paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/outputs/ \
# #   data.num_workers=1 \
# #   data.batch_size=1 \
# #   data.training_samples=4 \
# #   data.valid_samples=4

# # python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg \        
# #         model.params.cfg.arch=SwinUNETR
# # python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/train/run.py

# ################ FINAL RUNS ##################
# # cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# # python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/self_models/optimized_eoformer_gt_copy.py

# ## trial run for simple UNET (SWINUnet)
# # python simple_unet_run.py \
# #   --data_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData \
# #   --id_pickle /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets/cv_split_5folds_brats_56789.pkl \
# #   --max_epochs 50 \
# #   --batch_size 2



# pip uninstall mlpipeline -y

## Install the package locally

#Splitting the Dataset - This has been done
# cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/
# python SiNGR-tumor/mlpipeline/utils/split_brats.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets --seed 56789 
# python SiNGR-tumor/mlpipeline/utils/split_lggflair.py --root /data1/courses/2024-2025/4343SADL6/Tumor_Group/ --output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets --seed 56789 

#SiNG transform
cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# pip install -e .

cd mlpipeline/utils
python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/utils/gt_final.py \
--dataset "BraTS" \
--mode "fast" \
--input_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
--output_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT

# Starting a trial run to see model
# python -m  mlpipeline.train.run /
#         experiment=brats_uncertainty_sem_seg    /
#         model.params.cfg.arch=SwinUNETR

# python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg model.params.cfg.arch=SwinUNETR
# python -m mlpipeline.train.run \
#   experiment=brats_uncertainty_sem_seg \
#   model.params.cfg.arch=NestedFormer \
#   paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/outputs/ \
#   data.num_workers=1 \
#   data.batch_size=1 \
#   data.training_samples=4 \
#   data.valid_samples=4

# # Starting a trial run to see model
# # python -m  mlpipeline.train.run /
# #         experiment=brats_uncertainty_sem_seg    /
# #         model.params.cfg.arch=SwinUNETR

# # python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg model.params.cfg.arch=SwinUNETR
# python -m mlpipeline.train.run \
#   experiment=brats_uncertainty_sem_seg \
#   model.params.cfg.arch=NestedFormer \
#   paths.output_dir=/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/outputs/ \
#   data.num_workers=1 \
#   data.batch_size=1 \
#   data.training_samples=4 \
#   data.valid_samples=4

# python -m mlpipeline.train.run experiment=brats_uncertainty_sem_seg \        
#         model.params.cfg.arch=SwinUNETR
# python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/mlpipeline/train/run.py

################ FINAL RUNS ##################
# cd /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor
# python /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/self_models/optimized_eoformer_gt_copy.py

## trial run for simple UNET (SWINUnet)
# python simple_unet_run.py \
#   --data_dir /data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData \
#   --id_pickle /data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SiNGR-tumor/workdir/datasets/cv_split_5folds_brats_56789.pkl \
#   --max_epochs 50 \
#   --batch_size 2
