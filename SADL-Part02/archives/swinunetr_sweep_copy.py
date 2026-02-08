import os
import argparse
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from monai.data import Dataset
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandRotate90d,
    RandShiftIntensityd, RandScaleIntensityd, ToTensord, ResizeWithPadOrCropd
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from sklearn.model_selection import train_test_split
import random
import nibabel as nib
import wandb

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 1e-5, 1e-6]},
        "epochs": {"values": [10, 20, 30]},
        "optimizer": {"values": ["Adam", "SGD", "AdamW"]},
        "batch_size": {"values": [1]},
        "loss_function": {"values": ["DiceLoss", "DiceCELoss", "DiceFocalLoss", "GeneralizedDiceLoss", "GeneralizedDiceFocalLoss"]},
    }
}

def get_data(data_dir, id_list):
    input_names = ['t1', 't2', 'flair', 't1ce']
    data = []
    for id_num in id_list:
        patient_id = "BraTS20_Training_{:03d}".format(int(id_num))
        base_dir = os.path.join(data_dir, patient_id)
        images = [os.path.join(base_dir, "{}_{}.nii".format(patient_id, name)) for name in input_names]
        label = os.path.join(base_dir, "{}_seg.nii".format(patient_id))

        if not all(os.path.exists(img) for img in images) or not os.path.exists(label):
            continue
        data.append({"image": images, "label": label})
    return data

def get_data_with_gt(data_dir, geo_dir, id_list):
    input_names = ['t1', 't2', 'flair', 't1ce']
    data = []
    for id_num in id_list:
        patient_id = "BraTS20_Training_{:03d}".format(int(id_num))
        base_dir = os.path.join(data_dir, patient_id)
        geo_dir_values = os.path.join(geo_dir, patient_id)
        images = [os.path.join(base_dir, "{}_{}.nii".format(patient_id, name)) for name in input_names]
        label = os.path.join(geo_dir_values, "{}_seg_transformed.nii.gz".format(patient_id))
        if not all(os.path.exists(img) for img in images) or not os.path.exists(label):
            continue
        data.append({"image": images, "label": label})
    return data


def main():
    run = wandb.init()
    config = wandb.config

    data_dir ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    geo_dir ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"
    train_csv ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/train_ids_seed_34.csv"
    val_csv ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/val_ids_seed_34.csv"
    fold = 2
    

    train_csv = pd.read_csv(train_csv)
    val_csv = pd.read_csv(val_csv)

    if fold is not None:
        train_csv = train_csv[train_csv['fold'] == fold]
        val_csv = val_csv[val_csv['fold'] == fold]

        train_ids = train_csv['ID'].tolist()
        val_ids = val_csv['ID'].tolist()

        train_folds = train_csv['fold'].tolist()
        val_folds = val_csv['fold'].tolist()
    train_files = get_data_with_gt(data_dir, geo_dir, train_ids)                   
    val_files = get_data_with_gt(data_dir, geo_dir, val_ids)

    img_size = (64, 64, 64)

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=img_size),
        # RandFlipd(keys=["image", "label"], prob=0.2),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=img_size),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=img_size,
        in_channels=4,
        out_channels=1,
        feature_size=12,
        depths=(1, 1, 2, 1),  # instead of (2, 2, 2, 2)
        num_heads=(1, 2, 4, 8),
    ).to(device)
  
    if config.loss_function == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    elif config.loss_function == "DiceFocalLoss":
        loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif config.loss_function == "GeneralizedDiceLoss":
        loss_function = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif config.loss_function == "GeneralizedDiceFocalLoss":
        loss_function = GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True)
    else:
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # print("Model loaded")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True)
    ])


    min_loss_seen = float('inf')
    max_loss_seen = float('-inf')
    min_val_loss_seen = float('inf')
    max_val_loss_seen = float('-inf')

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch}")
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Optional: log batch loss
            # wandb.log({"batch_loss": loss.item()})

        loss_per_run = epoch_loss / len(train_loader)

        # Normalize training loss
        # min_loss_seen = min(min_loss_seen, loss_per_run)
        # max_loss_seen = max(max_loss_seen, loss_per_run)
        # normalized_loss = (loss_per_run - min_loss_seen) / (max_loss_seen - min_loss_seen) if max_loss_seen > min_loss_seen else 0.0

        # 🔍 VALIDATION LOSS
        model.eval()
        val_epoch_loss = 0
        dice_metric.reset()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_inputs)
                val_loss = loss_function(val_outputs, val_labels)
                val_epoch_loss += val_loss.item()

                # Optional: Dice metric
                val_outputs_post = post_pred(val_outputs)
                dice_metric(y_pred=val_outputs_post, y=val_labels)

        val_loss_avg = val_epoch_loss / len(val_loader)

        # Normalize validation loss
        # min_val_loss_seen = min(min_val_loss_seen, val_loss_avg)
        # max_val_loss_seen = max(max_val_loss_seen, val_loss_avg)
        # normalized_val_loss = (val_loss_avg - min_val_loss_seen) / (max_val_loss_seen - min_val_loss_seen) if max_val_loss_seen > min_val_loss_seen else 0.0

        # val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        # 📊 LOG TO WANDB
        wandb.log({
            "epoch": epoch + 1,
            "loss for epoch": round(loss_per_run, 4),
            # "normalized_loss": round(normalized_loss, 4),
            "val_loss": round(val_loss_avg, 4),
            # "normalized_val_loss": round(normalized_val_loss, 4),
            # "val_dice": round(val_dice, 4)
        })

        print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {loss_per_run:.4f} | Val Loss: {val_loss_avg:.4f}")

    run.finish()
    
    model.eval()
    with torch.no_grad():
        dice_metric.reset()
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, img_size, 1, model)
            val_outputs = post_pred(val_outputs)
            dice_metric(y_pred=val_outputs, y=val_labels)
        metric = dice_metric.aggregate().item()
        print(f"Validation Dice: {metric:.4f}")
        wandb.log({"val_dice": metric})
        dice_metric.reset()
    # run.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="SeminarAdvancesinDeepLearning", entity="universiteitleiden")
    wandb.agent(sweep_id, function=main, count=10)