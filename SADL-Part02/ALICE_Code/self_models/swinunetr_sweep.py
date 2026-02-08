import os
import argparse
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import pylab as pyl
import time
import datetime
from mpl_toolkits.mplot3d import Axes3D  

from monai.data import Dataset
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandRotate90d,
    RandShiftIntensityd, RandScaleIntensityd, ToTensord, ResizeWithPadOrCropd
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from monai.losses import DiceFocalLoss
from monai.losses import DiceCELoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, GeneralizedWassersteinDiceLoss


from sklearn.model_selection import train_test_split
import random
import nibabel as nib
import wandb

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 1e-5, 1e-3, 1e-6]},
        "epochs": {"values": [50]},
        "optimizer": {"values": ["Adam", "SGD", "AdamW"]},
        "batch_size": {"values": [1, 2]},
        "loss_function": {"values": ["DiceLoss", "DiceCELoss", "GeneralizedDiceLoss"]},
    }
}

def get_data_with_gt(data_dir, geo_dir, geo_map_dir, id_list):
    input_names = ['t1', 't2', 'flair', 't1ce']
    data = []
    for id_num in id_list:
        patient_id = f"BraTS20_Training_{int(id_num):03d}"
        base_dir = os.path.join(data_dir, patient_id)
        geo_label_dir = os.path.join(geo_dir, patient_id)
        geo_map_dir_patient = os.path.join(geo_map_dir, patient_id)
        images = [os.path.join(base_dir, f"{patient_id}_{name}.nii") for name in input_names]
        label = os.path.join(geo_label_dir, f"{patient_id}_seg_transformed.nii.gz")
        geo_maps = [os.path.join(geo_map_dir_patient, f"{patient_id}_fast_sg_{name}.nii.gz") for name in input_names]
        if not all(os.path.exists(p) for p in images + geo_maps + [label]):
            continue
        data.append({
            "image": images,
            "geo": geo_maps,
            "label": label
        })
    return data

def get_brat_region(onehot, region):
    if region == "WT":
        return onehot[:, 1:4].sum(dim=1, keepdim=True).clamp(0, 1)
    elif region == "TC":
        return onehot[:, [1, 3]].sum(dim=1, keepdim=True).clamp(0, 1)

    elif region == "ET":
        return onehot[:, 3:4] 
    else:
        raise ValueError(f"Unknown region {region}")

def visualize_prediction(gt_mask, pred_mask, save_dir=None, sample_id=None):
    print(f"gt_mask shape: {gt_mask.shape}, pred_mask shape: {pred_mask.shape}")
    
    # Convert to NumPy and squeeze
    gt_np = gt_mask.detach().cpu().numpy() if hasattr(gt_mask, "detach") else np.array(gt_mask)
    pred_np = pred_mask.detach().cpu().numpy() if hasattr(pred_mask, "detach") else np.array(pred_mask)

    gt_np = np.squeeze(gt_np)
    pred_np = np.squeeze(pred_np)
   
    fig = pyl.figure(figsize=(12, 6))
    # Ground truth plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.voxels(gt_np != 0, facecolors='blue', edgecolor='k', linewidth=0.2, alpha=0.6)
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Prediction plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.voxels(pred_np != 0, facecolors='red', edgecolor='k', linewidth=0.2, alpha=0.6)
    ax2.set_title("Prediction")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    if save_dir and sample_id:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{sample_id}_3d_voxel_comparison.png")
        pyl.savefig(save_path, bbox_inches='tight', dpi=300)
        pyl.close()
        print(f"Saved: {save_path}")
    else:
        pyl.tight_layout()
        pyl.show()

def main():
    run = wandb.init()
    config = wandb.config

    wandb.config.update({"architecture": "SwinUNETR"}, allow_val_change=True)

    data_dir ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    geo_dir ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"
    train_csv ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SADL-Part02/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/train_ids_seed_34.csv"
    val_csv ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SADL-Part02/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/val_ids_seed_34.csv"
    fold = 2

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    geo_map_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT/fast_sgc"
    save_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Results/BRATS_Results_swinunetr_timestamp_{}".format(timestamp)
    train_csv = pd.read_csv(train_csv)
    val_csv = pd.read_csv(val_csv)
    num_classes = 4

    if fold is not None:
        train_csv = train_csv[train_csv['fold'] == fold]
        val_csv = val_csv[val_csv['fold'] == fold]

        train_ids = train_csv['ID'].tolist()
        val_ids = val_csv['ID'].tolist()

        train_folds = train_csv['fold'].tolist()
        val_folds = val_csv['fold'].tolist()

    train_files = get_data_with_gt(data_dir, geo_dir,geo_map_dir, train_ids)      
    val_files = get_data_with_gt(data_dir, geo_dir,geo_map_dir, val_ids)

    img_size = (64, 64, 64)

    train_transforms = Compose([
        LoadImaged(keys=["image", "geo", "label"]),
        EnsureChannelFirstd(keys=["image", "geo", "label"]),
        Spacingd(keys=["image", "geo", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "bilinear", "nearest")),
        Orientationd(keys=["image", "geo", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image", "geo"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "geo", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "geo", "label"], spatial_size=img_size),
        RandFlipd(keys=["image", "geo", "label"], prob=0.2),
        ToTensord(keys=["image", "geo", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "geo", "label"]),
        EnsureChannelFirstd(keys=["image", "geo", "label"]),
        Spacingd(
            keys=["image", "geo", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "bilinear", "nearest")
            ),
        Orientationd(keys=["image", "geo", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image", "geo"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "geo", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "geo", "label"], spatial_size=img_size),
        ToTensord(keys=["image", "geo", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=img_size,
        # in_channels=4,
        in_channels=8,
        out_channels=4,  # 4 output channels for 4-class segmentation
        feature_size=12,
        depths=(1, 1, 2, 1),
        num_heads=(1, 2, 4, 8),
    ).to(device)
  
    if config.loss_function == "DiceCELoss":
        loss_function = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, reduction="mean")
    elif config.loss_function == "GeneralizedDiceLoss":
        loss_function =  GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, reduction="mean")
    else:
        loss_function =  DiceLoss(include_background=True, to_onehot_y=True, softmax=True, reduction="mean")

    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.8)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    dice_loss_fn_val = DiceLoss(include_background=False, reduction="mean")
    print("Model loaded")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch}")
        model.train()
        epoch_loss = 0
        vis_count = 0
        for batch_data in train_loader:
            image = batch_data["image"].to(device)              # shape: [1, 4, 64, 64, 64]
            geo = batch_data["geo"][:, :4].to(device)           # reduce 12 → 4 channels
            labels = batch_data["label"].to(device)              # shape: [1, 1, 64, 64, 64]
            labels = labels.clone()
            labels[labels == 4] = 3

            inputs = torch.cat((image, geo), dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
            if vis_count < 3:
                pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                visualize_prediction(gt_mask=labels, pred_mask=pred, save_dir=save_dir, sample_id=f"epoch{epoch+1}_sample{vis_count}")
                vis_count += 1

        avg_train_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch + 1}/{config.epochs}, Training Loss: {avg_train_loss:.4f}")
        run.log({"train_loss": avg_train_loss}, step=epoch)

        regions = ["WT", "TC", "ET"]
        val_region_dice = {r: [] for r in regions}

        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                image = val_data["image"].to(device)
                geo = val_data["geo"][:, :4].to(device)
                val_labels = val_data["label"].to(device)
                val_labels = val_labels.clone()
                
                val_labels[val_labels == 4] = 3
                val_inputs = torch.cat((image, geo), dim=1)
                val_outputs = model(val_inputs)

                pred_onehot = AsDiscrete(argmax=True, to_onehot=num_classes)(val_outputs)  # [B,4,H,W,D]
                val_labels = val_labels.squeeze(1)  # remove channel dim → [B,H,W,D]
                label_onehot = torch.nn.functional.one_hot(val_labels.long(), num_classes=num_classes)  # [B,H,W,D,C]
                label_onehot = label_onehot.permute(0, 4, 1, 2, 3).float()  # [B,C,H,W,D]

                for r in regions:
                    pred_region = get_brat_region(pred_onehot, r)   # [B,1,H,W,D]
                    label_region = get_brat_region(label_onehot, r) # [B,1,H,W,D]

                    pred_region = (pred_region > 0).float()
                    label_region = (label_region > 0).float()
                    pred_region = pred_region.mean(dim=0, keepdim=True)  # [1,1,H,W,D]
                    # score = dice_loss_fn_val(pred_region, label_region)
                    # score = score.mean()
                    # val_region_dice[r].append(score.detach().cpu().item())

                    
                    dice_loss = dice_loss_fn_val(pred_region, label_region)
                    dice_score = 1.0 - dice_loss
                    val_region_dice[r].append(dice_score.item())

        val_region_dice_avg = {}
        for r in regions:
            if len(val_region_dice[r]) == 0:
                val_region_dice_avg[r] = float("nan")
            else:
                val_region_dice_avg[r] = float(np.mean(val_region_dice[r]))
        log_dict = {f"val_dice_{r}": val_region_dice_avg[r] for r in regions}
        run.log(log_dict, step=epoch)
    run.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="SeminarAdvancesinDeepLearning02", entity="universiteitleiden")
    wandb.agent(sweep_id, function=main, count=72)