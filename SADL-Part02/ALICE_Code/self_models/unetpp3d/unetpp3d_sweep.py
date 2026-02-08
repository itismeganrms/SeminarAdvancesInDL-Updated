import os
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
import wandb
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, ResizeWithPadOrCropd,
    ToTensord, Activations, AsDiscrete
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.losses import DiceCELoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, GeneralizedWassersteinDiceLoss

import time
import datetime
from mpl_toolkits.mplot3d import Axes3D

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 1e-5, 1e-3, 1e-6]},
        "epochs": {"values": [30, 50, 75]},
        "optimizer": {"values": ["Adam", "SGD", "AdamW"]},
        "batch_size": {"values": [1, 2]},
        "loss_function": {"values": ["DiceLoss", "DiceCELoss", "DiceFocalLoss", "GeneralizedDiceLoss", "GeneralizedDiceFocalLoss", "GeneralizedWassersteinDiceLoss"]},
    }
}

# --- Model definition ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels=4, n_classes=1, base_n_filter=32, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        n1, n2, n3, n4 = base_n_filter, base_n_filter*2, base_n_filter*4, base_n_filter*8

        self.conv0_0 = ConvBlock(input_channels, n1)
        self.conv1_0 = ConvBlock(n1, n2)
        self.conv2_0 = ConvBlock(n2, n3)
        self.conv3_0 = ConvBlock(n3, n4)
        self.maxpool = nn.MaxPool3d(2, 2)

        self.up1_0 = nn.ConvTranspose3d(n2, n1, 2, 2)
        self.conv0_1 = ConvBlock(n1*2, n1)

        self.up2_0 = nn.ConvTranspose3d(n3, n2, 2, 2)
        self.conv1_1 = ConvBlock(n2*2, n2)
        self.up1_1 = nn.ConvTranspose3d(n2, n1, 2, 2)
        self.conv0_2 = ConvBlock(n1*3, n1)

        self.up3_0 = nn.ConvTranspose3d(n4, n3, 2, 2)
        self.conv2_1 = ConvBlock(n3*2, n3)
        self.up2_1 = nn.ConvTranspose3d(n3, n2, 2, 2)
        self.conv1_2 = ConvBlock(n2*3, n2)
        self.up1_2 = nn.ConvTranspose3d(n2, n1, 2, 2)
        self.conv0_3 = ConvBlock(n1*4, n1)

        self.final = nn.Conv3d(n1, n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        return self.final(x0_3)

# --- Data parsing ---
def get_data_with_geo(data_dir, geo_dir, geo_map_dir, id_list):
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


# --- Main training loop ---
def main():
    run = wandb.init()
    config = wandb.config

    if config.loss_function == "DiceCELoss":
        loss_function = DiceCELoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "DiceFocalLoss":
        loss_function = DiceFocalLoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "GeneralizedDiceLoss":
        loss_function = GeneralizedDiceLoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "GeneralizedDiceFocalLoss":
        loss_function = GeneralizedDiceFocalLoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "GeneralizedWassersteinDiceLoss":
        loss_function = GeneralizedWassersteinDiceLoss()
    else:
        loss_function = DiceLoss(sigmoid=True, reduction="mean")

    model = UNetPlusPlus(input_channels=8, n_classes=1).to("cuda" if torch.cuda.is_available() else "cpu")

    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.8)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", required=True)
    # # parser.add_argument("--train_csv", required=True)
    # # parser.add_argument("--val_csv", required=True)
    # parser.add_argument("--max_epochs", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--fold", type=int, default=None, help="Fold index to use for training and validation")
    # args = parser.parse_args()

    data_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    train_csv1="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/train_ids_seed_34.csv"
    val_csv1="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/val_ids_seed_34.csv"
    
    geo_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"
    geo_map_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT/fast_sgc"
    
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Results/BRATS_Results_unetpp3d_timestamp_{}".format(timestamp)
    
    # train_csv = pd.read_csv(train_csv)
    # val_csv = pd.read_csv(val_csv)
    # Wandb login and init
    train_csv = pd.read_csv(train_csv1)
    val_csv = pd.read_csv(val_csv1)
    fold = 3

    if fold is not None:
        train_csv = train_csv[train_csv["fold"] == fold]
        val_csv = val_csv[val_csv["fold"] == fold]

    train_ids = train_csv["ID"].tolist()
    val_ids = val_csv["ID"].tolist()

    train_files = get_data_with_geo(data_dir, geo_dir, geo_map_dir, train_ids)
    val_files = get_data_with_geo(data_dir, geo_dir, geo_map_dir, val_ids)
    patch_size = (64, 64, 64)

    print(f"Loaded training samples: {len(train_files)}", flush=True)
    print(f"Loaded validation samples: {len(val_files)}", flush=True)

    train_transforms = Compose([
        LoadImaged(keys=["image", "geo", "label"]),
        EnsureChannelFirstd(keys=["image", "geo", "label"]),
        Spacingd(keys=["image", "geo", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "bilinear", "nearest")),
        Orientationd(keys=["image", "geo", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image", "geo"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "geo", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "geo", "label"], spatial_size=(64, 64, 64)),
    ToTensord(keys=["image", "geo", "label"]),
    ])


    val_transforms = Compose([
        LoadImaged(keys=["image", "geo", "label"]),
        EnsureChannelFirstd(keys=["image", "geo", "label"]),
        Spacingd(keys=["image", "geo", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "bilinear", "nearest")),
        Orientationd(keys=["image", "geo", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image", "geo"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "geo", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "geo", "label"], spatial_size=(64, 64, 64)),
        ToTensord(keys=["image", "geo", "label"]),
    ])

    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds = Dataset(val_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)


    device = next(model.parameters()).device

    # loss_function = DiceLoss(sigmoid=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}", flush=True)

        # --- Training ---
        model.train()
        epoch_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_start = time.time()
            image = batch_data["image"].to(device)
            geo = batch_data["geo"][:, :4].to(device)
            inputs = torch.cat((image, geo), dim=1)  # [B, 8, 64, 64, 64]
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #print(f"Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Time: {time.time() - batch_start:.2f}s", flush=True)

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation Loss ---
        model.eval()
        val_loss = 0.0
        dice_metric.reset()

        with torch.no_grad():
            for val_data in val_loader:
                image = val_data["image"].to(device)
                geo = val_data["geo"][:, :4].to(device)
                val_inputs = torch.cat((image, geo), dim=1)
                val_labels = val_data["label"].to(device)

                val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()

                val_outputs = post_pred(val_outputs)  # sigmoid + threshold
                dice_metric(y_pred=val_outputs, y=val_labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = dice_metric.aggregate().item()

        print(f"Validation Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")
        wandb.log({
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
        })

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

    # Final Dice Score
    model.eval()
    with torch.no_grad():
        dice_metric.reset()
        for val_data in val_loader:
            image = val_data["image"].to(device)
            geo = val_data["geo"][:, :4].to(device)
            val_inputs = torch.cat((image, geo), dim=1)
            val_labels = val_data["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, patch_size, 1, model)
            val_outputs = post_pred(val_outputs)
            dice_metric(y_pred=val_outputs, y=val_labels)
        final_score = dice_metric.aggregate().item()
        print(f"Final Validation Dice Score: {final_score:.4f}", flush=True)
        wandb.log({"final_val_dice_score": final_score})
        dice_metric.reset()

    wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="SeminarAdvancesinDeepLearning", entity="universiteitleiden")
    wandb.agent(sweep_id, function=main, count=20)