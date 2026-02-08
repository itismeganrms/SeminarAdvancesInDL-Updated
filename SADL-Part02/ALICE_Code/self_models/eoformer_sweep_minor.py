import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import time
import datetime
import pylab as pyl
 
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, GeneralizedWassersteinDiceLoss
from monai.metrics import DiceMetric

from monai.losses import DiceFocalLoss
from monai.losses import DiceCELoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, GeneralizedWassersteinDiceLoss

sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "learning_rate": {"values": [1e-4, 1e-5, 1e-3, 1e-6]},
        "epochs": {"values": [30, 35, 40, 45, 50, 55, 60]},
        "optimizer": {"values": ["Adam", "SGD", "AdamW"]},
        "batch_size": {"values": [1, 2, 4]},
        "loss_function": {"values": ["DiceLoss", "DiceCELoss"]},
    }
}

class BraTSDataset(Dataset):
    def __init__(self, sample_ids, mri_root_dir, geodesic_root_dir, geo_map_dir, transform=None):
        self.sample_ids = sample_ids
        self.mri_root_dir = mri_root_dir
        self.geodesic_root_dir = geodesic_root_dir
        self.geo_map_dir = geo_map_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)
    
    @staticmethod
    def load_nifti(file_path):
        img = nib.load(file_path)        
        return img.get_fdata()

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        mri_dir = os.path.join(self.mri_root_dir, sample_id)
        geo_map_dir = os.path.join(self.geo_map_dir, sample_id)
        geo_path = os.path.join(self.geodesic_root_dir, sample_id, f"{sample_id}_seg_transformed.nii.gz")
        if not os.path.exists(geo_path):
            raise FileNotFoundError(f"Missing geodesic file: {geo_path}")
        
        input_names = ['t1', 't2', 'flair', 't1ce']
        t1, t1ce, t2, flair, geo = [], [], [], [], []
        for id_num in input_names:
            file_path = os.path.join(mri_dir, f"{sample_id}_{id_num}.nii")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing MRI file: {file_path}")
            img = self.load_nifti(file_path)
            if id_num == 't1':
                t1 = img                #print(f"t1 shape: {t1.shape}")
            elif id_num == 't2':
                t2 = img                #print(f"t2 shape: {t2.shape}")
            elif id_num == 'flair':
                flair = img             #print(f"flair shape: {flair.shape}")
            elif id_num == 't1ce':
                t1ce = img          #print(f"t1ce shape: {t1ce.shape}")
        
        input_np = np.stack([t1, t1ce, t2, flair], axis=0)
        input_np = (input_np - input_np.mean(axis=(1, 2, 3), keepdims=True)) / (input_np.std(axis=(1, 2, 3), keepdims=True) + 1e-8)
        
        geo = self.load_nifti(geo_path)
        #seg = self.load_nifti(os.path.join(mri_dir, f"{sample_id}_seg.nii"))
        geo[geo == 4] = 3  

        geo_map_t1, geo_map_t2, geo_map_flair, geo_map_t1ce = [], [], [], []
        for id_num in input_names:
            file_path = os.path.join(geo_map_dir, f"{sample_id}_fast_sg_{id_num}.nii.gz")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing MRI file: {file_path}")
            img = self.load_nifti(file_path)
            if id_num == 't1':
                geo_map_t1 = img
                geo_map_t1 = np.max(geo_map_t1, axis=-1) 
            elif id_num == 't2':
                geo_map_t2 = img
                geo_map_t2 = np.max(geo_map_t2, axis=-1)
            elif id_num == 'flair':
                geo_map_flair = img
                geo_map_flair = np.max(geo_map_flair, axis=-1) 
            elif id_num == 't1ce':
                geo_map_t1ce = img
                geo_map_t1ce = np.max(geo_map_t1ce, axis=-1)
        geo_map = np.stack([geo_map_t1, geo_map_t1ce, geo_map_t2, geo_map_flair], axis=0)

        input_tensor = torch.from_numpy(input_np).float()
        geo_tensor = torch.from_numpy(geo_map).float()
        target_tensor = torch.from_numpy(geo).long()
        return input_tensor, geo_tensor, target_tensor

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_p=0.3):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)


class EOFormerModel(nn.Module):
    def __init__(self, num_classes=1):
        super(EOFormerModel, self).__init__()
        self.layer1 = ConvBlock3D(8, 16)
        self.layer2 = ConvBlock3D(16, 32)
        self.layer3 = ConvBlock3D(32, 64)
        self.final_layer = ConvBlock3D(64, num_classes, kernel_size=1)

    def forward(self, x, geo_map):
        inputs = torch.cat((x, geo_map))
        print(f"Input shape: {inputs.shape}, Geodesic map shape: {geo_map.shape}")
        x = torch.cat((x, geo_map), dim=1) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_layer(x)
        return x  

def visualize_prediction(gt_mask, pred_mask, save_dir=None, sample_id=None):
    # print(f"gt_mask shape: {gt_mask.shape}, pred_mask shape: {pred_mask.shape}")
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run = wandb.init()
    run.tags = ["3D-CNN", "BraTS2020", "DiceLoss", "EOFormer"]
    config = wandb.config
    model = EOFormerModel(num_classes=1).to(device)

    if config.loss_function == "DiceCELoss":
        criterion = DiceCELoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "DiceFocalLoss":
        criterion = DiceFocalLoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "GeneralizedDiceLoss":
        criterion = GeneralizedDiceLoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "GeneralizedDiceFocalLoss":
        criterion = GeneralizedDiceFocalLoss(sigmoid=True, reduction="mean")
    elif config.loss_function == "GeneralizedWassersteinDiceLoss":
        criterion = GeneralizedWassersteinDiceLoss()
    else:
        criterion = DiceLoss(sigmoid=True, reduction="mean")

    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

    mri_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    geodesic_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"
    geo_map_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT/fast_sgc"
    save_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Results/BRATS_Results_eoformer_timestamp_{}".format(timestamp)
    train_csv ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/train_ids_seed_34.csv"
    val_csv ="/data1/courses/2024-2025/4343SADL6/Tumor_Group/SeminarAdvancesInDL/ALICE_Code/SwinUNETR/BRATS21/splits_data_csv/val_ids_seed_34.csv"

    train_csv = pd.read_csv(train_csv)
    val_csv = pd.read_csv(val_csv)   
    fold = 2
    
    if fold is not None:
        train_csv = train_csv[train_csv['fold'] == fold]
        val_csv = val_csv[val_csv['fold'] == fold]

        train_ids = train_csv['ID'].tolist()
        val_ids = val_csv['ID'].tolist()

        train_folds = train_csv['fold'].tolist()
        val_folds = val_csv['fold'].tolist()
    
    train_samples = [f"BraTS20_Training_{i:03d}" for i in train_ids]
    val_samples = [f"BraTS20_Training_{i:03d}" for i in val_ids]

    dataset = BraTSDataset(train_samples, mri_root_dir, geodesic_root_dir, geo_map_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_dataset = BraTSDataset(val_samples, mri_root_dir, geodesic_root_dir, geo_map_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    for epoch in range(config.epochs + 1):
        print(f"Starting epoch {epoch + 1}")
        model.train()
        total_loss = 0
        vis_count = 0

        # TRAINING LOOP
        for input_tensor, geo_map_tensor, gt_mask in dataloader:
            print(f"Entering training loop for {epoch + 1}")
            input_tensor, geo_map_tensor, gt_mask = input_tensor.to(device), geo_map_tensor.to(device), gt_mask.to(device)
            output = model(input_tensor, geo_map_tensor)
            unsqueezed_gtmask = gt_mask.unsqueeze(1)
            loss = criterion(output, unsqueezed_gtmask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Visualize only the first 3 batches of each epoch
            if vis_count < 3:
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                visualize_prediction(gt_mask, pred, save_dir=save_dir, sample_id=f"epoch{epoch+1}_sample{vis_count}")
                vis_count += 1

        avg_train_loss = round(total_loss / len(dataloader), 4)
        print(f"[Epoch {epoch + 1}] Avg Train Loss: {avg_train_loss}")
        run.log({"loss": avg_train_loss, "epoch": epoch + 1})

        # VALIDATION LOOP
        model.eval()
        val_total_loss = 0.0
        val_dice_scores = []

        with torch.no_grad():
            for idx, (input_tensor, geo_map_tensor, gt_mask) in enumerate(val_loader):
                print(f"Entering validation loop for {epoch + 1}")
                input_tensor = input_tensor.to(device)
                geo_map_tensor = geo_map_tensor.to(device)
                gt_mask = gt_mask.to(device)
                output = model(input_tensor, geo_map_tensor)
                unsqueezed_gtmask = gt_mask.unsqueeze(1)
                loss = criterion(output, unsqueezed_gtmask)
                val_total_loss += loss.item()
                val_dice_scores.append(1 - loss.item())
                
            val_avg_loss = round(val_total_loss / len(val_loader), 4)
            val_avg_dice = round(sum(val_dice_scores) / len(val_dice_scores), 4)
            run.log({"val_loss": val_avg_loss, "val_dice_score": val_avg_dice, "epoch": epoch + 1})

    run.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="SeminarAdvancesinDeepLearning", entity="universiteitleiden")
    wandb.agent(sweep_id, function=main, count=20)
