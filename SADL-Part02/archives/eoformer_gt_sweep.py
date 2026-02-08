import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import wandb

def load_nifti(path): return nib.load(path).get_fdata()

def get_input_tensor_with_geodesic(sample_id, mri_dir, geo_dir):
    t1 = load_nifti(os.path.join(mri_dir, f"{sample_id}_t1.nii"))
    t1ce = load_nifti(os.path.join(mri_dir, f"{sample_id}_t1ce.nii"))
    t2 = load_nifti(os.path.join(mri_dir, f"{sample_id}_t2.nii"))
    flair = load_nifti(os.path.join(mri_dir, f"{sample_id}_flair.nii"))
    geo_path = os.path.join(geo_dir, sample_id, f"{sample_id}_seg_transformed.nii.gz")
    geo = load_nifti(geo_path)
    x = np.stack([t1, t1ce, t2, flair, geo], axis=0)
    x = (x - x.mean(axis=(1,2,3), keepdims=True)) / (x.std(axis=(1,2,3), keepdims=True) + 1e-8)
    return torch.from_numpy(x).float().unsqueeze(0)

def get_ground_truth(geo_path, sample_id):
    seg = load_nifti(os.path.join(geo_path, f"{sample_id}_seg_transformed.nii.gz"))
    seg = np.clip(seg, 0, 1)
    return torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)

class EOFormerModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(5, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, num_classes, 1)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return self.conv3(x)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


def train():
    wandb.init(project="SeminarAdvancesinDeepLearning", entity="universiteitleiden", tags=["EOFormer", "sweep", "geodesic"])
    config = wandb.config
    

    mri_root = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    geo_root = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"
    train_samples = [f"BraTS20_Training_{i:03d}" for i in range(1, 370)]

    model = EOFormerModel(num_classes=2, dropout_rate=config.dropout_rate)
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    criterion = DiceLoss()

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for sample_id in train_samples:
            mri_dir = os.path.join(mri_root, sample_id)
            geo_path = os.path.join(geo_root, sample_id, f"{sample_id}_seg_transformed.nii.gz")
            if not os.path.exists(geo_path): continue

            input_tensor = get_input_tensor_with_geodesic(sample_id, mri_dir, geo_root).to(device)
            gt_mask = get_ground_truth(geo_root, sample_id).to(device)

            output = model(input_tensor)
            loss = criterion(output, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_samples)
        wandb.log({"loss for epoch": avg_loss, "epoch": epoch})

    wandb.finish()

sweep_config = {
    'method': 'random',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-4, 5e-5]},
        'dropout_rate': {'values': [0.2, 0.3, 0.5]},
        'optimizer': {'values': ['adam', 'SGD']},
        'epochs': {"values": [50, 100]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="SeminarAdvancesinDeepLearning", entity="universiteitleiden")
    wandb.agent(sweep_id, function=train, count=10)  
