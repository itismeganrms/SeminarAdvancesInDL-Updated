import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import wandb

# Define sweep configuration
sweep_configuration = {
    "method": "grid",
    "parameters": {
        "learning_rate": {"values": [1e-4, 1e-5, 1e-3]},
        "epochs": {"values": [10, 20, 30]},
        "optimizer": {"values": ["Adam", "SGD"]},
        # "batch_size": {"values": [16, 32, 64]},
        "loss_function": {"values": ["DiceLoss", "CrossEntropyLoss"]},
    }
}

# Utility Functions
def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def get_input_tensor(formatted_sample_num, sample_dir):
    t1 = load_nifti(os.path.join(sample_dir, f"{formatted_sample_num}_t1.nii"))
    t1ce = load_nifti(os.path.join(sample_dir, f"{formatted_sample_num}_t1ce.nii"))
    t2 = load_nifti(os.path.join(sample_dir, f"{formatted_sample_num}_t2.nii"))
    flair = load_nifti(os.path.join(sample_dir, f"{formatted_sample_num}_flair.nii"))

    input_np = np.stack([t1, t1ce, t2, flair], axis=0)
    input_np = (input_np - input_np.mean(axis=(1, 2, 3), keepdims=True)) / (
        input_np.std(axis=(1, 2, 3), keepdims=True) + 1e-8
    )
    input_tensor = torch.from_numpy(input_np).float().unsqueeze(0)
    return input_tensor

def get_ground_truth(sample_dir, sample_num):
    seg = load_nifti(os.path.join(sample_dir, f"{sample_num}_seg.nii"))
    seg = np.clip(seg, 0, 1)
    gt_tensor = torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)
    return gt_tensor

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(
            targets.squeeze(1), num_classes
        ).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

class EOFormerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EOFormerModel, self).__init__()
        self.conv1 = nn.Conv3d(4, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, num_classes, kernel_size=1)
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        return x

def main():
    run = wandb.init()
    config = wandb.config

    model = EOFormerModel(num_classes=2)

    if config.loss_function == "DiceLoss":
        criterion = DiceLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    train_samples = [f"BraTS20_Training_{i:03d}" for i in range(1, 370)]
    sample_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch}")
        model.train()
        total_loss = 0
        for sample_num in train_samples:
            sample_dir = os.path.join(sample_root_dir, sample_num)
            input_tensor = get_input_tensor(sample_num, sample_dir)
            gt_mask = get_ground_truth(sample_dir, sample_num)

            output = model(input_tensor)
            loss = criterion(output, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_per_run = round(total_loss / len(train_samples), 4)
        wandb.log({"epoch": epoch + 1, "loss": loss_per_run})
        print(f"Epoch {epoch + 1}, Loss: {loss_per_run}")

    run.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="SeminarAdvancesinDeepLearning", entity="universiteitleiden")
    wandb.agent(sweep_id, function=main, count=10)
