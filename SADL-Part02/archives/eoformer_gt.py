import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import wandb


num_epochs = 10

run = wandb.init(
    entity="universiteitleiden",
    project="SeminarAdvancesinDeepLearning",
    tags=["3D-CNN", "BraTS2020", "DiceLoss", "segmentation"],
    config={
        "learning_rate": 1e-4,
        "architecture": "EOFormerModel",
        "loss_function": "DiceLoss",
        "epochs": num_epochs,
        "optimizer": "Adam"
    }
)

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def get_input_tensor_with_geodesic(formatted_sample_num, mri_dir, geodesic_dir):
    t1 = load_nifti(os.path.join(mri_dir, f"{formatted_sample_num}_t1.nii"))
    t1ce = load_nifti(os.path.join(mri_dir, f"{formatted_sample_num}_t1ce.nii"))
    t2 = load_nifti(os.path.join(mri_dir, f"{formatted_sample_num}_t2.nii"))
    flair = load_nifti(os.path.join(mri_dir, f"{formatted_sample_num}_flair.nii"))

    # geo_path = os.path.join(geodesic_dir, formatted_sample_num, f"{formatted_sample_num}_seg_transformed.nii.gz")
    # geodesic = load_nifti(geo_path)

    input_np = np.stack([t1, t1ce, t2, flair], axis=0)
    input_np = (input_np - input_np.mean(axis=(1, 2, 3), keepdims=True)) / (
        input_np.std(axis=(1, 2, 3), keepdims=True) + 1e-8
    )
    return torch.from_numpy(input_np).float().unsqueeze(0)  # [1, 5, D, H, W]

def get_ground_truth(geo_file:
    seg = load_nifti(os.path.join(geo_path, f"{sample_id}_seg_transformed.nii.gz"))
    seg = np.clip(seg, 0, 1)
    return torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)

#### this was added to the original code
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
####

class EOFormerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EOFormerModel, self).__init__()
        # self.conv1 = nn.Conv3d(5, 16, kernel_size=3, padding=1)  # <- 5 input channels
        # self.bn1 = nn.BatchNorm3d(16)
        # self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm3d(32)
        # self.conv3 = nn.Conv3d(32, num_classes, kernel_size=1)
        # self.dropout = nn.Dropout3d(p=0.3)
        self.layer1 = ConvBlock3D(5, 16)
        self.layer2 = ConvBlock3D(16, 32)
        self.final_layer = ConvBlock3D(32, num_classes, kernel_size=1)

    def forward(self, x):
        # x = torch.relu(self.bn1(self.conv1(x)))
        # x = self.dropout(x)
        # x = torch.relu(self.bn2(self.conv2(x)))
        # x = self.dropout(x)
        # x = self.conv3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final_layer(x)
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

def visualize_prediction(gt_mask, pred_mask, slice_idx=77):
    gt_np = gt_mask.squeeze().detach().cpu().numpy()
    pred_np = pred_mask.squeeze().detach().cpu().numpy()

    if gt_np.ndim == 4: gt_np = gt_np[0]
    if pred_np.ndim == 4: pred_np = pred_np[0]

    gt_slice = gt_np[:, :, slice_idx]
    pred_slice = pred_np[:, :, slice_idx]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(gt_slice, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(pred_slice, cmap='gray'); plt.title("Prediction"); plt.axis('off')
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # model = EOFormerModel(num_classes=2)
    
    model = EOFormerModel(num_classes=2).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mri_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    geodesic_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"

    train_samples = [f"BraTS20_Training_{i:03d}" for i in range(1, 370)]

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        model.train()
        total_loss = 0

        for sample_num in train_samples:
            mri_dir = os.path.join(mri_root_dir, sample_num)
            geo_file = os.path.join(geodesic_root_dir, sample_num, f"{sample_num}_seg_transformed.nii.gz")

            if not os.path.exists(geo_file):
                print(f"[SKIP] {sample_num} - missing geodesic file.")
                continue

            input_tensor = get_input_tensor_with_geodesic(sample_num, mri_dir, geodesic_root_dir).to(device)
            gt_mask = get_ground_truth(geo_file, sample_num).to(device)

            output = model(input_tensor)
            loss = criterion(output, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = round(total_loss / len(train_samples), 4)
        run.log({"loss for epoch": avg_loss})
        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss}")

    run.finish()

    model.eval()
    with torch.no_grad():
        for sample_num in train_samples[:3]:  # visualize a few examples
            mri_dir = os.path.join(mri_root_dir, sample_num)
            geo_file = os.path.join(geodesic_root_dir, sample_num, f"{sample_num}_seg_transformed.nii.gz")
            if not os.path.exists(geo_file): continue

            input_tensor = get_input_tensor_with_geodesic(sample_num, mri_dir, geodesic_root_dir)
            gt_mask = get_ground_truth(mri_dir, sample_num)
            output = model(input_tensor)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

            visualize_prediction(gt_mask, pred)

# "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii"
# '/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii'