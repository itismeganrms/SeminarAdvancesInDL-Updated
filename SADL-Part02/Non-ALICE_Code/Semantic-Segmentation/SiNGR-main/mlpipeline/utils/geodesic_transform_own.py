import os
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import os
import SimpleITK as sitk


# Load the NIfTI image

# # Define the root directory pointing to your BRATS dataset
# root_dir = "/content/drive/My Drive/BRATS/MICCAI_BraTS2020_TrainingData"

# # Change to that directory (optional, only if you want to work from there)
# os.chdir(root_dir)

# # prompt: distance between point a and b

def distance(point_a, point_b):
  return np.linalg.norm(np.array(point_a) - np.array(point_b))


# Example usage
point_a = (1, 2, 3)
point_b = (4, 5, 6)
dist = distance(point_a, point_b)
print(f"The distance between {point_a} and {point_b} is: {dist}")

def merge_seg(seg_path):
    seg, _ = get_data(seg_path, is_seg=True)
    combined_seg = [(seg == 1) | (seg == 4), (seg == 1) | (seg == 4) | (seg == 2), seg == 4]
    combined_seg = np.stack(combined_seg, axis=0)
    return combined_seg

def get_data(input_name, is_seg=False):

    if not os.path.isfile(input_name):
        print("File not exists:", input_name)
        return -1

    img = sitk.ReadImage(input_name)
    np_img = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    if is_seg:
        return np.asarray(np_img, np.uint8), spacing_raw
    else:
        return np.asarray(np_img, np.float32), spacing_raw

def transform_geo_dist(dist, gt, label_name):
    margin = 0.5

    def _transform_layer(dist_layer, gt_layer):
        fg = (gt_layer == 1)
        bg = (gt_layer == 0)
        #dist_layer[bg] = torch.clamp_min(dist_layer[bg], -fg_max)
        dist_layer[fg] = torch.clamp_min(dist_layer[fg], 0.0)
        dist_layer[bg] = torch.clamp_max(dist_layer[bg], 0.0)

        if torch.sum(fg) == 0:
            return torch.full_like(dist_layer, -1)

        fg_max = dist_layer[fg].max()
        dist_layer[bg] = torch.clamp_min(dist_layer[bg], -fg_max)

        dist_layer = dist_layer / (fg_max + 1e-8)

        if label_name == "fast_sgc_margin":
            dist_layer[fg] = dist_layer[fg] * (1.0 - margin) + margin
            dist_layer[bg] = dist_layer[bg] * (1.0 - margin) - margin

        dist_layer = torch.clamp(dist_layer, -1.0, 1.0)

        if label_name == "fast_sgc_clamp":
            dist_layer[(gt_layer > 0) & (dist_layer <= 0)] = dist_layer[dist_layer > 0.0001].min()
            dist_layer[(gt_layer == 0) & (dist_layer >= 0)] = dist_layer[dist_layer < -0.0001].max()

        return dist_layer

    for i in range(dist.shape[-1]):
        dist[..., i] = _transform_layer(dist[..., i], gt[..., i])
    return dist

'''
def transform_gd_dist(dist, gt):
    def _transform_layer(dist_layer, gt_layer):
        fg = (gt_layer == 1)

        bg = (gt_layer == 0)
        print(f"dist_layer: {dist_layer}")
        print(f"dist_layer shape: {dist_layer.shape}")
        print(f"fg: {fg}")
        print(f"bg: {bg}")
        fg = fg.bool()
        bg = bg.bool()

        print("dist_layer shape:", dist_layer.shape)
        print("gt_layer shape:", gt_layer.shape)
        print("fg dtype:", fg.dtype, "shape:", fg.shape)


        #dist_layer[bg] = torch.clamp_min(dist_layer[bg], -fg)

        dist_layer[fg] = torch.clamp_min(dist_layer[fg], 0.52)
        dist_layer[bg] = torch.clamp_max(dist_layer[bg], 0.48)
        return dist_layer

    for i in range(dist.shape[-1]):
        dist[..., i] = _transform_layer(dist[..., i], gt[..., i])
    return dist
'''

def transform_gd_dist(dist, gt):
    def _transform_gd_dist_layer(dist_layer, gt_layer):
        # Ensure the shapes match and masks are boolean
        #assert dist_layer.shape == gt_layer.shape, "Shape mismatch!"

        fg = (gt_layer == 1)
        bg = (gt_layer == 0)

        dist_layer[fg] = torch.clamp_min(dist_layer[fg], 0.52)
        dist_layer[bg] = torch.clamp_max(dist_layer[bg], 0.48)

        return dist_layer

    # Sanity check for NaNs before processing
    #assert not torch.any(torch.isnan(dist)), "NaNs found in input distance tensor"

    # Loop over the last dimension (D)


    #if gt.ndim == 4 and gt.shape[-1] == 1:
    #new_gt = gt.squeeze(-1)
    gt = gt[..., 0]
    #print(f"new_gt.shape: {new_gt.shape}")
    print(f"gt.shape: {gt.shape}")


    for index in range(dist.shape[-1]):
        dist_layer = dist[..., index]
        gt_layer = gt[..., index]

        print("dist_layer shape:", dist_layer.shape)
        print("gt_layer shape:", gt_layer.shape)
        #print("fg dtype:", fg.dtype, "shape:", fg.shape)


        # Apply per-layer transformation
        dist[..., index] = _transform_gd_dist_layer(dist_layer, gt_layer)

    # Sanity check after processing
    assert not torch.any(torch.isnan(dist)), "NaNs introduced during transformation"

    return dist



def process_single_image(image_path, gt_path, output_path, label_name="fast_sgc_margin", dataset="BRATS", device="cpu"):
    print(f"Processing {image_path.name}")
    hard_gt_path = gt_path

    if dataset == "BRATS":
        #from your_custom_module import merge_seg  # replace with
        hard_gt = np.transpose(merge_seg(hard_gt_path), [3, 2, 1, 0])#your actual function
        #hard_gt = np.transpose(merge_seg(gt_path), [3, 2, 1, 0])
        #print(gt_path)
        #hard_gt = merge_seg(gt_path)
        #print(hard_gt)
    else:
        #from your_custom_module import get_data  # replace with your actual function
        hard_gt = get_data(gt_path, is_seg=True)[0]
        hard_gt = np.transpose(np.expand_dims(hard_gt, 0), [3, 2, 1, 0])

    hard_gt = torch.tensor(hard_gt, device=device)
    raw_geo = nib.load(image_path).get_fdata()
    geo_tensor = torch.tensor(raw_geo, device=device)

    if label_name == "gd_normed":
        transformed = transform_gd_dist(geo_tensor, hard_gt)
        print(f"transformed: {transformed}")
    else:
        transformed = transform_geo_dist(geo_tensor, hard_gt, label_name)

    result_np = transformed.cpu().numpy()
    print(f"RESULTS_NP: {result_np}")
    #result_np = np.transpose(result_np, [3, 2, 1, 0])
    print(f"output_path : {output_path}")
    os.makedirs(output_path.parent, exist_ok=True)
    nib.save(nib.Nifti1Image(result_np, affine=np.eye(4)), output_path)
    print(f"Saved to {output_path}")




if __name__ == "__main__":
    label_name = "gd_normed"
    dataset = "BRATS"

    #/content/drive/My Drixve/BRATS/MICCAI_BraTS2020_TrainingData

    image_path = Path("/content/drive/My Drive/BRATS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii")
    gt_path = Path("/content/drive/My Drive/BRATS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii")
    output_path = Path("/content/drive/My Drive/BRATS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/result.nii.gz")
    #/utils/data/MICCAI_BraTS2020_TrainingData/Patient001/pred_distance.nii.gz")
    #gt_path = Path("../utils/data/MICCAI_BraTS2020_TrainingData/Patient001/Patient001_seg.nii")
    #output_path = Path("../utils/data/MICCAI_BraTS2020_TrainingData/Patient001/Patient001_transformed.nii.gz")

    process_single_image(image_path, gt_path, output_path, label_name, dataset, device="cpu")

result_img = nib.load('/content/drive/My Drive/BRATS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/result.nii.gz')
result_data = result_img.get_fdata()

# Check if all elements are zero
all_zeros = np.any(result_data == 1)

print(f"Contents of result.nii.gz:\n{result_data}")
print(f"\nIs the array filled with only zeros?: {all_zeros}")



# If your code expects root_dir relative to some project structure
project_root = "/content/drive/My Drive/BRATS/SiNGR-main"
data_dir = os.path.join(project_root, "mlpipeline/utils/data/MICCAI_BraTS2020_TrainingData")



'''
import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# -----------------------
# Dataset Class
# -----------------------
class NiiSegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.sample_dirs = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]

        t1_file = [f for f in os.listdir(sample_path) if f.endswith("t1.nii")][0]
        t2_file = [f for f in os.listdir(sample_path) if f.endswith("t2.nii")][0]
        seg_file = [f for f in os.listdir(sample_path) if f.endswith("seg.nii")][0]

        t1 = nib.load(os.path.join(sample_path, t1_file)).get_fdata()
        t2 = nib.load(os.path.join(sample_path, t2_file)).get_fdata()
        seg = nib.load(os.path.join(sample_path, seg_file)).get_fdata()



        # Resize or crop to consistent shape if needed
        # For simplicity, assuming all images are the same shape

        # Normalize
        t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)
        t2 = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)

        # Stack channels
        image = np.stack([t1, t2], axis=0)  # shape (2, H, W)
        mask = seg[np.newaxis, ...]         # shape (1, H, W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# -----------------------
# ResNet-UNet Model
# -----------------------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        self.in_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        x0 = self.in_layer(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.center = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256 + 256, 128)
        self.dec2 = DecoderBlock(128 + 128, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x = self.center(x4, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x)
        return self.final(x)

# -----------------------
# Training Setup
# -----------------------
def train_model(model, dataloader, device, epochs=10):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

# -----------------------
# Run Everything
# -----------------------
if __name__ == "__main__":
    root_dir = "SiNGR-main/mlpipeline/utils/data/MICCAI_BraTS2020_TrainingData"  # Change this
    dataset = NiiSegmentationDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ResNetUNet(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, dataloader, device, epochs=10)


import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# -----------------------
# Dataset Class
# -----------------------
class NiiSegmentation2DSliceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for sample_dir in sorted(os.listdir(root_dir)):
            full_dir = os.path.join(root_dir, sample_dir)
            print(full_dir)

            if not os.path.isdir(full_dir):
                continue

            if not os.path.isdir(full_dir):
                continue

            if len(os.listdir(full_dir)) == 0:
                continue

            try:
                t1_file = [f for f in os.listdir(full_dir) if f.endswith("t1.nii")][0]
                t2_file = [f for f in os.listdir(full_dir) if f.endswith("t2.nii")][0]
                seg_file = [f for f in os.listdir(full_dir) if f.endswith("seg.nii")][0]
            except IndexError:
                print(f"Missing modality in {full_dir}, skipping.")
                continue

            #print(len(os.listdir(full_dir)))
            #t1_file = [f for f in os.listdir(full_dir) if f.endswith("t1.nii")][0]
            #t2_file = [f for f in os.listdir(full_dir) if f.endswith("t2.nii")][0]
            #seg_file = [f for f in os.listdir(full_dir) if f.endswith("seg.nii")][0]

            t1_data = nib.load(os.path.join(full_dir, t1_file)).get_fdata()
            t2_data = nib.load(os.path.join(full_dir, t2_file)).get_fdata()
            seg_data = nib.load(os.path.join(full_dir, seg_file)).get_fdata()

            # Normalize
            t1_data = (t1_data - t1_data.min()) / (t1_data.max() - t1_data.min() + 1e-8)
            t2_data = (t2_data - t2_data.min()) / (t2_data.max() - t2_data.min() + 1e-8)

            depth = t1_data.shape[2]
            for i in range(depth):
                self.samples.append((
                    t1_data[:, :, i],
                    t2_data[:, :, i],
                    seg_data[:, :, i]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t1, t2, seg = self.samples[idx]
        image = np.stack([t1, t2], axis=0)        # (2, H, W)
        mask = np.expand_dims(seg, axis=0)        # (1, H, W)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


    def __getitem__(self, idx):
        t1, t2, seg = self.samples[idx]
        x = torch.cat([t1, t2], dim=0)  # [2, H, W]
        if x.shape[0] == 2:
            third_channel = torch.zeros_like(t1)
            x = torch.cat([x, third_channel], dim=0)


        image = np.stack([t1, t2], axis=0)        # (2, H, W)
        mask = np.expand_dims(seg, axis=0)        # (1, H, W)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# -----------------------
# ResNet-UNet Model
# -----------------------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(pretrained=True)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4
   
        resnet = models.resnet34(pretrained=True)
        self.initial = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Copy the pretrained weights for remaining layers
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4


    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        skip = nn.functional.interpolate(skip, size=(16, 16), mode='nearest')  # or mode='bilinear' if it's an image
        x = torch.cat([x, skip], dim=1)

        #skip = nn.functional.interpolate(skip, size=(16, 16), mode='nearest')  # or mode='bilinear' if it's an image
        #x = torch.cat([x, skip], dim=1)


        #x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        print(x.shape, skip.shape)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.center = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256 + 256, 128)
        self.dec2 = DecoderBlock(128 + 128, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x = self.center(x4, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x)
        return self.final(x)

# -----------------------
# Training Function
# -----------------------
def train_model(model, dataloader, device, epochs=5):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

# -----------------------
# Main Entry Point
# -----------------------
if __name__ == "__main__":
    project_root = "/content/drive/My Drive/BRATS/"
    root_dir = os.path.join(project_root, "MICCAI_BraTS2020_TrainingData")


    #root_dir = "SiNGR-main/mlpipeline/utils/data/MICCAI_BraTS2020_TrainingData"  # 
    dataset = NiiSegmentation2DSliceDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = ResNetUNet(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, dataloader, device, epochs=10)




!pip install nibabel

'''
