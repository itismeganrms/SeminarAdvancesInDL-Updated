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


num_epochs = 100

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
                geo_map_t1 = np.max(geo_map_t1, axis=-1)  # shape: (240, 240, 155)    print(f"Geodesic map t1 shape: {geo_map_t1.shape}")
            elif id_num == 't2':
                geo_map_t2 = img
                geo_map_t2 = np.max(geo_map_t2, axis=-1)    #print(f"Geodesic map t2 shape: {geo_map_t2.shape}")
            elif id_num == 'flair':
                geo_map_flair = img
                geo_map_flair = np.max(geo_map_flair, axis=-1) #print(f"Geodesic map flair shape: {geo_map_flair.shape}")
            elif id_num == 't1ce':
                geo_map_t1ce = img
                geo_map_t1ce = np.max(geo_map_t1ce, axis=-1)  #print(f"Geodesic map t1ce shape: {geo_map_t1ce.shape}")
        geo_map = np.stack([geo_map_t1, geo_map_t1ce, geo_map_t2, geo_map_flair], axis=0)

        input_tensor = torch.from_numpy(input_np).float()   #print(f"Input tensor shape: {input_tensor.shape}")
        # input_tensor = torch.from_numpy(input_np).unsqueeze(0)
        # print(f"Input tensor after unsqueeze: {input_tensor.shape}")

        # geo_map_prior = np.clip(geo_map, 0, 1)     
        geo_tensor = torch.from_numpy(geo_map).float()#.unsqueeze(0)    print(f"Geodesic map tensor final shape: {geo_tensor.shape}")
 
        target_tensor = torch.from_numpy(geo).long()#.unsqueeze(0)    print(f"Target tensor shape: {target_tensor.shape}")
        # target_tensor = torch.from_numpy(np.clip(seg, 0, 1)).long()
        return input_tensor, geo_tensor, target_tensor

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_cam(self, input_tensor, geo_map_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_()
        geo_map_tensor = geo_map_tensor.requires_grad_()
        
        output = self.model(input_tensor, geo_map_tensor)
        if class_idx is None:
            class_idx = torch.argmax(torch.softmax(output, dim=1))

        # One-hot for selected class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize CAM
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


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
    def __init__(self, num_classes=4):
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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1) # print(f"Logits shape: {logits.shape}, Probs shape: {probs.shape}, Targets shape: {targets.shape}")
        #targets_onehot = torch.nn.functional.one_hot(targets.squeeze(1), num_classes).permute(0, 4, 1, 2, 3).float()
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float() # print(f"Targets one-hot shape: {targets_onehot.shape}")
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

def visualize_prediction(gt_mask, pred_mask,save_dir=None, sample_id=None):
    gt_np = gt_mask.squeeze().detach().cpu().numpy()
    pred_np = pred_mask.squeeze().detach().cpu().numpy()

    if gt_np.ndim == 4: gt_np = gt_np[0]
    if pred_np.ndim == 4: pred_np = pred_np[0]
    slice_idx = gt_np.shape[0] // 2  # middle slice
    gt_slice = gt_np[:, :, slice_idx]
    pred_slice = pred_np[:, :, slice_idx]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(gt_slice, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(pred_slice, cmap='gray'); plt.title("Prediction"); plt.axis('off')


    if save_dir is not None and sample_id is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{sample_id}_slice{slice_idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Saved: {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # model = EOFormerModel(num_classes=2)
    
    model = EOFormerModel(num_classes=4).to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mri_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    geodesic_root_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_GeoLS"
    geo_map_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT/fast_sgc"
    save_dir = "/data1/courses/2024-2025/4343SADL6/Tumor_Group/Results/BRATS_Results_eoformer"
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
    
    for epoch in range(num_epochs+1):
        print(f"Starting epoch {epoch + 1}")
        model.train()
        total_loss = 0
        vis_count = 0
        for input_tensor, geo_map_tensor, gt_mask in dataloader:
            input_tensor, geo_map_tensor, gt_mask = input_tensor.to(device), geo_map_tensor.to(device), gt_mask.to(device)
            output = model(input_tensor, geo_map_tensor)
            loss = criterion(output, gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Visualize only the first 3 batches of each epoch
            if vis_count < 3:
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                visualize_prediction(gt_mask, pred, save_dir=save_dir, sample_id=f"epoch{epoch+1}_sample{vis_count}")
                vis_count += 1

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss}")

        model.eval()
        val_total_loss = 0.0
        val_dice_scores = []
        grad_cam = GradCAM3D(model, model.layer3) 

        with torch.no_grad():
            for i, (input_tensor, geo_map_tensor, gt_mask) in enumerate(val_loader):
                input_tensor, geo_map_tensor, gt_mask = input_tensor.to(device), geo_map_tensor.to(device), gt_mask.to(device)
                output = model(input_tensor, geo_map_tensor)
                pred_mask = torch.argmax(output, dim=1, keepdim=True)

                # Save visualization
                sample_id = val_samples[i]
                visualize_prediction(gt_mask, pred_mask, save_dir=save_dir, sample_id=f"{sample_id}_epoch{epoch}")

                # ---------------------
                # Run Grad-CAM on Class 1 (e.g., Tumor core)
                # ---------------------
                cam = grad_cam.generate_cam(input_tensor, geo_map_tensor, class_idx=1)  # Pick class_idx based on interest
                cam_np = cam.squeeze().detach().cpu().numpy()
                mid_slice = cam_np.shape[-1] // 2

                # Save Grad-CAM overlay
                plt.figure(figsize=(5, 5))
                plt.imshow(cam_np[0, 0, :, :, mid_slice], cmap='jet')  # Only one channel, 2D slice
                plt.title(f"Grad-CAM {sample_id} (Class 1)")
                plt.axis('off')
                os.makedirs(os.path.join(save_dir, "gradcam"), exist_ok=True)
                plt.savefig(os.path.join(save_dir, "gradcam", f"{sample_id}_cam_epoch{epoch}.png"))
                plt.close()

                # Just one sample per epoch
                break

        grad_cam.remove_hooks()
    
