import os
import torch
import click
import nibabel as nib
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import shutil

def merge_seg(seg_path):
    seg, _ = get_data(seg_path, is_seg=True)
    combined_seg = [
        (seg == 4),                          # Enhancing Tumor (ET)
        (seg == 1) | (seg == 4),             # Tumor Core (TC)
        (seg == 1) | (seg == 2) | (seg == 4) # Whole Tumor (WT)
    ]

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


def transform_gd_dist(dist, gt):
    def _transform_gd_dist_layer(dist_layer, gt_layer):
        fg = (gt_layer == 1)
        bg = (gt_layer == 0)
        dist_layer[fg] = torch.clamp_min(dist_layer[fg], 0.52)
        dist_layer[bg] = torch.clamp_max(dist_layer[bg], 0.48)
        return dist_layer
    gt = gt[..., 0]
    print(f"gt.shape: {gt.shape}")
    for index in range(dist.shape[-1]):
        dist_layer = dist[..., index]
        gt_layer = gt[..., index]
        print("dist_layer shape:", dist_layer.shape)
        print("gt_layer shape:", gt_layer.shape)
        dist[..., index] = _transform_gd_dist_layer(dist_layer, gt_layer)
    return dist

def process_single_image(image_path, gt_path, output_path, label_name="fast_sgc_margin", dataset="BRATS", device="cuda"):
    print(f"Processing {image_path.name}")
    hard_gt_path = gt_path
    print(hard_gt_path.shape)
    print(f"hard_gt_path: {hard_gt_path}")
    if dataset == "BRATS":
        hard_gt = np.transpose(merge_seg(hard_gt_path), [3, 2, 1, 0])
    else:
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



def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.0")

    cases = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    random.shuffle(cases)  # Shuffle cases for randomness

    num_cases = len(cases)
    num_train = int(num_cases * train_ratio)
    num_val = int(num_cases * val_ratio)
    num_test = num_cases - num_train - num_val  # Ensure all cases are used

    train_cases = cases[:num_train]
    val_cases = cases[num_train:num_train + num_val]
    test_cases = cases[num_train + num_val:]

    splits = {"train": train_cases, "val": val_cases, "test": test_cases}

    for split_name, case_list in splits.items():
        output_dir = Path(root_dir) / split_name
        output_dir.mkdir(exist_ok=True)
        for case_name in case_list:
            source_dir = Path(root_dir) / case_name
            destination_dir = output_dir / case_name
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
            print(f"Moved {case_name} to {destination_dir}")


def visualize_slice(image_path, slice_index=50):
    img = nib.load(str(image_path))
    data = img.get_fdata()

    # Ensure slice_index doesn't exceed dimensions
    slice_index = min(slice_index, data.shape[2] - 1)
    slice_data = data[:, :, slice_index]

    plt.imshow(slice_data.T, cmap='gray', origin='lower')
    plt.title(f"{image_path.parent.name} - {image_path.name} (Slice {slice_index})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    #DO GEODESIC TRANSFORM
    dataset = "BRATS"
    input_folder = Path("/data1/courses/2024-2025/4343SADL6/Tumor_Group/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    gt_folder = Path("/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/Updated_GT/fast_sgc")
    output_root = Path("/data1/courses/2024-2025/4343SADL6/Tumor_Group/Datasets/BraTS_UpdatedGT_OP")
    output_root.mkdir(exist_ok=True)

    input_suffixes = ["_t1.nii", "_t1ce.nii", "_seg.nii", "_flair.nii", "_t2.nii"]
    # suffixes = ["_seg.nii"]

    # for zip(case_dir in input_folder.iterdir(), gt_file_dir in gt_folder.iterdir()):        
    #     if case_dir.is_dir() and case_dir.name.startswith("BraTS20_Training_"):
    #         case_id = case_dir.name
    #         output_case_dir = output_root / case_id
    #         output_case_dir.mkdir(exist_ok=True)

    #         for suffix in input_suffixes:
    #             image_file = case_dir / f"{case_id}{suffix}"
    #             if image_file.exists():
    #                 output_file = output_case_dir / f"{case_id}{suffix.replace('.nii', '_transformed.nii.gz')}"
    #                 process_single_image(image_file, gt_file, output_file)
    #                 print(f"Processed {image_file} -> {output_file}")
    case_dirs = sorted([d for d in input_folder.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")])
    gt_files = sorted([d for d in gt_folder.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")])

    for case_dir, gt_file in zip(case_dirs, gt_files):
        case_id = case_dir.name
        output_case_dir = output_root / case_id
        output_case_dir.mkdir(exist_ok=True)

        for suffix in input_suffixes:
            image_file = case_dir / f"{case_id}{suffix}"
            print(f"image_file: {image_file}")
            gt_file = gt_file / f"{case_id}_fast_sg{suffix}.gz"
            print(f"gt_file: {gt_file}")
            if image_file.exists():
                output_file = output_case_dir / f"{case_id}{suffix.replace('.nii', '_transformed.nii.gz')}"
                process_single_image(image_file, gt_file, output_file)