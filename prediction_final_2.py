# -*- coding: utf-8 -*-
"""Prediction_final-2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mUQJawko9B4XIIDA4DPudq0kGMzkE0m0
"""

import os
import numpy as np
import nibabel as nib
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    CenterSpatialCropd
)
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
test_dir = "/content/Task01_BrainTumour1Test"
model_path = "/content/best_metric_model_fold2.pth"

# Define model
model = UNet(
    spatial_dims=3,
    in_channels=4,     # 4 input channels for BRATS data (T1, T1ce, T2, FLAIR)
    out_channels=3,    # 3 output channels for tumor regions (ET, TC, WT)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Load the trained model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define test transforms
test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(
        keys=["image"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear"),
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    CenterSpatialCropd(keys=["image"], roi_size=[128, 128, 128]),
])

# Define post-processing transforms
post_transforms = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5)
])

# Prepare test dataset
test_image_dir = os.path.join(test_dir, "imagesTs")
test_image_files = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(".nii.gz")])
test_data_dicts = [{"image": img} for img in test_image_files]

# Create test dataset and dataloader
test_dataset = Dataset(data=test_data_dicts, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Run inference on test data
predictions = []
with torch.no_grad():
    for test_data in tqdm(test_loader, desc="Predicting"):
        test_image = test_data["image"].to(device)

        # Use sliding window inference for better results on large volumes
        pred_output = sliding_window_inference(
            inputs=test_image,
            roi_size=[128, 128, 128],
            sw_batch_size=4,
            predictor=model,
            overlap=0.5
        )

        # Apply post-processing
        pred_output = post_transforms(pred_output)
        pred_output = [i for i in decollate_batch(pred_output)]
        predictions.append(pred_output)

# Function to visualize a slice from 3D volumes
def visualize_segmentation(image, prediction, slice_num=69):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the FLAIR image (usually the most informative for tumor visualization)
    if image.shape[0] == 4:  # If we have all 4 modalities
        axes[0].imshow(image[3, :, :, slice_num], cmap="gray")  # FLAIR is typically the 4th channel
    else:
        axes[0].imshow(image[2, :, :, slice_num], cmap="gray")

    axes[0].set_title("Original Image")

    # Create a color overlay for the different tumor regions
    seg_map = np.zeros((image.shape[1], image.shape[2], 3))

    # Color coding: Red for ET (label 4), Green for TC (labels 1,4), Blue for WT (labels 1,2,4)
    if prediction.shape[0] == 3:  # If our prediction has 3 channels
        seg_map[:, :, 0] = prediction[0, :, :, slice_num]  # ET - Red
        seg_map[:, :, 1] = prediction[1, :, :, slice_num]  # TC - Green
        seg_map[:, :, 2] = prediction[2, :, :, slice_num]  # WT - Blue

    axes[1].imshow(image[0, :, :, slice_num], cmap="gray")
    axes[1].imshow(seg_map, alpha=0.6)
    axes[1].set_title("Segmentation Overlay")

    plt.tight_layout()
    plt.show()

# Select a test sample and visualize
sample_idx = 0  # Change this index to visualize different test samples
test_sample = test_dataset[sample_idx]["image"]
prediction_sample = predictions[sample_idx][0].cpu().numpy()

# Visualize multiple slices
for slice_num in [60, 70, 80, 90]:
    print(f"Visualizing slice {slice_num}")
    visualize_segmentation(test_sample, prediction_sample, slice_num=slice_num)

print("Segmentation completed")