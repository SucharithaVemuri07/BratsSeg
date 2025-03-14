# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import glob
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    Resize
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dir = "/content/Task01_BrainTumour1Test"
model_path = "/content/best_metric_model_fold4.pth"

images_dir = os.path.join(test_dir, "imagesTs")
labels_dir = os.path.join(test_dir, "labelsTs")

# Load test images
test_images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
test_labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

# Create test dataset
test_data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)]

test_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest"),
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),
])

# Post-process and resize predictions
def post_process_and_resize(pred_tensor, target_shape):
    pred_tensor = torch.sigmoid(pred_tensor)
    pred_tensor = (pred_tensor > 0.5).float()
    resize = Resize(spatial_size=target_shape, mode="trilinear", align_corners=True)
    resized_pred = resize(pred_tensor[0])

    return resized_pred.unsqueeze(0)

test_dataset = Dataset(data=test_data_dicts, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,

)
# Load the trained model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Metrics
dice_metric = DiceMetric(include_background=True, reduction="mean")
hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)

# Run inference and compute metrics
dice_metric.reset()
hausdorff_metric.reset()

with torch.no_grad():
    for test_data in tqdm(test_loader, desc="Testing"):
        test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)
        print(f"Label shape: {test_labels.shape}")
        test_outputs = model(test_inputs)

        # Resize the output to the original label size
        resized_outputs = post_process_and_resize(test_outputs, target_shape=test_labels.shape[2:])
        print(f"Prediction shape: {resized_outputs.shape}")
        # Compute metrics with resized predictions
        dice_metric(y_pred=resized_outputs, y=test_labels)
        hausdorff_metric(y_pred=resized_outputs, y=test_labels)

# Aggregate and print results
dice_score = dice_metric.aggregate().item()
hausdorff_score = hausdorff_metric.aggregate().item()

print(f"\nTest Dice Score: {dice_score:.4f}")
print(f"Test Hausdorff Distance: {hausdorff_score:.4f}")


def plot_results(image_path, label_path, pred_tensor):
    img = nib.load(image_path).get_fdata()  # Load MRI image
    print(f"image shape: {img.shape}")
    lbl = nib.load(label_path).get_fdata()  # Load ground truth
    print(f"label shape: {lbl.shape}")
    pred = pred_tensor.cpu().numpy()  # Convert prediction tensor to NumPy
    print(f"prediction shape: {pred.shape}")

    slice_idx = img.shape[2] // 2  # Select middle slice

    t1ce_image = img[..., 2] #selected t1ce 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1ce_image[:, :, slice_idx], cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(lbl[:, :, slice_idx], cmap="viridis")  
    axes[1].set_title("Label")
    axes[1].axis("off")
    
    axes[2].imshow(pred[0, 0, :, :, slice_idx], cmap="viridis")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

sample_idx = 0
test_sample = test_dataset[sample_idx]
test_image_path = test_data_dicts[sample_idx]["image"]
test_label_path = test_data_dicts[sample_idx]["label"]

model.eval()
with torch.no_grad():
    test_input = test_sample["image"].unsqueeze(0).to(device)  
    test_output = model(test_input)
    test_output = post_process_and_resize(test_output, target_shape=(240, 240, 155))
    
plot_results(test_image_path, test_label_path, test_output)