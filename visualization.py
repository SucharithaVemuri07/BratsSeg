import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Define file paths for both cases
cases = ["BRATS_003", "BRATS_004"]  # Add more cases if needed
base_dir = "/Users/sucharitha/Task01_BrainTumour1"
image_dir = os.path.join(base_dir, "imagesTr")
mask_dir = os.path.join(base_dir, "labelsTr")

modalities = ["FLAIR", "T1", "T1ce", "T2"]  # MRI modality names

for case in cases:
    # Load the MRI scan and corresponding segmentation mask
    image_path = os.path.join(image_dir, f"{case}.nii.gz")
    mask_path = os.path.join(mask_dir, f"{case}.nii.gz")

    try:
        image_nib = nib.load(image_path)
        mask_nib = nib.load(mask_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        continue

    # Convert to numpy arrays
    image_data = image_nib.get_fdata()  # Shape: (H, W, D, 4)
    mask_data = mask_nib.get_fdata()    # Shape: (H, W, D)

    # Select a middle slice
    slice_idx = image_data.shape[2] // 2  # Middle slice along depth axis

    # Create a figure with 2 rows: (1) Raw MRI, (2) MRI with Mask Overlay
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Visualization for {case}", fontsize=16)

    for i in range(4):
        # Display MRI modality
        axes[0, i].imshow(image_data[:, :, slice_idx, i], cmap="gray")
        axes[0, i].set_title(f"{modalities[i]} MRI")
        axes[0, i].axis("off")

        # Overlay segmentation mask
        axes[1, i].imshow(image_data[:, :, slice_idx, i], cmap="gray")
        axes[1, i].imshow(mask_data[:, :, slice_idx], alpha=0.5, cmap="jet")  # Overlay mask
        axes[1, i].set_title(f"{modalities[i]} with Mask")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()