import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from monai.config import print_config
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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
)
from monai.utils import set_determinism
import matplotlib.pyplot as plt
set_determinism(seed=0)
print_config()
import warnings
warnings.filterwarnings("ignore")

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Step 1: Data loading
base_dir = "/Users/sucharitha/Task01_BrainTumour1"  
image_dir = os.path.join(base_dir, "imagesTr")
label_dir = os.path.join(base_dir, "labelsTr")

#Extract only .nii files
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".nii.gz")])
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]

#Dataset Information
print(f"Total training samples: {len(image_files)}")
print(f"Total training labels: {len(label_files)}")

#Step 2: Transformations
transform = Compose(
    [
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
        RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]) #Post transform for validation
dataset = Dataset(data=data_dicts, transform=transform)

#Filtering non tumour labels(non zero)
data_dicts_f = []
for sample in tqdm(dataset, desc="Checking for Empty Labels"):
    label = sample["label"]  
    if torch.any(label > 0):  
        data_dicts_f.append(sample)
print(f"Original samples: {len(data_dicts)}, Filtered samples: {len(data_dicts_f)}")
data_dicts = data_dicts_f
#Update dataset
dataset_brats = Dataset(data=data_dicts_f, transform=transform)

#Post process traindata infor
train_ds = dataset[1]
print(f"Image shape: {train_ds['image'].shape}")  # (H, W, D, C)
print(f"Label shape: {train_ds['label'].shape}")  # (H, W, D)

#Step 3: Model
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,  
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True, batch=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

dice_metric = DiceMetric(include_background=True, reduction="mean")
hausdroff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)

subset_size = int(len(dataset) * 0.5) #Splitting the dataset into subsets for memory optimization
sub_dataset = Subset(dataset, range(subset_size))

#Fold
kfold = KFold(n_splits=5, shuffle=True)
scaler = torch.cuda.amp.GradScaler()
root_dir = "/Users/sucharitha"

#Training 
def train_and_validate():
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    max_epochs = 12
    best_fold = None  

    for fold, (train_index, val_index) in enumerate(kfold.split(sub_dataset)):  
        print(f"\nFold {fold + 1} Starting...")
        
        train_subset = Subset(sub_dataset, train_index)
        val_subset = Subset(sub_dataset, val_index)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)

        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0.0

            for batch_data in tqdm(train_loader, desc=f"Training Fold {fold + 1}, Epoch {epoch + 1}"):
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            epoch_loss_values.append(epoch_loss)
            print(f"Train Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}")  
            model.eval()
            dice_metric.reset()
            hausdroff_metric.reset()
            
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc=f"Validating Fold {fold + 1}, Epoch {epoch + 1}"):
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    val_outputs = [post_transforms(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    hausdroff_metric(y_pred=val_outputs, y=val_labels)

            dice_val = dice_metric.aggregate().item()
            hausdorff_val = hausdroff_metric.aggregate().item()
            print(f"Validation Dice: {dice_val:.4f}, Validation Hausdorff: {hausdorff_val:.4f}")  

            if dice_val > best_metric:
                best_metric = dice_val
                best_metric_epoch = epoch + 1
                best_fold = fold + 1  
                torch.save(model.state_dict(), f"best_metric_model_fold{fold + 1}.pth")
                print(f"New Best Dice: {best_metric:.4f} at Epoch {best_metric_epoch}")

    if best_fold is None:
        best_fold = 1 
    print(f"\nTraining complete. Best Dice found: {best_metric:.4f} at epoch {best_metric_epoch} in Fold {best_fold}")

if __name__ == '__main__':
    train_and_validate() 



