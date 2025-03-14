# Deep Learning-based Brain Tumor Segmentation Using MRI

## Overview
This work walks through a 3D U-Net architecture implementation on brain tumor segmentation downstream task using BraTs challenge data. The multi-modal MRI data consists of four-time points: FLAIR, T1w, t1gd, and T2w. The goal is to build a **2D or 3D U-Net** architecture for segmentation, training the model under the following labeled classes:
- **0** - Background
- **1** - Necrotic and Non-enhancing Tumor
- **2** - Peritumoral Edema
- **4** - GD-enhancing Tumor
The training procedure includes **five-fold cross-validation** to improve generalization.
