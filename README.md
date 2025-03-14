# Deep Learning-based Brain Tumor Segmentation Using MRI

## Overview
This work walks through a 3D U-Net architecture implementation on brain tumor segmentation downstream task using BraTs challenge data. The multi-modal MRI data consists of four-time points: FLAIR, T1w, t1gd, and T2w. The goal is to build a **2D or 3D U-Net** architecture for segmentation, training the model under the following labeled classes:
- **0** - Background
- **1** - Necrotic and Non-enhancing Tumor
- **2** - Peritumoral Edema
- **4** - GD-enhancing Tumor
The training procedure includes **five-fold cross-validation** to improve generalization.

## Dataset
The dataset used in this work is from the BraTS Challenge 2016 and 2017, consisting of multi-modal MRI images and corresponding labels. The train and test data is stored in the Data folder. Download the training data from the text file which consists of two folders - imagesTr(samples) and labelsTr(labels), additionally imagesTs and labelsTs are test samples, labels derived from the train set manually.

- Structure
```plaintext
- Data/
-  │── imagesTs.zip
-  │── labelsTs.zip
-  │── training set.txt
