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

```plaintext
Data/
 │── imagesTs.zip
 │── labelsTs.zip
 │── training set.txt 
```
## Install Dependencies 
The model built and implemented on M1 ARM64 architecture supported Metal Backend for Acceleration. The below steps are followed to work on M1. Usage of NVIDIA GPU, create conda environment and execute the main model(final.py)
### Create environment
```bash
brew install --cask miniconda
conda init zsh
source ~/.zshrc
conda create -n {env-name} python=3.10
conda activate {env-name}
```
### Install packages
```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal
pip install numpy nibabel scikit-learn matplotlib scipy tqdm monai
```
## Usage
1. ### Training the Model
To train the 3D U-Net model with cross-validation:
```bash
python final.py
```
2. ### Performing Predictions
To generate predictions using the trained model:
```bash
python prediction_final.py
```

## Acknowledgment 
This project is intended for research purposes only. 
BraTs - Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS). *IEEE Transactions on Medical Imaging, 34*(10), 1993-2024. https://doi.org/10.1109/TMI.2015.2467942
Monai - https://github.com/Project-MONAI/tutorials
