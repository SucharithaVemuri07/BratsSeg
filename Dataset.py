from monai.data import Dataset

# Define BraTS dataset class
class BraTSDataset(Dataset):
    def __init__(self, data_files, transforms):
        self.data_files = data_files
        self.transforms = transforms

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        return self.transforms(self.data_files[index])



