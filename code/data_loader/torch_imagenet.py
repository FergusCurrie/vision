import torch 
from torch.utils.data import Dataset


class PascalImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, batch_size, train_test_val="train"):
        self.image_size = image_size
        self.train_test_val = train_test_val
        self.batch_size = batch_size
        self.load_data()  # set data to list of filenames based on train_testA_val


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = None, None 
        return image, label