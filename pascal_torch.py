"""
Load in pascal dataset to torch data generator. 

This avoids importing tensorflow 
"""


import os
from skimage import io
import random
import numpy as np
from skimage.transform import resize
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import torch

# from sklearn.preprocessing import LabelEncoder
from PIL import Image


class PascalImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, batch_size, train_test_val="train"):
        self.image_size = image_size
        self.train_test_val = train_test_val
        self.batch_size = batch_size
        self.load_data()  # set data to list of filenames based on train_testA_val

    def load_data(self):
        with open(f"data/PASCAL_VOC_2010/VOC2010/ImageSets/Segmentation/train.txt") as f:
            train = f.read().splitlines()
        with open(f"data/PASCAL_VOC_2010/VOC2010/ImageSets/Segmentation/trainval.txt") as f:
            trainval = f.read().splitlines()
        with open(f"data/PASCAL_VOC_2010/VOC2010/ImageSets/Segmentation/val.txt") as f:
            val = f.read().splitlines()
        # Select appropriate data
        if self.train_test_val == "train":
            self.data = train
        if self.train_test_val == "trainval":
            self.data = trainval
        if self.train_test_val == "val":
            self.data = val

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
                (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]
        image = io.imread(f"data/PASCAL_VOC_2010/VOC2010/JPEGImages/{filename}.jpg")
        label = io.imread(f"data/PASCAL_VOC_2010/VOC2010/SegmentationClass/{filename}.png")[..., :3]
        label = self.encode_segmap(label)
        label = label[..., np.newaxis]
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image = image.reshape(image.shape[-1], image.shape[0], image.shape[1])
        label = label.reshape(label.shape[-1], label.shape[0], label.shape[1])
        # print(image.shape, label.shape)
        image = torchvision.transforms.Resize(
            (self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )(image)
        label = torchvision.transforms.Resize(
            (self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )(label)
        # print(image.shape, label.shape)

        return (image / 255), label


def get_pascal_torch(image_size, batch_size):
    # Initalise the dataset
    train = PascalImageDataset(image_size=image_size, batch_size=batch_size)
    val = PascalImageDataset(image_size=image_size, batch_size=batch_size, train_test_val="val")
    # Convert to torch dataloader
    train = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val = torch.utils.data.DataLoader(val, batch_size=batch_size)
    return train, val


if __name__ == "__main__":
    train, val = get_pascal_torch(224, 8)
    print(train)
    for x, y in train:
        print(x[0].shape)
        break
