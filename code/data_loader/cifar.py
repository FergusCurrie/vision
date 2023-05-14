'''
Imagenet is slow to experiment with. Going to see about using CIFAR
to get models up and running then give them a shot at imagenet.
'''

import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
CIFAR_DIRECTORY = '/home/fergus/data/cifar/'

# Define the transform to apply to the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# data = unpickle(f'{CIFAR_DIRECTORY}/cifar-10-batches-py/data_batch_1')
# print(data.keys()) # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
# # print(data[b'data'].shape) # (10000, 3072)
# # print(data[b'filenames'])
# # print(trainset)
# # print(len(trainloader)) # length of loader is number of batches

class Cifar(Dataset):
    def __init__(self):
        self.images = []
        self.filenames = []
        self.labels = []
        self.load()

    def load(self):
        for i in range(1,6):
            data = unpickle(f'{CIFAR_DIRECTORY}/cifar-10-batches-py/data_batch_{i}')
            for j, filename in enumerate(data[b'filenames']):
                self.filenames.append(filename)
                self.images.append(data[b'data'][j])
                self.labels.append(data[b'labels'][j])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = self.images[idx].reshape(3, 32,32) # good example of reshaping gone wrong
        image = np.transpose(image, (1,2,0)) 
        assert(image.shape == (32,32,3 )) # preprocessing fails without this 
        if 0:
            tensor = transform(image)
        else:
            tensor =  torch.from_numpy(image).permute(2,0,1)
        assert(tensor.shape == (3, 32, 32))
        return tensor.float(), filename 

def get_cifar(batch_size, shuffle=True):
    training_data = Cifar()
    return DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)

cifar = Cifar()
next(iter(cifar))