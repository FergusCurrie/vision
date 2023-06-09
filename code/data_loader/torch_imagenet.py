'''

In imagesets txts:
['n01440764/n01440764_10026 1\n', 'n01440764/n01440764_10027 2\n', 'n01440764/n01440764_10029 3\n', 'n01440764/n01440764_10040 4\n', 'n01440764/n01440764_10042 5\n', 'n01440764/n01440764_10043 6\n', 'n01440764/n01440764_10048 7\n', 'n01440764/n01440764_10066 8\n', 'n01440764/n01440764_10074 9\n', 'n01440764/n01440764_10095 10\n']

In Data/n07714990: 
['n07714990_1878.JPEG', 'n07714990_2501.JPEG', 'n07714990_6696.JPEG', 'n07714990_71.JPEG', 'n07714990_8125.JPEG', 'n07714990_1491.JPEG', 'n07714990_456.JPEG', 'n07714990_2674.JPEG', 'n07714990_481.JPEG', 'n07714990_4779.JPEG']
'''

import torch 
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
import torchvision.transforms as transforms
import json


IMAGENET_DIR = '/home/fergus/data/ImageNet'


def load_imagenet_filenames(str_instance_label=True):
    _, _, synsetid2idx = get_imagenet_class_id_dictionaries()
    filenames = []
    labels = []
    with open(f'{IMAGENET_DIR}/ImageSets/CLS-LOC/train_cls.txt', 'r') as f:
        train_files = f.readlines()
    for file_str in train_files:
        file = file_str.strip().split(' ')
        filenames.append(file[0])
        label = file[0].split('/')[0]
        if str_instance_label == True:
            labels.append(file[0]) # n01440764/n01440764_10026
        else:
            labels.append(synsetid2idx[label])
    return filenames, labels 

def load_imagenet_image(image_filename) -> torch.Tensor:
    # load image 
    image = Image.open(f'{IMAGENET_DIR}/Data/CLS-LOC/train/{image_filename}.JPEG')
    
    # Define the transformation to convert the image to a tensor
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Apply the transformation to the image
    tensor = transform(image)

    if tensor.shape ==  (1, 224, 224):
        tensor = tensor.repeat(3, 1, 1)

    if tensor.shape == (4, 224, 224):
        print('got shape (4,224,224)')
        tensor = tensor[:3,:,:]
    return tensor


def get_imagenet_class_id_dictionaries():
    '''
    Help functions to get between synset id, id and label string of imagnet. 
    '''
    # class idx maps from id to label string 
    with open(f"{IMAGENET_DIR}/imagenet_class_index.json", 'r') as class_index_file:
        class_idx = json.load(class_index_file)

        # create idx mapping to sysnsetid and label
        idx2synsetid = {}
        idx2label = {}
        for k, idx in enumerate(class_idx):
            idx2synsetid[idx] = class_idx[str(k)][0]
            idx2label[idx] = class_idx[str(k)][1]
    
    # make synsetid to index map
    synsetid2idx = {v:k for k,v in idx2synsetid.items()}   
    return idx2label, idx2synsetid, synsetid2idx

def get_imagenet(batch_size, str_instance_label=False, iterable=False):
    if iterable:
        training_dataing = IterableImagenet(batch_size=batch_size)
        return DataLoader(training_dataing, batch_size=batch_size)
    else:
        training_data = ImageNet(str_instance_label=str_instance_label)
        return DataLoader(training_data, batch_size=batch_size, shuffle=True)

class IterableImagenet(IterableDataset):
    def __init__(self, batch_size):
        self.filenames, self.labels = load_imagenet_filenames()
        self.batch_size = batch_size
    def __len__(self):
        return len(self.filenames)
    
    def __iter__(self):
        # batches itself, iter should get get the ith item? 
        for i in range(len(self.filenames)):
            yield load_imagenet_image(self.filenames[i]), self.labels[i]
            # yield self.filenames[i], self.labels[i]

        # batch_images = []
        # batch_labels = [] 
        # for i in range(len(self.filenames)):
        #     batch_images.append(load_imagenet_image(self.filenames[i]))
        #     batch_labels.append(self.labels[i])
            
        #     if len(batch_images) == self.batch_size:
        #         print(batch_images[0].shape)
        #         yield torch.stack(batch_images), batch_labels
        #         batch_images = []
        #         batch_labels = []


class ImageNet(Dataset):
    def __init__(self, str_instance_label=False):
        self.str_instance_label = str_instance_label
        self.filenames, self.labels = load_imagenet_filenames()
        print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_filename = self.filenames[idx]
        label = self.labels[idx]

        tensor = load_imagenet_image(image_filename)

        return tensor, label
    


if __name__ == '__main__':
    a,b,c = get_imagenet_class_id_dictionaries()
    print(b)
    imagenet = get_imagenet(batch_size=1, str_instance_label=True, iterable=True)
    print(len(imagenet))
    file_str_set = set()
    for batch_idx, (filename, file_strs) in enumerate(imagenet):
        file_str_set.add(file_strs)
    print(len(file_str_set))