'''
DeepCluster is an approach to self-supervised learning belonging to the 
deep learning family. 

https://arxiv.org/pdf/1807.05520.pdf

todo;
- random flips and crops
- dropout
- l2 pen on weights 
- momentum 0.9
- constant step size
- for clustering features are pca reduced to 256 dim, whitended and l2 normalised. 
- kmeans implementation is:  Billion-scale similarity search with gpus. arXiv
- ^  remove from gpu 

rename dc to something better, have consistent ordering in args 
^ I think go through and have a good think about some of this naming

what can be easily abstracted out to use in the future 

check any TODOS

test epoch on datagenerator - does it work?

logging + log levels? 

random init 

replace all these dictionaries with a nicer solution 


using wrong clustering 
k

Follow up paper to deep cluster: 
https://openaccess.thecvf.com/content_ICCV_2019/papers/Caron_Unsupervised_Pre-Training_of_Image_Features_on_Non-Curated_Data_ICCV_2019_paper.pdf

swav inspiration:
https://arxiv.org/pdf/1805.01978.pdf
'''

import pickle 
import torch 
import torch.nn as nn
import torchvision.models as models 
from data_loader.torch_imagenet import get_imagenet, get_imagenet_class_id_dictionaries
import numpy as np 
import faiss
import logging 
import os 
from cluster import Cluster
from utils import get_logger, log_start_end

NUM_CLUSTERS = 1000 # TODO: 10000
IMAGENET_DIR = '/home/fergus/data/ImageNet'

logger = get_logger()

class DeepClusterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet_encoder = models.alexnet(weights=None).features
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(256*6*6, NUM_CLUSTERS) 

    def forward(self, x):
        x = self.alexnet_encoder(x)
        features = self.flatten(x)
        class_preds = self.linear(features)
        return class_preds, features 


class DeepCluster():
    def __init__(self, epochs=500):
        self.dataset = get_imagenet(batch_size=64, str_instance_label=True, iterable=False)
        #self.init_instance2cluster_idx()
        #assert (len(self.instance2cluster_idx) == len(self.dataset))
        self.model = DeepClusterModule()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epochs = epochs

    @log_start_end(logger=logger)
    def init_instance2cluster_idx(self):
        self.instance2cluster_idx = {}
        for _, (_, instance) in enumerate(self.dataset):
            self.instance2cluster_idx[instance] = np.random.randint(NUM_CLUSTERS)
    
    def loss_function(self, ypred, y, cluster_weightings):
        '''
        Args:
            y_pred is model output (batch_size,NUM_CLUSTERS)
            y is (batch_size,) of synsetid strs
        Returns
            negative log liklihood loss, per instance weighted inversely to size of cluster
        '''
        ytrue = np.array([clean_label(y_i, self.instance2cluster_idx) for y_i in y])
        weights = np.array([value for _,value in cluster_weightings.items()]) # TODO: make sure this is in correct order
        
        # puts ytrue and weights on gpu
        ytrue = torch.from_numpy(ytrue).type(torch.LongTensor).to(self.device)
        weights = torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)
        negative_log_liklihood = torch.nn.NLLLoss(weight=weights)
        logger.debug(f'ytrue: {ytrue} ypred:{ypred}')
        return negative_log_liklihood(ypred, ytrue)

    def epoch(self, dataloader):
        '''
        Train one epoch on model. 
        Args:
            dataloader: dataloader for imagenet dataset
            model: deepcluster model
            cluster_assignments: dictionary mapping file_str to cluster idx
            optimizer: optimizer for deepcluster model
        '''
        cluster_weightings = calculuate_cluster_weightings(self.instance2cluster_idx)
        
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y

            # Compute prediction error
            y_pred, _  = self.model(X)
            loss = self.loss_function(y_pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                logger.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def train(self):
        '''
        Training loop for Deep Cluster.
        Args:
            dc: deepcluster model
            imagenet: dataloader for imagenet dataset  
            epochs: number of epochs to train for
        '''
        

        for epoch_idx in range(self.epochs):
            print(f'Epoch {epoch_idx} of {self.epochs}')
            logger.debug(f'Epoch {epoch_idx} of {self.epochs}')

            # Save on 25th epoch 
            if epoch_idx !=0 and epoch_idx % 25 == 0:
                torch.save(self.model.state_dict(), f"local/DeepCluster_{epoch_idx}.pth")
            
            # DO CLUSTER: (load from pickle if exists)
            if os.path.exists(f'cluster_assignment_{epoch_idx}.pkl'):
                with open(f'cluster_assignment_{epoch_idx}.pkl', 'rb') as f:
                    self.instance2cluster_idx = pickle.load(f)
            else:
                cluster = Cluster(self.model, NUM_CLUSTERS)
                self.instance2cluster_idx = cluster.cluster()
            with open(f'cluster_assignment_{epoch_idx}.pkl', 'wb') as f:
                pickle.dump(self.instance2cluster_idx, f)
            
            # Train one epoch 
            self.epoch(self.dataset)


def calculuate_cluster_weightings(instance2cluster_idx):
    '''
    Calculates inverse of size of cluster. 
    Args:
        instance2cluster_idx: dictionary mapping instance (file_str) to cluster idx
    Returns:
        cluster_weightings: dictionary mapping cluster idx to cluster weighting
    '''
    print(len(instance2cluster_idx))  # 1200384
    assert(len(instance2cluster_idx) == 1281167) # 1281167
    cluster_counts, cluster_weightings = {}, {}
    for i in range(NUM_CLUSTERS):
        cluster_counts[i] = 0
    for _, cluster in instance2cluster_idx.items():
        cluster_counts[cluster] += 1

    for cluster, count in cluster_counts.items():
        cluster_weightings[cluster] = 1 / count
    return cluster_weightings

def clean_label(file_str, cluster_assignments):
    file = file_str.strip().split(' ')
    # print(list(cluster_assignments.keys())[:10])
    # print(len(cluster_assignments)) # 1200384
    # i = 0
    # for key, value in cluster_assignments.items():
    #     print(f'key={key}')
    #     i+= 1
    #     if i > 5: 
    #         break
    # # print(file[0]) # n04008634/n04008634_5181
    c = cluster_assignments[file[0]] # cluster assignment for this point 
    return c
    



if __name__ == '__main__':
    logger = get_logger()
    logger.debug('Starting')

    deep_cluster = DeepCluster()
    deep_cluster.train()












# testing cluster assignment - 
# epoch_idx = 0
# if os.path.exists(f'cluster_assignment_{epoch_idx}.pkl'):
#     with open(f'cluster_assignment_{epoch_idx}.pkl', 'rb') as f:
#         cluster_assignments = pickle.load(f)

# for batch_idx, (images, file_strs) in enumerate(imagenet):
#     for file_str in file_strs:
#         if file_str in cluster_assignments:
#             print('yes')
#         else:
#             print('no')