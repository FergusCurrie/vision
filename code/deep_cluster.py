'''
DeepCluster is an approach to self-supervised learning belonging to the 
deep learning family. 

https://arxiv.org/pdf/1807.05520.pdf

todo;
- empty cluster update
- inverse weighting on cluster size to loss 
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
'''

import torch 
import torch.nn as nn
import torchvision.models as models 
from data_loader.torch_imagenet import get_imagenet, get_imagenet_class_id_dictionaries
import numpy as np 
import faiss

NUM_CLUSTERS = 1000

# Load the AlexNext model
# alexnet = models.alexnet(weights=None)

# Load the ImageNet dataset
#imagenet = get_imagenet(batch_size=16)

# Print the model architecture

class DeepCluster(nn.Module):
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

def calculuate_cluster_weightings(cluster_assignments):
    # first calculate cluster weightings as inverse of size of clusters 
    cluster_counts, cluster_weightings = {}, {}
    for i in range(NUM_CLUSTERS):
        cluster_counts[i] = 0
    for _, cluster in cluster_counts.items():
        cluster_counts[cluster] += 1

    for cluster, count in cluster_counts.items():
        cluster_weightings[cluster] = 1 / count
    return cluster_weightings

def loss(y_pred, y, cluster_weightings, synsetid2idx):
    '''
    y_pred is (batch_size,NUM_CLUSTERS)
    y is (batch_size,) of synsetid strs 
    cluster_weightings maps cluster idx to weightign (based on inverse of size)
    sysnsetid2idx maps synset idx to cluster idx 

    '''
    ytrue = [synsetid2idx[y_i] for y_i in y]
    weights = [value for item,value in cluster_weightings.items()] # TODO: make sure this is in correct order
    negative_log_liklihood = torch.nn.NLLLoss(weight=weights)
    return negative_log_liklihood(y_pred, ytrue)

def epoch(dataloader, model, cluster_assignments, optimizer, synsetid2idx):
    cluster_weightings = calculuate_cluster_weightings(cluster_assignments)

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        _, y_pred = model(X)
        loss = loss(y_pred, y, cluster_weightings, synsetid2idx)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def forward_full_dataset(imagenet, dc):
    # First do a foward pass of the dataset 
    file_str2embedding = {}
    with torch.no_grad():
        for batch_idx, (images, file_strs) in enumerate(imagenet):
            # print(f'file_strs={file_strs}')
            # put images on gpu 
            images = images.to(device)

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx} of {len(imagenet)}')
                if batch_idx != 0:
                    break
            
            # forward pass
            ypred, embeddings  = dc(images)
            
            # put embeddings in dictionary 
            for i, file_str in enumerate(file_strs):
                file_str2embedding[file_str] = embeddings[i].cpu().numpy()
    return file_str2embedding

def faiss_pca(X):
    # PCA transform with faiss-gpu
    mat = faiss.PCAMatrix(512, 128) # d_in=512, d_out=128
    mat.train(X)
    assert mat.is_trained
    X = mat.apply(X)
    return X 

def standardise(X):
    X -= np.mean(X)
    X /= np.std(X)
    X /=  np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    return X

def faiss_kmeans(X):
    # Init settings 
    ncentroids = 5 
    niter = 20
    verbose = True

    d = X.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(X)
    _, cluster_assignments = kmeans.index.search(X, 1)
    assert(cluster_assignments.shape == (len(X), 1))
    return cluster_assignments

def cluster(imagenet, dc):
    file_str2embedding = forward_full_dataset(imagenet, dc)

    # Stack into data matrix 
    X = np.array([value for _,value in file_str2embedding.items()])
    
    # Perform fast PCA using the faiss library 
    X = faiss_pca(X)
    
    # Standardise data - l2 norm after standardisation, is that wierd? 
    X = standardise(X)

    # Kmeans transform with faiss-gpu   
    cluster_assignments = faiss_kmeans(X)

    # Create dicitonary from file_str to cluster assignment
    file_str2cluster_assignment = {}
    for file_str, cluster_assignment in zip(file_str2embedding.keys(), cluster_assignments):
        file_str2cluster_assignment[file_str] = cluster_assignment[0] # array of len 1
    return file_str2cluster_assignment


def train(dc, imagenet, epochs=500):
    adam = torch.optim.Adam(dc.parameters(), lr=1e-4)
    _, _, synsetid2idx = get_imagenet_class_id_dictionaries()

    for epoch_idx in range(epochs):
        print(f'Epoch {epoch_idx} of {epochs}')
        if epoch_idx !=0 and epoch_idx % 25 == 0:
            torch.save(dc.state_dict(), f"local/DeepCluster_{epoch_idx}.pth")

        cluster_assignments = cluster(imagenet, dc)
        epoch(imagenet, dc, cluster_assignments, adam, synsetid2idx)

# test alexnet shape 
# print(models.alexnet(weights=None).features(torch.randn(1, 3, 224, 224)).shape) # torch.Size([1, 256, 6, 6])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)
dc = DeepCluster().to(device)

imagenet = get_imagenet(batch_size=64, str_instance_label=True)

train(dc, imagenet, epochs=500)



