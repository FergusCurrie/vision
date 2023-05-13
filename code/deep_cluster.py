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

NUM_CLUSTERS = 1000 # TODO: 10000
IMAGENET_DIR = '/home/fergus/data/ImageNet'

# Load the AlexNext model
# alexnet = models.alexnet(weights=None)

# Load the ImageNet dataset
#imagenet = get_imagenet(batch_size=16)

# Print the model architecture

def log_start_end(function):
    def wrapper(*args, **kwargs):
        logger.debug(f'STARTED: {function.__name__}')
        func = function(*args, **kwargs)
        logger.debug(f'FINISHED: {function.__name__}')
        return func
    return wrapper

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
    '''
    Calculates inverse of size of cluster. 
    Args:
        cluster_assignments: dictionary mapping file_str to cluster idx
    Returns:
        cluster_weightings: dictionary mapping cluster idx to cluster weighting
    '''
    # print(len(cluster_assignments))
    # assert(len(cluster_assignments) == 1281167)
    cluster_counts, cluster_weightings = {}, {}
    for i in range(NUM_CLUSTERS):
        cluster_counts[i] = 0
    for _, cluster in cluster_assignments.items():
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
    

def loss_function(ypred, y, cluster_weightings, cluster_assignments):
    '''
    Args:
        y_pred is model output (batch_size,NUM_CLUSTERS)
        y is (batch_size,) of synsetid strs
        cluster_weightings maps cluster idx to weightings (based on inverse of size)
        synsetid2idx maps synset idx to cluster idx
    Returns
        negative log liklihood loss, per instance weighted inversely to size of cluster
    '''
    ytrue = np.array([clean_label(y_i, cluster_assignments) for y_i in y])
    weights = np.array([value for _,value in cluster_weightings.items()]) # TODO: make sure this is in correct order
    
    # puts ytrue and weights on gpu
    ytrue = torch.from_numpy(ytrue).type(torch.LongTensor).to(device)
    weights = torch.from_numpy(weights).type(torch.FloatTensor).to(device)
    negative_log_liklihood = torch.nn.NLLLoss(weight=weights)
    logger.debug(f'ytrue: {ytrue} ypred:{ypred}')
    return negative_log_liklihood(ypred, ytrue)

def epoch(dataloader, model, cluster_assignments, optimizer):
    '''
    Train one epoch on model. 
    Args:
        dataloader: dataloader for imagenet dataset
        model: deepcluster model
        cluster_assignments: dictionary mapping file_str to cluster idx
        optimizer: optimizer for deepcluster model
    '''
    cluster_weightings = calculuate_cluster_weightings(cluster_assignments)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y

        # Compute prediction error
        y_pred, _  = model(X)
        loss = loss_function(y_pred, y, cluster_weightings, cluster_assignments)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            logger.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

@log_start_end
def forward_full_dataset(imagenet, dc):
    '''
    Args:
        imagenet: dataloader for imagenet dataset
        dc: deepcluster model
    Returns:
        file_str2embedding: dictionary mapping file_str to deep cluster model feature embedding
    '''
    # First do a foward pass of the dataset 
    file_str2embedding = {}
    with torch.no_grad():
        for batch_idx, (images, file_strs) in enumerate(imagenet):
            # print(f'file_strs={file_strs}')
            # put images on gpu 
            images = images.to(device)

            if batch_idx % 100 == 0:
                print(f'-- Full Forward Cluster predictions {batch_idx} of {len(imagenet)}')
                logger.debug(f'-- Full Forward Cluster predictions {batch_idx} of {len(imagenet)}')
            
            # forward pass
            _, embeddings  = dc(images)
            
            # put embeddings in dictionary 
            for i, file_str in enumerate(file_strs):
                file_str2embedding[file_str] = embeddings[i].cpu().numpy()
    return file_str2embedding


@log_start_end
def batched_pca_forward_full_dataset(imagenet, dc):
    '''
    Differs from previous function by batching calls to pca so that memory doesn't explode. 
    Args:
        imagenet: dataloader for imagenet dataset
        dc: deepcluster model
    Returns:
        file_str2embedding: dictionary mapping file_str to deep cluster model feature embedding
    '''
    # First do a foward pass of the dataset 
    file_str2embedding = {}
    pca_batch_size = 100000
    embeddingsX = [] # holds data matrix of embeddings for pca 
    namesX = [] # holds file_strs for embeddings in embeddingsX
    with torch.no_grad():
        for batch_idx, (images, file_strs) in enumerate(imagenet):
            # put images on gpu 
            images = images.to(device)

            if batch_idx % 100 == 0:
                print(f'-- Full Forward Cluster predictions {batch_idx} of {len(imagenet)}')
                logger.debug(f'-- Full Forward Cluster predictions {batch_idx} of {len(imagenet)}')
            
            # forward pass
            _, embeddings  = dc(images)

            for i, file_str in enumerate(file_strs):
                e_i = embeddings[i].cpu().numpy()
                if e_i.shape != (256*6*6,):
                    print(f'embedding shape = {e_i.shape}')
                embeddingsX.append(e_i)
                namesX.append(file_str)

            # once we have enough embedddings for batch, run pca and add to file_str2embedding
            # then clear memory on embeddingsX and namesX
            if len(embeddingsX) >= pca_batch_size:
                pca_embeddings = faiss_pca(np.array(embeddingsX))
                for i, file_str in enumerate(namesX):
                    file_str2embedding[file_str] = pca_embeddings[i]
                embeddingsX = []
                namesX = []

    return file_str2embedding

@log_start_end
def faiss_pca(X):
    '''
    Uses faiss library to do PCA transform on X.
    Args:
        X: numpy array of shape (len(dataset), n_features=256*6*6)
    Returns:
        pca transform of X 
    '''
    logger.debug(f'X shape = {X.shape}')
    mat = faiss.PCAMatrix(d_in=9216, d_out=256) 
    mat.train(X)
    X = mat.apply(X)
    return X 

def standardise(X):
    X -= np.mean(X)
    X /= np.std(X)
    X /=  np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    return X

@log_start_end
def faiss_kmeans(X):
    '''
    Calculate kmeans cluster assignments for X.
    Args:
        X: numpy array of shape (len(dataset), n_features=256*6*6)
    Returns:
        cluster_assignments: numpy array of shape (len(dataset), 1) of cluster assignments
    '''
    kmeans = faiss.Clustering(X.shape[1], NUM_CLUSTERS)
    kmeans.seed = np.random.randint(9000)
    kmeans.niter=20
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, X.shape[1], flat_config)
    kmeans.train(X, index)
    _, cluster_assignments = index.search(X, 1)
    assert(cluster_assignments.shape == (len(X), 1))
    return cluster_assignments


@log_start_end
def cluster(dc):
    '''
    Find cluster assignments for all images in imagenet dataset.
    Args:
        imagenet: dataloader for imagenet dataset
        dc: deepcluster model
    Returns:
        file_str2cluster_assignment: dictionary mapping file_str to cluster assignment
    '''
    imagenet = get_imagenet(batch_size=64, str_instance_label=True, iterable=True)
    file_str2embedding = batched_pca_forward_full_dataset(imagenet, dc)

    # Stack into data matrix 
    X = np.array([value for _,value in file_str2embedding.items()])
    
    # Perform fast PCA using the faiss library 
    # X = faiss_pca(X) # this is already done with batching 
    
    # Standardise data - l2 norm after standardisation, is that wierd? 
    X = standardise(X)

    # Kmeans transform with faiss-gpu   
    cluster_assignments = faiss_kmeans(X)

    # Create dicitonary from file_str to cluster assignment
    file_str2cluster_assignment = {}
    for file_str, cluster_assignment in zip(file_str2embedding.keys(), cluster_assignments):
        file_str2cluster_assignment[file_str] = cluster_assignment[0] # array of len 1
    return file_str2cluster_assignment



@log_start_end
def forward_full_test():
    '''
    For testing. Get file_str2embedding pointing to None. 
    Args:
        imagenet: dataloader for imagenet dataset
        dc: deepcluster model
    Returns:
        file_str2embedding: dictionary mapping file_str to deep cluster model feature embedding
    '''
    logger.debug('STARTED: forward_full_test ')
    file_str2embedding = {}
    with open(f'{IMAGENET_DIR}/ImageSets/CLS-LOC/train_cls.txt', 'r') as f:
        train_files = f.readlines()
        for file_str in train_files:
            file = file_str.strip().split(' ')
            file_str = file[0]
            file_str2embedding[file_str] = None
    logger.debug('finshed : forward_full_test ')
    return file_str2embedding

def cluster_test(imagenet, dc):
    '''
    returns cluster assignmetns of 0. allows for skipping clustering on test. 
    '''
    file_str2embedding = forward_full_test(imagenet, dc)
    logger.debug('STARTED: cluster test ')
    file_str2cluster_assignment = {}
    for file_str in file_str2embedding.keys():
        file_str2cluster_assignment[file_str] =  np.random.randint(NUM_CLUSTERS) # array of len 1
    logger.debug('FINISHED : cluster test')
    return file_str2cluster_assignment


def train(dc, imagenet, epochs=500):
    '''
    Training loop for Deep Cluster.
    Args:
        dc: deepcluster model
        imagenet: dataloader for imagenet dataset  
        epochs: number of epochs to train for
    '''
    adam = torch.optim.Adam(dc.parameters(), lr=1e-4)
    _, _, synsetid2idx = get_imagenet_class_id_dictionaries()

    for epoch_idx in range(epochs):
        print(f'Epoch {epoch_idx} of {epochs}')
        logger.debug(f'Epoch {epoch_idx} of {epochs}')
        if epoch_idx !=0 and epoch_idx % 25 == 0:
            torch.save(dc.state_dict(), f"local/DeepCluster_{epoch_idx}.pth")

        if os.path.exists(f'cluster_assignment_{epoch_idx}.pkl'):
            with open(f'cluster_assignment_{epoch_idx}.pkl', 'rb') as f:
                cluster_assignments = pickle.load(f)
        else:
            cluster_assignments = cluster(dc)
        
        # pickle cluster assignments so I don't have to wait this long again 
        with open(f'cluster_assignment_{epoch_idx}.pkl', 'wb') as f:
            pickle.dump(cluster_assignments, f)
        
        epoch(imagenet, dc, cluster_assignments, adam)



# Logging 
logging.basicConfig(filename='example.log', level=logging.DEBUG)
logger = logging.getLogger('deep_cluster_logger')
my_handler = logging.FileHandler('deep_cluster_logger.log')
my_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
my_handler.setFormatter(formatter)
logger.addHandler(my_handler)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)
dc = DeepCluster().to(device)

logger.debug('Loading imagnet')
imagenet = get_imagenet(batch_size=64, str_instance_label=True)

logger.debug('Starting train')
train(dc, imagenet, epochs=500)



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