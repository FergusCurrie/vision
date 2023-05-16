'''
Class for applying the deep cluster model to a dataset.
'''
import torch
import logging 
import faiss
import numpy as np
from utils import log_start_end, get_logger
from data_loader.torch_imagenet import get_imagenet

logger = get_logger()

class Cluster():
    def __init__(self, model, num_clusters):
        '''
        dataset : iterable dataset (can do 1 forward pass on)
        '''
        self.dataset = get_imagenet(batch_size=64, str_instance_label=True, iterable=True)
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clusters = num_clusters

    @log_start_end(logger=logger)
    def forward_full_dataset(self):
        '''
        Forward pass the entire dataset through the model and return the embeddings
        Returns:
            instance2embedding: dictionary mapping file_str to deep cluster model feature embedding
        '''
        # First do a foward pass of the dataset 
        instance2embedding = {}
        with torch.no_grad():
            for batch_idx, (images, file_strs) in enumerate(self.dataset):
                # print(f'file_strs={file_strs}')
                # put images on gpu 
                images = images.to(self.device)

                if batch_idx % 100 == 0:
                    print(f'-- Full Forward Cluster predictions {batch_idx} of {len(self.dataset)}')
                    logger.debug(f'-- Full Forward Cluster predictions {batch_idx} of {len(self.dataset)}')
                
                # forward pass
                _, embeddings  = self.model(images)
                
                # put embeddings in dictionary 
                for i, file_str in enumerate(file_strs):
                    instance2embedding[file_str] = embeddings[i].cpu().numpy()
        return instance2embedding


    @log_start_end(logger)
    def batched_pca_forward_full_dataset(self):
        '''
        Differs from previous function by batching calls to pca so that memory doesn't explode. 
        Returns:
            instance2embedding: dictionary mapping file_str to deep cluster model feature embedding
        '''
        # First do a foward pass of the dataset 
        instance2embedding = {}
        pca_batch_size = 100000
        embeddingsX = [] # holds data matrix of embeddings for pca 
        namesX = [] # holds file_strs for embeddings in embeddingsX
        with torch.no_grad():
            for batch_idx, (images, file_strs) in enumerate(self.dataset):
                # put images on gpu 
                images = images.to(self.device)

                if batch_idx % 100 == 0:
                    print(f'-- Full Forward Cluster predictions {batch_idx} of {len(self.dataset)}')
                    logger.debug(f'-- Full Forward Cluster predictions {batch_idx} of {len(self.dataset)}')
                
                # forward pass
                _, embeddings  = self.model(images)

                for i, file_str in enumerate(file_strs):
                    e_i = embeddings[i].cpu().numpy()
                    if e_i.shape != (256*6*6,):
                        print(f'embedding shape = {e_i.shape}')
                    embeddingsX.append(e_i)
                    namesX.append(file_str)

                # once we have enough embedddings for batch, run pca and add to file_str2embedding
                # then clear memory on embeddingsX and namesX
                if len(embeddingsX) >= pca_batch_size:
                    pca_embeddings = self.faiss_pca(np.array(embeddingsX))
                    for i, file_str in enumerate(namesX):
                        instance2embedding[file_str] = pca_embeddings[i]
                    embeddingsX = []
                    namesX = []

        return instance2embedding

    @log_start_end(logger)
    def faiss_pca(self, X):
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

    def standardise(self, X):
        X -= np.mean(X)
        X /= np.std(X)
        X /=  np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        return X

    @log_start_end(logger)
    def faiss_kmeans(self, X):
        '''
        Calculate kmeans cluster assignments for X.
        Args:
            X: numpy array of shape (len(dataset), n_features=256*6*6)
        Returns:
            isntance2cluster_idx: numpy array of shape (len(dataset), 1) of cluster assignments
        '''
        kmeans = faiss.Clustering(X.shape[1], self.num_clusters)
        kmeans.seed = np.random.randint(9000)
        kmeans.niter=20
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, X.shape[1], flat_config)
        kmeans.train(X, index)
        _, isntance2cluster_idx = index.search(X, 1)
        assert(isntance2cluster_idx.shape == (len(X), 1))
        return isntance2cluster_idx


    @log_start_end(logger)
    def cluster(self):
        '''
        Find cluster assignments for all images in imagenet dataset.
        Returns:
            isntance2cluster_idx: dictionary mapping file_str to cluster assignment
        '''
        # imagenet = get_imagenet(batch_size=64, str_instance_label=True, iterable=True)
        instance2embedding = self.batched_pca_forward_full_dataset()

        # Stack into data matrix 
        X = np.array([value for _,value in instance2embedding.items()])
        
        # Perform fast PCA using the faiss library 
        # X = faiss_pca(X) # this is already done with batching 
        
        # Standardise data - l2 norm after standardisation, is that wierd? 
        X = self.standardise(X)

        # Kmeans transform with faiss-gpu   
        cluster_assignments = self.faiss_kmeans(X)

        # Create dicitonary from file_str to cluster assignment
        instance2cluster_idx = {}
        for file_str, cluster_assignment in zip(instance2embedding.keys(), cluster_assignments):
            instance2cluster_idx[file_str] = cluster_assignment[0] # array of len 1
        return instance2cluster_idx



    # @log_start_end
    # def forward_full_test():
    #     '''
    #     For testing. Get file_str2embedding pointing to None. 
    #     Args:
    #         imagenet: dataloader for imagenet dataset
    #         dc: deepcluster model
    #     Returns:
    #         file_str2embedding: dictionary mapping file_str to deep cluster model feature embedding
    #     '''
    #     logger.debug('STARTED: forward_full_test ')
    #     file_str2embedding = {}
    #     with open(f'{IMAGENET_DIR}/ImageSets/CLS-LOC/train_cls.txt', 'r') as f:
    #         train_files = f.readlines()
    #         for file_str in train_files:
    #             file = file_str.strip().split(' ')
    #             file_str = file[0]
    #             file_str2embedding[file_str] = None
    #     logger.debug('finshed : forward_full_test ')
    #     return file_str2embedding

    # def cluster_test(imagenet, dc):
    #     '''
    #     returns cluster assignmetns of 0. allows for skipping clustering on test. 
    #     '''
    #     file_str2embedding = forward_full_test(imagenet, dc)
    #     logger.debug('STARTED: cluster test ')
    #     file_str2cluster_assignment = {}
    #     for file_str in file_str2embedding.keys():
    #         file_str2cluster_assignment[file_str] =  np.random.randint(NUM_CLUSTERS) # array of len 1
    #     logger.debug('FINISHED : cluster test')
    #     return file_str2cluster_assignment