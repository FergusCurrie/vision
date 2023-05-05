'''
DeepCluster is an approach to self-supervised learning belonging to the 
deep learning family. 

https://arxiv.org/pdf/1807.05520.pdf
'''

import torch 
import torchvision.models as models 


# Load the AlexNext model

alexnet = models.alexnet(weights=None)

# Print the model architecture
print(alexnet)



