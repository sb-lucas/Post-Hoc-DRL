import torch
import torch.nn as nn
import numpy as np

class ConditionalSimNet(nn.Module):
    def __init__(self, n_conditions, embedding_size):
        """ 
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
        """

        super(ConditionalSimNet, self).__init__()
        
        # define masks with gradients
        self.masks = torch.nn.Embedding(n_conditions, embedding_size)
        # initialize weights
        self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005

    def forward(self, x, c):
        self.mask = self.masks(c)
        self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = x * self.mask

        return masked_embedding, self.mask.norm(1)