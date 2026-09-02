import torch
import torch.nn as nn
import numpy as np

class AttentionPool(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.attn = nn.Linear(emb_size, 1)

    def forward(self, x, mask):  # x: (B, N, D), mask: (B, N) True=real
        scores = self.attn(x).squeeze(-1)               # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)           # (B, N)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1) # (B, D)

        return pooled

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, pooling, learnedmask):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            pooling: String defining the pooling method to use, either "mean" or "attention"
            learnedmask: Boolean indicating whether masks are learned or fixed
        """
        super(ConditionalSimNet, self).__init__()

        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionPool(embedding_size)

        # create the mask
        if learnedmask:
            # define masks with gradients
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize random weights
            self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)

            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c):
        x, attn_mask = x  # Unpack the tuple (padded_embeddings, attention_mask)

        if self.pooling == "attention":
            x = self.pool(x, attn_mask)   # (B, N, D) -> (B, D)

        embedded_x = self.embeddingnet(x)  # (B, D) -> (B, D)
        mask = self.masks(c)

        if self.learnedmask:
            mask = torch.nn.functional.relu(mask)

        masked_embedding = embedded_x * mask

        return masked_embedding, mask, embedded_x
