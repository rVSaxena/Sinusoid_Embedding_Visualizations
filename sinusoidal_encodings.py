import math
import torch
import numpy as np
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, embedding_size, information_density=1000):
        super().__init__()
        self.embedding_size=embedding_size
        self.information_density=information_density

        # The block below is to facilitate an unnecessarily complicated way to combine the sin
        # and cos embeddings by interspersing. Any sensible person would use the uncomplicated approach as in 'The Annotated Transformer' guide.
        # Some people just need to derive matrix operations as matrix multiplications.

        I=np.identity(self.embedding_size//2)
        R1=np.repeat(I, 2, axis=1)
        R2=np.repeat(I, 2, axis=1)
        
        self.sin_placement_matrix=torch.from_numpy(np.where(np.arange(R1.shape[1]) % 2, 0, R1)).type(torch.float32)
        self.cos_placement_matrix=torch.from_numpy(np.where((1+np.arange(R1.shape[1])) % 2, 0, R1)).type(torch.float32)


    def forward(self, time):

        half_dim=self.embedding_size//2
        embeddings=math.log(self.information_density)/(half_dim - 1)
        embeddings=torch.exp(torch.arange(half_dim)*-embeddings)
        embeddings=time[:, None]*embeddings[None, :]

        embeddings=embeddings.sin().squeeze()@self.sin_placement_matrix+embeddings.cos().squeeze()@self.cos_placement_matrix
        embeddings=embeddings[:, None]
        return embeddings

