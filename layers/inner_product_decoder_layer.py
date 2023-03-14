import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class InnerProductDecoder(nn.Module):

    def __init__(self, activation, dropout):
        super().__init__()
        self.activation = activation                
        self.dropout = dropout            

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj