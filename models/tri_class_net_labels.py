import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch.conv as conv
import torch.nn.functional as F

from layers import GSNLayer, ApplyGraphFunc, ApplyNodeFunc, GINLayer, MLP, SumPredictor, GatedGCN_layer
from data import GraphBatch

class TriCount(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        
        # self.GNN_layers = nn.ModuleList([
        #     GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)
        # ])
        self.GNN_layers = nn.ModuleList([
            GINLayer(
                ApplyNodeFunc(
                    MLP(2, hidden_dim, hidden_dim, hidden_dim)
                ), 
                aggr_type='sum',
                dropout=0.2, 
                batch_norm=True, 
                residual=True, 
                init_eps=0, 
                learn_eps=True
            ) for _ in range(L)
        ])
        
        self.mlp = MLP(
            num_layers=6, 
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            dropout=0.2
        )


    def forward(self, g: GraphBatch, X, E):
        H = self.embedding_h(X)
        
        # graph convnet layers
        for GNN_layer in self.GNN_layers:
            H = GNN_layer(g.graph, H)
        
        H = self.mlp(H)    
        g.graph.ndata['H'] = H            

        return F.sigmoid(H)
