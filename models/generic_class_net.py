import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch.conv as conv
import torch.nn.functional as F

from layers import GSNLayer, ApplyGraphFunc, ApplyNodeFunc, GINLayer, MLP, SumPredictor, GatedGCN_layer
from data import GraphBatch

class GenericClassNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        
        
        self.gnn_layers_g = nn.ModuleList([
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

        self.gnn_layers_q = nn.ModuleList([
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
        
        self.mlp_g = MLP(
            num_layers=8, 
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            dropout=0.2
        )

        self.mlp_q = MLP(
            num_layers=8, 
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            dropout=0.2
        )

        self.predict_mlp = MLP(
            num_layers=16, 
            input_dim=2*output_dim, 
            hidden_dim=hidden_dim, 
            output_dim=1, 
            dropout=0.2
        )


    def _forward_graph(self, g: GraphBatch, X, gnn_layers, mlp_layer):
        H = self.embedding_h(X)
        
        # graph convnet layers
        for gnn_layer in gnn_layers:
            H = gnn_layer(g.graph, H)
        
        H = mlp_layer(H)    
        g.graph.ndata['H'] = H
        return H

    def forward(self, g: GraphBatch, X, E, q: GraphBatch, X_q, E_q):
        H = self._forward_graph(g, X, self.gnn_layers_g, self.mlp_g)  
        self._forward_graph(q, X_q, self.gnn_layers_q, self.mlp_q)  
        #--------------

        q_representation = dgl.mean_nodes(q.graph, 'H')
        batch_size = len(g.sizes_n)
        m = torch.zeros(g.graph.num_nodes(), batch_size)            

        offset = 0
        for i in range(batch_size):
            g_i = dgl.slice_batch(g.graph, i)
            for j in range(g_i.num_nodes()):
                m[j + offset][i] = 1.0
            offset += g_i.num_nodes()
        
        q_ = torch.matmul(m, q_representation)
        
        x = torch.cat([H, q_], dim=1)        
        pred = self.predict_mlp(x)

        return F.sigmoid(pred)
