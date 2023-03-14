import enum
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch.conv as conv

from layers import *
from data import GraphBatch


class CascadeGNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.g_embedding_h = nn.Linear(input_dim, hidden_dim)
        self.q_embedding_h = nn.Linear(input_dim, hidden_dim)
        
        # self.g_layers = nn.ModuleList([
        #     GINLayer(
        #         ApplyNodeFunc(
        #             MLP(2, 2*hidden_dim, hidden_dim, hidden_dim)
        #         ), 
        #         aggr_type='sum',
        #         dropout=0.2, 
        #         batch_norm=True, 
        #         residual=True, 
        #         init_eps=0, 
        #         learn_eps=True
        #     ) for _ in range(L)
        # ])
        # self.q_layers = nn.ModuleList([
        #     GINLayer(
        #         ApplyNodeFunc(
        #             MLP(1, hidden_dim, hidden_dim, hidden_dim)
        #         ), 
        #         aggr_type='sum',
        #         dropout=0.2, 
        #         batch_norm=True, 
        #         residual=True, 
        #         init_eps=0, 
        #         learn_eps=True
        #     ) for _ in range(L)
        # ])
        
        self.g_layers = nn.ModuleList([
            conv.AGNNConv() for _ in range(L)
        ])
        self.g_reduce = nn.ModuleList([
            MLP(1, 2*hidden_dim, hidden_dim, hidden_dim) for _ in range(L)
        ])

        self.q_layers = nn.ModuleList([
            conv.AGNNConv() for _ in range(L)
        ])
        # self.aggregator_layer = GraphAggregator_layer(hidden_dim, hidden_dim)
        # self.MLP_layer = MLP_layer(hidden_dim, hidden_dim)
        self.Predictor_layer = SumPredictor(hidden_dim, hidden_dim, output_dim)        

    def forward(self, g: GraphBatch, q: GraphBatch, X, E, X_q, E_q):
        h_g = self.g_embedding_h(X)
        h_q = self.q_embedding_h(X_q)

        for g_layer, g_reduce, q_layer in zip(self.g_layers, self.g_reduce, self.q_layers):
            h_q = q_layer(q.graph, h_q)
            q.graph.ndata['H'] = h_q
            h_q_aggr = dgl.sum_nodes(q.graph, 'H')

            h_g_with_h_q = torch.zeros(len(h_g), 2*self.hidden_dim)
            i = 0                        
            for j, aggr_h_q in enumerate(h_q_aggr):                
                for k in range(g.sizes_n[j][0].int()):
                    h_g_with_h_q[i] = torch.cat([h_g[i], aggr_h_q], 0)
                    i += 1
        
            h_g = g_layer(g.graph, h_g_with_h_q)
            h_g = g_reduce(h_g)

        g.graph.ndata['H'] = h_g        

        y = self.Predictor_layer(g, q)
        
        return y
