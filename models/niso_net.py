import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch.conv as conv

from layers import *
from data import GraphBatch

class GatedGCNMean(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.GatedGCN_layers = nn.ModuleList([
            GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)
        ])
        self.aggregator_layer = GraphAggregator_layer(hidden_dim, hidden_dim)
        self.MLP_layer = MLP_layer(2*hidden_dim, output_dim)

    def __process_graph(self, g, X, E, snorm_n, snorm_e):
        # input embedding
        H = self.embedding_h(X)
        E = self.embedding_e(E)
        
        # graph convnet layers
        for GGCN_layer in self.GatedGCN_layers:
            H, E = GGCN_layer(g, H, E, snorm_n, snorm_e)
        
        # MLP classifier
        g.ndata['H'] = H

    def forward(self, g, q, X, E, X_q, E_q, snorm_n, snorm_e, snorm_n_q, snorm_e_q):
        self.__process_graph(g, X, E, snorm_n, snorm_e)
        self.__process_graph(q, X_q, E_q, snorm_n_q, snorm_e_q)

        y_g = dgl.mean_nodes(g, 'H')
        y_q = dgl.mean_nodes(q, 'H')

        y = torch.cat([y_g, y_q], 1)
        y = self.MLP_layer(y)
        
        return y

class GatedGCNAggregate(nn.Module):
    
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
        # self.GNN_layers = nn.ModuleList([
        #     conv.GINConv(
        #         MLP(2, hidden_dim, hidden_dim, hidden_dim),
        #         aggregator_type='sum',
        #         init_eps=0,
        #         learn_eps=True
        #     ) for _ in range(L)
        # ])

        # self.GNN_layers = nn.ModuleList([
        #     conv.RelGraphConv(
        #         in_feat=hidden_dim, out_feat=hidden_dim, num_rels=2,
        #         regularizer='bdd', num_bases=5, activation=torch.relu
        #     ) for _ in range(L)
        # ])
        # self.GNN_layers = nn.ModuleList([
        #     conv.GatedGraphConv(
        #         in_feats=hidden_dim,
        #         out_feats=hidden_dim,
        #         n_steps=L,
        #         n_etypes=1
        #     )
        # ])

        # self.aggregator_layer = GraphAggregator_layer(hidden_dim, hidden_dim)
        # self.MLP_layer = MLP_layer(hidden_dim, hidden_dim)
        self.Predictor_layer = SumPredictor(hidden_dim, hidden_dim, output_dim)

    def __process_graph(self, g: GraphBatch, X, E):
        # input embedding
        H = self.embedding_h(X)
        
        # graph convnet layers
        for GGCN_layer in self.GNN_layers:
            H = GGCN_layer(g.graph, H)
            # H = GGCN_layer(g.graph, H, E)
        
        g.graph.ndata['H'] = H

        # MLP layer
        # g.graph.ndata['H'] = self.MLP_layer(H)

    def forward(self, g: GraphBatch, q: GraphBatch, X, E, X_q, E_q):
        self.__process_graph(g, X, E)
        self.__process_graph(q, X_q, E_q)            

        y = self.Predictor_layer(g, q)
        
        return y
