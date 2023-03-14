import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch.conv as conv

from layers import GSNLayer, ApplyGraphFunc, ApplyNodeFunc, GINLayer, MLP, SumPredictor
from data import GraphBatch

class GSN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        
        # self.GNN_layers = nn.ModuleList([
        #     GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)
        # ])
        self.GNN_layers = nn.ModuleList([
            GSNLayer(
                ApplyGraphFunc(
                    GINLayer(
                        ApplyNodeFunc(
                            MLP(2, hidden_dim+2, hidden_dim, hidden_dim)
                        ), 
                        aggr_type='sum',
                        dropout=0.2, 
                        batch_norm=True, 
                        residual=True, 
                        init_eps=0, 
                        learn_eps=True
                    )
                )
            ) for _ in range(L)
        ])
        # self.GNN_layers = nn.ModuleList([
        #     GINLayer(
        #         ApplyNodeFunc(
        #             MLP(2, hidden_dim, hidden_dim, hidden_dim)
        #         ), 
        #         aggr_type='sum',
        #         dropout=0.2, 
        #         batch_norm=True, 
        #         residual=True, 
        #         init_eps=0, 
        #         learn_eps=True
        #     ) for _ in range(L)
        # ])
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

        # self.MLP_layer = MLP_layer(hidden_dim, hidden_dim)
        self.predictor = SumPredictor(hidden_dim, hidden_dim, output_dim)

    def __process_graph(self, g: GraphBatch, X, E):
        # input embedding
        H = self.embedding_h(X)
        
        # graph convnet layers
        for GNN_layer in self.GNN_layers:
            H = GNN_layer(g.graph, H)
        
        g.graph.ndata['H'] = H

        # MLP layer
        # g.graph.ndata['H'] = self.MLP_layer(H)

    def forward(self, g: GraphBatch, q: GraphBatch, X, E, X_q, E_q):
        self.__process_graph(g, X, E)
        self.__process_graph(q, X_q, E_q)            

        y = self.predictor(g, q)
        return y
