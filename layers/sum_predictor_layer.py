import dgl
from dgl import DGLGraph
import torch
import torch.nn as nn

from data import GraphBatch

class SumPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(SumPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.lin_g = nn.Linear(input_dim, hidden_dim)
        self.lin_q = nn.Linear(input_dim, hidden_dim)
        self.lin_pred = nn.Linear(4*hidden_dim+4, hidden_dim)
        self.lin_pred_final = nn.Linear(hidden_dim+4, output_dim)

        for layer in [self.lin_g, self.lin_q, self.lin_pred]:
            nn.init.normal_(layer.weight, 0.0, 1/(hidden_dim**0.5))
            nn.init.zeros_(layer.bias)
        for layer in [self.lin_pred_final]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, g: GraphBatch, q: GraphBatch):
        y_g = self.lin_g(dgl.sum_nodes(g.graph, 'H'))
        y_g = self.dropout(y_g)
        y_q = self.lin_q(dgl.sum_nodes(q.graph, 'H'))
        y_q = self.dropout(y_q)
        
        # y_q = self.lin_q(dgl.sum_nodes(q.graph, 'H'))
        # y_q = self.dropout(y_q)
        # g.graph.ndata['out'] = self.dropout(self.lin_g(g.graph.ndata['H']))
        # y_g = dgl.sum_nodes(g.graph, 'out')

        inv_sizes_n = 1.0 / g.sizes_n
        inv_sizes_n_q = 1.0 / q.sizes_n

        y = self.lin_pred(torch.cat([y_g, y_q, y_g-y_q, y_g*y_q, g.sizes_n, q.sizes_n, inv_sizes_n, inv_sizes_n_q], dim=1))
        y = torch.relu(y)
        y = self.lin_pred_final(torch.cat([y, g.sizes_n, q.sizes_n, inv_sizes_n, inv_sizes_n_q], dim=1))

        return y