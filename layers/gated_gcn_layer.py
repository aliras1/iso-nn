import torch
import torch.nn as nn

from data import GraphBatch

class GatedGCN_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        # self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bx_j = edges.src['BX']
        # e_j = Ce_j + Dxj + Ex
        e_j = edges.src['DX'] + edges.dst['EX'] #+ edges.data['CE']
        edges.data['E'] = e_j
        return {'Bx_j' : Bx_j, 'e_j' : e_j}

    def reduce_func(self, nodes):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        e_j = nodes.mailbox['e_j']
        # sigma_j = σ(e_j)
        σ_j = torch.sigmoid(e_j)
        # h = Ax + Σ_j η_j * Bxj
        h = Ax + torch.sum(σ_j * Bx_j, dim=1) / torch.sum(σ_j, dim=1)
        return {'H' : h}
    
    def forward(self, g: GraphBatch, X):        
        g.ndata['H']  = X
        g.ndata['AX'] = self.A(X) 
        g.ndata['BX'] = self.B(X) 
        g.ndata['DX'] = self.D(X)
        g.ndata['EX'] = self.E(X) 
        # g.graph.edata['E']  = E_X
        # g.graph.edata['CE'] = self.C(E_X)
        
        g.update_all(self.message_func, self.reduce_func)
        
        H = g.ndata['H'] # result of graph convolution
        # E = g.graph.edata['E'] # result of graph convolution
        
        snorm_n = 1.0 / g.number_of_nodes()
        H *= snorm_n # normalize activation w.r.t. graph node size
        # E *= g.snorm_e # normalize activation w.r.t. graph edge size
        
        H = self.bn_node_h(H) # batch normalization  
        # E = self.bn_node_e(E) # batch normalization  
        
        H = torch.relu(H) # non-linear activation
        # E = torch.relu(E) # non-linear activation
        
        H = X + H # residual connection
        # E = E_X + E # residual connection
        
        return H#, E