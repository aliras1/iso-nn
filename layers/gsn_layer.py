import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class ApplyGraphFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, g, h):
        h = self.mlp(g, h)
        return h



class GSNLayer(nn.Module):

    def __init__(self, apply_func: ApplyGraphFunc):
        super().__init__()
        self.apply_func = apply_func

    def forward(self, g, h):
        structs = g.ndata["struct"]
        h = torch.cat([h, structs], 1)
        y = self.apply_func(g, h)

        return y
    

