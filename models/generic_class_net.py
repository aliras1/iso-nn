import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch.conv as conv
import torch.nn.functional as F
from functools import reduce

from layers import (
    GSNLayer,
    ApplyGraphFunc,
    ApplyNodeFunc,
    GINLayer,
    MLP,
    SumPredictor,
    GatedGCN_layer,
)
from data import GraphBatch


class GenericClassNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, L: int):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)

        self.gnn_layers_g = nn.ModuleList(
            [
                GINLayer(
                    ApplyNodeFunc(MLP(2, hidden_dim, hidden_dim, hidden_dim)),
                    aggr_type="sum",
                    dropout=0.2,
                    batch_norm=True,
                    residual=True,
                    init_eps=0,
                    learn_eps=True,
                )
                for _ in range(L)
            ]
        )

        self.gnn_layers_p = nn.ModuleList(
            [
                GINLayer(
                    ApplyNodeFunc(MLP(2, hidden_dim, hidden_dim, hidden_dim)),
                    aggr_type="sum",
                    dropout=0.2,
                    batch_norm=True,
                    residual=True,
                    init_eps=0,
                    learn_eps=True,
                )
                for _ in range(L)
            ]
        )

        self.mlp_g = MLP(
            num_layers=8,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.2,
        )

        self.mlp_p = MLP(
            num_layers=8,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.2,
        )

        self.predict_mlp = MLP(
            num_layers=16,
            input_dim=2 * output_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=0.2,
        )

    def _forward_graph(
        self,
        g: GraphBatch,
        X: torch.Tensor,
        gnn_layers: nn.ModuleList,
        mlp_layer: nn.Module,
    ) -> torch.Tensor:
        H = self.embedding_h(X)

        for gnn_layer in gnn_layers:
            H = gnn_layer(g.graph, H)

        H = mlp_layer(H)
        g.graph.ndata["H"] = H
        return H

    def _extend_H_with_p(
        self, H: torch.Tensor, p_representation: torch.Tensor, g: GraphBatch
    ) -> torch.Tensor:
        extension = torch.matmul(g.extension_matrix, p_representation)
        return torch.cat([H, extension], dim=1)

    def forward(
        self, g: GraphBatch, X: torch.Tensor, p: GraphBatch, X_p: torch.Tensor
    ) -> torch.Tensor:
        H = self._forward_graph(g, X, self.gnn_layers_g, self.mlp_g)
        self._forward_graph(p, X_p, self.gnn_layers_p, self.mlp_p)

        p_representation = dgl.mean_nodes(p.graph, "H")
        H = self._extend_H_with_p(H, p_representation, g)

        pred = self.predict_mlp(H)

        return F.sigmoid(pred)
