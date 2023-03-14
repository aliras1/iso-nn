import dgl
import torch
import torch.nn as nn

class GraphAggregator_layer(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(GraphAggregator_layer, self).__init__()
		self.lin = nn.Linear(input_dim, output_dim)
		self.lin_gate = nn.Linear(input_dim, output_dim)
		self.lin_final = nn.Linear(output_dim, output_dim)

	def forward(self, g, H):
		h_states = self.lin(H)
		x_gates = torch.relu(self.lin_gate(H))
		g.ndata['out'] = h_states * x_gates		
		h_states = dgl.sum_nodes(g, 'out')
		h_states = self.lin_final(h_states)
		return h_states