import matplotlib.pyplot as plt
import numpy as np

import data
from .common import *
from res.plot_lib import set_default
from .scaler import NormStandardScaler
from utils import *
import torch.nn.functional as F

def node_match(u: dict, v: dict):
    return u['l'] == v['l']    

def gen_dataset(num_g, min_num_nodes, max_num_nodes, edge_probability):
    def gen_g():
        yield from gen_smb_random(num_g, 5, min_num_nodes, max_num_nodes, edge_probability)
    def gen_q():
        yield from gen_smb_random(num_g, 1, 4, 6, 0.51, force_connected=True)


    zipped_data = [] 
    
    for g, q in zip(gen_g(), gen_q()):
        # draw(g, "")
        # draw(q, "")
                
        match = nx_iso.GraphMatcher(g, q, node_match=node_match)
        subisos = match.subgraph_isomorphisms_iter()         

        num_v = g.number_of_nodes()
        y = torch.zeros([num_v, 1], dtype=torch.float)
        for subiso in subisos:
            for v in subiso.keys():
                y[v][0] = 1.0    

            
        g_with_features = create_artificial_features(g)
        q_with_features = create_artificial_features(q)
        zipped_data.append((g_with_features, q_with_features, y))

    return zipped_data
    

# create artifical data feature (= in degree) for each node
def create_artificial_features(graph: nx.Graph):
    labels = [graph.nodes[v]['l'] for v in range(graph.number_of_nodes())]
    vertex_labels = torch.LongTensor(labels)
    vertex_labels_one_hot = F.one_hot(vertex_labels, 4)
    
    graph = dgl.from_networkx(graph)    
    vertex_degrees = graph.in_degrees().view(-1, 1).float()    

    graph.ndata['feat'] = torch.cat([vertex_labels_one_hot, vertex_degrees], 1)    
    graph.edata['feat'] = torch.ones(graph.number_of_edges()).int()
    return graph

def save_dataset(dataset, name):
    save_graphs(f'./data/{name}_generic_g.dat', [graph_query_label[0] for graph_query_label in dataset])
    save_graphs(f'./data/{name}_generic_q.dat', [graph_query_label[1] for graph_query_label in dataset])
    with open(f'./data/{name}_generic_l.dat', 'wb') as fp:
        pickle.dump([graph_query_label[2] for graph_query_label in dataset], fp)

def load_dataset(name):
    gs = load_graphs(f'./data/{name}_generic_g.dat')
    qs = load_graphs(f'./data/{name}_generic_q.dat')
    with open(f'./data/{name}_generic_l.dat', 'rb') as fp:
        ls = pickle.load(fp)
    
    dataset = [(g, q, l) for g, q, l in zip(gs[0], qs[0], ls)]
    return dataset