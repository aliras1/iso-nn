import matplotlib.pyplot as plt
import numpy as np
from typing import Generator

import data
from .common import *
from res.plot_lib import set_default
from .scaler import NormStandardScaler
from utils import *
import torch.nn.functional as F

def node_match(u: dict, v: dict):
    return u['l'] == v['l']    


def gen_dataset(num_g, min_num_nodes, max_num_nodes, edge_probability):
    gen_g = gen_smb_random(num_g, 5, min_num_nodes, max_num_nodes, edge_probability)
    gen_q = gen_smb_random(num_g, 1, 7, 8, 0.2, force_connected=True)

    return gen_dataset(gen_g, gen_q)

def gen_dataset(num_g: int, gen_g: Generator[nx.Graph, None, None]):
    gen_q = gen_smb_random(num_g, 1, 7, 8, 0.2, force_connected=True)
    return gen_dataset(gen_g, gen_q)

def gen_dataset(gen_g: Generator[nx.Graph, None, None], gen_q: Generator[nx.Graph, None, None]):
    zipped_data = [] 
    
    for g, q in zip(gen_g, gen_q):
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
    

def create_dataset_from_dgl(num_graphs):
    def gen_g():
        cora_dataset = CoraGraphDataset()
        # return (dgl.to_networkx(g) for g in cora_dataset[:1])
        g = cora_dataset[0]
        g.ndata['l'] = torch.randint(0, 4, (g.num_nodes(),), dtype=torch.long)
        yield dgl.to_networkx(g, node_attrs='l')
    gen_q = gen_smb_random(num_graphs, 1, 7, 8, 0.2, force_connected=True)
    return gen_dataset(gen_g(), gen_q)

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

def create_artificial_features_dgl(dgl_graph: DGLGraph):    
    vertex_labels = torch.randint(0, 4, (dgl_graph.num_nodes(),), dtype=torch.long)
    vertex_labels_one_hot = F.one_hot(vertex_labels, 4)    
    vertex_degrees = dgl_graph.in_degrees().view(-1, 1).float()    

    dgl_graph.ndata['label'] = vertex_labels
    dgl_graph.ndata['feat'] = torch.cat([vertex_labels_one_hot, vertex_degrees], 1)    
    dgl_graph.edata['feat'] = torch.ones(dgl_graph.number_of_edges()).int()
    return dgl_graph

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