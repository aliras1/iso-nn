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
        yield from gen_smb_random(num_g, min_num_nodes, max_num_nodes, edge_probability)
    
    zipped_data = []

    q = nx.Graph(nx.cycle_graph(n=3))
    for v in range(q.number_of_nodes()):
        q.nodes[v]['l'] = random.randint(0, 3)  
    
    for g in gen_g():
        # draw(g, "")
        
        num_v = g.number_of_nodes()
        y = torch.zeros([num_v, 1], dtype=torch.float)
        match = nx_iso.GraphMatcher(g, q, node_match=node_match)
        subisos = match.subgraph_isomorphisms_iter()
        for subiso in subisos:
            for v in subiso.keys():
                y[v][0] = 1.0              
            
        g_with_features = create_artificial_features(g)
        zipped_data.append((g_with_features, y))

    # qs = list(gen_all_triangles())
    # for g in gen_g():
    #     for q in qs:
    #         y = [0 for v in range(g.number_of_nodes())]
    #         match = nx_iso.GraphMatcher(g, q, node_match=node_match)
    #         subisos = match.subgraph_isomorphisms_iter()
    #         for subiso in subisos:
    #             for v in subiso.keys():
    #                 y[v] = 1
    #         zipped_data.append(g, q, )

    return zipped_data, dgl.from_networkx(q)
    

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

def save_dataset(dataset, q, name):
    save_graphs(f'./data/{name}_labels_g.dat', [graph_and_label[0] for graph_and_label in dataset])
    save_graphs(f'./data/{name}_labels_q.dat', [q])
    with open(f'./data/{name}_labels_l.dat', 'wb') as fp:
        pickle.dump([graph_and_label[1] for graph_and_label in dataset], fp)

def load_dataset(name):
    gs = load_graphs(f'./data/{name}_labels_g.dat')
    q = load_graphs(f'./data/{name}_labels_q.dat')
    with open(f'./data/{name}_labels_l.dat', 'rb') as fp:
        ls = pickle.load(fp)
    
    dataset = [(g, l) for g, l in zip(gs[0], ls)]
    return dataset, q