import matplotlib.pyplot as plt
import numpy as np

import data
from .common import *
from res.plot_lib import set_default
from .scaler import NormStandardScaler
from utils import *

def gen_dataset(num_g, num_q, min_num_nodes, max_num_nodes, edge_probability):
    def gen_g():
        yield from gen_gnp_random(num_g, min_num_nodes, max_num_nodes, edge_probability)
    
    # qs = list(gen_gnp_random(num_q, 4, 6, 0.2))
    qs = list(gen_gnp_random(1, 3, 3, 1.0))
    for q in qs:
        draw(q, "")

    gs = list(gen_g())

    zipped_data = []
    for g in gs:
        for q in qs:
            insert_q_into_g(g, q)
        for q in qs:
            zipped_data.append((g, q, count_subisos(g, q)))

    gs, qs, labels = zip(*zipped_data)

    labels_tensor = torch.FloatTensor(labels)
    # print(labels_tensor)
    # set_default()
    bins = int(max(labels))
    ax = plt.gca()
    ax.margins(0.20)
    plt.hist([int(l) for l in labels], bins)
    plt.show()

    scaler = NormStandardScaler()
    normalized_labels = scaler.fit_tensor(labels_tensor)    
    normalized_labels = normalized_labels.tolist()
    plt.hist(normalized_labels)
    # plt.show()
    # print(normalized_labels)
    
    dataset = [convert_to_dgl(g, q, numisos) for g, q, numisos in zip(gs, qs, normalized_labels)]
    # random.shuffle(dataset)
    
    return dataset, scaler
    

# create artifical data feature (= in degree) for each node
def create_artificial_features(graph: nx.Graph):
    labels = [graph.nodes[v]['l'] for v in range(graph.number_of_nodes())]
    vertex_labels = torch.FloatTensor(labels).view(-1, 1)
    
    graph = dgl.from_networkx(graph)    
    vertex_degrees = graph.in_degrees().view(-1, 1).float()    

    graph.ndata['feat'] = torch.cat([vertex_labels, vertex_degrees], 1)
    graph.edata['feat'] = torch.ones(graph.number_of_edges()).int()
    return graph

def convert_to_dgl(g: nx.Graph, q: nx.Graph, num_isos: int):
    g = create_artificial_features(g)
    q = create_artificial_features(q)

    return ((g, q), num_isos)