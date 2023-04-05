import imp
import math
import pickle
import random
from itertools import combinations_with_replacement

import dgl
from dgl import DGLGraph
from dgl.data import *
import networkx as nx
import networkx.algorithms.isomorphism as nx_iso
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.utils.data import DataLoader
from res.plot_lib import draw


class GraphBatch():
    def __init__(self, g, sizes_n, snorm_n, snorm_e):
        self.graph = g
        self.sizes_n = sizes_n
        self.snorm_n = snorm_n
        self.snorm_e = snorm_e

def collate_graphs(graphs: dgl.DGLGraph):    
    sizes_n = [graph.number_of_nodes() for graph in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization 
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    batch = GraphBatch(batched_graph, torch.FloatTensor(sizes_n).view(-1, 1), snorm_n, snorm_e) # wrap data into an object
    return batch

# Collate function to prepare graphs
def collate(samples):
    gs_and_qs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    gs, qs = map(list, zip(*gs_and_qs))  # samples is a list of pairs (graph, label)
    labels = torch.tensor(labels).view(-1, 1)
    batched_g = collate_graphs(gs)
    batched_q= collate_graphs(qs)
    return batched_g, batched_q, labels


def gen_smb_random(num_graphs, num_blocks, min_num_v, max_num_v, p, force_connected=False):
    for _ in range(num_graphs):
        sizes = [random.randint(min_num_v, max_num_v) for i in range(num_blocks)]
        ps = [[p for j in range(num_blocks)] for i in range(num_blocks)]

        g = nx.Graph(nx.stochastic_block_model(sizes=sizes, p=ps))
        while force_connected and not nx.is_connected(g):
            g = nx.Graph(nx.stochastic_block_model(sizes=sizes, p=ps))

        for v in range(g.number_of_nodes()):
            g.nodes[v]['l'] = random.randint(0, 3)        
        
        yield g

def gen_all_triangles():
    label_combinations = combinations_with_replacement([0, 1, 2, 3], 3)
    for label_combination in range(label_combinations):
        g = nx.Graph(nx.cycle_graph(n=3))
        
        for idx, label in enumerate(label_combination):
            g.nodes[idx]['l'] = label
        yield g

def gen_gnp_random(num_graphs, min_num_v, max_num_v, p):
    for _ in range(num_graphs):
        num_v = random.randint(min_num_v, max_num_v)        
        
        g = nx.Graph(nx.fast_gnp_random_graph(num_v, p))

        # convert g into a connected graph
        for u in range (g.number_of_nodes() - 1):
            v = u + 1
            if not g.has_edge(u, v):
                g.add_edge(u, v)

        for v in range(g.number_of_nodes()):
            g.nodes[v]['l'] = random.randint(0, 4)        
        yield g

def insert_q_into_g(g: nx.Graph, q: nx.Graph):
    size = g.number_of_nodes()

    # add q to g
    for v in range(q.number_of_nodes()):
        g.add_node(v + size, l=q.nodes[v]['l'])
    for u, v in q.edges:
        g.add_edge(u + size, v + size)
    
    # connect q and g
    for _ in range(int(0.2 * q.number_of_nodes())):
        u = random.randint(0, q.number_of_nodes()-1) + size
        v = random.randint(0, g.number_of_nodes()-1)
        g.add_edge(u, v)

    return g


def node_match(u: dict, v: dict):
    return u['l'] == v['l']    

def count_subisos(g: nx.Graph, q: nx.Graph, node_match=node_match):
    # draw(dgl.from_networkx(g), '')
    # draw(dgl.from_networkx(q), '')
    match = nx_iso.GraphMatcher(g, q, node_match=node_match)
    subisos = match.subgraph_isomorphisms_iter()
    num_subisos = sum((1 for _ in subisos))
    # num_subisos /= math.factorial(q.number_of_nodes())
    return float(num_subisos)


def save_dataset(dataset, name):
    save_graphs(f'./data/{name}_g.dat', [graph_and_label[0][0] for graph_and_label in dataset])
    save_graphs(f'./data/{name}_q.dat', [graph_and_label[0][1] for graph_and_label in dataset])
    with open(f'./data/{name}_l.dat', 'wb') as fp:
        pickle.dump([graph_and_label[1] for graph_and_label in dataset], fp)

def load_dataset(name):
    gs = load_graphs(f'./data/{name}_g.dat')
    qs = load_graphs(f'./data/{name}_q.dat')
    with open(f'./data/{name}_l.dat', 'rb') as fp:
        ls = pickle.load(fp)
    
    dataset = [((g, q), l) for g, q, l in zip(gs[0], qs[0], ls)]
    return dataset