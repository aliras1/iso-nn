from re import sub
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import networkx.generators as nx_gen
import networkx.algorithms.isomorphism as nx_iso

import data
from .common import *
from res.plot_lib import set_default
from .scaler import NormStandardScaler
from res.plot_lib import draw_nx


def gen_dataset(num_g, num_q):
    def gen_g():
        yield from gen_gnp_random(num_g, 20, 40, 0.2)

    qs = list(gen_gnp_random(num_q, 3, 6, 0.2))
    gs = list(gen_g())

    zipped_data = []
    for g in gs:
        for q in qs:
            insert_q_into_g(g, q)
        for q in qs:
            zipped_data.append((g, q, count_subisos(g, q)))

    gs, qs, labels = zip(*zipped_data)

    labels_tensor = torch.FloatTensor(labels)

    scaler = NormStandardScaler()
    normalized_labels = scaler.fit_tensor(labels_tensor)
    normalized_labels = normalized_labels.tolist()

    dataset = [
        convert_to_dgl(g, q, numisos)
        for g, q, numisos in zip(gs, qs, normalized_labels)
    ]

    return dataset, scaler


def create_artificial_features(graph: nx.Graph):
    labels = [graph.nodes[v]["l"] for v in range(graph.number_of_nodes())]
    vertex_labels = torch.FloatTensor(labels).view(-1, 1)

    dgl_graph = dgl.from_networkx(graph)
    vertex_degrees = dgl_graph.in_degrees().view(-1, 1).float()

    substructs = [nx_gen.complete_graph(3), nx_gen.path_graph(3)]
    substructs_counts = [[0 for _ in substructs] for _ in graph.nodes]
    for i, substruct in enumerate(substructs):
        match = nx_iso.GraphMatcher(graph, substruct)
        subisos = match.subgraph_isomorphisms_iter()
        for subiso in subisos:
            for v in graph.nodes:
                if v in subiso:
                    substructs_counts[v][i] += 1
    substructs_count_tensor = torch.tensor(substructs_counts)

    dgl_graph.ndata["feat"] = torch.cat([vertex_labels, vertex_degrees], 1)
    dgl_graph.ndata["struct"] = substructs_count_tensor
    dgl_graph.edata["feat"] = torch.ones(dgl_graph.number_of_edges()).int()
    return dgl_graph


def count_substruct_isos(g: nx.Graph, v: int, substruct: nx.Graph):
    v_ng = set()
    for distance in range(substruct.number_of_nodes()):
        v_ng.update(nx.descendants_at_distance(g, v, distance))
    ng = g.subgraph(v_ng)
    print(f"{g.number_of_nodes()}: {len(v_ng)}")
    match = nx_iso.GraphMatcher(ng, substruct)
    subisos = match.subgraph_isomorphisms_iter()
    return sum(1 for iso in subisos if v in iso)


def convert_to_dgl(g: nx.Graph, q: nx.Graph, num_isos: int):
    g = create_artificial_features(g)
    q = create_artificial_features(q)

    return ((g, q), num_isos)
