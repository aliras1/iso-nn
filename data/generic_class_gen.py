from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from typing import Generator, Callable, List, TypeVar
from dgl.data import PATTERNDataset

import data
from .common import *
from res.plot_lib import set_default
from .scaler import NormStandardScaler
from utils import *
import torch.nn.functional as F

GraphGenerator = Callable[[], nx.Graph]


class DatasetWrapper:
    def __init__(self, num_graphs, dataset) -> None:
        self._cur_idx = 0
        self._len = len(dataset)
        self.num_graphs = num_graphs
        self.dataset = dataset

    def next(self) -> dgl.DGLGraph:
        g = self.dataset[self._cur_idx]
        self._cur_idx = (self._cur_idx + 1) % self._len
        print(f"next {self._cur_idx}")
        return g


def node_match(u: dict, v: dict) -> bool:
    return u["l"] == v["l"]


def gen_dataset(
    dataset_size: int, min_num_nodes: int, max_num_nodes: int, edge_probability: float
) -> List[DatasetEntry]:
    # gen_g = lambda: gen_smb_random(5, min_num_nodes, max_num_nodes, edge_probability)
    # gen_p = lambda: gen_smb_random(1, 6, 8, 0.2, force_connected=True)

    gen_g = lambda: gen_gnp_random(min_num_nodes, max_num_nodes, edge_probability)
    gen_p = lambda: gen_gnp_random(6, 8, 0.2, force_connected=True)

    return _gen_dataset(dataset_size, gen_g, gen_p)


def _label_graph(g: nx.Graph, p: nx.Graph) -> DatasetEntry:
    return (
        _create_artificial_features(g),
        _create_artificial_features(p),
        _classify_nodes(g, p),
    )


def _classify_nodes(g: nx.Graph, p: nx.Graph) -> torch.FloatTensor:
    match = nx_iso.GraphMatcher(g, p, node_match=node_match)
    subisos = match.subgraph_isomorphisms_iter()
    subiso_lists = map(lambda subiso: subiso.keys(), subisos)
    subiso_node_list = reduce(
        lambda subiso1, subiso2: chain(subiso1, subiso2), subiso_lists, []
    )
    subiso_nodes = torch.LongTensor(list(set(subiso_node_list)))
    draw(p)
    y = torch.zeros(
        [
            g.number_of_nodes(),
        ],
        dtype=torch.float,
    )
    y.index_fill_(0, subiso_nodes, 1.0)

    return y.view(-1, 1)


def _gen_dataset(
    dataset_size: int, gen_g: GraphGenerator, gen_p: GraphGenerator
) -> List[DatasetEntry]:
    g_and_p = ((gen_g(), gen_p()) for _ in range(dataset_size))
    g_and_p_and_l = map(lambda gp: _label_graph(*gp), g_and_p)

    return list(g_and_p_and_l)


def create_dataset_from_dgl_karate(num_graphs: int) -> List[DatasetEntry]:
    def gen_g():
        dataset = KarateClubDataset()
        g = dataset[0]

        g.ndata["l"] = torch.randint(0, 4, (g.num_nodes(),), dtype=torch.long)
        return dgl.to_networkx(g, node_attrs="l")

    gen_p = lambda: gen_gnp_random(3, 4, 0.2, force_connected=True)
    return _gen_dataset(num_graphs, gen_g, gen_p)


def convert_to_internal(
    g: dgl.DGLGraph,
) -> DatasetEntry:
    g.ndata["l"] = g.ndata["feat"]
    pnode = g.ndata["label"].nonzero().squeeze()
    p = g.subgraph(pnode)
    p.ndata["l"] = p.ndata["feat"]
    labels = g.ndata["label"].float().unsqueeze(dim=1)
    g = _create_artificial_features_dgl(g)
    p = _create_artificial_features_dgl(p)

    return (g, p, labels)


def create_dataset_from_dgl_pattern(
    num_graphs: int, mode: str = "train"
) -> List[DatasetEntry]:
    data = PATTERNDataset(mode=mode)
    return list(map(convert_to_internal, data[:num_graphs]))


def create_dataset_from_dgl_cora(num_graphs: int) -> List[DatasetEntry]:
    dataset = CoraGraphDataset()

    def gen_g():
        g = dataset[0].clone()
        g.ndata["l"] = torch.randint(0, 4, (g.num_nodes(),), dtype=torch.long)
        return dgl.to_networkx(g, node_attrs="l")

    gen_p = lambda: gen_gnp_random(3, 4, 0.2, force_connected=True)
    return _gen_dataset(num_graphs, gen_g, gen_p)


def create_dataset_from_dgl_mutag(num_graphs: int) -> List[DatasetEntry]:
    dataset = MUTAGDataset()

    def gen_g():
        g = dgl.to_homogeneous(dataset[0])
        g.ndata["l"] = torch.randint(0, 4, (g.num_nodes(),), dtype=torch.long)
        return dgl.to_networkx(g, node_attrs="l")

    gen_p = lambda: gen_gnp_random(3, 8, 0.2, force_connected=True)
    return _gen_dataset(num_graphs, gen_g, gen_p)


def create_dataset_from_dgl_ppi(num_graphs: int) -> List[DatasetEntry]:
    raw_data = PPIDataset()
    dataset = DatasetWrapper(num_graphs=num_graphs, dataset=raw_data)

    def gen_g():
        g = dataset.next().clone()
        g.ndata["l"] = torch.randint(0, 4, (g.num_nodes(),), dtype=torch.long)
        return nx.Graph(dgl.to_networkx(g, node_attrs="l").to_undirected())

    gen_p = lambda: gen_gnp_random(3, 3, 0.5, force_connected=True)
    return _gen_dataset(num_graphs, gen_g, gen_p)


def create_dataset_from_dgl_sbmmix(num_graphs: int) -> List[DatasetEntry]:
    raw_data = SBMMixtureDataset(n_graphs=num_graphs, n_nodes=256, n_communities=8)
    dataset = DatasetWrapper(num_graphs=num_graphs, dataset=raw_data)

    def gen_g():
        g = dataset.next()[0]
        g.ndata["l"] = torch.randint(0, 4, (g.num_nodes(),), dtype=torch.long)
        return dgl.to_networkx(g, node_attrs="l")

    gen_p = lambda: gen_gnp_random(3, 4, 0.2, force_connected=True)
    return _gen_dataset(num_graphs, gen_g, gen_p)


def _create_artificial_features_dgl(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """create artifical data feature
    (= in degree + one-hot(label)) for each node"""
    vertex_labels = graph.ndata["l"]
    vertex_labels_one_hot = F.one_hot(vertex_labels, 4)
    vertex_degrees = graph.in_degrees().view(-1, 1).float()

    graph.ndata["label"] = vertex_labels
    graph.ndata["feat"] = torch.cat([vertex_labels_one_hot, vertex_degrees], 1)
    graph.edata["feat"] = torch.ones(graph.number_of_edges()).int()
    return graph


def _create_artificial_features(graph: nx.Graph) -> dgl.DGLGraph:
    """create artifical data feature
    (= in degree + one-hot(label)) for each node"""
    labels = [graph.nodes[v]["l"] for v in range(graph.number_of_nodes())]
    vertex_labels = torch.LongTensor(labels)
    vertex_labels_one_hot = F.one_hot(vertex_labels, 4)

    graph = dgl.from_networkx(graph)
    vertex_degrees = graph.in_degrees().view(-1, 1).float()

    graph.ndata["label"] = vertex_labels
    graph.ndata["feat"] = torch.cat([vertex_labels_one_hot, vertex_degrees], 1)
    graph.edata["feat"] = torch.ones(graph.number_of_edges()).int()
    return graph


def create_artificial_features_dgl(dgl_graph: dgl.DGLGraph) -> dgl.DGLGraph:
    vertex_labels = torch.randint(0, 4, (dgl_graph.num_nodes(),), dtype=torch.long)
    vertex_labels_one_hot = F.one_hot(vertex_labels, 4)
    vertex_degrees = dgl_graph.in_degrees().view(-1, 1).float()

    dgl_graph.ndata["label"] = vertex_labels
    dgl_graph.ndata["feat"] = torch.cat([vertex_labels_one_hot, vertex_degrees], 1)
    dgl_graph.edata["feat"] = torch.ones(dgl_graph.number_of_edges()).int()
    return dgl_graph


def save_dataset(dataset: List[DatasetEntry], name: str) -> None:
    save_graphs(
        f"./data/{name}_generic_g.dat",
        [graph_query_label[0] for graph_query_label in dataset],
    )
    save_graphs(
        f"./data/{name}_generic_q.dat",
        [graph_query_label[1] for graph_query_label in dataset],
    )
    with open(f"./data/{name}_generic_l.dat", "wb") as fp:
        pickle.dump([graph_query_label[2] for graph_query_label in dataset], fp)


def load_dataset(name: str) -> List[DatasetEntry]:
    gs = load_graphs(f"./data/{name}_generic_g.dat")
    qs = load_graphs(f"./data/{name}_generic_q.dat")
    with open(f"./data/{name}_generic_l.dat", "rb") as fp:
        ls = pickle.load(fp)

    dataset = [(g, q, l) for g, q, l in zip(gs[0], qs[0], ls)]
    return dataset
