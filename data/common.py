import dgl
from dgl.data import *
from itertools import combinations_with_replacement
import networkx as nx
import networkx.algorithms.isomorphism as nx_iso
import pickle
import random
import torch


DatasetEntry = tuple[nx.Graph, nx.Graph, torch.FloatTensor]


class GraphBatch:
    def __init__(
        self,
        g: dgl.DGLGraph,
        sizes_n: torch.Tensor,
        snorm_n: torch.Tensor,
        snorm_e: torch.Tensor,
    ):
        self.graph: dgl.DGLGraph = g
        self.sizes_n: torch.Tensor = sizes_n
        self.snorm_n: torch.Tensor = snorm_n
        self.snorm_e: torch.Tensor = snorm_e
        self.extension_matrix: torch.Tensor = self._compute_extension_matrix()

    def _compute_extension_matrix(self) -> torch.Tensor:
        batch_size = len(self.sizes_n)
        offset = 0
        m = torch.zeros(self.graph.num_nodes(), 0)

        for i in range(batch_size):
            g_i = dgl.slice_batch(self.graph, i)
            rest = self.graph.num_nodes() - offset - g_i.num_nodes()
            column_list = [0] * offset + [1] * g_i.num_nodes() + [0] * rest
            column = torch.LongTensor(column_list).view(-1, 1)
            m = torch.cat([m, column], 1)
            offset += g_i.num_nodes()

        return m


def _collate_graphs(graphs: dgl.DGLGraph):
    sizes_n = [graph.number_of_nodes() for graph in graphs]  # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization
    sizes_e = [graph.number_of_edges() for graph in graphs]  # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    batch = GraphBatch(
        batched_graph, torch.FloatTensor(sizes_n).view(-1, 1), snorm_n, snorm_e
    )  # wrap data into an object
    return batch


# Collate function to prepare graphs
def collate(samples: list[DatasetEntry]):
    gs, qs, labels = map(
        list, zip(*samples)
    )  # samples is a list of tuples (graph, patern, label)
    labels = torch.cat(labels)
    batched_g = _collate_graphs(gs)
    batched_q = _collate_graphs(qs)
    return batched_g, batched_q, labels


def gen_sbm_random(
    num_blocks: int,
    min_num_v: int,
    max_num_v: int,
    p: float,
    force_connected: bool = False,
) -> nx.Graph:
    """Generates a random labeled graph with the stochastic
    block model. If force_connected flag is set, the resulting
    graph is guaranteed to be fully connected."""
    block_sizes = [random.randint(min_num_v, max_num_v) for _ in range(num_blocks)]

    # edge probabilities between blocks
    # ps[i][j] = probability of placing edges between block_i and block_j
    ps = [[p if j == i else 0.1 for j in range(num_blocks)] for i in range(num_blocks)]

    g = nx.Graph(nx.stochastic_block_model(sizes=block_sizes, p=ps))
    while force_connected and not nx.is_connected(g):
        g = nx.Graph(nx.stochastic_block_model(sizes=block_sizes, p=ps))

    # generate labels
    for v in range(g.number_of_nodes()):
        g.nodes[v]["l"] = random.randint(0, 3)

    return g


def gen_gnp_random(
    min_num_v: int,
    max_num_v: int,
    p: float,
    force_connected: bool = False,
) -> nx.Graph:
    num_v = random.randint(min_num_v, max_num_v)
    g = nx.Graph(nx.fast_gnp_random_graph(num_v, p))

    while force_connected and not nx.is_connected(g):
        g = nx.Graph(nx.fast_gnp_random_graph(num_v, p))

    for v in range(g.number_of_nodes()):
        g.nodes[v]["l"] = random.randint(0, 3)

    return g


def node_match(u: dict, v: dict):
    return u["l"] == v["l"]


def save_dataset(dataset: list[DatasetEntry], name: str) -> None:
    print(f'Saving dataset "{name}"...')
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


def load_dataset(name: str) -> list[DatasetEntry]:
    print(f'Loading dataset "{name}"...')
    gs = load_graphs(f"./data/{name}_generic_g.dat")
    qs = load_graphs(f"./data/{name}_generic_q.dat")
    with open(f"./data/{name}_generic_l.dat", "rb") as fp:
        ls = pickle.load(fp)

    dataset = [(g, q, l) for g, q, l in zip(gs[0], qs[0], ls)]
    return dataset
