import dgl
import networkx as nx
from res.plot_lib import set_default
import matplotlib.pyplot as plt
import networkx.algorithms.isomorphism as nx_iso
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgl.data import CoraGraphDataset
from gnnlens import Writer
import torch

from data import generic_class_gen
from data.common import collate_graphs


model = torch.load('./models/model_generic_class_net_10000_medium_0.9009.pth')

num_g = 64
num_q = 8

def draw(g):
    g_nx = dgl.to_networkx(g)
    pos = nx.drawing.layout.spring_layout(
        g_nx,
        k=2)
    nx.draw_networkx(
        g_nx,
        pos=pos,
        node_size=300,
        edge_color='grey',
        cmap=plt.cm.Reds,
        vmin=0,
        vmax=1,)

    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

def collate(samples):
    gs, qs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    labels = torch.cat(labels)
    batched_g = collate_graphs(gs)
    batched_q = collate_graphs(qs)
    return batched_g, batched_q, labels

# dataset = generic_class_gen.load_dataset("generic_class_1000")
dataset = generic_class_gen.create_dataset_from_dgl(1)

# writer = Writer('tutorial_graph')
# nlabels = graph.ndata['label']
# num_classes = 4
# writer.add_graph(name='rnd', graph=graph,
                # nlabels=nlabels, num_nlabel_types=num_classes)


# dataset = generic_class_gen.gen_dataset(1, 20, 30, 0.1)
# dataset = dataset[:10]
# dataset_split = int(len(dataset) * 0.77)
# testset = dataset[dataset_split:]
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
threshold = 0.6
loss_fn = nn.BCELoss()


def visualize(g, h):
    g_nx = dgl.to_networkx(g.graph)
    pos = nx.drawing.layout.spring_layout(
        g_nx,
        k=2)
    nx.draw_networkx(
        g_nx,
        pos=pos,
        node_size=300,
        edge_color='grey',
        with_labels=True,
        node_color=h,
        cmap=plt.cm.Reds,
        vmin=0,
        vmax=1,)

    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

def node_match(u: dict, v: dict):
    return (u['l'] == v['l']).item()

def evaluate_accuracy(scores, labels, loss_fn, threshold):
    J = loss_fn(scores, labels)
    loss = J.detach().item()
    classification = (scores > threshold).long()
    acc = torch.sum(classification == labels).item()
    nb_data = labels.size(0)

    return (loss, acc, nb_data)

def evaluate_iso(g, q, scores, threshold):
    def get_label_from_feature(feature):        
        return torch.argmax(feature[:, :-1], dim=1)

    g.ndata['l'] = get_label_from_feature(g.ndata['feat'])
    g_nx = g.to_networkx(node_attrs=['l'])

    q.ndata['l'] = get_label_from_feature(q.ndata['feat'])
    q_nx = q.to_networkx(node_attrs=['l'])

    classification = scores > threshold
    node_tensor = classification.nonzero()
    nodes = node_tensor[:, :1].flatten().tolist()
    g_reduced = g_nx.subgraph(nodes).copy()

    match_ml = nx_iso.GraphMatcher(g_reduced, q_nx, node_match=node_match)
    subisos_ml = match_ml.subgraph_isomorphisms_iter()
    num_ml = len(list(subisos_ml)) + 1

    match_orig = nx_iso.GraphMatcher(g_nx, q_nx, node_match=node_match)
    subisos_orig = match_orig.subgraph_isomorphisms_iter()
    num_orig = len(list(subisos_orig)) + 1

    acc = num_ml / num_orig                        
    return acc

def save_visuals(batch_scores, writer, threshold):
    classification = (batch_scores > threshold).long()
    writer.add_model(graph_name='rnd', model_name='GNN',
            nlabels=classification)


def evaluate(model, data_loader, threshold):
    model.eval()

    iso_acc = acc = nb_data = loss = 0

    with torch.no_grad():
        for iter, (batch_gs, batch_qs, batch_labels) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata['feat']
            batch_E = batch_gs.graph.edata['feat']

            batch_X_q = batch_qs.graph.ndata['feat']
            batch_E_q = batch_qs.graph.edata['feat']

            batch_scores = model(batch_gs, batch_X, batch_E, batch_qs, batch_X_q, batch_E_q)

            # save_visuals(batch_scores)
            curr_loss, curr_acc, curr_nb_data = evaluate_accuracy(batch_scores, batch_labels, loss_fn, threshold)
            acc += curr_acc
            loss += curr_loss
            loss /= iter + 1
            acc /= curr_nb_data
            nb_data += curr_nb_data
            iso_acc += evaluate_iso(batch_gs.graph, batch_qs.graph, batch_scores, threshold)
            iso_acc /= iter + 1
    return (iso_acc, acc, loss)


res = evaluate(model, test_loader, threshold)
print(res)