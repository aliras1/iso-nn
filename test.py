import dgl
import networkx as nx
from res.plot_lib import set_default
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import classification_gen
from data.common import collate_graphs


model = torch.load('./models/model.pth')

num_g = 64
num_q = 8

def collate(samples):
    gs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    labels = torch.cat(labels)
    batched_g = collate_graphs(gs)
    return batched_g, labels

# dataset = load_dataset('artificial_graphs_1000')
dataset = classification_gen.load_dataset("tri_match_5000")
dataset = dataset[:10]
dataset_split = int(len(dataset) * 0.77)
testset = dataset[dataset_split:]
test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate)

loss = nn.BCELoss()

import torch


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

def evaluate(model, data_loader, loss, debug):
    model.eval()
    epoch_test_loss = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_gs, batch_labels) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata['feat']
            batch_E = batch_gs.graph.edata['feat']            

            batch_scores = model(batch_gs, batch_X, batch_E)
            J = loss(batch_scores, batch_labels)

            if debug:
                sc_orig = batch_scores # scaler.rescale(batch_scores)
                labels_orig = batch_labels # scaler.rescale(batch_labels)
                visualize(batch_gs, sc_orig)
                print(sc_orig)
                sc_predict = torch.clone(sc_orig)                
                
                for sc in sc_predict:
                    sc[0] = 1.0 if sc[0] > 0.6 else 0.0
                print(sc_predict)
                print(labels_orig)

            epoch_test_loss += J.detach().item()
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)    
    
    return epoch_test_loss

loss = evaluate(model, test_loader, loss, True)
print(loss)