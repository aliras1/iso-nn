# Import libs
from collections import OrderedDict
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

from models import TriCount
os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl
from dgl import DGLGraph
from dgl.data import *

import time

import numpy as np
import networkx as nx
from res.plot_lib import set_default
import matplotlib.pyplot as plt

from data import semi_supervised_generic_gen
from data.common import collate_graphs

def collate(samples):
    gs, qs, _ = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    batched_g = collate_graphs(gs)
    batched_q = collate_graphs(qs)
    batched_subiso = [set() for _ in range(batched_g.graph.num_nodes())]
    n_g = 0
    n_q = 0

    for g, q, subisos in samples:
        for subiso in subisos:
            for k_, v_ in subiso.items():
                k = k_ + n_g
                v = v_ + n_q                
                batched_subiso[k].add(v)
        n_g += g.num_nodes()
        n_q += q.num_nodes()
    
    # mark nodes that are not in any isomorphisms with -1
    for mapping in batched_subiso:
        if len(mapping) == 0:
            mapping.add(-1)

    return batched_g, batched_q, batched_subiso


# set_default(figsize=(3, 3), dpi=150)
def draw(G, title):
    pos = nx.drawing.layout.spring_layout(G.to_networkx())
    nx.draw(G.to_networkx(), pos=pos, with_labels=True)

    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

# Generate artifical graph dataset
num_g = 1000
num_q = 1
min_num_nodes = 15
max_num_nodes = 25
edge_probability = 0.2

start_tic = time.perf_counter()
dataset = semi_supervised_generic_gen.gen_dataset(num_g, min_num_nodes, max_num_nodes, edge_probability)
semi_supervised_generic_gen.save_dataset(dataset, "generic_match_1000")
dataset = semi_supervised_generic_gen.load_dataset("generic_match_1000")

dataset_size = len(dataset)
trainset_perc = 0.77
testset_perc = 1.0 - 0.77
dataset_split = int(trainset_perc * dataset_size)
num_epochs = 100
trainset = dataset[:dataset_split]
testset = dataset[dataset_split:]
end_tic = time.perf_counter()
print(f'dataset gen: {end_tic-start_tic}s')


def train(model, data_loader, loss):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_gs, batch_qs, batch_mappings) in enumerate(data_loader):
        start_tic = time.perf_counter()
                
        batch_X = batch_gs.graph.ndata['feat']
        batch_E = batch_gs.graph.edata['feat']
        batch_g_scores = model(batch_gs, batch_X, batch_E)

        batch_X_q = batch_qs.graph.ndata['feat']
        batch_E_q = batch_qs.graph.edata['feat']
        batch_q_scores = model(batch_qs, batch_X_q, batch_E_q)

        mappings = [random.choice(tuple(m)) for m in batch_mappings]
        q_examples = [
            batch_q_scores[v] if v != -1 
            else batch_q_scores[random.randrange(0, len(batch_q_scores))] 
                for v in mappings]
        q_examples_tensor = torch.stack(q_examples)

        labels = [1 if v != -1 else -1 for v in mappings]
        labels_tensor = torch.FloatTensor(labels)

        J = loss(batch_g_scores, q_examples_tensor, labels_tensor)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        
        epoch_loss += J.detach().item()
        # epoch_train_acc += accuracy(batch_scores, batch_labels)
        # nb_data += batch_mappings.size(0)

        end_tic = time.perf_counter()
        print(f'batch {iter}: {end_tic-start_tic}s')

    epoch_loss /= (iter + 1)
    # epoch_train_acc /= nb_data

    return epoch_loss#, epoch_train_acc

import torch

def accuracy(g_scores, q_scores, labels, loss):
    losses = [loss(g_score, q_score, label) for g_score, q_score, label in zip(g_scores, q_scores, labels)]
    classifications = torch.clone(g_scores)                           
    for sc in classifications:
        sc[0] = 1.0 if sc[0] > 0.6 else 0.0
    return torch.sum(classifications == labels).item()

def evaluate(model, data_loader, loss):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_gs, batch_qs, batch_mappings) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata['feat']
            batch_E = batch_gs.graph.edata['feat']
            batch_g_scores = model(batch_gs, batch_X, batch_E)

            batch_X_q = batch_qs.graph.ndata['feat']
            batch_E_q = batch_qs.graph.edata['feat']
            batch_q_scores = model(batch_qs, batch_X_q, batch_E_q)

            mappings = [random.choice(tuple(m)) for m in batch_mappings]
            q_examples = [
                batch_q_scores[v] if v != -1 
                else batch_q_scores[random.randrange(0, len(batch_q_scores))] 
                    for v in mappings]
            q_examples_tensor = torch.stack(q_examples)

            labels = [1 if v != -1 else -1 for v in mappings]
            labels_tensor = torch.FloatTensor(labels)

            J = loss(batch_g_scores, q_examples_tensor, labels_tensor)

            epoch_test_loss += J.detach().item()
            # epoch_test_acc += accuracy(batch_g_scores, q_examples_tensor, labels_tensor, loss)
            nb_data += len(batch_mappings)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
    
    return epoch_test_loss, epoch_test_acc

# datasets
train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)

# Create model
model = TriCount(input_dim=5, hidden_dim=100, output_dim=50, L=6)
loss = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

epoch_train_losses = []
epoch_test_losses = []
epoch_train_accs = []
epoch_test_accs = []

plt.plot([],[])
plt.plot([],[])

for epoch in range(num_epochs):
    
    start = time.time()
    train_loss = train(model, train_loader, loss)
    test_loss, test_acc = evaluate(model, test_loader, loss)

    print(f'Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, test_acc {test_acc:.4f}')
    
    epoch_train_losses.append(train_loss)
    epoch_test_accs.append(test_acc)
    indices = [i for i in range(epoch+1)]

    plt.gca().lines[0].set_xdata(indices)
    plt.gca().lines[0].set_ydata(epoch_train_losses)
    plt.gca().lines[1].set_xdata(indices)
    plt.gca().lines[1].set_ydata(epoch_test_accs)
    
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.pause(0.05)
        
torch.save(model, './models/model_tri_class_net_labels.pth')
plt.show()