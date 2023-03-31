# Import libs
import pickle
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

from data import classification_gen
from data.common import collate_graphs

def collate(samples):
    gs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    labels = torch.cat(labels)
    batched_g = collate_graphs(gs)
    return batched_g, labels


# set_default(figsize=(3, 3), dpi=150)
def draw(G, title):
    pos = nx.drawing.layout.spring_layout(G.to_networkx())
    nx.draw(G.to_networkx(), pos=pos, with_labels=True)

    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

# Generate artifical graph dataset
num_g = 10000
num_q = 1
min_num_nodes = 20
max_num_nodes = 40
edge_probability = 0.2

start_tic = time.perf_counter()
# dataset = classification_gen.gen_dataset(num_g, min_num_nodes, max_num_nodes, edge_probability)
# classification_gen.save_dataset(dataset, "tri_match_10000")
dataset = classification_gen.load_dataset("tri_match_10000")

dataset_size = len(dataset)
trainset_perc = 0.77
testset_perc = 1.0 - 0.77
dataset_split = int(trainset_perc * dataset_size)
num_epochs = 35
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

    for iter, (batch_gs, batch_labels) in enumerate(data_loader):
        start_tic = time.perf_counter()
                
        batch_X = batch_gs.graph.ndata['feat']
        batch_E = batch_gs.graph.edata['feat']


        batch_scores = model(batch_gs, batch_X, batch_E)
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        
        epoch_loss += J.detach().item()
        # epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

        end_tic = time.perf_counter()
        print(f'batch {iter}: {end_tic-start_tic}s')

    epoch_loss /= (iter + 1)
    # epoch_train_acc /= nb_data

    return epoch_loss#, epoch_train_acc

import torch

def accuracy(scores, labels):
    classifications = torch.clone(scores)                           
    for sc in classifications:
        sc[0] = 1.0 if sc[0] > 0.6 else 0.0
    return torch.sum(classifications == labels).item()

def evaluate(model, data_loader, loss):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_gs, batch_labels) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata['feat']
            batch_E = batch_gs.graph.edata['feat']

            batch_scores = model(batch_gs, batch_X, batch_E)
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
    
    return epoch_test_loss, epoch_test_acc

# datasets
train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)

# Create model
model = TriCount(input_dim=1, hidden_dim=100, output_dim=1, L=6)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

epoch_train_losses = []
epoch_test_losses = []
epoch_train_accs = []
epoch_test_accs = []

for epoch in range(num_epochs):
    
    start = time.time()
    train_loss = train(model, train_loader, loss)
    test_loss, test_acc = evaluate(model, test_loader, loss)
        
    print(f'Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, test_acc {test_acc:.4f}')

torch.save(model, './models/model.pth')