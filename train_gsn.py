# Import libs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

from models import GSN
os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset

import time

import numpy as np
import networkx as nx
from res.plot_lib import set_default
import matplotlib.pyplot as plt

from data import *
from models import GSN

# Generate artifical graph dataset
num_g = 512
num_q = 8
# num_g = 1000
# num_q = 15
dataset_size = num_g * num_q
trainset_perc = 0.77
testset_perc = 1.0 - 0.77
dataset_split = int(trainset_perc * dataset_size)

start_tic = time.perf_counter()
trainset, scaler = data_gsn.gen_dataset(int(num_g * trainset_perc), num_q)
testset, _ = data_gsn.gen_dataset(int(num_g * testset_perc), num_q)
end_tic = time.perf_counter()
print(f'dataset gen: {end_tic-start_tic}s')


def train(model, data_loader, loss):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_gs, batch_qs, batch_labels) in enumerate(data_loader):
        batch_X = batch_gs.graph.ndata['feat']
        batch_E = batch_gs.graph.edata['feat']
        batch_X_q = batch_qs.graph.ndata['feat']
        batch_E_q = batch_qs.graph.edata['feat']


        batch_scores = model(batch_gs, batch_qs, batch_X, batch_E, batch_X_q, batch_E_q)
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        
        epoch_loss += J.detach().item()
        # epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

    epoch_loss /= (iter + 1)
    # epoch_train_acc /= nb_data

    return epoch_loss

import torch

def evaluate(model, data_loader, loss):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_gs, batch_qs, batch_labels) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata['feat']
            batch_E = batch_gs.graph.edata['feat']

            batch_X_q = batch_qs.graph.ndata['feat']
            batch_E_q = batch_qs.graph.edata['feat']

            batch_scores = model(batch_gs, batch_qs, batch_X, batch_E, batch_X_q, batch_E_q)
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            # epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        # epoch_test_acc /= nb_data
    
    return epoch_test_loss

# datasets
train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)

# Create model
model = GSN(input_dim=2, hidden_dim=100, output_dim=1, L=6)
loss = nn.L1Loss()
# for n, p in model.named_parameters():
#     print(n)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

epoch_train_losses = []
epoch_test_losses = []
epoch_train_accs = []
epoch_test_accs = []

for epoch in range(15):
    
    start = time.time()
    train_loss = train(model, train_loader, loss)
    test_loss = evaluate(model, test_loader, loss)
        
    print(f'Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')

torch.save(model, './models/model.pth')