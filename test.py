import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import collate, numisos

model = torch.load('./models/model.pth')

num_g = 64
num_q = 8

testset, scaler = numisos.gen_dataset(num_g, num_q)
test_loader = DataLoader(testset, batch_size=10, shuffle=False, collate_fn=collate)

loss = nn.L1Loss()

import torch

def evaluate(model, data_loader, loss, debug):
    model.eval()
    epoch_test_loss = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_gs, batch_qs, batch_labels) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata['feat']
            batch_E = batch_gs.graph.edata['feat']

            batch_X_q = batch_qs.graph.ndata['feat']
            batch_E_q = batch_qs.graph.edata['feat']

            batch_scores = model(batch_gs, batch_qs, batch_X, batch_E, batch_X_q, batch_E_q)
            J = loss(batch_scores, batch_labels)

            if debug:
                sc_orig = scaler.rescale(batch_scores)
                labels_orig = scaler.rescale(batch_labels)
                print(sc_orig)
                print(labels_orig)

            epoch_test_loss += J.detach().item()
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)    
    
    return epoch_test_loss

loss = evaluate(model, test_loader, loss, True)
print(loss)