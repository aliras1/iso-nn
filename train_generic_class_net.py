import matplotlib.pyplot as plt
import networkx as nx
import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy

os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl
from dgl import DGLGraph
from dgl.data import *

from data import *
from data.common import collate_graphs
from models.generic_class_net import GenericClassNet


def train(
    model: nn.Module, data_loader: DataLoader, metrics: MulticlassAccuracy
) -> tuple[float, float]:
    model.train()
    epoch_loss = 0

    start_tic = time.perf_counter()

    for iter, (g, q, labels) in enumerate(data_loader):
        X = g.graph.ndata["feat"]
        X_q = q.graph.ndata["feat"]

        scores = model(g, X, q, X_q)

        #####
        weights = (torch.clone(labels) * 3.0).clamp_min(min=1.0)  # sparse
        # weights = ((1.0 - torch.clone(labels)) * 1.2).clamp_min(min=1.0)  # dense
        loss_fn = nn.BCELoss(weight=weights)
        #####
        J = loss_fn(scores, labels)
        l = J.detach().item()
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        # scheduler.step(metrics=l)
        epoch_loss += J.detach().item()
        metrics.update(scores.squeeze(), labels.squeeze())

    epoch_loss /= iter + 1

    end_tic = time.perf_counter()
    print(f"trained in: {end_tic-start_tic}s")

    acc = metrics.compute()
    metrics.reset()
    return epoch_loss, acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss: nn.Module,
    metrics: MulticlassAccuracy,
) -> None:
    model.eval()
    epoch_test_loss = 0

    with torch.no_grad():
        for iter, (g, q, labels) in enumerate(data_loader):
            X = g.graph.ndata["feat"]
            X_q = q.graph.ndata["feat"]

            scores = model(g, X, q, X_q)

            J = loss(scores, labels)
            epoch_test_loss += J.detach().item()
            metrics.update(scores.squeeze(), labels.squeeze())

        epoch_test_loss /= iter + 1

    acc = metrics.compute()
    metrics.reset()
    return epoch_test_loss, acc


# Generate artifical graph dataset
num_g = 10000
min_num_nodes = 80
max_num_nodes = 150
edge_probability = 0.05

start_tic = time.perf_counter()
dataset = gen_dataset(num_g, min_num_nodes, max_num_nodes, edge_probability)
save_dataset(dataset, "generic_class_10000_train")
validset = gen_dataset(num_g, min_num_nodes, max_num_nodes, edge_probability)
save_dataset(validset, "generic_class_10000_valid")
# dataset = load_dataset("generic_class_10000_train")

dataset_size = len(dataset)
trainset_perc = 0.77
testset_perc = 1.0 - 0.77
dataset_split = int(trainset_perc * dataset_size)
num_epochs = 300
trainset = dataset[:dataset_split]
testset = dataset[dataset_split:]
end_tic = time.perf_counter()
print(f"dataset loaded: {end_tic-start_tic}s")

# datasets
train_loader = DataLoader(trainset, batch_size=50, shuffle=True, collate_fn=collate)
test_loader = DataLoader(testset, batch_size=50, shuffle=False, collate_fn=collate)

# Create model
model = GenericClassNet(input_dim=5, hidden_dim=100, output_dim=50, L=8)
# model = torch.load("./models/model_generic_class_net7.pth")
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
train_metrics = BinaryAccuracy(threshold=0.6)
test_metrics = BinaryAccuracy(threshold=0.6)

epoch_train_losses = []
epoch_test_losses = []
epoch_train_accs = []
epoch_test_accs = []
best_test_acc = 0.0

plt.plot([], [])
plt.plot([], [])

logfile = open("logs.csv", "w")

for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc = train(model, train_loader, train_metrics)
    test_loss, test_acc = evaluate(model, test_loader, loss, test_metrics)

    logfile.write(f"{train_loss},{train_acc},{test_loss},{test_acc}\n")
    logfile.flush()
    print(
        f"Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}, test_acc {test_acc:.4f}"
    )

    if test_acc > best_test_acc:
        torch.save(model, "./models/model_generic_class_net.pth")
        best_test_acc = test_acc

    epoch_train_losses.append(train_loss)
    epoch_test_accs.append(test_acc)
    indices = [i for i in range(epoch + 1)]

    plt.gca().lines[0].set_xdata(indices)
    plt.gca().lines[0].set_ydata(epoch_train_losses)
    plt.gca().lines[1].set_xdata(indices)
    plt.gca().lines[1].set_ydata(epoch_test_accs)

    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.pause(0.05)

logfile.close()
plt.show()
