import dgl
from gnnlens import Writer
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as nx_iso
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryAUROC,
)
from torcheval.tools import get_module_summary

from data import *


def line_to_epoch(line: str) -> tuple[float, float, float, float]:
    strs = line.rstrip().split(",")
    return tuple(map(lambda x: float(x), strs))


def draw_training_plot() -> None:
    with open("logs.csv") as logs:
        epochs = map(line_to_epoch, logs.readlines())
        train_losses, train_accs, test_losses, test_accs = map(list, zip(*epochs))

        print(f"accuracies: {max(train_accs)}, {max(test_accs)}")

        fig, ax = plt.subplots()
        ax.plot(train_losses, label="Train loss")
        ax.plot(train_accs, label="Train accuracy")
        leg = ax.legend()
        plt.savefig("train_plot.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.clf()
        plt.cla()

        fig, ax = plt.subplots()
        ax.plot(test_losses, label="Test loss")
        ax.plot(test_accs, label="Test accuracy")
        leg = ax.legend()
        plt.savefig("test_plot.pdf", format="pdf", bbox_inches="tight")
        plt.show()


def analyze_dataset(dataset: list[DatasetEntry]) -> None:
    print(f"dataset size: {len(dataset)}")
    graphs = [entry[0] for entry in dataset]

    nodes = torch.Tensor([graph.num_nodes() for graph in graphs])
    avg_nodes = torch.mean(nodes)

    edges = torch.Tensor([graph.num_edges() for graph in graphs])
    avg_edges = torch.mean(edges)

    batch = dgl.batch(graphs, ndata=["feat"])
    degrees = torch.mean(batch.in_degrees().float())

    print(f"G: {{avg node: {avg_nodes}, avg edge: {avg_edges}, avg deg: {degrees}}}")

    ###############

    patterns = [entry[1] for entry in dataset]
    nodes = torch.Tensor([graph.num_nodes() for graph in patterns])
    avg_nodes = torch.mean(nodes)

    edges = torch.Tensor([graph.num_edges() for graph in patterns])
    avg_edges = torch.mean(edges)

    batch = dgl.batch(patterns, ndata=["feat"])
    degrees = torch.mean(batch.in_degrees().float())

    print(f"P: {{avg node: {avg_nodes}, avg edge: {avg_edges}, avg deg: {degrees}}}")

    ##############

    labels = [entry[2] for entry in dataset]
    l = torch.concat(labels).squeeze()
    size = l.size(dim=0)
    sum = l.sum()
    print(
        f"#labels: {size}, #class1: {sum}, #class1/#labels: {sum / size}, #labels/#class1: {size / sum}"
    )


def visualize_predictions(dir: str) -> None:
    writer = Writer(dir)
    model.eval()

    for i in range(min(5, len(dataset))):
        graph = dataset[i][0]
        pattern = dataset[i][1]
        nlabels = graph.ndata["label"]
        nlabels = torch.squeeze_copy(dataset[i][2])
        writer.add_graph(
            name=f"rnd-{i}", graph=graph, nlabels=nlabels, num_nlabel_types=4
        )
        writer.add_graph(
            name=f"rnd-pattern-{i}",
            graph=pattern,
            nlabels=pattern.ndata["label"],
            num_nlabel_types=4,
        )

        batch_gs, batch_qs, labels = collate([dataset[i]])
        batch_X = batch_gs.graph.ndata["feat"]
        batch_X_q = batch_qs.graph.ndata["feat"]

        batch_scores = model(batch_gs, batch_X, batch_qs, batch_X_q)
        pred = (batch_scores > 0.6).long()

        writer.add_model(graph_name=f"rnd-{i}", model_name="GNN", nlabels=pred)
        writer.add_model(
            graph_name=f"rnd-{i}", model_name="labels", nlabels=graph.ndata["label"]
        )

    writer.close()


def evaluate_iso(
    g: dgl.DGLGraph, q: dgl.DGLGraph, scores: torch.Tensor, threshold: float
) -> float:
    def get_label_from_feature(feature):
        return torch.argmax(feature[:, :-1], dim=1)

    g.ndata["l"] = get_label_from_feature(g.ndata["feat"])
    g_nx = g.to_networkx(node_attrs=["l"])

    q.ndata["l"] = get_label_from_feature(q.ndata["feat"])
    q_nx = q.to_networkx(node_attrs=["l"])

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


def evaluate(model: nn.Module, data_loader: DataLoader, threshold: float) -> tuple:
    model.eval()

    iso_acc = loss = 0

    with torch.no_grad():
        for iter, (batch_gs, batch_qs, batch_labels) in enumerate(data_loader):
            batch_X = batch_gs.graph.ndata["feat"]
            batch_X_q = batch_qs.graph.ndata["feat"]

            batch_scores = model(batch_gs, batch_X, batch_qs, batch_X_q)

            # save_visuals(batch_scores)
            J = loss_fn(batch_scores, batch_labels)
            loss += J.detach().item()
            acc_metric.update(batch_scores.squeeze(), batch_labels.squeeze())
            f1_metric.update(batch_scores.squeeze(), batch_labels.squeeze())
            auroc_metric.update(batch_scores.squeeze(), batch_labels.squeeze())
            for m in confusion_metrics:
                m.update(batch_scores.squeeze(), batch_labels.squeeze().long())

            # iso_acc += evaluate_iso(
            #     batch_gs.graph, batch_qs.graph, batch_scores, threshold
            # )

        loss /= iter + 1
        iso_acc /= iter + 1
    return (
        iso_acc,
        acc_metric.compute(),
        f1_metric.compute(),
        auroc_metric.compute(),
        [m.compute() for m in confusion_metrics],
        loss,
    )


def plot_roc_curve(confusion_matrices: list[torch.Tensor]) -> None:
    # (true positive, false negative) ,
    # (false positive, true negative)

    tpr = torch.Tensor(
        [c[0][0] / (c[0][0] + c[0][1]) for c in confusion_matrices]
    )  # tp / (tp + fn)
    fpr = torch.Tensor(
        [c[1][0] / (c[1][0] + c[1][1]) for c in confusion_matrices]
    )  # fp / (fp + tn)
    gmeans = torch.sqrt(tpr * (1 - fpr))
    idx = torch.argmax(gmeans)  # idx of best G-mean
    best_threshold = idx * 0.05

    plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    plt.plot(fpr, tpr, marker=".", label="Model")
    plt.scatter(
        fpr[idx],
        tpr[idx],
        marker="o",
        color="black",
        label=f"Best (threshold = {best_threshold:.2f})",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()


model = torch.load("./models/iso_net.pth")
dataset = load_dataset("generic_class_10000_valid")

ms = get_module_summary(model)
print(ms)

# draw_training_plot()
analyze_dataset(dataset)
# visualize_predictions("visu")

test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
threshold = 0.45
loss_fn = nn.BCELoss()
acc_metric = BinaryAccuracy(threshold=threshold)
f1_metric = BinaryF1Score(threshold=threshold)
confusion_metrics = [BinaryConfusionMatrix(threshold=t * 0.05) for t in range(21)]
auroc_metric = BinaryAUROC()

res = evaluate(model, test_loader, threshold)
consfusion_matrices = res[4]
print(res)
plot_roc_curve(consfusion_matrices)
