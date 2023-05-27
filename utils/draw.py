import networkx as nx
import matplotlib.pyplot as plt


def draw(graph):
    pos = nx.drawing.layout.spring_layout(graph)
    # pos = nx.drawing.layout.circular_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True)

    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
