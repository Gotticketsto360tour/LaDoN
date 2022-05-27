from typing import List
import matplotlib.pyplot as plt
import numpy as np
from ladon.classes.agent import Agent
import networkx as nx
import random
from statistics import mean
import seaborn as sns


def rename_plot(g: sns.FacetGrid, titles: List, legend: str) -> plt.Figure:
    for ax, title in zip(g.axes.flatten(), titles):
        ax.set_title(title)
    g.legend.set(title=legend)
    g.legend.set_frame_on(True)
    return g.figure


def find_distance(A: Agent, B: Agent) -> float:
    return abs(A.opinion - (B.opinion))


def get_main_component(network: nx.Graph) -> nx.Graph:
    largest_cc = max(nx.connected_components(network), key=len)
    return network.subgraph(largest_cc)


def find_average_path(network: nx.Graph) -> float:
    nodes = list(network.nodes())
    n_samples = 1000
    sampling_shortest_path = []
    for _ in range(n_samples):
        n1, n2 = random.choices(nodes, k=2)
        sampling_shortest_path.append(
            nx.shortest_path_length(network, source=n1, target=n2)
        )

    return mean(sampling_shortest_path)
