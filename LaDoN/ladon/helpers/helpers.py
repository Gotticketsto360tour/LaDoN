from typing import List
import matplotlib.pyplot as plt
import numpy as np
from ladon.classes.agent import Agent
import networkx as nx
from collections import Counter
from scipy.spatial import distance
import seaborn as sns

# from numba import jit


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


def find_shortest_path(network: nx.Graph, nodes: list) -> int:
    n1, n2 = np.random.choice(nodes, size=2, replace=False)
    return nx.shortest_path_length(network, source=n1, target=n2)


def get_shortest_path_distribution(network: nx.Graph, n_samples=1000) -> np.array:
    nodes = np.array(network.nodes())
    sampling_shortest_path = np.array(
        [find_shortest_path(network=network, nodes=nodes) for _ in range(n_samples)]
    )
    return sampling_shortest_path


def find_average_path(network: nx.Graph, n_samples=1000) -> float:
    return get_shortest_path_distribution(network, n_samples).mean()


def make_probability_vectors(A: np.array, B: np.array):
    max_value = np.max([np.max(A), np.max(B)])
    A_counter, B_counter = Counter(A), Counter(B)
    A_prop = np.zeros(max_value, dtype=np.int32)
    B_prop = np.zeros(max_value, dtype=np.int32)
    for i in range(max_value):
        A_prop[i] = A_counter[i + 1]
        B_prop[i] = B_counter[i + 1]

    return A_prop, B_prop


def calculate_jsd_for_paths(A: np.array, B: np.array):
    A_prop, B_prop = make_probability_vectors(A, B)
    return distance.jensenshannon(A_prop, B_prop, 2)


def calculate_jsd_from_path_distributions(
    A: nx.Graph, B: nx.Graph, n_samples=1000
) -> float:
    A_dist, B_dist = get_shortest_path_distribution(
        A, n_samples
    ), get_shortest_path_distribution(B, n_samples)
    return calculate_jsd_for_paths(A_dist, B_dist)
