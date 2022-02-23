import numpy as np
from agent import Agent
import networkx as nx
import random
from statistics import mean


def find_distance(A: Agent, B: Agent):
    return abs(A.opinion - (B.opinion))


def compare_values(A: Agent, B: Agent):
    return np.linalg.norm((A.opinion - B.opinion))


def get_main_component(network):
    largest_cc = max(nx.connected_components(network), key=len)
    return network.subgraph(largest_cc)


def find_average_path(network):
    sub = get_main_component(network)
    nodes = list(sub.nodes())
    n_samples = 1000
    sampling_shortest_path = []
    for _ in range(n_samples):
        n1, n2 = random.choices(nodes, k=2)
        sampling_shortest_path.append(
            nx.shortest_path_length(sub, source=n1, target=n2)
        )

    return mean(sampling_shortest_path)


def compare_vectors(A: Agent, B: Agent, neigbor_number: int):
    weight = A.social_memory[neigbor_number] / (A.social_memory[neigbor_number] + 4)
    comparison = weight * np.linalg.norm((A.inner_vector - B.inner_vector)) + (
        1 - weight
    ) * np.linalg.norm((A.outer_vector - B.outer_vector))
    return comparison
