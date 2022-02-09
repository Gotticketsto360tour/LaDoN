import numpy as np
from prettytable import RANDOM
from agent import Agent
import networkx as nx

THRESHOLDS = [0.7, 0.8, 0.9, 1, 1.1, 1.2]
RANDOMNESS = [0.1, 0.2, 0.3, 0.4, 0.5]
POSITIVE_LEARNING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
NEGATIVE_LEARNING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]


def find_distance(A: Agent, B: Agent):
    return abs(A.opinion - (B.opinion))


def compare_values(A: Agent, B: Agent):
    return np.linalg.norm((A.opinion - B.opinion))


def find_average_path(network):
    largest_cc = max(nx.connected_components(network), key=len)
    sub = network.subgraph(largest_cc)
    return nx.algorithms.average_shortest_path_length(sub)


def compare_vectors(A: Agent, B: Agent, neigbor_number: int):
    weight = A.social_memory[neigbor_number] / (A.social_memory[neigbor_number] + 4)
    comparison = weight * np.linalg.norm((A.inner_vector - B.inner_vector)) + (
        1 - weight
    ) * np.linalg.norm((A.outer_vector - B.outer_vector))
    return comparison
