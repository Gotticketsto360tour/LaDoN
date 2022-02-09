from network import Network
import networkx as nx
import numpy as np
from helpers import (
    THRESHOLDS,
    RANDOMNESS,
    POSITIVE_LEARNING_RATES,
    NEGATIVE_LEARNING_RATES,
)
import itertools
import multiprocessing as mp
import pickle as pkl


dictionary = {
    "THRESHOLD": 1.2,
    "N_TARGET": 1000,
    "RANDOMNESS": 0.5,
    "N_TIMESTEPS": 10,
    "POSITIVE_LEARNING_RATE": 0.2,
    "NEGATIVE_LEARNING_RATE": 0.3,
    "STOP_AT_TARGET": True,
}

my_network = Network(dictionary)


def make_one_simulation(dictionary):
    networks = [Network(dictionary) for _ in range(10)]
    for network in networks:
        network.run_simulation()

    threshold = dictionary.get("THRESHOLD")
    randomness = dictionary.get("RANDOMNESS")
    positive_learning_rate = dictionary.get("POSITIVE_LEARNING_RATE")
    negative_learning_rate = dictionary.get("NEGATIVE_LEARNING_RATE")

    out_dict = {
        "threshold": threshold,
        "randomness": randomness,
        "positive_learning_rate": positive_learning_rate,
        "negative_learning_rate": negative_learning_rate,
        "opinions": np.hstack(
            [network.get_opinion_distribution() for network in networks]
        ),
        "initial_opinions": np.hstack(
            [network.get_initial_opinion_distribution() for network in networks]
        ),
        "degrees": np.hstack(
            [network.get_degree_distribution() for network in networks]
        ),
        "distances": np.hstack(
            [network.get_opinion_distances() for network in networks]
        ),
        "centrality": np.hstack([network.get_centrality() for network in networks]),
    }

    with open(
        f"S{threshold}-{randomness}-{positive_learning_rate}-{negative_learning_rate}.pkl",
        "wb",
    ) as handle:
        pkl.dump(out_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


my_dictionary = make_one_simulation(dictionary)


def make_all_simulations():

    values = [THRESHOLDS, RANDOMNESS, POSITIVE_LEARNING_RATES, NEGATIVE_LEARNING_RATES]

    combinations = list(itertools.product(*values))

    dictionary = {
        "THRESHOLD": 1.2,
        "N_TARGET": 1000,
        "RANDOMNESS": 0.5,
        "N_TIMESTEPS": 10,
        "POSITIVE_LEARNING_RATE": 0.2,
        "NEGATIVE_LEARNING_RATE": 0.3,
        "STOP_AT_TARGET": True,
    }
