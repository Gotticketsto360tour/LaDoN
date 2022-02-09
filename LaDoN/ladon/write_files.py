from scipy import rand
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


def make_one_simulation(
    threshold, randomness, positive_learning_rate, negative_learning_rate
):

    dictionary = {
        "THRESHOLD": threshold,
        "N_TARGET": 1000,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": 10,
        "POSITIVE_LEARNING_RATE": positive_learning_rate,
        "NEGATIVE_LEARNING_RATE": negative_learning_rate,
        "STOP_AT_TARGET": True,
    }

    networks = [Network(dictionary) for _ in range(10)]
    for network in networks:
        network.run_simulation()

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
        f"analysis/data/simulations/S{threshold}-{randomness}-{positive_learning_rate}-{negative_learning_rate}.pkl",
        "wb",
    ) as handle:
        pkl.dump(out_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def make_all_simulations():

    values = [THRESHOLDS, RANDOMNESS, POSITIVE_LEARNING_RATES, NEGATIVE_LEARNING_RATES]

    combinations = list(itertools.product(*values))

    pool = mp.Pool(mp.cpu_count())

    results = [pool.apply(make_one_simulation, args=arg) for arg in combinations]

    pool.close()


if __name__ == "__main__":
    make_all_simulations()
