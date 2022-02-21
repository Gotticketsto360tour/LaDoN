from scipy import rand
from network import Network
import networkx as nx
import numpy as np
from config import (
    THRESHOLDS,
    RANDOMNESS,
    POSITIVE_LEARNING_RATES,
    NEGATIVE_LEARNING_RATES,
    TIE_DISSOLUTIONS,
)
import random
import itertools
import multiprocessing as mp
import pickle as pkl


def make_network_by_seed(dictionary, run):
    random.seed(run)
    np.random.seed(run)
    network = Network(dictionary)
    network.run_simulation()
    return network


def make_one_simulation(
    threshold,
    randomness,
    positive_learning_rate,
    negative_learning_rate,
    tie_dissolution,
):
    dictionary = {
        "THRESHOLD": threshold,
        "N_TARGET": 1000,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": 10000,
        "POSITIVE_LEARNING_RATE": positive_learning_rate,
        "NEGATIVE_LEARNING_RATE": negative_learning_rate,
        "P": 0.4,
        "K": 7,
        "TIE_DISSOLUTION": tie_dissolution,
        "RECORD": True,
    }

    networks = [make_network_by_seed(dictionary, run) for run in range(10)]

    out_dict_final_state = {
        "threshold": threshold,
        "randomness": randomness,
        "positive_learning_rate": positive_learning_rate,
        "negative_learning_rate": negative_learning_rate,
        "tie_dissolution": tie_dissolution,
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
        "clustering": np.hstack([network.get_clustering() for network in networks]),
    }

    out_dict_over_time = {
        "threshold": threshold,
        "randomness": randomness,
        "positive_learning_rate": positive_learning_rate,
        "negative_learning_rate": negative_learning_rate,
        "tie_dissolution": tie_dissolution,
        "mean_distance": np.hstack(
            [np.array(network.MEAN_DISTANCE) for network in networks]
        ),
        "negative_ties_dissoluted": np.hstack(
            [np.array(network.NEGATIVE_TIES_DISSOLUTED) for network in networks]
        ),
        "mean_absolute_opinion": np.hstack(
            [np.array(network.MEAN_ABSOLUTE_OPINIONS) for network in networks]
        ),
        "sd_absolute_opinion": np.hstack(
            [np.array(network.SD_ABSOLUTE_OPINIONS) for network in networks]
        ),
        "timestep": np.array(
            [timestep for network in networks for timestep in range(0, 10000, 10)]
        ),
    }

    with open(
        f"analysis/data/simulations/final_state/S{threshold}-{randomness}-{positive_learning_rate}-{negative_learning_rate}_{tie_dissolution}.pkl",
        "wb",
    ) as handle:
        pkl.dump(out_dict_final_state, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(
        f"analysis/data/simulations/over_time/S{threshold}-{randomness}-{positive_learning_rate}-{negative_learning_rate}_{tie_dissolution}.pkl",
        "wb",
    ) as handle:
        pkl.dump(out_dict_over_time, handle, protocol=pkl.HIGHEST_PROTOCOL)


def make_all_simulations():

    values = [
        THRESHOLDS,
        RANDOMNESS,
        POSITIVE_LEARNING_RATES,
        NEGATIVE_LEARNING_RATES,
        TIE_DISSOLUTIONS,
    ]

    combinations = list(itertools.product(*values))

    pool = mp.Pool(mp.cpu_count())

    results = [pool.apply(make_one_simulation, args=arg) for arg in combinations]

    pool.close()


if __name__ == "__main__":
    make_all_simulations()
