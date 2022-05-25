from statistics import mean
from unittest import result
from ladon.config import NAME_DICTIONARY
from ladon.classes.network import Network, NoOpinionNetwork, ScaleFreeNetwork
import networkx as nx
import numpy as np
import optuna
import netrd
from tqdm import tqdm
from ladon.helpers.helpers import find_average_path, get_main_component
import random
import pickle as pkl
import multiprocessing as mp
import joblib
import plotly.io as pio
import math


def make_barabasi_network_by_seed(dictionary, run):
    random.seed(run)
    np.random.seed(run)
    network = ScaleFreeNetwork(dictionary)
    # network.run_simulation()
    return network


def run_single_simulation(dictionary, run, target, target_dictionary) -> float:
    my_network = make_barabasi_network_by_seed(dictionary=dictionary, run=run)

    clustering_diff = abs(
        nx.algorithms.cluster.average_clustering(my_network.graph)
        - (target_dictionary.get("clustering"))
    )
    network_avg_path = find_average_path(my_network.graph)

    average_path_diff = abs(
        network_avg_path - target_dictionary.get("average_path")
    ) / target_dictionary.get("denominator")

    distance_algorithm = netrd.distance.DegreeDivergence()
    JSD = distance_algorithm.dist(my_network.graph, target)
    minimize_array = np.array([clustering_diff, average_path_diff, JSD])
    mean = np.mean(minimize_array)
    return mean


def objective(trial, target, repeats, target_dictionary) -> float:
    N_TARGET = target.number_of_nodes()
    K = trial.suggest_int("K", 1, 20)
    dictionary = {
        "N_TARGET": N_TARGET,
        "N_TIMESTEPS": N_TARGET * 20,
        "K": K,
        "RECORD": False,
    }

    results = [
        run_single_simulation(dictionary, run, target, target_dictionary)
        for run in range(repeats)
    ]

    return mean(results)


if __name__ == "__main__":
    resulting_dictionary = {}
    for name, network in NAME_DICTIONARY.items():
        print(f"--- NOW RUNNING: {name} ---")
        network = get_main_component(network)
        average_path = find_average_path(network=network)
        study = optuna.create_study(study_name=name, direction="minimize")
        target_dictionary = {
            "clustering": nx.algorithms.cluster.average_clustering(network),
            "assortativity": nx.algorithms.assortativity.degree_assortativity_coefficient(
                network
            ),
            "average_path": average_path,
            "denominator": average_path + 2,
        }
        study.optimize(
            lambda trial: objective(trial, network, 1, target_dictionary), n_trials=500
        )
        study.best_params
        resulting_dictionary[name] = study.best_params
        joblib.dump(
            study, f"../analysis/data/optimization/{name}_study_no_barabasi.pkl"
        )
    with open(
        f"../analysis/data/optimization/best_optimization_results_barabasi.pkl",
        "wb",
    ) as handle:
        pkl.dump(resulting_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)