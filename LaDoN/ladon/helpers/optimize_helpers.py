from statistics import mean
from typing import Callable, Dict
from unittest import result

from ladon.config import NAME_DICTIONARY
from ladon.classes.network import Network, NoOpinionNetwork, ScaleFreeNetwork
import networkx as nx
import numpy as np
import optuna
import netrd
from ladon.helpers.helpers import find_average_path, get_main_component
import random
import pickle as pkl
import joblib


def make_network_by_seed(
    dictionary: Dict, run: int, run_simulation: bool = True
) -> Network:
    random.seed(run)
    np.random.seed(run)
    network = Network(dictionary)
    if run_simulation:
        network.run_simulation()
    return network


def make_barabasi_network_by_seed(dictionary: Dict, run: int) -> ScaleFreeNetwork:
    random.seed(run)
    np.random.seed(run)
    network = ScaleFreeNetwork(dictionary)
    # network.run_simulation()
    return network


def make_no_opinion_network_by_seed(dictionary: Dict, run: int) -> NoOpinionNetwork:
    random.seed(run)
    np.random.seed(run)
    network = NoOpinionNetwork(dictionary)
    network.run_simulation()
    return network


def make_theoretical_network_by_seed(dictionary: Dict, run: int) -> NoOpinionNetwork:
    random.seed(run)
    np.random.seed(run)
    network = NoOpinionNetwork(dictionary)
    return network


def run_single_simulation(
    dictionary: Dict, run: int, target: nx.Graph(), target_dictionary: Dict, type: str
) -> float:
    """Run a single simulation and return the mean of the vector of differences.

    Args:
        dictionary (Dict): Dictionary specifying how the Network class should be initiated.
        run (int): Integer specifying which seed should be set for reproducibility
        target (nx.Graph): Empirical data to match by the generated network.
        target_dictionary (Dict): Dictionary containing precomputed characteristics from the target network
        type (str): Specifying which type of network generating method to be used.

    Returns:
        float: Mean of the vector of differences
    """
    type_dict = {
        "barabasi": make_barabasi_network_by_seed,
        "theoretical": make_theoretical_network_by_seed,
        "opinion": make_network_by_seed,
        "no_opinion": make_no_opinion_network_by_seed,
    }

    network_function = type_dict.get(type)

    my_network = network_function(dictionary=dictionary, run=run)

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


def run_optimization(objective: Callable, type: str) -> None:
    resulting_dictionary = {}

    for name, network in NAME_DICTIONARY.items():
        network = get_main_component(network=network)
        average_path = find_average_path(network=network)
        print(f"--- NOW RUNNING: {name} ---")
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
            lambda trial: objective(
                trial=trial,
                target=network,
                target_dictionary=target_dictionary,
                repeats=3,
            ),
            n_trials=400,
            n_jobs=-1,
        )
        resulting_dictionary[name] = study.best_params
        joblib.dump(study, f"../analysis/data/optimization/{name}_study{type}.pkl")
    with open(
        f"../analysis/data/optimization/best_optimization_results_{type}.pkl",
        "wb",
    ) as handle:
        pkl.dump(resulting_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)
