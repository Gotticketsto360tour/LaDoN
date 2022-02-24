from statistics import mean
from typing import Dict
from unittest import result
from helpers import get_main_component
from network import Network
import networkx as nx
import numpy as np
import optuna
import netrd
from tqdm import tqdm
from helpers import find_average_path
import random
import pickle as pkl
import multiprocessing as mp
import joblib
import plotly.io as pio
import math

pio.renderers.default = "notebook"

# study = joblib.load("analysis/data/optimization/polbooks_study.pkl")

# fig = optuna.visualization.plot_param_importances(study)
# fig.show()

# fig = optuna.visualization.plot_optimization_history(study)
# fig.show()

# fig = optuna.visualization.plot_edf(study)
# fig.show()

# fig = optuna.visualization.plot_slice(study)
# fig.show()


def make_network_by_seed(dictionary, run):
    random.seed(run)
    np.random.seed(run)
    network = Network(dictionary)
    network.run_simulation()
    return network


def run_single_simulation(
    dictionary: Dict, run: int, target: nx.Graph(), target_dictionary: Dict
):
    """Run a single simulation and return the eucledian norm of the vector of differences.

    Args:
        dictionary (Dict): Dictionary specifying how the Network class should be initiated.
        run (int): _description_
        target (nx.Graph): _description_
        target_dictionary (Dict): _description_

    Returns:
        _type_: _description_
    """
    my_network = make_network_by_seed(dictionary=dictionary, run=run)

    clustering_diff = abs(
        nx.algorithms.cluster.average_clustering(my_network.graph)
        - (target_dictionary.get("clustering"))
    )
    assortativity_diff = abs(
        nx.algorithms.assortativity.degree_assortativity_coefficient(my_network.graph)
        - (target_dictionary.get("assortativity"))
    )
    network_avg_path = find_average_path(my_network.graph)

    average_path_diff = abs(
        network_avg_path - target_dictionary.get("average_path")
    ) / max([network_avg_path, target_dictionary.get("average_path")])

    distance_algorithm = netrd.distance.DegreeDivergence()
    JSD = distance_algorithm.dist(my_network.graph, target)
    minimize_array = np.array(
        [clustering_diff, assortativity_diff, average_path_diff, JSD]
    )
    norm = np.linalg.norm(minimize_array)
    return norm


def objective(trial, target, repeats, target_dictionary):
    N_TARGET = target.number_of_nodes()
    N_EDGES = target.number_of_edges()
    K = round(N_EDGES / N_TARGET)
    threshold = trial.suggest_float("threshold", 0.5, 2)
    randomness = trial.suggest_float("randomness", 0.1, 1)
    positive_learning_rate = trial.suggest_float("positive_learning_rate", 0.05, 0.5)
    negative_learning_rate = trial.suggest_float("negative_learning_rate", 0.05, 0.5)
    tie_dissolution = trial.suggest_float("tie_dissolution", 0.1, 1)

    dictionary = {
        "THRESHOLD": threshold,
        "N_TARGET": N_TARGET,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": N_TARGET * 20,
        "POSITIVE_LEARNING_RATE": positive_learning_rate,
        "NEGATIVE_LEARNING_RATE": negative_learning_rate,
        "P": 0.1,
        "K": 2 * K,
        "TIE_DISSOLUTION": tie_dissolution,
        "RECORD": False,
    }

    results = [
        run_single_simulation(dictionary, run, target, target_dictionary)
        for run in range(repeats)
    ]

    return mean(results)


netscience = nx.read_gml(path="analysis/data/netscience/netscience.gml")

facebook = nx.read_edgelist(
    "analysis/data/facebook_combined.txt", create_using=nx.Graph(), nodetype=int
)

astrophysics = nx.read_edgelist(
    "analysis/data/dimacs10-astro-ph/out.dimacs10-astro-ph",
    create_using=nx.Graph(),
    nodetype=int,
)

theoretical_physics = nx.read_edgelist(
    "analysis/data/physics/ca-HepTh.txt",
    create_using=nx.Graph(),
    nodetype=int,
)

karate = nx.karate_club_graph()

polblogs = nx.read_gml(
    path="analysis/data/polblogs/polblogs.gml",
)
polblogs = nx.Graph(polblogs.to_undirected())

polbooks = nx.read_gml(
    path="analysis/data/polbooks/polbooks.gml",
)

dolphin = nx.read_gml(path="analysis/data/dolphins/dolphins.gml")

if __name__ == "__main__":
    resulting_dictionary = {}
    name_dictionary = {
        "karate": karate,
        "dolphin": dolphin,
        "polbooks": polbooks,
        "netscience": netscience,
        # "astrophysics": astrophysics,
        "polblogs": polblogs,
        # "facebook": facebook,
    }

    for name, network in name_dictionary.items():
        network = get_main_component(network=network)
        print(f"--- NOW RUNNING: {name} ---")
        study = optuna.create_study(study_name=name, direction="minimize")
        target_dictionary = {
            "clustering": nx.algorithms.cluster.average_clustering(network),
            "assortativity": nx.algorithms.assortativity.degree_assortativity_coefficient(
                network
            ),
            "average_path": find_average_path(network),
        }
        study.optimize(
            lambda trial: objective(trial, network, 5, target_dictionary), n_trials=1000
        )
        study.best_params
        resulting_dictionary[name] = study.best_params
        joblib.dump(study, f"analysis/data/optimization/{name}_study.pkl")
    with open(
        f"analysis/data/optimization/best_optimization_results.pkl",
        "wb",
    ) as handle:
        pkl.dump(resulting_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)
