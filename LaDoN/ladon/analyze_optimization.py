from statistics import mean
from unittest import result
from click import ParamType
from network import Network, NoOpinionNetwork
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

pio.renderers.default = "notebook"

netscience = nx.read_gml(path="analysis/data/netscience/netscience.gml")

facebook = nx.read_edgelist(
    "analysis/data/facebook_combined.txt", create_using=nx.Graph(), nodetype=int
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

name_dictionary = {
    "karate": karate,
    "dolphin": dolphin,
    "polbooks": polbooks,
    "polblogs": polblogs,
    "netscience": netscience,
    # "facebook": facebook,
}


def run_single_simulation(dictionary, run, target, precomputed_target_metrics):
    random.seed(run)

    my_network = Network(dictionary=dictionary)
    my_network.run_simulation()
    clustering_diff = abs(
        nx.algorithms.cluster.average_clustering(my_network.graph)
        - precomputed_target_metrics.get("clustering")
    )
    assortativity_diff = abs(
        nx.algorithms.assortativity.degree_assortativity_coefficient(my_network.graph)
        - precomputed_target_metrics.get("assortativity")
    )
    network_avg_path = find_average_path(my_network.graph)
    target_avg_path = precomputed_target_metrics.get("average_path")

    average_path_diff = abs(
        find_average_path(my_network.graph) - find_average_path(target)
    ) / max([network_avg_path, target_avg_path])

    distance_algorithm = netrd.distance.DegreeDivergence()
    JSD = distance_algorithm.dist(my_network.graph, target)
    minimize_array = np.array(
        [clustering_diff, assortativity_diff, average_path_diff, JSD]
    )
    return minimize_array


def get_result_from_optimizing(target_string: str):
    study = joblib.load(f"analysis/data/optimization/{target_string}_study.pkl")
    parameters = study.best_params
    print(parameters)
    target = name_dictionary.get(target_string)
    N_TARGET = target.number_of_nodes()
    N_EDGES = target.number_of_edges()
    K = round(N_EDGES / N_TARGET)
    precomputed_target_metrics = {
        "clustering": nx.algorithms.cluster.average_clustering(target),
        "assortativity": nx.algorithms.assortativity.degree_assortativity_coefficient(
            target
        ),
        "average_path": find_average_path(target),
    }
    dictionary = {
        "THRESHOLD": parameters.get("threshold"),
        "N_TARGET": N_TARGET,
        "RANDOMNESS": parameters.get("randomness"),
        "N_TIMESTEPS": N_TARGET * 10,
        "POSITIVE_LEARNING_RATE": parameters.get("positive_learning_rate"),
        "NEGATIVE_LEARNING_RATE": parameters.get("negative_learning_rate"),
        "STOP_AT_TARGET": True,
        "P": 0.4,
        "K": K,
        "TIE_DISSOLUTION": parameters.get("tie_dissolution"),
        "RECORD": False,
    }

    results = [
        run_single_simulation(
            dictionary=dictionary,
            run=run,
            target=name_dictionary.get(target_string),
            precomputed_target_metrics=precomputed_target_metrics,
        )
        for run in range(10)
    ]
    return results


result_dictionary_1 = {
    name: mean([np.linalg.norm(i) for i in get_result_from_optimizing(name)])
    for name in name_dictionary
}
[i for i in get_result_from_optimizing("netscience")]
