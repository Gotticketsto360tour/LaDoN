from statistics import mean
from network import Network
import networkx as nx
import numpy as np
import optuna
import netrd
from tqdm import tqdm
from helpers import find_average_path


def objective(trial, target, repeats):

    threshold = trial.suggest_float("threshold", 0, 2)
    randomness = trial.suggest_float("randomness", 0.1, 1)
    positive_learning_rate = trial.suggest_float("positive_learning_rate", 0, 0.5)
    negative_learning_rate = trial.suggest_float("negative_learning_rate", 0, 0.5)

    N_TARGET = target.number_of_nodes()
    list_of_norms = []
    for run in range(repeats):

        dictionary = {
            "THRESHOLD": threshold,
            "N_TARGET": N_TARGET,
            "RANDOMNESS": randomness,
            "N_TIMESTEPS": N_TARGET * 3,
            "POSITIVE_LEARNING_RATE": positive_learning_rate,
            "NEGATIVE_LEARNING_RATE": negative_learning_rate,
            "STOP_AT_TARGET": True,
        }

        my_network = Network(dictionary=dictionary)
        my_network.run_simulation()

        clustering_diff = abs(
            nx.algorithms.cluster.average_clustering(my_network.graph)
            - nx.algorithms.cluster.average_clustering(target)
        )
        assortativity_diff = abs(
            nx.algorithms.assortativity.degree_assortativity_coefficient(
                my_network.graph
            )
            - nx.algorithms.assortativity.degree_assortativity_coefficient(target)
        )
        network_avg_path = find_average_path(my_network.graph)
        target_avg_path = find_average_path(target)

        average_path_diff = abs(
            find_average_path(my_network.graph) - find_average_path(target)
        ) / max([network_avg_path, target_avg_path])

        distance_algorithm = netrd.distance.DegreeDivergence()
        JSD = np.sqrt(distance_algorithm.dist(my_network.graph, target))
        minimize_array = np.array(
            [clustering_diff, assortativity_diff, average_path_diff, JSD]
        )
        norm = np.linalg.norm(minimize_array)
        list_of_norms.append(norm)

    return mean(list_of_norms)


target = nx.read_gml(path="analysis/data/netscience/netscience.gml")

# target = nx.read_edgelist(
#    "analysis/data/facebook_combined.txt", create_using=nx.Graph(), nodetype=int
# )

# target = nx.karate_club_graph()

# target = nx.read_gml(
#    path="analysis/data/polblogs/polblogs.gml",
# )
# target = nx.Graph(target.to_undirected())

# target = nx.read_gml(
#     path="analysis/data/polbooks/polbooks.gml",
# )

# target = nx.read_gml(path="analysis/data/netscience/netscience.gml")


study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, target, 1), n_trials=100, n_jobs=1)
study.best_params
