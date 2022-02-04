from network import Network
import networkx as nx
import numpy as np
import optuna
import netrd


def objective(trial, target):

    threshold = trial.suggest_float("threshold", 0, 1)
    randomness = trial.suggest_float("randomness", 0, 1)
    positive_learning_rate = trial.suggest_float("positive_learning_rate", 0, 1)
    negative_learning_rate = trial.suggest_float("negative_learning_rate", 0, 1)

    N_TARGET = target.number_of_nodes()

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
        nx.algorithms.assortativity.degree_assortativity_coefficient(my_network.graph)
        - nx.algorithms.assortativity.degree_assortativity_coefficient(target)
    )
    distance_algorithm = netrd.distance.DegreeDivergence()
    JSD = np.sqrt(distance_algorithm.dist(my_network.graph, target))
    minimize_array = np.array([clustering_diff, assortativity_diff, JSD])
    norm = np.linalg.norm(minimize_array)

    return norm


target = nx.read_gml(path="analysis/data/netscience/netscience.gml")

study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, target), n_trials=100)
