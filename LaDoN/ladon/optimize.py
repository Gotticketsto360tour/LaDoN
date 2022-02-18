from statistics import mean
from unittest import result
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


def run_single_simulation(dictionary, run, target, target_dictionary):
    random.seed(run)

    my_network = Network(dictionary=dictionary)
    my_network.run_simulation()

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
    JSD = np.sqrt(distance_algorithm.dist(my_network.graph, target))
    minimize_array = np.array(
        [clustering_diff, assortativity_diff, average_path_diff, JSD]
    )
    norm = np.linalg.norm(minimize_array)
    return norm


def objective(trial, target, repeats):
    target_dictionary = {
        "clustering": nx.algorithms.cluster.average_clustering(target),
        "assortativity": nx.algorithms.assortativity.degree_assortativity_coefficient(
            target
        ),
        "average_path": find_average_path(target),
    }
    N_TARGET = target.number_of_nodes()
    N_EDGES = target.number_of_edges()
    K = round(N_EDGES / N_TARGET)
    threshold = trial.suggest_float("threshold", 0.5, 2)
    randomness = trial.suggest_float("randomness", 0.1, 1)
    positive_learning_rate = trial.suggest_float("positive_learning_rate", 0, 0.5)
    negative_learning_rate = trial.suggest_float("negative_learning_rate", 0, 0.5)
    tie_dissolution = trial.suggest_float("tie_dissolution", 0.1, 1)

    dictionary = {
        "THRESHOLD": threshold,
        "N_TARGET": N_TARGET,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": N_TARGET * 10,
        "POSITIVE_LEARNING_RATE": positive_learning_rate,
        "NEGATIVE_LEARNING_RATE": negative_learning_rate,
        "P": 0.4,
        "K": K,
        "TIE_DISSOLUTION": tie_dissolution,
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
        "polblogs": polblogs,
        "netscience": netscience,
        "facebook": facebook,
    }
    for name, network in name_dictionary.items():
        print(f"--- NOW RUNNING: {name} ---")
        study = optuna.create_study(study_name=name, direction="minimize")
        study.optimize(
            lambda trial: objective(trial, network, 5), n_trials=1000, n_jobs=-1
        )
        study.best_params
        resulting_dictionary[name] = study.best_params
        joblib.dump(study, f"analysis/data/optimization/{name}_study.pkl")
    with open(
        f"analysis/data/optimization/best_optimization_results.pkl",
        "wb",
    ) as handle:
        pkl.dump(resulting_dictionary, handle, protocol=pkl.HIGHEST_PROTOCOL)
