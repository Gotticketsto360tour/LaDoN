import pickle as pkl
from typing import Dict, List
import networkx as nx
from ladon.helpers.helpers import (
    get_main_component,
    find_average_path,
    get_shortest_path_distribution,
    calculate_jsd_for_paths,
)
from ladon.helpers.optimize_helpers import (
    make_network_by_seed,
    make_no_opinion_network_by_seed,
    make_theoretical_network_by_seed,
    make_barabasi_network_by_seed,
)
import joblib
import netrd
import numpy as np
from ladon.config import NAME_DICTIONARY


def load_pickle(string: str) -> dict:
    with open(string, "rb") as input_file:
        file = pkl.load(input_file)
    return file


def get_mean(
    model,
    target_dictionary: Dict,
    target_path_distribution: np.array,
    network: nx.Graph,
) -> float:
    """Returns the mean of the vector of differences

    Args:
        model (Network or NoOpinionNetwork): Model to be evaluated
        target_dictionary (Dict): Dictionary containing precomputed values
        network (nx.Graph): Target network
        denominator (float): Denominator for normalizing the average path length

    Returns:
        float: Mean of the vector of differences
    """
    clustering_diff = abs(
        nx.algorithms.cluster.average_clustering(model.graph)
        - (target_dictionary.get("clustering")[0])
    )
    path_distribution = get_shortest_path_distribution(model.graph, 10000)
    JSD_path = calculate_jsd_for_paths(target_path_distribution, path_distribution)

    distance_algorithm = netrd.distance.DegreeDivergence()
    JSD_degree = distance_algorithm.dist(model.graph, network)
    minimize_array = np.array([clustering_diff, JSD_path, JSD_degree])
    mean = np.mean(minimize_array)
    return mean


def generate_network_dataframe(repeats: int) -> List[Dict]:
    """Generate the dataframe for evaluating the models

    Args:
        repeats (int): Integer giving how many repeats of each network needs to be simulated

    Returns:
        list: List of dictionary in json style
    """

    list_of_dictionaries = []
    types = ["", "barabasi", "no_opinion", "theoretical"]
    best_values_dict = {
        one_type: joblib.load(
            f"../analysis/data/optimization/best_optimization_results_{one_type}.pkl"
        )
        for one_type in types
    }
    for name, network in NAME_DICTIONARY.items():
        network = get_main_component(network=network)
        target_dictionary = {
            "type": ["Target"],
            "clustering": [nx.algorithms.cluster.average_clustering(network)],
            "average_path": [find_average_path(network)],
            "network": [name],
            "JSD_degree": [0],
            "JSD_paths": [0],
            "run": [0],
            "mean": [0],
        }
        list_of_dictionaries.append(target_dictionary)

        N_TARGET = network.number_of_nodes()
        N_EDGES = network.number_of_edges()
        K = round(N_EDGES / N_TARGET)

        target_path_distribution = get_shortest_path_distribution(network, 10000)

        dictionary = {
            "THRESHOLD": best_values_dict.get("").get(name).get("threshold"),
            "N_TARGET": N_TARGET,
            "RANDOMNESS": 0.1,
            "N_TIMESTEPS": N_TARGET * 10,
            "POSITIVE_LEARNING_RATE": best_values_dict.get("")
            .get(name)
            .get("positive_learning_rate"),
            "NEGATIVE_LEARNING_RATE": best_values_dict.get("")
            .get(name)
            .get("negative_learning_rate"),
            "P": 0.5,
            "K": 2 * K,
            "TIE_DISSOLUTION": best_values_dict.get("")
            .get(name)
            .get("tie_dissolution"),
            "RECORD": False,
        }

        best_study_opinion_networks = [
            make_network_by_seed(dictionary=dictionary, run=run)
            for run in range(repeats)
        ]

        distance_algorithm = netrd.distance.DegreeDivergence()

        opinion_dictionaries = [
            {
                "type": ["Opinion_Model"],
                "clustering": [nx.algorithms.cluster.average_clustering(model.graph)],
                "average_path": [find_average_path(model.graph)],
                "network": [name],
                "JSD_degree": [distance_algorithm.dist(model.graph, network)],
                "JSD_paths": [
                    calculate_jsd_for_paths(
                        get_shortest_path_distribution(model.graph, 10000),
                        target_path_distribution,
                    )
                ],
                "run": [run],
                "mean": [
                    get_mean(
                        model=model,
                        target_dictionary=target_dictionary,
                        network=network,
                        target_path_distribution=target_path_distribution,
                    )
                ],
            }
            for run, model in enumerate(best_study_opinion_networks)
        ]

        list_of_dictionaries.extend(opinion_dictionaries)

        dictionary = {
            "N_TARGET": N_TARGET,
            "RANDOMNESS": best_values_dict.get("no_opinion")
            .get(name)
            .get("randomness"),
            "N_TIMESTEPS": N_TARGET * 10,
            "P": 0.5,
            "K": 2 * K,
            "RECORD": False,
        }

        best_study_no_opinion_networks = [
            make_no_opinion_network_by_seed(dictionary=dictionary, run=run)
            for run in range(repeats)
        ]

        distance_algorithm = netrd.distance.DegreeDivergence()

        no_opinion_dictionaries = [
            {
                "type": ["No_Opinion_Model"],
                "clustering": [nx.algorithms.cluster.average_clustering(model.graph)],
                "average_path": [find_average_path(model.graph)],
                "network": [name],
                "JSD_degree": [distance_algorithm.dist(model.graph, network)],
                "JSD_paths": [
                    calculate_jsd_for_paths(
                        get_shortest_path_distribution(model.graph, 10000),
                        target_path_distribution,
                    )
                ],
                "run": [run],
                "mean": [
                    get_mean(
                        model=model,
                        target_dictionary=target_dictionary,
                        network=network,
                        target_path_distribution=target_path_distribution,
                    )
                ],
            }
            for run, model in enumerate(best_study_no_opinion_networks)
        ]

        list_of_dictionaries.extend(no_opinion_dictionaries)

        dictionary = {
            "N_TARGET": N_TARGET,
            # "RANDOMNESS": best_parameters.get("randomness"),
            "N_TIMESTEPS": N_TARGET * 10,
            "P": best_values_dict.get("theoretical").get(name).get("P"),
            "K": 2 * K,
            "RECORD": False,
        }

        best_study_theoretical_networks = [
            make_theoretical_network_by_seed(dictionary=dictionary, run=run)
            for run in range(repeats)
        ]

        distance_algorithm = netrd.distance.DegreeDivergence()

        theoretical_dictionaries = [
            {
                "type": ["Small-world Network"],
                "clustering": [nx.algorithms.cluster.average_clustering(model.graph)],
                "average_path": [find_average_path(model.graph)],
                "network": [name],
                "JSD_degree": [distance_algorithm.dist(model.graph, network)],
                "JSD_paths": [
                    calculate_jsd_for_paths(
                        get_shortest_path_distribution(model.graph, 10000),
                        target_path_distribution,
                    )
                ],
                "run": [run],
                "mean": [
                    get_mean(
                        model=model,
                        target_dictionary=target_dictionary,
                        network=network,
                        target_path_distribution=target_path_distribution,
                    )
                ],
            }
            for run, model in enumerate(best_study_theoretical_networks)
        ]

        list_of_dictionaries.extend(theoretical_dictionaries)

        dictionary = {
            "N_TARGET": N_TARGET,
            # "RANDOMNESS": best_parameters.get("randomness"),
            "N_TIMESTEPS": N_TARGET * 10,
            # "P": best_parameters.get("randomness"),
            "K": best_values_dict.get("barabasi").get(name).get("K"),
            "RECORD": False,
        }

        best_study_barabasi_networks = [
            make_barabasi_network_by_seed(dictionary=dictionary, run=run + 3)
            for run in range(repeats)
        ]

        distance_algorithm = netrd.distance.DegreeDivergence()

        barabasi_dictionaries = [
            {
                "type": ["Scale-free Network"],
                "clustering": [nx.algorithms.cluster.average_clustering(model.graph)],
                "average_path": [find_average_path(model.graph)],
                "network": [name],
                "JSD_degree": [distance_algorithm.dist(model.graph, network)],
                "JSD_paths": [
                    calculate_jsd_for_paths(
                        get_shortest_path_distribution(model.graph, 10000),
                        target_path_distribution,
                    )
                ],
                "run": [run],
                "mean": [
                    get_mean(
                        model=model,
                        target_dictionary=target_dictionary,
                        network=network,
                        target_path_distribution=target_path_distribution,
                    )
                ],
            }
            for run, model in enumerate(best_study_barabasi_networks)
        ]

        list_of_dictionaries.extend(barabasi_dictionaries)
    return list_of_dictionaries


if __name__ == "__main__":
    data = generate_network_dataframe(repeats=20)
    with open(
        f"../analysis/data/optimization/data_from_all_runs.pkl",
        "wb",
    ) as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
