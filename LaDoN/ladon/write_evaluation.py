import pickle as pkl
import networkx as nx
import pandas as pd
from helpers import get_main_component, find_average_path
from optimize import make_network_by_seed
from optimize_no_opinion import make_no_opinion_network_by_seed
import joblib
from network import Network, NoOpinionNetwork
import netrd
import seaborn as sns
import numpy as np

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)


def get_norm(model, target_dictionary, network):
    clustering_diff = abs(
        nx.algorithms.cluster.average_clustering(model.graph)
        - (target_dictionary.get("clustering")[0])
    )
    assortativity_diff = abs(
        nx.algorithms.assortativity.degree_assortativity_coefficient(model.graph)
        - (target_dictionary.get("assortativity")[0])
    )
    network_avg_path = find_average_path(model.graph)

    average_path_diff = abs(
        network_avg_path - target_dictionary.get("average_path")[0]
    ) / max([network_avg_path, target_dictionary.get("average_path")[0]])

    distance_algorithm = netrd.distance.DegreeDivergence()
    JSD = distance_algorithm.dist(model.graph, network)
    minimize_array = np.array(
        [clustering_diff, assortativity_diff, average_path_diff, JSD]
    )
    norm = np.linalg.norm(minimize_array)
    return norm


def generate_network_dataframe():

    with open(
        "analysis/data/fb-pages-government/fb-pages-government.nodes", "rb+"
    ) as f:
        data = [str(node, "utf-8").strip().split(",")[-1] for node in f.readlines()[1:]]

    politicians = nx.Graph()

    politicians.add_nodes_from(data)

    with open(
        "analysis/data/fb-pages-government/fb-pages-government.edges", "rb+"
    ) as f:
        data = [str(node, "utf-8").strip().split(",") for node in f.readlines()]

    politicians.add_edges_from(data)

    netscience = nx.read_gml(path="analysis/data/netscience/netscience.gml")

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
        "netscience": netscience,
        "polblogs": polblogs,
        "politicians": politicians,
    }
    list_of_dictionaries = []
    for name, network in name_dictionary.items():
        network = get_main_component(network=network)
        target_dictionary = {
            "type": ["Target"],
            "clustering": [nx.algorithms.cluster.average_clustering(network)],
            "assortativity": [
                nx.algorithms.assortativity.degree_assortativity_coefficient(network)
            ],
            "average_path": [find_average_path(network)],
            "network": [name],
            "JSD": [0],
            "run": [0],
            "norm": [0],
        }
        list_of_dictionaries.append(target_dictionary)
        best_study_opinions = joblib.load(
            f"analysis/data/optimization/{name}_study.pkl"
        )

        best_parameters = best_study_opinions.best_params

        N_TARGET = network.number_of_nodes()
        N_EDGES = network.number_of_edges()
        K = round(N_EDGES / N_TARGET)

        dictionary = {
            "THRESHOLD": best_parameters.get("threshold"),
            "N_TARGET": N_TARGET,
            "RANDOMNESS": 0.1,
            "N_TIMESTEPS": N_TARGET * 20,
            "POSITIVE_LEARNING_RATE": best_parameters.get("positive_learning_rate"),
            "NEGATIVE_LEARNING_RATE": best_parameters.get("negative_learning_rate"),
            "P": 0.1,
            "K": 2 * K,
            "TIE_DISSOLUTION": best_parameters.get("tie_dissolution"),
            "RECORD": False,
        }

        best_study_opinion_networks = [
            make_network_by_seed(dictionary=dictionary, run=run) for run in range(5)
        ]

        distance_algorithm = netrd.distance.DegreeDivergence()

        opinion_dictionaries = [
            {
                "type": ["Opinion_Model"],
                "clustering": [nx.algorithms.cluster.average_clustering(model.graph)],
                "assortativity": [
                    nx.algorithms.assortativity.degree_assortativity_coefficient(
                        model.graph
                    )
                ],
                "average_path": [find_average_path(model.graph)],
                "network": [name],
                "JSD": [distance_algorithm.dist(model.graph, network)],
                "run": [run],
                "norm": [
                    get_norm(
                        model=model,
                        target_dictionary=target_dictionary,
                        network=network,
                    )
                ],
            }
            for run, model in enumerate(best_study_opinion_networks)
        ]

        list_of_dictionaries.extend(opinion_dictionaries)

        best_study_non_opinions = joblib.load(
            f"analysis/data/optimization/{name}_study_no_opinion.pkl"
        )

        best_parameters = best_study_non_opinions.best_params

        dictionary = {
            "N_TARGET": N_TARGET,
            "RANDOMNESS": best_parameters.get("randomness"),
            "N_TIMESTEPS": N_TARGET * 20,
            "P": 0.1,
            "K": 2 * K,
            "RECORD": False,
        }

        best_study_no_opinion_networks = [
            make_no_opinion_network_by_seed(dictionary=dictionary, run=run)
            for run in range(5)
        ]

        distance_algorithm = netrd.distance.DegreeDivergence()

        no_opinion_dictionaries = [
            {
                "type": ["No_Opinion_Model"],
                "clustering": [nx.algorithms.cluster.average_clustering(model.graph)],
                "assortativity": [
                    nx.algorithms.assortativity.degree_assortativity_coefficient(
                        model.graph
                    )
                ],
                "average_path": [find_average_path(model.graph)],
                "network": [name],
                "JSD": [distance_algorithm.dist(model.graph, network)],
                "run": [run],
                "norm": [
                    get_norm(
                        model=model,
                        target_dictionary=target_dictionary,
                        network=network,
                    )
                ],
            }
            for run, model in enumerate(best_study_no_opinion_networks)
        ]

        list_of_dictionaries.extend(no_opinion_dictionaries)
    # pd.concat([pd.DataFrame(data) for data in list_of_dictionaries])
    return list_of_dictionaries


# test = generate_network_dataframe()

# sns.barplot(data=test, x="network", y="clustering", hue="type")
# sns.barplot(data=test, x="network", y="average_path", hue="type")
# sns.barplot(data=test, x="network", y="assortativity", hue="type")
# sns.barplot(data=test, x="network", y="JSD", hue="type")

# sns.barplot(data=test.query("type != 'Target'"), x="network", y="norm", hue="type")

# test.groupby(["network", "type"]).agg(
#     clustering=("clustering", "mean"),
#     average_path=("average_path", "mean"),
#     assortativity=("assortativity", "mean"),
#     JSD=("JSD", "mean"),
# ).reset_index()

if __name__ == "__main__":
    data = generate_network_dataframe()
    with open(
        f"analysis/data/optimization/data_from_all_runs.pkl",
        "wb",
    ) as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
