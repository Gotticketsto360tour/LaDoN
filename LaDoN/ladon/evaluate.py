import pickle as pkl
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import optuna
import joblib
import networkx as nx
from helpers import find_average_path, get_main_component
from config import NAME_DICTIONARY

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)


def change_labels(string: str):
    if string == "Opinion_Model":
        return "Co-evolutionary model"
    if string == "No_Opinion_Model":
        return "Network Formation model"
    else:
        return "Empirical network"


def change_network_labels(string: str):
    translation = {
        "dolphin": "Dolphins",
        "karate": "Karate Club",
        "netscience": "Citation Network",
        "polblogs": "Political Blogs",
        "polbooks": "Political Books",
        "politicians": "Politicians",
        "tvshows": "TV Shows",
    }
    return translation.get(string)


with open(
    f"analysis/data/optimization/data_from_all_runs.pkl",
    "rb",
) as handle:
    data = pkl.load(handle)

data = pd.concat([pd.DataFrame(x) for x in data])

data["type"] = data["type"].apply(lambda x: change_labels(x))
data["network"] = data["network"].apply(lambda x: change_network_labels(x))
g = sns.barplot(
    data=data,
    y="network",
    x="clustering",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "TV Shows",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
    capsize=0.07,
    hue_order=["Co-evolutionary model", "Network Formation model", "Empirical network"],
)
g.set(xlabel="Average Clustering Coefficient", ylabel="")
plt.legend(
    title="Network",
)
plt.savefig(
    "plots/overall/Model_Evaluation_Average_Clustering.png",
    dpi=300,
    bbox_inches="tight",
)


g = sns.barplot(
    data=data,
    y="network",
    x="average_path",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "TV Shows",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
    capsize=0.07,
    hue_order=["Co-evolutionary model", "Network Formation model", "Empirical network"],
)
g.set(xlabel=r"$APL$", ylabel="")
plt.legend(title="Network", loc="upper right")
plt.savefig("plots/overall/Model_Evaluation_APL.png", dpi=300, bbox_inches="tight")

g = sns.barplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="JSD",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "TV Shows",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
    capsize=0.07,
)
g.set(xlabel=r"$JSD$", ylabel="")
plt.legend(
    title="Network",
)
plt.savefig("plots/overall/Model_Evaluation_JSD.png", dpi=300, bbox_inches="tight")


g = sns.barplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="mean",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "TV Shows",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
    capsize=0.07,
)
g.set(xlabel="Mean Difference", ylabel="")
plt.legend(title="Network", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
plt.savefig("plots/overall/Model_Evaluation.png", dpi=300, bbox_inches="tight")

pio.renderers.default = "notebook"


def get_important_parameters(network):
    study = joblib.load(f"analysis/data/optimization/{network}_study.pkl")
    dict_of_values = dict(optuna.importance.get_param_importances(study))
    dict_of_values["network"] = network
    return dict_of_values


def rename_value_names(string):
    old_names = [
        "randomness",
        "tie_dissolution",
        "threshold",
        "negative_learning_rate",
        "positive_learning_rate",
    ]
    new_names = [r"$R$", r"$P(D)$", r"$T$", r"$\beta$", r"$\alpha$"]
    translation = {old_names[i]: new_names[i] for i in range(len(old_names))}
    return translation.get(string)


list_of_parameters = [get_important_parameters(network) for network in NAME_DICTIONARY]

importance_df = pd.DataFrame(list_of_parameters)

importance_df = importance_df.melt(id_vars="network")

importance_df["variable"] = importance_df["variable"].apply(
    lambda x: rename_value_names(x)
)
importance_df["network"] = importance_df["network"].apply(
    lambda x: change_network_labels(x)
)

sns.barplot(
    data=importance_df,
    x="variable",
    y="value",
    hue="network",
    palette="muted",
    hue_order=[
        "Karate Club",
        "Dolphins",
        "Citation Network",
        "TV Shows",
        "Political Books",
        "Political Blogs",
        "Politicians",
    ],
).set(
    ylabel="Parameter Importance",
    xlabel="",
)
plt.legend(title=r"Network")
plt.savefig(
    "plots/overall/Parameter_Importance.png",
    dpi=300,
    bbox_inches="tight",
)


def get_important_network_characteristics(name: str, network: nx.Graph()):
    network = get_main_component(network=network)
    degrees = np.array([degree[1] for degree in network.degree()])

    return {
        "name": name,
        "nodes": len(degrees),
        "edges": len(network.edges()),
        "mean_degree": degrees.mean(),
        "sd_degree": degrees.std(),
        "clustering coefficient": nx.average_clustering(network),
        "average_path_length": find_average_path(network),
    }


pd.DataFrame(
    [
        get_important_network_characteristics(name, network)
        for name, network in NAME_DICTIONARY.items()
    ]
)


fig = optuna.visualization.plot_param_importances(study)
fig.show()

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_edf(study)
fig.show()

fig = optuna.visualization.plot_slice(study)
fig.show()
