from visualize import plot_graph, generate_network_plots
from network import Network
import networkx as nx
import numpy as np
import seaborn as sns
import random
import pandas as pd
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")


def make_network_by_seed(dictionary, run):
    random.seed(run)
    np.random.seed(run)
    network = Network(dictionary)
    # network.run_simulation()
    return network


dictionary = {
    "THRESHOLD": 0.8,
    "N_TARGET": 500,
    "RANDOMNESS": 0.2,
    "N_TIMESTEPS": 10000,
    "POSITIVE_LEARNING_RATE": 0.1,
    "NEGATIVE_LEARNING_RATE": 0.1,
    "P": 0.4,
    "K": 7,
    "TIE_DISSOLUTION": 0.8,
    "RECORD": True,
}

dictionary = {
    "THRESHOLD": 0.6,
    "N_TARGET": 500,
    "RANDOMNESS": 0.01,
    "N_TIMESTEPS": 10000,
    "POSITIVE_LEARNING_RATE": 0.10,
    "NEGATIVE_LEARNING_RATE": 0.01,
    "P": 0.5,
    "K": 7,
    "TIE_DISSOLUTION": 1,
    "RECORD": True,
}

my_network = make_network_by_seed(dictionary=dictionary, run=5)
my_network.run_simulation()

sns.lineplot(data=my_network.NEGATIVE_TIES_DISSOLUTED)
sns.lineplot(data=my_network.AVERAGE_CLUSTERING)
sns.lineplot(data=my_network.ASSORTATIVITY)
sns.lineplot(data=my_network.MEAN_ABSOLUTE_OPINIONS)
sns.lineplot(data=my_network.AVERAGE_PATH_LENGTH)
sns.lineplot(data=my_network.MEAN_DISTANCE)
sns.lineplot(data=my_network.SD_ABSOLUTE_OPINIONS)


plot_graph(my_network, plot_type="agent_type", save_path="MyHtmlPlot.html")

opinions = my_network.get_opinion_distribution()

plotting = sns.displot(
    data=opinions,
    stat="percent",
    common_norm=False,
    # binwidth=0.05,
    kde=True,
    height=8.27,
    aspect=11.7 / 8.27,
).set(xlabel=r"$O_F$")

plotting = sns.histplot(
    data=my_network.get_degree_distribution(),
    stat="percent",
    common_norm=False,
    # binwidth=0.05,
    kde=True,
    # height=8.27,
    # aspect=11.7 / 8.27,
    discrete=True,
).set(xlabel="Degree")

sns.histplot(
    data=my_network.get_opinion_distances_without_none(),
    stat="percent",
    common_norm=False,
    # binwidth=0.05,
    kde=True,
    # height=8.27,
    # aspect=11.7 / 8.27,
    # discrete=True,
).set(xlabel=r"$O_F$")

# THIS IS A PRETTY GREAT WAY OF SHOWING
# WHAT HAPPENS TO THE NETWORK OVER TIME

my_network = Network(dictionary=dictionary)

for _ in range(500):
    my_network.take_turn()

# my_network.run_simulation()

plot_graph(my_network, plot_type="agent_type")

dictionary = {
    "THRESHOLD": 0.8,
    "N_TARGET": 500,
    "RANDOMNESS": 0.1,
    "N_TIMESTEPS": 10000,
    "POSITIVE_LEARNING_RATE": 0.15,
    "NEGATIVE_LEARNING_RATE": 0.1,
    "P": 0.5,
    "K": 7,
    "TIE_DISSOLUTION": 1,
    "RECORD": False,
}
my_network = make_network_by_seed(dictionary, run=7)

generate_network_plots(
    my_network,
    plot_type="agent_type",
    # save_path="plots/networks/network_example",
    run=7,
)
