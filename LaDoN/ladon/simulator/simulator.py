from ladon.visualize import plot_graph, generate_network_plots
from ladon.classes.network import Network
from ladon.helpers.optimize_helpers import make_network_by_seed
import networkx as nx
import numpy as np
import seaborn as sns
import random
import pandas as pd
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")


# dictionary = {
#     "THRESHOLD": 0.8,
#     "N_TARGET": 500,
#     "RANDOMNESS": 0.1,
#     "N_TIMESTEPS": 10000,
#     "POSITIVE_LEARNING_RATE": 0.2,
#     "NEGATIVE_LEARNING_RATE": 0,
#     "P": 0.5,
#     "K": 7,
#     "TIE_DISSOLUTION": 1,
#     "RECORD": True,
# }

# my_network = make_network_by_seed(dictionary=dictionary, run=1, run_simulation=False)

# my_network.run_simulation()

# sns.lineplot(data=my_network.NEGATIVE_TIES_DISSOLUTED)
# sns.lineplot(data=my_network.AVERAGE_CLUSTERING)
# sns.lineplot(data=my_network.ASSORTATIVITY)
# sns.lineplot(data=my_network.MEAN_ABSOLUTE_OPINIONS)
# sns.lineplot(data=my_network.AVERAGE_PATH_LENGTH)
# sns.lineplot(data=my_network.MEAN_DISTANCE)
# sns.lineplot(data=my_network.SD_ABSOLUTE_OPINIONS)


# plot_graph(my_network, plot_type="agent_type")

# opinions = my_network.get_opinion_distribution()

# plotting = sns.displot(
#     data=opinions,
#     stat="percent",
#     common_norm=False,
#     # binwidth=0.05,
#     kde=True,
#     height=8.27,
#     aspect=11.7 / 8.27,
# ).set(xlabel=r"$O_F$")

# plotting = sns.histplot(
#     data=my_network.get_degree_distribution(),
#     stat="percent",
#     common_norm=False,
#     # binwidth=0.05,
#     kde=True,
#     # height=8.27,
#     # aspect=11.7 / 8.27,
#     discrete=True,
# ).set(xlabel="Degree")

# # THIS IS A PRETTY GREAT WAY OF SHOWING
# # WHAT HAPPENS TO THE NETWORK OVER TIME

# my_network = Network(dictionary=dictionary)

# for _ in range(500):
#     my_network.take_turn()

# # my_network.run_simulation()

# plot_graph(my_network, plot_type="agent_type")

# data_specific_random = data[
#     (data["threshold"] == 0.9)
#     & (data["positive_learning_rate"] == 0.2)
#     & (data["negative_learning_rate"] == 0.2)
#     & (data["tie_dissolution"] == 1)
#     & (data["run"] == 8)
# ]

for randomness in [0.1, 0.3, 0.5]:
    # randomness = 0.1
    dictionary = {
        "THRESHOLD": 0.9,
        "N_TARGET": 1000,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": 10000,
        "POSITIVE_LEARNING_RATE": 0.2,
        "NEGATIVE_LEARNING_RATE": 0.2,
        "P": 0.5,
        "K": 7,
        "TIE_DISSOLUTION": 1,
        "RECORD": False,
    }
    my_network = make_network_by_seed(dictionary, run=8, run_simulation=False)
    generate_network_plots(
        my_network,
        plot_type="agent_type",
        save_path=f"../plots/networks/network_example_R{randomness}",
        run=8,
    )
