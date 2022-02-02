from visualize import plot_graph
from network import Network
import networkx as nx
import numpy as np
import seaborn as sns

# TODO:
# Have agents interact with a certain level of noise, centered around 0.

# Specification:
# This could make relations asymmetric. It could be modelled as


# NOTE:
# One of the reasons why we are not seeing the tails of the distribution match
# is because of two ceiling effects, which can be corrected by having upper and
# lower limits for opinions in the model.

dictionary = {
    "THRESHOLD": 0.7,
    "N_TARGET": 300,
    "RANDOMNESS": 0.3,
    "N_TIMESTEPS": 8000,
    "POSITIVE_LEARNING_RATE": 0.4,
    "NEGATIVE_LEARNING_RATE": 0.05,
}

my_network = Network(dictionary=dictionary)

my_network.run_simulation()

plot_graph(my_network, plot_type="agent_type")

degrees = my_network.get_degree_distribution()

opinions = my_network.get_opinion_distribution()
initial_opinions = my_network.get_initial_opinion_distribution()
sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.scatterplot(x=initial_opinions, y=degrees)
sns.scatterplot(x=degrees, y=opinions)

plotting = sns.histplot(opinions, stat="percent", binwidth=0.2, kde=True)
plotting.set(xlim=(-10, 10))

nx.algorithms.cluster.average_clustering(my_network.graph)

nx.algorithms.assortativity.degree_assortativity_coefficient(my_network.graph)

plot_graph(my_network, plot_type="agent_type")

plot_graph(my_network, plot_type="community")
