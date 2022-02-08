from turtle import distance

from matplotlib.pyplot import title, xlabel, ylabel
from visualize import plot_graph
from network import Network
import networkx as nx
import numpy as np
import seaborn as sns
import scipy

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")

# TODO:
# Have agents interact with a certain level of noise, centered around 0.

# Specification:
# This could make relations asymmetric. It could be modelled as

# NOTE:
# There seems to be a deep correlation between
# centrality and extremeness of opinions.
# The fringes in opinions come from the fringes of the network

# NOTE:
# Larger xenophobia rates produce smaller distances
# to neighbors opinions

# NOTE: The correlation between centrality and polarization
# shifts sign when the network polarizes.

# NOTE:
# Something crazy is happening around 0.7

# dictionary = {
#     "THRESHOLD": 0.7,
#     "N_TARGET": 1589,
#     "RANDOMNESS": 0.5,
#     "N_TIMESTEPS": 1589 * 3,
#     "POSITIVE_LEARNING_RATE": 0.2,
#     "NEGATIVE_LEARNING_RATE": 0.05,
#     "STOP_AT_TARGET": True,
# }

dictionary = {
    "THRESHOLD": 0.9,
    "N_TARGET": 1589,
    "RANDOMNESS": 0.2,
    "N_TIMESTEPS": 1589 * 3,
    "POSITIVE_LEARNING_RATE": 0.5,
    "NEGATIVE_LEARNING_RATE": 0.1,
    "STOP_AT_TARGET": False,
}

my_network = Network(dictionary=dictionary)

my_network.run_simulation()

distances = np.array(
    [distance for distance in my_network.get_opinion_distances() if distance]
)
plotting = sns.histplot(distances, stat="percent", binwidth=0.02, kde=True).set(
    title="Average distance to neighbor's opinion", xlabel="Average distance"
)
plot_graph(my_network, plot_type="agent_type")

centralities = np.array(
    list(nx.algorithms.centrality.betweenness_centrality(my_network.graph).values())
)
degrees = my_network.get_degree_distribution()

opinions = my_network.get_opinion_distribution()

sns.regplot(centralities, np.array([abs(x) for x in opinions]))

initial_opinions = my_network.get_initial_opinion_distribution()
sns.scatterplot(x=degrees, y=initial_opinions).set(
    title="Initial opinion as a function of Degree",
    xlabel="Degree",
    ylabel="Initial Opinion",
)
sns.scatterplot(x=degrees, y=opinions).set(
    title="Opinion as a function of Degree", xlabel="Degree", ylabel="Opinion"
)
sns.regplot(x=degrees, y=centralities).set(
    title="Centrality as a function of Degree",
    xlabel="Degree",
    ylabel="Betweeness Centrality",
)
sns.histplot(degrees, stat="percent", binwidth=1, discrete=True, kde=True).set(
    title="Degree Distribution", xlabel="Degree"
)
plotting = sns.histplot(opinions, stat="percent", binwidth=0.05, kde=True).set(
    title="Opinion Distribution", xlabel="Opinion"
)
plotting.set(xlim=(-10, 10))

nx.algorithms.cluster.average_clustering(my_network.graph)

nx.algorithms.assortativity.degree_assortativity_coefficient(my_network.graph)

plot_graph(my_network, plot_type="agent_type")

plot_graph(my_network, plot_type="community")

g = nx.read_gml(path="analysis/data/netscience/netscience.gml")

g = nx.karate_club_graph()

nx.algorithms.cluster.average_clustering(g)
nx.algorithms.assortativity.degree_assortativity_coefficient(g)
degrees_g = [x[1] for x in list(g.degree())]
sns.histplot(degrees_g, stat="percent")

from netrd.distance import DegreeDivergence

distance_algorithm = DegreeDivergence()
distance_algorithm.dist(my_network.graph, nx.karate_club_graph())
