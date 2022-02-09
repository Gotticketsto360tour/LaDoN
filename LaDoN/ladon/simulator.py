from matplotlib.pyplot import title, xlabel, xlim, ylabel, ylim
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
# It might make sense to have, say, 10 runs within one
# file for storing data, visualized "as one"

# NOTE:
# When threshold is low, randomness increases polarization,
# and when threshold is high, randomness decreases polarization

# TODO:
# 1. Run optimization to see more robust values
# 2. Make functions for recording data
# 3. Make first visualizations of what is happening
# 4. Schedule meeting with Paul regarding the model
# 5. Opinion as a function of initial opinion is really nice

dictionary = {
    "THRESHOLD": 1.5150662773714703,
    "N_TARGET": 1589,
    "RANDOMNESS": 0.1448096424703561,
    "N_TIMESTEPS": 1589 * 3,
    "POSITIVE_LEARNING_RATE": 0.44084799100234867,
    "NEGATIVE_LEARNING_RATE": 0.36968366749403275,
    "STOP_AT_TARGET": True,
}

dictionary = {
    "THRESHOLD": 1.2,
    "N_TARGET": 1000,
    "RANDOMNESS": 0.5,
    "N_TIMESTEPS": 10,
    "POSITIVE_LEARNING_RATE": 0.2,
    "NEGATIVE_LEARNING_RATE": 0.3,
    "STOP_AT_TARGET": True,
}

[0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
8 * 5 * 5 * 5

my_network = Network(dictionary=dictionary)

my_network.run_simulation()

numbers = my_network.get_agent_numbers()

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

sns.regplot(numbers, degrees)
sns.regplot(numbers, centralities)
sns.regplot(numbers, opinions)

sns.regplot(
    centralities, np.array([abs(x) for x in opinions]), scatter_kws={"alpha": 0.7}
).set(
    title="Absolute value of opinions as a function of Betweeness Centrality",
    xlabel="Betweeness Centrality",
    ylabel="Absolute value of Opinion",
    ylim=(-0.02, 1.02),
)

initial_opinions = my_network.get_initial_opinion_distribution()
np.corrcoef(initial_opinions, opinions)
sns.regplot(initial_opinions, opinions, scatter_kws={"alpha": 0.7}).set(
    title="Opinion as a function of initial opinion",
    xlabel="Initial opinion",
    ylabel="Opinion",
)

sns.scatterplot(x=degrees, y=initial_opinions, alpha=0.7).set(
    title="Initial opinion as a function of Degree",
    xlabel="Degree",
    ylabel="Initial Opinion",
)
sns.scatterplot(x=degrees, y=opinions, alpha=0.7).set(
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

np.median(np.array([abs(x) for x in opinions]))

plotting.set(xlim=(-10, 10))

nx.algorithms.cluster.average_clustering(my_network.graph)

nx.algorithms.assortativity.degree_assortativity_coefficient(my_network.graph)

plot_graph(my_network, plot_type="agent_type")

plot_graph(my_network, plot_type="community")

g = nx.read_gml(path="analysis/data/netscience/netscience.gml")

# g = nx.karate_club_graph()

nx.algorithms.cluster.average_clustering(g)
nx.algorithms.assortativity.degree_assortativity_coefficient(g)
degrees_g = [x[1] for x in list(g.degree())]
sns.histplot(degrees_g, stat="percent")

from netrd.distance import DegreeDivergence

distance_algorithm = DegreeDivergence()
distance_algorithm.dist(my_network.graph, g)
