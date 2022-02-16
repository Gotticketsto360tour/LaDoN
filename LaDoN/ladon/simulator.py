from matplotlib.pyplot import title, xlabel, xlim, ylabel, ylim
from visualize import plot_graph
from network import Network, NoOpinionNetwork
import networkx as nx
import numpy as np
import seaborn as sns
import scipy
import random

random.seed(10)

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
# In stable networks the hub is often composed of
# old, high degree nodes. These will have high betweeness
# centrality as they connect to so many nodes.
# Because centrality is so correlated with age,
# central nodes will often be nodes that have been
# under the influence of the conditions of the network
# for a longer time. If the network tends towards stability,
# the absolute value of the opinion of central nodes will tend towards 0.
# As the network becomes more destabilized, this correlation
# dissapears and even reverses.

# Central nodes are very often high degree nodes.
# The likelihood of getting a high degree becomes much
# greater, if your opinion is as close to neutral as possible.
# In these cases, you can cater to both sides of the spectrum,
# and get higher degrees.
# Moreover, nodes with high degrees in stable society
# will by definition be the nodes exposed to the most opinions in the network.
# As the network is stable, most of these opinions will be within the
# bounded confidence interval. Therefore, the high degree nodes
# will tend towards the mean of the population when the population is stable.

# NOTE: The correlation between degree and centrality
# fades as the network destabilizes. This suggests that
# as the network becomes more polarized, some agents
# become important bridges in the network without having a
# degree.
# This should be checked further by finding the instances where
# polarization is high, degree is low, but centrality is high.

# NOTE: The correlation between centrality and polarization
# shifts sign when the network polarizes.

# TODO:
# Currently, I don't really use the fact
# that ties are severed to say anything.
# Especially problematic is that randomness,
# doesn't seem to affect any of the important results.
# This is to some extent interesting in
# its own right, but it might make more sense to
# focus the probability of severing ties.
# This will no doubt have interesting
# interactions with negative learning rate
# and threshold.

# What could make sense is to have this as
# the baseline model, and next model could have
# randomness at a fixed level (say 0.1) and then
# vary how much "defriending" there is

# NOTE:
# Taking stock of the current status of things here:
# 1. It seems like the model can generate networks
# quite well in networks where the cost associated
# with a tie is relatively high. This could be
# another argument for messing more with tie dissolution
# instead of degree of randomness.

# 2. The types of opinion distributions I have
# can be fitted really well with the model.
# However, the data seems relatively strange.
# The greatest amount of political consensus
# seems to be in Estonia and the least
# in Scandinavia. In other words, something
# weird is going on. One possible explanation
# could be that the range of possible political
# parties within the middle ranges are larger.
# That could be the explanation for Denmark,
# but that doesn't explain Norway and Sweden.
# An important thing to keep in mind is that
# these are real people on the streets.
# People in highly polarized places
# might report central positions to
# avoid backlash.

# 3. There are some very interesting ties to
# affective polarization. Some of the surveys
# especially pertaining to the US could point
# to high negative learning rates, which
# could polarize the country. NOTE: These should be
# explored further

# TODO:
# 1. Check up on data; can we include clustering in measure?
# 2. Try with a pre-specified network, without birth or death of agents
# 3. Include parameter for probability of tie dissolution.
# 4. How does number of edges evolve over time?


dictionary = {
    "THRESHOLD": 1.5150662773714703,
    "N_TARGET": 1589,
    "RANDOMNESS": 0.1448096424703561,
    "N_TIMESTEPS": 1589 * 3,
    "POSITIVE_LEARNING_RATE": 0.44084799100234867,
    "NEGATIVE_LEARNING_RATE": 0.36968366749403275,
    "STOP_AT_TARGET": True,
}

g = nx.read_gml(path="analysis/data/polbooks/polbooks.gml")

dictionary = {
    "THRESHOLD": 1.775823633928921,
    "N_TARGET": 105,
    "RANDOMNESS": 0.3480999193670037,
    "N_TIMESTEPS": 10,
    "POSITIVE_LEARNING_RATE": 0.13439617121418493,
    "NEGATIVE_LEARNING_RATE": 0.18960028610011792,
    "STOP_AT_TARGET": True,
}

my_network = Network(dictionary=dictionary)

my_network.run_simulation()

plot_graph(my_network, plot_type="agent_type")
