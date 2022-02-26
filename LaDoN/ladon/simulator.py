from visualize import plot_graph
from network import Network, NoOpinionNetwork
import networkx as nx
import numpy as np
import seaborn as sns
import scipy
import random

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
# 1. Write up sections regarding the missing
# tie dissolution in previous work (relate)
# to the article from Smaldino
# 2. Write-up examples of the importance of
# co-evolution instead of "bi-evolution"
# 3. Include the physics networks in modelling
# 4. Start simulating for all different conditions


# NOTE:
# If randomness of tie-dissolution is included, I can actually test
# whether tie dissolution is the reason why polarization doesn't happen

# NOTE:
# For low thresholds, high randomness, low negative_learning_rate
# this generates classic connected caveman graphs.

# NOTE:
# There does exist parameter combinations,
# that seem reasonable,
# which produces polarization sometimes
# and diversity at other times. Very nice!

# NOTE:
# When tie-dissolution is 0, there is no correlation between distance in opinion space and distance in the network.
# As tie-dissolution rises, the correspondance between average path length and absolute opinion becomes stronger.
# This makes sense as tie-dissolution is the only way for agents to change their connections in relation to their opinion.
# When tie-dissolution is 0, polarization of opinion does not lead to polarization of the network or vice versa.
# These conditions resemble the classic models in the literature. The results of polarization often happens directly because
# the agents do not reflect their opinions in their connections.
# As soon as tie-dissolution is greater than 0, agents will over time have less and less distance to their neighbors.
# This is because assimilation will draw agents close and distant agents will gradually be discarded.
# As a result of this process, communities evolve where opinions are similar.
# Because of triadic closure, these neighborhoods become more connected and more similar, creating a self-reinforcing loop.
# In other words, tie-dissolution results in homophily.
# As a result, distance in network space and distance in opinion space becomes more and more alike.
# When the network polarizes in opinion space, the result will be a polarization in the network, where the two poles push each other further and further away from each other, resulting in bi-polarization.
# As only a few connections remain between the two parts of the graph, the average path length increases substantially.
# In these circumstances, random connections that connect across the network will often be severed, and will therefore be unlikely.
# When the network is best characterized by consensus, the network consists of one primary core. Long-range connections are more likely to be sustained, creating a "small-world" effect.

# NOTE:
# Make the visualization of networks evolving over time - really sells the point.

# NOTE:
# Randomness doesn't affect qualitative results - it affects the speed of convergence.

# TODO:
# 1. Find better networks to match as targets
# 2. Read and understand the polarization of politics paper
# 3. Make connectedpapers and read for papers connected to "cooperation tie dissolution" paper
# 4. Run full sweep of parameters
# 5. Write up an abstract of the idea for friday

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
    "THRESHOLD": 0.8,
    "N_TARGET": 500,
    "RANDOMNESS": 0.1,
    "N_TIMESTEPS": 10000,
    "POSITIVE_LEARNING_RATE": 0.15,
    "NEGATIVE_LEARNING_RATE": 0.05,
    "P": 0.5,
    "K": 7,
    "TIE_DISSOLUTION": 0.9,
    "RECORD": True,
}

my_network = Network(dictionary=dictionary)

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

my_network.run_simulation()

plot_graph(my_network, plot_type="agent_type")
