from visualize import plot_graph
from network import Network
import networkx
import numpy as np

# NOTE:
# A measure could be:
# What percentage of a community consists of only one type of agent?

# NOTE:
# Shouldn't one of the measures be distribution of distances to
# each neighbor?

# NOTE:
# For visualization purposes, wouldn't it
# be much nicer to let everyone have either 1, 2 or
# 3 values instead of 5? Why 5?

# TODO:
# Rethink how agents connect based on an agents expectations.
# They need memory as well as a counter for how many times
# they have seen the other agents.
# Next step is also to combine inner and outer vectors

my_network = Network(graph="smallworld")

plot_graph(my_network, plot_type="agent_type")

my_network.run_simulation()

plot_graph(my_network, plot_type="agent_type")

plot_graph(my_network, plot_type="community")
