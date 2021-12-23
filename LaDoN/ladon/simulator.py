from ladon.visualize import plot_graph
from ladon.network import Network

# NOTE:
# A measure could be:
# What percentage of a community consists of only one type of agent?

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
