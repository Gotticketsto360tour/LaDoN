from visualize import plot_graph
from network import Network
import networkx
import numpy as np

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

main_component = networkx.node_connected_component(my_network.graph, 78)

[
    my_network.agents.get(agent).inner_vector
    for agent in my_network.agents
    if agent in main_component
]

np.array(
    [
        my_network.agents.get(agent).outer_mean
        for agent in my_network.agents
        if agent in main_component
    ]
).mean()

np.array(
    [
        my_network.agents.get(agent).outer_mean
        for agent in my_network.agents
        if agent not in main_component
    ]
).mean()

plot_graph(my_network, plot_type="agent_type")

plot_graph(my_network, plot_type="community")
