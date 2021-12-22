from ladon.visualize import plot_graph
from ladon.network import Network


my_network = Network(graph="smallworld")

plot_graph(my_network, plot_type="agent_type")

my_network.run_simulation()

plot_graph(my_network, plot_type="agent_type")

plot_graph(my_network, plot_type="community")
