import networkx as nx
from networkx.generators.random_graphs import watts_strogatz_graph
import numpy as np
from ladon.visualize import plot_graph


class Network:
    def __init__(self):
        self.graph = watts_strogatz_graph(100, 4, 0.1)


plot_graph(Network().graph)
