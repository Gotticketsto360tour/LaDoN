import networkx as nx
from networkx.generators.random_graphs import (
    watts_strogatz_graph,
    barabasi_albert_graph,
)
import numpy as np
from ladon.visualize import plot_graph
from ladon.agent import Agent
from ladon.config import AGENT_CONFIG


class Network:
    def __init__(self, graph: str = "smallworld"):
        N_AGENTS = 100

        if graph == "smallworld":
            self.graph = watts_strogatz_graph(100, 4, 0.1)
        elif graph == "scalefree":
            self.graph = barabasi_albert_graph(100, 4)

        self.agents = {i: Agent(AGENT_CONFIG) for i in N_AGENTS}


plot_graph(Network(graph="scalefree").graph)
