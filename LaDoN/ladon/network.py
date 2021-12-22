from typing import Dict
import networkx as nx
from networkx.generators.random_graphs import (
    watts_strogatz_graph,
    barabasi_albert_graph,
)
import numpy as np
from ladon.agent import Agent
from ladon.config import CONFIGS
from random import sample
from ladon.helpers import compare_vectors


class Network:
    def __init__(self, graph: str = "smallworld"):
        self.N_AGENTS = 100
        self.N_GROUPS = len(CONFIGS)
        self.agents = {}

        if graph == "smallworld":
            self.graph = watts_strogatz_graph(100, 4, 0.1)
        elif graph == "scalefree":
            self.graph = barabasi_albert_graph(100, 4)

        self.initialize_network(CONFIGS)

    def initialize_network(self, CONFIGS: Dict):
        for agent in range(self.N_AGENTS):
            agent_type = sample(list(CONFIGS.keys()), 1)[0]
            self.agents[agent] = Agent(CONFIGS[agent_type])

    def take_turn(self):
        sampled_agent = sample(self.graph.nodes, 1)[0]
        sampled_agent_neighbors = list(self.graph.neighbors(sampled_agent))
        for neighbor in sampled_agent_neighbors:
            distance = compare_vectors(
                self.agents.get(sampled_agent),
                self.agents.get(neighbor),
            )
            if distance >= 1:
                self.graph.remove_edge(sampled_agent, neighbor)
            else:
                neighbors_neighbor = list(self.graph.neighbors(neighbor))
                self.graph.add_edge(sampled_agent, sample(neighbors_neighbor, 1)[0])

    def run_simulation(self):
        for turn in range(100):
            self.take_turn()
