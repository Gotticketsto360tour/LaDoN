from typing import Dict
import networkx as nx
from networkx.generators.random_graphs import (
    watts_strogatz_graph,
    barabasi_albert_graph,
)
import numpy as np
from agent import Agent
from config import CONFIGS
from random import sample
from random import random
from helpers import compare_vectors


class Network:
    def __init__(self, graph: str = "smallworld"):
        self.THRESHOLD = 2
        self.N_AGENTS = 1000
        self.N_GROUPS = len(CONFIGS)
        self.agents = {}

        if graph == "smallworld":
            self.graph = watts_strogatz_graph(self.N_AGENTS, 4, 0.1)
        elif graph == "scalefree":
            self.graph = barabasi_albert_graph(self.N_AGENTS, 4)

        self.initialize_network(CONFIGS)

    def generate_and_sever_connections(self, sampled_agent):
        sampled_agent_neighbors = list(self.graph.neighbors(sampled_agent))
        for neighbor in sampled_agent_neighbors:
            distance = compare_vectors(
                self.agents.get(sampled_agent), self.agents.get(neighbor), neighbor
            )
            if distance >= self.THRESHOLD:
                self.graph.remove_edge(sampled_agent, neighbor)
                return False, neighbor
            else:
                return True, neighbor
        return False, None

    def initialize_network(self, CONFIGS: Dict):
        for agent in range(self.N_AGENTS):
            agent_type = sample(list(CONFIGS.keys()), 1)[0]
            self.agents[agent] = Agent(CONFIGS[agent_type])

    def record_interactions(self, sampled_agent):
        sampled_agent_neighbors = list(self.graph.neighbors(sampled_agent))
        for neighbor in sampled_agent_neighbors:
            self.agents.get(sampled_agent).social_memory[neighbor] += 1

    def take_turn(self):
        sampled_agent = sample(self.graph.nodes, 1)[0]
        random_number = random()
        if random_number < 0.1:
            random_agent = sample(self.graph.nodes, 1)[0]
            while random_agent == sampled_agent:
                random_agent = sample(self.graph.nodes, 1)[0]
            self.graph.add_edge(sampled_agent, random_agent)
        else:
            flag, neighbor = self.generate_and_sever_connections(sampled_agent)
            self.record_interactions(sampled_agent)
            if flag:
                neighbors_neighbor = [
                    neighbor
                    for neighbor in self.graph.neighbors(neighbor)
                    if neighbor != sampled_agent
                ]
                for new_neighbor in neighbors_neighbor:
                    distance = compare_vectors(
                        self.agents.get(sampled_agent),
                        self.agents.get(new_neighbor),
                        new_neighbor,
                    )
                    if distance <= self.THRESHOLD:
                        self.graph.add_edge(sampled_agent, new_neighbor)
                        break

    def run_simulation(self):
        for _ in range(100000):
            self.take_turn()
