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
from helpers import compare_vectors, compare_values


class Network:
    def __init__(self, graph: str = "smallworld"):
        self.THRESHOLD = 0.1
        self.N_AGENTS = 200
        self.RANDOMNESS = 0.1
        self.N_GROUPS = len(CONFIGS)
        self.N_TIMESTEPS = 10000
        self.learning_rate = 0.1
        self.agents = {}

        if graph == "smallworld":
            self.graph = watts_strogatz_graph(self.N_AGENTS, 4, 0.1)
        elif graph == "scalefree":
            self.graph = barabasi_albert_graph(self.N_AGENTS, 4)

        self.initialize_network(CONFIGS)

    def update_values(self, sampled_agent, neigbor, flag):
        agent_vector = self.agents.get(sampled_agent).outer_vector
        neigbor_vector = self.agents.get(neigbor).outer_vector

        if flag:
            self.agents.get(sampled_agent).outer_vector = (
                agent_vector + (neigbor_vector - agent_vector) * self.learning_rate
            )
            self.agents.get(neigbor).outer_vector = (
                neigbor_vector + (agent_vector - neigbor_vector) * self.learning_rate
            )

    def generate_and_sever_connections(self, sampled_agent):
        sampled_agent_neighbors = list(self.graph.neighbors(sampled_agent))
        for neighbor in sampled_agent_neighbors:
            # distance = compare_vectors(
            #     self.agents.get(sampled_agent), self.agents.get(neighbor), neighbor
            # )
            distance = compare_values(
                self.agents.get(sampled_agent), self.agents.get(neighbor)
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
        if random_number < self.RANDOMNESS:
            random_agent = sample(self.graph.nodes, 1)[0]
            while random_agent == sampled_agent:
                random_agent = sample(self.graph.nodes, 1)[0]
            self.graph.add_edge(sampled_agent, random_agent)
            flag, neighbor = self.generate_and_sever_connections(sampled_agent)
            self.update_values(sampled_agent, neighbor, flag)
            self.record_interactions(sampled_agent)
        else:
            flag, neighbor = self.generate_and_sever_connections(sampled_agent)
            if neighbor:
                self.update_values(sampled_agent, neighbor, flag)
                self.record_interactions(sampled_agent)
            if flag:
                neighbors_neighbor = [
                    neighbor
                    for neighbor in self.graph.neighbors(neighbor)
                    if neighbor != sampled_agent
                ]
                for new_neighbor in neighbors_neighbor:
                    # distance = compare_vectors(
                    #     self.agents.get(sampled_agent),
                    #     self.agents.get(new_neighbor),
                    #     new_neighbor,
                    # )
                    distance = compare_values(
                        self.agents.get(sampled_agent), self.agents.get(new_neighbor)
                    )
                    if distance <= self.THRESHOLD:
                        self.graph.add_edge(sampled_agent, new_neighbor)
                        self.update_values(sampled_agent, new_neighbor, True)
                        break

    def run_simulation(self):
        for _ in range(self.N_TIMESTEPS):
            self.take_turn()
