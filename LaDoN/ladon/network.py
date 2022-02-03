from typing import Dict
import networkx as nx
from agent import Agent
from config import CONFIGS
from random import sample
from random import random
from helpers import find_distance
from tqdm import tqdm
import numpy as np


class Network:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        self.N_GROUPS = len(CONFIGS)
        self.N_AGENTS = 0
        self.agent_number = 0
        self.agents = {}
        self.graph = nx.Graph()

    def get_opinion_distribution(self):
        return np.array(
            [float(self.agents.get(agent).opinion) for agent in self.agents]
        )

    def get_initial_opinion_distribution(self):
        return np.array(
            [float(self.agents.get(agent).initial_opinion) for agent in self.agents]
        )

    def get_degree_distribution(self):
        return np.array([degree[1] for degree in list(self.graph.degree())])

    def update_values(self, sampled_agent, neigbor):
        list_of_agents = [self.agents.get(sampled_agent), self.agents.get(neigbor)]
        max_agent, min_agent = max(list_of_agents, key=lambda x: x.opinion), min(
            list_of_agents, key=lambda x: x.opinion
        )
        V = find_distance(max_agent, min_agent)

        if V <= self.THRESHOLD:
            V = V * self.POSITIVE_LEARNING_RATE
            max_agent.opinion -= V
            min_agent.opinion += V
        else:
            V = V * self.NEGATIVE_LEARNING_RATE
            max_agent.opinion += V
            min_agent.opinion -= V

    def add_new_connection_randomly(self, agent_on_turn):
        nodes_without_new_agent = [
            agent
            for agent in self.graph.nodes
            if agent != agent_on_turn
            and agent not in self.graph.neighbors(agent_on_turn)
        ]
        if nodes_without_new_agent:
            sampled_agent = sample(nodes_without_new_agent, 1)[0]
            self.graph.add_edge(agent_on_turn, sampled_agent)

    def add_new_connection_through_neighbors(self, agent_on_turn):

        # NOTE: If a neighborhood is fully connected, this action will still be wasted

        neighbors = list(self.graph.neighbors(agent_on_turn))

        while neighbors:
            sampled_neighbor = sample(neighbors, 1)[0]
            # will ensure that no turn is "wasted" by making already existing edges
            candidate_neighbors = [
                agent
                for agent in list(self.graph.neighbors(sampled_neighbor))
                if agent not in neighbors and agent != agent_on_turn
            ]
            if candidate_neighbors:
                new_neigbor = sample(candidate_neighbors, 1)[0]
                self.graph.add_edge(agent_on_turn, new_neigbor)
                break
            else:
                neighbors.remove(sampled_neighbor)

    def generate_or_eliminate_agent(self, CONFIGS: Dict):
        P_d = self.N_AGENTS / (2 * self.N_TARGET)
        if random() >= P_d:
            self.graph.add_node(self.agent_number)
            agent_type = sample(list(CONFIGS.keys()), 1)[0]
            new_agent = self.agent_number
            self.agents[self.agent_number] = Agent(CONFIGS[agent_type])
            self.agent_number += 1
            self.N_AGENTS += 1
            if self.N_AGENTS >= 2:
                self.add_new_connection_randomly(new_agent)

                # this probability could be a different probability than the other random process

                if random() < self.RANDOMNESS:
                    self.add_new_connection_randomly(new_agent)
                else:
                    self.add_new_connection_through_neighbors(new_agent)

        else:
            sampled_agent = sample(self.graph.nodes, 1)[0]
            self.graph.remove_node(sampled_agent)
            del self.agents[sampled_agent]
            self.N_AGENTS -= 1

    def take_turn(self):
        self.generate_or_eliminate_agent(CONFIGS)
        if self.N_AGENTS >= 2:
            sampled_agent = sample(self.graph.nodes, 1)[0]
            if random() < self.RANDOMNESS:
                self.add_new_connection_randomly(sampled_agent)
            else:
                self.add_new_connection_through_neighbors(sampled_agent)

            for neighbor in self.graph.neighbors(sampled_agent):
                self.update_values(sampled_agent, neighbor)
            # map(
            #     lambda x: self.update_values(sampled_agent, x),
            #     self.graph.neighbors(sampled_agent),
            # )

            negative_relations = [
                neighbor
                for neighbor in self.graph.neighbors(sampled_agent)
                if find_distance(
                    self.agents.get(sampled_agent), self.agents.get(neighbor)
                )
                > self.THRESHOLD
            ]
            for neighbor in negative_relations:
                self.graph.remove_edge(sampled_agent, neighbor)
            # map(lambda x: self.graph.remove_edge(sampled_agent, x), negative_relations)

    def run_simulation(self):
        for _ in tqdm(range(self.N_TIMESTEPS)):
            self.take_turn()
        if self.STOP_AT_TARGET:
            while self.N_AGENTS < self.N_TARGET:
                self.take_turn()
            return "DONE"
