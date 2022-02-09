from statistics import mean
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

    def get_opinion_distances(self):
        return np.array(
            [
                mean(
                    [
                        abs(
                            self.agents.get(agent).opinion
                            - (self.agents.get(neighbor).opinion)
                        )
                        for neighbor in self.graph.neighbors(agent)
                    ]
                )
                if list(self.graph.neighbors(agent))
                else None
                for agent in self.agents
            ]
        )

    def get_agent_numbers(self):
        return np.array([agent for agent in self.agents])

    def get_centrality(self):
        return np.array(
            list(nx.algorithms.centrality.betweenness_centrality(self.graph).values())
        )

    def write_data(self):
        data = {
            "opinions": self.get_opinion_distribution(),
            "initial_opinions": self.get_initial_opinion_distribution(),
            "degrees": self.get_degree_distribution(),
            "distances": self.get_opinion_distances(),
            "centrality": self.get_centrality(),
        }

    def update_values(self, sampled_agent, neigbor):
        list_of_agents = [self.agents.get(sampled_agent), self.agents.get(neigbor)]
        max_agent, min_agent = max(list_of_agents, key=lambda x: x.opinion), min(
            list_of_agents, key=lambda x: x.opinion
        )
        distance = find_distance(max_agent, min_agent)

        if distance <= self.THRESHOLD:
            V = 0.5 * distance * self.POSITIVE_LEARNING_RATE
            max_agent.opinion -= V
            min_agent.opinion += V
            if max_agent.opinion < -1:
                max_agent.opinion = -1
            if min_agent.opinion > 1:
                min_agent.opinion = 1
        else:
            V = 0.5 * distance * self.NEGATIVE_LEARNING_RATE
            max_agent.opinion += V
            min_agent.opinion -= V
            if max_agent.opinion > 1:
                max_agent.opinion = 1
            if min_agent.opinion < -1:
                min_agent.opinion = -1

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
                self.update_all_values(new_agent)

        else:
            sampled_agent = sample(self.graph.nodes, 1)[0]
            self.graph.remove_node(sampled_agent)
            del self.agents[sampled_agent]
            self.N_AGENTS -= 1

    def update_all_values(self, agent):
        for neighbor in self.graph.neighbors(agent):
            self.update_values(agent, neighbor)

        negative_relations = [
            neighbor
            for neighbor in self.graph.neighbors(agent)
            if find_distance(self.agents.get(agent), self.agents.get(neighbor))
            > self.THRESHOLD
        ]
        for neighbor in negative_relations:
            self.graph.remove_edge(agent, neighbor)

    def take_turn(self):
        self.generate_or_eliminate_agent(CONFIGS)
        if self.N_AGENTS >= 2:
            sampled_agent = sample(self.graph.nodes, 1)[0]
            if random() < self.RANDOMNESS:
                self.add_new_connection_randomly(sampled_agent)
            else:
                self.add_new_connection_through_neighbors(sampled_agent)

            self.update_all_values(sampled_agent)

    def run_simulation(self):
        for _ in tqdm(range(self.N_TIMESTEPS)):
            self.take_turn()
        if self.STOP_AT_TARGET:
            while self.N_AGENTS < self.N_TARGET:
                self.take_turn()
            return "DONE"
