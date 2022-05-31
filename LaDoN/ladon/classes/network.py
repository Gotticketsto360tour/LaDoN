from statistics import mean
import networkx as nx
from ladon.classes.agent import Agent
from random import sample
from random import random
from random import seed
from ladon.helpers.helpers import find_distance, find_average_path
from tqdm import tqdm
import numpy as np


class Network:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        self.N_AGENTS = 0
        self.agent_number = 0
        self.EDGE_SURPLUS = 0
        self.N_TIE_DISSOLUTIONS = 0
        self.agents = {}
        self.graph = nx.generators.random_graphs.watts_strogatz_graph(
            n=self.N_TARGET, k=self.K, p=self.P
        )
        self.initialize_network()
        self.initialize_vectors()

    def initialize_network(self):
        """Fills the dictionary of agents with class instantiations
        of the "Agent" datatype
        """
        for agents in range(self.N_TARGET):
            self.agents[agents] = Agent()

    def initialize_vectors(self):
        self.MEAN_ABSOLUTE_OPINIONS = []
        self.SD_ABSOLUTE_OPINIONS = []
        self.NEGATIVE_TIES_DISSOLUTED = []
        self.MEAN_DISTANCE = []
        self.AVERAGE_PATH_LENGTH = []
        self.AVERAGE_CLUSTERING = []
        self.ASSORTATIVITY = []
        self.OPINION_DISTRIBUTIONS = []

    def record_opinion_distributions(self):
        self.OPINION_DISTRIBUTIONS.append(self.get_opinion_distribution())

    def record_time_step(self):
        """Records measures of interest over time"""
        absolute_opinions = abs(self.get_opinion_distribution())
        mean_absolute_opinions = np.mean(absolute_opinions)
        standard_deviation = np.std(absolute_opinions)
        negative_ties_dissoluted = self.N_TIE_DISSOLUTIONS
        mean_distances = np.mean(self.get_opinion_distances_without_none())
        self.MEAN_ABSOLUTE_OPINIONS.append(mean_absolute_opinions)
        self.SD_ABSOLUTE_OPINIONS.append(standard_deviation)
        self.NEGATIVE_TIES_DISSOLUTED.append(negative_ties_dissoluted)
        self.MEAN_DISTANCE.append(mean_distances)
        self.AVERAGE_CLUSTERING.append(nx.average_clustering(self.graph))
        self.AVERAGE_PATH_LENGTH.append(find_average_path(self.graph))
        self.ASSORTATIVITY.append(
            nx.algorithms.assortativity.degree_assortativity_coefficient(self.graph)
        )

    def get_opinion_distribution(self) -> np.array:
        return np.array(
            [float(self.agents.get(agent).opinion) for agent in self.agents]
        )

    def get_initial_opinion_distribution(self) -> np.array:
        return np.array(
            [float(self.agents.get(agent).initial_opinion) for agent in self.agents]
        )

    def get_degree_distribution(self) -> np.array:
        return np.array([degree[1] for degree in list(self.graph.degree())])

    def get_opinion_distances(self) -> np.array:
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

    def get_opinion_distances_without_none(self) -> np.array:
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
                for agent in self.agents
                if list(self.graph.neighbors(agent))
            ]
        )

    def get_clustering(self) -> np.array:
        return np.array(list(nx.algorithms.clustering(self.graph).values()))

    def get_centrality(self) -> np.array:
        return np.array(
            list(nx.algorithms.centrality.betweenness_centrality(self.graph).values())
        )

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
            self.EDGE_SURPLUS -= 1
            sampled_agent = sample(nodes_without_new_agent, 1)[0]
            self.graph.add_edge(agent_on_turn, sampled_agent)

    def add_new_connection_through_neighbors(self, agent_on_turn):

        candidate_neighbors = [
            neighbors_neighbor
            for neighbor in list(self.graph.neighbors(agent_on_turn))
            for neighbors_neighbor in list(self.graph.neighbors(neighbor))
            if neighbors_neighbor != agent_on_turn
            and neighbors_neighbor not in self.graph.neighbors(agent_on_turn)
        ]

        if candidate_neighbors:
            self.EDGE_SURPLUS -= 1
            sampled_neigbor = sample(candidate_neighbors, 1)[0]
            self.graph.add_edge(agent_on_turn, sampled_neigbor)

    def delete_tie(self, agent: int, neighbor: int):
        if random() <= self.TIE_DISSOLUTION:
            self.graph.remove_edge(agent, neighbor)
            self.N_TIE_DISSOLUTIONS += 1
            self.EDGE_SURPLUS += 1
            self.ensure_no_lone_agents(agent, neighbor)

    def update_all_values(self, agent: int):

        neighbor_list = list(self.graph.neighbors(agent))
        for neighbor in sample(neighbor_list, k=len(neighbor_list)):
            self.update_values(agent, neighbor)

        negative_relations = [
            neighbor
            for neighbor in self.graph.neighbors(agent)
            if find_distance(self.agents.get(agent), self.agents.get(neighbor))
            > self.THRESHOLD
        ]
        remove_ties = [
            self.delete_tie(agent, neighbor) for neighbor in negative_relations
        ]

    def ensure_no_lone_agents(self, sampled_agent: int, neighbor: int):
        """Function for ensuring that the network only has one component.

        Args:
            sampled_agent (int): Integer which specifies the sampled agent
            neighbor (int): Integer which specifies the neighbor
        """
        sampled_agents_neighbors = list(self.graph.neighbors(sampled_agent))
        neighbors_neighbors = list(self.graph.neighbors(neighbor))
        if len(sampled_agents_neighbors) == 0:
            self.add_new_connection_randomly(sampled_agent)
            return False
        elif len(neighbors_neighbors) == 0:
            self.add_new_connection_randomly(neighbor)
            return False
        elif not nx.has_path(
            self.graph, sampled_agents_neighbors[0], neighbors_neighbors[0]
        ):
            self.graph.add_edge(sampled_agent, neighbor)
            self.EDGE_SURPLUS -= 1
            return False
        else:
            return True

    def take_turn(self):
        sampled_agent = sample(self.graph.nodes, 1)[0]
        list_of_neighbors = list(self.graph.neighbors(sampled_agent))
        ensured_no_lone_agents = True
        if self.EDGE_SURPLUS < 1 and len(list_of_neighbors) > 0:
            self.EDGE_SURPLUS += 1
            removed_edge = sample(list_of_neighbors, 1)[0]
            self.graph.remove_edge(sampled_agent, removed_edge)
            ensured_no_lone_agents = self.ensure_no_lone_agents(
                sampled_agent, removed_edge
            )
        if ensured_no_lone_agents:
            if random() >= self.RANDOMNESS and list_of_neighbors:
                self.add_new_connection_through_neighbors(sampled_agent)
            else:
                self.add_new_connection_randomly(sampled_agent)

        self.update_all_values(sampled_agent)

    def run_simulation(self):
        for timestep in tqdm(range(1, self.N_TIMESTEPS + 1)):
            self.take_turn()
            if timestep % 20 == 0 and self.RECORD:
                self.record_time_step()
            if timestep % 500 == 0 and self.RECORD:
                self.record_opinion_distributions()
        return "DONE"


class NoOpinionNetwork(Network):
    """Child of the Network Class.
    Includes all the same methods, but
    the take turn function is edited so
    that it does not include any opinion
    dynamics.

    Args:
        Network (Network): The Co-evolutionary Network Class
    """

    def take_turn(self):
        sampled_agent = sample(self.graph.nodes, 1)[0]
        list_of_neighbors = list(self.graph.neighbors(sampled_agent))
        ensured_no_lone_agents = True
        if self.EDGE_SURPLUS < 1 and list_of_neighbors:
            self.EDGE_SURPLUS += 1
            removed_edge = sample(list_of_neighbors, 1)[0]
            self.graph.remove_edge(sampled_agent, removed_edge)
            ensured_no_lone_agents = self.ensure_no_lone_agents(
                sampled_agent, removed_edge
            )
        if ensured_no_lone_agents:
            if random() >= self.RANDOMNESS and list_of_neighbors:
                self.add_new_connection_through_neighbors(sampled_agent)
            else:
                self.add_new_connection_randomly(sampled_agent)


class ScaleFreeNetwork(Network):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        self.N_AGENTS = 0
        self.agent_number = 0
        self.EDGE_SURPLUS = 0
        self.N_TIE_DISSOLUTIONS = 0
        self.agents = {}
        self.graph = nx.generators.random_graphs.barabasi_albert_graph(
            n=self.N_TARGET, m=self.K
        )
        self.initialize_network()
        self.initialize_vectors()
