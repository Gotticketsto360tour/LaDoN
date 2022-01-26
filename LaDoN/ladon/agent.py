import numpy as np
from typing import Dict
from config import AGENT_CONFIG


class Agent:
    def __init__(self, agent_config: Dict, simulation_type: str = "uniform"):
        if simulation_type == "normal":
            self.type = agent_config.get("type")
            self.opinion = np.random.normal(
                agent_config.get("opinion").get("mu"),
                agent_config.get("opinion").get("sigma"),
                agent_config.get("opinion").get("vector_size"),
            )
        if simulation_type == "uniform":
            self.type = 0
            self.opinion = np.random.uniform(low=-1, high=1, size=None)
        self.initial_opinion = self.opinion


agent = Agent(AGENT_CONFIG)
agent_2 = Agent(AGENT_CONFIG)
