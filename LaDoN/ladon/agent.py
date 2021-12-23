import numpy as np
from typing import Dict
from ladon.config import AGENT_CONFIG
from collections import defaultdict


class Agent:
    def __init__(self, agent_config: Dict):
        self.type = agent_config.get("type")
        self.inner_vector = np.random.normal(
            agent_config.get("inner_vector").get("mu"),
            agent_config.get("inner_vector").get("sigma"),
            agent_config.get("inner_vector").get("vector_size"),
        )
        self.inner_mean = self.inner_vector.mean()
        self.outer_vector = np.random.normal(
            agent_config.get("outer_vector").get("mu"),
            agent_config.get("outer_vector").get("sigma"),
            agent_config.get("outer_vector").get("vector_size"),
        )
        self.outer_mean = self.outer_vector.mean()
        self.social_memory = defaultdict(lambda: 0)


agent = Agent(AGENT_CONFIG)
agent_2 = Agent(AGENT_CONFIG)
