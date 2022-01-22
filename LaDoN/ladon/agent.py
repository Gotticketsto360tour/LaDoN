import numpy as np
from typing import Dict
from config import AGENT_CONFIG


class Agent:
    def __init__(self, agent_config: Dict):
        self.type = agent_config.get("type")
        self.opinion = np.random.normal(
            agent_config.get("opinion").get("mu"),
            agent_config.get("opinion").get("sigma"),
            agent_config.get("opinion").get("vector_size"),
        )


agent = Agent(AGENT_CONFIG)
agent_2 = Agent(AGENT_CONFIG)
