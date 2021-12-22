import numpy as np
from typing import Dict
from ladon.config import AGENT_CONFIG


class Agent:
    def __init__(self, agent_config: Dict):
        self.inner_vector = np.random.normal(
            agent_config.get("mu"),
            agent_config.get("sigma"),
            agent_config.get("vector_size"),
        )

    def compare_vectors(self, OtherAgent):
        return np.square(np.subtract(self.inner_vector, OtherAgent.inner_vector)).mean()


agent = Agent(AGENT_CONFIG)
agent_2 = Agent(AGENT_CONFIG)

agent.compare_vectors(agent_2)
