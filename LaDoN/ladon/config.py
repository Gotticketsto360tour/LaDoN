AGENT_CONFIG = {"mu": 0, "sigma": 1, "vector_size": 5, "type": 1}
AGENT_CONFIG_2 = {"mu": 1, "sigma": 1, "vector_size": 5, "type": 2}
AGENT_CONFIG_3 = {"mu": 0.5, "sigma": 0.5, "vector_size": 5, "type": 3}

CONFIGURATIONS = [AGENT_CONFIG, AGENT_CONFIG_2, AGENT_CONFIG_3]

CONFIGS = {i: CONFIG for i, CONFIG in enumerate(CONFIGURATIONS)}
