from numpy.core.numeric import outer


AGENT_CONFIG = {
    "opinion": {"mu": 1, "sigma": 1, "vector_size": 1},
    "type": 1,
}
AGENT_CONFIG_2 = {
    "opinion": {"mu": 2, "sigma": 1, "vector_size": 1},
    "type": 2,
}
AGENT_CONFIG_3 = {
    "opinion": {"mu": 1.5, "sigma": 0.5, "vector_size": 1},
    "type": 3,
}

CONFIGURATIONS = [AGENT_CONFIG, AGENT_CONFIG_2, AGENT_CONFIG_3]  # AGENT_CONFIG_3

CONFIGS = {i: CONFIG for i, CONFIG in enumerate(CONFIGURATIONS)}
