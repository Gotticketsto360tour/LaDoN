from numpy.core.numeric import outer


AGENT_CONFIG = {
    "inner_vector": {"mu": 1, "sigma": 0.5, "vector_size": 1},
    "outer_vector": {"mu": 0, "sigma": 1, "vector_size": 1},
    "type": 1,
}
AGENT_CONFIG_2 = {
    "inner_vector": {"mu": 1, "sigma": 0.5, "vector_size": 1},
    "outer_vector": {"mu": 2, "sigma": 1, "vector_size": 1},
    "type": 2,
}
AGENT_CONFIG_3 = {
    "inner_vector": {"mu": 1, "sigma": 0.5, "vector_size": 1},
    "outer_vector": {"mu": 2, "sigma": 0.5, "vector_size": 1},
    "type": 3,
}

CONFIGURATIONS = [AGENT_CONFIG, AGENT_CONFIG_2]  # AGENT_CONFIG_3

CONFIGS = {i: CONFIG for i, CONFIG in enumerate(CONFIGURATIONS)}
