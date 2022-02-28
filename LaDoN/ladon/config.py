from numpy.core.numeric import outer

THRESHOLDS = [0.8, 0.9, 1, 1.1, 1.2]
RANDOMNESS = [0.1]
POSITIVE_LEARNING_RATES = [0.05, 0.1, 0.15, 0.2, 0.25]
NEGATIVE_LEARNING_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
TIE_DISSOLUTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]

AGENT_CONFIG = {
    "opinion": {"mu": 0, "sigma": 1, "vector_size": 1},
    "type": 1,
}
AGENT_CONFIG_2 = {
    "opinion": {"mu": 2, "sigma": 1, "vector_size": 1},
    "type": 2,
}
AGENT_CONFIG_3 = {
    "opinion": {"mu": 0, "sigma": 0.5, "vector_size": 1},
    "type": 3,
}

CONFIGURATIONS = [AGENT_CONFIG, AGENT_CONFIG_3]  # AGENT_CONFIG_3

CONFIGS = {i: CONFIG for i, CONFIG in enumerate(CONFIGURATIONS)}
