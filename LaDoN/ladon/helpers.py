import numpy as np
from ladon.agent import Agent


def compare_vectors(A: Agent, B: Agent):
    return np.square(np.subtract(A.inner_vector, B.inner_vector)).mean()
