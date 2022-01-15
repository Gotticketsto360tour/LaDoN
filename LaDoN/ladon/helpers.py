import numpy as np
from agent import Agent

def compare_values(A: Agent, B: Agent):
    return np.linalg.norm((A.outer_vector - B.outer_vector))

def compare_vectors(A: Agent, B: Agent, neigbor_number: int):
    weight = A.social_memory[neigbor_number] / (A.social_memory[neigbor_number] + 4)
    comparison = weight * np.linalg.norm((A.inner_vector - B.inner_vector)) + (
        1 - weight
    ) * np.linalg.norm((A.outer_vector - B.outer_vector))
    return comparison
