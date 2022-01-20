import numpy
from random import random
import matplotlib.pyplot as plt
import networkx as nx


def generate_agents(target: int, timesteps: int):
    current_agents = [0]
    for _ in range(timesteps):
        n = current_agents[-1]
        p_d = n / (2 * target)
        if random() >= p_d:
            n += 1
        else:
            n -= 1
        current_agents.append(n)

    return current_agents


data = generate_agents(target=200, timesteps=10000)

plt.plot(data)
