import numpy as np


class Agent:
    def __init__(self):
        self.initial_opinion = np.random.uniform(low=-1, high=1, size=None)
        self.opinion = self.initial_opinion
