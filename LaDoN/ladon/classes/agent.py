import numpy as np


class Agent:
    def __init__(self):
        self.opinion = np.random.uniform(low=-1, high=1, size=None)
        self.initial_opinion = self.opinion
