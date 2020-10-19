from typing import List

import numpy as np

from environment import Environment


class Route:

    def __init__(self, env: Environment, order: np.ndarray):
        self.env = env
        self.order = order
        self.length = 0

    def get_length(self):
        return np.sum(self.env.dists[self.order[:-1], self.order[1:]])

    def fitness(self):
        return 1/self.length

    @staticmethod
    def calc_lengths(routes: List['Route']):
        sources = np.hstack([route.order[:-1] for route in routes])
        targets = np.hstack([route.order[1:] for route in routes])
        lengths = np.sum(routes[0].env.dists[sources, targets].reshape(len(routes), -1), axis=1)
        for i, l in enumerate(lengths):
            routes[i].length = l
