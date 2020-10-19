from typing import List

from scipy.spatial.distance import cdist
import numpy as np


class Environment:

    def __init__(self, cities: np.ndarray) -> None:
        self.cities = cities
        self.amount = cities.shape[0]
        self.dists = cdist(cities, cities)
