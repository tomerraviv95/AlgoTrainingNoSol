from typing import List

from environment import Environment
from pool import Pool
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def load(data_path: str) -> Environment:
    data = pd.read_csv(data_path, sep=' ')
    cities = []
    for x, y in data[['x', 'y']].values:
        cities.append([x, y])
    return Environment(np.array(cities))


def draw_results(pool: Pool):
    plt.plot(range(len(pool.best_route_by_generation)), pool.best_route_by_generation)
    plt.plot(range(len(pool.best_route_by_generation)), [564] * len(pool.best_route_by_generation), color='r')
    plt.xlabel('generation')
    plt.ylabel('shortest path so far')
    plt.title('solve TSP with genetic algorithm')
    plt.show()


def draw_route(pool: Pool):
    plt.figure()
    cities_x, cities_y = pool.env.cities.T
    plt.scatter(cities_x, cities_y, color='black')
    order = pool.population[np.argmin([route.length for route in pool.population])].order
    for i in range(len(cities_x) - 1):
        plt.plot([cities_x[order[i]], cities_x[order[i+1]]], [cities_y[order[i]], cities_y[order[i+1]]], color='r')
    plt.show()


if __name__ == '__main__':
    env = load('datasets/dataset_1.csv')
    pool = Pool(env, 150)
    pool.run(15000)
    draw_results(pool)
    draw_route(pool)
