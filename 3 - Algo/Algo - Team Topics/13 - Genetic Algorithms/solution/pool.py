import numpy as np

from environment import Environment
from route import Route

ELITE = 0.2
MUTATION = 0.01


class Pool:

    def __init__(self, env: Environment, size: int):
        self.population = []
        self.env = env
        self.population_size = size
        for _ in range(size):
            self.population.append(Route(env, np.random.permutation(env.amount)))
        Route.calc_lengths(self.population)
        self.best_route_by_generation = []
        self.best_route_by_generation.append(np.min([route.length for route in self.population]))

    def crossover(self, route1, route2):  # contains mutation
        start, end = sorted(np.random.choice(np.arange(self.env.amount), size=2))
        route2_remains = route2.order[~np.isin(route2.order, route1.order[start:end])]

        new_order = np.hstack([route1.order[start:end], route2_remains]).astype(int)

        # mutation
        mutations = np.random.random(len(new_order)) < MUTATION
        for i, to_mutate in enumerate(mutations):
            if not to_mutate:
                continue
            swap_idx = np.random.randint(len(new_order))

            original_city = new_order[i]
            new_city = new_order[swap_idx]
            new_order[swap_idx] = original_city
            new_order[i] = new_city

        return Route(self.env, new_order)

    def next_generation(self):
        population_probabilities = self.get_population_probabilities()
        new_population = []
        elite_size = int(ELITE*self.population_size)
        new_population += sorted(self.population, key=lambda x: x.fitness(), reverse=True)[:elite_size]
        for _ in range(self.population_size - elite_size):
            route1, route2 = np.random.choice(self.population, size=2, replace=False, p=population_probabilities)
            new_population.append(self.crossover(route1, route2))
        Route.calc_lengths(new_population)
        self.population = new_population

    def get_population_probabilities(self):
        fitness = np.array([route.fitness() for route in self.population])
        return fitness/np.sum(fitness)

    def run(self, generation_amount=50):
        for _ in range(generation_amount):
            if (_ + 1) % 10 == 0:
                print(f'Creating generation #{len(self.best_route_by_generation)}')
            self.next_generation()
            self.best_route_by_generation.append(np.min([route.length for route in self.population]))