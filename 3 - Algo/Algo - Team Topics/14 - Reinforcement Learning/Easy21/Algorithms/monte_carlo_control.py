from typing import Dict, Tuple, List
import numpy as np

from Algorithms.utils import find_optimal_policy, find_optimal_v, sample
from utils.action import Action
from utils.environment import Environment
from utils.state import State


def monte_carlo_control(env: Environment) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = find_optimal_q(env)

    v = find_optimal_v(env, q)
    policy = find_optimal_policy(env, q)
    return q, v, policy


def find_optimal_q(env: Environment, episodes: int = 10000000) -> np.ndarray:
    """ using monte carlo control to find the optimal state-action value function

    :param env: environment
    :param episodes: amount of episodes to sample
    :return: array q such that q[s.id][a] is the optimal state-action value
    """

    state_counter = np.zeros(env.nS)
    state_action_counter = np.zeros((env.nS, env.nA))
    q = np.zeros((env.nS, env.nA))

    for k in range(1, episodes + 1):

        if k % 1000 == 0:
            print(k)

        states, actions, rewards = sample(env, q, state_counter)
        returns = np.ones(len(states)) * rewards[-1]

        for s, a, r in zip(states, actions, returns):
            state_counter[s.id] += 1
            state_action_counter[s.id, a.value] += 1
            q[s.id, a.value] += 1 / state_action_counter[s.id, a.value] * (r - q[s.id, a.value])

    return q
