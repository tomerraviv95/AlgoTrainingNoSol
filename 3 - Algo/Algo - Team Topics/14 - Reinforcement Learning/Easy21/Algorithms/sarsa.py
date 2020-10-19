from typing import Tuple, Optional
import numpy as np

from Algorithms.utils import find_optimal_v, find_optimal_policy, get_random_initial_state, choose_epsilon_greedily
from utils.action import Action
from utils.environment import Environment


def sarsa(env: Environment, c: float, episodes: int = 10000, opt_q: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """

    :param env: Environment
    :param c: lambda coefficient
    :param episodes: number of episodes
    :param opt_q: optimal value for error calculations
    :return:
    """
    q = find_optimal_q(env, c, episodes, opt_q)

    v = find_optimal_v(env, q)
    policy = find_optimal_policy(env, q)
    return q, v, policy


def find_optimal_q(env: Environment, c: float, episodes, opt_q: Optional[np.ndarray] = None) -> np.ndarray:
    """ using monte carlo control to find the optimal state-action value function

    :param env: environment
    :param c: lambda coefficient
    :param episodes: amount of episodes to sample
    :param opt_q: optimal value for error calculations
    :return: array q such that q[s.id][a] is the optimal state-action value
    """

    print(c)
    state_counter = np.zeros(env.nS)
    state_action_counter = np.zeros((env.nS, env.nA))
    q = np.zeros((env.nS, env.nA))
    error = []
    compute_errors = opt_q is not None
    if compute_errors:
        error.append(np.mean((opt_q - q) ** 2))

    for k in range(1, episodes + 1):

        E = np.zeros((env.nS, env.nA))
        s = get_random_initial_state()

        a = choose_epsilon_greedily(env, s, q, state_counter).value

        while not s.terminal:

            state_counter[s.id] += 1
            state_action_counter[s.id, a] += 1
            s1, r = env.step(s, Action(a))

            if not s1.terminal:
                a1 = choose_epsilon_greedily(env, s1, q, state_counter).value
                delta = r + q[s1.id, a1] - q[s.id, a]
            else:
                a1 = 1
                delta = r - q[s.id, a]

            E[s.id, a] += 1
            alpha = 1 / (state_action_counter[s.id, a])

            q += alpha * delta * E
            E *= c

            s = s1
            a = a1

        if compute_errors:
            error.append(np.mean((opt_q - q) ** 2))

    if compute_errors:
        np.save(f'outputs/sarsa/errors/{int(c * 10)}.npy', error)

    return q
