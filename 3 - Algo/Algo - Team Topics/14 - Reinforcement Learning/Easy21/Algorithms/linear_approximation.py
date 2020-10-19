from typing import Tuple
import numpy as np

from Algorithms.utils import find_optimal_v, find_optimal_policy, get_random_initial_state
from utils.action import Action
from utils.environment import Environment
from utils.state import State


def linear_approximation(env: Environment, c: float, episodes: int = 10000) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    q = approximate_q(env, c, episodes)

    v = find_optimal_v(env, q)
    policy = find_optimal_policy(env, q)

    return q, v, policy


def approximate_q(env: Environment, c: float, episodes: int) -> np.ndarray:
    w = np.zeros(36)

    for _ in range(episodes):
        E = np.zeros(36)
        s = get_random_initial_state()

        a = choose_epsilon_greedily(env, s, w).value
        f = to_feature_vector(s, a)

        while not s.terminal:
            s1, r = env.step(s, Action(a))

            if not s1.terminal:
                a1 = choose_epsilon_greedily(env, s1, w).value
                f1 = to_feature_vector(s1, a1)
                delta = r + np.dot(f1, w) - np.dot(f, w)
            else:
                a1 = 1
                f1 = 1
                delta = r - np.dot(f, w)

            E += f
            w += 0.01 * delta * E
            E *= c

            s = s1
            a = a1
            f = f1

    q = np.zeros((env.nS, env.nA))

    for s in env.states:
        for a in env.actions:
            f = to_feature_vector(s, a.value)
            q[s.id, a.value] = np.dot(f, w)

    return q


def to_feature_vector(s: State, a: Action) -> np.ndarray:
    f = np.zeros((3, 6, 2))

    dealer_feature = np.array([x <= s.dealer <= y for x, y in [(1, 4), (4, 7), (7, 10)]]).astype(int)
    player_feature = np.array(
        [x <= s.player <= y for x, y in [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]]).astype(int)

    for i in np.where(dealer_feature)[0]:
        for j in np.where(player_feature)[0]:
            f[i, j, a] = 1

    return f.flatten()


def choose_epsilon_greedily(env: Environment, s: State, w: np.ndarray) -> Action:

    hit = np.dot(to_feature_vector(s, Action.HIT.value), w)
    stick = np.dot(to_feature_vector(s, Action.STICK.value), w)
    epsilon = 0.05
    probs = np.ones(env.nA) * epsilon / env.nA
    if hit > stick:
        probs[0] += (1 - epsilon)
    else:
        probs[1] += (1 - epsilon)
    a = np.random.choice(env.actions, p=probs)
    return a