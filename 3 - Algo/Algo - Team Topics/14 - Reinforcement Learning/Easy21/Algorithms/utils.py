from typing import Tuple

import numpy as np

from utils.action import Action
from utils.environment import Environment
from utils.state import State


def find_optimal_v(env: Environment, q: np.ndarray) -> np.array:
    """ compute the optimal state value function

    :param env: environment
    :param q: optimal state-action value function
    :return: optimal state value function, such that v[s.id] if the optimal value for state s
    """

    return np.array([np.max(q[s.id]) for s in env.states])


def find_optimal_policy(env: Environment, q: np.ndarray) -> np.ndarray:
    """ compute the optimal policy from the state-action value function

    :param env: environment
    :param q: the state-action value function
    :return: policy such that policy[s.id] is the chosen action for state s
    """
    return np.array([np.argmax(q[s.id]) for s in env.states])


def get_random_initial_state() -> State:
    return State(player=np.random.randint(1, 11), dealer=np.random.randint(1, 11))


def sample(env: Environment, q: np.ndarray, state_counter: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Sample an episode from randomly chosen initial state

    :param env: environment
    :param q: the current state-action value function
    :param state_counter: amount of times every states has been visited
    :return: (states, actions, rewards)
    """

    s = get_random_initial_state()
    states = []
    rewards = []
    actions = []

    while not s.terminal:
        states.append(s)
        a = choose_epsilon_greedily(env, s, q, state_counter)
        s, r = env.step(s, a)
        rewards.append(r)
        actions.append(a)

    return np.array(states), np.array(actions), np.array(rewards)


def choose_epsilon_greedily(env: Environment, s: State, q: np.ndarray, state_counter: np.array) -> Action:
    """ choose action according to epsilon greedy policy based on current state-action value function

    :param env: Environment
    :param s: current state
    :param q: current state-action value function
    :param state_counter: state counter
    :return: chosen action
    """

    epsilon = 100 / (100 + state_counter[s.id])
    probs = np.ones(env.nA) * epsilon / env.nA
    probs[np.argmax(q[s.id])] += (1 - epsilon)
    a = np.random.choice(env.actions, p=probs)
    return a
