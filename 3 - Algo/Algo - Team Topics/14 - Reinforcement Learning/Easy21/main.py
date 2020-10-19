from mpl_toolkits.mplot3d import Axes3D

from Algorithms.linear_approximation import linear_approximation
from Algorithms.monte_carlo_control import monte_carlo_control
from Algorithms.sarsa import sarsa
from utils.action import Action
from utils.environment import Environment

import numpy as np
from matplotlib import pyplot as plt

from utils.state import State


def plot_policy(env: Environment, policy: np.ndarray) -> None:
    p = np.zeros((21, 10))
    for s in env.states:
        p[s.player - 1, s.dealer - 1] = policy[s.id]

    plt.imshow(p)
    plt.xticks(range(10), range(1, 11))
    plt.yticks(range(21), range(1, 22))
    plt.show()
    print(p)


def plot_v(env: Environment, v: np.array) -> None:
    x = np.arange(1, 11)
    y = np.arange(1, 22)
    X, Y = np.meshgrid(x, y)
    V = np.zeros((21, 10))
    for s in env.states:
        V[s.player - 1, s.dealer - 1] = v[s.id]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()


def sample(env: Environment, policy: np.array):
    s = State(player=np.random.randint(1, 11), dealer=np.random.randint(1, 11))
    r = 0

    while not s.terminal:
        s, r = env.step(s, Action(policy[s.id]))

    return r


def test(env: Environment, policy: np.array):
    test_size = 10000
    score = 0
    for _ in range(test_size):
        if sample(env, policy) == 1:
            score += 1

    print(score / test_size)


def sarsa_main(recompute=True):
    env = Environment()

    opt_q = np.load('outputs/mc/q.npy')

    if recompute:
        for i, c in enumerate(np.arange(11) / 10):
            sarsa(env, c, episodes=15000, opt_q=opt_q)

    x = np.arange(15001)
    for i in np.arange(11):
        error = np.load(f'outputs/sarsa/errors/{i}.npy')
        plt.plot(x, error)

    plt.legend(np.arange(11) / 10)
    plt.xlabel('# of episodes')
    plt.ylabel('MSE')
    plt.title('SARSA with different lambda values')
    plt.show()


def linear_approx_main():
    env = Environment()

    opt_q = np.load('outputs/mc/q.npy')

    c_values = np.arange(11)/10
    error = []
    for c in c_values:
        print(c)
        q, v, policy = linear_approximation(env, c)
        error.append(np.mean((opt_q - q) ** 2))

    plt.plot(c_values, error)
    plt.xlabel('lambda value')
    plt.ylabel('MSE')
    plt.title('Linear approximation')
    plt.show()


if __name__ == '__main__':

    sarsa_main(recompute=False)
