from typing import Tuple, List
import numpy as np

from utils.action import Action
from utils.state import State, LOSER, WINNER, DRAW


class Environment:

    def __init__(self) -> None:

        self.states, self.actions = self.initialize()
        self.nS = len(self.states)
        self.nA = len(self.actions)

    @staticmethod
    def initialize() -> Tuple[List[State], List[Action]]:

        states = []
        for dealer in range(1, 11):
            for player in range(1, 22):
                states.append(State(player=player, dealer=dealer))

        actions = [Action.HIT, Action.STICK]

        return states, actions

    @staticmethod
    def draw_card() -> Tuple[int, int]:
        """ Draw card fro, the deck

        :return: (value, color) where value is 1-10 in uniform distribution and sign is -1/+1 with 33%/66% distribution
        """

        n = np.random.randint(low=1, high=11)
        sign = -1 if np.random.random() < 1 / 3 else +1
        return n, sign

    def step(self, s: State, a: Action) -> Tuple[State, int]:
        """ Advances the current state in one step according to chosen action

        :param s: current state
        :param a: chosen action
        :return: (next state, reward)
        """

        assert s in self.states
        assert a in self.actions

        if a == Action.HIT:
            n, sign = self.draw_card()
            player = s.player + sign * n

            return (State(player=player, dealer=s.dealer), 0) if 1 <= player <= 21 else (LOSER, -1)

        dealer = s.dealer
        while 1 <= dealer <= 16:
            n, sign = self.draw_card()
            dealer = dealer + sign * n

        if 17 <= dealer <= 21:
            return (WINNER, 1) if s.player > dealer else (DRAW, 0) if s.player == dealer else (LOSER, -1)

        return WINNER, 1
