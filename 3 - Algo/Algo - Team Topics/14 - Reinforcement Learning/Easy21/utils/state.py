class State:

    def __init__(self, player: int, dealer: int, id=-1):
        self.player = player
        self.dealer = dealer
        self.terminal = player == 0 or dealer == 0
        self.id = id if id != -1 else self.get_id()

    def get_id(self) -> int:
        return self.player - 1 + 21*(self.dealer - 1)

    def __eq__(self, other) -> bool:
        return self.player == other.player and self.dealer == other.dealer

    def __hash__(self) -> int:
        return 100*self.player + self.dealer


def get_state_by_id(id: int) -> State:
    player = id % 21
    dealer = id // 21
    return State(player, dealer)


WINNER = State(player=21, dealer=0, id=210)
LOSER = State(player=0, dealer=21, id=211)
DRAW = State(player=0, dealer=0, id=212)
