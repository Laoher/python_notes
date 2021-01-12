import math
import random

# 1 for cross, 0 for circle

class State:
    def __init__(self, board, turn=1):
        self.board = board
        self.turn = turn

    def __str__(self):
        return str(self.board) + str(self.turn)

    def __eq__(self, other):
        return self.board == other.board and self.turn == other.turn


class Model:
    CROSS_WIN = 1
    CIRCLE_WIN = -1
    DRAW = 0
    NOT_FINISHED = 2333

    def __init__(self, learning_rate, explore_rate, training_epoch=1000):
        self.value_table = {}
        self.learning_rate = learning_rate
        self.explore_rate = explore_rate
        self.training_epoch = training_epoch
        pass

    @staticmethod
    def result(state):
        board = state.board
        if (board[0] + board[3] + board[6] == 3
                or board[1] + board[4] + board[7] == 3
                or board[2] + board[5] + board[8] == 3
                or board[0] + board[1] + board[2] == 3
                or board[3] + board[4] + board[5] == 3
                or board[6] + board[7] + board[8] == 3
                or board[0] + board[4] + board[8] == 3
                or board[2] + board[4] + board[6] == 3):
            return Model.CROSS_WIN

        if (board[0] + board[3] + board[6] == -3
                or board[1] + board[4] + board[7] == -3
                or board[2] + board[5] + board[8] == -3
                or board[0] + board[1] + board[2] == -3
                or board[3] + board[4] + board[5] == -3
                or board[6] + board[7] + board[8] == -3
                or board[0] + board[4] + board[8] == -3
                or board[2] + board[4] + board[6] == -3):
            return Model.CIRCLE_WIN

        if sum(map(abs, board)) == 9:
            return Model.DRAW

        return Model.NOT_FINISHED

    def get_next_states(self, state):
        next_state_ids = []
        for i in range(len(state.board)):
            if state.board[i] == 0:
                next_state_ids.append(i)
        return next_state_ids

    def next_state(self, state, i):
        board = state.board[:]
        board[i] = state.turn
        return State(board, -state.turn)

    def explore(self, state):
        # get all states
        next_state_ids = self.get_next_states(state)
        if len(next_state_ids) == 0:
            return -1, -1

        # if state not in value table, set an initial value
        for i in next_state_ids:
            next_state = self.next_state(state, i)

            key = str(next_state)
            if key not in self.value_table:
                self.value_table[key] = 0.5

        # select next state randomly
        return random.choice(next_state_ids)

    def exploit(self, state):
        # select the state with highest (or lowest) value
        next_state_ids = self.get_next_states(state)
        if len(next_state_ids) == 0:
            return -1, -1

        if state.turn == 1:
            # cross turn, select the highest
            next_step = -1
            value = -math.inf
            for i in next_state_ids:
                next_state = self.next_state(state, i)
                key = str(next_state)

                # select the highest value
                if key in self.value_table:
                    if self.value_table[key] > value:
                        value = self.value_table[key]
                        next_step = i
                # set initial value for states not in value table
                else:
                    self.value_table[key] = 0.5     # set initial value of new state
                    if next_step == -1:
                        value = 0.5
                        next_step = i
            return next_step, value
        elif state.turn == -1:
            # circle turn, select lowest
            next_step = -1
            value = math.inf
            for i in next_state_ids:
                next_state = self.next_state(state, i)
                key = str(next_state)

                # select the lowest value
                if key in self.value_table:
                    if self.value_table[key] < value:
                        value = self.value_table[key]
                        next_step = i
                # set initial value for states not in value table
                else:
                    self.value_table[key] = 0.5  # set initial value of new state
                    if next_step == -1:
                        value = 0.5
                        next_step = i
            return next_step, value

    def train(self):
        for i in range(self.training_epoch):
            # get initial state, circle first
            board = [0 for _ in range(9)]
            board[random.randint(0, 8)] = -1
            state = State(board, 1)
            self.value_table[str(state)] = 0.5

            print("Train game %d: " % i, end="")

            # play one game
            while True:
                if self.result(state) == Model.CROSS_WIN:
                    self.value_table[str(state)] = 1
                    print("cross win", end=" ")
                    break
                elif self.result(state) == Model.CIRCLE_WIN:
                    self.value_table[str(state)] = 0
                    print("circle win", end=" ")
                    break
                elif self.result(state) == Model.DRAW:
                    self.value_table[str(state)] = 0.5
                    print("draw", end=" ")
                    break
                else:
                    print(str(state.board), end=" ")
                    if random.uniform(0, 1) < self.explore_rate:
                        next_step = self.explore(state)
                        if next_step == -1:
                            break
                        state = self.next_state(state, next_step)
                    else:
                        next_step, value = self.exploit(state)
                        if next_step == -1:
                            break

                        self.value_table[str(state)] += \
                            self.learning_rate * (value - self.value_table[str(state)])
                        state = self.next_state(state, next_step)
            print("")

    def test(self, test_epochs=10000):
        cnt_cross, cnt_circle, cnt_draw = 0, 0, 0
        for i in range(test_epochs):
            # get initial state, circle first
            board = [0 for _ in range(9)]
            board[random.randint(0, 8)] = -1
            state = State(board, 1)
            self.value_table[str(state)] = 0.5

            print("Test game %d: " % i, end="")

            while True:
                if self.result(state) == Model.CROSS_WIN:
                    self.value_table[str(state)] = 1
                    print("cross win", end=" ")
                    cnt_cross += 1
                    break
                elif self.result(state) == Model.CIRCLE_WIN:
                    self.value_table[str(state)] = 0
                    print("circle win", end=" ")
                    cnt_circle += 1
                    break
                elif self.result(state) == Model.DRAW:
                    self.value_table[str(state)] = 0.5
                    print("draw", end=" ")
                    cnt_draw += 1
                    break
                else:
                    print(str(state.board), end=" ")
                    next_step, value = self.exploit(state)
                    if next_step == -1:
                        break
                    state = self.next_state(state, next_step)
            print("")
        print("Cross win %d, Circle win %d, Draw %d" % (cnt_cross, cnt_circle, cnt_draw))
        print(repr(self.value_table))


model = Model(learning_rate=0.01, explore_rate=0.2, training_epoch=5000000)
model.train()
print("Start Testing ...")
model.test(10000)
