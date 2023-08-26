import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = np.array([0, 3])
START = np.array([2, 0])

class State:
    def __init__(self, state=START, determine=True):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.size = np.array([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1 #obstacles
        self.state = state
        self.determine = determine
        self.transition = {
                'l' : np.array([0, -1]),
                'r' : np.array([0, 1]),
                'u' : np.array([-1, 0]),
                'd' : np.array([1, 0])
                }

    def isEnd(self):
        if (self.state == WIN_STATE).all():
            return True
        else: 
            return False

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        # if the random, the pick arbitrary action with 0.2 probability
        if self.determine == False and np.random.rand() <= 0.2:
            action = np.random.choice(['u', 'd', 'l', 'r'])

        newstate = self.state + self.transition[action]
        newstate = np.clip(newstate, (0,0), self.size-1)

        if self.board[tuple(newstate)] == 0.:
            self.state = newstate

        reward = -1.
        return reward 

    def showBoard(self):
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if  ([i,j] == self.state).all():
                    token = '*'
                elif ([i,j] == WIN_STATE).all():
                    token = 'E'
                elif self.board[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

    def reset(self):
        self.state = START

class Agent:
    def __init__(self):
        self.actions = ["u", "d", "l", "r"]
        self.State = State()
        self.lr = 0.7
        self.debug = True
        if self.debug == True:
            self.State.showBoard()
        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0.  # Q value is a dict of dict

    def showQ(self):
        for i in range(0, BOARD_ROWS):
            print('---------------------------------------------')
            out = '|'
            for j in range(0, BOARD_COLS):
                token = f'   {self.Q_values[(i,j)]["u"]:4.2f}    '
                out += token + '|'
            print(out)
            out = '|'
            for j in range(0, BOARD_COLS):
                token = f'{self.Q_values[(i,j)]["l"]:04.2f}  {self.Q_values[(i,j)]["r"]:04.2f}'
                out += token + '|'
            print(out)
            out = '|'
            for j in range(0, BOARD_COLS):
                token = f'   {self.Q_values[(i,j)]["d"]:04.2f}    '
                out += token + '|'
            print(out)
        print('---------------------------------------------')

    def step(self):
        action = np.random.choice(self.actions)
        current_state = self.State.state
        reward = self.State.nxtPosition(action)
        next_state = self.State.state
        TD = reward + max(self.Q_values[tuple(next_state)].values()) - self.Q_values[tuple(current_state)][action]
        self.Q_values[tuple(current_state)][action] += self.lr * TD  
        if self.State.isEnd() == True:
            self.State.reset()

        if self.debug == True:
            self.State.showBoard()
            self.showQ()
