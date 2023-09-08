import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_DISCRETE_ACTIONS = 4 # up, left, right, down
BOARD_ROWS = 3
BOARD_COLS = 4
OBSTACLE = np.array([1, 1], dtype = np.int8)
WIN_STATE = np.array([0, 3], dtype = np.int8)
START = np.array([2, 0], dtype=np.int8)

class GridworldEnv(gym.Env):
    """Custom Grid World Environment that follows gym interface."""

    def __init__(self, determined=False):
        super().__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=np.array([BOARD_ROWS-1, BOARD_COLS-1]), dtype=np.int8)
        self.state= START 
        self.transition = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int8)
        self.determined = determined

    def step(self, action):
        if self.determined == False and np.random.rand() <= 0.2:
            action = self.action_space.sample()
        newstate = self.state + self.transition[action,:]

        # avoid going out of the board or into the obstacles
        if self.observation_space.contains(newstate) and np.array_equal(newstate, OBSTACLE) == False:
            self.state = newstate
        observation = self.state
        reward = -1.
        if np.array_equal(self.state, WIN_STATE):
            terminated = True
        else:
            terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state= START
        info = {}
        observation = self.state
        return observation, info

    def close(self):
        return


from stable_baselines3 import A2C
env = GridworldEnv(determined=False)
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10000)
