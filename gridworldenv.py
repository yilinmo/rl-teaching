import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

N_DISCRETE_ACTIONS = 4 # up, left, right, down
BOARD_ROWS = 3
BOARD_COLS = 4
OBSTACLE = np.array([1, 1], dtype=np.int8)
WIN_STATE = np.array([3, 0], dtype=np.int8)
START = np.array([0, 2], dtype=np.int8)

class GridworldEnv(gym.Env):
    """Custom Grid World Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, determined=False, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=np.array([BOARD_COLS-1, BOARD_ROWS-1]), dtype=np.int8)
        self.state= START 
        self.transition = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int8) # up, down, left, right
        self.determined = determined
        self.window_width = BOARD_COLS * 128
        self.window_height = BOARD_ROWS * 128

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

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

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state= START
        info = {}
        observation = self.state

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def render(self):

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        pix_square_size = 128 # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * WIN_STATE,
                (pix_square_size, pix_square_size),
            ),
        )
        # Next we draw the obstacle
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                pix_square_size * OBSTACLE,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.state + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Finally, add some gridlines
        for x in range(BOARD_ROWS + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_width, pix_square_size * x),
                width=3,
            )
        for x in range(BOARD_COLS + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
