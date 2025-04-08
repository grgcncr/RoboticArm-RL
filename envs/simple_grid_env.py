import numpy as np
import gym
from gym import spaces
from gym_environment_interface import GymEnvironmentInterface

class SimpleGridEnv(gym.Env, GymEnvironmentInterface):
    def __init__(self):
        super(SimpleGridEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.float32)
        self.goal_position = (4, 4)
        self.state = None

    def reset(self):
        self.state = (0, 0)
        return self._get_observation()

    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:  # Move up
            x -= 1
        elif action == 1 and x < 4:  # Move down
            x += 1
        elif action == 2 and y > 0:  # Move left
            y -= 1
        elif action == 3 and y < 4:  # Move right
            y += 1

        self.state = (x, y)
        done = self.state == self.goal_position
        reward = self.calculate_reward(self.state, done)
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros(self.observation_space.shape)
        grid[self.state] = 1
        grid[self.goal_position] = 0.5
        print(grid)

    def calculate_reward(self, new_state, done):
        if done:
            return 10  # Reward for reaching the goal
        return -1  # Penalty for each step taken

    def _get_observation(self):
        grid = np.zeros(self.observation_space.shape)
        grid[self.state] = 1
        return grid
