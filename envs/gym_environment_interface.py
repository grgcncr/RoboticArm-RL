from abc import ABC, abstractmethod
import gym
from gym import spaces
import numpy as np

class GymEnvironmentInterface(ABC):
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode='human'):
        pass

    @abstractmethod
    def calculate_reward(self, new_state, done):
        pass
