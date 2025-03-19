from abc import ABC, abstractmethod

class GymEnvironmentInterface(ABC):
    """
    This abstract base class defines the standard structure for custom gym environments.
    It ensures that all environments implement the necessary methods required by 
    reinforcement learning algorithms.
    """

    @abstractmethod
    def step(self, action):
        """
        Apply an action to the environment. This method is a critical part of the environment's
        interface, handling the logic for state transitions and determining the outcomes of actions
        taken by an agent.

        Parameters:
        action (int): The action taken by the agent.

        Returns:
        tuple: A tuple containing:
            - observation (object): The new state of the environment after the action.
            - reward (float): The amount of reward returned after transitioning to the new state.
            - done (bool): Whether the episode has ended, in which case further step() calls should be avoided.
            - info (dict): Diagnostic information useful for debugging. It can sometimes
                           also be used for learning (e.g., might contain the raw probabilities
                           behind the environment's last state change).
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to an initial state and return an initial observation.

        Returns:
        object: The initial observation that agents use to learn and make decisions from.
        """
        pass

    @abstractmethod
    def render(self, mode='human'):
        """
        Render one frame of the environment (optional). The visual representation
        of the environment after an action is taken.

        Parameters:
        mode (str): The mode to render with. Common modes are:
                    - 'human': render to the current display or terminal and
                      return nothing. Usually for human consumption.
                    - 'rgb_array': Return an numpy.ndarray with shape (x, y, 3),
                      representing RGB values for an x-by-y pixel image, suitable
                      for turning into a video.

        Returns:
        None or numpy.ndarray: Depending on the rendering mode.
        """
        pass

    @abstractmethod
    def calculate_reward(self, new_state, done):
        """
        Calculate the reward received after transitioning to a new state. This method
        allows for the customization of the reward structure depending on the specifics
        of the state transition and whether the state is terminal.

        Parameters:
        new_state (object): The new state after the action is taken.
        done (bool): Whether the episode has ended.

        Returns:
        float: The reward for the transition, which may be positive for achieving the goal,
               negative for making undesirable decisions, or zero.
        """
        pass
