import random
from envs.simple_grid_env import SimpleGridEnv

def test_random_agent(num_episodes=10):
    # Load the environment
    env = SimpleGridEnv()
    
    # Run the random agent for a number of episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()  # Take a random action
            obs, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == '__main__':
    test_random_agent()
