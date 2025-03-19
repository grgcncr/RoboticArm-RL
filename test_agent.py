import os
from stable_baselines3 import PPO
from envs.simple_grid_env import SimpleGridEnv

def test_agent(model_path, num_episodes=10):
    # Load the environment
    env = SimpleGridEnv()

    # Load the trained model
    model = PPO.load(model_path)

    # Run the model for a number of episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == '__main__':
    model_path = 'saved_models/PPO_grid_model'
    test_agent(model_path)
