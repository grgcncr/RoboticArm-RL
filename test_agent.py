import os
from stable_baselines3 import PPO
from envs.franka_gym import FrankaGym
from stable_baselines3.common.monitor import Monitor
from simapp_cfg import simapp_cfg
import gymnasium as gym

def test_agent(model_path, num_episodes=100):
    # Start the Simulation App
    simapp_cfg()

    # Create the environment
    env = Monitor(gym.make('FrankaGymEnv'))
    # Load the trained model
    model = PPO.load(model_path)

    # Run the model for a number of episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # print("before",observation)
            observation, reward, terminated, truncated, info = env.step(action)
            print("after",observation)
            # env.render()
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == '__main__':
    model_path = '/home/mushroomgeorge/robotics-rl/checkpoints/model_100000_steps.zip'
    test_agent(model_path)
