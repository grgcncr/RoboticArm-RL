import os
from stable_baselines3 import PPO
from envs.franka_gym import FrankaGym
from stable_baselines3.common.monitor import Monitor
from simapp_cfg import simapp_cfg
import gymnasium as gym

def test_agent(num_episodes=100):
    # Start the Simulation App
    simapp_cfg()

    # Create the environment
    env = Monitor(gym.make('FrankaGymEnv'))
    # Load the trained model
    model = PPO.load('/home/mushroomgeorge/robotics-rl/checkpoints/model64_1024000_steps.zip', env=env)

    # Run the model for a number of episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        while not done:
            # Use deterministic=True for consistent testing
            action, _states = model.predict(observation=obs, deterministic=True)
             
            # DEBUG: Print action and current observation (first few steps only to avoid spam)
            if step_count < 5:
                print(f"Episode {episode+1}, Step {step_count}: Action = {action}")
                print(f"Episode {episode+1}, Step {step_count}: Obs = {obs}")
            # print("action",action)
            obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            # print("action:", action)
            # print("next observation:", obs)
            # print("reward:", reward)
            # print("done:", done)
            total_reward += reward
            done = terminated or truncated
            step_count += 1
        print(f"\n\nEpisode {episode + 1}: Total Reward = {total_reward}\n\n")
    env.close()

if __name__ == '__main__':
    test_agent()
