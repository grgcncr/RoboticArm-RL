from stable_baselines3 import PPO
from envs.franka_gym import FrankaGym
import numpy as np
import os
import gymnasium as gym
def train_agent():
    # Create the environment
    env = gym.make('FrankaGymEnv')
    # obs, info = env.reset()
    # print("Initial observation:", [f"{x:.4f}" for x in obs.tolist()])
    # Instantiate the agent
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        n_steps=256,           # Steps per rollout
        batch_size=64,     # Batch size for training
        learning_rate=0.003,                 # Learning rate
        n_epochs=10,                        # Number of epochs per update
        gamma=0.99,                         # Discount factor
        gae_lambda=0.95,                    # GAE lambda
        clip_range=0.2,                     # PPO clipping range
    )
    # Train the agent
    model.learn(total_timesteps=1024*4)
    
    # while simulation_app.is_running():
    #     scene.write_data_to_sim()
    #     sim.step()
    #     env.reset()

    # Save the model
    # model_path = os.path.join('saved_models', 'PPO_grid_model')
    # model.save(model_path)
    # print("Model saved to", model_path)

    # Optionally, add code here to load and evaluate the model

    # print("--------------------Simulation Shutdown--------------------")
    # simulation_app.app.shutdown()

if __name__ == '__main__':
    train_agent()

