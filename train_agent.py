from stable_baselines3 import PPO
import gymnasium as gym
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.monitor import Monitor
from simapp_cfg import simapp_cfg
from checkpoint_callback import CheckpointCallback

def train_agent():
    # Start the Simulation App
    simapp_cfg()

    # Create the environment
    env = Monitor(gym.make('FrankaGymEnv'))

    checkpoint_dir = Path("~/robotics-rl/checkpoints/").expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("~/robotics-rl/logs").expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the agent
    model = PPO(
        'MlpPolicy', 
        env,
        tensorboard_log=logs_dir
    )
    
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir)

    # Train the agent
    model.learn(total_timesteps=100000, callback=checkpoint_callback, progress_bar=True)
    
    # Save the model
    # model_path = os.path.join('saved_models', 'PPO_grid_model')
    # model.save(model_path)
    # print("Model saved to", model_path)

if __name__ == '__main__':
    train_agent()

