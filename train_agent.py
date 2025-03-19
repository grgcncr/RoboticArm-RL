from stable_baselines3 import PPO
from envs.simple_grid_env import SimpleGridEnv
import os

def train_agent():
    # Create the environment
    env = SimpleGridEnv()

    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=20000)

    # Save the model
    model_path = os.path.join('saved_models', 'PPO_grid_model')
    model.save(model_path)
    print("Model saved to", model_path)

    # Optionally, add code here to load and evaluate the model

if __name__ == '__main__':
    train_agent()
