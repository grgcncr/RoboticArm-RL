from stable_baselines3 import PPO
import gymnasium as gym
from pathlib import Path
import torch
from stable_baselines3.common.monitor import Monitor
from simapp_cfg import simapp_cfg
from custom_callback import CustomCallback

def train_agent():
    # Start the Simulation App
    simapp_cfg()

    # Create the environment
    env = Monitor(gym.make('FrankaGymEnv'))

    # Initialize save directories 
    checkpoint_dir = Path("~/robotics-rl/checkpoints/").expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("~/robotics-rl/logs").expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.002,                  
        # clip_range=0.1,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]),
        tensorboard_log=logs_dir,
    )

    # model = PPO.load('/home/mushroomgeorge/robotics-rl/checkpoints/model_1843200_steps.zip', env=env)
    
    # Custom Callback --> saves checkpoints, alternates and logs learing rate, logs success rate
    custom_callback = CustomCallback(save_freq=1024 * 1000, save_path=checkpoint_dir, initial_lr=0.0003)

    # Train the agent
    model.learn(total_timesteps=1024 * 10000, callback=custom_callback, progress_bar=True)
    env.close()

if __name__ == '__main__':
    train_agent()




#     model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0003,           # Higher for robotic tasks
#     n_steps=2048,                 # Good for your setup
#     batch_size=256,               # Larger for stability
#     n_epochs=5,                   # Conservative to avoid overfitting
#     gamma=0.995,                  # Higher for long-horizon tasks
#     gae_lambda=0.95,              # Good default
#     target_kl=0.02,              # Prevent policy collapse
#     ent_coef=0.01,               # Maintain exploration
#     vf_coef=0.5,                 # Balance value/policy learning
#     clip_range=0.2,              # Standard for manipulation
#     max_grad_norm=0.5,           # Gradient clipping for stability
#     policy_kwargs=dict(
#         net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # Larger networks
#         activation_fn=torch.nn.ReLU,
#     ),
#     tensorboard_log=logs_dir,
# )