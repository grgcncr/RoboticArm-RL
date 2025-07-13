from stable_baselines3 import PPO
import gymnasium as gym
from pathlib import Path
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
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        tensorboard_log=logs_dir,
    )
    
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(save_freq=1024 * 100, save_path=checkpoint_dir)

    # Train the agent
    model.learn(total_timesteps=1024 * 1000, callback=checkpoint_callback, progress_bar=True)
    env.close()
    # Save the model
    # model_path = os.path.join('saved_models', 'PPO_grid_model')
    # model.save(model_path)
    # print("Model saved to", model_path)

if __name__ == '__main__':
    train_agent()

