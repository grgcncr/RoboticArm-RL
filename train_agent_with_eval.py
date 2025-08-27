from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path
import torch
import gymnasium as gym

from simapp_cfg import simapp_cfg
from custom_callback import CustomCallback


def make_env():
    """Utility to create a monitored Franka env (for parallelization)."""
    return Monitor(gym.make("FrankaGymEnv"))


def train_agent():
    # Start the simulation app
    simapp_cfg()

    checkpoint_dir = Path("~/robotics-rl/checkpoints/").expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("~/robotics-rl/logs").expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    eval_env = make_env()

    custom_policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.002,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=custom_policy_kwargs,
        tensorboard_log=logs_dir,
    )

    custom_callback = CustomCallback(
        save_freq=1024 * 1000,
        save_path=checkpoint_dir,
        initial_lr=3e-4,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=logs_dir,
        eval_freq=100000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=1024 * 10000,
        callback=[custom_callback, eval_callback],
        progress_bar=True,
    )

    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_agent()
