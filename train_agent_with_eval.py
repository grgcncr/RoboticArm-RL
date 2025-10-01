from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path
import gymnasium as gym
from simapp_cfg import simapp_cfg
from custom_callback import CustomCallback


def make_env():
    # return Monitor(gym.make("FrankaGymEnv"))
    return Monitor(gym.make("FrankaGymEnvLvl2"))

def train_agent():
    # Start the simulation app
    simapp_cfg()

    checkpoint_dir = Path("~/robotics-rl/checkpoints/").expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("~/robotics-rl/logs").expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    # custom_policy_kwargs = dict(
    #     net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])],
    # )

    # model = PPO(
    #     "MlpPolicy",
    #     train_env,
    #     learning_rate=0.0003,
    #     n_steps=2048,
    #     batch_size=128,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     ent_coef=0.002,
    #     policy_kwargs=custom_policy_kwargs,
    #     tensorboard_log=logs_dir,
    # )

    # from stable_baselines3 import SAC

    # model = SAC(
    #     "MlpPolicy",
    #     train_env,
    #     learning_rate=0.0003,
    #     buffer_size=1000000,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     ent_coef='auto',  # Automatic entropy tuning
    #     policy_kwargs=dict(net_arch=[256, 256]),
    #     tensorboard_log=logs_dir,
    # )
    
    td3_policy_kwargs = dict(
        net_arch=[256, 256, 128],
    )

    from stable_baselines3 import TD3

    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=0.0005,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        # train_freq=1,
        # gradient_steps=1,
        policy_kwargs=td3_policy_kwargs,
        tensorboard_log=logs_dir,
    )

    # model = TD3.load('/home/mushroomgeorge/robotics-rl/checkpoints/td3_10_6144000_steps.zip', env=train_env)

    custom_callback = CustomCallback(
                       # 1000
        save_freq=1024 * 10,
        save_path=checkpoint_dir,
        initial_lr=0.0003,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=logs_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
                             # 10000
        total_timesteps=1024 * 10000,
        callback=[custom_callback, eval_callback],
        # reset_num_timesteps=False, # <---
        progress_bar=True,
    )

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    train_agent()
