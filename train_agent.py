
def train_agent():
    import argparse

    from isaaclab.app import AppLauncher

    # add argparse arguments
    parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from stable_baselines3 import PPO
    from envs.franka_gym import FrankaGym
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from set_up_cfg import SceneCfg
    import torch
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 2.0, 1.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # franka = scene["robot"]
    # init_pos_target = torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.00, 0.00], dtype=torch.float32, device="cuda:0").unsqueeze(0)
    # franka.write_joint_position_to_sim(init_pos_target)
    # franka.write_data_to_sim()
    # scene.write_data_to_sim()
    # sim.step()
    # scene.update(sim.get_physics_dt())
    # Now we are ready!
    print("----------------Setup complete----------------")
    # Create the environment
    env = FrankaGym(sim,scene)
    obs = env.reset()
    print("Initial observation:", [f"{x:.4f}" for x in obs])
    # Instantiate the agent
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        n_steps=256,           # Steps per rollout
        batch_size=64,     # Batch size for training
        learning_rate=3e-4,                 # Learning rate
        n_epochs=10,                        # Number of epochs per update
        gamma=0.99,                         # Discount factor
        gae_lambda=0.95,                    # GAE lambda
        clip_range=0.2,                     # PPO clipping range
    )



    # Train the agent
    model.learn(total_timesteps=1024)
    
    # Save the model
    # model_path = os.path.join('saved_models', 'PPO_grid_model')
    # model.save(model_path)
    # print("Model saved to", model_path)

    # Optionally, add code here to load and evaluate the model

    # print("--------------------Simulation Shutdown--------------------")
    # simulation_app.app.shutdown()

if __name__ == '__main__':
    train_agent()

