def simapp_cfg():
    # Start the Simulation App
    import argparse

    from isaaclab.app import AppLauncher

    # add argparse arguments
    parser = argparse.ArgumentParser(description="Gymnasium Enviroment for Franka trainning")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from scene_cfg import SceneCfg
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

    from gymnasium.envs.registration import register
    register(
        id='FrankaGymEnv',
        entry_point=lambda: 
        __import__(
            'envs.franka_gym', 
            fromlist=['FrankaGym']
            ).FrankaGym(app_launcher.app, sim, scene)
    )
    register(
        id='FrankaGymEnvLvl2',
        entry_point=lambda: 
        __import__(
            'envs.franka_gym_lvl2', 
            fromlist=['FrankaGymLvl2']
            ).FrankaGymLvl2(app_launcher.app, sim, scene)
    )