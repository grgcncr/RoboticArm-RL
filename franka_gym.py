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

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObject, RigidObjectCfg
import numpy as np
import time
import os
from pathlib import Path
import imageio.v2 as imageio
from envs.gym_environment_interface import GymEnvironmentInterface
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    FRANKA_PANDA_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },# prev start pos (0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05)
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
        prim_path = "{ENV_REGEX_NS}/Franka",
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka")
    
    cube = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001,rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
                metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.025), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    hand_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Franka/panda_hand/hand_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.7), 
            rot=(0.70711, 0.0, 0.0, 0.70711),  # (x,w,z,y))!!
            convention="ros",
        ),
    )

    side_camera = CameraCfg(
        prim_path="/World/side_camera",  
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 3.5, 0.5),  
            rot=(0.0, 0.0, 0.70711, -0.70711), 
        ),
    )

    
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Franka/.*",
        filter_prim_paths_expr=["/World/Cube"],
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )

class FrankaGym(GymEnvironmentInterface):
    move_by_vel = False
    def step(sim: sim_utils.SimulationContext, scene: InteractiveScene, rep):
        sim_dt = sim.get_physics_dt()        
        franka = scene["robot"]
        contact_sensor = scene["contact_sensor"]
        cube = scene["cube"]
        
        joint_pos = get_random_joint_positions()
        joint_vel = get_random_joint_velocities()
        franka.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Capture images for every repetition
        # capture_images(scene,rep,"side_camera", "~/robotics-rl/camera_output/side_camera") # <---------------- uncomment
        # capture_images(scene,rep,"hand_camera", "~/robotics-rl/camera_output/hand_camera") # <---------------- uncomment

        max_force = torch.max(contact_sensor.data.net_forces_w).item()
        if max_force != 0.0:
            print("Received max contact force of :",max_force)
        force_magnitudes = torch.norm(contact_sensor.data.force_matrix_w, dim=-1)
        contact_detected = (force_magnitudes > 0.1).any().item()
        if contact_detected:
            time_point = time.time() + 4
            print(time.time())
            print(time_point)            
            while time.time() < time_point:{"print(time.time())"}
            print("--------------------Franka Touched the Cube--------------------")
            #time.sleep(5.0)
            # FrankaGym.close()
            reward = -10.0
            done = True
        else:
            reward = FrankaGym.calculate_reward(scene)
            done = False

        info = {"Cube collision": contact_detected}
         
        observation = FrankaGym.get_observation(scene)       
        return observation, reward, done, info


    def reset(sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Reset environment"""
        sim.reset()
        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene["cube"].write_root_com_pose_to_sim() # default_root_state 
        return FrankaGym.get_observation(scene)
    
    def render(self, mode='human'):
        """Nothing needed since Isaac Sim is rendering automatically."""
        pass

    # def calculate_reward(self, new_state, done):
    #     """Simple distance reward"""
    #     gripper_pos = self.my_franka.end_effector.get_local_pose()[0]
    #     cube_pos = self.cube.get_local_poses()[0]

    #     distance = np.linalg.norm(gripper_pos - cube_pos)
    #     reward = -distance
    #     return reward
    
    def calculate_reward(scene: InteractiveScene) -> torch.Tensor:
        franka = scene["robot"]
        cube = scene["cube"]
        fdata = franka.data

        link_names = fdata.body_names
        idx_left = link_names.index("panda_leftfinger")
        idx_right = link_names.index("panda_rightfinger")

        pos_left = fdata.body_link_pos_w[:, idx_left]
        pos_right = fdata.body_link_pos_w[:, idx_right]
        gripper_center = (pos_left + pos_right) / 2
        cube_pos = cube.data.body_pos_w[:, 0]  # only one body
        
        distance = torch.norm(gripper_center - cube_pos, dim=-1).cpu().numpy().reshape(-1)
        reward = -distance

        return reward

    def get_observation(scene: InteractiveScene):
        # Get observation: robot joints + cube position
        joint_pos = scene["robot"].data.joint_pos.cpu().numpy().reshape(-1)
        cube_pos = scene["cube"].data.body_pos_w.cpu().numpy().reshape(-1)
        observation = np.concatenate([joint_pos, cube_pos])
        return observation
    
    def close():
        print("--------------------Simulation Shutdown--------------------")
        simulation_app.app.shutdown()

def capture_images(scene: InteractiveScene, rep, camera, path):
    rgb_tensor = scene[camera].data.output["rgb"]
    rgb_np = rgb_tensor.squeeze(0).cpu().numpy()  
    save_dir = Path(path).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    image_path = save_dir / f"camera_output_{rep}.png"
    imageio.imwrite(str(image_path), rgb_np)
    # print(f"--------Saved {image_path}--------")
    # # For 10 images
    # if rep == 10:
    #     FrankaGym.close()
    


def get_random_joint_velocities(scale_arm=100.0, scale_gripper=2.5):
    arm_vel = (np.random.rand(7) - 0.5) * 2 * scale_arm
    gripper_vel = np.random.uniform(-1, 1) * scale_gripper
    gripper = np.array([gripper_vel, gripper_vel])
    joint_vel = np.concatenate([arm_vel, gripper])
    return torch.tensor(joint_vel, dtype=torch.float32, device="cuda:0").unsqueeze(0)

def get_random_joint_positions():
    joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_limits_upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    
    arm_positions = np.random.uniform(joint_limits_lower, joint_limits_upper)
    gripper_position = np.random.uniform(0.02, 0.05)
    gripper = np.array([gripper_position, gripper_position])
    joint_pos = np.concatenate([arm_positions, gripper])
    return torch.tensor(joint_pos, dtype=torch.float32, device="cuda:0").unsqueeze(0)

def simulation_run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    rep = 0
    # franka = scene["robot"] 
    # contact_sensor = scene["contact_sensor"]
    # cube = scene["cube"]
    
    # Main Loop 
    while simulation_app.is_running():
        count += 1
        if count % 20 == 0:
            rep += 1
            print("Repetition :",rep)
            FrankaGym.step(sim,scene,rep)
        else:
            scene.write_data_to_sim()
            sim.step()
            sim_time += sim_dt
            scene.update(sim_dt)

        
def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 2.0, 1.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("----------------Setup complete----------------")
    # Run the simulation
    simulation_run(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
