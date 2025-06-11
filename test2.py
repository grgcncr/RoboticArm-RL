import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Franka RL in Gymnasium")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import gym
from gym import spaces
from envs.gym_environment_interface import GymEnvironmentInterface
import time
import math
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf, Usd, UsdLux, UsdShade, Sdf, Tf, Vt, UsdPhysics, PhysxSchema
from isaacsim.core.utils.prims import get_prim_at_path
from omni.physx.scripts import physicsUtils
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from omni.physx.scripts import utils
from isaacsim.sensors.physics import ContactSensor
from isaacsim.sensors.physics import _sensor
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
import torch
import sys
import carb
import asyncio
import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.api.physics_context import PhysicsContext
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaacsim.core.prims import XFormPrim
import omni.kit.app
import omni.replicator.core as rep
from pathlib import Path
import os
from pxr import PhysxSchema
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.assets import RigidObject, RigidObjectCfg





def get_random_joint_velocities(scale_arm=100.0, scale_gripper=2.5):
    arm_vel = (np.random.rand(7) - 0.5) * 2 * scale_arm
    
    gripper_vel = np.random.uniform(-1, 1) * scale_gripper
    gripper = np.array([gripper_vel, gripper_vel])
    return np.concatenate([arm_vel, gripper])

def get_random_joint_positions():
    # Joint limits for Franka arm (in radians)
    joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_limits_upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    arm_positions = np.random.uniform(joint_limits_lower, joint_limits_upper)

    gripper_position = np.random.uniform(0.02, 0.05)
    gripper = np.array([gripper_position, gripper_position])
    return np.concatenate([arm_positions, gripper])

def print_contact(sensor_path):
        _contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        raw_data = _contact_sensor_interface.get_contact_sensor_raw_data(sensor_path)
        if raw_data.size > 0:
            body_name1 = _contact_sensor_interface.decode_body_name(raw_data[0][3])
            body_name0 = _contact_sensor_interface.decode_body_name(raw_data[0][2])
            impulse = raw_data[0][6]
            if body_name0 != "/World/defaultGroundPlane/GroundPlane/CollisionPlane" and body_name1 != "/World/defaultGroundPlane/GroundPlane/CollisionPlane":
                if body_name0 == "/World/Cube" or body_name1 == "/World/Cube":#and (impulse[0]>0 or impulse[1]>0 or impulse[2]>0):
                    # print(sensor.get_current_frame())
                    print(f"{body_name0} in contact with: {body_name1}, with impulse : {impulse}")
                    return True
                else:
                    return False
                
        else:
            # print(f"{part} has no contact.")
            return False

class FrankaGym(GymEnvironmentInterface):
    from isaacsim.sensors.physics import _sensor
    _contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
    
    from isaaclab.scene import InteractiveScene
    move_by_vel = False
    def step(self, action, scene: InteractiveScene):
        import omni.physx
        if self.move_by_vel:
            self.my_franka.set_joint_velocities(action)
        else:
            self.my_franka.set_joint_positions(action)

        
        # self.capture_images() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        observation = self.get_observation()
        collision_r = False
        collision_l = False
        collision_hand = False
        # collision_r = FrankaGym.print_contact(self.rsp)
        # collision_l = FrankaGym.print_contact(self.lsp)
        # collision_hand = FrankaGym.print_contact(self.hsp)
        # collision_r = FrankaGym.print_contact("/World/Cube/Contact_Sensor")
        # print(self.contact_sensor.data)
        print(scene[self.contact_sensor_rf])
        if (collision_l | collision_r | collision_hand):
            reward = -10.0
            done = True
        else:
            reward = self.calculate_reward(observation, done=False)
            done = False

        info = {"collision": (collision_l | collision_r | collision_hand)}
        self.sim.step()        
        return observation, reward, done, info


    def reset(self):
        """Reset environment"""
        self.sim.reset()
        self.my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
        self.my_franka.gripper.set_default_state(self.my_franka.gripper.joint_opened_positions)
        self.cube.set_local_poses(position=np.array([0.3, 0.3, 0.3]))
        return self.get_observation()
    
    def render(self, mode='human'):
        """Nothing needed since Isaac Sim is rendering automatically."""
        pass

    def calculate_reward(self, new_state, done):
        """Simple distance reward"""
        gripper_pos = self.my_franka.end_effector.get_local_pose()[0]
        cube_pos = self.cube.get_local_poses()[0]

        distance = np.linalg.norm(gripper_pos - cube_pos)
        reward = -distance
        return reward
    
    def get_observation(self):
        """Get observation: robot joints + cube position"""
        joint_pos = self.my_franka.get_joint_positions()
        cube_pos = self.cube.get_local_poses()[0].reshape(-1)
        observation = np.concatenate([joint_pos, cube_pos])
        return observation
    
    def close(self):
        self.simulation_app.close()

@configclass
class ConfigureScene(InteractiveSceneCfg):
    # PhysicsContext()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
    # args, unknown = parser.parse_known_args()

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # my_world = World(stage_units_in_meters=1.0)
    # my_world.scene.add_default_ground_plane()

    # asset_path = "/home/mushroomgeorge/robotics-rl/isaacsim assets/franka.usd"
    # add_reference_to_stage(usd_path=asset_path, prim_path="/World")
    # gripper = ParallelGripper(
    #     end_effector_prim_path="/World/franka/panda_rightfinger",
    #     joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    #     joint_opened_positions=np.array([0.05, 0.05]),
    #     joint_closed_positions=np.array([0.02, 0.02]),
    #     action_deltas=np.array([0.01, 0.01]),
    # )
    # my_franka = scene.add(
    #     SingleManipulator(
    #         prim_path="/World/franka", name="my_franka", end_effector_prim_name="panda_rightfinger", gripper=gripper
    #     )
    # )

    # robot = ArticulationCfg(
    #     prim_path="/World/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/Isaac/Robots/Franka/franka.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             enabled_self_collisions=True,
    #         ),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.0),  # Position in world
    #         rot=(1.0, 0.0, 0.0, 0.0),  # Rotation (quaternion)
    #         # Initialize joints to a good pose
    #         joint_pos={
    #             # Arm joints
    #             "panda_joint1": 0.0,
    #             "panda_joint2": -0.569,
    #             "panda_joint3": 0.0,
    #             "panda_joint4": -2.810,
    #             "panda_joint5": 0.0,
    #             "panda_joint6": 3.037,
    #             "panda_joint7": 0.741,
    #             # Gripper joints (open position)
    #             "panda_finger_joint1": 0.04,  # Left finger
    #             "panda_finger_joint2": 0.04,  # Right finger
    #         },
    #         actuators={"*": actuators.ImplicitActuatorCfg()},
    #     ),
    # )

    franka_robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka")

    print(dir(sim_utils))
    print("franka ok")
    # Small cube for grasping
    cube = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm cube - good size for Franka gripper
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # Light weight for easy manipulation
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # Red color
                metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.5),  # Position in front of robot
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    print("cube ok")
    # add_reference_to_stage(usd_path="/home/mushroomgeorge/robotics-rl/isaacsim assets/Cube.usd", prim_path="/World")
    # cube = scene.add(
    #     XFormPrim(
    #         "/World/Cube",
    #         name="cube"
    #     )
    # )
    # # cube.set_local_scales(np.array([[3.3, 3.3, 3.3]]))
    # cube.set_world_poses(np.array([[0.3, 0.3, 0.3]]))    

    # my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
    # my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)

    

    # stage = get_current_stage()
    # ground_path = "/World/defaultGroundPlane/GroundPlane/CollisionPlane"
    # ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
    # cube_prim = stage.GetPrimAtPath("/World/Cube")
    # franka_prim = stage.GetPrimAtPath("/World/franka")
    # rfinger_prim = stage.GetPrimAtPath("/World/franka/panda_rightfinger")
    # lfinger_prim = stage.GetPrimAtPath("/World/franka/panda_leftfinger")
    # hand_prim = stage.GetPrimAtPath("/World/franka/panda_hand")
    
    # scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    # scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    # scene.CreateGravityMagnitudeAttr().Set(981.0)
    




    # # print(dir(utils))
    # utils.setCollider(lfinger_prim, approximation="convexHull")
    # # utils.setRigidBody(lfinger_prim, approximationShape="convexDecomposition", kinematic=True)
    # utils.setCollider(rfinger_prim, approximation="convexHull")
    # # utils.setRigidBody(rfinger_prim, approximationShape="convexDecomposition", kinematic=True)
    # utils.setCollider(hand_prim, approximation="convexHull")
    # # utils.setRigidBody(hand_prim, approximationShape="convexDecomposition", kinematic=True)
    # utils.setCollider(cube_prim)
    # utils.setRigidBody(cube_prim, approximationShape="boundingCube", kinematic=False)
    
    
    contact_sensor_rf = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Franka/",
        update_period=0.0,
        history_length=6,
        debug_vis=True
        # filter_prim_paths_expr=["/World/Cube"],
    )
    print("sensor ok")
    # contact_sensor = ContactSensor(contact_sensor_rf)
    # my_world.scene.add(contact_sensor)
    # from isaaclab.sensors.contact_sensor import ContactSensor
    # print(dir(contact_sensor))
    # mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
    # mass_api.CreateMassAttr(10.0)
    # rigid_cube = UsdPhysics.RigidBodyAPI.Apply(cube_prim)
    # rigid_cube.CreateRigidBodyEnabledAttr(True)
    # rigid_cube.CreateKinematicEnabledAttr(False)
    
    rsp = "/World/franka/panda_rightfinger/Contact_Sensor"
    # rfinger_sensor = ContactSensor(
    #     prim_path=rsp,
    #     name="Contact_Sensor_r",
    #     frequency=60,
    #     translation=np.array([0, 0, 0]),
    #     min_threshold=0.1,
    #     max_threshold=10000000,
    #     radius=-1
    # )

    lsp = "/World/franka/panda_leftfinger/Contact_Sensor"
    # lfinger_sensor = ContactSensor(
    #     prim_path=lsp,
    #     name="Contact_Sensor_l",
    #     frequency=60,
    #     translation=np.array([0, 0, 0]),
    #     min_threshold=0.1,
    #     max_threshold=10000000,
    #     radius=-1
    # )

    hsp = "/World/franka/panda_hand/Contact_Sensor"
    # hand_sensor = ContactSensor(
    #     prim_path=hsp,
    #     name="Contact_Sensor_hand",
    #     frequency=60,
    #     translation=np.array([0, 0, 0]),
    #     min_threshold=0.1,
    #     max_threshold=10000000,
    #     radius=-1
    # )

    # cube_sensor = ContactSensor(
    #     prim_path="/World/Cube/Contact_Sensor",
    #     name="Contact_Sensor_cube",
    #     frequency=60,
    #     translation=np.array([0, 0, 0]),
    #     min_threshold=0.1,
    #     max_threshold=10000000,
    #     radius=-1
    # )

    # stage = omni.usd.get_context().get_stage()
    
    # UsdPhysics.CollisionAPI.Apply(cube_prim)
    # UsdPhysics.CollisionAPI.Apply(hand_prim)
    # UsdPhysics.CollisionAPI.Apply(rfinger_prim)
    # UsdPhysics.CollisionAPI.Apply(lfinger_prim)
    
    # UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
    # UsdPhysics.MeshCollisionAPI.Apply(hand_prim)
    # UsdPhysics.MeshCollisionAPI.Apply(rfinger_prim)
    # UsdPhysics.MeshCollisionAPI.Apply(lfinger_prim)

    # filtered_api = UsdPhysics.FilteredPairsAPI.Apply(cube_prim)
    # filtered_api.CreateFilteredPairsRel().AddTarget(Sdf.Path(ground_path))
    # cube_collision = PhysxSchema.PhysxCollisionAPI.Apply(cube_prim)
    # hand_collision = PhysxSchema.PhysxCollisionAPI.Apply(hand_prim)
    # rfinger_colliion = PhysxSchema.PhysxCollisionAPI.Apply(rfinger_prim)
    # lfinger_collision = PhysxSchema.PhysxCollisionAPI.Apply(lfinger_prim)

    # contact_report_cube = PhysxSchema.PhysxContactReportAPI.Apply(cube_prim)
    # contact_report_rf = PhysxSchema.PhysxContactReportAPI.Apply(rfinger_prim)
    # contact_report_lf = PhysxSchema.PhysxContactReportAPI.Apply(lfinger_prim)
    # contact_report_hand = PhysxSchema.PhysxContactReportAPI.Apply(hand_prim)

    # contact_report_cube.CreateThresholdAttr(0.1)
    # contact_report_rf.CreateThresholdAttr(0.1)
    # contact_report_lf.CreateThresholdAttr(0.1)
    # contact_report_hand.CreateThresholdAttr(0.1)


    # cube_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("boundingCube")
    # hand_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    # rfinger_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    # lfinger_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")







    # arm_path = "/World/franka/panda_hand/arm_camera"
    # arm_camera = UsdGeom.Camera.Define(stage, arm_path)
    # arm_camera.AddTranslateOp().Set(Gf.Vec3f(-0.011052506998599807, 0.002351993160557201, -1.6135926818935435))
    # arm_camera.AddOrientOp().Set(Gf.Quatf(0.0, 0.7, 0.7, 0.0))

    # static_path = "/World/static_camera"
    # static_camera = UsdGeom.Camera.Define(stage, static_path)
    # static_camera.AddTranslateOp().Set(Gf.Vec3f(-0.025128380734830664, 6.88345, 0.735964839641152))
    # static_camera.AddOrientOp().Set(Gf.Quatf(-0.00355, -0.005, 0.6779, 0.73513))

    # arm_camera_prim = stage.GetPrimAtPath(arm_path)
    # static_camera_prim = stage.GetPrimAtPath(static_path)

    # if not arm_camera_prim.IsValid() or not static_camera_prim.IsValid():
    #     print("Camera prim(s) not valid!")

    # arm_rp = rep.create.render_product(arm_path, resolution=(640, 480))
    # static_rp = rep.create.render_product(static_path, resolution=(640, 480))

    # writer = rep.WriterRegistry.get("BasicWriter")

    # output_dir = Path(os.path.expanduser("~/robotics-rl/camera_output"))
    # output_dir.mkdir(exist_ok=True)

    # writer.initialize(output_dir=str(output_dir), rgb=True)

def capture_images(self):
    self.writer.attach([self.arm_rp, self.static_rp])
    for _ in range(3):
        self.my_world.step(render=True)
    self.writer.detach()

def simulation_run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    env = FrankaGym()
    i=0
    toggle_interval = 0.2
    last_toggle_time = time.time()
    # Main Loop
    while env.simulation_app.is_running():
        sim.step()

        if env.my_world.is_stopped():
            env.simulation_app.close()

        if env.my_world.is_playing():

            current_time = time.time()
            if current_time - last_toggle_time > toggle_interval:
                print("Repetition",i+1)
                action = get_random_joint_velocities() if env.move_by_vel else get_random_joint_positions()
                obs, reward, done, info = env.step(action)
                i+=1
                # print(obs)
                # print(obs[9],obs[10],obs[11])
                last_toggle_time = current_time

def main():
    # try:
    #     for i in range(900):
    #         print("Repetition",i+1)
    #         action = get_random_joint_velocities() if env.move_by_vel else get_random_joint_positions()
    #         obs, reward, done, info = env.step(action)
    #         # print(f"Reward: {reward}\nStep {i}\n{info}")
            
    #         time.sleep(0.01)
    # finally:
    #     env.close()

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((1.5, 2.0, 1.5), (0.0, 0.0, 0.0))
    print(dir(sim))
    # sim_context = SimulationContext()
    scene_cfg = ConfigureScene(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(dir(scene))
    print("--------Setup Copmlete--------")
    sim.reset()
    simulation_run(sim,scene)

if __name__ == "__main__":
    main()
    simulation_app.close()