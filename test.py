from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view


parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()


assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
set_camera_view((1.5, 2.0, 1.5), (0.0, 0.0, 0.0))


asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.02, 0.02]),
    action_deltas=np.array([0.01, 0.01]),
)
my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka", name="my_franka", end_effector_prim_name="panda_rightfinger", gripper=gripper
    )
)

cube = my_world.scene.add(
    DynamicCuboid(
        name="cube",
        position=np.array([0.3, 0.3, 0.3]),
        prim_path="/World/Cube",
        scale=np.array([0.0515, 0.0515, 0.0515]),
        size=1.0,
        color=np.array([0, 0, 1]),
    )
)
my_world.scene.add_default_ground_plane()
my_world.reset()

my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)

my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()

import time
import numpy as np

toggle_interval = 3.5 
last_toggle_time = time.time()

while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_stopped():
        simulation_app.close()
    
    
    current_time = time.time()
    if current_time - last_toggle_time > toggle_interval:
        
        print("Franka joints position : ",my_franka.get_joint_positions(),"\n" )
        print("Gripper position : ",my_franka.gripper.get_joint_positions(),"\n")
        print("Cube position : ",cube.get_local_pose()[0],"\n")
   
        
        joint_positions = my_franka.gripper.get_joint_positions()[-2:]
        if joint_positions[0] > 0.03 and joint_positions[1] > 0.03:
            my_franka.gripper.close()  
        else:
            my_franka.gripper.open() 
        
        last_toggle_time = current_time


