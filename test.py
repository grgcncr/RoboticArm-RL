from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import carb
import numpy as np
import asyncio
import time
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view

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
add_reference_to_stage(usd_path="/home/mushroomgeorge/Desktop/franka.usd", prim_path="/World")
gripper = ParallelGripper(
    end_effector_prim_path="/World/franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.02, 0.02]),
    action_deltas=np.array([0.01, 0.01]),
)
my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/franka", name="my_franka", end_effector_prim_name="panda_rightfinger", gripper=gripper
    )
)

# from isaacsim.core.utils.stage import open_stage
# open_stage("/home/mushroomgeorge/Desktop/Cube.usd")



# cube = my_world.scene.add(
#     DynamicCuboid(
#         name="cube",
#         position=np.array([0.3, 0.3, 0.3]),
#         prim_path="/World/Cube",
#         scale=np.array([0.0515, 0.0515, 0.0515]),
#         size=1.0,
#         color=np.array([0, 0, 1]),
#     )
# )
add_reference_to_stage(usd_path="/home/mushroomgeorge/Desktop/Cube.usd", prim_path="/World")
from isaacsim.core.prims import XFormPrim
cube = my_world.scene.add(
    XFormPrim(
        "/World/Cube",
        name="cube"
    )
)
cube.set_local_scales(np.array([[1.3, 1.3, 1.3]]))
cube.set_world_poses(np.array([[0.3, 0.3, 0.3]]))

my_world.scene.add_default_ground_plane()
my_world.reset()

# my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
# my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
# my_franka.enable_rigid_body_physics()
# my_franka.gripper.enable_rigid_body_physics()

from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf

stage = get_current_stage()

arm_path = "/World/franka/panda_hand/arm_camera"
arm_camera = UsdGeom.Camera.Define(stage, arm_path)
arm_camera.AddTranslateOp().Set(Gf.Vec3f(-0.011052506998599807, 0.002351993160557201, -1.6135926818935435))
arm_camera.AddOrientOp().Set(Gf.Quatf(0.0, 0.7, 0.7, 0.0))

static_path = "/World/static_camera"
static_camera = UsdGeom.Camera.Define(stage, static_path)
static_camera.AddTranslateOp().Set(Gf.Vec3f(-0.025128380734830664, 6.88345, 0.735964839641152))
static_camera.AddOrientOp().Set(Gf.Quatf(-0.00355, -0.005, 0.6779, 0.73513))

import omni.kit.app
import omni.replicator.core as rep
from pathlib import Path
import os

arm_camera_prim = stage.GetPrimAtPath(arm_path)
static_camera_prim = stage.GetPrimAtPath(static_path)

if not arm_camera_prim.IsValid() or not static_camera_prim.IsValid():
    print("Camera prim(s) not valid!")

arm_rp = rep.create.render_product(arm_path, resolution=(640, 480))
static_rp = rep.create.render_product(static_path, resolution=(640, 480))

writer = rep.WriterRegistry.get("BasicWriter")

output_dir = Path(os.path.expanduser("~/robotics-rl/camera_output"))
output_dir.mkdir(exist_ok=True)

writer.initialize(output_dir=str(output_dir), rgb=True)
app = omni.kit.app.get_app()

# import threading
# import queue

# save_queue = queue.Queue()

# def save_images_worker():
#     while True:
#         render_products = save_queue.get()
#         if render_products is None:
#             break
#         writer.attach(render_products)
#         save_queue.task_done()

# save_thread = threading.Thread(target=save_images_worker, daemon=True)
# save_thread.start()


def get_random_joint_velocities(scale_arm=5.0, scale_gripper=2.5):
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

# async def capture_camera_images():
#     writer.attach([arm_rp, static_rp])
#     await asyncio.sleep(0.09)
#     writer.detach()
#     print("Captured Image")


capture_count = 0
toggle_interval = 3
last_toggle_time = time.time()
move_by_vel = True
# Main Loop
while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_stopped():
        simulation_app.close()
    
    if my_world.is_playing():

        current_time = time.time()
        if current_time - last_toggle_time > toggle_interval:
            # Use velocirties or positions
            # if move_by_vel:
            #     my_franka.set_joint_velocities(get_random_joint_velocities())
            # else:
            #     my_franka.set_joint_positions(get_random_joint_positions())
            
            # print("Franka joints position : ",my_franka.get_joint_positions(),"\n" )
            # print("Gripper position : ",my_franka.gripper.get_joint_positions(),"\n")
            # print("Cube position : ",cube.get_local_poses()[0],"\n")
            
            # writer.attach([arm_rp, static_rp])
            # for _ in range(2):  # Frames
            #     app.update()
            #     time.sleep(0.01)
            # writer.detach()

            # asyncio.ensure_future(capture_camera_images())
            # save_queue.put([arm_rp, static_rp])
            capture_count += 1
            print(f"Iteration number {capture_count}")  
            last_toggle_time = current_time

# save_queue.put(None)
# save_thread.join()
