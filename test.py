import math
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import sys
import random
import carb
import numpy as np
import asyncio
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
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
# my_franka.enable_rigid_body_physics()
my_franka.gripper.enable_rigid_body_physics()

from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as R


stage = get_current_stage()

arm_path = "/World/Franka/panda_hand/arm_camera"
arm_camera = UsdGeom.Camera.Define(stage, arm_path)
arm_camera.AddTranslateOp().Set(Gf.Vec3f(-0.011052506998599807, 0.002351993160557201, -1.6135926818935435))
arm_camera.AddOrientOp().Set(Gf.Quatf(0.0, 0.7, 0.7, 0.0))

static_path = "/World/static_camera"
static_camera = UsdGeom.Camera.Define(stage, static_path)
static_camera.AddTranslateOp().Set(Gf.Vec3f(-0.025128380734830664, 6.88345, 0.735964839641152))
static_camera.AddOrientOp().Set(Gf.Quatf(-0.00355, -0.005, 0.6779, 0.73513))

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

import omni.kit.app
import os
import numpy as np
from PIL import Image
import omni.replicator.core as rep
from isaacsim.core.utils.prims import get_prim_at_path

# async def capture_images_from_cameras(camera_paths, resolution=(1280, 960), output_dir="/home/mushroomgeorge/robotics-rl/camera_output"):
#     while True:
        
#             # Clean output dir
#         # os.makedirs(output_dir, exist_ok=True)
#         # for f in os.listdir(output_dir):
#         #     os.remove(os.path.join(output_dir, f))

#         # Create render products using Replicator's proper API
#         render_products = []

#         for cam_path in camera_paths:
#             cam_prim = get_prim_at_path(cam_path)
#             if not cam_prim or not cam_prim.IsDefined():
#                 print(f"[ERROR] Camera prim at {cam_path} not found or invalid.")
#                 continue

#             rp = rep.create.render_product(cam_path, resolution=resolution)
#             render_products.append(rp)

#         if not render_products:
#             print("[ERROR] No valid render products created.")
#             return {}

#         writer = rep.BasicWriter(
#         output_dir=output_dir,
#         rgb=True,
#         frame_padding=4
#         )
#         writer.attach(render_products)

#         # Use a frame trigger — must be OUTSIDE async def
#         # with rep.trigger.on_frame():
#         #     rep.orchestrator.run()

#         # await rep.orchestrator.step_async()

#         with rep.trigger.on_frame():
#             rep.orchestrator.run(num_frames=1)

#         await asyncio.sleep(0.5)

#         images = {}
#         print("before tha save")
#         # for i, cam_path in enumerate(camera_paths):
#         #     npy_path = os.path.join(output_dir, f"rgb_{i:04d}.npy")
#         #     try:
#         #         rgb_array = np.load(npy_path)
#         #         images[cam_path] = rgb_array
#         #         Image.fromarray(rgb_array).save(os.path.join(output_dir, f"rgb_{i:04d}.png"))
#         #     except Exception as e:
#         #         print(f"Failed to process image for {cam_path}: {e}")

#         await asyncio.sleep(0.01)
#         return images
        

# async def capture_images_task():
#     return capture_images_from_cameras([arm_path, static_path])









import omni.replicator.core as rep
import os
from PIL import Image
from pathlib import Path
capture_count = 0
MAX_CAPTURES = 1
def capture_camera_images(output_dir="camera_output"):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print(dir(rep.trigger))
    # Ensure replicator is initialized
    # rep.settings.set_render_product_resolution(640, 480)

    # Get stage and cameras
    # stage = omni.usd.get_context().get_stage()
    arm_camera_prim = stage.GetPrimAtPath(arm_path)
    static_camera_prim = stage.GetPrimAtPath(static_path)

    if not arm_camera_prim.IsValid() or not static_camera_prim.IsValid():
        print("Camera prim(s) not valid!")
        return

    arm_rp = rep.create.render_product(arm_path, resolution=(1280, 960))
    static_rp = rep.create.render_product(static_path, resolution=(1280, 960))

    writer = rep.WriterRegistry.get("BasicWriter")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    writer.initialize(output_dir=str(output_dir), rgb=True)

    with rep.trigger.on_time(3):
        # writer.attach([arm_rp, static_rp])
        global capture_count
        if capture_count < MAX_CAPTURES:  # Ensure only one capture
            writer.attach([arm_rp, static_rp])
            capture_count += 1
            print(f"Captured and saved image {capture_count}")
    print("end")





# import omni.replicator.core as rep
# from pathlib import Path

# output_dir = "camera_images"
# Path(output_dir).mkdir(exist_ok=True)

# # # Set resolution
# # rep.settings.set_render_product_resolution(640, 480)

# # Create render products from the camera paths
# arm_render_product = rep.create.render_product(arm_path, (640, 480))
# static_render_product = rep.create.render_product(static_path, (640, 480))

# # Set up writer
# writer = rep.WriterRegistry.get("BasicWriter")
# writer.initialize(
#     output_dir=output_dir,
#     rgb=True,
#     semantic_segmentation=False
# )
# writer.attach([arm_render_product, static_render_product])






import time

toggle_interval = 3.5 
last_toggle_time = time.time()

move_by_vel = False
# Main Loop
while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_stopped():
        simulation_app.close()
    
    if my_world.is_playing():

        current_time = time.time()
        if current_time - last_toggle_time > toggle_interval:
            # Use velocirties or positions
            if move_by_vel:
                my_franka.set_joint_velocities(get_random_joint_velocities())
            else:
                my_franka.set_joint_positions(get_random_joint_positions())
            
            print("Franka joints position : ",my_franka.get_joint_positions(),"\n" )
            print("Gripper position : ",my_franka.gripper.get_joint_positions(),"\n")
            print("Cube position : ",cube.get_local_pose()[0],"\n")
            # print(dir(omni.kit.app.get_app()))

            # omni.kit.app.get_app().next_update_async(capture_images_task)

            # asyncio.ensure_future(capture_images_from_cameras([arm_path, static_path]))
            # from writer import setup_replicator_cameras
            # setup_replicator_cameras([arm_path,static_path],(1280,960),"/home/mushroomgeorge/robotics-rl/camera_output")
            # capture_single_image(arm_prim, "/home/mushroomgeorge/robotics-rl/arm_output")
            # capture_single_image(static_prim, "/home/mushroomgeorge/robotics-rl/static_output")
            # capture_camera_view(arm_prim, "/home/mushroomgeorge/robotics-rl/1.png")
            # capture_camera_view(static_prim, "/home/mushroomgeorge/robotics-rl/2.png")
            # print(dir(rep))
            # print(dir(rep.trigger))
            # rep.trigger.on_time(writer.write(),3)
            
            capture_camera_images()
            last_toggle_time = current_time


