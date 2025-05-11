import asyncio
import numpy as np
import gym
from gym import spaces
from envs.gym_environment_interface import GymEnvironmentInterface
import time

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

class FrankaGym(GymEnvironmentInterface):

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    import argparse
    import sys
    import carb
    import numpy as np
    import asyncio
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
    cube.enable_rigid_body_physics()

    # my_world.scene.add_default_ground_plane()
    # print(dir(my_world._physics_context))
    # print(dir(my_world._physics_context))
    # my_world._physics_context.enable_contact_collection(True)

    # my_world.reset()

    my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    # my_franka.enable_rigid_body_physics()
    my_franka.gripper.enable_rigid_body_physics()


    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom, Gf, Usd, UsdLux, UsdShade, Sdf, Tf, Vt, UsdPhysics, PhysxSchema


    stage = get_current_stage()
    cube_prim = stage.GetPrimAtPath("/World/Cube")
    franka_prim = stage.GetPrimAtPath("/World/Franka")
    gripper_prim = stage.GetPrimAtPath("/World/Franka/panda_rightfinger")

    from isaacsim.core.utils.prims import get_prim_at_path
    from omni.physx.scripts import physicsUtils
    
    from omni.physx.scripts import utils
    # print(dir(utils))
    utils.setCollider(gripper_prim)
    utils.setCollider(cube_prim)
    # utils.setRigidBody(franka_prim, approximationShape="convexDecomposition", kinematic=False)
    # utils.setRigidBody(cube_prim, approximationShape="boundingCube", kinematic=True)
    my_world.reset()
    
    # from omni.physx import get_physx_interface, get_shysx_simulation_interface

    # UsdPhysics.CollisionAPI.Apply(cube_prim)
    # UsdPhysics.CollisionAPI.Apply(franka_prim)
    # print(dir(UsdPhysics))
    # print(dir(UsdPhysics.CollisionAPI))
    
    
    
    
    # physicsUtils.add_rigid_xform(stage, "/World/Franka/panda_rightfinger")
    # physicsUtils.add_collision_to_collision_group("/World/Franka/panda_rightfinger")

    # physicsUtils.add_rigid_xform(stage, "/World/Cube")
    # physicsUtils.add_collision_to_collision_group("/World/Cube")

    arm_path = "/World/Franka/panda_hand/arm_camera"
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

    capture_count = 0
    toggle_interval = 3
    last_toggle_time = time.time()
    move_by_vel = False

    # def check_collision(self):
    #     import omni.physx
    #     contact_report_interface = omni.physx.get_contact_report_interface()
    #     contacts = contact_report_interface.get_contacts()

    #     for contact in contacts:
    #         prim0 = contact.actor0
    #         prim1 = contact.actor1

    #         if not prim0 or not prim1:
    #             continue

    #         prim0_path = str(prim0.get_path())
    #         prim1_path = str(prim1.get_path())

    #         # You can adjust these based on exact paths of gripper/arm/cube
    #         if ("/World/Franka/panda_rightfinger" in prim0_path or "/World/Franka/panda_rightfinger" in prim1_path) and "/World/Cube" in (prim0_path + prim1_path):
    #             return True  # Collision detected

    #     return False


    def capture_images(self):
    # Capture one image from each camera and save it
        self.writer.attach([self.arm_rp, self.static_rp])
        # Give time for the render to occur (depends on Isaac Sim version, often needed)
        for _ in range(3):  # step 3 frames just to be safe
            self.my_world.step(render=True)
        self.writer.detach()

    # from isaacsim.core import SimulationContext
    # print(dir(isaacsim.core))


    def step(self, action):
        import omni.physx

        if self.move_by_vel:
            self.my_franka.set_joint_velocities(action)
        else:
            self.my_franka.set_joint_positions(action)

        self.my_world.step(render=True)

        self.capture_images()

        observation = self.get_observation()

        
        # Check for collision
        # sim_context = SimulationContext()
        # contacts = sim_context.get_contact_report()
    
        collision = False
        # for contact in contacts:
        #     if ("/World/Franka/panda_rightfinger" in contact['body0'] and "/World/Cube" in contact['body1']) or \
        #     ("/World/Cube" in contact['body0'] and "/World/Franka/panda_rightfinger" in contact['body1']):
        #         collision = True
        #         break

        # observation = self.get_observation()
        
        physx_sim_interface = omni.physx.acquire_physx_simulation_interface()
        # physx_sim_interface.set_contact_report_enabled(True)
        # print(dir(physx_sim_interface))
        # Retrieve contact report data
        contact_headers, contact_data = physx_sim_interface.get_contact_report()

        # Define the paths to the Franka robot and the cube
        franka_path = "/World/Franka/panda_rightfinger"
        cube_path = "/World/Cube"
        # Iterate through contact headers to check for collisions
        


        # Your handler function for contact reports
        def on_contact_report(contacts, contact_data):
            print(contacts,contact_data)
            for contact in contacts:
                print("Contact Report Received:")
                print("  Actor0:", contact.actor0)
                print("  Actor1:", contact.actor1)
        
            # Example collision check
            if "/World/Franka/panda_rightfinger" in contact.actor0 and "/World/Cube" in contact.actor1:
                print("Collision detected between gripper and cube!")

            # Subscribe to contact reports
        p = physx_sim_interface.subscribe_contact_report_events(on_contact_report)
        print(p)


        # print(f"Checking collisions between: {franka_path} and {cube_path}")
        # print(f"Number of contact headers: {len(contact_headers)}")
        # for header in contact_headers:
        #     print(f"Header: actor0={header.actor0}, actor1={header.actor1}")

        # print(contact_data,contact_headers)
        # for header in contact_headers:
        #     print("Actor0:", header.actor0)
        #     print("Actor1:", header.actor1)
        
        # for header in contact_headers:
        #     actor0 = header.actor0
        #     actor1 = header.actor1

        #     # Check if either actor is part of the Franka robot and the other is the cube
        #     if  (actor0.startswith(franka_path) and actor1 == cube_path) or \
        #         (actor1.startswith(franka_path) and actor0 == cube_path):
        #         print(f"Collision detected between {actor0} and {actor1}")
        #         # Apply reward deduction and set done flag
        #         reward = -10
        #         collision = True
        #         done = True
        #         break
        
        
        if collision:
            reward = -10.0
            done = True
        else:
            reward = self.calculate_reward(observation, done=False)
            done = False
        # === Collision Detection ===
        # collision_detected = False
        # contact_pairs = self.my_world._physics_context.get_contact_pairs()
        # for pair in contact_pairs:
        #     prim1 = pair["body0"]
        #     prim2 = pair["body1"]
        #     if ("/World/Cube" in prim1 and "/World/Franka" in prim2) or ("/World/Cube" in prim2 and "/World/Franka" in prim1):
        #         collision_detected = True
        #         break

        # === Reward & Termination Logic ===
        # if collision_detected:
        #     reward = -10.0
        #     done = True
        # else:
        #     reward = self.calculate_reward(observation, done=False)
        #     done = False

        info = {"collision": collision}

        return observation, reward, done, info


    def reset(self):
        """Reset environment"""
        self.my_world.reset()
        self.my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
        self.my_franka.gripper.set_default_state(self.my_franka.gripper.joint_opened_positions)
        self.cube.set_local_pose(position=np.array([0.3, 0.3, 0.3]))
        return self.get_observation()
    
    def render(self, mode='human'):
        """Nothing needed since Isaac Sim is rendering automatically."""
        pass

    def calculate_reward(self, new_state, done):
        """Simple distance reward"""
        gripper_pos = self.my_franka.end_effector.get_local_pose()[0]
        cube_pos = self.cube.get_local_pose()[0]

        distance = np.linalg.norm(gripper_pos - cube_pos)
        reward = -distance
        return reward
    
    def get_observation(self):
        """Get observation: robot joints + cube position"""
        joint_pos = self.my_franka.get_joint_positions()
        cube_pos = self.cube.get_local_pose()[0]
        observation = np.concatenate([joint_pos, cube_pos])
        return observation
    
    def close(self):
        self.simulation_app.close()

if __name__ == "__main__":
    env = FrankaGym()
    i=0
    try:
        for i in range(100):  # Run 100 steps as an example
            action = get_random_joint_velocities() if env.move_by_vel else get_random_joint_positions()
            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward}\nStep {i}\n{info}")
            time.sleep(0.05)  # Slow down stepping if needed
    finally:
        env.close()