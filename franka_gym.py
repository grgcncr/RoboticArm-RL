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

    from isaacsim.core.api.simulation_context import SimulationContext

    # Start simulation context
    

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
    from isaacsim.core.api.physics_context import PhysicsContext
    from isaaclab.sensors import ContactSensorCfg
    
    PhysicsContext()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
    args, unknown = parser.parse_known_args()

    def print_contact(sensor_path, part):
        from isaacsim.sensors.physics import _sensor
        _contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        raw_data = _contact_sensor_interface.get_contact_sensor_raw_data(sensor_path)
        body_name = _contact_sensor_interface.decode_body_name(raw_data[0][3])
        if raw_data.size > 0:
            if body_name == "/World/Cube":
                print(f"{part} in contact with: {body_name}, with impolse : {raw_data[0][6]}")
                return True
            else:
                return False
        else:
            # print(f"{part} has no contact.")
            return False

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit()

    my_world = World(stage_units_in_meters=1.0)
    # print(dir(my_world.stage))
    my_world.scene.add_default_ground_plane()
    set_camera_view((1.5, 2.0, 1.5), (0.0, 0.0, 0.0))
    sim_context = SimulationContext()

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
            color=np.array([0, 0, 7]),
        )
    )
    # cube.enable_rigid_body_physics()

    # my_world.scene.add_default_ground_plane()
    # print(dir(my_world._physics_context))
    # my_world._physics_context.enable_contact_collection(True)

    # my_world.reset()

    my_franka.set_joint_positions([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05])
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    # my_franka.enable_rigid_body_physics()
    # my_franka.gripper.enable_rigid_body_physics()


    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom, Gf, Usd, UsdLux, UsdShade, Sdf, Tf, Vt, UsdPhysics, PhysxSchema

    stage = get_current_stage()
    cube_prim = stage.GetPrimAtPath("/World/Cube")
    franka_prim = stage.GetPrimAtPath("/World/Franka")
    rfinger_prim = stage.GetPrimAtPath("/World/Franka/panda_rightfinger")
    lfinger_prim = stage.GetPrimAtPath("/World/Franka/panda_leftfinger")
    hand_prim = stage.GetPrimAtPath("/World/Franka/panda_hand")
    
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)
    

    # import omni.kit.commands

    # # Set convex decomposition for articulated links
    # omni.kit.commands.execute("ChangeColliderApproximationCommand",
    #     approximationShape="convexDecomposition",
    #     primPaths=["/World/Franka/panda_hand", 
    #            "/World/Franka/panda_leftfinger", 
    #            "/World/Franka/panda_rightfinger"]
    # )


    from isaacsim.core.utils.prims import get_prim_at_path
    from omni.physx.scripts import physicsUtils
    
    from omni.physx.scripts import utils
    # # print(dir(utils))
    # utils.setCollider(lfinger_prim)
    # # utils.setRigidBody(gripper_prim, approximationShape="convexHull", kinematic=True)
    # utils.setCollider(cube_prim)
    # utils.setRigidBody(cube_prim, approximationShape="boundingCube", kinematic=False)
    mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
    mass_api.CreateMassAttr(1.0)
    rigid_cube = UsdPhysics.RigidBodyAPI.Apply(cube_prim)
    rigid_cube.CreateRigidBodyEnabledAttr(True)
    rigid_cube.CreateKinematicEnabledAttr(False)
    # utils.setCollider(rfinger_prim)
    # utils.setCollider(hand_prim)
    
    from isaacsim.sensors.physics import ContactSensor
    import numpy as np

    # cube_sensor = ContactSensor(
    #     prim_path="/World/Cube/Contact_Sensor",
    #     name="Contact_Sensor_cube",
    #     frequency=30,
    #     translation=np.array([0, 0, 0]),
    #     min_threshold=0,
    #     max_threshold=10000000,
    #     radius=-1
    # )

    # set_collision_approximation(hand_prim)
    # set_collision_approximation(lfinger_prim)
    # set_collision_approximation(rfinger_prim) 

    # franka_csensor = ContactSensorCfg(
    #     prim_path="/World/Franka", update_period=0.0, history_length=6, debug_vis=True
    # )
    
    # print(dir(utils.setCollider(hand_prim)))

    
    rfinger_sensor = ContactSensor(
        prim_path="/World/Franka/panda_rightfinger/Contact_Sensor",
        name="Contact_Sensor_r",
        frequency=60,
        translation=np.array([0, 0, 0]),
        min_threshold=0.1,
        max_threshold=10000000,
        radius=-1
    )

    lfinger_sensor = ContactSensor(
        prim_path="/World/Franka/panda_leftfinger/Contact_Sensor",
        name="Contact_Sensor_l",
        frequency=60,
        translation=np.array([0, 0, 0]),
        min_threshold=0.1,
        max_threshold=10000000,
        radius=-1
    )

    hand_sensor = ContactSensor(
        prim_path="/World/Franka/panda_hand/Contact_Sensor",
        name="Contact_Sensor_hand",
        frequency=60,
        translation=np.array([0, 0, 0]),
        min_threshold=0.1,
        max_threshold=10000000,
        radius=-1
    )

    # franka_sensor = ContactSensor(
    #     prim_path="/World/Franka/Contact_Sensor",
    #     name="Contact_Sensor_franka",
    #     frequency=60,
    #     translation=np.array([0, 0, 0]),
    #     min_threshold=0.1,
    #     max_threshold=10000000,
    #     radius=-1
    # )

    # from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns

    # contact_forces = ContactSensorCfg(
    #     prim_path="/World/Franka", update_period=0.0, history_length=6, debug_vis=True
    # )

    import omni
    from pxr import PhysxSchema

    # stage = omni.usd.get_context().get_stage()
    contact_report_cube = PhysxSchema.PhysxContactReportAPI.Apply(cube_prim)
    contact_report_rf = PhysxSchema.PhysxContactReportAPI.Apply(rfinger_prim)
    contact_report_lf = PhysxSchema.PhysxContactReportAPI.Apply(lfinger_prim)
    contact_report_hand = PhysxSchema.PhysxContactReportAPI.Apply(hand_prim)
    # contact_report_franka = PhysxSchema.PhysxContactReportAPI.Apply(franka_prim)

    cube_collision = PhysxSchema.PhysxCollisionAPI.Apply(cube_prim)
    # print(dir(cube_collision))
    # cube_collision.CreateApproximationAttr("boundingCube")
    hand_collision = PhysxSchema.PhysxCollisionAPI.Apply(hand_prim)
    rfinger_colliion = PhysxSchema.PhysxCollisionAPI.Apply(rfinger_prim)
    lfinger_collision = PhysxSchema.PhysxCollisionAPI.Apply(lfinger_prim)
    # franka_collision = PhysxSchema.PhysxCollisionAPI.Apply(franka_prim)

    # Set a minimum threshold for the contact report to zero
    contact_report_cube.CreateThresholdAttr(0.0)
    contact_report_rf.CreateThresholdAttr(0.0)
    contact_report_lf.CreateThresholdAttr(0.0)
    contact_report_hand.CreateThresholdAttr(0.0)
    # contact_report_franka.CreateThresholdAttr(0.0)
    # print(dir(franka_collision))
    # print(dir(franka_sensor))
    # print(dir(cube_prim))

    cube_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("boundingCube")
    hand_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    rfinger_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    lfinger_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
    # franka_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")

       

    # cube_collision.CreateApproximationAttr().Set("convexDecomposition")
    # hand_collision.CreateApproximationAttr().Set("convexDecomposition")
    # rfinger_colliion.CreateApproximationAttr().Set("convexDecomposition")
    # lfinger_collision.CreateApproximationAttr().Set("convexDecomposition")


    # contact_report_cube.CreateCollisionEnabledAttr().Set(True)
    # contact_report_hand.CreateCollisionEnabledAttr().Set(True)
    # contact_report_lf.CreateCollisionEnabledAttr().Set(True)
    # contact_report_rf.CreateCollisionEnabledAttr().Set(True)

    # contact_report_cube.CreateContactOffsetAttr().Set(0.01)
    # contact_report_hand.CreateContactOffsetAttr().Set(0.01)
    # contact_report_lf.CreateContactOffsetAttr().Set(0.01)
    # contact_report_rf.CreateContactOffsetAttr().Set(0.01)


    # contact_report_cube.CreateApproximationAttr().Set("convexDecomposition")
    # contact_report_hand.CreateApproximationAttr().Set("convexDecomposition")
    # contact_report_lf.CreateApproximationAttr().Set("convexDecomposition")
    # contact_report_rf.CreateApproximationAttr().Set("convexDecomposition")

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

    #------------------------------ WORLD RESET ---------------------------------------#
    my_world.reset()


    capture_count = 0
    toggle_interval = 3
    last_toggle_time = time.time()
    move_by_vel = False

    def capture_images(self):
        self.writer.attach([self.arm_rp, self.static_rp])
        for _ in range(3):
            self.my_world.step(render=True)
        self.writer.detach()

    from isaacsim.sensors.physics import _sensor
    _contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
    from isaacsim.core.api.sensors import RigidContactView
    print(dir(RigidContactView))
    contact_view = RigidContactView(prim_paths_expr="/World/Franka/*", filter_paths_expr="/World/Cube")    
    contact_view.initialize(sim_context.physics_sim_view)
    # print(dir(_contact_sensor_interface))
    # print(dir(hand_sensor))
    
    def step(self, action):
        import omni.physx

        if self.move_by_vel:
            self.my_franka.set_joint_velocities(action)
        else:
            self.my_franka.set_joint_positions(action)

        

        self.capture_images()

        observation = self.get_observation()
    
        collision_r = False
        collision_l = False
        collision_hand = False
        
        # physx_sim_interface = omni.physx.acquire_physx_simulation_interface()
        # contact_headers, contact_data = physx_sim_interface.get_contact_report()

        # Define the paths to the Franka robot and the cube
        franka_path = "/World/Franka/panda_rightfinger"
        cube_path = "/World/Cube"
        # print("Cube : ",self.cube_sensor.get_current_frame())
        

        # print_contact("/World/Franka/panda_rightfinger/Contact_Sensor", "Right finger")
        # print_contact("/World/Franka/panda_leftfinger/Contact_Sensor", "Left finger")
        # print_contact("/World/Franka/panda_hand/Contact_Sensor", "Hand")

        # from isaacsim.core.utils.physics import wake_up_prim

        # wake_up_prim(self.cube_prim)
        raw_rigid_cube_data = self._contact_sensor_interface.get_rigid_body_raw_data("/World/Cube")
        raw_data_r = self._contact_sensor_interface.get_contact_sensor_raw_data("/World/Franka/panda_rightfinger/Contact_Sensor")
        raw_data_l = self._contact_sensor_interface.get_contact_sensor_raw_data("/World/Franka/panda_leftfinger/Contact_Sensor")
        raw_data_hand = self._contact_sensor_interface.get_contact_sensor_raw_data("/World/Franka/panda_hand/Contact_Sensor")
        # raw_data_franka = self._contact_sensor_interface.get_contact_sensor_raw_data("/World/Franka/Contact_Sensor")
        # reading = _contact_sensor_interface.get_sensor_reading("/World/Franka/panda_rightfinger/Contact_Sensor")
        # 200193, 29361409
        
        import math

        # if :
        #     print("True contact")

        # if raw_data_franka.size > 0:
        #     body = self._contact_sensor_interface.decode_body_name(raw_data_franka[0][3])
        #     impolse = raw_data_franka[0][6]
        #     # print(impolse)
        #     print(body)
        #     if(body == "/World/Cube"):
        #         # collision_r = True
        #         print(f"Franka --> cube : {impolse}")
        # print(raw_rigid_cube_data)
        collision_r = self.print_contact("/World/Franka/panda_rightfinger/Contact_Sensor","Right Finger")
            
        # print("Hand : ",self.hand_sensor.get_current_frame())
        # print("Left Finger : ",self.lfinger_sensor.get_current_frame())
        # print("Right FInger : ",self.rfinger_sensor.get_current_frame())

        collision_l = self.print_contact("/World/Franka/panda_leftfinger/Contact_Sensor","Left Finger")
        
        collision_hand = self.print_contact("/World/Franka/panda_hand/Contact_Sensor","Panda Hand")


        if (collision_l | collision_r | collision_hand):
            reward = -10.0
            done = True
        else:
            reward = self.calculate_reward(observation, done=False)
            done = False

        info = {"collision": (collision_l | collision_r | collision_hand)}
        self.my_world.step(render=True)
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
        for i in range(500):
            print("Repetition",i)
            action = get_random_joint_velocities() if env.move_by_vel else get_random_joint_positions()
            obs, reward, done, info = env.step(action)
            # print(f"Reward: {reward}\nStep {i}\n{info}")
            
            time.sleep(0.05)
    finally:
        env.close()