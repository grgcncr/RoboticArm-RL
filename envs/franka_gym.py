import torch
import numpy as np
import time
from pathlib import Path
from gym import spaces
import gym
import imageio.v2 as imageio
from envs.gym_environment_interface import GymEnvironmentInterface
class FrankaGym(gym.Env, GymEnvironmentInterface):
    def __init__(self, sim, scene):
        super(FrankaGym, self).__init__()
        self.sim = sim
        self.scene = scene
        joint_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        joint_upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        
        # Observation: 3D distance + 7D joints + 1D gripper state = 11 dimensions
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*3 + joint_lower + [0]),    
            high=np.array([np.inf]*3 + joint_upper + [1]),    
        )
        
        # Action: 7 joint positions + 1 gripper state = 8 dimensions
        self.action_space = spaces.Box(
            low=np.array(joint_lower + [0]),      
            high=np.array(joint_upper + [1]),     
            dtype=np.float32
        )

        self.joint_pos = None
        self.count = 0
        self.cube_contact = False
        print("Environment initialized successfully")
    
    def step(self, action):
        self.count += 1
        print(f"Repetition : {self.count}")
        sim_dt = self.sim.get_physics_dt()        
        franka = self.scene["robot"]
        cube = self.scene["cube"]        
        cube_pos = cube.data.body_pos_w
        g_state = 0.04
        if action[7] > 0.0:
            g_state = 0.0
        self.joint_pos = list(action[:7]) + [g_state, g_state]
        joint_pos = torch.tensor(list(action[:7]) + [g_state, g_state], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        franka.write_joint_position_to_sim(joint_pos)
        franka.write_data_to_sim()
        # franka.update(sim_dt)
        # franka.data.update(sim_dt)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)

        # Capture images for every repetition
        # capture_images(scene,rep,"side_camera", "~/robotics-rl/camera_output/side_camera") # <---------------- uncomment
        # capture_images(scene,rep,"hand_camera", "~/robotics-rl/camera_output/hand_camera") # <---------------- uncomment
        reward = self.calculate_reward() 
        done = self.cube_contact

        info = {
                "Cube collision": self.cube_contact,
                "Done" : done
               }
         
        observation = self._get_observation()       
        return observation, reward, done, info


    def reset(self):
        """Reset environment"""
        franka = self.scene["robot"]
        franka.write_joint_position_to_sim(torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.00, 0.00], dtype=torch.float32, device="cuda:0").unsqueeze(0))
        cube = self.scene["cube"]
        env_ids = torch.tensor([0], dtype=torch.long, device="cuda:0")
        cube_pose = torch.tensor([0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        cube.write_root_pose_to_sim(cube_pose, env_ids)
        # cube.write_root_velocity_to_sim(
        #     torch.zeros((1, 3), device=cube.device),  # Clear linear velocity
        #     torch.zeros((1, 3), device=cube.device)   # Clear angular velocity
        # )
        franka.write_data_to_sim()
        cube.write_data_to_sim()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())
        observation = self._get_observation()
        print("Enviroment Reseted")
        return observation
    
    def render(self, mode='human'):
        """Nothing needed since Isaac Sim is rendering automatically."""
        pass

    def calculate_reward(self):
        contact_sensor = self.scene["contact_sensor"]
        max_force = torch.max(contact_sensor.data.net_forces_w).item()
        # print("Received max contact force of :",max_force)
        force_magnitudes = torch.norm(contact_sensor.data.force_matrix_w, dim=-1)
        contact_detected = (force_magnitudes > 0.1).any().item()
        distance = torch.norm(self.calculate_distance())
        if max_force != 0.0 and contact_detected:          
            print("------------------------Franka Touched the Cube------------------------")
            # FrankaGym.close()
            self.cube_contact = True
            reward = -(distance*5)
            print(f"Reward : {reward}")
            self.reset()
            time.sleep(10)
            return reward
        else:
            self.cube_contact = False    
            reward = -distance
            print(f"Reward : {reward}")
            return reward


    def calculate_distance(self):
        franka = self.scene["robot"]
        cube = self.scene["cube"]
        fdata = franka.data

        link_names = fdata.body_names
        idx_left = link_names.index("panda_leftfinger")
        idx_right = link_names.index("panda_rightfinger")

        pos_left = fdata.body_link_pos_w[:, idx_left]
        pos_right = fdata.body_link_pos_w[:, idx_right]
        gripper_center = (pos_left + pos_right) / 2
        cube_pos = cube.data.body_pos_w[:, 0]  # only one body
        distance = torch.stack([
            gripper_center[0][0] - cube_pos[0][0],
            gripper_center[0][1] - cube_pos[0][1],
            gripper_center[0][2] - cube_pos[0][2]
        ])
        return distance

    def _get_observation(self):
        # Get observation: 3D distance + 7D joint positions + 1D gripper state
        franka = self.scene["robot"]
        joint_pos = franka.data.joint_pos
        joint_pos_7 = joint_pos[0][:7]
        gr_pos = joint_pos[0][7:9]
        # print("joint pos ",joint_pos_7[0])
        g_status = torch.tensor([1], dtype=torch.float32, device="cuda:0")
        if gr_pos[0].item() < 0.03:
            g_status = torch.tensor([0], dtype=torch.float32, device="cuda:0")
        distance = self.calculate_distance()
        # print("distance ", distance[0])
        # print("gstatus",g_status[0])
        obs = torch.cat([distance,joint_pos_7, g_status], dim=0)
        obs_numpy = obs.cpu().numpy().astype(np.float32)
        return obs_numpy



















# def capture_images(self, rep, camera, path):
#     rgb_tensor = self.scene[camera].data.output["rgb"]
#     rgb_np = rgb_tensor.squeeze(0).cpu().numpy()  
#     save_dir = Path(path).expanduser()
#     save_dir.mkdir(parents=True, exist_ok=True)
#     image_path = save_dir / f"camera_output_{rep}.png"
#     imageio.imwrite(str(image_path), rgb_np)
#     # print(f"--------Saved {image_path}--------")
#     # # For 10 images
#     # if rep == 10:
#     #     FrankaGym.close()
    


# def get_random_joint_velocities(scale_arm=4.0, scale_gripper=2.5):
#     arm_vel = (np.random.rand(7) - 0.5) * 2 * scale_arm
#     gripper_vel = np.random.uniform(-1, 1) * scale_gripper
#     gripper = np.array([gripper_vel, gripper_vel])
#     joint_vel = np.concatenate([arm_vel, gripper])
#     return torch.tensor(joint_vel, dtype=torch.float32, device="cuda:0").unsqueeze(0)

# def get_random_joint_positions(grip_open):
#     joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
#     joint_limits_upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    
#     arm_positions = np.random.uniform(joint_limits_lower, joint_limits_upper)
#     # gripper_position = np.random.uniform(0.01, 0.04)
#     if grip_open:
#         gripper_position = 0.00
#     else:
#         gripper_position = 0.04
#     gripper = np.array([gripper_position, gripper_position])
#     joint_pos = np.concatenate([arm_positions, gripper])
#     return torch.tensor(joint_pos, dtype=torch.float32, device="cuda:0").unsqueeze(0)
#     # return torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.04, 0.04], dtype=torch.float32, device="cuda:0").unsqueeze(0)


# def simulation_run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
#     rep = 0
#     franka = scene["robot"] 
#     # Init state
#     init_target = torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.01, 0.01], dtype=torch.float32, device="cuda:0").unsqueeze(0)
#     joint_pos = init_target
#     franka.write_joint_position_to_sim(init_target)
#     franka.update(sim_dt)
#     franka.data.update(sim_dt)
#     # contact_sensor = scene["contact_sensor"]
#     # cube = scene["cube"]
    
#     scene.write_data_to_sim()
#     sim.step()
#     sim_time += sim_dt
#     scene.update(sim_dt)
#     # Main Loop 
#     while simulation_app.is_running():
#         count += 1
#         if count % 10 == 0:
#             rep += 1
#             print("Repetition :",rep)
#             franka
#             *_, joint_pos = FrankaGym.step(sim,scene,rep)
#         else:
#             franka.write_joint_position_to_sim(joint_pos)
#             franka.write_data_to_sim()
#             scene.write_data_to_sim()
#             sim.step()
#             sim_time += sim_dt
#             scene.update(sim_dt)


