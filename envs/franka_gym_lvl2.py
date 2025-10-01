import torch
import numpy as np
import math
import time
from pathlib import Path
from gymnasium import spaces
import gymnasium as gym
import random
import imageio.v2 as imageio
from envs.gym_environment_interface import GymEnvironmentInterface
class FrankaGymLvl2(gym.Env, GymEnvironmentInterface):
    metadata = {"render_modes": [], 'render_fps' : 30}
    def __init__(self, app, sim, scene):
        super(FrankaGymLvl2, self).__init__()
        self.sim = sim
        self.scene = scene
        self.app = app
        joint_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        joint_upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        
        # Action: 7 joint positions + 1 gripper state = 8 dimensions
        self.action_space = spaces.Box(
            low=np.array(joint_lower + [0.0]),      
            high=np.array(joint_upper + [1.0]),     
            dtype=np.float64
        )

        margin = 0.05
        joint_lower = [j - margin for j in joint_lower]
        joint_upper = [j + margin for j in joint_upper]

        # Observation: 3 distance values + 1 delta orientation + 7 joint positions + 1 gripper state = 12 dimensions
        self.observation_space = spaces.Box(
            low=np.array([-10.0]*3 + [-np.pi] + joint_lower + [0.0]),    
            high=np.array([10.0]*3 + [np.pi] + joint_upper + [1.0]),    
            dtype=np.float64
        )
        
        self.joint_pos_history = []
        self.count = 0
        self.total_count = 0
        self.suc_count = 0
        self.ep_suc_count = 0
        self.ep_count = 0
        self.suc_rate = 0
        self.flag = False
        self.cube_contact = False
        self.prev_grip_dis = False
        self.prev_distance = None
        self.prev_gripper_center = None
        self.position_history = []
        self.history_size = 15
        self.distance_history = []
        self.progress_history = []
        self.max_steps = 100
        self.delta_orientation = 1.5
        self.gripper_idx = self.scene["robot"].data.body_names.index("panda_rightfinger")
        self.world_down = torch.tensor([[0.0, 0.0, -1.0]], device="cuda:0")
        self.env_ids = torch.tensor([0], dtype=torch.long, device="cuda:0")

        print("Environment initialized successfully")
    
    def step(self, action):
        self.flag = False
        self.count += 1
        self.total_count += 1
        # if self.total_count % (1024 * 200) == 0:
            # print(f"{self.total_count} Fanka action: {action}")
            # Capture images
            # self.capture_images("side_camera", "~/robotics-rl/camera_output/side_camera") # <---------------- uncomment
            # self.capture_images("hand_camera", "~/robotics-rl/camera_output/hand_camera") # <---------------- uncomment

        terminated = False
        trancated = False
        info = {}
        if self.count >= self.max_steps:
            trancated = True
        # print(f"Repetition : {self.count}")
        sim_dt = self.sim.get_physics_dt()        
        franka = self.scene["robot"]
        cube = self.scene["cube"]
        # cube_pose = torch.tensor([-0.4105,  0.1556, 0.0250, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        # cube.write_root_pose_to_sim(cube_pose, self.env_ids)
        # cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device="cuda:0"), self.env_ids)
        # cube.write_data_to_sim()
        contact_sensor = self.scene["contact_sensor"]
        cube_pos = cube.data.body_pos_w
        cx = cube_pos.squeeze(1)[0][0].item()
        cy = cube_pos.squeeze(1)[0][1].item()  
        cz = cube_pos.squeeze(1)[0][2].item()     
        g_state = 0.0
        if action[7] >= 0.5:
            g_state = 0.04
        self.joint_pos = list(action[:7]) + [g_state, g_state]
        joint_pos = torch.tensor(list(action[:7]) + [g_state, g_state], dtype=torch.float64, device="cuda:0").unsqueeze(0)
        # joint_pos = torch.tensor([1.56358981, 0.91938818, 1.15408039, -2.39678192, -1.2282356, 2.16234732, -2.8973, 0.04, 0.04,], dtype=torch.float64, device="cuda:0").unsqueeze(0)
        franka.write_joint_position_to_sim(joint_pos)
        franka.write_data_to_sim()
        
        self.delta_orientation = self.get_delta_orientation()

        # if abs(cx) > 0.51 or abs(cy) > 0.51 or (abs(cx) < 0.4 and abs(cy) < 0.4) or cz > 1.0:
        #     if random.randint(1, 3) == 1:
        #         if random.randint(1,2) == 1:
        #             cube_pose = torch.tensor([random.uniform(-0.2, 0.2), random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), 0.02, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        #         else:
        #             cube_pose = torch.tensor([random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), random.uniform(-0.2, 0.2), 0.02, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        #     else:    
        #         cube_pose = torch.tensor([random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), 0.02, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
            
        cube_distance_2d = math.sqrt(cx*cx + cy*cy)
        if cube_distance_2d < 0.5 or cube_distance_2d > 0.55 or cz > 1.0:
            angle = random.uniform(0, 2 * math.pi)
            radius = math.sqrt(random.uniform(0.5**2, 0.55**2))
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            cube_pose = torch.tensor([x, y, 0.02, 1.0, 0.0, 0.0, 0.0],dtype=torch.float32, device="cuda:0").unsqueeze(0)

            zero_vel = torch.zeros((1, 6), dtype=torch.float32, device="cuda:0")  # [linear(3), angular(3)]
            cube.write_root_pose_to_sim(cube_pose, self.env_ids)
            cube.write_root_velocity_to_sim(zero_vel, self.env_ids)
            cube.write_data_to_sim()
        
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)

        reward = self.calculate_reward() 
        
        # False cube contact 
        if reward == -1000:
            terminated = True
            reward = 0

        if reward == 100:   
            # print(f"Cube pos: {cube_pos}\n Franka pos: {action}\n")
            terminated = True

        if reward == -1:
            terminated = True
    
        if trancated or terminated:
            self.ep_count += 1
            self.flag = True # update suc_rate only in the 1st 100,200,300...th step
            # print(self.ep_count)

        if self.ep_count % 100 == 0 and self.flag:
            self.suc_rate = self.ep_suc_count / 100
            # print(self.ep_count)

        info["success_rate"] = self.suc_rate
        # info["distance"] = self.prev_distance


        observation = self._get_observation()
        return observation, reward, terminated, trancated, info


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        franka = self.scene["robot"]
        cube = self.scene["cube"]
        franka.write_joint_position_to_sim(torch.tensor(
            [0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.00, 0.00],
            dtype=torch.float64, device="cuda:0"
        ).unsqueeze(0))

        franka.write_data_to_sim()
        
        # if random.randint(1, 3) == 1:
        #     if random.randint(1,2) == 1:
        #         cube_pose = torch.tensor([random.uniform(-0.2, 0.2), random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), 0.02, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        #     else:
        #         cube_pose = torch.tensor([random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), random.uniform(-0.2, 0.2), 0.02, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        # else:    
        #     cube_pose = torch.tensor([random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), random.uniform(0.4, 0.5) if random.random() < 0.5 else random.uniform(-0.5, -0.4), 0.02, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda:0").unsqueeze(0)
        
        angle = random.uniform(0, 2 * math.pi)
        radius = math.sqrt(random.uniform(0.5**2, 0.55**2))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        cube_pose = torch.tensor([x, y, 0.02, 1.0, 0.0, 0.0, 0.0],dtype=torch.float32, device="cuda:0").unsqueeze(0)

        zero_vel = torch.zeros((1, 6), dtype=torch.float32, device="cuda:0")  # [linear(3), angular(3)]
        cube.write_root_pose_to_sim(cube_pose, self.env_ids)
        cube.write_root_velocity_to_sim(zero_vel, self.env_ids)
        cube.write_data_to_sim()

        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())

        observation = self._get_observation()
        self.joint_pos_history = []
        self.cube_contact = False
        self.prev_distance = None
        self.prev_gripper_center = None
        self.position_history = []
        self.history_size = 15
        self.distance_history = []
        self.progress_history = []
        self.count = 0
        if self.ep_count % 100 == 0:
            self.ep_suc_count = 0
            # print("from reset")
        return observation , {}
    
    def render(self, mode='human'):
        """Nothing needed since Isaac Sim is rendering automatically."""
        pass

    def calculate_reward(self):
        # self.ep_suc_count += 1
        cube = self.scene["cube"]
        # self.prev_grip_dis = False
        contact_sensor = self.scene["contact_sensor"]
        max_force = torch.max(contact_sensor.data.net_forces_w).item()
        force_magnitudes = torch.norm(contact_sensor.data.force_matrix_w, dim=-1)
        contact_detected = (force_magnitudes != 0.0).any().item()

        dis_xy, dis_z = self.calculate_distance_xy_z()
        dis = self.calculate_distance(False)
        obs = self._get_observation()
        joint3  = obs[7]
        g_status = obs[11]
        delta_orientation = self.delta_orientation
        link3_pos = self.get_link_pos("panda_link3")
        link4_pos = self.get_link_pos("panda_link4")
        link6_pos = self.get_link_pos("panda_link6")
        link_meanz = (link3_pos[0][2] + link4_pos[0][2]) / 2

        bonus = 0.0

        # if self.count <= 5:
        #     print(f"distance: {dis}")
        #     print(f"link3-z: {link3_pos[0][2]}")
        #     print("xy:",dis_xy,"\nz: ", dis_z, "\nori: ",delta_orientation)
        
        # Ground contact
        if max_force > 0.0 and not contact_detected:
            # Ground contact (non-cube collision)
            if self.total_count % 512000 == 0:
                print(f"Success counter: {self.suc_count}")
            return -1  # Immediate penalty and terminate
        
        # Terminal condition: successful grasp (open gripper close to cube)
        # if self.cube_contact == True:
        #     print(f", G_status: {g_status}, Ori: {delta_orientation:.2f},\n Dis: {dis:.2f}, Dis_xy: {dis_xy:.2f}, Dis_z: {dis_z:.2f} Lmz: {link_meanz:.2f}")
        self.cube_contact = False
        if dis_z <= 0.06 and dis_xy <= 0.015 and delta_orientation < 0.5 and link_meanz > 0.4:
            # print(f"\n===SUCCESS=== --> Ori: {delta_orientation:.2f}, linkmean-z: {link_meanz:.2f}\n Distance: {dis:.2f}, x-y: {dis_xy}, z: {dis_z} ")
            
            if contact_detected and g_status == 0:    
                self.suc_count += 1
                self.ep_suc_count += 1
                # print(f"Success counter: {self.suc_count}")
                # print(f"Success episode counter: {self.ep_suc_count}")
                if self.total_count % 512000 == 0:
                    print(f"Success counter: {self.suc_count}")
                return 100  # Success 
            self.prev_grip_dis = True
        if self.total_count % 512000 == 0:
            print(f"Success counter: {self.suc_count}")
        # if self.count <= 5:
        #     print(f"link3 - z: {link3_pos[0][2]}, Joit3: {joint3}")
        #     print(f"link4 - z: {link4_pos[0][2]}")
        
        if max_force > 0.0 and contact_detected: # False cube contact 
            self.cube_contact = True
            cube_vel = cube.data.body_vel_w
            max_cvel = cube_vel.abs().max().item()
            # print(f"{self.total_count} Contact: {max_cvel:.2f}", end="")
            if max_cvel > 1.0:
                if self.total_count % 512000 == 0:
                    print(f"Success counter: {self.suc_count}")
                return -1000
        
        xy_reward = math.exp(-3.5 * dis_xy)
        # print("xy_reward: ",xy_reward)
        z_reward = math.exp(-3.5 * dis_z)
        # print("z_reward: ",z_reward)
        orientation_reward = math.exp(-1.5 * delta_orientation)
        # print("ori_reward: ", orientation_reward)
        reward = 0.45 * xy_reward + 0.35 * z_reward + 0.2 * orientation_reward
        # print(f"gstatus {g_status}")
        # time.sleep(5)

        if delta_orientation <= 0.5 and dis < 0.7:
            bonus += 0.05

        if g_status == 1.0:
            bonus += 0.05
        
        if g_status == 0.0 and dis <= 0.17:
            bonus += 0.06

        if joint3 <= -0.7 and joint3 >= -2.7:
            bonus += 0.025
        #     self.suc_count += 1
        #     self.ep_suc_count += 1
        # if self.total_count % 100 == 0:
        #     print(f"Success counter: {self.suc_count}")
        #     print(f"Success episode counter: {self.ep_suc_count}")

        if link_meanz > 0.4 and link_meanz <= 0.62 and dis < 0.7:#and dis_z < 0.4: and link3_pos[0][2] < 0.6
            bonus += 0.025

        if link6_pos[0][2] < link_meanz:
            bonus += 0.02

        if delta_orientation < 0.5:
            if dis <= 0.07:
                bonus += 0.2
            elif dis <= 0.08:
                bonus += 0.15
            elif dis <= 0.09:
                bonus += 0.1
            elif dis <= 0.12:
                bonus += 0.05
            # elif dis <= 0.2:
            #     bonus += 0.1     

        penalty = 0

        if joint3 > -0.7:
            penalty += 0.15

        if joint3 < -2.7:
            penalty += 0.3

        if  link_meanz <= 0.4 or link_meanz > 0.62:
            penalty += 0.15

        if link6_pos[0][2] >= link_meanz:
            penalty += 0.15

        if dis > 0.7:
            penalty += 0.2
        elif dis > 0.5:
            penalty += 0.1
        # elif dis > 0.3:
        #     penalty += 0.1

        # if dis_z < 0.02:
        #     penalty += 0.1  

        if delta_orientation > 1.5:
            penalty += 0.3
        elif delta_orientation > 1:
            penalty += 0.2
        # elif delta_orientation > 0.75:
        #     penalty += 0.15
        # elif delta_orientation > 0.5:
        #     penalty += 0.05
    
        # if self.total_count >= 2000000:
        #     if dis > 0.5:
        #         penalty += 0.3
        #     elif dis > 0.4:
        #         penalty += 0.2
        #     if delta_orientation > 0.7:
        #         penalty += 0.1
            # if dis > 0.5:
            #     penalty += 0.6
            # if dis > 0.6:
            #     penalty += 0.9

        # if self.total_count >= 3000000:
        #     if dis > 0.3:
        #         penalty += 0.2
        #     if delta_orientation > 0.5:
        #         penalty += 0.1
        #     if dis > 0.4:
        #         penalty += 0.6
        #     if dis > 0.5:
        #         penalty += 0.9
        
        # if self.total_count >= 4000000:
        #     if dis > 0.2:
        #         penalty += 0.1
        #     if dis > 0.3:
        #         penalty += 0.6
        #     if dis > 0.4:
        #         penalty += 0.9

        # if self.total_count >= 6000000:
        #     if dis > 0.15:
        #         penalty += 0.15
        #     if dis > 0.2:
        #         penalty += 0.6
        #     if dis > 0.3:
        #         penalty += 0.9
    
        # Total reward
        total_reward = reward + bonus - penalty
        # print("-------------\nDistance",dis,"\nReward",reward,"\nTotal reward",total_reward,"\n-------------\n")
        return float(total_reward)

    @torch.no_grad()
    def get_delta_orientation(self):
        q = self.scene["robot"].data.body_link_quat_w[:, self.gripper_idx]
        z_z = 1 - 2 * (q[:, 1]**2 + q[:, 2]**2)
        return float(torch.acos(torch.clamp(-z_z, -1.0, 1.0)).item())

    def calculate_distance_xy_z(self):
        cube = self.scene["cube"]
        gripper_center = (self.get_link_pos("panda_rightfinger") + self.get_link_pos("panda_leftfinger")) / 2
        cube_pos = cube.data.body_pos_w[:, 0]  # only one body
        
        # Calculate XY distance (horizontal plane)
        xy_distance = torch.norm(cube_pos[0][:2] - gripper_center[0][:2])
        
        # Calculate Z distance (vertical)
        z_distance = abs(cube_pos[0][2] - gripper_center[0][2])
        
        return xy_distance, z_distance

    def calculate_distance(self, array):
        cube = self.scene["cube"]
        gripper_center = (self.get_link_pos("panda_rightfinger") + self.get_link_pos("panda_leftfinger")) / 2
        # print("center",gripper_center)
        cube_pos = cube.data.body_pos_w[:, 0]  # only one body
        # print("cubepos",cube_pos)
        if array:
            distance_array = torch.stack([
                cube_pos[0][0] - gripper_center[0][0],
                cube_pos[0][1] - gripper_center[0][1],
                cube_pos[0][2] - gripper_center[0][2]
            ])
            return distance_array
        distance = torch.norm(cube_pos[0] - gripper_center[0])
        return distance

    def _get_observation(self):
        # Get observation: 3D distance + 1D delta orientation + 7D joint positions + 1D gripper state
        franka = self.scene["robot"]
        joint_pos = franka.data.joint_pos
        joint_pos_7 = joint_pos[0][:7]
        gr_pos = joint_pos[0][7:9]
        # print("joint pos ",joint_pos_7[0])
        g_status = torch.tensor([0.0 if gr_pos[0] < 0.02 else 1.0], device="cuda:0")
        distance = self.calculate_distance(True)
        delta_orientation = torch.tensor([self.delta_orientation], dtype=torch.float64, device="cuda:0")
        # print("distance ", distance[0])
        # print("gstatus",g_status[0])
        obs = torch.cat([distance, delta_orientation, joint_pos_7, g_status], dim=0)
        obs_numpy = obs.cpu().numpy().astype(np.float64)
        return obs_numpy

    def close(self):
        print(f"Success-rate: {self.suc_count} / {self.ep_count}")
        print("Simulation Shutdown")
        self.app.app.shutdown()        

    def get_link_pos(self,link_name):
        fdata = self.scene["robot"].data
        idx = fdata.body_names.index(link_name)
        link_pos = fdata.body_link_pos_w[:, idx]
        return link_pos
