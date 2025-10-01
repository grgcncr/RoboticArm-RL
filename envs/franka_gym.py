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
class FrankaGym(gym.Env, GymEnvironmentInterface):
    metadata = {"render_modes": [], 'render_fps' : 30}
    def __init__(self, app, sim, scene):
        super(FrankaGym, self).__init__()
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

        # ===== CONTACT HANDLING =====
        max_force = torch.max(contact_sensor.data.net_forces_w).item()
        force_magnitudes = torch.norm(contact_sensor.data.force_matrix_w, dim=-1)
        contact_detected = (force_magnitudes != 0.0).any().item()

        if max_force > 0.0 and contact_detected:  # Cube contact
            self.cube_contact = True
            cube_vel = cube.data.body_vel_w
            max_cvel = cube_vel.abs().max().item()
            # print(f"{self.total_count} Contact: {max_cvel:.2f}", end="")
            if max_cvel > 1.0:
                terminated = True

        reward = self.calculate_reward() 
        
        if reward == 100:#or reward == -1: # reward == -1 or reward == 2 or     
            # print(f"Cube pos: {cube_pos}\n Franka pos: {action}\n")
            terminated = True

        if reward == -1:
            terminated = True
    
        if trancated or terminated:
            self.ep_count += 1
            self.flag = True # update suc_rate only in the 1st 100,200,300...th step

        if self.ep_count % 100 == 0 and self.flag:
            self.suc_rate = self.ep_suc_count / 100

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
        return observation , {}
    
    def render(self, mode='human'):
        """Nothing needed since Isaac Sim is rendering automatically."""
        pass

    def calculate_reward(self):
        # cube = self.scene["cube"]
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

        # if self.count <= 5:
        #     print(f"distance: {dis}")
        #     print(f"link3-z: {link3_pos[0][2]}")
        #     print("xy:",dis_xy,"\nz: ", dis_z, "\nori: ",delta_orientation)
        
        # Contact Handling: terminate only if ground contact (not cube contact)
        if max_force > 0.0 and not contact_detected:
            # Ground contact (non-cube collision)
            if self.total_count % 512000 == 0:
                print(f"Success counter: {self.suc_count}")
            return -1  # Immediate penalty and terminate
        
        # Terminal condition: successful grasp (open gripper close to cube)
        # if self.cube_contact == True:
        #     print(f", G_status: {g_status}, Ori: {delta_orientation:.2f},\n Dis: {dis:.2f}, Dis_xy: {dis_xy:.2f}, Dis_z: {dis_z:.2f} Lmz: {link_meanz:.2f}")
        self.cube_contact = False
        if not contact_detected and dis_z <= 0.06 and dis_xy <= 0.015 and g_status == 1.0 and delta_orientation < 0.5 and link_meanz > 0.4:
            # print(f"\n===SUCCESS=== --> Ori: {delta_orientation:.2f}, linkmean-z: {link_meanz:.2f}\n Distance: {dis:.2f}, x-y: {dis_xy}, z: {dis_z} ")
            self.suc_count += 1
            self.ep_suc_count += 1
            print(f"Success counter: {self.suc_count}")
            print(f"Success episode counter: {self.ep_suc_count}")
            if self.total_count % 512000 == 0:
                print(f"Success counter: {self.suc_count}")
            return 100  # Success
        if self.total_count % 512000 == 0:
            print(f"Success counter: {self.suc_count}")
        # if self.count <= 5:
        #     print(f"link3 - z: {link3_pos[0][2]}, Joit3: {joint3}")
        #     print(f"link4 - z: {link4_pos[0][2]}")
        xy_reward = math.exp(-3.5 * dis_xy)
        # print("xy_reward: ",xy_reward)
        z_reward = math.exp(-3.5 * dis_z)
        # print("z_reward: ",z_reward)
        orientation_reward = math.exp(-1.5 * delta_orientation)
        # print("ori_reward: ", orientation_reward)
        reward = 0.45 * xy_reward + 0.35 * z_reward + 0.2 * orientation_reward
        # print(f"gstatus {g_status}")
        # time.sleep(5)
        
        bonus = 0.0

        if delta_orientation <= 0.5 and dis < 0.7:
            bonus += 0.05

        if g_status == 1.0:
            bonus += 0.05

        if joint3 <= -0.7 and joint3 >= -2.7:
            bonus += 0.025

        if link_meanz > 0.4 and link_meanz <= 0.62 and dis < 0.7:#and dis_z < 0.4: and link3_pos[0][2] < 0.6
            bonus += 0.025

        if link6_pos[0][2] < link_meanz:
            bonus += 0.02

        if not contact_detected and g_status == 1.0 and delta_orientation < 0.5:
            if dis <= 0.07:
                bonus += 1.0
            elif dis <= 0.08:
                bonus += 0.5
            elif dis <= 0.09:
                bonus += 0.3
            elif dis <= 0.12:
                bonus += 0.1
            # elif dis <= 0.2:
            #     bonus += 0.1     

        penalty = 0

        if joint3 > -0.7:
            penalty += 0.15

        if joint3 < -2.7:
            penalty += 0.2

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



    # def capture_images(self, camera, path):
    #     rgb_tensor = self.scene[camera].data.output["rgb"]
    #     rgb_np = rgb_tensor.squeeze(0).cpu().numpy()  
    #     save_dir = Path(path).expanduser()
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     image_path = save_dir / f"camera_output_{self.total_count}.png"
    #     imageio.imwrite(str(image_path), rgb_np)
    #     # print(f"--------Saved {image_path}--------")
    #     # # For 10 images
    #     # if rep == 10:
    #     #     FrankaGym.close()



    # def calculate_reward(self):
    #     contact_sensor = self.scene["contact_sensor"]
    #     cube = self.scene["cube"]
    #     cube_pos = cube.data.body_pos_w[0, :3].squeeze(0)
    #     # Get current observations
    #     obs = self._get_observation()
    #     gripper_center = self.get_gripper_center().squeeze(0)
    #     gripper_status = obs[10]  # Gripper state (0=closed, 1=open)
    #     fdata = self.scene["robot"].data
    #     link_names = fdata.body_names
    #     # idx_link3 = link_names.index("panda_link3")
    #     # idx_link6 = link_names.index("panda_link6")
    #     # pos_link3 = fdata.body_link_pos_w[:, idx_link3].squeeze(0)
    #     # pos_link6 = fdata.body_link_pos_w[:, idx_link6].squeeze(0)
    #     penalty = 0.0
    #     progress_reward = 0.0
    #     alignment_reward = 0.0 
    #     distance_reward = 0.0
    #     gripper_state_reward = 0.0
    #     # Calculate distance to cube
    #     distance = self.calculate_distance(False)
    #     # Initialize tracking variables on first call
    #     if self.prev_distance is None:
    #         self.prev_distance = distance
    #         self.prev_gripper_center = gripper_center.clone()
    #         self.position_history = [gripper_center.clone()]
    #         self.distance_history = [distance]
    #         self.progress_history = [0]            
    #         self.joint_pos_history = [torch.tensor((list(obs[-8:-1]) + [0.0, 0.0] if obs[-1]==0.0 else [0.04, 0.04]), dtype=torch.float64, device="cuda:0")]
    #     distance_progress = self.prev_distance - distance

    #     # Update position and distance history
    #     self.position_history.append(gripper_center.clone())
    #     self.distance_history.append(distance)
    #     self.progress_history.append(distance_progress)
    #     self.joint_pos_history = [torch.tensor((list(obs[-8:-1]) + [0.0, 0.0] if obs[-1]==0.0 else [0.04, 0.04]), dtype=torch.float64, device="cuda:0")]
    #     if len(self.position_history) > 10:
    #         self.position_history.pop(0)
    #     if len(self.distance_history) > 10:
    #         self.distance_history.pop(0)
    #     if len(self.progress_history) > 10:
    #         self.progress_history.pop(0)
    #     if len(self.joint_pos_history) > 10:
    #         self.joint_pos_history.pop(0)
            
    #     # if self.count <= 300000:
    #     #     distance_reward = 0.8 * (1 - (distance / 1.5))
    #     #     gripper_state_reward = 0.0
    #     #     if gripper_status == 1.0 and distance < 0.3:
    #     #         gripper_state_reward += 0.2

    #     # else:
    #     # ===== COMPONENT 1: DISTANCE REWARD (35% of total) =====
    #     distance_reward = 0.35 * (1 - (distance / 1.5))
    #     gripper_state_reward = 0.0
    #     # ===== COMPONENT 2: GRIPPER STATE REWARD (15% of total) =====
    #     if gripper_status == 1.0 and distance < 0.3:
    #         gripper_state_reward += 0.15
    #     # ===== COMPONENT 3: PROGRESS REWARD (25% of total) =====
        
    #     # Progress based on distance improvement
        
    #     if distance_progress > 0:
    #         progress_reward += 0.125 * min(distance_progress * 5, 1.0)  # Reduced scaling from 10 to 5
    #     elif distance_progress < -0.01:  # Moving away penalty
    #         progress_reward -= 0.125
            
    #     # Velocity-based progress (moving toward cube)
    #     if self.prev_gripper_center is not None:
    #         gr_movement = torch.norm(gripper_center - self.prev_gripper_center)
    #         if gr_movement > 0.001:  # Only if there's actual movement
    #             # Check if movement is toward the cube
    #             old_cube_dist = torch.norm(self.prev_gripper_center - cube_pos)
    #             new_cube_dist = torch.norm(gripper_center - cube_pos)
    #             if new_cube_dist < old_cube_dist:  # Moving closer
    #                 progress_reward += 0.125 * min(gr_movement * 10, 1.0)
        
    #     # Store current progress for next iteration
    #     # current_progress = distance_progress # if distance_progress > 0 else 0.0
        
    #     # Consistent gripper state reward regardless of distance

        
    #     # ===== COMPONENT 4: ALIGNMENT REWARD (25% of total) =====       
    #     robot_base_pos = torch.tensor([0.0, 0.0, 0.0], device=cube_pos.device)
    #     cube_to_base_vector = robot_base_pos - cube_pos
    #     cube_to_base_distance = torch.norm(cube_to_base_vector)
    #     cube_to_base_direction = cube_to_base_vector / cube_to_base_distance
        
    #     try:
    #         # Get link names and find panda_link6 index
    #         link_names = fdata.body_names
    #         idx_link6 = link_names.index("panda_link6")
            
    #         # Get orientation quaternion for panda_link6
    #         link6_quat = fdata.body_quat_w[0, idx_link6]  # [w, x, y, z] format
            
    #         # Convert quaternion to rotation matrix to get orientation
    #         w, x, y, z = link6_quat[0], link6_quat[1], link6_quat[2], link6_quat[3]
            
    #         # Get the Z-axis direction of the link (third column of rotation matrix)
    #         z_axis = torch.tensor([
    #             2.0 * (x * z + w * y),
    #             2.0 * (y * z - w * x),
    #             1.0 - 2.0 * (x * x + y * y)
    #         ], device=cube_pos.device)
            
    #         # Check if Z-axis is pointing vertically (close to [0, 0, 1] or [0, 0, -1])
    #         # vertical_up = torch.tensor([0.0, 0.0, 1.0], device=cube_pos.device)
    #         vertical_down = torch.tensor([0.0, 0.0, -1.0], device=cube_pos.device)
            
    #         # dot_up = torch.dot(z_axis, vertical_up)
    #         dot_down = torch.dot(z_axis, vertical_down)
            
    #         # Reward for being vertical (either up or down)
    #         vertical_alignment = abs(dot_down) # max(abs(dot_up), )
    #         alignment_reward += 0.125 * vertical_alignment
            
    #     except (ValueError, IndexError):
    #         # If panda_link6 not found, no penalty but no reward
    #         pass
        
    #     try:
    #         # Get panda_link3 position
    #         idx_link3 = link_names.index("panda_link3")
    #         link3_pos = fdata.body_pos_w[0, idx_link3, :3]
            
    #         # Check height constraint first
    #         if link3_pos[2] < 0.5:
    #             alignment_reward -= 0.1  # Penalty for being too low (capped at -15%)
    #         else:
    #             # Check alignment with cube-to-base line
    #             if cube_to_base_distance > 0.001:
    #                 link3_to_base_vector = robot_base_pos - link3_pos
    #                 link3_to_base_distance = torch.norm(link3_to_base_vector)
                    
    #                 if link3_to_base_distance > 0.001:
    #                     link3_to_base_direction = link3_to_base_vector / link3_to_base_distance
                        
    #                     # Calculate alignment with cube-to-base line
    #                     link3_alignment_dot = torch.dot(cube_to_base_direction, link3_to_base_direction)
    #                     link3_alignment_score = max(0.0, link3_alignment_dot)
                        
    #                     # Distance from link3 to the ideal line
    #                     link3_to_cube_vector = link3_pos - cube_pos
    #                     link3_projection_length = torch.dot(link3_to_cube_vector, cube_to_base_direction)
    #                     link3_projection_point = cube_pos + link3_projection_length * cube_to_base_direction
    #                     link3_distance_to_line = torch.norm(link3_pos - link3_projection_point)
                        
    #                     # Reward being close to the line
    #                     link3_line_proximity = max(0.0, 1.0 - link3_distance_to_line * 2.0)
                        
    #                     alignment_reward += 0.125 * (link3_alignment_score + link3_line_proximity)
                        
    #     except (ValueError, IndexError):
    #         # If panda_link3 not found, no penalty but no reward
    #         pass
        
        
    #     # PENALTY 1: Staying in the same position (stagnation)
    #     if len(self.joint_pos_history) >= 3:
    #         all_joint_positions = torch.stack(self.joint_pos_history) 
    #         joint_variance = torch.var(all_joint_positions, dim=0).mean()
    #         if joint_variance < 0.0001 and distance > 0.2:
    #             penalty += 0.15

    #     if len(self.progress_history) >= 5:
    #         all_progress = torch.stack(self.joint_pos_history).mean()
    #         if all_progress < 0.0:
    #             penalty += 0.2

    #     if len(self.distance_history) >= 5:
    #         distance_variance = np.var(self.distance_history)
    #         mean_distance = np.mean(self.distance_history)
            
    #         # If staying at similar distance (low variance) and distance is small but not decreasing
    #         if distance_variance < 0.001 and 0.03 < mean_distance < 0.6:
    #             penalty += 0.15  # Capped at -15%

    #     # PENALTY 4: Oscillation / Hovering between two points
    #     if len(self.position_history) >= 6:  # Need enough history
    #         recent_positions = torch.stack(self.position_history[-6:])
            
    #         # Compute pairwise distances (to detect position switches)
    #         dists = torch.norm(recent_positions[1:] - recent_positions[:-1], dim=1)
            
    #         # Check if the distances form a high-low-high-low pattern (oscillation)
    #         # Compute difference between adjacent steps
    #         delta_signs = torch.sign(dists[1:] - dists[:-1])
            
    #         # Count how often the sign changes (indicative of bouncing behavior)
    #         sign_changes = (delta_signs[:-1] * delta_signs[1:] < 0).sum().item()
            
    #         # Apply penalty if many alternations and not making progress
    #         if sign_changes >= 2 and distance > 0.1:
    #             penalty += 0.15  # Adjust as needed
        
    #     # ===== CONTACT HANDLING =====
    #     max_force = torch.max(contact_sensor.data.net_forces_w).item()
    #     force_magnitudes = torch.norm(contact_sensor.data.force_matrix_w, dim=-1)
    #     contact_detected = (force_magnitudes != 0.0).any().item()
        
    #     if max_force > 0.0 and contact_detected:  # Cube contact
    #         self.cube_contact = True
    #         cube_vel = cube.data.body_vel_w
    #         max_cvel = cube_vel.abs().max().item()
    #         print("Franka Touched the Cube --> ", max_cvel)
    #         if max_cvel > 1.0:  # High force contact
    #             return -1.0
    #         else:
    #             penalty += 0.1  # Gentle contact penalty (capped at -15%)
    #     elif max_force > 0.0:  # Ground contact
    #         return -2.0
            
    #     # ===== TERMINAL SUCCESS =====
    #     if distance < 0.03 and gripper_status == 1.0:
    #         return 100.0
            
    #     # ===== COMBINE ALL REWARDS =====
    #     total_reward = distance_reward + progress_reward + gripper_state_reward + alignment_reward - penalty
    #     if torch.is_tensor(total_reward):
    #         total_reward = total_reward.cpu().numpy().item()
    #     # Update tracking variables
    #     self.prev_distance = distance
    #     self.prev_gripper_center = gripper_center.clone()
        
    #     # Ensure reward stays within bounds and max reward is 1.0 (excluding terminal)
    #     return max(-1.0, min(total_reward, 1.0))



        # cube_center_x = cube_pos[0]
        # cube_center_y = cube_pos[1]
        # cube_center_line_a = cube_center_y / cube_center_x
        # # Robot alignment reward - check if robot is aligned with cube-to-base line
        # # robot_base_pos = torch.tensor([0.0, 0.0, 0.0], device=cube_pos.device)
        # fdata = self.scene["robot"].data
        # link_names = fdata.body_names
        # idx_l3 = link_names.index("panda_link3")
        # pos_l3 = fdata.body_link_pos_w[:, idx_l3].squeeze(0).cpu().numpy()
        # l3_x = pos_l3[0]
        # l3_y = pos_l3[1]
        # l3_a = l3_y / l3_x
        
        # sector_point_x  = (l3_x * l3_a) - (cube_center_x * cube_center_line_a)
        # sector_point_y = sector_point_x * cube_center_line_a
        # sector_dis = np.sqrt((sector_point_x - l3_x)**2 + (sector_point_y - l3_y)**2)



# def calculate_reward(self):
#         contact_sensor = self.scene["contact_sensor"]
#         cube = self.scene["cube"]        
#         max_force = torch.max(contact_sensor.data.net_forces_w).item()
#         force_magnitudes = torch.norm(contact_sensor.data.force_matrix_w, dim=-1)
#         contact_detected = (force_magnitudes != 0.0).any().item()
#         distance = self.calculate_distance(False)
#         if self.prev_distance is None:
#             self.prev_distance = distance
#         reward = 0

#         reward = 1 - (distance / 1.5)

#         # Gripper state
#         g_status = self._get_observation()[10]
#         if g_status == 1.0:
#             if distance < 0.1:
#                 reward += 0.2
#             elif distance < 0.3:
#                 reward += 0.1
#         else :
#             if distance < 0.2:
#                 reward -= 0.2
        
#         # Contact handling
#         if max_force > 0.0 and contact_detected: # cube contact         
#             self.cube_contact = True
#             cube_vel = cube.data.body_vel_w
#             max_cvel = cube_vel.abs().max().item()
#             print("Franka Touched the Cube --> ", max_cvel)
#             if max_cvel > 1.0: # with force
#                 return -1
#             else:
#                 reward -= 0.2 # without force
#         elif max_force > 0.0 : # the only other contact is the groundplane
#             return -2

#         # Terminal condition
#         if distance < 0.03 and g_status == 1.0:
#             return 100.0

#         return min(reward, 1.0)








    


# def get_random_joint_velocities(scale_arm=4.0, scale_gripper=2.5):
#     arm_vel = (np.random.rand(7) - 0.5) * 2 * scale_arm
#     gripper_vel = np.random.uniform(-1, 1) * scale_gripper
#     gripper = np.array([gripper_vel, gripper_vel])
#     joint_vel = np.concatenate([arm_vel, gripper])
#     return torch.tensor(joint_vel, dtype=torch.float64, device="cuda:0").unsqueeze(0)

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
#     return torch.tensor(joint_pos, dtype=torch.float64, device="cuda:0").unsqueeze(0)
#     # return torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.04, 0.04], dtype=torch.float64, device="cuda:0").unsqueeze(0)


# def simulation_run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
#     rep = 0
#     franka = scene["robot"] 
#     # Init state
#     init_target = torch.tensor([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.01, 0.01], dtype=torch.float64, device="cuda:0").unsqueeze(0)
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











        
        
        # ===SUCCESS=== --> Ori: 0.33, link3-z: 0.57
        # Distance: 0.04, x-y: 0.01, z: 0.04 
        # Cube pos: tensor([[[ 0.5305, -0.1336,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [ 0.14995119  0.69500041 -0.39965433 
        # -1.94331467  0.34194258  2.22981501
        # -2.8973      0.67055678]

        # ===SUCCESS=== --> Ori: 0.12, link3-z: 0.59
        # Distance: 0.05, x-y: 0.03, z: 0.03 
        # Cube pos: tensor([[[0.4147, 0.4214, 0.0250]]], 
        # device='cuda:0')
        # Franka pos: [ 0.69487011  0.61776996  0.09330231 
        # -2.07425714 -0.06326482  2.56873226
        # -2.45136905  1.        ]

        # ===SUCCESS=== --> Ori: 0.24, link3-z: 0.58
        # Distance: 0.04, x-y: 0.01, z: 0.04 
        # Cube pos: tensor([[[ 0.5198, -0.1569,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [ 0.13000549  0.66015613 -0.43844473 
        # -2.03546238  0.37563819  2.36542869
        # -2.8973      1.        ]

        # ===SUCCESS=== --> Ori: 0.28, link3-z: 0.56
        # Distance: 0.05, x-y: 0.04, z: 0.04 
        # Cube pos: tensor([[[-0.4545,  0.4271,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [ 2.10418344  0.76042145  0.28428364 
        # -1.78696728 -0.47059396  2.23366666
        # 0.48970163  1.        ]

        # ===SUCCESS=== --> Ori: 0.20, link3-z: 0.57
        # Distance: 0.04, x-y: 0.02, z: 0.04 
        # Cube pos: tensor([[[ 0.4114, -0.4205,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [-0.32192889  0.70365196 -0.51841462 
        # -2.00791502  0.80550206  2.39520764
        # -2.8973      1.        ]

        # ===SUCCESS=== --> Ori: 0.08, link3-z: 0.55
        # Distance: 0.05, x-y: 0.03, z: 0.03 
        # Cube pos: tensor([[[ 0.4533, -0.4515,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [-0.2036905   0.80050337 -0.53826392 
        # -1.93972349  0.6071614   2.59789896
        # -2.8973      1.        ]






        # ===SUCCESS=== --> Ori: 0.27, link3-z: 0.60
        # Distance: 0.05, x-y: 0.01, z: 0.05 
        # Cube pos: tensor([[[-0.1567, -0.4650,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [-1.30468345  0.5637759  -0.59946591 
        # -2.25347853  0.64572418  2.3631084
        # 0.52186209  1.        ]

        # ===SUCCESS=== --> Ori: 0.05, link3-z: 0.61
        # Distance: 0.05, x-y: 0.01, z: 0.05 
        # Cube pos: tensor([[[0.1515, 0.4629, 0.0250]]], 
        # device='cuda:0')
        # Franka pos: [ 0.94213212  0.47197792  0.29257882 
        # -2.37295818 -0.2980344   2.77117467
        # -0.08048934  1.        ]







        # ===SUCCESS=== --> Ori: 0.16, link3-z: 0.53
        # Distance: 0.06, x-y: 0.0146022355183959, z: 
        # 0.054579611867666245 
        # Cube pos: tensor([[[-0.4105,  0.1556,  0.0250]]], 
        # device='cuda:0')
        # Franka pos: [ 1.56358981  0.91938818  1.15408039 
        # -2.39678192 -1.2282356   2.16234732
        # -2.8973      1.        ]



