# RoboticArm-RL  

An RL project where an agent learns to control a Franka Emika Panda robotic arm to grasp a cube.
Built and tested with NVIDIA Isaac Sim and Isaac Lab.

---

## Features & Goals

- Use reinforcement learning to teach a robot arm (Franka Emika Panda) to pick up a cube.
- Train and evaluate in simulation (using Isaac Sim or other robotics-sim frameworks).
- Modular environment and training code to facilitate experimentation.
- Save and load trained models for inference or further training.
- (Optionally) Visualize arm motion, grasp success rate, and metrics over time.

---

## Repository Structure

```
RoboticArm-RL/
│
├── envs/
│ ├── __init__.py
│ └── franka_gym.py
| └── franka_gym_lvl2.py
| └── gym_environment_interface.py
| └── simple_grid_env.py
│
├── isaacsim assets/
│ ├── Cube.usd
| └── franka.usd
│
├── saved_models/
│ └── … (trained models, checkpoints)
│
├── custom_callback.py
├── scene_cfg.py
├── simapp_cfg.py
├── train_agent.py
├── train_agent_with_eval.py
├── test_agent.py
├── test_environment.py
├── test2.py
├── test_agent.py
├── test_environment.py
└── requirements.txt
```

## Setup Instructions

---

## Setup & Installation

### 1. Prerequisites
- **Isaac Sim** must be installed and working (see [Isaac Sim docs](https://developer.nvidia.com/isaac-sim)).  
- **Isaac Lab** must be installed and configured (see [Isaac Lab repo](https://github.com/isaac-sim/IsaacLab)).  
- GPU drivers and CUDA toolkit (matching Isaac Sim requirements).  

Make sure you can launch Isaac Sim standalone before continuing.

### 2. Clone the Repository

Next clone the repository to your local machine:

```bash
git clone [URL-to-your-repo]
cd grid_problem
```

### 3. Create Conda Environment
```bash
conda create -n roboticarm-rl python=3.10
conda activate roboticarm-rl
```

### 4. Run the scrips
```bash
roboticarm-rl train_agent.py # add "--headless" for no ui
```
