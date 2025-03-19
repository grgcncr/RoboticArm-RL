
# Simple Grid Environment for RL Training

This project demonstrates how to create and use a simple grid-based environment for reinforcement learning using Python, Gym, and Stable Baselines3.

## Project Structure

```
robotics-rl/
│
├── envs/
│   ├── __init__.py
│   ├── gym_environment_interface.py
│   └── simple_grid_env.py
│
├── train_agent.py
├── test_agent.py
│
├── saved_models/
│
└── requirements.txt
```

## Setup Instructions

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone [URL-to-your-repo]
cd grid_problem
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv rl-examples

# Activate the virtual environment
# On Windows
rl-examples\Scripts\activate
# On MacOS/Linux
source rl-examples/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Using the Environment

### Preparing the Environment

Before training or testing the agents, ensure the custom environment `SimpleGridEnv` is correctly set up. This environment is defined in the `envs/simple_grid_env.py` file and uses an interface defined in `envs/gym_environment_interface.py`.

### Training the Agent

To train the agent using the provided environment:

```bash
python3 train_agent.py
```

This script configures and trains an agent using the PPO algorithm, saving the model in the `saved_models/` directory.

### Testing the Agent

To test the trained model:

```bash
python test_agent.py
```

This script loads the trained model and evaluates its performance across multiple episodes, rendering the environment state to visualize the agent's actions.

## Using the Interface

The `GymEnvironmentInterface` defined in `gym_environment_interface.py` provides an abstract base class that ensures all custom environments implement necessary methods like `step`, `reset`, `render`, and `calculate_reward`. Implement this interface to ensure compatibility and consistency when creating new environments.

