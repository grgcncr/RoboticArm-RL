from gymnasium.envs.registration import register
register (
    id = 'FrankaGymEnv',
    entry_point = 'envs.franka_gym:FrankaGym',
)
