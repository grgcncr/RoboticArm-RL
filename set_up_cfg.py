from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObject, RigidObjectCfg
from pathlib import Path
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    FRANKA_PANDA_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.6,
                "panda_joint3": 0.0,
                "panda_joint4": -2.2,
                "panda_joint5": 0.0,
                "panda_joint6": 1.7,
                "panda_joint7": 0.8,
                "panda_finger_joint.*": 0.04,
            },# prev start pos (0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8, 0.05, 0.05)
        ),
        # actuators={
        #     "panda_shoulder": ImplicitActuatorCfg(
        #         joint_names_expr=["panda_joint[1-4]"],
        #         effort_limit=87.0,
        #         velocity_limit=2.175,
        #         stiffness=80.0,
        #         damping=1.0,
        #         control_mode="velocity",
        #     ),
        #     "panda_forearm": ImplicitActuatorCfg(
        #         joint_names_expr=["panda_joint[5-7]"],
        #         effort_limit=12.0,
        #         velocity_limit=2.61,
        #         stiffness=80.0,
        #         damping=1.0,
        #         control_mode="velocity",
        #     ),
        #     "panda_hand": ImplicitActuatorCfg(
        #         joint_names_expr=["panda_finger_joint.*"],
        #         effort_limit=200.0,
        #         velocity_limit=0.2,
        #         stiffness=2e3,
        #         damping=10.0,
        #         control_mode="velocity",
        #     ),
        # },
        actuators = {
            "panda_shoulder": IdealPDActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                effort_limit_sim=200.0,
                velocity_limit=2.175,
                velocity_limit_sim=2.175,
                stiffness=50.0,   # ensure no spring override
                damping=2.0       # minimal damping for stability
            ),
            "panda_forearm": IdealPDActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                effort_limit_sim=50.0,
                velocity_limit=2.61,
                velocity_limit_sim=2.61,
                stiffness=50.0,
                damping=2.0
            ),
            "panda_hand": IdealPDActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                effort_limit_sim=400.0,
                velocity_limit=0.2,
                velocity_limit_sim=0.2,
                stiffness=50.0,
                damping=2.0
            ),
        },
        soft_joint_pos_limit_factor=1.0,
        prim_path = "{ENV_REGEX_NS}/Franka",
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka")
    
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),  # 5cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001,rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
                metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # hand_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Franka/panda_hand/hand_camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.05, 10.0),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, -0.7), 
    #         rot=(0.70711, 0.0, 0.0, 0.70711),  # (x,w,z,y))!!
    #         convention="ros",
    #     ),
    # )

    # side_camera = CameraCfg(
    #     prim_path="/World/side_camera",  
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 100.0),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 3.5, 0.5),  
    #         rot=(0.0, 0.0, 0.70711, -0.70711), 
    #     ),
    # )

    
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Franka/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )