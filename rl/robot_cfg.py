from pathlib import Path
import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.entity import EntityCfg

ROBOT_XML: Path = (Path(__file__).resolve().parent.parent / "xmls" / "open_duck_mini_v5.xml")
assert ROBOT_XML.exists(), f"Robot MJCF not found: {ROBOT_XML}"

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ROBOT_XML))

JOINT_NAMES = (
    "L_hip_yaw",
    "L_hip_roll",
    "L_hip_pitch",
    "L_knee",
    "L_ankle",
    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "R_ankle",
)

ROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES,),),
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.275),
    rot=(1.0, 0.0, 0.0, 0.0),

    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),

    joint_pos={
        "L_hip_yaw":   0.0,
        "L_hip_roll": -0.174532925,
        "L_hip_pitch": 0.349065850,
        "L_knee":     -0.698131701,
        "L_ankle":     0.349065850,

        "R_hip_yaw":   0.0,
        "R_hip_roll":  0.174532925,
        "R_hip_pitch": -0.349065850,
        "R_knee":      0.698131701,
        "R_ankle":    -0.349065850,
    },

    joint_vel={".*": 0.0},
)

FOOT_GEOM_NAMES = (
    "left_foot_collision",
    "right_foot_collision",
)

FOOT_SITE_NAMES = (
    "L6_IK",
    "R6_IK",
)

def get_robot_cfg() -> EntityCfg:
    """Return the Open Duck Mini V2 robot configuration."""

    return EntityCfg(
        spec_fn=get_spec,
        articulation=ROBOT_ARTICULATION,
        init_state=HOME_KEYFRAME,
    )