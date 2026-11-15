from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.envs import mdp as env_mdp
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from rl.gait_observation import gait_observation, gait_reference_contact
from mjlab.envs.mdp.actions import JointPositionActionCfg
from rl.robot_cfg import JOINT_NAMES
from mjlab.managers.reward_manager import RewardTermCfg
from rl.gait_reward import gait_contact_matching
from rl.cwc_reward import cwc_penalty
from mjlab.managers.scene_entity_config import SceneEntityCfg
import math
from mjlab.envs import mdp as env_mdp
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

# Physics simulation
PHYSICS_DT = 0.0005

# Policy control
DECIMATION = 40
POLICY_DT = PHYSICS_DT * DECIMATION

# Episode
EPISODE_LENGTH_S = 10.0
IS_FINITE_HORIZON = False
SCALE_REWARDS_BY_DT = True

def get_sim_cfg() -> SimulationCfg:
    """Return the Open Duck Mini V2 simulation configuration."""

    return SimulationCfg(
        mujoco=MujocoCfg(
            timestep=PHYSICS_DT,
            integrator="implicitfast",
            solver="newton",
            cone="elliptic",
            tolerance=1e-10,
            disableflags=(),
        ),
    )

def get_command_cfg() -> UniformVelocityCommandCfg:
    """Return the velocity command configuration."""

    return UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        rel_world_envs=0.0,
        heading_command=False,
        init_velocity_prob=0.0,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.1),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )

def get_observation_cfg() -> dict[str, ObservationGroupCfg]:
    """Return actor and critic observation configurations."""

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=env_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),

        "base_ang_vel": ObservationTermCfg(
            func=env_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),

        "projected_gravity": ObservationTermCfg(
            func=env_mdp.projected_gravity,
        ),

        "joint_pos": ObservationTermCfg(
            func=env_mdp.joint_pos_rel,
            params={"biased": True},
        ),

        "joint_vel": ObservationTermCfg(
            func=env_mdp.joint_vel_rel,
        ),

        "actions": ObservationTermCfg(
            func=env_mdp.last_action,
        ),

        "command": ObservationTermCfg(
            func=env_mdp.generated_commands,
            params={"command_name": "twist"},
        ),

        "gait_phase": ObservationTermCfg(
            func=gait_observation,
            params={"command_name": "twist"},
        ),
    }

    critic_terms = {
        **actor_terms,

        "joint_pos": ObservationTermCfg(
            func=env_mdp.joint_pos_rel,
        ),

        "gait_contact_ref": ObservationTermCfg(
            func=gait_reference_contact,
            params={"command_name": "twist"},
        ),

        "foot_contact": ObservationTermCfg(
            func=velocity_mdp.foot_contact,
            params={"sensor_name": "feet_contact"},
        ),

        "foot_air_time": ObservationTermCfg(
            func=velocity_mdp.foot_air_time,
            params={"sensor_name": "feet_contact"},
        ),
    }

    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),

        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

ACTION_SCALE = 0.15

def get_action_cfg() -> JointPositionActionCfg:
    """Return the joint position action configuration."""

    return JointPositionActionCfg(
        entity_name="robot",
        actuator_names=JOINT_NAMES,
        scale=ACTION_SCALE,
        use_default_offset=True,
    )

def get_reward_cfg() -> dict[str, RewardTermCfg]:
    """Return the Open Duck Mini V2 reward configuration."""
    return {
        "track_linear_velocity": RewardTermCfg(
            func=velocity_mdp.track_linear_velocity,
            weight=3.0,
            params={
                "command_name": "twist",
                "std": 0.1,
            },
        ),

        "track_angular_velocity": RewardTermCfg(
            func=velocity_mdp.track_angular_velocity,
            weight=0.5,
            params={
                "command_name": "twist",
                "std": 0.5,
            },
        ),


        "action_rate_l2": RewardTermCfg(
            func=env_mdp.action_rate_l2,
            weight=-0.002,
        ),


        "joint_torques_l2": RewardTermCfg(
            func=env_mdp.joint_torques_l2,
            weight=-0.00005,
        ),

        "posture": RewardTermCfg(
            func=env_mdp.posture,
            weight=0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JOINT_NAMES,
                ),
                "std": {
                    r".*": 0.35,
                },
            },
        ),

        "gait_contact_matching": RewardTermCfg(
            func=gait_contact_matching,
            weight=0.1,
            params={
                "command_name": "twist",
                "sensor_name": "feet_contact",
            },
        ),

        "cwc_penalty": RewardTermCfg(
            func=cwc_penalty,
            weight=-0.1,
            params={
                "sensor_name": "feet_wrench",
                "mu": 0.8,
                "min_normal_force": 5.0,
                "safe_margin": 0.2,
                "weight_friction": 1.0,
                "weight_cop": 1.0,
                "weight_yaw": 1.0,
            },
        ),
    }

def get_termination_cfg() -> dict[str, TerminationTermCfg]:
    return {
        "bad_orientation": TerminationTermCfg(
            func=env_mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0),},
        ),

        "nan_detection": TerminationTermCfg(func=env_mdp.nan_detection,),

        "time_out": TerminationTermCfg(func=env_mdp.time_out,time_out=True,),
    }

def get_curriculum_cfg() -> dict[str, CurriculumTermCfg]:
    return {
        "action_rate_curriculum": CurriculumTermCfg(
            func=env_mdp.reward_curriculum,
            params={
                "reward_name": "action_rate_l2",
                "stages": [
                    {"step": 0,    "weight": -0.002},
                    {"step": 600,  "weight": -0.004},
                    {"step": 1200, "weight": -0.006},
                    {"step": 1800, "weight": -0.008},
                    {"step": 2400, "weight": -0.010},
                ],
            },
        ),

        "joint_torque_curriculum": CurriculumTermCfg(
            func=env_mdp.reward_curriculum,
            params={
                "reward_name": "joint_torques_l2",
                "stages": [
                    {"step": 0,    "weight": -0.00005},
                    {"step": 600,  "weight": -0.00010},
                    {"step": 1200, "weight": -0.00020},
                    {"step": 1800, "weight": -0.00035},
                    {"step": 2400, "weight": -0.00050},
                ],
            },
        ),
    }