# step_timing_adaptation/run.py
#
# Current experiment:
#   - 9 s automatic walking
#   - vx = +0.035 m/s
#   - vy = -0.020 m/s
#   - COM_HEIGHT = 0.215 m
#   - SWING_HEIGHT = 0.020 m
#   - external push disabled
#
# Contact processing:
#   - old contact_stability.py is no longer used
#   - no friction cone / unilateral-force / torque-limit check
#   - q is generated kinematically by differential IK integration
#   - qdot is taken directly from the differential-IK solution
#     and converted from Pinocchio to MuJoCo velocity convention
#   - qdd is reconstructed offline from this direct qdot history
#   - MuJoCo provides M(q), qfrc_bias, qfrc_passive,
#     contact points, and contact Jacobians
#   - only the first six floating-base equations are used:
#
#       M_b(q) qdd + h_b(q, qdot) = J_c,b(q)^T f_c
#
#   - when the contact-force distribution is underdetermined,
#     the minimum-norm solution is selected

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ROBOT_XML = ROOT_DIR / "xmls" / "open_duck_mini_v2.xml"
SCENE_XML = ROOT_DIR / "xmls" / "scene_flat_terrain.xml"


# ============================================================
# LOCAL MODULES
# ============================================================

if __package__:

    from .adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from .online_swing_trajectory import (
        OnlineQuinticSwingTrajectory,
    )

    from .lipm_state_generator import (
        PointFootLIPM,
    )

    from .simulation_plot import (
        SimulationLog,
        plot_simulation_results,
    )

    from .contact_force_reconstruction import (
        ContactForceReconstructor,
        plot_contact_force_results,
    )

    from .data_logger import (
        JointAngleDataLogger,
    )

else:

    from adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from online_swing_trajectory import (
        OnlineQuinticSwingTrajectory,
    )

    from lipm_state_generator import (
        PointFootLIPM,
    )

    from simulation_plot import (
        SimulationLog,
        plot_simulation_results,
    )

    from contact_force_reconstruction import (
        ContactForceReconstructor,
        plot_contact_force_results,
    )

    from data_logger import (
        JointAngleDataLogger,
    )


# ============================================================
# EXISTING PROJECT MODULES
# ============================================================

from lipm_mpc.run import settle_robot

from footstep_planning.pinocchio_model import (
    PinocchioModel,
)

from footstep_planning.differential_ik import (
    TRUNK_FRAME,
    solve_single_support_ik,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# EXECUTOR / EXPERIMENT
# ============================================================

DT = 0.0005
SIMULATION_DURATION = 9.0

FIRST_STANCE_SIDE = "left"

SHOW_VIEWER = True
REALTIME_PLAYBACK = True

VIEWER_SYNC_PERIOD = 0.02
STATUS_PRINT_PERIOD = 0.10

TIME_TOLERANCE = 1.0e-10


# ============================================================
# CONTACT FORCE RECONSTRUCTION
# ============================================================
#
# True:
#   - record qpos and direct differential-IK qvel at every sample
#   - reconstruct qdd offline from the direct qvel history
#   - reconstruct point-contact forces from the first six
#     floating-base dynamic equations
#
# False:
#   - skip this reconstruction completely
#
# No friction-cone or unilateral-force constraints are applied.
# ============================================================

ENABLE_CONTACT_FORCE_RECONSTRUCTION = True


# ============================================================
# VELOCITY COMMAND
# ============================================================

def automatic_velocity_profile(
    current_time: float,
) -> tuple[float, float]:

    _ = float(
        current_time
    )

    return (
        0.035,
        -0.020,
    )


# ============================================================
# LIPM
# ============================================================

GRAVITY = 9.81
COM_HEIGHT = 0.215


# ============================================================
# EXTERNAL PUSH
# ============================================================

ENABLE_AUTOMATIC_PUSH = False

PUSH_TIME = 5.0

PUSH_FORCE_X = 0.0
PUSH_FORCE_Y = 0.0

PUSH_DURATION = 0.05


# ============================================================
# STEP BOUNDS
# ============================================================

STEP_LENGTH_MIN = -0.03
STEP_LENGTH_MAX = +0.03

STEP_WIDTH_MIN = -0.01
STEP_WIDTH_MAX = +0.01

STEP_TIME_MIN = 0.25
STEP_TIME_MAX = 0.35


# ============================================================
# STEP QP
# ============================================================

STEP_QP_ALPHA_LOCATION = 1.0
STEP_QP_ALPHA_TIMING = 5.0
STEP_QP_ALPHA_DCM = 1000.0
STEP_QP_ALPHA_VIABILITY = 1.0e6


# ============================================================
# ONLINE TIMING CAUSALITY
# ============================================================

STEP_TIMING_GAP = 0.05


# ============================================================
# SWING FOOT
# ============================================================

SWING_HEIGHT = 0.02


# ============================================================
# DIFFERENTIAL IK
# ============================================================

IK_DAMPING = 1.0e-8
IK_RCOND = 1.0e-10

SUPPORT_POSITION_KP = 25.0
SWING_POSITION_KP = 20.0
COM_POSITION_KP = 10.0
TRUNK_ORIENTATION_KP = 10.0


# ============================================================
# LEG JOINTS FOR LOGGING
# ============================================================

LEG_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


# ============================================================
# COMMAND SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class VelocityCommandSnapshot:
    x: float
    y: float


# ============================================================
# BASIC HELPERS
# ============================================================

def opposite_side(
    side: str,
) -> str:

    if side == "left":
        return "right"

    if side == "right":
        return "left"

    raise ValueError(
        f"Invalid side: {side}"
    )


def stance_leg_from_side(
    side: str,
) -> StanceLeg:

    if side == "left":
        return StanceLeg.LEFT

    if side == "right":
        return StanceLeg.RIGHT

    raise ValueError(
        f"Invalid side: {side}"
    )


def get_contact_position(
    side,
    left_contact,
    right_contact,
):

    if side == "left":
        return left_contact

    if side == "right":
        return right_contact

    raise ValueError(
        f"Invalid side: {side}"
    )


# ============================================================
# LEG JOINT LOGGING
# ============================================================

def get_leg_joint_limits(
    mj_model,
):

    joint_limits = {}

    for joint_name in LEG_JOINT_NAMES:

        joint_id = int(
            mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
        )

        if joint_id < 0:

            raise RuntimeError(
                f"Joint '{joint_name}' was not found."
            )

        if bool(
            mj_model.jnt_limited[
                joint_id
            ]
        ):

            lower = float(
                mj_model.jnt_range[
                    joint_id,
                    0,
                ]
            )

            upper = float(
                mj_model.jnt_range[
                    joint_id,
                    1,
                ]
            )

        else:

            lower = -np.inf
            upper = +np.inf

        joint_limits[
            joint_name
        ] = (
            lower,
            upper,
        )

    return joint_limits


def get_leg_joint_angles(
    *,
    mj_model,
    mj_data,
):

    joint_angles = {}

    for joint_name in LEG_JOINT_NAMES:

        joint_id = int(
            mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
        )

        if joint_id < 0:

            raise RuntimeError(
                f"Joint '{joint_name}' was not found."
            )

        qpos_address = int(
            mj_model.jnt_qposadr[
                joint_id
            ]
        )

        angle = float(
            mj_data.qpos[
                qpos_address
            ]
        )

        if not np.isfinite(
            angle
        ):

            raise RuntimeError(
                f"Joint '{joint_name}' contains NaN/Inf."
            )

        joint_angles[
            joint_name
        ] = angle

    return joint_angles


# ============================================================
# PINOCCHIO VELOCITY -> MUJOCO QVEL
# ============================================================

def pin_velocity_to_mujoco(
    *,
    robot,
    q_pin,
    v_pin,
    mj_model,
):
    """
    Convert Pinocchio generalized velocity into MuJoCo qvel.

    Mapping is performed by joint name.

    Free-flyer convention:

    Pinocchio:
        v[0:3] -> linear velocity expressed in the child/body frame
        v[3:6] -> angular velocity expressed in the child/body frame

    MuJoCo free joint:
        qvel[0:3] -> translational velocity in the world frame
        qvel[3:6] -> angular velocity in the child/body frame

    Therefore only the free-base linear velocity must be rotated:

        v_world = R_world_body @ v_body

    Scalar hinge/slide joint velocities are copied directly by
    matching joint names.
    """

    q_pin = np.asarray(
        q_pin,
        dtype=float,
    )

    v_pin = np.asarray(
        v_pin,
        dtype=float,
    )

    if q_pin.shape != (
        robot.model.nq,
    ):

        raise ValueError(
            "q_pin has wrong shape."
        )

    if v_pin.shape != (
        robot.model.nv,
    ):

        raise ValueError(
            "v_pin has wrong shape."
        )

    if (
        not np.all(
            np.isfinite(
                q_pin
            )
        )
        or
        not np.all(
            np.isfinite(
                v_pin
            )
        )
    ):

        raise ValueError(
            "q_pin/v_pin contains NaN or Inf."
        )

    q_mj_reference = (
        robot.pin_to_mujoco(
            q_pin
        )
    )

    v_mj = np.zeros(
        mj_model.nv,
        dtype=float,
    )

    for mj_joint_id in range(
        mj_model.njnt
    ):

        joint_name = mujoco.mj_id2name(
            mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            mj_joint_id,
        )

        if joint_name is None:

            continue

        pin_joint_id = int(
            robot.model.getJointId(
                joint_name
            )
        )

        if (
            pin_joint_id
            <=
            0
            or
            pin_joint_id
            >=
            robot.model.njoints
            or
            robot.model.names[
                pin_joint_id
            ]
            !=
            joint_name
        ):

            raise RuntimeError(
                f"Pinocchio joint '{joint_name}' "
                "was not found."
            )

        mj_vadr = int(
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        )

        mj_qadr = int(
            mj_model.jnt_qposadr[
                mj_joint_id
            ]
        )

        pin_vadr = int(
            robot.model.idx_vs[
                pin_joint_id
            ]
        )

        pin_nv = int(
            robot.model.nvs[
                pin_joint_id
            ]
        )

        mj_joint_type = int(
            mj_model.jnt_type[
                mj_joint_id
            ]
        )

        # ====================================================
        # FREE JOINT
        # ====================================================

        if (
            mj_joint_type
            ==
            int(
                mujoco.mjtJoint.mjJNT_FREE
            )
        ):

            if pin_nv != 6:

                raise RuntimeError(
                    f"Free joint '{joint_name}' "
                    "does not have nv=6 in Pinocchio."
                )

            v_linear_body = (
                v_pin[
                    pin_vadr:
                    pin_vadr + 3
                ]
            )

            omega_body = (
                v_pin[
                    pin_vadr + 3:
                    pin_vadr + 6
                ]
            )

            # MuJoCo free-joint quaternion:
            #
            #     [qw, qx, qy, qz]
            #
            qw = float(
                q_mj_reference[
                    mj_qadr + 3
                ]
            )

            qx = float(
                q_mj_reference[
                    mj_qadr + 4
                ]
            )

            qy = float(
                q_mj_reference[
                    mj_qadr + 5
                ]
            )

            qz = float(
                q_mj_reference[
                    mj_qadr + 6
                ]
            )

            quaternion_norm = float(
                np.sqrt(
                    qw * qw
                    +
                    qx * qx
                    +
                    qy * qy
                    +
                    qz * qz
                )
            )

            if (
                not np.isfinite(
                    quaternion_norm
                )
                or
                quaternion_norm
                <=
                1.0e-12
            ):

                raise RuntimeError(
                    "Invalid free-base quaternion."
                )

            qw /= quaternion_norm
            qx /= quaternion_norm
            qy /= quaternion_norm
            qz /= quaternion_norm

            # Rotation from child/body frame to world frame.
            R_world_body = np.array(
                [
                    [
                        1.0
                        -
                        2.0
                        *
                        (
                            qy * qy
                            +
                            qz * qz
                        ),
                        2.0
                        *
                        (
                            qx * qy
                            -
                            qw * qz
                        ),
                        2.0
                        *
                        (
                            qx * qz
                            +
                            qw * qy
                        ),
                    ],
                    [
                        2.0
                        *
                        (
                            qx * qy
                            +
                            qw * qz
                        ),
                        1.0
                        -
                        2.0
                        *
                        (
                            qx * qx
                            +
                            qz * qz
                        ),
                        2.0
                        *
                        (
                            qy * qz
                            -
                            qw * qx
                        ),
                    ],
                    [
                        2.0
                        *
                        (
                            qx * qz
                            -
                            qw * qy
                        ),
                        2.0
                        *
                        (
                            qy * qz
                            +
                            qw * qx
                        ),
                        1.0
                        -
                        2.0
                        *
                        (
                            qx * qx
                            +
                            qy * qy
                        ),
                    ],
                ],
                dtype=float,
            )

            v_mj[
                mj_vadr:
                mj_vadr + 3
            ] = (
                R_world_body
                @
                v_linear_body
            )

            # Both Pinocchio and MuJoCo use child/body-frame
            # angular velocity for the rotational part of a
            # free joint.
            v_mj[
                mj_vadr + 3:
                mj_vadr + 6
            ] = (
                omega_body
            )

        # ====================================================
        # HINGE / SLIDE JOINT
        # ====================================================

        elif (
            mj_joint_type
            in (
                int(
                    mujoco.mjtJoint.mjJNT_HINGE
                ),
                int(
                    mujoco.mjtJoint.mjJNT_SLIDE
                ),
            )
        ):

            if pin_nv != 1:

                raise RuntimeError(
                    f"Joint '{joint_name}' "
                    "does not have nv=1 in Pinocchio."
                )

            v_mj[
                mj_vadr
            ] = (
                v_pin[
                    pin_vadr
                ]
            )

        else:

            raise NotImplementedError(
                f"Unsupported MuJoCo joint type "
                f"for '{joint_name}'."
            )

    if not np.all(
        np.isfinite(
            v_mj
        )
    ):

        raise RuntimeError(
            "Converted MuJoCo qvel contains NaN or Inf."
        )

    return v_mj


# ============================================================
# MUJOCO STATE UPDATE
# ============================================================

def update_mujoco_from_pinocchio(
    *,
    robot,
    q_pin,
    q_pin_velocity_reference,
    qvel_pin,
    mj_model,
    mj_data,
):
    """
    Set the kinematically generated Pinocchio state in MuJoCo.

    Unlike the previous implementation, qvel is NOT reconstructed
    from two consecutive MuJoCo qpos samples.

    Instead:

        differential IK
            -> qdot_full (Pinocchio generalized velocity)
            -> frame-convention conversion
            -> mj_data.qvel

    q_pin_velocity_reference is the Pinocchio configuration at which
    qdot_full was computed. It is used only for converting the
    free-base linear velocity from body coordinates to world
    coordinates.
    """

    q_mj = (
        robot.pin_to_mujoco(
            q_pin
        )
    )

    qvel_mj = (
        pin_velocity_to_mujoco(
            robot=(
                robot
            ),
            q_pin=(
                q_pin_velocity_reference
            ),
            v_pin=(
                qvel_pin
            ),
            mj_model=(
                mj_model
            ),
        )
    )

    mj_data.qpos[:] = q_mj
    mj_data.qvel[:] = qvel_mj

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    mujoco.mj_subtreeVel(
        mj_model,
        mj_data,
    )

    return (
        q_mj,
        qvel_mj,
    )


def get_mujoco_centroidal_quantities(
    *,
    mj_data,
    robot_root_body_id,
):

    body_id = int(
        robot_root_body_id
    )

    p_G = (
        mj_data.subtree_com[
            body_id
        ].copy()
    )

    v_G = (
        mj_data.subtree_linvel[
            body_id
        ].copy()
    )

    L_G = (
        mj_data.subtree_angmom[
            body_id
        ].copy()
    )

    for name, value in (
        ("p_G", p_G),
        ("v_G", v_G),
        ("L_G", L_G),
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise RuntimeError(
                f"MuJoCo centroidal quantity {name} "
                "contains NaN/Inf."
            )

    return (
        p_G,
        v_G,
        L_G,
    )


def compute_robot_mass(
    mj_model,
) -> float:

    mass = float(
        np.sum(
            mj_model.body_mass
        )
    )

    if (
        not np.isfinite(
            mass
        )
        or
        mass <= 0.0
    ):

        raise RuntimeError(
            f"Invalid robot mass: {mass}"
        )

    return mass


# ============================================================
# NOMINAL STEP
# ============================================================

def compute_nominal_step(
    *,
    planner,
    stance_side,
    desired_velocity_x,
    desired_velocity_y,
):

    return planner.compute_nominal_step(
        desired_velocity_x=(
            desired_velocity_x
        ),
        desired_velocity_y=(
            desired_velocity_y
        ),
        stance_leg=(
            stance_leg_from_side(
                stance_side
            )
        ),
    )


def compute_nominal_steps(
    *,
    planner,
    velocity_command,
):

    nominal_left = (
        compute_nominal_step(
            planner=(
                planner
            ),
            stance_side="left",
            desired_velocity_x=(
                velocity_command.x
            ),
            desired_velocity_y=(
                velocity_command.y
            ),
        )
    )

    nominal_right = (
        compute_nominal_step(
            planner=(
                planner
            ),
            stance_side="right",
            desired_velocity_x=(
                velocity_command.x
            ),
            desired_velocity_y=(
                velocity_command.y
            ),
        )
    )

    return (
        nominal_left,
        nominal_right,
    )


# ============================================================
# ADAPTIVE STEP QP
# ============================================================

def solve_adaptive_step(
    *,
    planner,
    nominal_step,
    dcm,
    stance_position,
    elapsed_time,
):

    return planner.solve_adaptive_step(
        nominal_step=(
            nominal_step
        ),
        dcm_measured=(
            dcm
        ),
        stance_position=(
            stance_position[
                0:2
            ]
        ),
        elapsed_time=(
            elapsed_time
        ),
        alpha_location=(
            STEP_QP_ALPHA_LOCATION
        ),
        alpha_timing=(
            STEP_QP_ALPHA_TIMING
        ),
        alpha_dcm=(
            STEP_QP_ALPHA_DCM
        ),
        alpha_viability=(
            STEP_QP_ALPHA_VIABILITY
        ),
        timing_gap=(
            STEP_TIMING_GAP
        ),
    )


# ============================================================
# CONSISTENT INITIAL DCM
# ============================================================

def compute_nominal_initial_dcm(
    *,
    nominal_step,
    stance_position,
):

    u0 = np.asarray(
        stance_position[
            0:2
        ],
        dtype=float,
    )

    delta_u = np.array(
        [
            nominal_step.step_displacement_x,
            nominal_step.step_displacement_y,
        ],
        dtype=float,
    )

    b_nom = np.array(
        [
            nominal_step.dcm_offset_x,
            nominal_step.dcm_offset_y,
        ],
        dtype=float,
    )

    return (
        u0
        +
        (
            delta_u
            +
            b_nom
        )
        /
        nominal_step.tau
    )


def compute_consistent_initial_com_velocity(
    *,
    planner,
    com_position,
    nominal_step,
    stance_position,
):

    xi0 = (
        compute_nominal_initial_dcm(
            nominal_step=(
                nominal_step
            ),
            stance_position=(
                stance_position
            ),
        )
    )

    com_xy = np.asarray(
        com_position[
            0:2
        ],
        dtype=float,
    )

    velocity_xy = (
        planner.omega
        *
        (
            xi0
            -
            com_xy
        )
    )

    velocity = np.array(
        [
            velocity_xy[0],
            velocity_xy[1],
            0.0,
        ],
        dtype=float,
    )

    return (
        velocity,
        xi0,
    )


# ============================================================
# START NEW STEP
# ============================================================

def start_new_step(
    *,
    planner,
    robot,
    swing_trajectory,
    stance_side,
    left_contact_position,
    right_contact_position,
    nominal_left_step,
    nominal_right_step,
    current_dcm,
):

    swing_side = (
        opposite_side(
            stance_side
        )
    )

    if stance_side == "left":

        nominal_step = (
            nominal_left_step
        )

    else:

        nominal_step = (
            nominal_right_step
        )

    stance_position = (
        get_contact_position(
            stance_side,
            left_contact_position,
            right_contact_position,
        )
    )

    planner_result = (
        solve_adaptive_step(
            planner=(
                planner
            ),
            nominal_step=(
                nominal_step
            ),
            dcm=(
                current_dcm
            ),
            stance_position=(
                stance_position
            ),
            elapsed_time=0.0,
        )
    )

    if swing_side == "left":

        (
            swing_start,
            _,
        ) = (
            robot.get_left_foot_pose()
        )

        landing_z = (
            left_contact_position[
                2
            ]
        )

    else:

        (
            swing_start,
            _,
        ) = (
            robot.get_right_foot_pose()
        )

        landing_z = (
            right_contact_position[
                2
            ]
        )

    landing_position = (
        swing_start.copy()
    )

    landing_position[
        0
    ] = (
        planner_result.step_location_x
    )

    landing_position[
        1
    ] = (
        planner_result.step_location_y
    )

    landing_position[
        2
    ] = (
        landing_z
    )

    swing_trajectory.reset(
        initial_position=(
            swing_start
        ),
        initial_velocity=np.zeros(
            3,
            dtype=float,
        ),
        initial_acceleration=np.zeros(
            3,
            dtype=float,
        ),
        start_time=0.0,
    )

    return (
        swing_side,
        nominal_step,
        planner_result,
        landing_position,
        planner_result.step_time,
    )


# ============================================================
# ADVANCE LIPM WITH OPTIONAL PUSH
# ============================================================

def advance_lipm(
    *,
    lipm,
    dt,
    robot_mass,
    push_time_remaining,
):

    dt = float(
        dt
    )

    push_time_remaining = max(
        0.0,
        float(
            push_time_remaining
        ),
    )

    if (
        push_time_remaining
        <=
        TIME_TOLERANCE
    ):

        lipm.advance(
            dt
        )

        return 0.0

    push_dt = min(
        dt,
        push_time_remaining,
    )

    force_xy = np.array(
        [
            PUSH_FORCE_X,
            PUSH_FORCE_Y,
        ],
        dtype=float,
    )

    lipm.advance(
        push_dt,
        external_force_xy=(
            force_xy
        ),
        mass=(
            robot_mass
        ),
    )

    remaining_dt = (
        dt
        -
        push_dt
    )

    if (
        remaining_dt
        >
        TIME_TOLERANCE
    ):

        lipm.advance(
            remaining_dt
        )

    return max(
        0.0,
        push_time_remaining
        -
        push_dt,
    )


# ============================================================
# RUN WALK
# ============================================================

def run_walk(
    *,
    mj_model,
    mj_data,
    robot,
    planner,
    viewer,
):

    physics_dt = float(
        mj_model.opt.timestep
    )

    if not np.isclose(
        physics_dt,
        DT,
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "MuJoCo physics timestep does not match DT: "
            f"model={physics_dt:.12f} s, "
            f"DT={DT:.12f} s"
        )

    robot_root_body_id = int(
        mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base",
        )
    )

    if robot_root_body_id < 0:

        raise RuntimeError(
            "MuJoCo body 'base' was not found."
        )

    robot_mass = (
        compute_robot_mass(
            mj_model
        )
    )

    # ========================================================
    # CONTACT FORCE RECONSTRUCTOR
    # ========================================================

    if ENABLE_CONTACT_FORCE_RECONSTRUCTION:

        contact_force_reconstructor = (
            ContactForceReconstructor(
                mj_model=(
                    mj_model
                )
            )
        )

    else:

        contact_force_reconstructor = None

    # ========================================================
    # INITIAL SETTLED CONFIGURATION
    # ========================================================

    q_pin = (
        robot.mujoco_to_pin(
            mj_data.qpos.copy()
        )
    )

    robot.update(
        q_pin
    )

    (
        left_contact_position,
        _,
    ) = (
        robot.get_left_foot_pose()
    )

    (
        right_contact_position,
        _,
    ) = (
        robot.get_right_foot_pose()
    )

    left_contact_position = (
        left_contact_position.copy()
    )

    right_contact_position = (
        right_contact_position.copy()
    )

    initial_com_actual = (
        robot.get_com()
    )

    # ========================================================
    # TRUNK ORIENTATION REFERENCE
    # ========================================================

    (
        _,
        trunk_rotation_ref,
    ) = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    trunk_rotation_ref = (
        trunk_rotation_ref.copy()
    )

    # ========================================================
    # LIPM HEIGHT
    # ========================================================

    support_plane_z = (
        0.5
        *
        (
            left_contact_position[
                2
            ]
            +
            right_contact_position[
                2
            ]
        )
    )

    com_world_z = (
        support_plane_z
        +
        COM_HEIGHT
    )

    initial_com_reference = (
        initial_com_actual.copy()
    )

    initial_com_reference[
        2
    ] = (
        com_world_z
    )

    measured_settled_com_height = float(
        initial_com_actual[
            2
        ]
        -
        support_plane_z
    )

    print(
        f"Settled CoM height above support plane: "
        f"{measured_settled_com_height:.6f} m"
    )

    print(
        f"Configured LIPM CoM height: "
        f"{COM_HEIGHT:.6f} m"
    )

    # ========================================================
    # INITIAL COMMAND
    # ========================================================

    (
        command_x,
        command_y,
    ) = (
        automatic_velocity_profile(
            0.0
        )
    )

    command = (
        VelocityCommandSnapshot(
            x=(
                command_x
            ),
            y=(
                command_y
            ),
        )
    )

    (
        nominal_left_step,
        nominal_right_step,
    ) = (
        compute_nominal_steps(
            planner=(
                planner
            ),
            velocity_command=(
                command
            ),
        )
    )

    # ========================================================
    # FIRST STANCE
    # ========================================================

    stance_side = (
        FIRST_STANCE_SIDE
    )

    stance_position = (
        get_contact_position(
            stance_side,
            left_contact_position,
            right_contact_position,
        )
    )

    if stance_side == "left":

        first_nominal_step = (
            nominal_left_step
        )

    else:

        first_nominal_step = (
            nominal_right_step
        )

    # ========================================================
    # CONSISTENT INITIAL DCM / COM VELOCITY
    # ========================================================

    (
        initial_com_velocity,
        initial_dcm,
    ) = (
        compute_consistent_initial_com_velocity(
            planner=(
                planner
            ),
            com_position=(
                initial_com_reference
            ),
            nominal_step=(
                first_nominal_step
            ),
            stance_position=(
                stance_position
            ),
        )
    )

    # ========================================================
    # POINT-FOOT LIPM
    # ========================================================

    lipm = (
        PointFootLIPM(
            gravity=(
                GRAVITY
            ),
            com_height=(
                COM_HEIGHT
            ),
            initial_position=(
                initial_com_reference
            ),
            initial_velocity=(
                initial_com_velocity
            ),
            support_position=(
                stance_position
            ),
        )
    )

    lipm_sample = (
        lipm.sample()
    )

    initial_dcm_error = float(
        np.linalg.norm(
            lipm_sample.dcm
            -
            initial_dcm
        )
    )

    if (
        initial_dcm_error
        >
        1.0e-10
    ):

        raise RuntimeError(
            "Initial DCM mismatch: "
            f"{initial_dcm_error:.3e}"
        )

    # ========================================================
    # SWING GENERATOR
    # ========================================================

    swing_trajectory = (
        OnlineQuinticSwingTrajectory()
    )

    # ========================================================
    # FIRST STEP
    # ========================================================

    (
        swing_side,
        current_nominal_step,
        planner_result,
        landing_position,
        current_step_time,
    ) = (
        start_new_step(
            planner=(
                planner
            ),
            robot=(
                robot
            ),
            swing_trajectory=(
                swing_trajectory
            ),
            stance_side=(
                stance_side
            ),
            left_contact_position=(
                left_contact_position
            ),
            right_contact_position=(
                right_contact_position
            ),
            nominal_left_step=(
                nominal_left_step
            ),
            nominal_right_step=(
                nominal_right_step
            ),
            current_dcm=(
                lipm_sample.dcm
            ),
        )
    )

    # ========================================================
    # RUNTIME STATE
    # ========================================================

    phase_time = 0.0
    kinematic_time = 0.0

    step_index = 1

    planner_frozen = False

    push_time_remaining = 0.0
    automatic_push_applied = False

    previous_com_actual = (
        initial_com_actual.copy()
    )

    com_velocity_actual = np.zeros(
        3,
        dtype=float,
    )

    # ========================================================
    # LEG JOINT LIMITS / LOG
    # ========================================================

    leg_joint_limits = (
        get_leg_joint_limits(
            mj_model
        )
    )

    simulation_log = (
        SimulationLog(
            leg_joint_angles={
                joint_name: []
                for joint_name
                in LEG_JOINT_NAMES
            },
            leg_joint_limits=(
                leg_joint_limits
            ),
        )
    )

    # ========================================================
    # JOINT ANGLE DATA LOGGER
    # ========================================================

    joint_data_logger = (
        JointAngleDataLogger(
            joint_names=(
                LEG_JOINT_NAMES
            ),
            output_file=(
                CURRENT_DIR
                /
                "data"
                /
                "joint_angles.csv"
            ),
        )
    )

    next_print_time = 0.0
    next_viewer_sync_time = 0.0

    wall_start = (
        time.perf_counter()
    )

    print()
    print(
        "Point-foot LIPM Step Timing Adaptation"
    )

    print(
        f"Automatic experiment duration: "
        f"{SIMULATION_DURATION:.1f} s"
    )

    print(
        f"Physics/data sampling: dt={physics_dt:.7f} s"
        f" | fs={1.0 / physics_dt:.1f} Hz"
    )

    print(
        "Velocity command:"
    )

    print(
        f"  vx={command.x:+.3f} m/s, "
        f"vy={command.y:+.3f} m/s"
    )

    print(
        f"COM height: {COM_HEIGHT:.3f} m"
        f" | swing height: {SWING_HEIGHT:.3f} m"
    )

    print(
        "Automatic push: "
        f"{'ENABLED' if ENABLE_AUTOMATIC_PUSH else 'DISABLED'}"
    )

    print(
        "Floating-base contact-force reconstruction: "
        f"{'ENABLED' if ENABLE_CONTACT_FORCE_RECONSTRUCTION else 'DISABLED'}"
    )

    print()

    print(
        "Leg joint limits:"
    )

    for joint_name in LEG_JOINT_NAMES:

        lower, upper = (
            leg_joint_limits[
                joint_name
            ]
        )

        print(
            f"  {joint_name:>18s}: "
            f"[{np.rad2deg(lower):+8.3f}, "
            f"{np.rad2deg(upper):+8.3f}] deg"
        )

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        if (
            viewer is not None
            and
            not viewer.is_running()
        ):

            break

        if (
            kinematic_time
            >=
            SIMULATION_DURATION
            -
            TIME_TOLERANCE
        ):

            break

        # ====================================================
        # AUTOMATIC VELOCITY PROFILE
        # ====================================================

        (
            new_command_x,
            new_command_y,
        ) = (
            automatic_velocity_profile(
                kinematic_time
            )
        )

        command_changed = (
            not np.isclose(
                command.x,
                new_command_x,
            )
            or
            not np.isclose(
                command.y,
                new_command_y,
            )
        )

        if command_changed:

            command = (
                VelocityCommandSnapshot(
                    x=(
                        new_command_x
                    ),
                    y=(
                        new_command_y
                    ),
                )
            )

            (
                nominal_left_step,
                nominal_right_step,
            ) = (
                compute_nominal_steps(
                    planner=(
                        planner
                    ),
                    velocity_command=(
                        command
                    ),
                )
            )

            if stance_side == "left":

                current_nominal_step = (
                    nominal_left_step
                )

            else:

                current_nominal_step = (
                    nominal_right_step
                )

            print(
                f"[COMMAND] t={kinematic_time:.3f} s"
                f" -> vx={command.x:+.3f} m/s"
                f", vy={command.y:+.3f} m/s"
            )

        # ====================================================
        # AUTOMATIC PUSH
        # ====================================================

        if (
            ENABLE_AUTOMATIC_PUSH
            and
            not automatic_push_applied
            and
            kinematic_time
            >=
            PUSH_TIME
            -
            TIME_TOLERANCE
        ):

            push_time_remaining = (
                PUSH_DURATION
            )

            automatic_push_applied = True

            print(
                f"[PUSH] t={kinematic_time:.3f} s"
                f" -> F=({PUSH_FORCE_X:+.2f},"
                f" {PUSH_FORCE_Y:+.2f}) N"
                f" for {PUSH_DURATION:.3f} s"
            )

        # ====================================================
        # CURRENT LIPM
        # ====================================================

        lipm_sample = (
            lipm.sample()
        )

        # ====================================================
        # TOUCHDOWN / SUPPORT SWITCH
        # ====================================================

        if (
            phase_time
            >
            current_step_time
            +
            TIME_TOLERANCE
        ):

            robot.update(
                q_pin
            )

            if swing_side == "left":

                left_contact_position = (
                    landing_position.copy()
                )

            else:

                right_contact_position = (
                    landing_position.copy()
                )

            stance_side = (
                swing_side
            )

            lipm.set_support_position(
                landing_position
            )

            lipm_sample = (
                lipm.sample()
            )

            step_index += 1
            phase_time = 0.0
            planner_frozen = False

            (
                swing_side,
                current_nominal_step,
                planner_result,
                landing_position,
                current_step_time,
            ) = (
                start_new_step(
                    planner=(
                        planner
                    ),
                    robot=(
                        robot
                    ),
                    swing_trajectory=(
                        swing_trajectory
                    ),
                    stance_side=(
                        stance_side
                    ),
                    left_contact_position=(
                        left_contact_position
                    ),
                    right_contact_position=(
                        right_contact_position
                    ),
                    nominal_left_step=(
                        nominal_left_step
                    ),
                    nominal_right_step=(
                        nominal_right_step
                    ),
                    current_dcm=(
                        lipm_sample.dcm
                    ),
                )
            )

        # ====================================================
        # ONLINE STEP LOCATION / TIMING ADAPTATION
        # ====================================================

        if (
            not planner_frozen
            and
            phase_time
            >
            TIME_TOLERANCE
        ):

            if (
                phase_time
                >=
                current_step_time
                -
                STEP_TIMING_GAP
                -
                TIME_TOLERANCE
            ):

                planner_frozen = True

            else:

                stance_position = (
                    get_contact_position(
                        stance_side,
                        left_contact_position,
                        right_contact_position,
                    )
                )

                try:

                    new_result = (
                        solve_adaptive_step(
                            planner=(
                                planner
                            ),
                            nominal_step=(
                                current_nominal_step
                            ),
                            dcm=(
                                lipm_sample.dcm
                            ),
                            stance_position=(
                                stance_position
                            ),
                            elapsed_time=(
                                phase_time
                            ),
                        )
                    )

                    planner_result = (
                        new_result
                    )

                    landing_position[
                        0
                    ] = (
                        planner_result.step_location_x
                    )

                    landing_position[
                        1
                    ] = (
                        planner_result.step_location_y
                    )

                    current_step_time = (
                        planner_result.step_time
                    )

                    if (
                        phase_time
                        >=
                        current_step_time
                        -
                        STEP_TIMING_GAP
                        -
                        TIME_TOLERANCE
                    ):

                        planner_frozen = True

                except RuntimeError as error:

                    if (
                        "timing adaptation window is closed"
                        in
                        str(
                            error
                        ).lower()
                    ):

                        planner_frozen = True

                    else:

                        raise

        # ====================================================
        # LIPM COM REFERENCE
        # ====================================================

        lipm_sample = (
            lipm.sample()
        )

        com_position_ref = (
            lipm_sample.position.copy()
        )

        com_velocity_ref = (
            lipm_sample.velocity.copy()
        )

        # ====================================================
        # ONLINE SWING TRAJECTORY
        # ====================================================

        swing_sample = (
            swing_trajectory.update(
                current_time=(
                    min(
                        phase_time,
                        current_step_time,
                    )
                ),
                landing_time=(
                    current_step_time
                ),
                target_position=(
                    landing_position
                ),
                swing_height=(
                    SWING_HEIGHT
                ),
            )
        )

        if swing_side == "left":

            p_left_ref = (
                swing_sample.position.copy()
            )

            p_right_ref = (
                right_contact_position.copy()
            )

        else:

            p_left_ref = (
                left_contact_position.copy()
            )

            p_right_ref = (
                swing_sample.position.copy()
            )

        support_position_ref = (
            get_contact_position(
                stance_side,
                p_left_ref,
                p_right_ref,
            )
        )

        swing_position_ref = (
            get_contact_position(
                swing_side,
                p_left_ref,
                p_right_ref,
            )
        )

        # ====================================================
        # DIFFERENTIAL IK
        # ====================================================

        (
            qdot_full,
            _,
            _,
        ) = (
            solve_single_support_ik(
                robot=(
                    robot
                ),
                q_pin=(
                    q_pin
                ),
                support_side=(
                    stance_side
                ),
                support_position_ref=(
                    support_position_ref
                ),
                swing_position_ref=(
                    swing_position_ref
                ),
                swing_linear_velocity_ref=(
                    swing_sample.velocity
                ),
                com_position_ref=(
                    com_position_ref
                ),
                com_velocity_ref=(
                    com_velocity_ref
                ),
                trunk_rotation_ref=(
                    trunk_rotation_ref
                ),
                support_position_gain=(
                    SUPPORT_POSITION_KP
                ),
                swing_position_gain=(
                    SWING_POSITION_KP
                ),
                com_position_gain=(
                    COM_POSITION_KP
                ),
                trunk_orientation_gain=(
                    TRUNK_ORIENTATION_KP
                ),
                damping=(
                    IK_DAMPING
                ),
                rcond=(
                    IK_RCOND
                ),
            )
        )

        if not np.all(
            np.isfinite(
                qdot_full
            )
        ):

            raise RuntimeError(
                "Differential IK returned NaN/Inf."
            )

        # ====================================================
        # PINOCCHIO INTEGRATION
        # ====================================================
        #
        # qdot_full was computed at the CURRENT q_pin.
        # Keep that configuration for the Pinocchio -> MuJoCo
        # free-base velocity-frame conversion.
        # ====================================================

        q_pin_velocity_reference = (
            q_pin.copy()
        )

        q_pin = (
            robot.integrate(
                q_pin=(
                    q_pin
                ),
                v_pin=(
                    qdot_full
                ),
                dt=(
                    DT
                ),
            )
        )

        robot.update(
            q_pin
        )

        # ====================================================
        # MUJOCO SET STATE
        # ====================================================

        (
            current_qpos_mj,
            current_qvel_mj,
        ) = (
            update_mujoco_from_pinocchio(
                robot=(
                    robot
                ),
                q_pin=(
                    q_pin
                ),
                q_pin_velocity_reference=(
                    q_pin_velocity_reference
                ),
                qvel_pin=(
                    qdot_full
                ),
                mj_model=(
                    mj_model
                ),
                mj_data=(
                    mj_data
                ),
            )
        )

        # Keep the returned variables available for debugging.
        # Contact-force reconstruction records mj_data.qpos/qvel
        # immediately below.
        _ = (
            current_qpos_mj,
            current_qvel_mj,
        )

        # ====================================================
        # RECORD q, qdot FOR FLOATING-BASE INVERSE DYNAMICS
        # ====================================================

        if ENABLE_CONTACT_FORCE_RECONSTRUCTION:

            contact_force_reconstructor.record_sample(
                time=(
                    kinematic_time
                ),
                mj_data=(
                    mj_data
                ),
            )

        # ====================================================
        # LEG JOINT ANGLES
        # ====================================================

        leg_joint_angles = (
            get_leg_joint_angles(
                mj_model=(
                    mj_model
                ),
                mj_data=(
                    mj_data
                ),
            )
        )

        joint_data_logger.append(
            time=(
                kinematic_time
            ),
            joint_angles=(
                leg_joint_angles
            ),
        )

        # ====================================================
        # WHOLE-BODY CENTROIDAL QUANTITIES
        # ====================================================

        (
            whole_body_com_position,
            whole_body_com_velocity,
            whole_body_angular_momentum,
        ) = (
            get_mujoco_centroidal_quantities(
                mj_data=(
                    mj_data
                ),
                robot_root_body_id=(
                    robot_root_body_id
                ),
            )
        )

        # ====================================================
        # ACTUAL ROBOT COM VELOCITY
        # ====================================================

        p_com_actual = (
            robot.get_com()
        )

        com_velocity_actual = (
            p_com_actual
            -
            previous_com_actual
        ) / DT

        previous_com_actual = (
            p_com_actual.copy()
        )

        push_active = (
            ENABLE_AUTOMATIC_PUSH
            and
            push_time_remaining
            >
            TIME_TOLERANCE
        )

        # ====================================================
        # DATA LOG -- EVERY PHYSICS STEP
        # ====================================================

        (
            left_foot_position_log,
            _,
        ) = (
            robot.get_left_foot_pose()
        )

        (
            right_foot_position_log,
            _,
        ) = (
            robot.get_right_foot_pose()
        )

        simulation_log.append(
            time=(
                kinematic_time
            ),
            command_vx=(
                command.x
            ),
            command_vy=(
                command.y
            ),
            left_foot_position=(
                left_foot_position_log
            ),
            right_foot_position=(
                right_foot_position_log
            ),
            com_position=(
                lipm_sample.position
            ),
            dcm=(
                lipm_sample.dcm
            ),
            landing_position=(
                landing_position
            ),
            step_time=(
                current_step_time
            ),
            push_active=(
                push_active
            ),
            whole_body_com_position=(
                whole_body_com_position
            ),
            whole_body_com_velocity=(
                whole_body_com_velocity
            ),
            angular_momentum=(
                whole_body_angular_momentum
            ),
            leg_joint_angles=(
                leg_joint_angles
            ),
        )

        # ====================================================
        # TERMINAL STATUS
        # ====================================================

        if (
            kinematic_time
            >=
            next_print_time
            -
            TIME_TOLERANCE
        ):

            print(
                f"t={kinematic_time:7.3f}"
                f" | step={step_index:03d}"
                f" | vCmd="
                f"({command.x:+.3f},"
                f"{command.y:+.3f}) m/s"
                f" | uT="
                f"({landing_position[0]:+.4f},"
                f"{landing_position[1]:+.4f}) m"
                f" | T={current_step_time:.4f} s"
                f" | vCoM_actual="
                f"({com_velocity_actual[0]:+.4f},"
                f"{com_velocity_actual[1]:+.4f}) m/s"
                f" | push="
                f"{'ON' if push_active else 'OFF'}"
            )

            next_print_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # VIEWER
        # ====================================================

        if (
            viewer is not None
            and
            kinematic_time
            >=
            next_viewer_sync_time
            -
            TIME_TOLERANCE
        ):

            viewer.sync()

            next_viewer_sync_time += (
                VIEWER_SYNC_PERIOD
            )

        # ====================================================
        # EXACT LIPM PROPAGATION
        # ====================================================

        push_time_remaining = (
            advance_lipm(
                lipm=(
                    lipm
                ),
                dt=(
                    DT
                ),
                robot_mass=(
                    robot_mass
                ),
                push_time_remaining=(
                    push_time_remaining
                ),
            )
        )

        # ====================================================
        # LOGICAL TIME
        # ====================================================

        phase_time += (
            DT
        )

        kinematic_time += (
            DT
        )

        mj_data.time = (
            kinematic_time
        )

        # ====================================================
        # REAL-TIME PLAYBACK
        # ====================================================

        if REALTIME_PLAYBACK:

            target_wall_time = (
                wall_start
                +
                kinematic_time
            )

            remaining = (
                target_wall_time
                -
                time.perf_counter()
            )

            if remaining > 0.0:

                time.sleep(
                    remaining
                )

    print()

    print(
        f"Walking stopped"
        f" | t={kinematic_time:.3f} s"
        f" | steps={step_index}"
        f" | auto_push_applied="
        f"{automatic_push_applied}"
    )

    # ========================================================
    # SAVE JOINT ANGLE DATA
    # ========================================================

    joint_angle_file = (
        joint_data_logger.save_csv()
    )

    print(
        f"Joint-angle data saved to: "
        f"{joint_angle_file}"
    )

    print(
        f"Joint-angle samples     : "
        f"{joint_data_logger.number_samples()}"
    )

    print()

    # ========================================================
    # OFFLINE FLOATING-BASE CONTACT FORCE RECONSTRUCTION
    # ========================================================

    if ENABLE_CONTACT_FORCE_RECONSTRUCTION:

        print()

        print(
            "Reconstructing contact forces from "
            "floating-base dynamics..."
        )

        contact_force_results = (
            contact_force_reconstructor.solve_all()
        )

        contact_force_reconstructor.print_summary(
            contact_force_results
        )

    else:

        print()

        print(
            "Contact-force reconstruction skipped "
            "(ENABLE_CONTACT_FORCE_RECONSTRUCTION=False)."
        )

        contact_force_results = None

    return (
        simulation_log,
        contact_force_results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # FILE CHECK
    # ========================================================

    if not SCENE_XML.exists():

        raise FileNotFoundError(
            f"Scene file not found: "
            f"{SCENE_XML}"
        )

    if not ROBOT_XML.exists():

        raise FileNotFoundError(
            f"Robot XML not found: "
            f"{ROBOT_XML}"
        )

    # ========================================================
    # MUJOCO
    # ========================================================

    mj_model = (
        mujoco.MjModel.from_xml_path(
            str(
                SCENE_XML
            )
        )
    )

    mj_data = (
        mujoco.MjData(
            mj_model
        )
    )

    # ========================================================
    # INITIAL PHYSICAL SETTLING
    # ========================================================

    settle_robot(
        mj_model,
        mj_data,
    )

    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    # ========================================================
    # PINOCCHIO
    # ========================================================

    robot = (
        PinocchioModel(
            mjcf_path=(
                ROBOT_XML
            ),
            mujoco_model=(
                mj_model
            ),
        )
    )

    # ========================================================
    # MEASURE SETTLED FOOT SEPARATION
    # ========================================================

    q_pin_initial = (
        robot.mujoco_to_pin(
            mj_data.qpos.copy()
        )
    )

    robot.update(
        q_pin_initial
    )

    (
        left_position_initial,
        _,
    ) = (
        robot.get_left_foot_pose()
    )

    (
        right_position_initial,
        _,
    ) = (
        robot.get_right_foot_pose()
    )

    measured_step_width = float(
        abs(
            left_position_initial[
                1
            ]
            -
            right_position_initial[
                1
            ]
        )
    )

    if (
        not np.isfinite(
            measured_step_width
        )
        or
        measured_step_width
        <=
        1.0e-6
    ):

        raise RuntimeError(
            "Invalid measured foot separation: "
            f"{measured_step_width}"
        )

    print(
        f"Measured default step width: "
        f"{measured_step_width:.6f} m"
    )

    # ========================================================
    # ADAPTIVE STEP PLANNER
    # ========================================================

    planner = (
        AdaptiveStepPlanner(
            StepPlannerParameters(
                gravity=(
                    GRAVITY
                ),
                com_height=(
                    COM_HEIGHT
                ),
                default_step_width=(
                    measured_step_width
                ),
                step_length_min=(
                    STEP_LENGTH_MIN
                ),
                step_length_max=(
                    STEP_LENGTH_MAX
                ),
                step_width_min=(
                    STEP_WIDTH_MIN
                ),
                step_width_max=(
                    STEP_WIDTH_MAX
                ),
                step_time_min=(
                    STEP_TIME_MIN
                ),
                step_time_max=(
                    STEP_TIME_MAX
                ),
            )
        )
    )

    # ========================================================
    # RUN SIMULATION
    # ========================================================

    if SHOW_VIEWER:

        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            show_right_ui=True,
        ) as viewer:

            (
                simulation_log,
                contact_force_results,
            ) = (
                run_walk(
                    mj_model=(
                        mj_model
                    ),
                    mj_data=(
                        mj_data
                    ),
                    robot=(
                        robot
                    ),
                    planner=(
                        planner
                    ),
                    viewer=(
                        viewer
                    ),
                )
            )

    else:

        (
            simulation_log,
            contact_force_results,
        ) = (
            run_walk(
                mj_model=(
                    mj_model
                ),
                mj_data=(
                    mj_data
                ),
                robot=(
                    robot
                ),
                planner=(
                    planner
                ),
                viewer=None,
            )
        )

    # ========================================================
    # CONTACT FORCE FIGURES
    # ========================================================
    #
    # Create these figures first with show=False.
    # plot_simulation_results() calls plt.show() at the end,
    # therefore all motion, joint-angle, and contact-force
    # figures are displayed together.
    # ========================================================

    if (
        contact_force_results
        is not None
    ):

        plot_contact_force_results(
            contact_force_results,
            mj_model=(
                mj_model
            ),
            show=False,
        )

    # ========================================================
    # PLOT RESULTS AFTER SIMULATION
    # ========================================================

    plot_simulation_results(
        simulation_log
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
