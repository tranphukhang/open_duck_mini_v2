# step_timing_adaptation/run.py

from __future__ import annotations

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

ROOT_DIR = (
    CURRENT_DIR.parent
)

SCENE_XML = (
    ROOT_DIR
    / "xmls"
    / "scene_flat_terrain_torque.xml"
)


# ============================================================
# IMPORTS
# ============================================================

if __package__:

    from .dynamics_model import (
        WholeBodyDynamicsModel,
    )

    from .adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from .swing_trajectory import (
        OnlineSwingFootTrajectory,
        VerticalSwingQPParameters,
    )

    from .whole_body_qp import (
        WholeBodyHQPConfig,
        WholeBodyHierarchicalInverseDynamics,
        DoubleSupportPreparationHierarchicalInverseDynamics,
    )


else:

    from dynamics_model import (
        WholeBodyDynamicsModel,
    )

    from adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from swing_trajectory import (
        OnlineSwingFootTrajectory,
        VerticalSwingQPParameters,
    )

    from whole_body_qp import (
        WholeBodyHQPConfig,
        WholeBodyHierarchicalInverseDynamics,
        DoubleSupportPreparationHierarchicalInverseDynamics,
    )



# ============================================================
# MODEL NAMES
# ============================================================

ROBOT_ROOT_BODY = "base"

LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"

LEFT_FOOT_GEOM = "left_foot_bottom_tpu"
RIGHT_FOOT_GEOM = "right_foot_bottom_tpu"

FLOOR_GEOM = "floor"


# ============================================================
# SIMULATION
# ============================================================

CONTROL_FREQUENCY = 1000.0

SHOW_VIEWER = True

REALTIME_FACTOR = 1.0

VIEWER_REFRESH_FREQUENCY = 60.0

STATUS_PRINT_PERIOD = 0.05


# ============================================================
# CONTACT
# ============================================================

FRICTION_COEFFICIENT = 0.6

# Shrink support rectangle inward.
SUPPORT_POLYGON_MARGIN = 0.002
# 2 mm


# ============================================================
# COM HEIGHT
# ============================================================

COM_HEIGHT_REFERENCE = 0.2044

COM_HEIGHT_KP = 80.0
COM_HEIGHT_KD = 16.0


# ============================================================
# ADAPTIVE STEP PLANNER
# ============================================================

# ------------------------------------------------------------
# Stage-2 QP weights
#
# Khadiv et al. simulation values:
# alpha_1 = 1
# alpha_2 = 5
# alpha_3 = 1000
# ------------------------------------------------------------

STEP_QP_ALPHA_LOCATION = 1.0
STEP_QP_ALPHA_TIMING = 5.0
STEP_QP_ALPHA_DCM = 1000.0
STEP_QP_ALPHA_VIABILITY = 1.0e6
STEP_TIMING_GAP = 0.02


# ------------------------------------------------------------
# Desired walking velocity
# ------------------------------------------------------------

DESIRED_VELOCITY_X = 0.05
DESIRED_VELOCITY_Y = 0.0


# ------------------------------------------------------------
# LIPM
# ------------------------------------------------------------

GRAVITY = 9.81


# ------------------------------------------------------------
# Nominal lateral foot spacing l_p
# ------------------------------------------------------------

DEFAULT_STEP_WIDTH = 0.16


# ------------------------------------------------------------
# Step location bounds
#
# Temporary planner bounds.
# Replace later with identified physical limits
# of the Open Duck Mini.
# ------------------------------------------------------------

STEP_LENGTH_MIN = -0.10
STEP_LENGTH_MAX = +0.10

STEP_WIDTH_MIN = -0.03
STEP_WIDTH_MAX = +0.03


# ------------------------------------------------------------
# Step timing bounds
# ------------------------------------------------------------

STEP_TIME_MIN = 0.20
STEP_TIME_MAX = 0.30


# ============================================================
# SWING FOOT TRAJECTORY
# ============================================================

# Desired swing-foot clearance.
SWING_HEIGHT_DESIRED = 0.03

# Hard upper bound for swing-foot height.
SWING_HEIGHT_MAX = 0.04

# Initial discretization of:
#
#     0 <= z(t) <= z_max
#
# Continuous extrema are checked and refined automatically.
SWING_VERTICAL_CONSTRAINT_SAMPLES = 41

# Small numerical regularization only.
SWING_VERTICAL_COEFFICIENT_REGULARIZATION = 1.0e-8

SWING_VERTICAL_BOUND_TOLERANCE = 1.0e-9

SWING_VERTICAL_MAX_REFINEMENTS = 8


# ============================================================
# ONLINE EXECUTION
# ============================================================

PLANNER_UPDATE_FREQUENCY = 100.0

SWING_TRACKING_KP = 250.0
SWING_TRACKING_KD = 30.0

# Contact is accepted only near the planned touchdown.
TOUCHDOWN_CONTACT_WINDOW = 0.02

# Keep final landing target for a short settling interval.
TOUCHDOWN_SETTLE_TIME = 0.0


# ============================================================
# DOUBLE-SUPPORT PREPARATION
# ============================================================

PREPARE_DURATION = 2.00
PREPARE_HOLD_TIME = 0.50

# CoM target is placed slightly toward the inside
# of the future LEFT stance foot.
PREPARE_LEFT_INNER_OFFSET = 0.010

PREPARE_COM_KP = 40.0
PREPARE_COM_KD = 12.0

# Desired final load distribution before lift-off.
PREPARE_LEFT_LOAD_FRACTION = 0.90

# Transition conditions.
PREPARE_RIGHT_LOAD_MAX = 0.15
PREPARE_DCM_MARGIN = 0.003
PREPARE_VELOCITY_Y_MAX = 0.04
PREPARE_MAX_TILT_DEG = 8.0

PREPARE_STATUS_PRINT_PERIOD = 0.10


# ============================================================
# POSTURE — LOWER PRIORITY
# ============================================================

POSTURE_KP = 50.0
POSTURE_KD = 14.0


# ============================================================
# NUMERICAL
# ============================================================

COM_JDOT_EPSILON = 1.0e-6


# ============================================================
# EMERGENCY
# ============================================================

EMERGENCY_BASE_TILT_DEG = 45.0

EMERGENCY_COM_DROP = 0.050


# ============================================================
# PRINT
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
)


def separator():

    print(
        "=" * 100
    )


# ============================================================
# OBJECT LOOKUP
# ============================================================

def require_object_id(
    model,
    object_type,
    object_name,
):

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        object_name,
    )

    if object_id < 0:

        raise RuntimeError(
            f"MuJoCo object not found: "
            f"'{object_name}'"
        )

    return int(
        object_id
    )


# ============================================================
# ACTUATED QPOS
# ============================================================

def build_actuated_qpos_indices(
    model,
):

    indices = []

    for actuator_id in range(
        model.nactuator
    ):

        joint_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if joint_id < 0:

            raise RuntimeError(
                f"Actuator {actuator_id} "
                "does not target a joint."
            )

        joint_type = int(
            model.jnt_type[
                joint_id
            ]
        )

        if (
            joint_type
            !=
            int(
                mujoco.mjtJoint.mjJNT_HINGE
            )
        ):

            raise RuntimeError(
                "Controller currently assumes "
                "direct hinge-joint actuators."
            )

        qpos_index = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        indices.append(
            qpos_index
        )

    return np.asarray(
        indices,
        dtype=int,
    )


# ============================================================
# BASE TILT
# ============================================================

def get_base_tilt_deg(
    data,
    base_body_id,
):

    rotation = (
        data.xmat[
            base_body_id
        ]
        .reshape(
            3,
            3,
        )
    )

    base_z_world = (
        rotation[
            :,
            2
        ]
    )

    cos_tilt = float(
        np.clip(
            base_z_world[
                2
            ],
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cos_tilt
            )
        )
    )


# ============================================================
# CONTACT CHECK
# ============================================================

def has_geom_contact(
    data,
    geom_a,
    geom_b,
):

    for contact_id in range(
        data.ncon
    ):

        contact = (
            data.contact[
                contact_id
            ]
        )

        geom1 = int(
            contact.geom1
        )

        geom2 = int(
            contact.geom2
        )

        if (
            (
                geom1 == geom_a
                and
                geom2 == geom_b
            )
            or
            (
                geom1 == geom_b
                and
                geom2 == geom_a
            )
        ):

            return True

    return False


# ============================================================
# SUPPORT RECTANGLE FROM COLLISION GEOM
# ============================================================

def get_support_rectangle_from_geom_aabb(
    model,
    data,
    geom_name,
    site_name,
    safety_margin,
):
    """
    Build a conservative rectangular support region from the
    collision geom's MuJoCo AABB.

    Result:

        [xmin, xmax, ymin, ymax]

    where x/y are WORLD-horizontal offsets measured from the
    corresponding foot-site origin.

    This is evaluated at the HOME configuration.

    Since the stance-foot task later constrains the foot pose
    in 6D, this rectangle remains the nominal contact support
    region during the standing test.
    """

    geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    site_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    aabb = np.asarray(
        model.geom_aabb[
            geom_id
        ],
        dtype=float,
    ).reshape(
        2,
        3,
    )

    center_local = (
        aabb[
            0,
            :
        ]
    )

    half_size_local = (
        aabb[
            1,
            :
        ]
    )

    geom_position_world = (
        data.geom_xpos[
            geom_id
        ]
        .copy()
    )

    geom_rotation_world = (
        data.geom_xmat[
            geom_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    site_position_world = (
        data.site_xpos[
            site_id
        ]
        .copy()
    )

    corners_world = []

    for sx in (
        -1.0,
        +1.0,
    ):

        for sy in (
            -1.0,
            +1.0,
        ):

            for sz in (
                -1.0,
                +1.0,
            ):

                corner_local = (
                    center_local
                    +
                    np.array(
                        [
                            sx
                            *
                            half_size_local[
                                0
                            ],

                            sy
                            *
                            half_size_local[
                                1
                            ],

                            sz
                            *
                            half_size_local[
                                2
                            ],
                        ],
                        dtype=float,
                    )
                )

                corner_world = (
                    geom_position_world
                    +
                    geom_rotation_world
                    @
                    corner_local
                )

                corners_world.append(
                    corner_world
                )

    corners_world = np.asarray(
        corners_world,
        dtype=float,
    )

    # Horizontal offsets from wrench-reference site.

    relative_xy = (
        corners_world[
            :,
            0:2
        ]
        -
        site_position_world[
            0:2
        ]
    )

    xmin = float(
        np.min(
            relative_xy[
                :,
                0
            ]
        )
    )

    xmax = float(
        np.max(
            relative_xy[
                :,
                0
            ]
        )
    )

    ymin = float(
        np.min(
            relative_xy[
                :,
                1
            ]
        )
    )

    ymax = float(
        np.max(
            relative_xy[
                :,
                1
            ]
        )
    )

    margin = float(
        safety_margin
    )

    xmin += margin
    xmax -= margin

    ymin += margin
    ymax -= margin

    if xmin >= xmax:

        raise RuntimeError(
            f"Invalid support x-range "
            f"for {geom_name}."
        )

    if ymin >= ymax:

        raise RuntimeError(
            f"Invalid support y-range "
            f"for {geom_name}."
        )

    # Basic sanity check.
    #
    # Open Duck feet are nowhere near 0.5 m long/wide.

    if (
        xmax
        -
        xmin
        >
        0.5
    ):

        raise RuntimeError(
            f"Suspicious support x-size "
            f"for {geom_name}."
        )

    if (
        ymax
        -
        ymin
        >
        0.5
    ):

        raise RuntimeError(
            f"Suspicious support y-size "
            f"for {geom_name}."
        )

    return np.array(
        [
            xmin,
            xmax,
            ymin,
            ymax,
        ],
        dtype=float,
    )


# ============================================================
# COP FROM OPTIMIZED WRENCH
# ============================================================

def compute_cop_from_wrench(
    wrench,
    contact_height,
):

    wrench = np.asarray(
        wrench,
        dtype=float,
    )

    fx = float(
        wrench[
            0
        ]
    )

    fy = float(
        wrench[
            1
        ]
    )

    fz = float(
        wrench[
            2
        ]
    )

    mx = float(
        wrench[
            3
        ]
    )

    my = float(
        wrench[
            4
        ]
    )

    if (
        fz
        <=
        1.0e-6
    ):

        return np.array(
            [
                np.nan,
                np.nan,
            ],
            dtype=float,
        )

    h = float(
        contact_height
    )

    x_cop = (
        -(
            my
            +
            h
            *
            fx
        )
        /
        fz
    )

    y_cop = (
        (
            mx
            -
            h
            *
            fy
        )
        /
        fz
    )

    return np.array(
        [
            x_cop,
            y_cop,
        ],
        dtype=float,
    )


# ============================================================
# COM Jdot*v
# ============================================================

def compute_com_jdot_v(
    model,
    data,
    root_body_id,
    scratch_plus,
    scratch_minus,
    epsilon,
):

    qvel = (
        data.qvel.copy()
    )

    if (
        np.linalg.norm(
            qvel
        )
        <
        1.0e-12
    ):

        return np.zeros(
            3,
            dtype=float,
        )

    # ========================================================
    # PLUS
    # ========================================================

    q_plus = (
        data.qpos.copy()
    )

    mujoco.mj_integratePos(
        model,
        q_plus,
        qvel,
        +epsilon,
    )

    scratch_plus.qpos[:] = (
        q_plus
    )

    scratch_plus.qvel[:] = (
        qvel
    )

    scratch_plus.ctrl[:] = (
        data.ctrl
    )

    mujoco.mj_forward(
        model,
        scratch_plus,
    )

    J_plus = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    mujoco.mj_jacSubtreeCom(
        model,
        scratch_plus,
        J_plus,
        root_body_id,
    )

    # ========================================================
    # MINUS
    # ========================================================

    q_minus = (
        data.qpos.copy()
    )

    mujoco.mj_integratePos(
        model,
        q_minus,
        qvel,
        -epsilon,
    )

    scratch_minus.qpos[:] = (
        q_minus
    )

    scratch_minus.qvel[:] = (
        qvel
    )

    scratch_minus.ctrl[:] = (
        data.ctrl
    )

    mujoco.mj_forward(
        model,
        scratch_minus,
    )

    J_minus = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    mujoco.mj_jacSubtreeCom(
        model,
        scratch_minus,
        J_minus,
        root_body_id,
    )

    Jdot = (
        J_plus
        -
        J_minus
    ) / (
        2.0
        *
        epsilon
    )

    return (
        Jdot
        @
        qvel
    )


def run_double_support_preparation(
    model,
    data,
    dynamics,
    controller,

    actuated_qpos_indices,
    posture_reference,

    home_torque,

    torque_lower,
    torque_upper,

    left_support_bounds,
    right_support_bounds,

    planner,

    viewer=None,
):

    dt = float(
        model.opt.timestep
    )

    control_decimation = int(
        round(
            (
                1.0
                /
                CONTROL_FREQUENCY
            )
            /
            dt
        )
    )

    if control_decimation < 1:

        raise RuntimeError(
            "Invalid preparation control decimation."
        )

    base_body_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ROBOT_ROOT_BODY,
    )

    left_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        LEFT_FOOT_GEOM,
    )

    right_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        RIGHT_FOOT_GEOM,
    )

    floor_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FLOOR_GEOM,
    )

    floor_z = float(
        data.geom_xpos[
            floor_geom_id,
            2
        ]
    )

    mujoco.mj_forward(
        model,
        data,
    )

    initial_com = dynamics.get_com_kinematics(
        data,
        ROBOT_ROOT_BODY,
    )

    initial_left = dynamics.get_site_kinematics(
        data,
        LEFT_FOOT_SITE,
    )

    initial_right = dynamics.get_site_kinematics(
        data,
        RIGHT_FOOT_SITE,
    )

    initial_com_y = float(
        initial_com.position[
            1
        ]
    )

    target_com_y = float(
        initial_left.position[
            1
        ]
        -
        PREPARE_LEFT_INNER_OFFSET
    )

    delta_y = (
        target_com_y
        -
        initial_com_y
    )

    robot_weight = (
        initial_com.mass
        *
        np.linalg.norm(
            model.opt.gravity
        )
    )

    scratch_plus = mujoco.MjData(
        model
    )

    scratch_minus = mujoco.MjData(
        model
    )

    phase_start_time = float(
        data.time
    )

    last_tau = (
        home_torque.copy()
    )

    last_solution = None

    step_index = 0

    next_print_time = 0.0

    total_duration = (
        PREPARE_DURATION
        +
        PREPARE_HOLD_TIME
    )

    print()

    separator()

    print(
        "DOUBLE-SUPPORT PREPARATION"
    )

    separator()

    print(
        f"initial CoM-y = "
        f"{initial_com_y:+.6f} m"
    )

    print(
        f"left foot y   = "
        f"{initial_left.position[1]:+.6f} m"
    )

    print(
        f"target CoM-y  = "
        f"{target_com_y:+.6f} m"
    )

    while True:

        mujoco.mj_step1(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        elapsed_time = (
            current_time
            -
            phase_start_time
        )

        com = dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )

        left = dynamics.get_site_kinematics(
            data,
            LEFT_FOOT_SITE,
        )

        right = dynamics.get_site_kinematics(
            data,
            RIGHT_FOOT_SITE,
        )

        left_contact = has_geom_contact(
            data,
            left_geom_id,
            floor_geom_id,
        )

        right_contact = has_geom_contact(
            data,
            right_geom_id,
            floor_geom_id,
        )

        tilt = get_base_tilt_deg(
            data,
            base_body_id,
        )

        if (
            not left_contact
            or
            not right_contact
        ):

            raise RuntimeError(
                "A foot lost contact during "
                "double-support preparation."
            )

        # --------------------------------------------
        # Quintic preparation profile
        # --------------------------------------------

        if (
            elapsed_time
            <
            PREPARE_DURATION
        ):

            r = float(
                np.clip(
                    elapsed_time
                    /
                    PREPARE_DURATION,
                    0.0,
                    1.0,
                )
            )

            s = (
                10.0 * r**3
                -
                15.0 * r**4
                +
                6.0 * r**5
            )

            s_dot = (
                (
                    30.0 * r**2
                    -
                    60.0 * r**3
                    +
                    30.0 * r**4
                )
                /
                PREPARE_DURATION
            )

            s_ddot = (
                (
                    60.0 * r
                    -
                    180.0 * r**2
                    +
                    120.0 * r**3
                )
                /
                PREPARE_DURATION**2
            )

        else:

            s = 1.0
            s_dot = 0.0
            s_ddot = 0.0

        desired_y = (
            initial_com_y
            +
            delta_y * s
        )

        desired_vy = (
            delta_y
            *
            s_dot
        )

        desired_ay_ff = (
            delta_y
            *
            s_ddot
        )

        desired_com_acceleration_y = (
            desired_ay_ff

            +
            PREPARE_COM_KP
            *
            (
                desired_y
                -
                com.position[1]
            )

            +
            PREPARE_COM_KD
            *
            (
                desired_vy
                -
                com.velocity[1]
            )
        )

        desired_left_fraction = (
            0.5
            +
            (
                PREPARE_LEFT_LOAD_FRACTION
                -
                0.5
            )
            *
            s
        )

        desired_right_fraction = (
            1.0
            -
            desired_left_fraction
        )

        if (
            step_index
            %
            control_decimation
            ==
            0
        ):

            terms = dynamics.compute(
                data,
                forward=False,
            )

            com_jdot_v = compute_com_jdot_v(
                model=model,
                data=data,
                root_body_id=base_body_id,
                scratch_plus=scratch_plus,
                scratch_minus=scratch_minus,
                epsilon=COM_JDOT_EPSILON,
            )

            desired_com_acceleration_z = (
                COM_HEIGHT_KP
                *
                (
                    COM_HEIGHT_REFERENCE
                    -
                    com.position[2]
                )
                -
                COM_HEIGHT_KD
                *
                com.velocity[2]
            )

            q_actuated = (
                data.qpos[
                    actuated_qpos_indices
                ]
            )

            v_actuated = (
                data.qvel[
                    dynamics.actuated_dof_indices
                ]
            )

            desired_posture_acceleration = (
                POSTURE_KP
                *
                (
                    posture_reference
                    -
                    q_actuated
                )
                -
                POSTURE_KD
                *
                v_actuated
            )

            last_solution = (
                controller.solve_prepare(
                    mass_matrix=(
                        terms.mass_matrix
                    ),

                    effective_bias=(
                        terms.effective_bias
                    ),

                    selection_matrix=(
                        terms.selection_matrix
                    ),

                    left_jacobian=(
                        left.jacobian
                    ),

                    left_jdot_v=(
                        left.jacobian_dot_velocity
                    ),

                    right_jacobian=(
                        right.jacobian
                    ),

                    right_jdot_v=(
                        right.jacobian_dot_velocity
                    ),

                    com_jacobian=(
                        com.jacobian
                    ),

                    com_jdot_v_y=(
                        com_jdot_v[1]
                    ),

                    com_jdot_v_z=(
                        com_jdot_v[2]
                    ),

                    desired_com_acceleration_y=(
                        desired_com_acceleration_y
                    ),

                    desired_com_acceleration_z=(
                        desired_com_acceleration_z
                    ),

                    desired_left_fz=(
                        desired_left_fraction
                        *
                        robot_weight
                    ),

                    desired_right_fz=(
                        desired_right_fraction
                        *
                        robot_weight
                    ),

                    desired_posture_acceleration=(
                        desired_posture_acceleration
                    ),

                    left_support_bounds=(
                        left_support_bounds
                    ),

                    right_support_bounds=(
                        right_support_bounds
                    ),

                    left_contact_height=(
                        left.position[2]
                        -
                        floor_z
                    ),

                    right_contact_height=(
                        right.position[2]
                        -
                        floor_z
                    ),

                    torque_lower=(
                        torque_lower
                    ),

                    torque_upper=(
                        torque_upper
                    ),
                )
            )

            last_tau = (
                last_solution.torque.copy()
            )

        data.ctrl[:] = (
            last_tau
        )

        # --------------------------------------------
        # Status
        # --------------------------------------------

        if (
            elapsed_time
            >=
            next_print_time
            -
            0.5 * dt
        ):

            dcm_y = float(
                com.position[1]
                +
                com.velocity[1]
                /
                planner.omega
            )

            if last_solution is None:

                left_fraction = np.nan
                right_fraction = np.nan

                r2 = 0.0
                r3 = 0.0
                r4 = 0.0

            else:

                left_fraction = (
                    last_solution.left_wrench[2]
                    /
                    robot_weight
                )

                right_fraction = (
                    last_solution.right_wrench[2]
                    /
                    robot_weight
                )

                r2 = (
                    last_solution.rank2_residual
                )

                r3 = (
                    last_solution.rank3_residual
                )

                r4 = (
                    last_solution.rank4_residual
                )

            print(
                f"prepare t={elapsed_time:5.2f}"
                f" | CoMy={com.position[1]:+.4f}"
                f" | DCM_y={dcm_y:+.4f}"
                f" | yref={desired_y:+.4f}"
                f" | Vy={com.velocity[1]:+.4f}"
                f" | load L/R="
                f"{100.0 * left_fraction:5.1f}/"
                f"{100.0 * right_fraction:5.1f}%"
                f" | tilt={tilt:.2f}"
                f" | R2={r2:.2e}"
                f" | R3={r3:.2e}"
                f" | R4={r4:.2e}"
            )

            next_print_time += (
                PREPARE_STATUS_PRINT_PERIOD
            )

        if (
            viewer is not None
        ):

            viewer.sync()

        mujoco.mj_step2(
            model,
            data,
        )

        step_index += 1

        if (
            elapsed_time
            >=
            total_duration
        ):

            break

    # ========================================================
    # READINESS
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )

    final_com = dynamics.get_com_kinematics(
        data,
        ROBOT_ROOT_BODY,
    )

    final_left = dynamics.get_site_kinematics(
        data,
        LEFT_FOOT_SITE,
    )

    final_dcm_y = float(
        final_com.position[1]
        +
        final_com.velocity[1]
        /
        planner.omega
    )

    left_support_y_min = (
        final_left.position[1]
        +
        left_support_bounds[2]
        +
        PREPARE_DCM_MARGIN
    )

    left_support_y_max = (
        final_left.position[1]
        +
        left_support_bounds[3]
        -
        PREPARE_DCM_MARGIN
    )

    if last_solution is None:

        raise RuntimeError(
            "Preparation produced no HQP solution."
        )

    final_right_load_fraction = (
        last_solution.right_wrench[2]
        /
        robot_weight
    )

    final_tilt = get_base_tilt_deg(
        data,
        base_body_id,
    )

    ready = (
        left_support_y_min
        <=
        final_dcm_y
        <=
        left_support_y_max

        and

        abs(
            final_com.velocity[1]
        )
        <=
        PREPARE_VELOCITY_Y_MAX

        and

        final_right_load_fraction
        <=
        PREPARE_RIGHT_LOAD_MAX

        and

        final_tilt
        <=
        PREPARE_MAX_TILT_DEG
    )

    print()

    separator()

    print(
        "DOUBLE-SUPPORT PREPARATION RESULT"
    )

    separator()

    print(
        f"final CoM-y       = "
        f"{final_com.position[1]:+.6f} m"
    )

    print(
        f"final DCM-y       = "
        f"{final_dcm_y:+.6f} m"
    )

    print(
        f"LEFT support y    = "
        f"[{left_support_y_min:+.6f}, "
        f"{left_support_y_max:+.6f}] m"
    )

    print(
        f"final Vy          = "
        f"{final_com.velocity[1]:+.6f} m/s"
    )

    print(
        f"right load        = "
        f"{100.0 * final_right_load_fraction:.2f}%"
    )

    print(
        f"final tilt        = "
        f"{final_tilt:.3f} deg"
    )

    print(
        f"Rank-2 residual   = "
        f"{last_solution.rank2_residual:.6e}"
    )

    print(
        f"Rank-3 residual   = "
        f"{last_solution.rank3_residual:.6e}"
    )

    print(
        f"Rank-4 residual   = "
        f"{last_solution.rank4_residual:.6e}"
    )

    print(
        f"ready LEFT SS     = "
        f"{ready}"
    )

    if not ready:

        raise RuntimeError(
            "Double-support preparation did not "
            "reach a valid LEFT single-support state."
        )


# ============================================================
# SINGLE-SUPPORT / PLANNER-IN-THE-LOOP VALIDATION
# ============================================================

def run_single_support_step_validation(
    model,
    data,
    dynamics,
    controller,
    planner,

    actuated_qpos_indices,
    posture_reference,

    home_torque,

    torque_lower,
    torque_upper,

    left_support_bounds,
    right_support_bounds,

    viewer=None,
):

    # ========================================================
    # FIXED TEST PHASE
    #
    # Stage 4A validates ONE step only:
    #
    #     LEFT stance -> RIGHT swing
    #
    # No alternating gait state machine yet.
    # ========================================================

    stance_leg = StanceLeg.LEFT

    dt = float(
        model.opt.timestep
    )

    simulation_frequency = (
        1.0
        /
        dt
    )

    requested_control_period = (
        1.0
        /
        CONTROL_FREQUENCY
    )

    control_decimation = int(
        round(
            requested_control_period
            /
            dt
        )
    )

    if control_decimation < 1:

        raise RuntimeError(
            "WBC frequency is greater than MuJoCo frequency."
        )

    actual_control_period = (
        control_decimation
        *
        dt
    )

    if (
        abs(
            actual_control_period
            -
            requested_control_period
        )
        >
        1.0e-12
    ):

        raise RuntimeError(
            "CONTROL_FREQUENCY is not an integer "
            "decimation of simulation frequency."
        )

    planner_period = (
        1.0
        /
        PLANNER_UPDATE_FREQUENCY
    )

    if planner_period < actual_control_period:

        raise RuntimeError(
            "PLANNER_UPDATE_FREQUENCY must not exceed "
            "the WBC frequency in this validation."
        )

    # ========================================================
    # IDS
    # ========================================================

    base_body_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ROBOT_ROOT_BODY,
    )

    left_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        LEFT_FOOT_GEOM,
    )

    right_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        RIGHT_FOOT_GEOM,
    )

    floor_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FLOOR_GEOM,
    )

    floor_plane_z = float(
        data.geom_xpos[
            floor_geom_id,
            2
        ]
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    com = dynamics.get_com_kinematics(
        data,
        ROBOT_ROOT_BODY,
    )

    left = dynamics.get_site_kinematics(
        data,
        LEFT_FOOT_SITE,
    )

    right = dynamics.get_site_kinematics(
        data,
        RIGHT_FOOT_SITE,
    )

    if stance_leg is StanceLeg.LEFT:

        stance = left
        swing = right

        stance_support_bounds = (
            left_support_bounds
        )

        stance_geom_id = (
            left_geom_id
        )

        swing_geom_id = (
            right_geom_id
        )

    else:

        stance = right
        swing = left

        stance_support_bounds = (
            right_support_bounds
        )

        stance_geom_id = (
            right_geom_id
        )

        swing_geom_id = (
            left_geom_id
        )

    stance_initial_position = (
        stance.position.copy()
    )

    swing_initial_position = (
        swing.position.copy()
    )

    # ========================================================
    # NOMINAL STEP
    # ========================================================

    nominal_step = planner.compute_nominal_step(
        desired_velocity_x=(
            DESIRED_VELOCITY_X
        ),
        desired_velocity_y=(
            DESIRED_VELOCITY_Y
        ),
        stance_leg=(
            stance_leg
        ),
    )

    # ========================================================
    # INITIAL PLANNER SOLUTION
    # ========================================================

    dcm_xy = (
        np.asarray(
            com.position[
                0:2
            ],
            dtype=float,
        )
        +
        np.asarray(
            com.velocity[
                0:2
            ],
            dtype=float,
        )
        /
        planner.omega
    )

    planner_result = planner.solve_adaptive_step(
        nominal_step=(
            nominal_step
        ),
        dcm_measured=(
            dcm_xy
        ),
        stance_position=np.asarray(
            stance.position[
                0:2
            ],
            dtype=float,
        ),
        elapsed_time=0.0,
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

    # ========================================================
    # SWING TRAJECTORY
    # ========================================================

    vertical_parameters = VerticalSwingQPParameters(
        desired_height=(
            SWING_HEIGHT_DESIRED
        ),
        maximum_height=(
            SWING_HEIGHT_MAX
        ),
        constraint_samples=(
            SWING_VERTICAL_CONSTRAINT_SAMPLES
        ),
        coefficient_regularization=(
            SWING_VERTICAL_COEFFICIENT_REGULARIZATION
        ),
        bound_tolerance=(
            SWING_VERTICAL_BOUND_TOLERANCE
        ),
        max_refinements=(
            SWING_VERTICAL_MAX_REFINEMENTS
        ),
    )

    swing_trajectory = (
        OnlineSwingFootTrajectory()
    )

    swing_trajectory.reset_3d(
        initial_position=(
            swing_initial_position
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

    trajectory_sample = (
        swing_trajectory.update_3d(
            current_time=0.0,
            landing_time=(
                planner_result.step_time
            ),
            landing_position_xy=np.array(
                [
                    planner_result.step_location_x,
                    planner_result.step_location_y,
                ],
                dtype=float,
            ),
            vertical_parameters=(
                vertical_parameters
            ),
        )
    )

    trajectory_sample_time = 0.0

    # ========================================================
    # STANCE WRENCH REFERENCE
    # ========================================================

    robot_weight = (
        com.mass
        *
        np.linalg.norm(
            model.opt.gravity
        )
    )

    stance_wrench_reference = np.array(
        [
            0.0,
            0.0,
            robot_weight,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # ========================================================
    # SCRATCH DATA FOR COM Jdot*v
    # ========================================================

    scratch_plus = mujoco.MjData(
        model
    )

    scratch_minus = mujoco.MjData(
        model
    )

    # ========================================================
    # PHASE STATE
    # ========================================================

    phase_start_time = float(
        data.time
    )

    next_planner_update = (
        planner_period
    )

    planner_frozen = False

    freeze_time = None

    touchdown_time = None

    # ========================================================
    # CONTROL STATE
    # ========================================================

    last_tau = (
        home_torque.copy()
    )

    last_solution = None

    # ========================================================
    # METRICS
    # ========================================================

    max_swing_position_error = 0.0
    max_stance_slip = 0.0
    max_tilt = 0.0
    max_tau_utilization = 0.0

    total_hqp_times = []

    rank5_fallback_count = 0

    # ========================================================
    # PRINT / VIEWER
    # ========================================================

    next_print_elapsed = (
        STATUS_PRINT_PERIOD
    )

    next_viewer_time = float(
        data.time
    )

    viewer_period = (
        1.0
        /
        VIEWER_REFRESH_FREQUENCY
    )

    wall_start = (
        time.perf_counter()
    )

    simulation_start_time = float(
        data.time
    )

    step_index = 0

    print()

    print(
        f"simulation frequency = "
        f"{simulation_frequency:.1f} Hz"
    )

    print(
        f"WBC frequency        = "
        f"{CONTROL_FREQUENCY:.1f} Hz"
    )

    print(
        f"planner frequency    = "
        f"{PLANNER_UPDATE_FREQUENCY:.1f} Hz"
    )

    print(
        f"stance leg           = "
        f"{stance_leg.value}"
    )

    print(
        f"initial planner T    = "
        f"{planner_result.step_time:.6f} s"
    )

    print(
        "initial planner uT   = "
        f"({planner_result.step_location_x:+.6f}, "
        f"{planner_result.step_location_y:+.6f}) m"
    )

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        if (
            viewer is not None
            and
            not viewer.is_running()
        ):

            break

        mujoco.mj_step1(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        elapsed_time = (
            current_time
            -
            phase_start_time
        )

        # ----------------------------------------------------
        # Fresh robot state
        # ----------------------------------------------------

        com = dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )

        left = dynamics.get_site_kinematics(
            data,
            LEFT_FOOT_SITE,
        )

        right = dynamics.get_site_kinematics(
            data,
            RIGHT_FOOT_SITE,
        )

        if stance_leg is StanceLeg.LEFT:

            stance = left
            swing = right

        else:

            stance = right
            swing = left

        tilt = get_base_tilt_deg(
            data,
            base_body_id,
        )

        stance_contact = has_geom_contact(
            data,
            stance_geom_id,
            floor_geom_id,
        )

        swing_contact = has_geom_contact(
            data,
            swing_geom_id,
            floor_geom_id,
        )

        stance_slip = float(
            np.linalg.norm(
                stance.position[
                    0:2
                ]
                -
                stance_initial_position[
                    0:2
                ]
            )
        )

        max_stance_slip = max(
            max_stance_slip,
            stance_slip,
        )

        max_tilt = max(
            max_tilt,
            tilt,
        )

        # ----------------------------------------------------
        # Emergency
        # ----------------------------------------------------

        if (
            tilt
            >
            EMERGENCY_BASE_TILT_DEG
        ):

            raise RuntimeError(
                "Emergency base tilt at "
                f"t={elapsed_time:.6f} s."
            )

        if (
            com.position[
                2
            ]
            <
            COM_HEIGHT_REFERENCE
            -
            EMERGENCY_COM_DROP
        ):

            raise RuntimeError(
                "Emergency CoM drop at "
                f"t={elapsed_time:.6f} s."
            )

        # ----------------------------------------------------
        # DCM
        # ----------------------------------------------------

        dcm_xy = (
            np.asarray(
                com.position[
                    0:2
                ],
                dtype=float,
            )
            +
            np.asarray(
                com.velocity[
                    0:2
                ],
                dtype=float,
            )
            /
            planner.omega
        )

        # ----------------------------------------------------
        # Freeze planner near landing.
        #
        # Once:
        #
        #     t >= T - T_gap
        #
        # the latest landing target is kept fixed.
        # ----------------------------------------------------

        if (
            not planner_frozen
            and
            elapsed_time
            >=
            planner_result.step_time
            -
            STEP_TIMING_GAP
        ):

            planner_frozen = True

            freeze_time = float(
                elapsed_time
            )

        # ----------------------------------------------------
        # Planner + trajectory regeneration
        # ----------------------------------------------------

        if (
            elapsed_time
            >=
            next_planner_update
            -
            0.5
            *
            dt
            and
            elapsed_time
            <
            planner_result.step_time
            -
            0.5
            *
            dt
        ):

            if not planner_frozen:

                planner_result = planner.solve_adaptive_step(
                    nominal_step=(
                        nominal_step
                    ),
                    dcm_measured=(
                        dcm_xy
                    ),
                    stance_position=np.asarray(
                        stance.position[
                            0:2
                        ],
                        dtype=float,
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

                if (
                    elapsed_time
                    >=
                    planner_result.step_time
                    -
                    STEP_TIMING_GAP
                ):

                    planner_frozen = True

                    freeze_time = float(
                        elapsed_time
                    )

            trajectory_sample = (
                swing_trajectory.update_3d(
                    current_time=(
                        elapsed_time
                    ),
                    landing_time=(
                        planner_result.step_time
                    ),
                    landing_position_xy=np.array(
                        [
                            planner_result.step_location_x,
                            planner_result.step_location_y,
                        ],
                        dtype=float,
                    ),
                    vertical_parameters=(
                        vertical_parameters
                    ),
                )
            )

            trajectory_sample_time = float(
                elapsed_time
            )

            while (
                next_planner_update
                <=
                elapsed_time
                +
                0.5
                *
                dt
            ):

                next_planner_update += (
                    planner_period
                )

        # ----------------------------------------------------
        # Desired swing state.
        #
        # Between planner/trajectory updates, propagate the
        # last sampled state with constant acceleration.
        # This avoids a zero-order hold on desired position.
        # ----------------------------------------------------

        if (
            elapsed_time
            <
            planner_result.step_time
        ):

            hold_time = max(
                0.0,
                elapsed_time
                -
                trajectory_sample_time,
            )

            desired_swing_position = (
                trajectory_sample.position
                +
                trajectory_sample.velocity
                *
                hold_time
                +
                0.5
                *
                trajectory_sample.acceleration
                *
                hold_time**2
            )

            desired_swing_velocity = (
                trajectory_sample.velocity
                +
                trajectory_sample.acceleration
                *
                hold_time
            )

            desired_swing_feedforward_acceleration = (
                trajectory_sample.acceleration.copy()
            )

        else:

            desired_swing_position = np.array(
                [
                    planner_result.step_location_x,
                    planner_result.step_location_y,
                    swing_initial_position[
                        2
                    ],
                ],
                dtype=float,
            )

            desired_swing_velocity = np.zeros(
                3,
                dtype=float,
            )

            desired_swing_feedforward_acceleration = np.zeros(
                3,
                dtype=float,
            )

        swing_position_error = (
            desired_swing_position
            -
            swing.position
        )

        swing_velocity_error = (
            desired_swing_velocity
            -
            swing.linear_velocity
        )

        swing_position_error_norm = float(
            np.linalg.norm(
                swing_position_error
            )
        )

        max_swing_position_error = max(
            max_swing_position_error,
            swing_position_error_norm,
        )

        desired_swing_linear_acceleration = (
            desired_swing_feedforward_acceleration
            +
            SWING_TRACKING_KP
            *
            swing_position_error
            +
            SWING_TRACKING_KD
            *
            swing_velocity_error
        )

        # ----------------------------------------------------
        # HQP update
        # ----------------------------------------------------

        if (
            step_index
            %
            control_decimation
            ==
            0
        ):

            terms = dynamics.compute(
                data,
                forward=False,
            )

            com_jdot_v = compute_com_jdot_v(
                model=model,
                data=data,
                root_body_id=(
                    base_body_id
                ),
                scratch_plus=(
                    scratch_plus
                ),
                scratch_minus=(
                    scratch_minus
                ),
                epsilon=(
                    COM_JDOT_EPSILON
                ),
            )

            desired_com_acceleration_z = (
                COM_HEIGHT_KP
                *
                (
                    COM_HEIGHT_REFERENCE
                    -
                    com.position[
                        2
                    ]
                )
                -
                COM_HEIGHT_KD
                *
                com.velocity[
                    2
                ]
            )

            q_actuated = (
                data.qpos[
                    actuated_qpos_indices
                ]
            )

            v_actuated = (
                data.qvel[
                    dynamics.actuated_dof_indices
                ]
            )

            desired_posture_acceleration = (
                POSTURE_KP
                *
                (
                    posture_reference
                    -
                    q_actuated
                )
                -
                POSTURE_KD
                *
                v_actuated
            )

            stance_contact_height = float(
                stance.position[
                    2
                ]
                -
                floor_plane_z
            )

            hqp_start = (
                time.perf_counter()
            )

            solution = controller.solve(
                mass_matrix=(
                    terms.mass_matrix
                ),

                effective_bias=(
                    terms.effective_bias
                ),

                selection_matrix=(
                    terms.selection_matrix
                ),

                stance_jacobian=(
                    stance.jacobian
                ),

                stance_jdot_v=(
                    stance.jacobian_dot_velocity
                ),

                swing_jacobian=(
                    swing.jacobian
                ),

                swing_jdot_v=(
                    swing.jacobian_dot_velocity
                ),

                com_jacobian=(
                    com.jacobian
                ),

                com_jdot_v_z=(
                    com_jdot_v[
                        2
                    ]
                ),

                desired_com_acceleration_z=(
                    desired_com_acceleration_z
                ),

                desired_swing_linear_acceleration=(
                    desired_swing_linear_acceleration
                ),

                desired_posture_acceleration=(
                    desired_posture_acceleration
                ),

                stance_wrench_reference=(
                    stance_wrench_reference
                ),

                stance_support_bounds=(
                    stance_support_bounds
                ),

                stance_contact_height=(
                    stance_contact_height
                ),

                torque_lower=(
                    torque_lower
                ),

                torque_upper=(
                    torque_upper
                ),
            )

            if not solution.rank5_used:

                rank5_fallback_count += 1

            last_solution = (
                solution
            )

            last_tau = (
                solution.torque.copy()
            )

            hqp_elapsed = (
                time.perf_counter()
                -
                hqp_start
            )

            total_hqp_times.append(
                hqp_elapsed
            )


        # ----------------------------------------------------
        # Apply torque
        # ----------------------------------------------------

        data.ctrl[:] = (
            last_tau
        )

        torque_limit_abs = np.maximum(
            np.abs(
                torque_lower
            ),
            np.abs(
                torque_upper
            ),
        )

        torque_utilization = float(
            np.max(
                np.abs(
                    last_tau
                )
                /
                torque_limit_abs
            )
        )

        max_tau_utilization = max(
            max_tau_utilization,
            torque_utilization,
        )

        # ----------------------------------------------------
        # Touchdown observation
        #
        # Ignore initial swing-foot contact. Only accept contact
        # near/after the planned landing time.
        # ----------------------------------------------------

        if (
            touchdown_time is None
            and
            swing_contact
            and
            elapsed_time
            >=
            planner_result.step_time
            -
            TOUCHDOWN_CONTACT_WINDOW
        ):

            touchdown_time = float(
                elapsed_time
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if (
            elapsed_time
            >=
            next_print_elapsed
            -
            0.5
            *
            dt
        ):

            if last_solution is None:

                rank2 = 0.0
                rank3 = 0.0
                rank4 = 0.0
                rank5 = 0.0

                hqp_ms = 0.0

                cop = np.array(
                    [
                        np.nan,
                        np.nan,
                    ],
                    dtype=float,
                )

            else:

                rank2 = (
                    last_solution.rank2_residual
                )

                rank3 = (
                    last_solution.rank3_residual
                )

                rank4 = (
                    last_solution.rank4_residual
                )

                rank5 = (
                    last_solution.rank5_residual
                )

                hqp_ms = (
                    1000.0
                    *
                    (
                        last_solution.solve_time_rank2
                        +
                        last_solution.solve_time_rank3
                        +
                        last_solution.solve_time_rank4
                        +
                        last_solution.solve_time_rank5
                    )
                )

                cop = compute_cop_from_wrench(
                    last_solution.stance_wrench,
                    (
                        stance.position[
                            2
                        ]
                        -
                        floor_plane_z
                    ),
                )

            print(
                f"t={elapsed_time:6.3f} s"

                f" | DCM="
                f"({dcm_xy[0]:+.4f},"
                f"{dcm_xy[1]:+.4f})"

                f" | uT="
                f"({planner_result.step_location_x:+.4f},"
                f"{planner_result.step_location_y:+.4f})"

                f" | T="
                f"{planner_result.step_time:.4f}"

                f" | frozen="
                f"{int(planner_frozen)}"

                f" | swing_err="
                f"{1000.0 * swing_position_error_norm:.2f} mm"

                f" | contact="
                f"{int(stance_contact)}/"
                f"{int(swing_contact)}"

                f" | CoP="
                f"({1000.0 * cop[0]:+.1f},"
                f"{1000.0 * cop[1]:+.1f}) mm"

                f" | tau="
                f"{100.0 * torque_utilization:.1f}%"

                f" | R2="
                f"{rank2:.2e}"

                f" | R3="
                f"{rank3:.2e}"

                f" | R4="
                f"{rank4:.2e}"

                f" | R5="
                f"{rank5:.2e}"

                f" | HQP="
                f"{hqp_ms:.2f} ms"
            )

            next_print_elapsed += (
                STATUS_PRINT_PERIOD
            )

        # ----------------------------------------------------
        # Viewer
        # ----------------------------------------------------

        if (
            viewer is not None
            and
            current_time
            >=
            next_viewer_time
            -
            0.5
            *
            dt
        ):

            viewer.sync()

            next_viewer_time += (
                viewer_period
            )

            simulation_elapsed = (
                current_time
                -
                simulation_start_time
            )

            target_wall_elapsed = (
                simulation_elapsed
                /
                REALTIME_FACTOR
            )

            current_wall_elapsed = (
                time.perf_counter()
                -
                wall_start
            )

            sleep_time = (
                target_wall_elapsed
                -
                current_wall_elapsed
            )

            if sleep_time > 0.0:

                time.sleep(
                    sleep_time
                )

        mujoco.mj_step2(
            model,
            data,
        )

        step_index += 1

        # ----------------------------------------------------
        # End after a short touchdown-settling period.
        # ----------------------------------------------------

        if (
            elapsed_time
            >=
            planner_result.step_time
            +
            TOUCHDOWN_SETTLE_TIME
        ):

            break

    # ========================================================
    # FINAL STATE
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )

    final_com = dynamics.get_com_kinematics(
        data,
        ROBOT_ROOT_BODY,
    )

    final_left = dynamics.get_site_kinematics(
        data,
        LEFT_FOOT_SITE,
    )

    final_right = dynamics.get_site_kinematics(
        data,
        RIGHT_FOOT_SITE,
    )

    if stance_leg is StanceLeg.LEFT:

        final_stance = final_left
        final_swing = final_right

    else:

        final_stance = final_right
        final_swing = final_left

    final_target = np.array(
        [
            planner_result.step_location_x,
            planner_result.step_location_y,
            swing_initial_position[
                2
            ],
        ],
        dtype=float,
    )

    final_swing_error = float(
        np.linalg.norm(
            final_swing.position
            -
            final_target
        )
    )

    final_stance_slip = float(
        np.linalg.norm(
            final_stance.position[
                0:2
            ]
            -
            stance_initial_position[
                0:2
            ]
        )
    )

    final_tilt = get_base_tilt_deg(
        data,
        base_body_id,
    )

    final_stance_contact = has_geom_contact(
        data,
        stance_geom_id,
        floor_geom_id,
    )

    final_swing_contact = has_geom_contact(
        data,
        swing_geom_id,
        floor_geom_id,
    )

    if total_hqp_times:

        mean_hqp_ms = (
            1000.0
            *
            float(
                np.mean(
                    total_hqp_times
                )
            )
        )

        max_hqp_ms = (
            1000.0
            *
            float(
                np.max(
                    total_hqp_times
                )
            )
        )

    else:

        mean_hqp_ms = 0.0
        max_hqp_ms = 0.0

    print()

    separator()

    print(
        "SINGLE-SUPPORT PLANNER/WBC RESULT"
    )

    separator()

    print()

    print(
        f"stance leg             = "
        f"{stance_leg.value}"
    )

    print(
        f"final planner uT       = "
        f"({planner_result.step_location_x:+.6f}, "
        f"{planner_result.step_location_y:+.6f}) m"
    )

    print(
        f"final planner T        = "
        f"{planner_result.step_time:.6f} s"
    )

    print(
        f"planner frozen         = "
        f"{planner_frozen}"
    )

    print(
        f"freeze time            = "
        f"{freeze_time}"
    )

    print(
        f"observed touchdown     = "
        f"{touchdown_time}"
    )

    print()

    print(
        f"final swing target     = "
        f"{final_target}"
    )

    print(
        f"final swing position   = "
        f"{final_swing.position}"
    )

    print(
        f"final swing error      = "
        f"{1000.0 * final_swing_error:.3f} mm"
    )

    print(
        f"max swing error        = "
        f"{1000.0 * max_swing_position_error:.3f} mm"
    )

    print(
        f"final stance slip      = "
        f"{1000.0 * final_stance_slip:.3f} mm"
    )

    print(
        f"max stance slip        = "
        f"{1000.0 * max_stance_slip:.3f} mm"
    )

    print(
        f"final base tilt        = "
        f"{final_tilt:.3f} deg"
    )

    print(
        f"max base tilt          = "
        f"{max_tilt:.3f} deg"
    )

    print(
        f"final stance contact   = "
        f"{final_stance_contact}"
    )

    print(
        f"final swing contact    = "
        f"{final_swing_contact}"
    )

    print(
        f"final CoM height       = "
        f"{final_com.position[2]:.8f} m"
    )

    print(
        f"max torque utilization = "
        f"{100.0 * max_tau_utilization:.2f}%"
    )

    print()

    print(
        f"mean HQP time          = "
        f"{mean_hqp_ms:.3f} ms"
    )

    print(
        f"max HQP time           = "
        f"{max_hqp_ms:.3f} ms"
    )

    print(
        f"Rank-5 fallbacks       = "
        f"{rank5_fallback_count}"
    )

    if last_solution is not None:

        print()

        print(
            f"Rank-2 residual        = "
            f"{last_solution.rank2_residual:.6e}"
        )

        print(
            f"Rank-3 residual        = "
            f"{last_solution.rank3_residual:.6e}"
        )

        print(
            f"Rank-4 residual        = "
            f"{last_solution.rank4_residual:.6e}"
        )

        print(
            f"Rank-5 residual        = "
            f"{last_solution.rank5_residual:.6e}"
        )

        print(
            f"max constraint violation = "
            f"{last_solution.max_constraint_violation:.6e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    
    # ========================================================
    # ADAPTIVE STEP PLANNER
    # ========================================================

    planner_parameters = StepPlannerParameters(

        gravity=(
            GRAVITY
        ),

        com_height=(
            COM_HEIGHT_REFERENCE
        ),

        default_step_width=(
            DEFAULT_STEP_WIDTH
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

    planner = AdaptiveStepPlanner(
        planner_parameters
    )

    separator()

    print(
        "STEP TIMING ADAPTATION"
    )

    print(
        "HIERARCHICAL INVERSE DYNAMICS"
    )

    print(
        "SINGLE-SUPPORT ADAPTIVE STEP TEST"
    )

    separator()

    print()

    print(
        f"MuJoCo = "
        f"{mujoco.__version__}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    if not SCENE_XML.exists():

        raise FileNotFoundError(
            f"Scene not found:\n"
            f"{SCENE_XML}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(
            SCENE_XML
        )
    )

    data = mujoco.MjData(
        model
    )

    print()

    print(
        f"nq = {model.nq}"
    )

    print(
        f"nv = {model.nv}"
    )

    print(
        f"nu = {model.nu}"
    )

    print(
        f"dt = {model.opt.timestep}"
    )

    print(
        f"integrator = "
        f"{mujoco.mjtIntegrator(model.opt.integrator).name}"
    )

    if (
        model.opt.integrator
        ==
        mujoco.mjtIntegrator.mjINT_RK4
    ):

        raise RuntimeError(
            "This split-step controller "
            "must not use RK4."
        )

    # ========================================================
    # HOME
    # ========================================================

    home_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    # Preserve HOME ctrl.

    home_torque = (
        data.ctrl.copy()
    )

    # Initial generalized velocity.

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    # ========================================================
    # DYNAMICS BACKEND
    # ========================================================

    dynamics = WholeBodyDynamicsModel(
        model
    )

    # ========================================================
    # ACTUATED QPOS
    # ========================================================

    actuated_qpos_indices = (
        build_actuated_qpos_indices(
            model
        )
    )

    posture_reference = (
        data.qpos[
            actuated_qpos_indices
        ]
        .copy()
    )

    # ========================================================
    # TORQUE LIMITS
    # ========================================================

    torque_lower = (
        model.actuator_ctrlrange[
            :,
            0
        ]
        .copy()
    )

    torque_upper = (
        model.actuator_ctrlrange[
            :,
            1
        ]
        .copy()
    )

    # ========================================================
    # SUPPORT POLYGONS
    # ========================================================

    left_support_bounds = (
        get_support_rectangle_from_geom_aabb(

            model=model,

            data=data,

            geom_name=(
                LEFT_FOOT_GEOM
            ),

            site_name=(
                LEFT_FOOT_SITE
            ),

            safety_margin=(
                SUPPORT_POLYGON_MARGIN
            ),
        )
    )

    right_support_bounds = (
        get_support_rectangle_from_geom_aabb(

            model=model,

            data=data,

            geom_name=(
                RIGHT_FOOT_GEOM
            ),

            site_name=(
                RIGHT_FOOT_SITE
            ),

            safety_margin=(
                SUPPORT_POLYGON_MARGIN
            ),
        )
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_com = (
        dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )
    )

    initial_left = (
        dynamics.get_site_kinematics(
            data,
            LEFT_FOOT_SITE,
        )
    )

    initial_right = (
        dynamics.get_site_kinematics(
            data,
            RIGHT_FOOT_SITE,
        )
    )

    floor_geom_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FLOOR_GEOM,
    )

    floor_z = float(
        data.geom_xpos[
            floor_geom_id,
            2
        ]
    )

    initial_left_height = float(
        initial_left.position[
            2
        ]
        -
        floor_z
    )

    initial_right_height = float(
        initial_right.position[
            2
        ]
        -
        floor_z
    )

    # ========================================================
    # PRINT ARCHITECTURE
    # ========================================================

    separator()

    print(
        "SINGLE-SUPPORT HQP ARCHITECTURE"
    )

    separator()

    print()

    print(
        "Decision vector:"
    )

    print(
        "  y = [qddot(20), lambda_stance(6)]"
    )

    print(
        "  total = 26 variables"
    )

    print()

    print(
        "Rank 1:"
    )

    print(
        "  floating-base Newton-Euler"
    )

    print(
        "  torque limits"
    )

    print(
        "  unilateral stance contact"
    )

    print(
        "  friction feasibility"
    )

    print(
        "  stance CoP feasibility"
    )

    print()

    print(
        "Rank 2:"
    )

    print(
        "  stance foot 6D"
    )

    print(
        "  CoM height"
    )

    print()

    print(
        "Rank 3:"
    )

    print(
        "  swing-foot translation XYZ"
    )

    print()

    print(
        "Rank 4:"
    )

    print(
        "  posture"
    )

    print()

    print(
        "Rank 5:"
    )

    print(
        "  stance-wrench regularization"
    )

    print()

    print(
        "No horizontal CoM control."
    )

    print(
        "No CoP tracking."
    )

    print(
        "No ZMP tracking."
    )

    print()

    print(
        f"Initial CoM height = "
        f"{initial_com.position[2]:.8f} m"
    )

    print(
        f"Desired CoM height = "
        f"{COM_HEIGHT_REFERENCE:.8f} m"
    )

    print()

    print(
        "Left support bounds "
        "[xmin xmax ymin ymax] [m]:"
    )

    print(
        left_support_bounds
    )

    print(
        "Right support bounds "
        "[xmin xmax ymin ymax] [m]:"
    )

    print(
        right_support_bounds
    )

    print()

    print(
        f"Initial left site height above ground  = "
        f"{initial_left_height:.6f} m"
    )

    print(
        f"Initial right site height above ground = "
        f"{initial_right_height:.6f} m"
    )

    print()

    print(
        "HOME torque:"
    )

    print(
        home_torque
    )

    print()

    print(
        f"CoM PD: "
        f"Kp={COM_HEIGHT_KP}, "
        f"Kd={COM_HEIGHT_KD}"
    )

    print(
        f"Posture PD: "
        f"Kp={POSTURE_KP}, "
        f"Kd={POSTURE_KD}"
    )

    separator()

    # ========================================================
    # WHOLE-BODY CONTROLLER
    # ========================================================

    wbc_config = WholeBodyHQPConfig(
        friction_coefficient=(
            FRICTION_COEFFICIENT
        ),

        numerical_regularization=(
            1.0e-4
        ),

        svd_tolerance=(
            1.0e-9
        ),

        constraint_tolerance=(
            1.0e-6
        ),

        solver_name="qrqp",
    )

    controller = (
        WholeBodyHierarchicalInverseDynamics(
            nv=(
                model.nv
            ),

            nu=(
                model.nu
            ),

            actuated_dof_indices=(
                dynamics.actuated_dof_indices
            ),

            config=(
                wbc_config
            ),
        )
    )

    prepare_controller = (
        DoubleSupportPreparationHierarchicalInverseDynamics(
            nv=(
                model.nv
            ),

            nu=(
                model.nu
            ),

            actuated_dof_indices=(
                dynamics.actuated_dof_indices
            ),

            config=(
                wbc_config
            ),
        )
    )

    # ========================================================
    # RUN
    # ========================================================

    if SHOW_VIEWER:

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            run_double_support_preparation(
                model=model,
                data=data,
                dynamics=dynamics,
                controller=prepare_controller,

                actuated_qpos_indices=(
                    actuated_qpos_indices
                ),

                posture_reference=(
                    posture_reference
                ),

                home_torque=(
                    home_torque
                ),

                torque_lower=(
                    torque_lower
                ),

                torque_upper=(
                    torque_upper
                ),

                left_support_bounds=(
                    left_support_bounds
                ),

                right_support_bounds=(
                    right_support_bounds
                ),

                planner=planner,

                viewer=viewer,
            )

            run_single_support_step_validation(

                model=model,

                data=data,

                dynamics=dynamics,

                controller=controller,

                planner=planner,

                actuated_qpos_indices=(
                    actuated_qpos_indices
                ),

                posture_reference=(
                    posture_reference
                ),

                home_torque=(
                    home_torque
                ),

                torque_lower=(
                    torque_lower
                ),

                torque_upper=(
                    torque_upper
                ),

                left_support_bounds=(
                    left_support_bounds
                ),

                right_support_bounds=(
                    right_support_bounds
                ),

                viewer=(
                    viewer
                ),
            )

    else:

        run_double_support_preparation(
            model=model,
            data=data,
            dynamics=dynamics,
            controller=prepare_controller,

            actuated_qpos_indices=(
                actuated_qpos_indices
            ),

            posture_reference=(
                posture_reference
            ),

            home_torque=(
                home_torque
            ),

            torque_lower=(
                torque_lower
            ),

            torque_upper=(
                torque_upper
            ),

            left_support_bounds=(
                left_support_bounds
            ),

            right_support_bounds=(
                right_support_bounds
            ),

            planner=planner,

            viewer=viewer,
        )

        run_single_support_step_validation(

            model=model,

            data=data,

            dynamics=dynamics,

            controller=controller,

            planner=planner,

            actuated_qpos_indices=(
                actuated_qpos_indices
            ),

            posture_reference=(
                posture_reference
            ),

            home_torque=(
                home_torque
            ),

            torque_lower=(
                torque_lower
            ),

            torque_upper=(
                torque_upper
            ),

            left_support_bounds=(
                left_support_bounds
            ),

            right_support_bounds=(
                right_support_bounds
            ),

            viewer=None,
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()