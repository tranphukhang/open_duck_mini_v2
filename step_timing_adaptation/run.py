# step_timing_adaptation/run.py

from __future__ import annotations

from pathlib import Path
import time

import casadi as ca
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

    from .whole_body_qp import (
        DoubleSupportHQPConfig,
        WholeBodyHierarchicalInverseDynamics,
    )

    from .adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

else:

    from dynamics_model import (
        WholeBodyDynamicsModel,
    )

    from whole_body_qp import (
        DoubleSupportHQPConfig,
        WholeBodyHierarchicalInverseDynamics,
    )

    from adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
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

SIMULATION_DURATION = 5.0

CONTROL_FREQUENCY = 1000.0

SHOW_VIEWER = True

REALTIME_FACTOR = 1.0

VIEWER_REFRESH_FREQUENCY = 60.0

STATUS_PRINT_PERIOD = 0.5


# ============================================================
# HOME BOOTSTRAP
# ============================================================

HOME_TORQUE_BOOTSTRAP_TIME = (
    1.0
    /
    CONTROL_FREQUENCY
)


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
# Stage currently under validation
# ------------------------------------------------------------

RUN_STAGE1_PLANNER_ONLY = False

# Stop after Stage-2 standalone validation.
# MuJoCo / WBC will not run while this is True.
RUN_STAGE2_PLANNER_ONLY = True


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

# Standalone nominal-consistency test time inside the step.
STAGE2_TEST_ELAPSED_TIME = 0.10


# ------------------------------------------------------------
# Desired walking velocity
# ------------------------------------------------------------

DESIRED_VELOCITY_X = 0.15
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
# These values are currently used for Stage-1 validation.
# They can later be replaced by the identified physical
# limits of the Open Duck Mini.
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


# ============================================================
# DYNAMIC DOUBLE-SUPPORT TEST
# ============================================================

def run_double_support(
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

    viewer=None,
):

    # ========================================================
    # TIMING
    # ========================================================

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
            "WBC frequency is greater "
            "than MuJoCo frequency."
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

    bootstrap_steps = int(
        round(
            HOME_TORQUE_BOOTSTRAP_TIME
            /
            dt
        )
    )

    bootstrap_steps = max(
        bootstrap_steps,
        1,
    )

    print()

    print(
        f"simulation frequency = "
        f"{simulation_frequency:.1f} Hz"
    )

    print(
        f"HQP frequency        = "
        f"{CONTROL_FREQUENCY:.1f} Hz"
    )

    print(
        f"HQP every            = "
        f"{control_decimation} MuJoCo steps"
    )

    print(
        f"HOME bootstrap       = "
        f"{1000.0 * bootstrap_steps * dt:.3f} ms"
    )

    # ========================================================
    # IDS
    # ========================================================

    base_body_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ROBOT_ROOT_BODY,
    )

    left_site_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        LEFT_FOOT_SITE,
    )

    right_site_id = require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        RIGHT_FOOT_SITE,
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
    # SCRATCH DATA
    # ========================================================

    scratch_plus = mujoco.MjData(
        model
    )

    scratch_minus = mujoco.MjData(
        model
    )

    # ========================================================
    # CONTACT WRENCH REFERENCE
    # ========================================================

    initial_com = (
        dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )
    )

    robot_weight = (
        initial_com.mass
        *
        np.linalg.norm(
            model.opt.gravity
        )
    )

    half_weight = (
        0.5
        *
        robot_weight
    )

    left_wrench_reference = np.array(
        [
            0.0,
            0.0,
            half_weight,

            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    right_wrench_reference = (
        left_wrench_reference.copy()
    )

    # ========================================================
    # INITIAL FOOT POSITIONS
    # ========================================================

    left_foot_initial = (
        data.site_xpos[
            left_site_id
        ]
        .copy()
    )

    right_foot_initial = (
        data.site_xpos[
            right_site_id
        ]
        .copy()
    )

    # ========================================================
    # INITIAL CONTROL
    # ========================================================

    last_tau = (
        home_torque.copy()
    )

    data.ctrl[:] = (
        last_tau
    )

    last_solution = None

    # ========================================================
    # METRICS
    # ========================================================

    max_z_error = 0.0

    max_tilt = 0.0

    max_left_slip = 0.0

    max_right_slip = 0.0

    max_tau_utilization = 0.0

    both_contact_samples = 0

    total_samples = 0

    total_hqp_times = []

    rank5_fallback_count = 0

    # ========================================================
    # TIME
    # ========================================================

    simulation_start_time = float(
        data.time
    )

    simulation_end_time = (
        simulation_start_time
        +
        SIMULATION_DURATION
    )

    next_print_time = (
        simulation_start_time
        +
        STATUS_PRINT_PERIOD
    )

    next_viewer_time = (
        simulation_start_time
    )

    viewer_period = (
        1.0
        /
        VIEWER_REFRESH_FREQUENCY
    )

    wall_start = (
        time.perf_counter()
    )

    step_index = 0

    # ========================================================
    # LOOP
    # ========================================================

    while (
        data.time
        <
        simulation_end_time
        -
        0.5
        *
        dt
    ):

        if (
            viewer is not None
            and
            not viewer.is_running()
        ):

            break

        # ====================================================
        # STEP 1
        # ====================================================

        mujoco.mj_step1(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        # ====================================================
        # GROUND-TRUTH STATE
        # ====================================================

        com = (
            dynamics.get_com_kinematics(
                data,
                ROBOT_ROOT_BODY,
            )
        )

        left = (
            dynamics.get_site_kinematics(
                data,
                LEFT_FOOT_SITE,
            )
        )

        right = (
            dynamics.get_site_kinematics(
                data,
                RIGHT_FOOT_SITE,
            )
        )

        tilt = (
            get_base_tilt_deg(
                data,
                base_body_id,
            )
        )

        left_contact = (
            has_geom_contact(
                data,
                left_geom_id,
                floor_geom_id,
            )
        )

        right_contact = (
            has_geom_contact(
                data,
                right_geom_id,
                floor_geom_id,
            )
        )

        left_slip = float(
            np.linalg.norm(
                left.position[
                    0:2
                ]
                -
                left_foot_initial[
                    0:2
                ]
            )
        )

        right_slip = float(
            np.linalg.norm(
                right.position[
                    0:2
                ]
                -
                right_foot_initial[
                    0:2
                ]
            )
        )

        z_error = float(
            COM_HEIGHT_REFERENCE
            -
            com.position[
                2
            ]
        )

        left_contact_height = float(
            left.position[
                2
            ]
            -
            floor_plane_z
        )

        right_contact_height = float(
            right.position[
                2
            ]
            -
            floor_plane_z
        )

        # ====================================================
        # METRICS
        # ====================================================

        max_z_error = max(
            max_z_error,
            abs(
                z_error
            ),
        )

        max_tilt = max(
            max_tilt,
            tilt,
        )

        max_left_slip = max(
            max_left_slip,
            left_slip,
        )

        max_right_slip = max(
            max_right_slip,
            right_slip,
        )

        total_samples += 1

        if (
            left_contact
            and
            right_contact
        ):

            both_contact_samples += 1

        # ====================================================
        # EMERGENCY
        # ====================================================

        if (
            tilt
            >
            EMERGENCY_BASE_TILT_DEG
        ):

            raise RuntimeError(
                "Emergency base tilt at "
                f"t={current_time:.6f} s."
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
                f"t={current_time:.6f} s."
            )

        # ====================================================
        # HOME TORQUE BOOTSTRAP
        # ====================================================

        if (
            step_index
            <
            bootstrap_steps
        ):

            last_tau = (
                home_torque.copy()
            )

        # ====================================================
        # HQP UPDATE
        # ====================================================

        elif (
            (
                step_index
                -
                bootstrap_steps
            )
            %
            control_decimation
            ==
            0
        ):

            terms = (
                dynamics.compute(
                    data,
                    forward=False,
                )
            )

            # ------------------------------------------------
            # CoM Jdot*v
            # ------------------------------------------------

            com_jdot_v = (
                compute_com_jdot_v(
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
            )

            # ------------------------------------------------
            # Rank 2:
            # CoM height acceleration
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Rank 4:
            # posture acceleration
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Solve HQP
            # ------------------------------------------------

            hqp_start = (
                time.perf_counter()
            )

            solution = (
                controller.solve_double_support(

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

                    com_jdot_v_z=(
                        com_jdot_v[
                            2
                        ]
                    ),

                    desired_com_acceleration_z=(
                        desired_com_acceleration_z
                    ),

                    desired_posture_acceleration=(
                        desired_posture_acceleration
                    ),

                    left_wrench_reference=(
                        left_wrench_reference
                    ),

                    right_wrench_reference=(
                        right_wrench_reference
                    ),

                    left_support_bounds=(
                        left_support_bounds
                    ),

                    right_support_bounds=(
                        right_support_bounds
                    ),

                    left_contact_height=(
                        left_contact_height
                    ),

                    right_contact_height=(
                        right_contact_height
                    ),

                    torque_lower=(
                        torque_lower
                    ),

                    torque_upper=(
                        torque_upper
                    ),
                )
            )

            hqp_elapsed = (
                time.perf_counter()
                -
                hqp_start
            )

            total_hqp_times.append(
                hqp_elapsed
            )

            if not solution.rank5_used:

                rank5_fallback_count += 1

            last_solution = (
                solution
            )

            last_tau = (
                solution.torque.copy()
            )

        # ====================================================
        # APPLY TORQUE
        # ====================================================

        data.ctrl[:] = (
            last_tau
        )

        # ====================================================
        # TORQUE UTILIZATION
        # ====================================================

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

        # ====================================================
        # PRINT STATUS
        # ====================================================

        if (
            current_time
            >=
            next_print_time
            -
            0.5
            *
            dt
        ):

            if last_solution is None:

                fz_left = 0.0
                fz_right = 0.0

                rank2 = 0.0
                rank4 = 0.0
                rank5 = 0.0

                rank5_mode = "HOME"

                hqp_ms = 0.0

                cop_left = np.array(
                    [
                        np.nan,
                        np.nan,
                    ]
                )

                cop_right = (
                    cop_left.copy()
                )

            else:

                fz_left = float(
                    last_solution.left_wrench[
                        2
                    ]
                )

                fz_right = float(
                    last_solution.right_wrench[
                        2
                    ]
                )

                rank2 = (
                    last_solution.rank2_residual
                )

                rank4 = (
                    last_solution.rank4_residual
                )

                rank5 = (
                    last_solution.rank5_residual
                )

                rank5_mode = (
                    "ON"
                    if
                    last_solution.rank5_used
                    else
                    "FALLBACK"
                )

                hqp_ms = (
                    1000.0
                    *
                    (
                        last_solution.solve_time_rank2
                        +
                        last_solution.solve_time_rank4
                        +
                        last_solution.solve_time_rank5
                    )
                )

                cop_left = (
                    compute_cop_from_wrench(
                        last_solution.left_wrench,
                        left_contact_height,
                    )
                )

                cop_right = (
                    compute_cop_from_wrench(
                        last_solution.right_wrench,
                        right_contact_height,
                    )
                )

            print(
                f"t={current_time:5.2f} s"

                f" | zerr="
                f"{1000.0 * abs(z_error):7.3f} mm"

                f" | tilt="
                f"{tilt:6.3f} deg"

                f" | slipL="
                f"{1000.0 * left_slip:7.3f} mm"

                f" | slipR="
                f"{1000.0 * right_slip:7.3f} mm"

                f" | c="
                f"{int(left_contact)}/"
                f"{int(right_contact)}"

                f" | tau="
                f"{100.0 * torque_utilization:6.2f}%"

                f" | Fz="
                f"{fz_left:6.2f}/"
                f"{fz_right:6.2f}"

                f" | CoPL="
                f"({1000.0 * cop_left[0]:6.1f},"
                f"{1000.0 * cop_left[1]:6.1f})mm"

                f" | CoPR="
                f"({1000.0 * cop_right[0]:6.1f},"
                f"{1000.0 * cop_right[1]:6.1f})mm"

                f" | R2="
                f"{rank2:.2e}"

                f" | R4="
                f"{rank4:.2e}"

                f" | R5="
                f"{rank5:.2e}"

                f" | L5="
                f"{rank5_mode}"

                f" | HQP="
                f"{hqp_ms:6.3f} ms"
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

        # ====================================================
        # STEP 2
        # ====================================================

        mujoco.mj_step2(
            model,
            data,
        )

        step_index += 1

    # ========================================================
    # FINAL FRESH STATE
    # ========================================================

    mujoco.mj_step1(
        model,
        data,
    )

    final_com = (
        dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )
    )

    final_left = (
        dynamics.get_site_kinematics(
            data,
            LEFT_FOOT_SITE,
        )
    )

    final_right = (
        dynamics.get_site_kinematics(
            data,
            RIGHT_FOOT_SITE,
        )
    )

    final_tilt = (
        get_base_tilt_deg(
            data,
            base_body_id,
        )
    )

    final_left_contact = (
        has_geom_contact(
            data,
            left_geom_id,
            floor_geom_id,
        )
    )

    final_right_contact = (
        has_geom_contact(
            data,
            right_geom_id,
            floor_geom_id,
        )
    )

    double_contact_ratio = (
        both_contact_samples
        /
        max(
            total_samples,
            1,
        )
    )

    if total_hqp_times:

        total_hqp_times = np.asarray(
            total_hqp_times,
            dtype=float,
        )

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

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    separator()

    print(
        "DOUBLE-SUPPORT HQP RESULT"
    )

    separator()

    print()

    print(
        f"max CoM-height error   = "
        f"{1000.0 * max_z_error:.3f} mm"
    )

    print(
        f"max base tilt          = "
        f"{max_tilt:.3f} deg"
    )

    print(
        f"max left-foot slip     = "
        f"{1000.0 * max_left_slip:.3f} mm"
    )

    print(
        f"max right-foot slip    = "
        f"{1000.0 * max_right_slip:.3f} mm"
    )

    print(
        f"double-contact ratio   = "
        f"{100.0 * double_contact_ratio:.2f}%"
    )

    print(
        f"max torque utilization = "
        f"{100.0 * max_tau_utilization:.2f}%"
    )

    print()

    print(
        f"mean total HQP time    = "
        f"{mean_hqp_ms:.3f} ms"
    )

    print(
        f"max total HQP time     = "
        f"{max_hqp_ms:.3f} ms"
    )

    print(
        f"Rank-5 fallbacks       = "
        f"{rank5_fallback_count}"
    )

    print()

    print(
        f"final CoM height       = "
        f"{final_com.position[2]:.8f} m"
    )

    print(
        f"final base tilt        = "
        f"{final_tilt:.3f} deg"
    )

    print(
        f"final left contact     = "
        f"{final_left_contact}"
    )

    print(
        f"final right contact    = "
        f"{final_right_contact}"
    )

    print(
        f"final left foot        = "
        f"{final_left.position}"
    )

    print(
        f"final right foot       = "
        f"{final_right.position}"
    )

    if last_solution is not None:

        print()

        print(
            "Final left wrench:"
        )

        print(
            last_solution.left_wrench
        )

        print(
            "Final right wrench:"
        )

        print(
            last_solution.right_wrench
        )

        print()

        print(
            f"Rank-2 residual = "
            f"{last_solution.rank2_residual:.6e}"
        )

        print(
            f"Rank-4 residual = "
            f"{last_solution.rank4_residual:.6e}"
        )

        print(
            f"Rank-5 residual = "
            f"{last_solution.rank5_residual:.6e}"
        )

        print(
            f"Rank-5 used     = "
            f"{last_solution.rank5_used}"
        )

        print(
            f"max HQP constraint violation = "
            f"{last_solution.max_constraint_violation:.6e}"
        )


# ============================================================
# STAGE 1 — NOMINAL STEP PLANNER VALIDATION
# ============================================================

def run_stage1_planner_validation():

    # ========================================================
    # CREATE PLANNER PARAMETERS
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

    # ========================================================
    # CREATE PLANNER
    # ========================================================

    planner = AdaptiveStepPlanner(
        planner_parameters
    )

    # ========================================================
    # COMPUTE NOMINAL STEP — LEFT STANCE
    # ========================================================

    left_step = planner.compute_nominal_step(

        desired_velocity_x=(
            DESIRED_VELOCITY_X
        ),

        desired_velocity_y=(
            DESIRED_VELOCITY_Y
        ),

        stance_leg=(
            StanceLeg.LEFT
        ),
    )

    # ========================================================
    # COMPUTE NOMINAL STEP — RIGHT STANCE
    # ========================================================

    right_step = planner.compute_nominal_step(

        desired_velocity_x=(
            DESIRED_VELOCITY_X
        ),

        desired_velocity_y=(
            DESIRED_VELOCITY_Y
        ),

        stance_leg=(
            StanceLeg.RIGHT
        ),
    )

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    separator()

    print(
        "STAGE 1 — NOMINAL STEP PLANNER"
    )

    separator()

    print()

    print(
        f"desired vx        = "
        f"{DESIRED_VELOCITY_X:+.6f} m/s"
    )

    print(
        f"desired vy        = "
        f"{DESIRED_VELOCITY_Y:+.6f} m/s"
    )

    print(
        f"CoM height        = "
        f"{COM_HEIGHT_REFERENCE:.6f} m"
    )

    print(
        f"omega0            = "
        f"{planner.omega:.6f} 1/s"
    )

    print(
        f"default step width = "
        f"{DEFAULT_STEP_WIDTH:.6f} m"
    )

    print()

    print(
        f"L bounds = "
        f"[{STEP_LENGTH_MIN:+.4f}, "
        f"{STEP_LENGTH_MAX:+.4f}] m"
    )

    print(
        f"W bounds = "
        f"[{STEP_WIDTH_MIN:+.4f}, "
        f"{STEP_WIDTH_MAX:+.4f}] m"
    )

    print(
        f"T bounds = "
        f"[{STEP_TIME_MIN:.4f}, "
        f"{STEP_TIME_MAX:.4f}] s"
    )

    # ========================================================
    # LEFT STANCE
    # ========================================================

    print()

    separator()

    print(
        "CASE 1: LEFT STANCE"
    )

    separator()

    print()

    print(
        f"B_l       = "
        f"{left_step.lower_time_bound:.6f} s"
    )

    print(
        f"B_u       = "
        f"{left_step.upper_time_bound:.6f} s"
    )

    print(
        f"T_nom     = "
        f"{left_step.step_time:.6f} s"
    )

    print(
        f"L_nom     = "
        f"{left_step.step_length:+.6f} m"
    )

    print(
        f"W_nom     = "
        f"{left_step.step_width_deviation:+.6f} m"
    )

    print(
        f"dx_nom    = "
        f"{left_step.step_displacement_x:+.6f} m"
    )

    print(
        f"dy_nom    = "
        f"{left_step.step_displacement_y:+.6f} m"
    )

    print(
        f"tau_nom   = "
        f"{left_step.tau:.6f}"
    )

    print(
        f"bx_nom    = "
        f"{left_step.dcm_offset_x:+.6f} m"
    )

    print(
        f"by_nom    = "
        f"{left_step.dcm_offset_y:+.6f} m"
    )

    # ========================================================
    # RIGHT STANCE
    # ========================================================

    print()

    separator()

    print(
        "CASE 2: RIGHT STANCE"
    )

    separator()

    print()

    print(
        f"B_l       = "
        f"{right_step.lower_time_bound:.6f} s"
    )

    print(
        f"B_u       = "
        f"{right_step.upper_time_bound:.6f} s"
    )

    print(
        f"T_nom     = "
        f"{right_step.step_time:.6f} s"
    )

    print(
        f"L_nom     = "
        f"{right_step.step_length:+.6f} m"
    )

    print(
        f"W_nom     = "
        f"{right_step.step_width_deviation:+.6f} m"
    )

    print(
        f"dx_nom    = "
        f"{right_step.step_displacement_x:+.6f} m"
    )

    print(
        f"dy_nom    = "
        f"{right_step.step_displacement_y:+.6f} m"
    )

    print(
        f"tau_nom   = "
        f"{right_step.tau:.6f}"
    )

    print(
        f"bx_nom    = "
        f"{right_step.dcm_offset_x:+.6f} m"
    )

    print(
        f"by_nom    = "
        f"{right_step.dcm_offset_y:+.6f} m"
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    tolerance = 1.0e-9

    # --------------------------------------------------------
    # Straight walking:
    #
    # vy = 0 -> W_nom = 0
    # --------------------------------------------------------

    if abs(
        DESIRED_VELOCITY_Y
    ) < tolerance:

        if not np.isclose(
            left_step.step_width_deviation,
            0.0,
            atol=tolerance,
        ):

            raise RuntimeError(
                "LEFT stance: "
                "W_nom must be zero for vy = 0."
            )

        if not np.isclose(
            right_step.step_width_deviation,
            0.0,
            atol=tolerance,
        ):

            raise RuntimeError(
                "RIGHT stance: "
                "W_nom must be zero for vy = 0."
            )

    # --------------------------------------------------------
    # L_nom = vx * T_nom
    # --------------------------------------------------------

    if not np.isclose(

        left_step.step_length,

        DESIRED_VELOCITY_X
        *
        left_step.step_time,

        atol=tolerance,
    ):

        raise RuntimeError(
            "LEFT stance: L_nom != vx * T_nom."
        )

    if not np.isclose(

        right_step.step_length,

        DESIRED_VELOCITY_X
        *
        right_step.step_time,

        atol=tolerance,
    ):

        raise RuntimeError(
            "RIGHT stance: L_nom != vx * T_nom."
        )

    # --------------------------------------------------------
    # For vy = 0:
    #
    # left stance  -> right foot -> delta_y = -l_p
    # right stance -> left foot  -> delta_y = +l_p
    # --------------------------------------------------------

    if abs(
        DESIRED_VELOCITY_Y
    ) < tolerance:

        if not np.isclose(
            left_step.step_displacement_y,
            -DEFAULT_STEP_WIDTH,
            atol=tolerance,
        ):

            raise RuntimeError(
                "LEFT stance lateral displacement "
                "has incorrect sign."
            )

        if not np.isclose(
            right_step.step_displacement_y,
            +DEFAULT_STEP_WIDTH,
            atol=tolerance,
        ):

            raise RuntimeError(
                "RIGHT stance lateral displacement "
                "has incorrect sign."
            )

        # ----------------------------------------------------
        # Lateral DCM offset must be antisymmetric
        # ----------------------------------------------------

        if not np.isclose(
            left_step.dcm_offset_y,
            -right_step.dcm_offset_y,
            atol=tolerance,
        ):

            raise RuntimeError(
                "Lateral nominal DCM offsets "
                "are not symmetric."
            )

    # --------------------------------------------------------
    # Sagittal terms must be identical for the two stance legs
    # --------------------------------------------------------

    if not np.isclose(
        left_step.step_time,
        right_step.step_time,
        atol=tolerance,
    ):

        raise RuntimeError(
            "LEFT/RIGHT T_nom mismatch."
        )

    if not np.isclose(
        left_step.step_length,
        right_step.step_length,
        atol=tolerance,
    ):

        raise RuntimeError(
            "LEFT/RIGHT L_nom mismatch."
        )

    if not np.isclose(
        left_step.dcm_offset_x,
        right_step.dcm_offset_x,
        atol=tolerance,
    ):

        raise RuntimeError(
            "LEFT/RIGHT bx_nom mismatch."
        )

    # ========================================================
    # RESULT
    # ========================================================

    print()

    separator()

    print(
        "STAGE 1 VALIDATION: PASSED"
    )

    separator()

    print()

    return (
        planner,
        left_step,
        right_step,
    )


# ============================================================
# STAGE 2 — ADAPTIVE STEP QP VALIDATION
# ============================================================

def run_stage2_planner_validation(
    planner,
    nominal_left_step,
    nominal_right_step,
):

    separator()

    print(
        "STAGE 2 — ADAPTIVE STEP QP"
    )

    separator()

    print()

    stance_position = np.array(
        [
            0.0,
            0.0,
        ],
        dtype=float,
    )

    elapsed_time = float(
        STAGE2_TEST_ELAPSED_TIME
    )

    for nominal_step in (
        nominal_left_step,
        nominal_right_step,
    ):

        uT_nominal = (
            stance_position
            +
            np.array(
                [
                    nominal_step.step_displacement_x,
                    nominal_step.step_displacement_y,
                ],
                dtype=float,
            )
        )

        b_nominal = np.array(
            [
                nominal_step.dcm_offset_x,
                nominal_step.dcm_offset_y,
            ],
            dtype=float,
        )

        # Synthetic DCM state exactly on the nominal LIPM
        # trajectory. Rearranged from Eq. (19):
        #
        # xi(t) = u0 + (uT - u0 + b) exp[-omega (T - t)]
        xi_measured = (
            stance_position
            +
            (
                uT_nominal
                -
                stance_position
                +
                b_nominal
            )
            *
            np.exp(
                -planner.omega
                *
                (
                    nominal_step.step_time
                    -
                    elapsed_time
                )
            )
        )

        adapted_step = (
            planner.solve_adaptive_step(

                nominal_step=(
                    nominal_step
                ),

                dcm_measured=(
                    xi_measured
                ),

                stance_position=(
                    stance_position
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
            )
        )

        separator()

        print(
            f"{nominal_step.stance_leg.value.upper()} STANCE"
        )

        separator()

        print()

        print(
            f"elapsed time = "
            f"{elapsed_time:.6f} s"
        )

        print(
            f"xi_measured = "
            f"[{xi_measured[0]:+.6f}, "
            f"{xi_measured[1]:+.6f}] m"
        )

        print()

        print(
            f"nominal uT  = "
            f"[{uT_nominal[0]:+.6f}, "
            f"{uT_nominal[1]:+.6f}] m"
        )

        print(
            f"adapted uT  = "
            f"[{adapted_step.step_location_x:+.6f}, "
            f"{adapted_step.step_location_y:+.6f}] m"
        )

        print()

        print(
            f"nominal T   = "
            f"{nominal_step.step_time:.6f} s"
        )

        print(
            f"adapted T   = "
            f"{adapted_step.step_time:.6f} s"
        )

        print()

        print(
            f"nominal tau = "
            f"{nominal_step.tau:.6f}"
        )

        print(
            f"adapted tau = "
            f"{adapted_step.tau:.6f}"
        )

        print()

        print(
            f"nominal b   = "
            f"[{b_nominal[0]:+.6f}, "
            f"{b_nominal[1]:+.6f}] m"
        )

        print(
            f"adapted b   = "
            f"[{adapted_step.dcm_offset_x:+.6f}, "
            f"{adapted_step.dcm_offset_y:+.6f}] m"
        )

        print()

        print(
            f"objective   = "
            f"{adapted_step.objective:.6e}"
        )

        print(
            f"eq residual = "
            f"{adapted_step.max_equality_residual:.6e}"
        )

        tolerance = 1.0e-7

        if not np.allclose(
            np.array(
                [
                    adapted_step.step_location_x,
                    adapted_step.step_location_y,
                ],
                dtype=float,
            ),
            uT_nominal,
            atol=tolerance,
        ):

            raise RuntimeError(
                "Stage-2 failed to recover nominal foot location."
            )

        if not np.isclose(
            adapted_step.tau,
            nominal_step.tau,
            atol=tolerance,
        ):

            raise RuntimeError(
                "Stage-2 failed to recover nominal tau."
            )

        if not np.isclose(
            adapted_step.step_time,
            nominal_step.step_time,
            atol=tolerance,
        ):

            raise RuntimeError(
                "Stage-2 failed to recover nominal step time."
            )

        if not np.allclose(
            np.array(
                [
                    adapted_step.dcm_offset_x,
                    adapted_step.dcm_offset_y,
                ],
                dtype=float,
            ),
            b_nominal,
            atol=tolerance,
        ):

            raise RuntimeError(
                "Stage-2 failed to recover nominal DCM offset."
            )

        if (
            adapted_step.max_equality_residual
            >
            tolerance
        ):

            raise RuntimeError(
                "Stage-2 Eq. (19) residual is too large."
            )

        print()

        print(
            "RESULT: PASSED"
        )

        print()

    separator()

    print(
        "STAGE 2 VALIDATION: PASSED"
    )

    separator()

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # STAGE 1 — ADAPTIVE STEP PLANNER VALIDATION
    # ========================================================

    (
        planner,
        nominal_left_step,
        nominal_right_step,
    ) = run_stage1_planner_validation()

    if RUN_STAGE1_PLANNER_ONLY:

        return

    # ========================================================
    # STAGE 2 — ADAPTIVE STEP QP VALIDATION
    # ========================================================

    run_stage2_planner_validation(
        planner=(
            planner
        ),
        nominal_left_step=(
            nominal_left_step
        ),
        nominal_right_step=(
            nominal_right_step
        ),
    )

    if RUN_STAGE2_PLANNER_ONLY:

        return


    separator()

    print(
        "STEP TIMING ADAPTATION"
    )

    print(
        "HIERARCHICAL INVERSE DYNAMICS"
    )

    print(
        "DOUBLE-SUPPORT TEST"
    )

    separator()

    print()

    print(
        f"MuJoCo = "
        f"{mujoco.__version__}"
    )

    print(
        f"CasADi = "
        f"{ca.__version__}"
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
        "HQP ARCHITECTURE"
    )

    separator()

    print()

    print(
        "Decision vector:"
    )

    print(
        "  y = [qddot(20), "
        "lambda_L(6), lambda_R(6)]"
    )

    print(
        "  total = 32 variables"
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
        "  unilateral contact"
    )

    print(
        "  friction feasibility"
    )

    print(
        "  CoP inside foot support polygon"
    )

    print()

    print(
        "Rank 2:"
    )

    print(
        "  left stance 6D"
    )

    print(
        "  right stance 6D"
    )

    print(
        "  CoM height only"
    )

    print()

    print(
        "Rank 3:"
    )

    print(
        "  none in double support"
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
        "  wrench regularization"
    )

    print(
        "  fallback to Rank 4 if Rank 5 "
        "is numerically singular"
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
    # CONTROLLER
    # ========================================================

    config = DoubleSupportHQPConfig(

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
                config
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

            run_double_support(

                model=model,

                data=data,

                dynamics=dynamics,

                controller=controller,

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

        run_double_support(

            model=model,

            data=data,

            dynamics=dynamics,

            controller=controller,

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