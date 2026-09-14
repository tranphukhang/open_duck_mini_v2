# step_timing_adaptation/run.py

from __future__ import annotations

import sys
import math
import time
from pathlib import Path

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

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


ROBOT_XML = (
    ROOT_DIR
    /
    "xmls"
    /
    "open_duck_mini_v2.xml"
)

SCENE_XML = (
    ROOT_DIR
    /
    "xmls"
    /
    "scene_flat_terrain.xml"
)


# ============================================================
# STEP TIMING ADAPTATION
# ============================================================

if __package__:

    from .adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from .swing_trajectory import (
        OnlineSwingFootTrajectory,
        VerticalSwingQPParameters,
    )

    from .adaptive_support_preview import (
        INITIAL_DOUBLE_SUPPORT,
        SINGLE_SUPPORT,
        build_adaptive_support_preview,
    )

else:

    from adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from swing_trajectory import (
        OnlineSwingFootTrajectory,
        VerticalSwingQPParameters,
    )

    from adaptive_support_preview import (
        INITIAL_DOUBLE_SUPPORT,
        SINGLE_SUPPORT,
        build_adaptive_support_preview,
    )


# ============================================================
# REUSE EXISTING LIPM-MPC
# ============================================================

from lipm_mpc.lipm_model import (
    LIPMModel1D,
)

from lipm_mpc.mpc_1d import (
    LIPMMPC1D,
)

from lipm_mpc.com_trajectory import (
    ConstantJerkCoMSegment,
)


# ============================================================
# REUSE EXISTING WHOLE-BODY KINEMATICS
# ============================================================

from footstep_planning.pinocchio_model import (
    PinocchioModel,
)

from footstep_planning.differential_ik import (
    TRUNK_FRAME,
    solve_single_support_ik,
    solve_double_support_ik,
)


# ============================================================
# NUMPY
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# TEST
#
# Current scope:
#
#       INITIAL DS
#           |
#           v
#       LEFT SUPPORT
#       RIGHT SWING
#           |
#           v
#       TOUCHDOWN
#
# One adaptive step only.
# ============================================================

STANCE_LEG = (
    StanceLeg.LEFT
)

STANCE_SIDE = (
    "left"
)

SWING_SIDE = (
    "right"
)


# ============================================================
# EXECUTOR
#
# Pure kinematic execution.
#
# NO mj_step().
# NO rigid-body dynamics.
# ============================================================

DT = 0.0005
# 2000 Hz differential IK / set-state


# ============================================================
# INITIAL DOUBLE SUPPORT
#
# Same purpose as lipm_mpc:
# allow MPC to move the CoM toward the future support foot
# before single support begins.
#
# This phase is still purely kinematic.
# ============================================================

INITIAL_DOUBLE_SUPPORT_DURATION = (
    0.36
)


# ============================================================
# LIPM / PLANNER
# ============================================================

GRAVITY = 9.81

COM_HEIGHT = 0.2044

DESIRED_VELOCITY_X = 0.05

DESIRED_VELOCITY_Y = 0.0

DEFAULT_STEP_WIDTH = 0.16


# ============================================================
# STEP BOUNDS
# ============================================================

STEP_LENGTH_MIN = -0.10
STEP_LENGTH_MAX = +0.10

STEP_WIDTH_MIN = -0.03
STEP_WIDTH_MAX = +0.03

STEP_TIME_MIN = 0.20
STEP_TIME_MAX = 0.30


# ============================================================
# STEP QP
# ============================================================

STEP_QP_ALPHA_LOCATION = 1.0

STEP_QP_ALPHA_TIMING = 5.0

STEP_QP_ALPHA_DCM = 1000.0

STEP_QP_ALPHA_VIABILITY = 1.0e6

STEP_TIMING_GAP = 0.02


# ============================================================
# PLANNER RATE
# ============================================================

PLANNER_UPDATE_FREQUENCY = 1.0 / DT

PLANNER_UPDATE_PERIOD = DT


# ============================================================
# SWING TRAJECTORY
# ============================================================

SWING_HEIGHT_DESIRED = 0.03

SWING_HEIGHT_MAX = 0.04

SWING_VERTICAL_CONSTRAINT_SAMPLES = (
    41
)

SWING_VERTICAL_COEFFICIENT_REGULARIZATION = (
    1.0e-6
)

SWING_VERTICAL_BOUND_TOLERANCE = (
    1.0e-9
)

SWING_VERTICAL_MAX_REFINEMENTS = (
    8
)


# ============================================================
# LIPM-MPC
#
# Reused from lipm_mpc implementation.
# ============================================================

MPC_TIMESTEP = 0.03

MPC_HORIZON_STEPS = 48

TERMINAL_POSITION_WEIGHT = 1.0

TERMINAL_VELOCITY_WEIGHT = 1.0

TERMINAL_ACCELERATION_WEIGHT = 1.0

CONTROL_WEIGHT = 2.0e-5

MPC_SOLVER_OPTIONS = {
    "ftol": 1.0e-10,
    "maxiter": 1000,
}


# ============================================================
# SUPPORT GEOMETRY
#
# Same values as lipm_mpc/run.py
# ============================================================

ZMP_SUPPORT_SCALE = 0.9

FOOT_TOE = 0.0645

FOOT_HEEL = 0.0386

FOOT_HALF_WIDTH = 0.02065


# ============================================================
# SINGLE-SUPPORT ZMP
#
# Step Timing Adaptation assumes:
#
#     xi_dot = omega * (xi - u0)
#
# where u0 is fixed during single support.
#
# Therefore LIPM-MPC is constrained to keep its ZMP
# very close to the stance-foot reference point u0.
#
# Exact equality cannot be used because LIPMMPC1D requires:
#
#     lower_bound < upper_bound
#
# Hence:
#
#     u0 - epsilon <= ZMP <= u0 + epsilon
#
# 0.5 mm is sufficiently small for the reduced-order
# validation while remaining numerically well defined.
# ============================================================

SINGLE_SUPPORT_ZMP_HALF_WIDTH = 0.0005


# ============================================================
# DIFFERENTIAL IK
#
# Same structure as lipm_mpc.
# ============================================================

IK_DAMPING = 1.0e-8

IK_RCOND = 1.0e-10

SUPPORT_POSITION_KP = 25.0

SWING_POSITION_KP = 20.0

COM_POSITION_KP = 10.0

TRUNK_ORIENTATION_KP = 10.0


# ============================================================
# DISTURBANCE
#
# Applied to the LIPM state.
#
# A desired DCM jump:
#
#       Delta xi
#
# is produced by:
#
#       Delta v = omega * Delta xi
#
# while keeping CoM position unchanged.
#
# Therefore the disturbance affects both:
#
#       planner
#       MPC
#
# rather than only faking a planner measurement.
# ============================================================

ENABLE_DISTURBANCE = False

DISTURBANCE_TIME = 0.10
# time from beginning of SINGLE SUPPORT

DISTURBANCE_DCM_X = +0.010
DISTURBANCE_DCM_Y = 0.000


# ============================================================
# VIEWER
# ============================================================

SHOW_VIEWER = True

REALTIME_PLAYBACK = True

VIEWER_SYNC_PERIOD = 0.02


# ============================================================
# PRINT
# ============================================================

STATUS_PRINT_PERIOD = 0.05


# ============================================================
# NUMERICAL
# ============================================================

TIME_TOLERANCE = 1.0e-10


# ============================================================
# MPC / IK RATIO
# ============================================================

MPC_IK_RATIO = (
    MPC_TIMESTEP
    /
    DT
)

MPC_IK_STEPS = int(
    round(
        MPC_IK_RATIO
    )
)

if not np.isclose(
    MPC_IK_RATIO,
    MPC_IK_STEPS,
    atol=1.0e-12,
):

    raise RuntimeError(
        "MPC_TIMESTEP must be an integer multiple of DT."
    )


# ============================================================
# HELPERS
# ============================================================

def separator():

    print(
        "=" * 100
    )


def update_mujoco_from_pinocchio(
    robot,
    q_pin,
    mj_model,
    mj_data,
):

    q_mj = (
        robot.pin_to_mujoco(
            q_pin
        )
    )

    mj_data.qpos[:] = (
        q_mj
    )

    mj_data.qvel[:] = (
        0.0
    )

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


# ============================================================
# MPC
# ============================================================

def create_axis_mpc():

    model = (
        LIPMModel1D(
            timestep=(
                MPC_TIMESTEP
            ),
            com_height=(
                COM_HEIGHT
            ),
            gravity=(
                GRAVITY
            ),
        )
    )

    terminal_weights = np.array(
        [
            TERMINAL_POSITION_WEIGHT,
            TERMINAL_VELOCITY_WEIGHT,
            TERMINAL_ACCELERATION_WEIGHT,
        ],
        dtype=float,
    )

    controller = (
        LIPMMPC1D(
            model=(
                model
            ),
            horizon_steps=(
                MPC_HORIZON_STEPS
            ),
            terminal_weights=(
                terminal_weights
            ),
            control_weight=(
                CONTROL_WEIGHT
            ),
        )
    )

    return (
        model,
        controller,
    )


def shift_control_sequence(
    control,
):

    if control is None:

        return None

    control = np.asarray(
        control,
        dtype=float,
    )

    shifted = np.empty_like(
        control
    )

    shifted[:-1] = (
        control[1:]
    )

    shifted[-1] = (
        control[-1]
    )

    return shifted


# ============================================================
# DCM
# ============================================================

def compute_dcm_from_lipm(
    x_state,
    y_state,
    omega,
):

    return np.array(
        [
            x_state[0]
            +
            x_state[1]
            /
            omega,

            y_state[0]
            +
            y_state[1]
            /
            omega,
        ],
        dtype=float,
    )


# ============================================================
# CURRENT CONTINUOUS LIPM STATE
# ============================================================

def get_current_lipm_state(
    x_state,
    y_state,
    segment,
    mpc_substep,
):

    if segment is None:

        return (
            x_state.copy(),
            y_state.copy(),
        )

    tau = min(
        mpc_substep
        *
        DT,
        MPC_TIMESTEP,
    )

    return (
        segment.get_x_state(
            tau
        ),
        segment.get_y_state(
            tau
        ),
    )


# ============================================================
# MPC SOLVE
# ============================================================

def solve_mpc_segment(
    *,
    current_phase,
    phase_time,

    x_state,
    y_state,

    x_mpc,
    y_mpc,

    previous_x_control,
    previous_y_control,

    left_initial_position,
    right_initial_position,

    left_rotation,
    right_rotation,

    landing_position,
    single_support_duration,
):

    preview = (
        build_adaptive_support_preview(
            current_phase=(
                current_phase
            ),
            phase_time=(
                phase_time
            ),

            initial_double_support_duration=(
                INITIAL_DOUBLE_SUPPORT_DURATION
            ),

            single_support_duration=(
                single_support_duration
            ),

            stance_side=(
                STANCE_SIDE
            ),

            left_initial_position=(
                left_initial_position
            ),

            right_initial_position=(
                right_initial_position
            ),

            landing_position=(
                landing_position
            ),

            left_rotation=(
                left_rotation
            ),

            right_rotation=(
                right_rotation
            ),

            timestep=(
                MPC_TIMESTEP
            ),

            horizon_steps=(
                MPC_HORIZON_STEPS
            ),

            foot_toe=(
                FOOT_TOE
            ),

            foot_heel=(
                FOOT_HEEL
            ),

            foot_half_width=(
                FOOT_HALF_WIDTH
            ),

            zmp_scale=(
                ZMP_SUPPORT_SCALE
            ),

            single_support_zmp_half_width=(
                SINGLE_SUPPORT_ZMP_HALF_WIDTH
            ),
        )
    )

    # --------------------------------------------------------
    # Terminal goal = center of final preview support region.
    # --------------------------------------------------------

    x_goal = np.array(
        [
            0.5
            *
            (
                preview.x_min[-1]
                +
                preview.x_max[-1]
            ),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    y_goal = np.array(
        [
            0.5
            *
            (
                preview.y_min[-1]
                +
                preview.y_max[-1]
            ),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    wall_start = (
        time.perf_counter()
    )

    x_result = (
        x_mpc.solve(
            current_state=(
                x_state
            ),
            goal_state=(
                x_goal
            ),
            lower_bounds=(
                preview.x_min
            ),
            upper_bounds=(
                preview.x_max
            ),
            initial_control=(
                shift_control_sequence(
                    previous_x_control
                )
            ),
            solver_options=(
                MPC_SOLVER_OPTIONS
            ),
        )
    )

    y_result = (
        y_mpc.solve(
            current_state=(
                y_state
            ),
            goal_state=(
                y_goal
            ),
            lower_bounds=(
                preview.y_min
            ),
            upper_bounds=(
                preview.y_max
            ),
            initial_control=(
                shift_control_sequence(
                    previous_y_control
                )
            ),
            solver_options=(
                MPC_SOLVER_OPTIONS
            ),
        )
    )

    solve_time = (
        time.perf_counter()
        -
        wall_start
    )

    if not x_result.success:

        raise RuntimeError(
            "X MPC failed: "
            f"{x_result.message}"
        )

    if not y_result.success:

        raise RuntimeError(
            "Y MPC failed: "
            f"{y_result.message}"
        )

    segment = (
        ConstantJerkCoMSegment(
            x_state=(
                x_state
            ),
            y_state=(
                y_state
            ),

            x_jerk=(
                x_result.first_control
            ),
            y_jerk=(
                y_result.first_control
            ),

            com_height=(
                COM_HEIGHT
            ),

            duration=(
                MPC_TIMESTEP
            ),
        )
    )

    return {
        "preview":
            preview,

        "x_result":
            x_result,

        "y_result":
            y_result,

        "segment":
            segment,

        "solve_time":
            solve_time,
    }


# ============================================================
# PLANNER
# ============================================================

def solve_step_planner(
    planner,
    nominal_step,
    dcm,
    stance_position_xy,
    elapsed_time,
):

    return (
        planner.solve_adaptive_step(
            nominal_step=(
                nominal_step
            ),

            dcm_measured=(
                dcm
            ),

            stance_position=(
                stance_position_xy
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
    )


# ============================================================
# EXECUTION
# ============================================================

def run_test(
    *,
    mj_model,
    mj_data,
    robot,
    planner,
    viewer=None,
):

    # ========================================================
    # INITIAL KINEMATIC STATE
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
        left_initial_position,
        left_rotation,
    ) = (
        robot.get_left_foot_pose()
    )

    (
        right_initial_position,
        right_rotation,
    ) = (
        robot.get_right_foot_pose()
    )

    initial_com = (
        robot.get_com()
    )

    # ========================================================
    # COM WORLD-Z REFERENCE
    #
    # COM_HEIGHT is the LIPM height measured relative to the
    # support plane:
    #
    #     h = z_CoM - z_support
    #
    # It is NOT an absolute world-z coordinate.
    #
    # Current validation:
    #
    #     LEFT stance -> RIGHT swing
    #
    # Therefore the left-foot site height is used as the fixed
    # support-plane reference.
    # ========================================================

    com_world_z_ref = (
        left_initial_position[2]
        +
        COM_HEIGHT
    )

    initial_com_height_above_support = (
        initial_com[2]
        -
        left_initial_position[2]
    )

    (
        _,
        trunk_rotation_ref,
    ) = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    stance_position_xy = (
        left_initial_position[
            0:2
        ].copy()
    )

    # ========================================================
    # NOMINAL STEP
    # ========================================================

    nominal_step = (
        planner.compute_nominal_step(
            stance_leg=(
                STANCE_LEG
            ),

            desired_velocity_x=(
                DESIRED_VELOCITY_X
            ),

            desired_velocity_y=(
                DESIRED_VELOCITY_Y
            ),
        )
    )

    nominal_landing_position = (
        right_initial_position.copy()
    )

    nominal_landing_position[0] = (
        stance_position_xy[0]
        +
        nominal_step.step_displacement_x
    )

    nominal_landing_position[1] = (
        stance_position_xy[1]
        +
        nominal_step.step_displacement_y
    )

    nominal_single_support_duration = (
        nominal_step.step_time
    )

    # ========================================================
    # LIPM STATE
    # ========================================================

    x_state = np.array(
        [
            initial_com[0],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    y_state = np.array(
        [
            initial_com[1],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # ========================================================
    # MPC
    # ========================================================

    (
        _,
        x_mpc,
    ) = (
        create_axis_mpc()
    )

    (
        _,
        y_mpc,
    ) = (
        create_axis_mpc()
    )

    current_segment = None

    mpc_substep = 0

    previous_x_control = None
    previous_y_control = None

    mpc_solve_count = 0

    max_mpc_solve_time = 0.0

    # ========================================================
    # SWING TRAJECTORY
    # ========================================================

    swing_trajectory = (
        OnlineSwingFootTrajectory()
    )

    vertical_parameters = (
        VerticalSwingQPParameters(
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
    )

    swing_initialized = False

    # ========================================================
    # PHASE
    # ========================================================

    phase = (
        INITIAL_DOUBLE_SUPPORT
    )

    phase_time = 0.0

    kinematic_time = 0.0

    # Before SS starts, MPC preview uses nominal step.

    landing_position = (
        nominal_landing_position.copy()
    )

    current_step_time = (
        nominal_single_support_duration
    )

    planner_result = None

    planner_frozen = False

    freeze_time = None

    next_planner_update = 0.0

    disturbance_applied = False

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    max_com_error = 0.0

    max_support_error = 0.0

    max_swing_error = 0.0

    max_viability_slack_x = 0.0

    max_viability_slack_y = 0.0

    next_print_time = 0.0

    # ========================================================
    # INFO
    # ========================================================

    separator()

    print(
        "STEP TIMING ADAPTATION"
    )

    print(
        "LIPM-MPC + HIERARCHICAL DIFFERENTIAL IK"
    )

    print(
        "KINEMATIC SET-STATE VALIDATION"
    )

    separator()

    print()

    print(
        "Execution architecture:"
    )

    print(
        "  adaptive step planner"
    )

    print(
        "       -> adaptive support preview"
    )

    print(
        "       -> LIPM-MPC CoM"
    )

    print(
        "       -> swing trajectory"
    )

    print(
        "       -> hierarchical differential IK"
    )

    print(
        "       -> Pinocchio integrate"
    )

    print(
        "       -> MuJoCo qpos set-state"
    )

    print()

    print(
        "No rigid-body dynamics."
    )

    print(
        "No WBC."
    )

    print(
        "No mj_step()."
    )

    print()

    print(
        f"Initial LEFT  = "
        f"{left_initial_position}"
    )

    print(
        f"Initial RIGHT = "
        f"{right_initial_position}"
    )

    print(
        f"Initial CoM   = "
        f"{initial_com}"
    )

    print(
        f"Initial CoM height above support = "
        f"{initial_com_height_above_support:.6f} m"
    )

    print(
        f"LIPM CoM height                  = "
        f"{COM_HEIGHT:.6f} m"
    )

    print(
        f"CoM world-z reference            = "
        f"{com_world_z_ref:.6f} m"
    )

    print(
        f"Initial vertical mismatch        = "
        f"{1000.0 * (com_world_z_ref - initial_com[2]):+.3f} mm"
    )

    print()

    print(
        f"Nominal uT    = "
        f"{nominal_landing_position}"
    )

    print(
        f"Nominal T     = "
        f"{nominal_step.step_time:.6f} s"
    )

    print(
        f"omega         = "
        f"{planner.omega:.6f} rad/s"
    )

    print()

    print(
        "IK hierarchy:"
    )

    print(
        "  support > CoM > swing > trunk"
    )

    print()

    # ========================================================
    # REAL-TIME
    # ========================================================

    wall_start = (
        time.perf_counter()
    )

    iteration = 0

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

        # ====================================================
        # INITIAL DS -> SINGLE SUPPORT
        # ====================================================

        if (
            phase
            ==
            INITIAL_DOUBLE_SUPPORT
            and
            phase_time
            >=
            INITIAL_DOUBLE_SUPPORT_DURATION
            -
            TIME_TOLERANCE
        ):

            # Current MPC state at exact transition.

            (
                x_state,
                y_state,
            ) = (
                get_current_lipm_state(
                    x_state=(
                        x_state
                    ),
                    y_state=(
                        y_state
                    ),
                    segment=(
                        current_segment
                    ),
                    mpc_substep=(
                        mpc_substep
                    ),
                )
            )

            current_segment = None

            mpc_substep = 0

            previous_x_control = None
            previous_y_control = None

            phase = (
                SINGLE_SUPPORT
            )

            phase_time = 0.0

            # ------------------------------------------------
            # Current DCM from MPC CoM state.
            # ------------------------------------------------

            dcm = (
                compute_dcm_from_lipm(
                    x_state=(
                        x_state
                    ),
                    y_state=(
                        y_state
                    ),
                    omega=(
                        planner.omega
                    ),
                )
            )

            planner_result = (
                solve_step_planner(
                    planner=(
                        planner
                    ),
                    nominal_step=(
                        nominal_step
                    ),
                    dcm=(
                        dcm
                    ),
                    stance_position_xy=(
                        stance_position_xy
                    ),
                    elapsed_time=0.0,
                )
            )

            landing_position = (
                right_initial_position.copy()
            )

            landing_position[0] = (
                planner_result.step_location_x
            )

            landing_position[1] = (
                planner_result.step_location_y
            )

            current_step_time = (
                planner_result.step_time
            )

            # ------------------------------------------------
            # Initialize online swing trajectory at lift-off.
            # ------------------------------------------------

            robot.update(
                q_pin
            )

            (
                current_right_position,
                _,
            ) = (
                robot.get_right_foot_pose()
            )

            swing_trajectory.reset_3d(
                initial_position=(
                    current_right_position
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

            swing_initialized = True

            next_planner_update = (
                PLANNER_UPDATE_PERIOD
            )

            separator()

            print(
                "START ADAPTIVE SINGLE SUPPORT"
            )

            separator()

            print(
                f"DCM at lift-off = "
                f"{dcm}"
            )

            print(
                f"Initial adapted uT = "
                f"{landing_position}"
            )

            print(
                f"Initial adapted T  = "
                f"{current_step_time:.6f} s"
            )

            print()

        # ====================================================
        # CURRENT CONTINUOUS LIPM STATE
        # ====================================================

        (
            current_x_state,
            current_y_state,
        ) = (
            get_current_lipm_state(
                x_state=(
                    x_state
                ),
                y_state=(
                    y_state
                ),
                segment=(
                    current_segment
                ),
                mpc_substep=(
                    mpc_substep
                ),
            )
        )

        # ====================================================
        # DISTURBANCE
        # ====================================================

        if (
            phase
            ==
            SINGLE_SUPPORT
            and
            ENABLE_DISTURBANCE
            and
            not disturbance_applied
            and
            phase_time
            >=
            DISTURBANCE_TIME
            -
            TIME_TOLERANCE
        ):

            # DCM:
            #
            # xi = c + c_dot / omega
            #
            # For fixed position:
            #
            # Delta v = omega * Delta xi

            current_x_state[1] += (
                planner.omega
                *
                DISTURBANCE_DCM_X
            )

            current_y_state[1] += (
                planner.omega
                *
                DISTURBANCE_DCM_Y
            )

            x_state = (
                current_x_state.copy()
            )

            y_state = (
                current_y_state.copy()
            )

            current_segment = None

            mpc_substep = 0

            previous_x_control = None
            previous_y_control = None

            disturbance_applied = True

            print()

            separator()

            print(
                "DCM DISTURBANCE APPLIED"
            )

            separator()

            print(
                f"t_SS = "
                f"{phase_time:.6f} s"
            )

            print(
                f"Delta DCM = "
                f"({DISTURBANCE_DCM_X:+.6f}, "
                f"{DISTURBANCE_DCM_Y:+.6f}) m"
            )

            print()

            current_x_state = (
                x_state.copy()
            )

            current_y_state = (
                y_state.copy()
            )

        # ====================================================
        # CURRENT DCM
        # ====================================================

        dcm = (
            compute_dcm_from_lipm(
                x_state=(
                    current_x_state
                ),
                y_state=(
                    current_y_state
                ),
                omega=(
                    planner.omega
                ),
            )
        )

        # ====================================================
        # ONLINE STEP TIMING / LOCATION ADAPTATION
        # ====================================================

        if (
            phase
            ==
            SINGLE_SUPPORT
            and
            not planner_frozen
        ):

            # Freeze current target when there is no longer
            # enough time to adapt safely.

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

                freeze_time = (
                    phase_time
                )

            elif (
                phase_time
                >=
                next_planner_update
                -
                TIME_TOLERANCE
            ):

                try:

                    new_result = (
                        solve_step_planner(
                            planner=(
                                planner
                            ),

                            nominal_step=(
                                nominal_step
                            ),

                            dcm=(
                                dcm
                            ),

                            stance_position_xy=(
                                stance_position_xy
                            ),

                            elapsed_time=(
                                phase_time
                            ),
                        )
                    )

                    planner_result = (
                        new_result
                    )

                    landing_position[0] = (
                        planner_result.step_location_x
                    )

                    landing_position[1] = (
                        planner_result.step_location_y
                    )

                    current_step_time = (
                        planner_result.step_time
                    )

                    max_viability_slack_x = max(
                        max_viability_slack_x,
                        planner_result.viability_slack_x,
                    )

                    max_viability_slack_y = max(
                        max_viability_slack_y,
                        planner_result.viability_slack_y,
                    )

                except RuntimeError as error:

                    if (
                        "timing adaptation window is closed"
                        in
                        str(
                            error
                        ).lower()
                    ):

                        planner_frozen = True

                        freeze_time = (
                            phase_time
                        )

                    else:

                        raise

                while (
                    next_planner_update
                    <=
                    phase_time
                    +
                    TIME_TOLERANCE
                ):

                    next_planner_update += (
                        PLANNER_UPDATE_PERIOD
                    )

        # ====================================================
        # MPC
        # ====================================================

        if current_segment is None:

            # The segment begins from the exact current state.

            x_state = (
                current_x_state.copy()
            )

            y_state = (
                current_y_state.copy()
            )

            mpc_info = (
                solve_mpc_segment(
                    current_phase=(
                        phase
                    ),

                    phase_time=(
                        phase_time
                    ),

                    x_state=(
                        x_state
                    ),

                    y_state=(
                        y_state
                    ),

                    x_mpc=(
                        x_mpc
                    ),

                    y_mpc=(
                        y_mpc
                    ),

                    previous_x_control=(
                        previous_x_control
                    ),

                    previous_y_control=(
                        previous_y_control
                    ),

                    left_initial_position=(
                        left_initial_position
                    ),

                    right_initial_position=(
                        right_initial_position
                    ),

                    left_rotation=(
                        left_rotation
                    ),

                    right_rotation=(
                        right_rotation
                    ),

                    landing_position=(
                        landing_position
                    ),

                    single_support_duration=(
                        current_step_time
                    ),
                )
            )

            current_segment = (
                mpc_info[
                    "segment"
                ]
            )

            x_result = (
                mpc_info[
                    "x_result"
                ]
            )

            y_result = (
                mpc_info[
                    "y_result"
                ]
            )

            previous_x_control = (
                x_result
                .control
                .copy()
            )

            previous_y_control = (
                y_result
                .control
                .copy()
            )

            mpc_substep = 0

            mpc_solve_count += 1

            max_mpc_solve_time = max(
                max_mpc_solve_time,
                mpc_info[
                    "solve_time"
                ],
            )

        # ====================================================
        # CURRENT CONTINUOUS COM REFERENCE
        # ====================================================

        tau = (
            mpc_substep
            *
            DT
        )

        com_reference = (
            current_segment.evaluate(
                tau
            )
        )

        # ====================================================
        # CONVERT LIPM COM REFERENCE TO WORLD COORDINATES
        #
        # ConstantJerkCoMSegment stores:
        #
        #     position[2] = COM_HEIGHT
        #
        # where COM_HEIGHT is the LIPM relative height h.
        #
        # Differential IK, however, expects the actual CoM
        # position in the world frame.
        #
        # Therefore:
        #
        #     z_CoM^W = z_support^W + h
        #
        # Horizontal x/y and their velocities remain exactly
        # those generated by LIPM-MPC.
        # ====================================================

        com_position_ref = (
            com_reference.position.copy()
        )

        com_velocity_ref = (
            com_reference.velocity.copy()
        )

        com_position_ref[2] = (
            com_world_z_ref
        )

        com_velocity_ref[2] = (
            0.0
        )

        # ====================================================
        # FOOT REFERENCES
        # ====================================================

        if (
            phase
            ==
            INITIAL_DOUBLE_SUPPORT
        ):

            p_left_ref = (
                left_initial_position
                .copy()
            )

            p_right_ref = (
                right_initial_position
                .copy()
            )

            v_swing_ref = None

        else:

            if not swing_initialized:

                raise RuntimeError(
                    "Swing trajectory was not initialized."
                )

            landing_now = (
                phase_time
                >=
                current_step_time
                -
                TIME_TOLERANCE
            )

            if landing_now:

                p_left_ref = (
                    left_initial_position
                    .copy()
                )

                p_right_ref = (
                    landing_position
                    .copy()
                )

                v_swing_ref = np.zeros(
                    3,
                    dtype=float,
                )

            else:

                swing_sample = (
                    swing_trajectory.update_3d(
                        current_time=(
                            phase_time
                        ),

                        landing_time=(
                            current_step_time
                        ),

                        landing_position_xy=(
                            landing_position[
                                0:2
                            ]
                        ),

                        vertical_parameters=(
                            vertical_parameters
                        ),
                    )
                )

                p_left_ref = (
                    left_initial_position
                    .copy()
                )

                p_right_ref = (
                    swing_sample
                    .position
                    .copy()
                )

                v_swing_ref = (
                    swing_sample
                    .velocity
                    .copy()
                )

        # ====================================================
        # DIFFERENTIAL IK
        # ====================================================

        if (
            phase
            ==
            INITIAL_DOUBLE_SUPPORT
        ):

            (
                qdot_full,
                diagnostics,
                Z,
            ) = (
                solve_double_support_ik(
                    robot=(
                        robot
                    ),

                    q_pin=(
                        q_pin
                    ),

                    left_position_ref=(
                        p_left_ref
                    ),

                    right_position_ref=(
                        p_right_ref
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

                    foot_position_gain=(
                        SUPPORT_POSITION_KP
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

        else:

            (
                qdot_full,
                diagnostics,
                Z,
            ) = (
                solve_single_support_ik(
                    robot=(
                        robot
                    ),

                    q_pin=(
                        q_pin
                    ),

                    support_side=(
                        STANCE_SIDE
                    ),

                    support_position_ref=(
                        p_left_ref
                    ),

                    swing_position_ref=(
                        p_right_ref
                    ),

                    swing_linear_velocity_ref=(
                        v_swing_ref
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
        # INTEGRATE KINEMATICS
        # ====================================================

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

        update_mujoco_from_pinocchio(
            robot=(
                robot
            ),

            q_pin=(
                q_pin
            ),

            mj_model=(
                mj_model
            ),

            mj_data=(
                mj_data
            ),
        )

        # ====================================================
        # TRACKING ERRORS
        # ====================================================

        p_com_actual = (
            robot.get_com()
        )

        (
            p_left_actual,
            _,
        ) = (
            robot.get_left_foot_pose()
        )

        (
            p_right_actual,
            _,
        ) = (
            robot.get_right_foot_pose()
        )

        com_error = float(
            np.linalg.norm(
                com_position_ref
                -
                p_com_actual
            )
        )

        max_com_error = max(
            max_com_error,
            com_error,
        )

        if (
            phase
            ==
            INITIAL_DOUBLE_SUPPORT
        ):

            left_error = float(
                np.linalg.norm(
                    p_left_ref
                    -
                    p_left_actual
                )
            )

            right_error = float(
                np.linalg.norm(
                    p_right_ref
                    -
                    p_right_actual
                )
            )

            support_error = max(
                left_error,
                right_error,
            )

            swing_error = 0.0

        else:

            support_error = float(
                np.linalg.norm(
                    p_left_ref
                    -
                    p_left_actual
                )
            )

            swing_error = float(
                np.linalg.norm(
                    p_right_ref
                    -
                    p_right_actual
                )
            )

        max_support_error = max(
            max_support_error,
            support_error,
        )

        max_swing_error = max(
            max_swing_error,
            swing_error,
        )

        # ====================================================
        # STATUS
        # ====================================================

        if (
            kinematic_time
            >=
            next_print_time
            -
            TIME_TOLERANCE
        ):

            if (
                phase
                ==
                INITIAL_DOUBLE_SUPPORT
            ):

                print(
                    f"t={kinematic_time:6.3f}"
                    f" | phase=INITIAL_DS"
                    f" | CoM="
                    f"({p_com_actual[0]:+.4f},"
                    f"{p_com_actual[1]:+.4f})"
                    f" | CoM err="
                    f"{1000.0 * com_error:.2f} mm"
                    f" | foot err="
                    f"{1000.0 * support_error:.2f} mm"
                )

            else:

                print(
                    f"tSS={phase_time:6.3f}"
                    f" | DCM="
                    f"({dcm[0]:+.4f},"
                    f"{dcm[1]:+.4f})"
                    f" | uT="
                    f"({landing_position[0]:+.4f},"
                    f"{landing_position[1]:+.4f})"
                    f" | T="
                    f"{current_step_time:.4f}"
                    f" | CoM err="
                    f"{1000.0 * com_error:.2f} mm"
                    f" | support err="
                    f"{1000.0 * support_error:.2f} mm"
                    f" | swing err="
                    f"{1000.0 * swing_error:.2f} mm"
                    f" | frozen="
                    f"{int(planner_frozen)}"
                )

            next_print_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # TOUCHDOWN / FINISH
        # ====================================================

        if (
            phase
            ==
            SINGLE_SUPPORT
            and
            phase_time
            >=
            current_step_time
            -
            TIME_TOLERANCE
        ):

            break

        # ====================================================
        # ADVANCE MPC INTERVAL
        # ====================================================

        mpc_substep += 1

        if (
            mpc_substep
            >=
            MPC_IK_STEPS
        ):

            x_state = (
                current_segment
                .get_terminal_x_state()
            )

            y_state = (
                current_segment
                .get_terminal_y_state()
            )

            current_segment = None

            mpc_substep = 0

        # ====================================================
        # TIME
        # ====================================================

        phase_time += (
            DT
        )

        kinematic_time += (
            DT
        )

        iteration += 1

        mj_data.time = (
            kinematic_time
        )

        # ====================================================
        # VIEWER
        # ====================================================

        if viewer is not None:

            viewer.sync()

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

    # ========================================================
    # FINAL
    # ========================================================

    robot.update(
        q_pin
    )

    final_com = (
        robot.get_com()
    )

    (
        final_left,
        _,
    ) = (
        robot.get_left_foot_pose()
    )

    (
        final_right,
        _,
    ) = (
        robot.get_right_foot_pose()
    )

    final_landing_error = float(
        np.linalg.norm(
            landing_position
            -
            final_right
        )
    )

    final_support_error = float(
        np.linalg.norm(
            left_initial_position
            -
            final_left
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    separator()

    print(
        "FINAL RESULT"
    )

    separator()

    print()

    print(
        f"Nominal uT = "
        f"{nominal_landing_position}"
    )

    print(
        f"Nominal T  = "
        f"{nominal_step.step_time:.6f} s"
    )

    print()

    print(
        f"Adapted uT = "
        f"{landing_position}"
    )

    print(
        f"Adapted T  = "
        f"{current_step_time:.6f} s"
    )

    print()

    print(
        "Adaptation:"
    )

    print(
        f"  Delta uT = "
        f"{landing_position - nominal_landing_position}"
    )

    print(
        f"  Delta T  = "
        f"{1000.0 * (current_step_time - nominal_step.step_time):+.3f} ms"
    )

    print()

    if planner_result is not None:

        print(
            f"b = "
            f"({planner_result.dcm_offset_x:+.6f}, "
            f"{planner_result.dcm_offset_y:+.6f}) m"
        )

        print(
            f"viability slack = "
            f"({planner_result.viability_slack_x:.3e}, "
            f"{planner_result.viability_slack_y:.3e})"
        )

        print(
            f"QP equality residual = "
            f"{planner_result.max_equality_residual:.3e}"
        )

    print()

    print(
        f"planner frozen      = "
        f"{planner_frozen}"
    )

    print(
        f"freeze time         = "
        f"{freeze_time}"
    )

    print(
        f"disturbance applied = "
        f"{disturbance_applied}"
    )

    print()

    print(
        f"final CoM           = "
        f"{final_com}"
    )

    print(
        f"final LEFT          = "
        f"{final_left}"
    )

    print(
        f"final RIGHT         = "
        f"{final_right}"
    )

    print()

    print(
        f"final landing error = "
        f"{1000.0 * final_landing_error:.3f} mm"
    )

    print(
        f"final support error = "
        f"{1000.0 * final_support_error:.3f} mm"
    )

    print()

    print(
        f"max CoM error       = "
        f"{1000.0 * max_com_error:.3f} mm"
    )

    print(
        f"max support error   = "
        f"{1000.0 * max_support_error:.3f} mm"
    )

    print(
        f"max swing error     = "
        f"{1000.0 * max_swing_error:.3f} mm"
    )

    print()

    print(
        f"max slack x         = "
        f"{max_viability_slack_x:.3e}"
    )

    print(
        f"max slack y         = "
        f"{max_viability_slack_y:.3e}"
    )

    print()

    print(
        f"MPC solves          = "
        f"{mpc_solve_count}"
    )

    print(
        f"max MPC solve time  = "
        f"{1000.0 * max_mpc_solve_time:.2f} ms"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  This validates planner -> MPC -> IK integration."
    )

    print(
        "  Motion is executed by kinematic set-state only."
    )

    print(
        "  Dynamic feasibility is NOT evaluated."
    )

    separator()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # STEP PLANNER
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
        )
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

    home_id = int(
        mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
    )

    if home_id < 0:

        raise RuntimeError(
            "HOME keyframe was not found."
        )

    mujoco.mj_resetDataKeyframe(
        mj_model,
        mj_data,
        home_id,
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
    # HEADER
    # ========================================================

    separator()

    print(
        "LOAD COMPLETE"
    )

    separator()

    print(
        f"MuJoCo = "
        f"{mujoco.__version__}"
    )

    print(
        f"nq     = "
        f"{mj_model.nq}"
    )

    print(
        f"nv     = "
        f"{mj_model.nv}"
    )

    print(
        f"DT     = "
        f"{DT:.6f} s"
    )

    print(
        f"MPC dt = "
        f"{MPC_TIMESTEP:.6f} s"
    )

    print(
        f"IK steps / MPC interval = "
        f"{MPC_IK_STEPS}"
    )

    print()

    # ========================================================
    # RUN
    # ========================================================

    if SHOW_VIEWER:

        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
        ) as viewer:

            run_test(
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

    else:

        run_test(
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


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()