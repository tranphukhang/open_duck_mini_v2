# step_timing_adaptation/run.py

from __future__ import annotations

import sys
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

if str(
    ROOT_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            ROOT_DIR
        ),
    )


ROBOT_XML = (
    ROOT_DIR
    / "xmls"
    / "open_duck_mini_v2.xml"
)

SCENE_XML = (
    ROOT_DIR
    / "xmls"
    / "scene_flat_terrain.xml"
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

    from .adaptive_support_preview import (
        INITIAL_DOUBLE_SUPPORT,
        SINGLE_SUPPORT,
        DOUBLE_SUPPORT,
        build_adaptive_support_preview,
    )

    from .online_swing_trajectory import (
        OnlineQuinticSwingTrajectory,
    )

else:

    from adaptive_step_planner import (
        AdaptiveStepPlanner,
        StepPlannerParameters,
        StanceLeg,
    )

    from adaptive_support_preview import (
        INITIAL_DOUBLE_SUPPORT,
        SINGLE_SUPPORT,
        DOUBLE_SUPPORT,
        build_adaptive_support_preview,
    )

    from online_swing_trajectory import (
        OnlineQuinticSwingTrajectory,
    )


# ============================================================
# REUSE LIPM-MPC
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
# REUSE WHOLE-BODY KINEMATICS
# ============================================================

from footstep_planning.pinocchio_model import (
    PinocchioModel,
)

from footstep_planning.differential_ik import (
    TRUNK_FRAME,
    solve_single_support_ik,
    solve_double_support_ik,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# WALKING TEST
# ============================================================

WALK_DURATION = 10.0

FIRST_STANCE_SIDE = "left"


# ============================================================
# EXECUTOR
# ============================================================

DT = 0.0005


# ============================================================
# GAIT TIMING
# ============================================================

INITIAL_DOUBLE_SUPPORT_DURATION = (
    0.36
)

DOUBLE_SUPPORT_DURATION = (
    0.27
)


# ============================================================
# LIPM / STEP PLANNER
# ============================================================

GRAVITY = 9.81

COM_HEIGHT = 0.2044

DESIRED_VELOCITY_X = 0.5
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
# STEP ADAPTATION QP
# ============================================================

STEP_QP_ALPHA_LOCATION = 1.0
STEP_QP_ALPHA_TIMING = 5.0
STEP_QP_ALPHA_DCM = 1000.0
STEP_QP_ALPHA_VIABILITY = 1.0e6

STEP_TIMING_GAP = 0.05


# ============================================================
# SWING
# ============================================================

SWING_HEIGHT = 0.04


# ============================================================
# LIPM-MPC
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
# ============================================================

ZMP_SUPPORT_SCALE = 0.9

FOOT_TOE = 0.0645
FOOT_HEEL = 0.0386
FOOT_HALF_WIDTH = 0.02065

SINGLE_SUPPORT_ZMP_HALF_WIDTH = (
    0.0005
)


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
# DISTURBANCE
# ============================================================

ENABLE_DISTURBANCE = False

DISTURBANCE_TIME = 0.10

DISTURBANCE_DCM_X = +0.010
DISTURBANCE_DCM_Y = 0.000


# ============================================================
# VIEWER / PRINT
# ============================================================

SHOW_VIEWER = True
REALTIME_PLAYBACK = True

VIEWER_SYNC_PERIOD = 0.02
STATUS_PRINT_PERIOD = 0.05


# ============================================================
# NUMERICAL
# ============================================================

TIME_TOLERANCE = 1.0e-10


# ============================================================
# MPC / EXECUTOR RATIO
# ============================================================

MPC_EXECUTOR_RATIO = (
    MPC_TIMESTEP
    /
    DT
)

MPC_EXECUTOR_STEPS = int(
    round(
        MPC_EXECUTOR_RATIO
    )
)

if not np.isclose(
    MPC_EXECUTOR_RATIO,
    MPC_EXECUTOR_STEPS,
    atol=1.0e-12,
):

    raise RuntimeError(
        "MPC_TIMESTEP must be "
        "an integer multiple of DT."
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def separator():

    print(
        "=" * 100
    )


def opposite_side(
    side,
):

    if side == "left":
        return "right"

    if side == "right":
        return "left"

    raise ValueError(
        f"Invalid side: {side}"
    )


def stance_leg_from_side(
    side,
):

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

        return (
            left_contact
        )

    if side == "right":

        return (
            right_contact
        )

    raise ValueError(
        f"Invalid side: {side}"
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
# NOMINAL STEP
# ============================================================

def compute_nominal_step(
    planner,
    stance_side,
):

    return (
        planner.compute_nominal_step(
            desired_velocity_x=(
                DESIRED_VELOCITY_X
            ),

            desired_velocity_y=(
                DESIRED_VELOCITY_Y
            ),

            stance_leg=(
                stance_leg_from_side(
                    stance_side
                )
            ),
        )
    )


def compute_nominal_landing(
    *,
    stance_side,
    left_contact,
    right_contact,
    nominal_step,
):

    support = (
        get_contact_position(
            stance_side,
            left_contact,
            right_contact,
        )
    )

    swing_side = (
        opposite_side(
            stance_side
        )
    )

    swing_contact = (
        get_contact_position(
            swing_side,
            left_contact,
            right_contact,
        )
    )

    target = (
        swing_contact.copy()
    )

    target[0] = (
        support[0]
        +
        nominal_step.step_displacement_x
    )

    target[1] = (
        support[1]
        +
        nominal_step.step_displacement_y
    )

    return target


# ============================================================
# MPC
# ============================================================

def create_axis_mpc():

    model = LIPMModel1D(
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

    terminal_weights = np.array(
        [
            TERMINAL_POSITION_WEIGHT,
            TERMINAL_VELOCITY_WEIGHT,
            TERMINAL_ACCELERATION_WEIGHT,
        ],
        dtype=float,
    )

    controller = LIPMMPC1D(
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

    if control.shape != (
        MPC_HORIZON_STEPS,
    ):

        raise ValueError(
            "Previous MPC control sequence "
            "has invalid shape."
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
# CONTINUOUS LIPM STATE
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
# STEP PLANNER
# ============================================================

def solve_step_planner(
    *,
    planner,
    nominal_step,
    dcm,
    stance_position,
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
    )


# ============================================================
# SOLVE MPC SEGMENT
# ============================================================

def solve_mpc_segment(
    *,
    current_phase,
    phase_time,

    stance_side,
    swing_side,

    x_state,
    y_state,

    x_mpc,
    y_mpc,

    previous_x_control,
    previous_y_control,

    left_contact_position,
    right_contact_position,

    landing_position,
    current_step_time,

    nominal_left_step,
    nominal_right_step,

    left_rotation,
    right_rotation,
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

            double_support_duration=(
                DOUBLE_SUPPORT_DURATION
            ),

            current_single_support_duration=(
                current_step_time
            ),

            stance_side=(
                stance_side
            ),

            swing_side=(
                swing_side
            ),

            left_contact_position=(
                left_contact_position
            ),

            right_contact_position=(
                right_contact_position
            ),

            current_landing_position=(
                landing_position
            ),

            nominal_left_step_displacement=np.array(
                [
                    nominal_left_step.step_displacement_x,
                    nominal_left_step.step_displacement_y,
                ],
                dtype=float,
            ),

            nominal_right_step_displacement=np.array(
                [
                    nominal_right_step.step_displacement_x,
                    nominal_right_step.step_displacement_y,
                ],
                dtype=float,
            ),

            nominal_left_step_time=(
                nominal_left_step.step_time
            ),

            nominal_right_step_time=(
                nominal_right_step.step_time
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

    # ========================================================
    # TERMINAL GOAL
    # ========================================================

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

    # ========================================================
    # MPC SOLVE
    # ========================================================

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
# RUN
# ============================================================

def run_walk(
    *,
    mj_model,
    mj_data,
    robot,
    planner,
    viewer=None,
):

    # ========================================================
    # INITIAL ROBOT STATE
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
        left_rotation,
    ) = (
        robot.get_left_foot_pose()
    )

    (
        right_contact_position,
        right_rotation,
    ) = (
        robot.get_right_foot_pose()
    )

    left_contact_position = (
        left_contact_position.copy()
    )

    right_contact_position = (
        right_contact_position.copy()
    )

    initial_com = (
        robot.get_com()
    )

    (
        _,
        trunk_rotation_ref,
    ) = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    # ========================================================
    # WORLD-Z COM REFERENCE
    #
    # Flat terrain:
    # use initial support-plane height.
    # ========================================================

    ground_z_reference = (
        0.5
        *
        (
            left_contact_position[2]
            +
            right_contact_position[2]
        )
    )

    com_world_z_reference = (
        ground_z_reference
        +
        COM_HEIGHT
    )

    # ========================================================
    # NOMINAL STEP FOR EACH STANCE LEG
    # ========================================================

    nominal_left_step = (
        compute_nominal_step(
            planner,
            "left",
        )
    )

    nominal_right_step = (
        compute_nominal_step(
            planner,
            "right",
        )
    )

    # ========================================================
    # INITIAL GAIT STATE
    # ========================================================

    stance_side = (
        FIRST_STANCE_SIDE
    )

    swing_side = (
        opposite_side(
            stance_side
        )
    )

    current_nominal_step = (
        nominal_left_step
        if
        stance_side
        ==
        "left"
        else
        nominal_right_step
    )

    landing_position = (
        compute_nominal_landing(
            stance_side=(
                stance_side
            ),

            left_contact=(
                left_contact_position
            ),

            right_contact=(
                right_contact_position
            ),

            nominal_step=(
                current_nominal_step
            ),
        )
    )

    current_step_time = (
        current_nominal_step.step_time
    )

    # ========================================================
    # INITIAL LIPM STATE
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

    _, x_mpc = (
        create_axis_mpc()
    )

    _, y_mpc = (
        create_axis_mpc()
    )

    current_segment = None

    mpc_substep = 0

    previous_x_control = None
    previous_y_control = None

    mpc_solve_count = 0

    max_mpc_solve_time = 0.0

    # ========================================================
    # SWING
    # ========================================================

    swing_trajectory = (
        OnlineQuinticSwingTrajectory()
    )

    swing_initialized = (
        False
    )

    # ========================================================
    # PHASE
    # ========================================================

    phase = (
        INITIAL_DOUBLE_SUPPORT
    )

    phase_time = (
        0.0
    )

    kinematic_time = (
        0.0
    )

    step_index = (
        0
    )

    planner_result = (
        None
    )

    planner_frozen = (
        False
    )

    disturbance_applied = (
        False
    )

    stop_requested = (
        False
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    max_com_error = 0.0

    max_support_error = 0.0

    max_swing_error = 0.0

    max_landing_error = 0.0

    last_landing_error = 0.0

    max_viability_slack_x = 0.0

    max_viability_slack_y = 0.0

    next_print_time = 0.0

    next_viewer_sync_time = 0.0

    # ========================================================
    # HEADER
    # ========================================================

    separator()

    print(
        "CONTINUOUS STEP TIMING ADAPTATION WALKING"
    )

    print(
        "LIPM-MPC + ONLINE QUINTIC SWING "
        "+ HIERARCHICAL DIFFERENTIAL IK"
    )

    separator()

    print()

    print(
        f"walk duration = "
        f"{WALK_DURATION:.3f} s"
    )

    print(
        f"DT            = "
        f"{DT:.6f} s"
    )

    print(
        f"MPC dt        = "
        f"{MPC_TIMESTEP:.6f} s"
    )

    print(
        f"initial DS    = "
        f"{INITIAL_DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"normal DS     = "
        f"{DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"timing gap    = "
        f"{STEP_TIMING_GAP:.3f} s"
    )

    print()

    print(
        f"Initial LEFT  = "
        f"{left_contact_position}"
    )

    print(
        f"Initial RIGHT = "
        f"{right_contact_position}"
    )

    print(
        f"Initial CoM   = "
        f"{initial_com}"
    )

    print()

    wall_start = (
        time.perf_counter()
    )

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
            WALK_DURATION
        ):

            stop_requested = (
                True
            )

        # ====================================================
        # TOUCHDOWN:
        # SINGLE SUPPORT -> DOUBLE SUPPORT
        #
        # One exact touchdown reference sample has already
        # been executed because transition occurs only after
        # phase_time exceeds T.
        # ====================================================

        if (
            phase
            ==
            SINGLE_SUPPORT
            and
            phase_time
            >
            current_step_time
            +
            TIME_TOLERANCE
        ):

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

            current_segment = (
                None
            )

            mpc_substep = (
                0
            )

            previous_x_control = (
                None
            )

            previous_y_control = (
                None
            )

            robot.update(
                q_pin
            )

            if (
                swing_side
                ==
                "left"
            ):

                (
                    actual_landing,
                    _,
                ) = (
                    robot.get_left_foot_pose()
                )

                left_contact_position = (
                    landing_position.copy()
                )

            else:

                (
                    actual_landing,
                    _,
                ) = (
                    robot.get_right_foot_pose()
                )

                right_contact_position = (
                    landing_position.copy()
                )

            last_landing_error = float(
                np.linalg.norm(
                    actual_landing
                    -
                    landing_position
                )
            )

            max_landing_error = max(
                max_landing_error,
                last_landing_error,
            )

            print()

            print(
                f"TOUCHDOWN STEP {step_index}"
                f" | {swing_side.upper()} foot"
                f" | error="
                f"{1000.0 * last_landing_error:.3f} mm"
            )

            # ------------------------------------------------
            # Landed foot becomes next stance.
            # ------------------------------------------------

            stance_side = (
                swing_side
            )

            swing_side = (
                opposite_side(
                    stance_side
                )
            )

            current_nominal_step = (
                nominal_left_step
                if
                stance_side
                ==
                "left"
                else
                nominal_right_step
            )

            # Nominal upcoming target during DS preview.

            landing_position = (
                compute_nominal_landing(
                    stance_side=(
                        stance_side
                    ),

                    left_contact=(
                        left_contact_position
                    ),

                    right_contact=(
                        right_contact_position
                    ),

                    nominal_step=(
                        current_nominal_step
                    ),
                )
            )

            current_step_time = (
                current_nominal_step.step_time
            )

            phase = (
                DOUBLE_SUPPORT
            )

            phase_time = (
                0.0
            )

            swing_initialized = (
                False
            )

            planner_result = (
                None
            )

            planner_frozen = (
                False
            )

            disturbance_applied = (
                False
            )

        # ====================================================
        # INITIAL DS -> SS
        # OR NORMAL DS -> SS
        # ====================================================

        start_single_support = (
            (
                phase
                ==
                INITIAL_DOUBLE_SUPPORT
                and
                phase_time
                >=
                INITIAL_DOUBLE_SUPPORT_DURATION
                -
                TIME_TOLERANCE
            )
            or
            (
                phase
                ==
                DOUBLE_SUPPORT
                and
                phase_time
                >=
                DOUBLE_SUPPORT_DURATION
                -
                TIME_TOLERANCE
                and
                not stop_requested
            )
        )

        if start_single_support:

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

            current_segment = (
                None
            )

            mpc_substep = (
                0
            )

            previous_x_control = (
                None
            )

            previous_y_control = (
                None
            )

            phase = (
                SINGLE_SUPPORT
            )

            phase_time = (
                0.0
            )

            step_index += (
                1
            )

            current_nominal_step = (
                nominal_left_step
                if
                stance_side
                ==
                "left"
                else
                nominal_right_step
            )

            stance_position = (
                get_contact_position(
                    stance_side,
                    left_contact_position,
                    right_contact_position,
                )
            )

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
                        current_nominal_step
                    ),

                    dcm=(
                        dcm
                    ),

                    stance_position=(
                        stance_position
                    ),

                    elapsed_time=0.0,
                )
            )

            swing_contact = (
                get_contact_position(
                    swing_side,
                    left_contact_position,
                    right_contact_position,
                )
            )

            landing_position = (
                swing_contact.copy()
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

            # ------------------------------------------------
            # Initialize online swing from ACTUAL foot pose.
            # ------------------------------------------------

            robot.update(
                q_pin
            )

            if (
                swing_side
                ==
                "left"
            ):

                (
                    swing_start,
                    _,
                ) = (
                    robot.get_left_foot_pose()
                )

            else:

                (
                    swing_start,
                    _,
                ) = (
                    robot.get_right_foot_pose()
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

            swing_initialized = (
                True
            )

            planner_frozen = (
                False
            )

            disturbance_applied = (
                False
            )

            print()

            separator()

            print(
                f"STEP {step_index}"
                f" | {stance_side.upper()} support"
                f" | {swing_side.upper()} swing"
            )

            separator()

            print(
                f"DCM       = {dcm}"
            )

            print(
                f"nominal T = "
                f"{current_nominal_step.step_time:.6f} s"
            )

            print(
                f"adapted T = "
                f"{current_step_time:.6f} s"
            )

            print(
                f"uT        = "
                f"{landing_position}"
            )

            print()

        # ====================================================
        # GRACEFUL FINISH AFTER FULL DS
        # ====================================================

        if (
            phase
            ==
            DOUBLE_SUPPORT
            and
            stop_requested
            and
            phase_time
            >=
            DOUBLE_SUPPORT_DURATION
            -
            TIME_TOLERANCE
        ):

            break

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

            current_segment = (
                None
            )

            mpc_substep = (
                0
            )

            previous_x_control = (
                None
            )

            previous_y_control = (
                None
            )

            disturbance_applied = (
                True
            )

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
        # ONLINE STEP ADAPTATION
        # ====================================================

        if (
            phase
            ==
            SINGLE_SUPPORT
            and
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

                planner_frozen = (
                    True
                )

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
                        solve_step_planner(
                            planner=(
                                planner
                            ),

                            nominal_step=(
                                current_nominal_step
                            ),

                            dcm=(
                                dcm
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

                    if (
                        phase_time
                        >=
                        current_step_time
                        -
                        STEP_TIMING_GAP
                        -
                        TIME_TOLERANCE
                    ):

                        planner_frozen = (
                            True
                        )

                except RuntimeError as error:

                    if (
                        "timing adaptation window is closed"
                        in
                        str(
                            error
                        ).lower()
                    ):

                        planner_frozen = (
                            True
                        )

                    else:

                        raise

        # ====================================================
        # MPC
        # ====================================================

        if current_segment is None:

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

                    stance_side=(
                        stance_side
                    ),

                    swing_side=(
                        swing_side
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

                    left_contact_position=(
                        left_contact_position
                    ),

                    right_contact_position=(
                        right_contact_position
                    ),

                    landing_position=(
                        landing_position
                    ),

                    current_step_time=(
                        current_step_time
                    ),

                    nominal_left_step=(
                        nominal_left_step
                    ),

                    nominal_right_step=(
                        nominal_right_step
                    ),

                    left_rotation=(
                        left_rotation
                    ),

                    right_rotation=(
                        right_rotation
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

            mpc_substep = (
                0
            )

            mpc_solve_count += (
                1
            )

            max_mpc_solve_time = max(
                max_mpc_solve_time,
                mpc_info[
                    "solve_time"
                ],
            )

        # ====================================================
        # CONTINUOUS COM REFERENCE
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

        com_position_ref = (
            com_reference.position.copy()
        )

        com_velocity_ref = (
            com_reference.velocity.copy()
        )

        com_position_ref[2] = (
            com_world_z_reference
        )

        com_velocity_ref[2] = (
            0.0
        )

        # ====================================================
        # FOOT REFERENCES
        # ====================================================

        if phase in (
            INITIAL_DOUBLE_SUPPORT,
            DOUBLE_SUPPORT,
        ):

            p_left_ref = (
                left_contact_position.copy()
            )

            p_right_ref = (
                right_contact_position.copy()
            )

            v_swing_ref = (
                None
            )

        else:

            if not swing_initialized:

                raise RuntimeError(
                    "Swing trajectory "
                    "is not initialized."
                )

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

            if (
                swing_side
                ==
                "left"
            ):

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

            v_swing_ref = (
                swing_sample.velocity.copy()
            )

        # ====================================================
        # DIFFERENTIAL IK
        # ====================================================

        if phase in (
            INITIAL_DOUBLE_SUPPORT,
            DOUBLE_SUPPORT,
        ):

            (
                qdot_full,
                _,
                _,
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
                "Differential IK "
                "returned NaN/Inf."
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

        if phase in (
            INITIAL_DOUBLE_SUPPORT,
            DOUBLE_SUPPORT,
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

            swing_error = (
                0.0
            )

        else:

            if (
                stance_side
                ==
                "left"
            ):

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

            else:

                support_error = float(
                    np.linalg.norm(
                        p_right_ref
                        -
                        p_right_actual
                    )
                )

                swing_error = float(
                    np.linalg.norm(
                        p_left_ref
                        -
                        p_left_actual
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
                SINGLE_SUPPORT
            ):

                print(
                    f"t={kinematic_time:6.3f}"
                    f" | STEP={step_index:02d}"
                    f" | SS {stance_side.upper()}"
                    f" | tSS={phase_time:.3f}"
                    f" | T={current_step_time:.3f}"
                    f" | uT="
                    f"({landing_position[0]:+.3f},"
                    f"{landing_position[1]:+.3f})"
                    f" | CoM err="
                    f"{1000.0 * com_error:.2f} mm"
                    f" | swing err="
                    f"{1000.0 * swing_error:.2f} mm"
                )

            else:

                print(
                    f"t={kinematic_time:6.3f}"
                    f" | {phase}"
                    f" | CoM err="
                    f"{1000.0 * com_error:.2f} mm"
                    f" | feet err="
                    f"{1000.0 * support_error:.2f} mm"
                )

            next_print_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # ADVANCE MPC SEGMENT
        # ====================================================

        mpc_substep += (
            1
        )

        if (
            mpc_substep
            >=
            MPC_EXECUTOR_STEPS
        ):

            x_state = (
                current_segment
                .get_terminal_x_state()
            )

            y_state = (
                current_segment
                .get_terminal_y_state()
            )

            current_segment = (
                None
            )

            mpc_substep = (
                0
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

    print()

    separator()

    print(
        "FINAL WALKING RESULT"
    )

    separator()

    print()

    print(
        f"logical walking time = "
        f"{kinematic_time:.3f} s"
    )

    print(
        f"completed steps      = "
        f"{step_index}"
    )

    print()

    print(
        f"final CoM   = "
        f"{final_com}"
    )

    print(
        f"final LEFT  = "
        f"{final_left}"
    )

    print(
        f"final RIGHT = "
        f"{final_right}"
    )

    print()

    print(
        f"last landing error = "
        f"{1000.0 * last_landing_error:.3f} mm"
    )

    print(
        f"max landing error  = "
        f"{1000.0 * max_landing_error:.3f} mm"
    )

    print()

    print(
        f"max CoM error      = "
        f"{1000.0 * max_com_error:.3f} mm"
    )

    print(
        f"max support error  = "
        f"{1000.0 * max_support_error:.3f} mm"
    )

    print(
        f"max swing error    = "
        f"{1000.0 * max_swing_error:.3f} mm"
    )

    print()

    print(
        f"max viability slack x = "
        f"{max_viability_slack_x:.3e}"
    )

    print(
        f"max viability slack y = "
        f"{max_viability_slack_y:.3e}"
    )

    print()

    print(
        f"MPC solves         = "
        f"{mpc_solve_count}"
    )

    print(
        f"max MPC solve time = "
        f"{1000.0 * max_mpc_solve_time:.2f} ms"
    )

    print()

    print(
        "Execution remains purely kinematic."
    )

    print(
        "Dynamic feasibility is not evaluated."
    )

    separator()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # PLANNER
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

    mj_data.qvel[:] = (
        0.0
    )

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
        f"DT     = "
        f"{DT:.6f} s"
    )

    print(
        f"executor = "
        f"{1.0 / DT:.1f} Hz"
    )

    print(
        f"MPC dt   = "
        f"{MPC_TIMESTEP:.6f} s"
    )

    print(
        f"MPC horizon = "
        f"{MPC_TIMESTEP * MPC_HORIZON_STEPS:.3f} s"
    )

    print()

    # ========================================================
    # RUN
    # ========================================================

    if SHOW_VIEWER:

        with (
            mujoco.viewer.launch_passive(
                mj_model,
                mj_data,
            )
            as viewer
        ):

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

    else:

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


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()