# lipm_mpc/run.py

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = Path(
    __file__
).resolve().parent

ROOT_DIR = (
    CURRENT_DIR.parent
)

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# LIPM-MPC IMPORTS
# ============================================================

if __package__:

    from .lipm_model import (
        LIPMModel1D,
    )

    from .support_preview import (
        build_support_preview,
    )

    from .mpc_1d import (
        LIPMMPC1D,
    )

    from .com_trajectory import (
        ConstantJerkCoMSegment,
    )

    from .zmp_visualization import (
        ZMPVisualizationConfig,
        LIPMZMPVisualizer,
    )

else:

    from lipm_model import (
        LIPMModel1D,
    )

    from support_preview import (
        build_support_preview,
    )

    from mpc_1d import (
        LIPMMPC1D,
    )

    from com_trajectory import (
        ConstantJerkCoMSegment,
    )

    from zmp_visualization import (
        ZMPVisualizationConfig,
        LIPMZMPVisualizer,
    )


# ============================================================
# FOOTSTEP / EXECUTOR IMPORTS
# ============================================================

from footstep_planning.walking_fsm import (
    WalkingFSM,
    WalkingPhase,
)

from footstep_planning.swing_trajectory import (
    compute_swing_trajectory,
)

from footstep_planning.pinocchio_model import (
    PinocchioModel,
)

from footstep_planning.differential_ik import (
    TRUNK_FRAME,
    solve_single_support_ik,
    solve_double_support_ik,
)

from footstep_planning.walking_visualization import (
    WalkingVisualizer,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# PATHS
# ============================================================

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
# EXECUTOR TIMING
# ============================================================

DT = 0.0005


# ============================================================
# WALKING PARAMETERS
# ============================================================

STEP_LENGTH = 0.04

FEET_SPACING = 0.16

SWING_HEIGHT = 0.04

COM_HEIGHT = 0.205

FIRST_SWING_SIDE = "right"


# ============================================================
# GAIT TIMING
# ============================================================

INITIAL_DOUBLE_SUPPORT_DURATION = 0.36

SINGLE_SUPPORT_DURATION = 0.18

DOUBLE_SUPPORT_DURATION = 0.27


# ============================================================
# MPC PARAMETERS
# ============================================================

MPC_TIMESTEP = 0.03

MPC_HORIZON_STEPS = 48

GRAVITY = 9.81


# ------------------------------------------------------------
# TERMINAL COST
# ------------------------------------------------------------

TERMINAL_POSITION_WEIGHT = 1.0

TERMINAL_VELOCITY_WEIGHT = 1.0

TERMINAL_ACCELERATION_WEIGHT = 1.0


# ------------------------------------------------------------
# CONTROL COST
# ------------------------------------------------------------

CONTROL_WEIGHT = 2e-5


# ------------------------------------------------------------
# SLSQP
# ------------------------------------------------------------

MPC_SOLVER_OPTIONS = {
    "ftol": 1e-10,
    "maxiter": 1000,
}


# ============================================================
# SUPPORT REGION
# ============================================================

ZMP_SUPPORT_SCALE = 0.9

FOOT_TOE = 0.0645

FOOT_HEEL = 0.0386

FOOT_HALF_WIDTH = 0.02065


# ============================================================
# IK PARAMETERS
# ============================================================

IK_DAMPING = 1e-8

IK_RCOND = 1e-10

SUPPORT_POSITION_KP = 25.0

SWING_POSITION_KP = 20.0

COM_POSITION_KP = 10.0

TRUNK_ORIENTATION_KP = 10.0


# ============================================================
# SETTLING PARAMETERS
# ============================================================

BASE_LIN_TOL = 5e-4

BASE_ANG_TOL = 1e-3

JOINT_VEL_TOL = 2.5e-3

STABLE_DURATION = 0.10

MIN_SETTLE_TIME = 0.50

MAX_SETTLE_TIME = 5.00


# ============================================================
# VIEWER
# ============================================================

VIEWER_FPS = 60.0

VIEWER_SYNC_STEPS = max(
    1,
    int(
        round(
            (
                1.0
                /
                VIEWER_FPS
            )
            /
            DT
        )
    ),
)

TRUNK_BODY_NAME = (
    "trunk_assembly"
)


# ============================================================
# ZMP VISUALIZATION PARAMETERS
# ============================================================
#
# These are visualization parameters only.
#
# All configuration remains in run.py.
# zmp_visualization.py contains only implementation logic.
# ============================================================

ZMP_VISUALIZATION_CONFIG = (
    ZMPVisualizationConfig(

        show_current=True,

        show_trail=True,

        show_preview=True,

        # ----------------------------------------------------
        # Height above ground
        # ----------------------------------------------------

        current_z=0.010,

        trail_z=0.008,

        preview_z=0.007,

        # ----------------------------------------------------
        # Marker sizes
        # ----------------------------------------------------

        current_radius=0.007,

        preview_radius=0.0028,

        # ----------------------------------------------------
        # Line widths
        # ----------------------------------------------------

        trail_width=5.0,

        preview_width=2.5,

        # ----------------------------------------------------
        # Trail
        # ----------------------------------------------------

        trail_min_distance=5e-4,

        trail_max_points=180,

        # ----------------------------------------------------
        # Draw one future sphere every N samples
        # ----------------------------------------------------

        preview_point_stride=2,

        # ----------------------------------------------------
        # Current ZMP = magenta
        # ----------------------------------------------------

        current_rgba=np.array(
            [
                1.00,
                0.00,
                1.00,
                1.00,
            ],
            dtype=np.float32,
        ),

        # ----------------------------------------------------
        # ZMP history
        # ----------------------------------------------------

        trail_rgba=np.array(
            [
                0.90,
                0.10,
                0.95,
                0.80,
            ],
            dtype=np.float32,
        ),

        # ----------------------------------------------------
        # Future ZMP points
        # ----------------------------------------------------

        preview_rgba=np.array(
            [
                0.72,
                0.28,
                1.00,
                0.72,
            ],
            dtype=np.float32,
        ),

        # ----------------------------------------------------
        # Future ZMP line
        # ----------------------------------------------------

        preview_line_rgba=np.array(
            [
                0.72,
                0.28,
                1.00,
                0.55,
            ],
            dtype=np.float32,
        ),
    )
)


# ============================================================
# MPC / IK RATE
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
    atol=1e-12,
):

    raise RuntimeError(
        "MPC_TIMESTEP must be an integer multiple of DT."
    )


# ============================================================
# PRINT
# ============================================================

MPC_PRINT_EVERY = 10


# ============================================================
# USER STOP REQUEST
# ============================================================

stop_requested = (
    threading.Event()
)


def keyboard_callback(
    keycode,
):

    if keycode in (
        ord("F"),
        ord("f"),
    ):

        if not stop_requested.is_set():

            stop_requested.set()

            print()

            print(
                "F pressed -> graceful stop requested."
            )


# ============================================================
# HELPERS
# ============================================================

def separator():

    print(
        "=" * 94
    )


def phase_name(
    phase,
):

    if hasattr(
        phase,
        "value",
    ):

        return str(
            phase.value
        )

    return str(
        phase
    )


def rotation_matrix_to_pitch(
    R,
):

    R = np.asarray(
        R,
        dtype=float,
    )

    pitch = np.arctan2(
        -R[2, 0],
        np.sqrt(
            R[0, 0] ** 2
            +
            R[1, 0] ** 2
        ),
    )

    return float(
        pitch
    )


def get_mujoco_body_pitch(
    model,
    data,
    body_name,
):

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:

        raise RuntimeError(
            f"Body '{body_name}' was not found."
        )

    R = (
        data.xmat[
            body_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    return rotation_matrix_to_pitch(
        R
    )


def get_trunk_local_y_error(
    robot,
    trunk_rotation_ref,
):

    _, R_actual = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    error_local = pin.log3(
        R_actual.T
        @
        trunk_rotation_ref
    )

    return float(
        error_local[1]
    )


# ============================================================
# SETTLING
# ============================================================

def settle_robot(
    model,
    data,
):

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id < 0:

        raise RuntimeError(
            "HOME keyframe was not found."
        )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    stable_time = 0.0

    settled = False

    max_steps = int(
        np.ceil(
            MAX_SETTLE_TIME
            /
            model.opt.timestep
        )
    )

    for _ in range(
        max_steps
    ):

        mujoco.mj_step(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        base_linear_speed = float(
            np.linalg.norm(
                data.qvel[
                    0:3
                ]
            )
        )

        base_angular_speed = float(
            np.linalg.norm(
                data.qvel[
                    3:6
                ]
            )
        )

        max_joint_speed = float(
            np.max(
                np.abs(
                    data.qvel[
                        6:
                    ]
                )
            )
        )

        stable_now = (
            current_time
            >=
            MIN_SETTLE_TIME
            and
            base_linear_speed
            <
            BASE_LIN_TOL
            and
            base_angular_speed
            <
            BASE_ANG_TOL
            and
            max_joint_speed
            <
            JOINT_VEL_TOL
        )

        if stable_now:

            stable_time += (
                model.opt.timestep
            )

        else:

            stable_time = 0.0

        if (
            stable_time
            >=
            STABLE_DURATION
        ):

            settled = True

            break

    if not settled:

        raise RuntimeError(
            "Robot did not satisfy settling criterion."
        )

    return {
        "time":
            float(
                data.time
            ),

        "base_linear_speed":
            float(
                np.linalg.norm(
                    data.qvel[
                        0:3
                    ]
                )
            ),

        "base_angular_speed":
            float(
                np.linalg.norm(
                    data.qvel[
                        3:6
                    ]
                )
            ),

        "max_joint_speed":
            float(
                np.max(
                    np.abs(
                        data.qvel[
                            6:
                        ]
                    )
                )
            ),
    }


# ============================================================
# PINOCCHIO -> MUJOCO
# ============================================================

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
# CREATE MPC
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

    mpc = LIPMMPC1D(
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
        mpc,
    )


# ============================================================
# MPC WARM START
# ============================================================

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
            "Previous MPC control sequence has invalid shape."
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

    return (
        shifted
    )


# ============================================================
# TERMINAL GOAL
# ============================================================

def compute_terminal_goals(
    preview,
):

    x_center = 0.5 * (
        preview.x_min[-1]
        +
        preview.x_max[-1]
    )

    y_center = 0.5 * (
        preview.y_min[-1]
        +
        preview.y_max[-1]
    )

    x_goal = np.array(
        [
            x_center,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    y_goal = np.array(
        [
            y_center,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    return (
        x_goal,
        y_goal,
    )


# ============================================================
# SOLVE ONE MPC INTERVAL
# ============================================================

def solve_mpc_segment(
    fsm,
    left_rotation,
    right_rotation,
    x_state,
    y_state,
    x_mpc,
    y_mpc,
    previous_x_control,
    previous_y_control,
):

    # ========================================================
    # SUPPORT PREVIEW
    # ========================================================

    preview = build_support_preview(
        fsm=(
            fsm
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
    )

    # ========================================================
    # TERMINAL GOAL
    # ========================================================

    (
        x_goal,
        y_goal,
    ) = compute_terminal_goals(
        preview
    )

    # ========================================================
    # SOLVE
    # ========================================================

    wall_start = (
        time.perf_counter()
    )

    x_result = x_mpc.solve(
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

    y_result = y_mpc.solve(
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

    solve_wall_time = (
        time.perf_counter()
        -
        wall_start
    )

    if not x_result.success:

        raise RuntimeError(
            "X-MPC failed: "
            f"{x_result.message}"
        )

    if not y_result.success:

        raise RuntimeError(
            "Y-MPC failed: "
            f"{y_result.message}"
        )

    # ========================================================
    # CONTINUOUS COM SEGMENT
    # ========================================================

    segment = ConstantJerkCoMSegment(
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

    return {
        "preview":
            preview,

        "x_goal":
            x_goal,

        "y_goal":
            y_goal,

        "x_result":
            x_result,

        "y_result":
            y_result,

        "segment":
            segment,

        "solve_wall_time":
            float(
                solve_wall_time
            ),
    }


# ============================================================
# FOOT REFERENCES
# ============================================================

def compute_foot_references(
    state,
):

    # ========================================================
    # SINGLE SUPPORT
    # ========================================================

    if (
        state.phase
        ==
        WalkingPhase.SINGLE_SUPPORT
    ):

        (
            p_swing_ref,
            v_swing_ref,
        ) = compute_swing_trajectory(
            start_position=(
                state.swing_start
            ),

            target_position=(
                state.swing_target
            ),

            phase_time=(
                state.phase_time
            ),

            duration=(
                state.phase_duration
            ),

            swing_height=(
                SWING_HEIGHT
            ),
        )

        if (
            state.swing_side
            ==
            "right"
        ):

            p_left_ref = (
                state
                .left_contact_position
                .copy()
            )

            p_right_ref = (
                p_swing_ref.copy()
            )

        elif (
            state.swing_side
            ==
            "left"
        ):

            p_left_ref = (
                p_swing_ref.copy()
            )

            p_right_ref = (
                state
                .right_contact_position
                .copy()
            )

        else:

            raise RuntimeError(
                f"Invalid swing side: "
                f"{state.swing_side}"
            )

        return (
            p_left_ref,
            p_right_ref,
            v_swing_ref.copy(),
        )

    # ========================================================
    # DOUBLE SUPPORT
    # ========================================================

    if state.phase in (
        WalkingPhase.INITIAL_DOUBLE_SUPPORT,
        WalkingPhase.DOUBLE_SUPPORT,
        WalkingPhase.FINAL_DOUBLE_SUPPORT,
    ):

        p_left_ref = (
            state
            .left_contact_position
            .copy()
        )

        p_right_ref = (
            state
            .right_contact_position
            .copy()
        )

        return (
            p_left_ref,
            p_right_ref,
            None,
        )

    raise RuntimeError(
        f"Cannot compute foot references "
        f"for phase {state.phase}."
    )


# ============================================================
# IK
# ============================================================

def solve_current_phase(
    robot,
    q_pin,
    state,
    p_left_ref,
    p_right_ref,
    p_com_ref,
    v_com_ref,
    v_swing_ref,
    trunk_rotation_ref,
):

    # ========================================================
    # SINGLE SUPPORT
    # ========================================================

    if (
        state.phase
        ==
        WalkingPhase.SINGLE_SUPPORT
    ):

        if (
            state.support_side
            ==
            "left"
        ):

            support_position_ref = (
                p_left_ref
            )

            swing_position_ref = (
                p_right_ref
            )

        elif (
            state.support_side
            ==
            "right"
        ):

            support_position_ref = (
                p_right_ref
            )

            swing_position_ref = (
                p_left_ref
            )

        else:

            raise RuntimeError(
                f"Invalid support side: "
                f"{state.support_side}"
            )

        return solve_single_support_ik(
            robot=(
                robot
            ),

            q_pin=(
                q_pin
            ),

            support_side=(
                state.support_side
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
                p_com_ref
            ),

            com_velocity_ref=(
                v_com_ref
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

    # ========================================================
    # DOUBLE SUPPORT
    # ========================================================

    return solve_double_support_ik(
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
            p_com_ref
        ),

        com_velocity_ref=(
            v_com_ref
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


# ============================================================
# MAIN
# ============================================================

def main():

    stop_requested.clear()

    # ========================================================
    # LOAD MUJOCO
    # ========================================================

    separator()

    print(
        "LOAD MUJOCO"
    )

    separator()

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
    # SETTLING
    # ========================================================

    separator()

    print(
        "SETTLING"
    )

    separator()

    settle_info = (
        settle_robot(
            mj_model,
            mj_data,
        )
    )

    print(
        f"settled at t = "
        f"{settle_info['time']:.4f} s"
    )

    # ========================================================
    # PINOCCHIO
    # ========================================================

    robot = PinocchioModel(
        mjcf_path=(
            ROBOT_XML
        ),

        mujoco_model=(
            mj_model
        ),
    )

    q_pin = (
        robot.mujoco_to_pin(
            mj_data.qpos.copy()
        )
    )

    robot.update(
        q_pin
    )

    (
        p_left_0,
        R_left_0,
    ) = robot.get_left_foot_pose()

    (
        p_right_0,
        R_right_0,
    ) = robot.get_right_foot_pose()

    p_com_0 = (
        robot.get_com()
    )

    (
        _,
        trunk_rotation_ref,
    ) = robot.get_frame_pose(
        TRUNK_FRAME
    )

    initial_trunk_pitch = (
        get_mujoco_body_pitch(
            mj_model,
            mj_data,
            TRUNK_BODY_NAME,
        )
    )

    initial_trunk_pitch_deg = float(
        np.degrees(
            initial_trunk_pitch
        )
    )

    # ========================================================
    # INITIAL STATE INFORMATION
    # ========================================================

    separator()

    print(
        "SETTLED INITIAL STATE"
    )

    separator()

    print(
        "LEFT foot  =",
        p_left_0,
    )

    print(
        "RIGHT foot =",
        p_right_0,
    )

    print(
        "CoM        =",
        p_com_0,
    )

    print(
        f"trunk pitch = "
        f"{initial_trunk_pitch_deg:+.4f} deg"
    )

    # ========================================================
    # FSM
    # ========================================================

    fsm = WalkingFSM(
        p_left_initial=(
            p_left_0
        ),

        p_right_initial=(
            p_right_0
        ),

        step_length=(
            STEP_LENGTH
        ),

        feet_spacing=(
            FEET_SPACING
        ),

        single_support_duration=(
            SINGLE_SUPPORT_DURATION
        ),

        double_support_duration=(
            DOUBLE_SUPPORT_DURATION
        ),

        first_swing_side=(
            FIRST_SWING_SIDE
        ),

        initial_double_support_duration=(
            INITIAL_DOUBLE_SUPPORT_DURATION
        ),
    )

    # ========================================================
    # INITIAL LIPM STATE
    # ========================================================

    x_state = np.array(
        [
            p_com_0[0],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    y_state = np.array(
        [
            p_com_0[1],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # ========================================================
    # MPC OBJECTS
    # ========================================================

    _, x_mpc = (
        create_axis_mpc()
    )

    _, y_mpc = (
        create_axis_mpc()
    )

    # ========================================================
    # WALKING VISUALIZER
    # ========================================================

    visualizer = (
        WalkingVisualizer(
            mj_model
        )
    )

    visualizer.initialize(
        robot
    )

    # ========================================================
    # LIPM ZMP VISUALIZER
    # ========================================================

    zmp_visualizer = (
        LIPMZMPVisualizer(
            config=(
                ZMP_VISUALIZATION_CONFIG
            ),

            com_height=(
                COM_HEIGHT
            ),

            gravity=(
                GRAVITY
            ),
        )
    )

    zmp_visualizer.initialize_from_states(
        x_state=(
            x_state
        ),

        y_state=(
            y_state
        ),
    )

    mj_data.qvel[:] = (
        0.0
    )

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    # ========================================================
    # INFO
    # ========================================================

    separator()

    print(
        "START LIPM-MPC + HIERARCHICAL IK WALKING"
    )

    separator()

    print(
        "Gait:"
    )

    print(
        f"  initial DS = "
        f"{INITIAL_DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"  SS         = "
        f"{SINGLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"  normal DS  = "
        f"{DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print()

    print(
        "MPC:"
    )

    print(
        f"  timestep   = "
        f"{MPC_TIMESTEP:.3f} s"
    )

    print(
        f"  horizon    = "
        f"{MPC_HORIZON_STEPS} steps "
        f"("
        f"{MPC_HORIZON_STEPS * MPC_TIMESTEP:.3f} s"
        f")"
    )

    print(
        f"  Qf         = "
        f"diag("
        f"{TERMINAL_POSITION_WEIGHT:.3f}, "
        f"{TERMINAL_VELOCITY_WEIGHT:.3f}, "
        f"{TERMINAL_ACCELERATION_WEIGHT:.3f}"
        f")"
    )

    print(
        f"  control w  = "
        f"{CONTROL_WEIGHT:.8f}"
    )

    print(
        f"  support scale = "
        f"{ZMP_SUPPORT_SCALE:.3f}"
    )

    print()

    print(
        "Executor:"
    )

    print(
        f"  IK timestep = "
        f"{DT:.6f} s"
    )

    print(
        f"  {MPC_IK_STEPS} IK steps / MPC interval"
    )

    print()

    print(
        "Single-support hierarchy:"
    )

    print(
        "  support > CoM > swing > trunk"
    )

    print()

    print(
        "Visualization:"
    )

    print(
        "  green   = left-foot trail"
    )

    print(
        "  red     = right-foot trail"
    )

    print(
        "  blue    = actual CoM trail"
    )

    print(
        "  orange  = current support polygon"
    )

    print(
        "  magenta sphere = current LIPM ZMP"
    )

    print(
        "  magenta trail  = LIPM ZMP history"
    )

    print(
        "  violet line/dots = future MPC ZMP preview"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  Displayed ZMP is generated by the LIPM."
    )

    print(
        "  It is NOT MuJoCo contact COP."
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "  After settling, execution is kinematic."
    )

    print(
        "  This test validates MPC -> CoM -> IK integration."
    )

    print(
        "  It is NOT yet a rigid-body dynamic walking test."
    )

    print()

    print(
        "Press F -> graceful stop."
    )

    print(
        "Close viewer -> exit."
    )

    # ========================================================
    # LOOP DATA
    # ========================================================

    kinematic_time = 0.0

    iteration = 0

    completed_steps = 0

    previous_step_index = -1

    stop_forwarded_to_fsm = False

    finished_announced = False

    # --------------------------------------------------------
    # MPC interval state
    # --------------------------------------------------------

    current_segment = None

    current_mpc_info = None

    mpc_substep = 0

    mpc_solve_count = 0

    previous_x_control = None

    previous_y_control = None

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    max_abs_x_jerk = 0.0

    max_abs_y_jerk = 0.0

    max_mpc_solve_wall_time = 0.0

    total_mpc_solve_wall_time = 0.0

    max_com_tracking_error = 0.0

    max_support_tracking_error = 0.0

    max_swing_tracking_error = 0.0

    # ========================================================
    # VIEWER
    # ========================================================

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        show_right_ui=True,
        key_callback=(
            keyboard_callback
        ),
    ) as viewer:

        visualizer.configure_viewer(
            viewer
        )

        zmp_visualizer.update_viewer(
            walking_visualizer=(
                visualizer
            ),

            viewer=(
                viewer
            ),

            robot=(
                robot
            ),

            state=(
                fsm.get_state()
            ),
        )

        wall_start = (
            time.perf_counter()
        )

        while viewer.is_running():

            # =================================================
            # FORWARD STOP REQUEST
            # =================================================

            if (
                stop_requested.is_set()
                and
                not stop_forwarded_to_fsm
            ):

                fsm.request_stop()

                stop_forwarded_to_fsm = (
                    True
                )

                print()

                print(
                    "Stop request forwarded to WalkingFSM."
                )

            # =================================================
            # CURRENT STATE
            # =================================================

            state = (
                fsm.get_state()
            )

            # =================================================
            # FINISHED
            # =================================================

            if state.finished:

                if not finished_announced:

                    separator()

                    print(
                        "GRACEFUL STOP COMPLETED"
                    )

                    separator()

                    print(
                        "FSM state = FINISHED"
                    )

                    print(
                        "Robot is holding final pose."
                    )

                    print(
                        "Close viewer to exit."
                    )

                    finished_announced = (
                        True
                    )

                zmp_visualizer.update_viewer(
                    walking_visualizer=(
                        visualizer
                    ),

                    viewer=(
                        viewer
                    ),

                    robot=(
                        robot
                    ),

                    state=(
                        state
                    ),
                )

                time.sleep(
                    0.03
                )

                continue

            # =================================================
            # NEW MPC INTERVAL
            # =================================================

            if current_segment is None:

                current_mpc_info = (
                    solve_mpc_segment(
                        fsm=(
                            fsm
                        ),

                        left_rotation=(
                            R_left_0
                        ),

                        right_rotation=(
                            R_right_0
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
                    )
                )

                current_segment = (
                    current_mpc_info[
                        "segment"
                    ]
                )

                x_result = (
                    current_mpc_info[
                        "x_result"
                    ]
                )

                y_result = (
                    current_mpc_info[
                        "y_result"
                    ]
                )

                # =============================================
                # UPDATE FUTURE ZMP VISUAL
                # =============================================

                zmp_visualizer.update_preview(
                    x_result=(
                        x_result
                    ),

                    y_result=(
                        y_result
                    ),
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

                max_abs_x_jerk = max(
                    max_abs_x_jerk,
                    abs(
                        x_result.first_control
                    ),
                )

                max_abs_y_jerk = max(
                    max_abs_y_jerk,
                    abs(
                        y_result.first_control
                    ),
                )

                solve_wall_time = (
                    current_mpc_info[
                        "solve_wall_time"
                    ]
                )

                total_mpc_solve_wall_time += (
                    solve_wall_time
                )

                max_mpc_solve_wall_time = max(
                    max_mpc_solve_wall_time,
                    solve_wall_time,
                )

                if (
                    mpc_solve_count
                    ==
                    1
                    or
                    mpc_solve_count
                    %
                    MPC_PRINT_EVERY
                    ==
                    0
                ):

                    preview = (
                        current_mpc_info[
                            "preview"
                        ]
                    )

                    print()

                    print(
                        f"MPC #{mpc_solve_count:04d}"
                        f" | t={kinematic_time:.3f} s"
                        f" | phase={phase_name(state.phase)}"
                        f" | support={state.support_side}"
                    )

                    print(
                        f"  jerk = "
                        f"("
                        f"{x_result.first_control:+.4f}, "
                        f"{y_result.first_control:+.4f}"
                        f") m/s^3"
                    )

                    print(
                        f"  terminal goal = "
                        f"("
                        f"{current_mpc_info['x_goal'][0]:+.4f}, "
                        f"{current_mpc_info['y_goal'][0]:+.4f}"
                        f") m"
                    )

                    print(
                        f"  horizon end = "
                        f"{preview.phase[-1]}"
                        f" / "
                        f"{preview.support_side[-1]}"
                    )

                    print(
                        f"  solver iterations = "
                        f"x:{x_result.iterations}, "
                        f"y:{y_result.iterations}"
                    )

                    print(
                        f"  solve wall time = "
                        f"{solve_wall_time * 1000.0:.2f} ms"
                    )

            # =================================================
            # REGISTER CURRENT PLANNED STEP
            # =================================================

            visualizer.register_step(
                state,
                robot,
            )

            # =================================================
            # NEW STEP MESSAGE
            # =================================================

            if (
                state.phase
                ==
                WalkingPhase.SINGLE_SUPPORT
                and
                state.step_index
                !=
                previous_step_index
            ):

                previous_step_index = (
                    state.step_index
                )

                separator()

                print(
                    f"START STEP "
                    f"#{state.step_index}"
                )

                print(
                    f"type    = "
                    f"{state.step_type.value}"
                )

                print(
                    f"support = "
                    f"{state.support_side}"
                )

                print(
                    f"swing   = "
                    f"{state.swing_side}"
                )

                print(
                    "start   =",
                    state.swing_start,
                )

                print(
                    "target  =",
                    state.swing_target,
                )

            # =================================================
            # CONTINUOUS COM REFERENCE
            # =================================================

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

            p_com_ref = (
                com_reference
                .position
                .copy()
            )

            v_com_ref = (
                com_reference
                .velocity
                .copy()
            )

            # -------------------------------------------------
            # Reference after this executor interval
            # -------------------------------------------------

            tau_next = min(
                tau
                +
                DT,
                MPC_TIMESTEP,
            )

            com_reference_next = (
                current_segment.evaluate(
                    tau_next
                )
            )

            # =================================================
            # UPDATE CURRENT LIPM ZMP
            # =================================================

            current_zmp_xy = (
                zmp_visualizer.update_current(
                    com_position=(
                        com_reference_next
                        .position
                    ),

                    com_acceleration=(
                        com_reference_next
                        .acceleration
                    ),
                )
            )

            # =================================================
            # FOOT REFERENCES
            # =================================================

            (
                p_left_ref,
                p_right_ref,
                v_swing_ref,
            ) = compute_foot_references(
                state
            )

            # =================================================
            # IK
            # =================================================

            (
                qdot_full,
                diagnostics,
                Z,
            ) = solve_current_phase(
                robot=(
                    robot
                ),

                q_pin=(
                    q_pin
                ),

                state=(
                    state
                ),

                p_left_ref=(
                    p_left_ref
                ),

                p_right_ref=(
                    p_right_ref
                ),

                p_com_ref=(
                    p_com_ref
                ),

                v_com_ref=(
                    v_com_ref
                ),

                v_swing_ref=(
                    v_swing_ref
                ),

                trunk_rotation_ref=(
                    trunk_rotation_ref
                ),
            )

            if not np.all(
                np.isfinite(
                    qdot_full
                )
            ):

                raise RuntimeError(
                    "Differential IK produced NaN/Inf."
                )

            # =================================================
            # INTEGRATE
            # =================================================

            q_pin = robot.integrate(
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

            # =================================================
            # TRACKING ERROR
            # =================================================

            p_com_actual = (
                robot.get_com()
            )

            com_tracking_error = float(
                np.linalg.norm(
                    com_reference_next.position
                    -
                    p_com_actual
                )
            )

            max_com_tracking_error = max(
                max_com_tracking_error,
                com_tracking_error,
            )

            if (
                state.phase
                ==
                WalkingPhase.SINGLE_SUPPORT
            ):

                p_left_actual, _ = (
                    robot.get_left_foot_pose()
                )

                p_right_actual, _ = (
                    robot.get_right_foot_pose()
                )

                if (
                    state.support_side
                    ==
                    "left"
                ):

                    support_error_now = float(
                        np.linalg.norm(
                            p_left_ref
                            -
                            p_left_actual
                        )
                    )

                    swing_error_now = float(
                        np.linalg.norm(
                            p_right_ref
                            -
                            p_right_actual
                        )
                    )

                else:

                    support_error_now = float(
                        np.linalg.norm(
                            p_right_ref
                            -
                            p_right_actual
                        )
                    )

                    swing_error_now = float(
                        np.linalg.norm(
                            p_left_ref
                            -
                            p_left_actual
                        )
                    )

                max_support_tracking_error = max(
                    max_support_tracking_error,
                    support_error_now,
                )

                max_swing_tracking_error = max(
                    max_swing_tracking_error,
                    swing_error_now,
                )

            # =================================================
            # FSM
            # =================================================

            phase_before = (
                state.phase
            )

            state_after = (
                fsm.update(
                    DT
                )
            )

            # =================================================
            # LANDING
            # =================================================

            if (
                phase_before
                ==
                WalkingPhase.SINGLE_SUPPORT
                and
                state_after.phase
                !=
                WalkingPhase.SINGLE_SUPPORT
            ):

                completed_steps += 1

                p_left_actual, _ = (
                    robot.get_left_foot_pose()
                )

                p_right_actual, _ = (
                    robot.get_right_foot_pose()
                )

                p_com_actual = (
                    robot.get_com()
                )

                # ---------------------------------------------
                # CURRENT SWING / SUPPORT FOOT
                # ---------------------------------------------

                if (
                    state.swing_side
                    ==
                    "left"
                ):

                    swing_actual = (
                        p_left_actual
                    )

                    support_actual = (
                        p_right_actual
                    )

                    support_target = (
                        state
                        .right_contact_position
                    )

                else:

                    swing_actual = (
                        p_right_actual
                    )

                    support_actual = (
                        p_left_actual
                    )

                    support_target = (
                        state
                        .left_contact_position
                    )

                # ---------------------------------------------
                # ERRORS
                # ---------------------------------------------

                landing_error = float(
                    np.linalg.norm(
                        state.swing_target
                        -
                        swing_actual
                    )
                )

                support_error = float(
                    np.linalg.norm(
                        support_target
                        -
                        support_actual
                    )
                )

                com_error = float(
                    np.linalg.norm(
                        com_reference_next.position
                        -
                        p_com_actual
                    )
                )

                # ---------------------------------------------
                # TRUNK
                # ---------------------------------------------

                trunk_pitch = (
                    get_mujoco_body_pitch(
                        mj_model,
                        mj_data,
                        TRUNK_BODY_NAME,
                    )
                )

                trunk_pitch_deg = float(
                    np.degrees(
                        trunk_pitch
                    )
                )

                trunk_pitch_drift_deg = float(
                    np.degrees(
                        trunk_pitch
                        -
                        initial_trunk_pitch
                    )
                )

                trunk_local_y_error = (
                    get_trunk_local_y_error(
                        robot,
                        trunk_rotation_ref,
                    )
                )

                trunk_local_y_error_deg = float(
                    np.degrees(
                        trunk_local_y_error
                    )
                )

                final_nullity = int(
                    Z.shape[
                        1
                    ]
                )

                # ---------------------------------------------
                # PRINT
                # ---------------------------------------------

                separator()

                print(
                    f"LANDING STEP "
                    f"#{state.step_index}"
                )

                print(
                    f"type = "
                    f"{state.step_type.value}"
                )

                print(
                    f"landing error = "
                    f"{landing_error:.6e} m"
                )

                print(
                    f"support error = "
                    f"{support_error:.6e} m"
                )

                print(
                    f"MPC CoM tracking error = "
                    f"{com_error:.6e} m"
                )

                print(
                    f"LIPM ZMP = "
                    f"("
                    f"{current_zmp_xy[0]:+.6f}, "
                    f"{current_zmp_xy[1]:+.6f}"
                    f") m"
                )

                print(
                    f"trunk pitch = "
                    f"{trunk_pitch_deg:+.4f} deg"
                )

                print(
                    f"trunk pitch drift = "
                    f"{trunk_pitch_drift_deg:+.4f} deg"
                )

                print(
                    f"trunk local-Y error = "
                    f"{trunk_local_y_error_deg:+.4f} deg"
                )

                print(
                    f"final nullity = "
                    f"{final_nullity}"
                )

            # =================================================
            # END OF MPC INTERVAL
            # =================================================

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

                current_mpc_info = None

                mpc_substep = 0

            # =================================================
            # TIME
            # =================================================

            iteration += 1

            kinematic_time += (
                DT
            )

            mj_data.time = (
                settle_info[
                    "time"
                ]
                +
                kinematic_time
            )

            # =================================================
            # VISUAL / REAL-TIME
            # =================================================

            if (
                iteration
                %
                VIEWER_SYNC_STEPS
                ==
                0
            ):

                zmp_visualizer.update_viewer(
                    walking_visualizer=(
                        visualizer
                    ),

                    viewer=(
                        viewer
                    ),

                    robot=(
                        robot
                    ),

                    state=(
                        state_after
                    ),
                )

                target_wall_time = (
                    wall_start
                    +
                    kinematic_time
                )

                remaining_time = (
                    target_wall_time
                    -
                    time.perf_counter()
                )

                if (
                    remaining_time
                    >
                    0.0
                ):

                    time.sleep(
                        remaining_time
                    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    separator()

    print(
        "PROGRAM ENDED"
    )

    separator()

    robot.update(
        q_pin
    )

    p_left_final, _ = (
        robot.get_left_foot_pose()
    )

    p_right_final, _ = (
        robot.get_right_foot_pose()
    )

    p_com_final = (
        robot.get_com()
    )

    final_trunk_pitch = (
        get_mujoco_body_pitch(
            mj_model,
            mj_data,
            TRUNK_BODY_NAME,
        )
    )

    final_trunk_pitch_deg = float(
        np.degrees(
            final_trunk_pitch
        )
    )

    final_trunk_pitch_drift_deg = float(
        np.degrees(
            final_trunk_pitch
            -
            initial_trunk_pitch
        )
    )

    # --------------------------------------------------------
    # LIPM state at actual stop time
    # --------------------------------------------------------

    if (
        current_segment
        is not None
    ):

        planner_tau = min(
            mpc_substep
            *
            DT,
            MPC_TIMESTEP,
        )

        planner_x_state = (
            current_segment.get_x_state(
                planner_tau
            )
        )

        planner_y_state = (
            current_segment.get_y_state(
                planner_tau
            )
        )

    else:

        planner_x_state = (
            x_state.copy()
        )

        planner_y_state = (
            y_state.copy()
        )

    final_lipm_zmp = (
        zmp_visualizer.compute_zmp_from_states(
            planner_x_state,
            planner_y_state,
        )
    )

    mean_mpc_solve_wall_time = (
        total_mpc_solve_wall_time
        /
        max(
            mpc_solve_count,
            1,
        )
    )

    # ========================================================
    # PRINT FINAL
    # ========================================================

    print(
        f"kinematic walking time = "
        f"{kinematic_time:.3f} s"
    )

    print(
        f"completed swing steps = "
        f"{completed_steps}"
    )

    print(
        f"MPC solves = "
        f"{mpc_solve_count}"
    )

    print()

    print(
        "LEFT foot final  =",
        p_left_final,
    )

    print(
        "RIGHT foot final =",
        p_right_final,
    )

    print(
        "CoM actual final =",
        p_com_final,
    )

    print(
        "LIPM X state      =",
        planner_x_state,
    )

    print(
        "LIPM Y state      =",
        planner_y_state,
    )

    print(
        "LIPM ZMP final    =",
        final_lipm_zmp,
    )

    print()

    print(
        f"max |x jerk| applied = "
        f"{max_abs_x_jerk:.6f} m/s^3"
    )

    print(
        f"max |y jerk| applied = "
        f"{max_abs_y_jerk:.6f} m/s^3"
    )

    print(
        f"max CoM tracking error = "
        f"{max_com_tracking_error:.6e} m"
    )

    print(
        f"max support tracking error = "
        f"{max_support_tracking_error:.6e} m"
    )

    print(
        f"max swing tracking error = "
        f"{max_swing_tracking_error:.6e} m"
    )

    print()

    print(
        f"mean MPC solve wall time = "
        f"{mean_mpc_solve_wall_time * 1000.0:.2f} ms"
    )

    print(
        f"max MPC solve wall time = "
        f"{max_mpc_solve_wall_time * 1000.0:.2f} ms"
    )

    print()

    print(
        f"initial trunk pitch = "
        f"{initial_trunk_pitch_deg:+.4f} deg"
    )

    print(
        f"final trunk pitch   = "
        f"{final_trunk_pitch_deg:+.4f} deg"
    )

    print(
        f"trunk pitch drift   = "
        f"{final_trunk_pitch_drift_deg:+.4f} deg"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()