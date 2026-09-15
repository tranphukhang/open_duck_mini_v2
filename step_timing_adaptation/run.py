# step_timing_adaptation/run.py

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

    sys.path.insert(
        0,
        str(ROOT_DIR),
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

from lipm_mpc.zmp_visualization import (
    ZMPVisualizationConfig,
    LIPMZMPVisualizer,
)

# IMPORTANT:
#
# Exactly reuse the settling procedure used by lipm_mpc.
#
# Settling uses MuJoCo dynamics ONLY before kinematic walking.
#
from lipm_mpc.run import (
    settle_robot,
)


# ============================================================
# REUSE WHOLE-BODY KINEMATICS / VISUALIZATION
# ============================================================

from footstep_planning.pinocchio_model import (
    PinocchioModel,
)

from footstep_planning.differential_ik import (
    TRUNK_FRAME,
    solve_single_support_ik,
    solve_double_support_ik,
)

from footstep_planning.walking_fsm import (
    WalkingPhase,
)

from footstep_planning.walking_visualization import (
    WalkingVisualizer,
    PlannedFootstep,
    MAX_PLANNED_FOOTSTEPS,
    add_sphere,
    add_line,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# WALKING
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

INITIAL_DOUBLE_SUPPORT_DURATION = 0.36

DOUBLE_SUPPORT_DURATION = 0.27


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

SINGLE_SUPPORT_ZMP_HALF_WIDTH = 0.0005


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
# VIEWER
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
# ZMP VISUALIZATION
# ============================================================

ZMP_VISUALIZATION_CONFIG = (
    ZMPVisualizationConfig(

        show_current=True,

        show_trail=True,

        show_preview=True,

        current_z=0.010,

        trail_z=0.008,

        preview_z=0.007,

        current_radius=0.007,

        preview_radius=0.0028,

        trail_width=5.0,

        preview_width=2.5,

        trail_min_distance=5.0e-4,

        trail_max_points=180,

        preview_point_stride=2,

        # Current ZMP
        current_rgba=np.array(
            [
                1.00,
                0.00,
                1.00,
                1.00,
            ],
            dtype=np.float32,
        ),

        # ZMP trail
        trail_rgba=np.array(
            [
                0.90,
                0.10,
                0.95,
                0.80,
            ],
            dtype=np.float32,
        ),

        # MPC preview points
        preview_rgba=np.array(
            [
                0.72,
                0.28,
                1.00,
                0.72,
            ],
            dtype=np.float32,
        ),

        # MPC preview line
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
# ADAPTIVE VISUALIZATION
# ============================================================

COM_GROUND_Z = 0.012

COM_GROUND_RADIUS = 0.006

COM_GROUND_RGBA = np.array(
    [
        0.05,
        0.40,
        1.00,
        1.00,
    ],
    dtype=np.float32,
)


DCM_GROUND_Z = 0.014

DCM_RADIUS = 0.007

DCM_RGBA = np.array(
    [
        1.00,
        0.90,
        0.00,
        1.00,
    ],
    dtype=np.float32,
)


PLANNER_TARGET_Z = 0.016

PLANNER_TARGET_RADIUS = 0.006

PLANNER_TARGET_RGBA = np.array(
    [
        0.00,
        1.00,
        0.85,
        1.00,
    ],
    dtype=np.float32,
)


COM_VERTICAL_LINE_WIDTH = 2.5

COM_VERTICAL_RGBA = np.array(
    [
        0.10,
        0.40,
        1.00,
        0.55,
    ],
    dtype=np.float32,
)


COM_DCM_LINE_WIDTH = 3.0

COM_DCM_LINE_RGBA = np.array(
    [
        1.00,
        0.80,
        0.05,
        0.70,
    ],
    dtype=np.float32,
)


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
# VISUAL STATE ADAPTER
# ============================================================

@dataclass
class VisualWalkingState:

    phase: WalkingPhase

    support_side: str | None


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

        return left_contact

    if side == "right":

        return right_contact

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

    # ========================================================
    # KINEMATIC SET-STATE
    # ========================================================

    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def make_visual_state(
    phase,
    support_side,
):

    if phase == INITIAL_DOUBLE_SUPPORT:

        walking_phase = (
            WalkingPhase.INITIAL_DOUBLE_SUPPORT
        )

    elif phase == SINGLE_SUPPORT:

        walking_phase = (
            WalkingPhase.SINGLE_SUPPORT
        )

    elif phase == DOUBLE_SUPPORT:

        walking_phase = (
            WalkingPhase.DOUBLE_SUPPORT
        )

    else:

        raise ValueError(
            f"Unsupported visual phase: {phase}"
        )

    return VisualWalkingState(
        phase=(
            walking_phase
        ),

        support_side=(
            support_side
        ),
    )


def append_planned_footstep(
    visualizer,
    *,
    side,
    position,
    rotation,
):

    footstep = PlannedFootstep(
        side=(
            side
        ),

        position=np.asarray(
            position,
            dtype=float,
        ).reshape(
            3
        ).copy(),

        rotation=np.asarray(
            rotation,
            dtype=float,
        ).reshape(
            3,
            3,
        ).copy(),
    )

    visualizer.planned_footsteps.append(
        footstep
    )

    if (
        len(
            visualizer.planned_footsteps
        )
        >
        MAX_PLANNED_FOOTSTEPS
    ):

        remove_count = (
            len(
                visualizer.planned_footsteps
            )
            -
            MAX_PLANNED_FOOTSTEPS
        )

        del visualizer.planned_footsteps[
            0:remove_count
        ]


def draw_adaptive_overlay(
    *,
    viewer,
    walking_visualizer,

    com_world,
    dcm_xy,

    phase,
    swing_side,

    landing_position,

    left_rotation,
    right_rotation,
):

    with viewer.lock():

        scene = (
            viewer.user_scn
        )

        # ====================================================
        # COM PROJECTION ON GROUND
        # ====================================================

        com_ground = np.array(
            [
                com_world[0],
                com_world[1],
                COM_GROUND_Z,
            ],
            dtype=float,
        )

        add_sphere(
            scene,
            com_ground,
            COM_GROUND_RADIUS,
            COM_GROUND_RGBA,
        )

        # ====================================================
        # VERTICAL COM PROJECTION
        # ====================================================

        add_line(
            scene,
            com_world,
            com_ground,
            COM_VERTICAL_LINE_WIDTH,
            COM_VERTICAL_RGBA,
        )

        # ====================================================
        # DCM
        # ====================================================

        dcm_ground = np.array(
            [
                dcm_xy[0],
                dcm_xy[1],
                DCM_GROUND_Z,
            ],
            dtype=float,
        )

        add_sphere(
            scene,
            dcm_ground,
            DCM_RADIUS,
            DCM_RGBA,
        )

        # ====================================================
        # COM -> DCM
        # ====================================================

        add_line(
            scene,
            com_ground,
            dcm_ground,
            COM_DCM_LINE_WIDTH,
            COM_DCM_LINE_RGBA,
        )

        # ====================================================
        # CURRENT ADAPTIVE FOOTHOLD
        # ====================================================

        if phase == SINGLE_SUPPORT:

            if swing_side == "left":

                target_rotation = (
                    left_rotation
                )

            else:

                target_rotation = (
                    right_rotation
                )

            current_target = PlannedFootstep(
                side=(
                    swing_side
                ),

                position=(
                    landing_position.copy()
                ),

                rotation=(
                    target_rotation.copy()
                ),
            )

            # Existing footprint renderer
            walking_visualizer.draw_planned_footprint(
                scene,
                current_target,
            )

            # Exact planner uT marker
            target_marker = np.array(
                [
                    landing_position[0],
                    landing_position[1],
                    PLANNER_TARGET_Z,
                ],
                dtype=float,
            )

            add_sphere(
                scene,
                target_marker,
                PLANNER_TARGET_RADIUS,
                PLANNER_TARGET_RGBA,
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

    # ========================================================
    # IMPORTANT
    #
    # Copy the current settled swing-foot position first.
    #
    # Therefore target z is exactly the settled ground
    # contact height, not a hardcoded HOME-frame z.
    # ========================================================

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
    # TERMINAL GOALS
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
# WALK
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
    # IMPORTANT
    #
    # At this point MuJoCo has ALREADY been settled.
    #
    # Therefore mj_data.qpos is the settled configuration,
    # not the raw HOME keyframe.
    # ========================================================

    q_pin = (
        robot.mujoco_to_pin(
            mj_data.qpos.copy()
        )
    )

    robot.update(
        q_pin
    )

    # ========================================================
    # INITIAL SETTLED FOOT POSES
    # ========================================================

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
    # COM WORLD HEIGHT
    #
    # LIPM h is RELATIVE height above support plane.
    #
    # World-z:
    #
    #     z_G = z_ground + h
    # ========================================================

    support_plane_z = (
        0.5
        *
        (
            left_contact_position[2]
            +
            right_contact_position[2]
        )
    )

    com_world_z_reference = (
        support_plane_z
        +
        COM_HEIGHT
    )

    initial_relative_com_height = (
        initial_com[2]
        -
        support_plane_z
    )

    # ========================================================
    # NOMINAL STEP
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
    # GAIT STATE
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
    # VISUALIZATION
    # ========================================================

    walking_visualizer = None
    zmp_visualizer = None

    if viewer is not None:

        walking_visualizer = (
            WalkingVisualizer(
                mj_model
            )
        )

        walking_visualizer.initialize(
            robot
        )

        walking_visualizer.configure_viewer(
            viewer
        )

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

    # ========================================================
    # SWING
    # ========================================================

    swing_trajectory = (
        OnlineQuinticSwingTrajectory()
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

    step_index = 0

    planner_result = None

    planner_frozen = False

    disturbance_applied = False

    stop_requested = False

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
        f"Scene         = "
        f"{SCENE_XML.name}"
    )

    print(
        f"walk duration = "
        f"{WALK_DURATION:.3f} s"
    )

    print(
        f"desired vx    = "
        f"{DESIRED_VELOCITY_X:.3f} m/s"
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
        f"MPC horizon   = "
        f"{MPC_TIMESTEP * MPC_HORIZON_STEPS:.3f} s"
    )

    print()

    print(
        "Settled initial state:"
    )

    print(
        f"  LEFT  = "
        f"{left_contact_position}"
    )

    print(
        f"  RIGHT = "
        f"{right_contact_position}"
    )

    print(
        f"  CoM   = "
        f"{initial_com}"
    )

    print(
        f"  support plane z = "
        f"{support_plane_z:+.6f} m"
    )

    print(
        f"  CoM height above support = "
        f"{initial_relative_com_height:.6f} m"
    )

    print(
        f"  LIPM height = "
        f"{COM_HEIGHT:.6f} m"
    )

    print(
        f"  CoM world-z reference = "
        f"{com_world_z_reference:.6f} m"
    )

    print()

    print(
        "Visualization:"
    )

    print(
        "  green trail      = LEFT foot"
    )

    print(
        "  red trail        = RIGHT foot"
    )

    print(
        "  blue 3D trail    = actual CoM"
    )

    print(
        "  orange polygon   = support polygon"
    )

    print(
        "  cyan footprint   = adaptive planner foothold uT"
    )

    print(
        "  blue ground dot  = LIPM CoM projection"
    )

    print(
        "  yellow dot       = DCM"
    )

    print(
        "  yellow line      = CoM -> DCM"
    )

    print(
        "  magenta          = current LIPM ZMP"
    )

    print(
        "  magenta trail    = ZMP history"
    )

    print(
        "  violet line/dots = future MPC ZMP"
    )

    print()

    print(
        "Walking execution after settling is kinematic."
    )

    print(
        "No mj_step() is used during walking."
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

        # ====================================================
        # WALK DURATION
        # ====================================================

        if (
            kinematic_time
            >=
            WALK_DURATION
        ):

            stop_requested = True

        # ====================================================
        # TOUCHDOWN
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

            current_segment = None

            mpc_substep = 0

            previous_x_control = None

            previous_y_control = None

            robot.update(
                q_pin
            )

            if swing_side == "left":

                (
                    actual_landing,
                    actual_rotation,
                ) = (
                    robot.get_left_foot_pose()
                )

                left_contact_position = (
                    landing_position.copy()
                )

                left_rotation = (
                    actual_rotation.copy()
                )

                landing_rotation = (
                    left_rotation.copy()
                )

            else:

                (
                    actual_landing,
                    actual_rotation,
                ) = (
                    robot.get_right_foot_pose()
                )

                right_contact_position = (
                    landing_position.copy()
                )

                right_rotation = (
                    actual_rotation.copy()
                )

                landing_rotation = (
                    right_rotation.copy()
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

            # ------------------------------------------------
            # Keep finished adaptive footsteps in GUI.
            # ------------------------------------------------

            if walking_visualizer is not None:

                append_planned_footstep(
                    walking_visualizer,

                    side=(
                        swing_side
                    ),

                    position=(
                        landing_position
                    ),

                    rotation=(
                        landing_rotation
                    ),
                )

            print()

            print(
                f"TOUCHDOWN STEP {step_index}"
                f" | {swing_side.upper()} foot"
                f" | target z="
                f"{landing_position[2]:+.6f} m"
                f" | actual z="
                f"{actual_landing[2]:+.6f} m"
                f" | error="
                f"{1000.0 * last_landing_error:.3f} mm"
            )

            # =================================================
            # NEXT STANCE
            # =================================================

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
                stance_side == "left"
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

            phase = (
                DOUBLE_SUPPORT
            )

            phase_time = 0.0

            swing_initialized = False

            planner_result = None

            planner_frozen = False

            disturbance_applied = False

        # ====================================================
        # DS -> SS
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

            current_segment = None

            mpc_substep = 0

            previous_x_control = None

            previous_y_control = None

            phase = (
                SINGLE_SUPPORT
            )

            phase_time = 0.0

            step_index += 1

            current_nominal_step = (
                nominal_left_step
                if
                stance_side == "left"
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

            # =================================================
            # SWING FOOT AT LIFT-OFF
            # =================================================

            robot.update(
                q_pin
            )

            if swing_side == "left":

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

            # =================================================
            # ADAPTIVE LANDING TARGET
            #
            # x/y from planner.
            #
            # z MUST remain the settled contact-plane height.
            # =================================================

            landing_position = (
                swing_start.copy()
            )

            landing_position[0] = (
                planner_result.step_location_x
            )

            landing_position[1] = (
                planner_result.step_location_y
            )

            # Keep ground-contact z.
            #
            # On flat terrain the previous contact z is the
            # desired next touchdown z.
            #
            if swing_side == "left":

                landing_position[2] = (
                    left_contact_position[2]
                )

            else:

                landing_position[2] = (
                    right_contact_position[2]
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

            # =================================================
            # ONLINE C2 SWING
            # =================================================

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

            swing_initialized = True

            planner_frozen = False

            disturbance_applied = False

            print()

            separator()

            print(
                f"STEP {step_index}"
                f" | {stance_side.upper()} support"
                f" | {swing_side.upper()} swing"
            )

            separator()

            print(
                f"DCM         = "
                f"{dcm}"
            )

            print(
                f"swing start = "
                f"{swing_start}"
            )

            print(
                f"landing z   = "
                f"{landing_position[2]:+.6f} m"
            )

            print(
                f"nominal T   = "
                f"{current_nominal_step.step_time:.6f} s"
            )

            print(
                f"adapted T   = "
                f"{current_step_time:.6f} s"
            )

            print(
                f"uT          = "
                f"{landing_position}"
            )

            print()

        # ====================================================
        # GRACEFUL FINISH
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
        # CURRENT LIPM STATE
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

            current_segment = None

            mpc_substep = 0

            previous_x_control = None

            previous_y_control = None

            disturbance_applied = True

            current_x_state = (
                x_state.copy()
            )

            current_y_state = (
                y_state.copy()
            )

            print(
                f"DISTURBANCE STEP {step_index}"
                f" | tSS={phase_time:.3f}"
                f" | Delta xi="
                f"({DISTURBANCE_DCM_X:+.3f},"
                f"{DISTURBANCE_DCM_Y:+.3f}) m"
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

                    # Planner changes x/y only.
                    #
                    # z remains fixed at the settled flat-ground
                    # contact height.
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

            # =================================================
            # ZMP PREVIEW
            # =================================================

            if zmp_visualizer is not None:

                zmp_visualizer.update_preview(
                    x_result=(
                        x_result
                    ),

                    y_result=(
                        y_result
                    ),
                )

            previous_x_control = (
                x_result.control.copy()
            )

            previous_y_control = (
                y_result.control.copy()
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
        # COM REFERENCE
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

        # IMPORTANT:
        #
        # Constant relative LIPM height converted to world z.
        #
        com_position_ref[2] = (
            com_world_z_reference
        )

        com_velocity_ref[2] = 0.0

        # ====================================================
        # CURRENT ZMP
        # ====================================================

        if zmp_visualizer is not None:

            zmp_visualizer.update_current(
                com_position=(
                    com_position_ref
                ),

                com_acceleration=(
                    com_reference.acceleration
                ),
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

            v_swing_ref = None

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

            v_swing_ref = (
                swing_sample.velocity.copy()
            )

        # ====================================================
        # IK
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
                "Differential IK returned NaN/Inf."
            )

        # ====================================================
        # PINOCCHIO INTEGRATION
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

        # ====================================================
        # MUJOCO SET-STATE
        #
        # NO DYNAMICS DURING WALKING.
        # ====================================================

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
        # ACTUAL STATES
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

        # ====================================================
        # ERRORS
        # ====================================================

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

            swing_error = 0.0

        else:

            if stance_side == "left":

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

            if phase == SINGLE_SUPPORT:

                swing_ref_z = (
                    p_left_ref[2]
                    if
                    swing_side == "left"
                    else
                    p_right_ref[2]
                )

                print(
                    f"t={kinematic_time:6.3f}"
                    f" | STEP={step_index:02d}"
                    f" | SS {stance_side.upper()}"
                    f" | tSS={phase_time:.3f}"
                    f" | T={current_step_time:.3f}"
                    f" | uT="
                    f"({landing_position[0]:+.3f},"
                    f"{landing_position[1]:+.3f})"
                    f" | zSwing={swing_ref_z:+.4f}"
                    f" | DCM="
                    f"({dcm[0]:+.3f},"
                    f"{dcm[1]:+.3f})"
                    f" | CoM err="
                    f"{1000.0 * com_error:.2f} mm"
                    f" | swing err="
                    f"{1000.0 * swing_error:.2f} mm"
                )

            else:

                print(
                    f"t={kinematic_time:6.3f}"
                    f" | {phase}"
                    f" | DCM="
                    f"({dcm[0]:+.3f},"
                    f"{dcm[1]:+.3f})"
                    f" | CoM err="
                    f"{1000.0 * com_error:.2f} mm"
                    f" | feet err="
                    f"{1000.0 * support_error:.2f} mm"
                )

            next_print_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # ADVANCE MPC
        # ====================================================

        mpc_substep += 1

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

            current_segment = None

            mpc_substep = 0

        # ====================================================
        # LOGICAL TIME
        # ====================================================

        phase_time += DT

        kinematic_time += DT

        # We are no longer using MuJoCo dynamic time after
        # settling. Use logical walking time in the GUI.
        mj_data.time = (
            kinematic_time
        )

        # ====================================================
        # GUI
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

            visual_support_side = (
                stance_side
                if
                phase == SINGLE_SUPPORT
                else
                None
            )

            visual_state = (
                make_visual_state(
                    phase,
                    visual_support_side,
                )
            )

            # ------------------------------------------------
            # Existing walking visualization.
            # ------------------------------------------------

            walking_visualizer.update(
                viewer,
                robot,
                visual_state,
            )

            # ------------------------------------------------
            # Adaptive planner + DCM + CoM projection.
            # ------------------------------------------------

            draw_adaptive_overlay(
                viewer=(
                    viewer
                ),

                walking_visualizer=(
                    walking_visualizer
                ),

                com_world=(
                    com_position_ref
                ),

                dcm_xy=(
                    dcm
                ),

                phase=(
                    phase
                ),

                swing_side=(
                    swing_side
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
            )

            # ------------------------------------------------
            # Existing LIPM ZMP visualization.
            # ------------------------------------------------

            zmp_visualizer.draw_overlay(
                viewer
            )

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
    # FINAL STATE
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

    # ========================================================
    # FINAL RESULT
    # ========================================================

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
        f"Scene = "
        f"{SCENE_XML.name}"
    )

    print()

    print(
        "MuJoCo dynamics were used ONLY for initial settling."
    )

    print(
        "Walking execution remained purely kinematic."
    )

    print(
        "Dynamic walking feasibility was NOT evaluated."
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
    # INITIAL SETTLING
    #
    # IMPORTANT:
    #
    # This is intentionally the same settling routine used by
    # lipm_mpc.
    #
    # During this short initialization MuJoCo physics is used
    # so that the feet obtain the correct physical height
    # relative to the floor.
    #
    # After this point, walking is kinematic set-state only.
    # ========================================================

    separator()

    print(
        "INITIAL PHYSICS SETTLING"
    )

    separator()

    settle_info = (
        settle_robot(
            mj_model,
            mj_data,
        )
    )

    print(
        f"settled at MuJoCo t = "
        f"{settle_info['time']:.4f} s"
    )

    print(
        f"base linear speed   = "
        f"{settle_info['base_linear_speed']:.6e} m/s"
    )

    print(
        f"base angular speed  = "
        f"{settle_info['base_angular_speed']:.6e} rad/s"
    )

    print(
        f"max joint speed     = "
        f"{settle_info['max_joint_speed']:.6e} rad/s"
    )

    # Freeze dynamic velocity after settling.

    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )

    # ========================================================
    # PINOCCHIO
    #
    # IMPORTANT:
    # Pinocchio starts from the SETTLED qpos.
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
        f"Scene  = "
        f"{SCENE_XML.name}"
    )

    print(
        f"Robot  = "
        f"{ROBOT_XML.name}"
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

        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            show_right_ui=True,
        ) as viewer:

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