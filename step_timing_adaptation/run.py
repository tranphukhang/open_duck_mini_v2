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

    from .online_swing_trajectory import (
        OnlineQuinticSwingTrajectory,
    )

    from .lipm_state_generator import (
        PointFootLIPM,
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


# ============================================================
# REUSE INITIAL SETTLING / VISUALIZATION
# ============================================================

from lipm_mpc.run import (
    settle_robot,
)

from lipm_mpc.zmp_visualization import (
    ZMPVisualizationConfig,
    LIPMZMPVisualizer,
)


# ============================================================
# WHOLE-BODY KINEMATICS
# ============================================================

from footstep_planning.pinocchio_model import (
    PinocchioModel,
)

from footstep_planning.differential_ik import (
    TRUNK_FRAME,
    solve_single_support_ik,
)


# ============================================================
# WALKING VISUALIZATION
# ============================================================

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
# BASE RATE
# ============================================================

DT = 0.0005


# ============================================================
# WALKING COMMAND
# ============================================================

FIRST_STANCE_SIDE = "left"

DESIRED_VELOCITY_X = 0.10
DESIRED_VELOCITY_Y = 0.05


# ============================================================
# LIPM
# ============================================================

GRAVITY = 9.81

COM_HEIGHT = 0.2044


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


# ============================================================
# ONLINE TIMING CAUSALITY
# ============================================================

STEP_TIMING_GAP = 0.05


# ============================================================
# SWING FOOT
# ============================================================

SWING_HEIGHT = 0.04


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
# OPTIONAL DCM DISTURBANCE
# ============================================================

ENABLE_DISTURBANCE = False

DISTURBANCE_TIME = 0.10

DISTURBANCE_DCM_X = +0.010
DISTURBANCE_DCM_Y = 0.000


# ============================================================
# GUI / TERMINAL
# ============================================================

SHOW_VIEWER = True

REALTIME_PLAYBACK = True

VIEWER_SYNC_PERIOD = 0.02

STATUS_PRINT_PERIOD = 0.10


# ============================================================
# NUMERICAL
# ============================================================

TIME_TOLERANCE = 1.0e-10


# ============================================================
# ZMP VISUALIZATION
#
# Point-foot LIPM:
#
#       p_ZMP = u0 = current stance foot
#
# No MPC preview.
# ============================================================

ZMP_VISUALIZATION_CONFIG = (
    ZMPVisualizationConfig(

        show_current=True,
        show_trail=True,
        show_preview=False,

        current_z=0.010,
        trail_z=0.008,
        preview_z=0.007,

        current_radius=0.007,
        preview_radius=0.0028,

        trail_width=5.0,
        preview_width=2.5,

        trail_min_distance=5.0e-4,
        trail_max_points=300,

        preview_point_stride=2,

        current_rgba=np.array(
            [
                1.00,
                0.00,
                1.00,
                1.00,
            ],
            dtype=np.float32,
        ),

        trail_rgba=np.array(
            [
                0.90,
                0.10,
                0.95,
                0.80,
            ],
            dtype=np.float32,
        ),

        preview_rgba=np.array(
            [
                0.72,
                0.28,
                1.00,
                0.72,
            ],
            dtype=np.float32,
        ),

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
# VISUAL STATE
# ============================================================

@dataclass
class VisualWalkingState:

    phase: WalkingPhase

    support_side: str


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

    mj_data.qvel[:] = 0.0

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
# CONSISTENT NOMINAL INITIAL DCM
# ============================================================

def compute_nominal_initial_dcm(
    *,
    nominal_step,
    stance_position,
):

    """
    Construct the beginning-of-step DCM that exactly satisfies
    Stage-2 Eq. (19) at t = 0 for the nominal solution.

        uT - (xi0-u0)*tau + b = u0

    Hence:

        xi0 =
            u0
            +
            (Delta_u_nom + b_nom) / tau_nom
    """

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

    xi0 = (
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

    return xi0


# ============================================================
# CONSISTENT INITIAL COM VELOCITY
# ============================================================

def compute_consistent_initial_com_velocity(
    *,
    planner,
    com_position,
    nominal_step,
    stance_position,
):

    """
    DCM:

        xi = c + c_dot / omega

    Therefore:

        c_dot = omega * (xi - c)

    xi is initialized from the nominal periodic step relation.
    """

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
# FOOTSTEP VISUAL HISTORY
# ============================================================

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


# ============================================================
# ADAPTIVE VISUAL OVERLAY
# ============================================================

def draw_adaptive_overlay(
    *,
    viewer,
    walking_visualizer,
    com_position,
    dcm,
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
        # COM GROUND PROJECTION
        # ====================================================

        com_ground = np.array(
            [
                com_position[0],
                com_position[1],
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

        add_line(
            scene,
            com_position,
            com_ground,
            COM_VERTICAL_LINE_WIDTH,
            COM_VERTICAL_RGBA,
        )

        # ====================================================
        # DCM
        # ====================================================

        dcm_ground = np.array(
            [
                dcm[0],
                dcm[1],
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

        add_line(
            scene,
            com_ground,
            dcm_ground,
            COM_DCM_LINE_WIDTH,
            COM_DCM_LINE_RGBA,
        )

        # ====================================================
        # CURRENT ADAPTIVE uT
        # ====================================================

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

        walking_visualizer.draw_planned_footprint(
            scene,
            current_target,
        )

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
# START A NEW STEP
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

    # ========================================================
    # STEP TIMING QP AT t = 0
    # ========================================================

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

    # ========================================================
    # ACTUAL SWING FOOT AT LIFT-OFF
    # ========================================================

    if swing_side == "left":

        (
            swing_start,
            _,
        ) = (
            robot.get_left_foot_pose()
        )

        landing_z = (
            left_contact_position[2]
        )

    else:

        (
            swing_start,
            _,
        ) = (
            robot.get_right_foot_pose()
        )

        landing_z = (
            right_contact_position[2]
        )

    # ========================================================
    # LANDING TARGET
    # ========================================================

    landing_position = (
        swing_start.copy()
    )

    landing_position[0] = (
        planner_result.step_location_x
    )

    landing_position[1] = (
        planner_result.step_location_y
    )

    # Flat terrain.
    landing_position[2] = (
        landing_z
    )

    # ========================================================
    # RESET ONLINE SWING
    # ========================================================

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
# RUN WALK
# ============================================================

def run_walk(
    *,
    mj_model,
    mj_data,
    robot,
    planner,
    measured_step_width,
    viewer,
):

    # ========================================================
    # SETTLED CONFIGURATION
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

    left_rotation = (
        left_rotation.copy()
    )

    right_rotation = (
        right_rotation.copy()
    )

    initial_com_actual = (
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
    # LIPM WORLD HEIGHT
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

    com_world_z = (
        support_plane_z
        +
        COM_HEIGHT
    )

    initial_com_reference = (
        initial_com_actual.copy()
    )

    initial_com_reference[2] = (
        com_world_z
    )

    # ========================================================
    # NOMINAL GAIT
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
    # CONSISTENT INITIAL DCM + CoM VELOCITY
    #
    # THIS IS THE IMPORTANT FIX.
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

    # ========================================================
    # VERIFY INITIAL DCM
    # ========================================================

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

    if initial_dcm_error > 1.0e-10:

        raise RuntimeError(
            "Initial LIPM DCM initialization mismatch: "
            f"{initial_dcm_error:.3e}"
        )

    # ========================================================
    # ONLINE SWING TRAJECTORY
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
    # VISUALIZATION
    # ========================================================

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

    x_initial = np.array(
        [
            lipm_sample.position[0],
            lipm_sample.velocity[0],
            lipm_sample.acceleration[0],
        ],
        dtype=float,
    )

    y_initial = np.array(
        [
            lipm_sample.position[1],
            lipm_sample.velocity[1],
            lipm_sample.acceleration[1],
        ],
        dtype=float,
    )

    zmp_visualizer.initialize_from_states(

        x_state=(
            x_initial
        ),

        y_state=(
            y_initial
        ),
    )

    # ========================================================
    # RUNTIME
    # ========================================================

    phase_time = 0.0

    kinematic_time = 0.0

    step_index = 1

    planner_frozen = False

    disturbance_applied = False

    previous_com_actual = (
        initial_com_actual.copy()
    )

    com_velocity_actual = np.zeros(
        3,
        dtype=float,
    )

    next_print_time = 0.0

    next_viewer_sync_time = 0.0

    wall_start = (
        time.perf_counter()
    )

    # ========================================================
    # STARTUP INFORMATION
    # ========================================================

    print()

    print(
        "Point-foot LIPM Step Timing Adaptation"
    )

    print(
        "Close MuJoCo GUI to stop."
    )

    print(
        f"v_des="
        f"({DESIRED_VELOCITY_X:+.3f}, "
        f"{DESIRED_VELOCITY_Y:+.3f}) m/s"
        f" | l_p={measured_step_width:.4f} m"
    )

    print(
        f"T_nom={first_nominal_step.step_time:.4f} s"
        f" | L_nom={first_nominal_step.step_length:.4f} m"
        f" | v0_LIPM="
        f"({initial_com_velocity[0]:+.4f},"
        f"{initial_com_velocity[1]:+.4f}) m/s"
    )

    print(
        f"first QP:"
        f" uT="
        f"({landing_position[0]:+.4f},"
        f"{landing_position[1]:+.4f}) m"
        f" | T={current_step_time:.4f} s"
    )

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        if not viewer.is_running():
            break

        # ====================================================
        # CURRENT LIPM STATE
        # ====================================================

        lipm_sample = (
            lipm.sample()
        )

        # ====================================================
        # TOUCHDOWN
        #
        # Use ">" rather than ">=" intentionally.
        #
        # At phase_time == T the swing trajectory receives one
        # final sample exactly at touchdown.
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

            # ------------------------------------------------
            # Commit landing target.
            # ------------------------------------------------

            if swing_side == "left":

                (
                    _,
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
                    _,
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

            # ------------------------------------------------
            # Store completed footprint.
            # ------------------------------------------------

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

            # =================================================
            # SUPPORT SWITCH
            #
            # Paper point-foot model:
            #
            #       u0_(k+1) = uT_k
            #
            # CoM position and velocity remain continuous.
            # =================================================

            stance_side = (
                swing_side
            )

            lipm.set_support_position(
                landing_position
            )

            lipm_sample = (
                lipm.sample()
            )

            # =================================================
            # NEXT STEP
            # =================================================

            step_index += 1

            phase_time = 0.0

            planner_frozen = False

            disturbance_applied = False

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
        # SYNTHETIC DCM DISTURBANCE
        # ====================================================

        if (
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

            lipm.apply_dcm_disturbance(

                DISTURBANCE_DCM_X,

                DISTURBANCE_DCM_Y,
            )

            disturbance_applied = True

            lipm_sample = (
                lipm.sample()
            )

        # ====================================================
        # ONLINE STEP LOCATION + TIMING ADAPTATION
        # ====================================================

        if (
            not planner_frozen
            and
            phase_time
            >
            TIME_TOLERANCE
        ):

            # ------------------------------------------------
            # Freeze adaptation once insufficient timing
            # margin remains.
            # ------------------------------------------------

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

                    landing_position[0] = (
                        planner_result.step_location_x
                    )

                    landing_position[1] = (
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

                    # ----------------------------------------
                    # Expected late-step condition.
                    # ----------------------------------------

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
        # CURRENT LIPM REFERENCE
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
        # SWING FOOT REFERENCE
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

        # ====================================================
        # SUPPORT / SWING TASK REFERENCES
        # ====================================================

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
        # HIERARCHICAL DIFFERENTIAL IK
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
        # KINEMATIC INTEGRATION
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
        # ACTUAL ROBOT CoM VELOCITY
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

        # ====================================================
        # ZMP
        #
        # p_ZMP = c - h/g * c_ddot = stance foot
        # ====================================================

        zmp_visualizer.update_current(

            com_position=(
                lipm_sample.position
            ),

            com_acceleration=(
                lipm_sample.acceleration
            ),
        )

        # ====================================================
        # TERMINAL OUTPUT
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
                f" | uT="
                f"({landing_position[0]:+.4f},"
                f"{landing_position[1]:+.4f}) m"
                f" | T={current_step_time:.4f} s"
                f" | vCoM="
                f"({com_velocity_actual[0]:+.4f},"
                f"{com_velocity_actual[1]:+.4f}) m/s"
            )

            next_print_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # GUI
        # ====================================================

        if (
            kinematic_time
            >=
            next_viewer_sync_time
            -
            TIME_TOLERANCE
        ):

            visual_state = (
                VisualWalkingState(

                    phase=(
                        WalkingPhase
                        .SINGLE_SUPPORT
                    ),

                    support_side=(
                        stance_side
                    ),
                )
            )

            # Existing foot / CoM / support visualization.
            walking_visualizer.update(

                viewer,

                robot,

                visual_state,
            )

            # Adaptive target + DCM + LIPM CoM.
            draw_adaptive_overlay(

                viewer=(
                    viewer
                ),

                walking_visualizer=(
                    walking_visualizer
                ),

                com_position=(
                    lipm_sample.position
                ),

                dcm=(
                    lipm_sample.dcm
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

            # Point-foot ZMP.
            zmp_visualizer.draw_overlay(
                viewer
            )

            next_viewer_sync_time += (
                VIEWER_SYNC_PERIOD
            )

        # ====================================================
        # EXACT LIPM PROPAGATION
        # ====================================================

        lipm.advance(
            DT
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
        # REALTIME PLAYBACK
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

    # ========================================================
    # STOP
    # ========================================================

    print()

    print(
        f"Walking stopped"
        f" | t={kinematic_time:.3f} s"
        f" | steps={step_index}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK FILES
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
    # PINOCCHIO MODEL
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
    #
    # This replaces the old hard-coded:
    #
    #       DEFAULT_STEP_WIDTH = 0.16
    #
    # The planner's l_p should correspond to the actual robot
    # nominal lateral foot separation.
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
            left_position_initial[1]
            -
            right_position_initial[1]
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

    # ========================================================
    # ADAPTIVE STEP PLANNER
    #
    # Construct AFTER settling because l_p is now measured
    # from the actual robot configuration.
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
    # RUN UNTIL GUI CLOSES
    # ========================================================

    if not SHOW_VIEWER:

        raise RuntimeError(
            "SHOW_VIEWER must be True."
        )

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

            measured_step_width=(
                measured_step_width
            ),

            viewer=(
                viewer
            ),
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()