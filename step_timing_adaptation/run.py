# step_timing_adaptation/run.py
#
# Automatic 9-second experiment:
#   0 <= t < 3 s : vx = +0.10 m/s, vy =  0.00 m/s
#   3 <= t < 6 s : vx = +0.10 m/s, vy = -0.05 m/s
#   6 <= t <= 9 s: vx = +0.10 m/s, vy =  0.00 m/s
#
# One external push is applied automatically at t = 5.0 s.
#
# After the simulation finishes, logged data are plotted by
# simulation_plot.py.

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


# ============================================================
# EXISTING PROJECT MODULES
# ============================================================

from lipm_mpc.run import (
    settle_robot,
)

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

DATA_LOG_PERIOD = 0.005

FIRST_STANCE_SIDE = "left"

SHOW_VIEWER = True
REALTIME_PLAYBACK = True
VIEWER_SYNC_PERIOD = 0.02
STATUS_PRINT_PERIOD = 0.10

TIME_TOLERANCE = 1.0e-10


# ============================================================
# AUTOMATIC VELOCITY COMMAND
# ============================================================

# Required experiment:
#
#   0 -> 3 s : vx = 0.10, vy =  0.00
#   3 -> 6 s : vx = 0.10, vy = -0.05
#   6 -> 9 s : vx = 0.10, vy =  0.00

def automatic_velocity_profile(
    current_time: float,
) -> tuple[float, float]:

    t = float(current_time)

    if t < 3.0:

        return (
            0.10,
            0.00,
        )

    if t < 6.0:

        return (
            0.10,
            -0.05,
        )

    return (
        0.10,
        0.00,
    )


# ============================================================
# LIPM
# ============================================================

GRAVITY = 9.81

COM_HEIGHT = 0.2044


# ============================================================
# AUTOMATIC PUSH
# ============================================================

PUSH_TIME = 5.0

PUSH_FORCE_X = +2.5
PUSH_FORCE_Y = +2.5

PUSH_DURATION = 0.05


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


def update_mujoco_from_pinocchio(
    *,
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

    mj_data.qpos[:] = q_mj

    # This run.py is kinematic at the full-body level:
    # q is supplied by differential IK.
    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
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
        not np.isfinite(mass)
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

    return (
        planner.compute_nominal_step(

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
    )


def compute_nominal_steps(
    *,
    planner,
    velocity_command,
):

    nominal_left = (
        compute_nominal_step(

            planner=planner,

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

            planner=planner,

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

    return (
        planner.solve_adaptive_step(

            nominal_step=(
                nominal_step
            ),

            dcm_measured=(
                dcm
            ),

            stance_position=(
                stance_position[0:2]
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
# CONSISTENT INITIAL DCM
# ============================================================

def compute_nominal_initial_dcm(
    *,
    nominal_step,
    stance_position,
):

    u0 = np.asarray(
        stance_position[0:2],
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
        com_position[0:2],
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

    landing_position = (
        swing_start.copy()
    )

    landing_position[0] = (
        planner_result.step_location_x
    )

    landing_position[1] = (
        planner_result.step_location_y
    )

    landing_position[2] = (
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

    robot_mass = (
        compute_robot_mass(
            mj_model
        )
    )

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
    # INITIAL COMMAND: profile at t = 0
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
            x=command_x,
            y=command_y,
        )
    )

    nominal_left_step, nominal_right_step = (
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

    simulation_log = (
        SimulationLog()
    )

    next_log_time = 0.0
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
        "Velocity profile:"
    )
    print(
        "  0 <= t < 3 s : "
        "vx=+0.10 m/s, vy=+0.00 m/s"
    )
    print(
        "  3 <= t < 6 s : "
        "vx=+0.10 m/s, vy=-0.05 m/s"
    )
    print(
        "  6 <= t <=9 s: "
        "vx=+0.10 m/s, vy=+0.00 m/s"
    )
    print(
        f"Automatic push: t={PUSH_TIME:.2f} s"
        f" | F=({PUSH_FORCE_X:+.2f}, "
        f"{PUSH_FORCE_Y:+.2f}) N"
        f" | duration={PUSH_DURATION:.3f} s"
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
                    x=new_command_x,
                    y=new_command_y,
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
        # AUTOMATIC PUSH -- EXACTLY ONCE
        # ====================================================

        if (
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

            automatic_push_applied = (
                True
            )

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

                    if (
                        "timing adaptation window is closed"
                        in
                        str(error).lower()
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
            push_time_remaining
            >
            TIME_TOLERANCE
        )

        # ====================================================
        # DATA LOG
        # ====================================================

        if (
            kinematic_time
            >=
            next_log_time
            -
            TIME_TOLERANCE
        ):

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
            )

            next_log_time += (
                DATA_LOG_PERIOD
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

    return (
        simulation_log
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

            simulation_log = (
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

        simulation_log = (
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
    # PLOT RESULTS AFTER SIMULATION
    # ========================================================

    plot_simulation_results(

        simulation_log,

        save_directory=(
            CURRENT_DIR
            /
            "results"
        ),

        push_time=(
            PUSH_TIME
        ),

        command_change_times=(
            3.0,
            6.0,
        ),

        show=True,
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
