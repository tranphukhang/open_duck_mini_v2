# step_timing_adaptation/run.py

from __future__ import annotations

import sys
import time
import threading

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import glfw
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
# SETTLING / ZMP VISUALIZATION
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
    add_sphere,
    add_line,
    draw_polyline,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# EXECUTOR
# ============================================================

DT = 0.0005


# ============================================================
# FIRST SUPPORT
# ============================================================

FIRST_STANCE_SIDE = "left"


# ============================================================
# INITIAL WALKING COMMAND
# ============================================================

INITIAL_DESIRED_VELOCITY_X = 0.0
INITIAL_DESIRED_VELOCITY_Y = 0.0


# ============================================================
# KEYBOARD COMMAND STEP
# ============================================================

VELOCITY_X_STEP = 0.05
VELOCITY_Y_STEP = 0.05


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
# COMMAND LIMITS
# ============================================================

VELOCITY_X_MIN = (
    STEP_LENGTH_MIN
    /
    STEP_TIME_MIN
)

VELOCITY_X_MAX = (
    STEP_LENGTH_MAX
    /
    STEP_TIME_MIN
)

VELOCITY_Y_MIN = (
    STEP_WIDTH_MIN
    /
    STEP_TIME_MIN
)

VELOCITY_Y_MAX = (
    STEP_WIDTH_MAX
    /
    STEP_TIME_MIN
)


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
# OPTIONAL SYNTHETIC DCM DISTURBANCE
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
# VISUAL HISTORY
# ============================================================

VISUAL_HISTORY_DURATION = 2.0


# ============================================================
# NUMERICAL
# ============================================================

TIME_TOLERANCE = 1.0e-10


# ============================================================
# VELOCITY ARROW VISUALIZATION
#
# MuJoCo native rendering-only geometry:
#
#     mjGEOM_ARROW
#
# Both arrows have exactly the same origin.
#
# Blue:
#     commanded velocity
#
# Red:
#     actual robot CoM velocity
#
# Arrow vector:
#
#     Delta p =
#         VELOCITY_ARROW_SCALE
#         *
#         [vx, vy, 0]
# ============================================================

VELOCITY_ARROW_HEIGHT_OFFSET = 0.16

# Displayed metres per m/s.
VELOCITY_ARROW_SCALE = 0.45

VELOCITY_ARROW_MIN_NORM = 1.0e-5


# ------------------------------------------------------------
# IMPORTANT:
#
# For mjGEOM_ARROW, width is a geometric width rather than
# the pixel-width convention of mjGEOM_LINE.
#
# Command is thicker.
# Actual velocity is thinner and drawn afterwards.
# ------------------------------------------------------------

COMMAND_ARROW_WIDTH = 0.008

CURRENT_VELOCITY_ARROW_WIDTH = 0.0045


COMMAND_ARROW_RGBA = np.array(
    [
        0.00,
        0.35,
        1.00,
        1.00,
    ],
    dtype=np.float32,
)


CURRENT_VELOCITY_ARROW_RGBA = np.array(
    [
        1.00,
        0.05,
        0.05,
        1.00,
    ],
    dtype=np.float32,
)


# ============================================================
# ZMP VISUALIZATION
#
# Built-in long history is disabled.
#
# A custom timestamped 4-second trail is drawn below.
# ============================================================

ZMP_VISUALIZATION_CONFIG = (
    ZMPVisualizationConfig(

        show_current=True,

        show_trail=False,

        show_preview=False,

        current_z=0.010,

        trail_z=0.008,

        preview_z=0.007,

        current_radius=0.007,

        preview_radius=0.0028,

        trail_width=5.0,

        preview_width=2.5,

        trail_min_distance=5.0e-4,

        trail_max_points=2,

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
# RECENT ZMP TRAIL
# ============================================================

RECENT_ZMP_TRAIL_WIDTH = 5.0

RECENT_ZMP_TRAIL_RGBA = np.array(
    [
        0.90,
        0.10,
        0.95,
        0.80,
    ],
    dtype=np.float32,
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
# VELOCITY COMMAND
# ============================================================

@dataclass(frozen=True)
class VelocityCommandSnapshot:

    x: float
    y: float
    version: int


class VelocityCommand:
    """
    Coordinate convention:

        +x : forward
        -x : backward

        +y : left
        -y : right

    Keys:

        UP    -> vx += 0.05
        DOWN  -> vx -= 0.05
        LEFT  -> vy += 0.05
        RIGHT -> vy -= 0.05
    """

    def __init__(
        self,
        *,
        initial_x,
        initial_y,
    ):

        self._lock = (
            threading.Lock()
        )

        self._x = float(
            np.clip(
                initial_x,
                VELOCITY_X_MIN,
                VELOCITY_X_MAX,
            )
        )

        self._y = float(
            np.clip(
                initial_y,
                VELOCITY_Y_MIN,
                VELOCITY_Y_MAX,
            )
        )

        self._version = 0


    def snapshot(
        self,
    ) -> VelocityCommandSnapshot:

        with self._lock:

            return VelocityCommandSnapshot(

                x=float(
                    self._x
                ),

                y=float(
                    self._y
                ),

                version=int(
                    self._version
                ),
            )


    def key_callback(
        self,
        keycode,
    ):

        with self._lock:

            old_x = (
                self._x
            )

            old_y = (
                self._y
            )

            if keycode == glfw.KEY_UP:

                self._x = float(
                    np.clip(
                        self._x
                        +
                        VELOCITY_X_STEP,

                        VELOCITY_X_MIN,
                        VELOCITY_X_MAX,
                    )
                )

            elif keycode == glfw.KEY_DOWN:

                self._x = float(
                    np.clip(
                        self._x
                        -
                        VELOCITY_X_STEP,

                        VELOCITY_X_MIN,
                        VELOCITY_X_MAX,
                    )
                )

            elif keycode == glfw.KEY_LEFT:

                self._y = float(
                    np.clip(
                        self._y
                        +
                        VELOCITY_Y_STEP,

                        VELOCITY_Y_MIN,
                        VELOCITY_Y_MAX,
                    )
                )

            elif keycode == glfw.KEY_RIGHT:

                self._y = float(
                    np.clip(
                        self._y
                        -
                        VELOCITY_Y_STEP,

                        VELOCITY_Y_MIN,
                        VELOCITY_Y_MAX,
                    )
                )

            else:

                return

            self._x = float(
                np.round(
                    self._x,
                    6,
                )
            )

            self._y = float(
                np.round(
                    self._y,
                    6,
                )
            )

            changed = (
                not np.isclose(
                    old_x,
                    self._x,
                )
                or
                not np.isclose(
                    old_y,
                    self._y,
                )
            )

            if changed:
                self._version += 1


# ============================================================
# VISUAL STATE
# ============================================================

@dataclass
class VisualWalkingState:

    phase: WalkingPhase
    support_side: str


# ============================================================
# TIMED FOOTSTEP
# ============================================================

@dataclass
class TimedFootstep:

    time: float
    footstep: PlannedFootstep


# ============================================================
# RECENT VISUAL HISTORY
# ============================================================

class RecentVisualHistory:

    def __init__(
        self,
        *,
        duration,
    ):

        self.duration = float(
            duration
        )

        if self.duration <= 0.0:

            raise ValueError(
                "Visual history duration must be positive."
            )

        self.left_trail = deque()
        self.right_trail = deque()
        self.com_trail = deque()
        self.zmp_trail = deque()
        self.footsteps = deque()


    def _prune_deque(
        self,
        data,
        current_time,
    ):

        threshold = (
            current_time
            -
            self.duration
        )

        while (
            len(data) > 0
            and
            data[0][0]
            <
            threshold
            -
            TIME_TOLERANCE
        ):

            data.popleft()


    def prune(
        self,
        current_time,
    ):

        self._prune_deque(
            self.left_trail,
            current_time,
        )

        self._prune_deque(
            self.right_trail,
            current_time,
        )

        self._prune_deque(
            self.com_trail,
            current_time,
        )

        self._prune_deque(
            self.zmp_trail,
            current_time,
        )

        threshold = (
            current_time
            -
            self.duration
        )

        while (
            len(self.footsteps) > 0
            and
            self.footsteps[0].time
            <
            threshold
            -
            TIME_TOLERANCE
        ):

            self.footsteps.popleft()


    def initialize(
        self,
        *,
        current_time,
        robot,
        zmp_world,
    ):

        (
            p_left,
            R_left,
        ) = (
            robot.get_left_foot_pose()
        )

        (
            p_right,
            R_right,
        ) = (
            robot.get_right_foot_pose()
        )

        p_com = (
            robot.get_com()
        )

        self.left_trail.append(
            (
                current_time,
                p_left.copy(),
            )
        )

        self.right_trail.append(
            (
                current_time,
                p_right.copy(),
            )
        )

        self.com_trail.append(
            (
                current_time,
                p_com.copy(),
            )
        )

        self.zmp_trail.append(
            (
                current_time,
                np.asarray(
                    zmp_world,
                    dtype=float,
                ).copy(),
            )
        )

        self.footsteps.append(
            TimedFootstep(

                time=current_time,

                footstep=PlannedFootstep(
                    side="left",
                    position=p_left.copy(),
                    rotation=R_left.copy(),
                ),
            )
        )

        self.footsteps.append(
            TimedFootstep(

                time=current_time,

                footstep=PlannedFootstep(
                    side="right",
                    position=p_right.copy(),
                    rotation=R_right.copy(),
                ),
            )
        )


    def record(
        self,
        *,
        current_time,
        robot,
        zmp_world,
    ):

        p_left, _ = (
            robot.get_left_foot_pose()
        )

        p_right, _ = (
            robot.get_right_foot_pose()
        )

        p_com = (
            robot.get_com()
        )

        self.left_trail.append(
            (
                current_time,
                p_left.copy(),
            )
        )

        self.right_trail.append(
            (
                current_time,
                p_right.copy(),
            )
        )

        self.com_trail.append(
            (
                current_time,
                p_com.copy(),
            )
        )

        self.zmp_trail.append(
            (
                current_time,
                np.asarray(
                    zmp_world,
                    dtype=float,
                ).copy(),
            )
        )

        self.prune(
            current_time
        )


    def add_footstep(
        self,
        *,
        current_time,
        side,
        position,
        rotation,
    ):

        self.footsteps.append(
            TimedFootstep(

                time=current_time,

                footstep=PlannedFootstep(

                    side=side,

                    position=np.asarray(
                        position,
                        dtype=float,
                    ).copy(),

                    rotation=np.asarray(
                        rotation,
                        dtype=float,
                    ).copy(),
                ),
            )
        )

        self.prune(
            current_time
        )


    def apply_to_walking_visualizer(
        self,
        walking_visualizer,
    ):

        walking_visualizer.left_trail = [
            point.copy()
            for _, point
            in self.left_trail
        ]

        walking_visualizer.right_trail = [
            point.copy()
            for _, point
            in self.right_trail
        ]

        walking_visualizer.com_trail = [
            point.copy()
            for _, point
            in self.com_trail
        ]

        walking_visualizer.planned_footsteps = [
            item.footstep
            for item
            in self.footsteps
        ]


    def get_zmp_points(
        self,
    ):

        return [
            point.copy()
            for _, point
            in self.zmp_trail
        ]


# ============================================================
# BASIC HELPERS
# ============================================================

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

    # --------------------------------------------------------
    # Kinematic set-state execution.
    # No mj_step() during walking.
    # --------------------------------------------------------

    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


# ============================================================
# VIEWER CONFIGURATION
# ============================================================

def configure_step_timing_viewer(
    *,
    viewer,
    walking_visualizer,
):
    """
    Configure only what is explicitly needed.

    IMPORTANT:
        viewer.opt.frame is NEVER touched here.

    Therefore:
        Rendering -> Frame

    remains at MuJoCo's default setting.

    Enabled Model Elements:
        Contact Force
        Transparent
    """

    with viewer.lock():

        # ====================================================
        # KEEP THE EXISTING TRACKING CAMERA BEHAVIOR
        #
        # This changes camera only.
        #
        # Rendering -> Frame is intentionally untouched.
        # ====================================================

        viewer.cam.type = (
            mujoco.mjtCamera
            .mjCAMERA_TRACKING
        )

        viewer.cam.trackbodyid = (
            walking_visualizer
            .base_body_id
        )

        # ====================================================
        # MODEL ELEMENTS
        # ====================================================

        viewer.opt.flags[
            mujoco.mjtVisFlag
            .mjVIS_CONTACTFORCE
        ] = True

        viewer.opt.flags[
            mujoco.mjtVisFlag
            .mjVIS_TRANSPARENT
        ] = True

    viewer.sync()


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
# ADAPTIVE OVERLAY
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
        # LIPM COM GROUND PROJECTION
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
        # CURRENT ADAPTIVE FOOTHOLD uT
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
# RECENT ZMP TRAIL
# ============================================================

def draw_recent_zmp_trail(
    *,
    viewer,
    visual_history,
):

    points = (
        visual_history
        .get_zmp_points()
    )

    if len(points) < 2:
        return

    with viewer.lock():

        draw_polyline(

            viewer.user_scn,

            points,

            RECENT_ZMP_TRAIL_WIDTH,

            RECENT_ZMP_TRAIL_RGBA,
        )


# ============================================================
# USER SCENE GEOMETRY ALLOCATION
# ============================================================

def allocate_user_geom(
    scene,
):

    if (
        scene.ngeom
        >=
        scene.maxgeom
    ):

        return None

    geom = (
        scene.geoms[
            scene.ngeom
        ]
    )

    scene.ngeom += 1

    return geom


# ============================================================
# MUJOCO-NATIVE ARROW
# ============================================================

def add_mujoco_arrow(
    *,
    scene,
    start,
    end,
    width,
    rgba,
):
    """
    Add a native MuJoCo rendering arrow.

    This uses:

        mjGEOM_ARROW

    together with:

        mjv_connector()

    rather than manually constructing an arrow from line
    segments.

    This is the same rendering-only arrow primitive used by
    MuJoCo's abstract visualization for vector decorations.
    """

    start = np.asarray(
        start,
        dtype=float,
    ).reshape(
        3
    )

    end = np.asarray(
        end,
        dtype=float,
    ).reshape(
        3
    )

    vector = (
        end
        -
        start
    )

    if (
        np.linalg.norm(
            vector
        )
        <
        1.0e-10
    ):

        return

    geom = (
        allocate_user_geom(
            scene
        )
    )

    if geom is None:
        return

    # ========================================================
    # INITIALIZE RENDERING-ONLY GEOMETRY
    # ========================================================

    mujoco.mjv_initGeom(

        geom,

        type=(
            mujoco.mjtGeom
            .mjGEOM_ARROW
        ),

        size=np.zeros(
            3,
            dtype=float,
        ),

        pos=np.zeros(
            3,
            dtype=float,
        ),

        mat=np.eye(
            3,
            dtype=float,
        ).reshape(
            9
        ),

        rgba=np.asarray(
            rgba,
            dtype=np.float32,
        ),
    )

    # ========================================================
    # CONNECT START -> END
    # ========================================================

    if hasattr(
        mujoco,
        "mjv_connector",
    ):

        mujoco.mjv_connector(

            geom=(
                geom
            ),

            type=(
                mujoco.mjtGeom
                .mjGEOM_ARROW
            ),

            width=float(
                width
            ),

            from_=(
                start
            ),

            to=(
                end
            ),
        )

    else:

        # Compatibility fallback for older MuJoCo versions.

        mujoco.mjv_makeConnector(

            geom,

            mujoco.mjtGeom
            .mjGEOM_ARROW,

            float(
                width
            ),

            float(start[0]),
            float(start[1]),
            float(start[2]),

            float(end[0]),
            float(end[1]),
            float(end[2]),
        )


# ============================================================
# DRAW ONE VELOCITY ARROW
# ============================================================

def draw_velocity_arrow(
    *,
    scene,
    origin,
    velocity_xy,
    rgba,
    width,
):

    origin = np.asarray(
        origin,
        dtype=float,
    ).reshape(
        3
    )

    velocity_xy = np.asarray(
        velocity_xy,
        dtype=float,
    ).reshape(
        2
    )

    if not np.all(
        np.isfinite(
            velocity_xy
        )
    ):

        return

    speed = float(
        np.linalg.norm(
            velocity_xy
        )
    )

    # --------------------------------------------------------
    # Zero velocity:
    #
    # simply no visible arrow.
    #
    # This has absolutely no effect on planner behaviour.
    # --------------------------------------------------------

    if speed < VELOCITY_ARROW_MIN_NORM:
        return

    velocity_world = np.array(
        [
            velocity_xy[0],
            velocity_xy[1],
            0.0,
        ],
        dtype=float,
    )

    endpoint = (
        origin
        +
        VELOCITY_ARROW_SCALE
        *
        velocity_world
    )

    add_mujoco_arrow(

        scene=(
            scene
        ),

        start=(
            origin
        ),

        end=(
            endpoint
        ),

        width=(
            width
        ),

        rgba=(
            rgba
        ),
    )


# ============================================================
# COMMAND + CURRENT VELOCITY ARROWS
# ============================================================

def draw_velocity_arrows(
    *,
    viewer,
    robot,
    command_velocity,
    current_velocity,
):
    """
    Both arrows:

        - have exactly the same origin;
        - are horizontal world-frame velocity vectors;
        - change direction with velocity direction;
        - change length with velocity magnitude.

    BLUE:
        command [vx_cmd, vy_cmd]

    RED:
        actual robot CoM velocity [vx, vy]

    The blue arrow is thicker.
    The red arrow is drawn afterwards and is thinner.

    Therefore if the two vectors are exactly coincident,
    the red arrow is visible inside the blue arrow.
    """

    (
        trunk_position,
        _,
    ) = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    arrow_origin = (
        trunk_position.copy()
    )

    arrow_origin[2] += (
        VELOCITY_ARROW_HEIGHT_OFFSET
    )

    command_velocity = np.asarray(
        command_velocity,
        dtype=float,
    ).reshape(
        2
    )

    current_velocity = np.asarray(
        current_velocity,
        dtype=float,
    ).reshape(
        2
    )

    with viewer.lock():

        scene = (
            viewer.user_scn
        )

        # ====================================================
        # BLUE COMMAND ARROW
        # ====================================================

        draw_velocity_arrow(

            scene=(
                scene
            ),

            origin=(
                arrow_origin
            ),

            velocity_xy=(
                command_velocity
            ),

            rgba=(
                COMMAND_ARROW_RGBA
            ),

            width=(
                COMMAND_ARROW_WIDTH
            ),
        )

        # ====================================================
        # RED ACTUAL VELOCITY ARROW
        # ====================================================

        draw_velocity_arrow(

            scene=(
                scene
            ),

            origin=(
                arrow_origin
            ),

            velocity_xy=(
                current_velocity
            ),

            rgba=(
                CURRENT_VELOCITY_ARROW_RGBA
            ),

            width=(
                CURRENT_VELOCITY_ARROW_WIDTH
            ),
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

    # ========================================================
    # STEP LOCATION / TIMING QP
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
    # CURRENT SWING FOOT
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
        planner_result
        .step_location_x
    )

    landing_position[1] = (
        planner_result
        .step_location_y
    )

    landing_position[2] = (
        landing_z
    )

    # ========================================================
    # ONLINE SWING RESET
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
# WALK
# ============================================================

def run_walk(
    *,
    mj_model,
    mj_data,
    robot,
    planner,
    measured_step_width,
    velocity_command,
    viewer,
):

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
    # INITIAL VELOCITY COMMAND
    # ========================================================

    command = (
        velocity_command
        .snapshot()
    )

    command_version = (
        command.version
    )

    # ========================================================
    # INITIAL NOMINAL GAIT
    # ========================================================

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

    if initial_dcm_error > 1.0e-10:

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
    # WALKING VISUALIZER
    # ========================================================

    walking_visualizer = (
        WalkingVisualizer(
            mj_model
        )
    )

    walking_visualizer.initialize(
        robot
    )

    # ========================================================
    # IMPORTANT:
    #
    # DO NOT CALL:
    #
    #     walking_visualizer.configure_viewer(viewer)
    #
    # because that old function forces:
    #
    #     Rendering -> Frame -> Site
    #
    # Instead use our configuration below, which does NOT
    # touch viewer.opt.frame.
    # ========================================================

    configure_step_timing_viewer(

        viewer=(
            viewer
        ),

        walking_visualizer=(
            walking_visualizer
        ),
    )

    # ========================================================
    # ZMP VISUALIZER
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
    # 4-SECOND VISUAL HISTORY
    # ========================================================

    visual_history = (
        RecentVisualHistory(

            duration=(
                VISUAL_HISTORY_DURATION
            )
        )
    )

    visual_history.initialize(

        current_time=0.0,

        robot=(
            robot
        ),

        zmp_world=(
            zmp_visualizer
            .current_zmp_world
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
    # HEADER
    # ========================================================

    print()

    print(
        "Point-foot LIPM Step Timing Adaptation"
    )

    print(
        "Close MuJoCo GUI to stop."
    )

    print()

    print(
        "Keyboard:"
    )

    print(
        "  UP    : vx +0.05 m/s"
    )

    print(
        "  DOWN  : vx -0.05 m/s"
    )

    print(
        "  LEFT  : vy +0.05 m/s"
    )

    print(
        "  RIGHT : vy -0.05 m/s"
    )

    print()

    print(
        "Velocity arrows:"
    )

    print(
        "  BLUE : command velocity"
    )

    print(
        "  RED  : actual CoM velocity"
    )

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        if not viewer.is_running():
            break

        # ====================================================
        # READ CURRENT COMMAND
        # ====================================================

        latest_command = (
            velocity_command
            .snapshot()
        )

        if (
            latest_command.version
            !=
            command_version
        ):

            command = (
                latest_command
            )

            command_version = (
                command.version
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

        # ====================================================
        # CURRENT LIPM
        # ====================================================

        lipm_sample = (
            lipm.sample()
        )

        # ====================================================
        # TOUCHDOWN
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

            # =================================================
            # SAVE COMPLETED FOOTSTEP
            # =================================================

            visual_history.add_footstep(

                current_time=(
                    kinematic_time
                ),

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
            # u0(k+1) = uT(k)
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
        # OPTIONAL DCM DISTURBANCE
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
                        planner_result
                        .step_location_x
                    )

                    landing_position[1] = (
                        planner_result
                        .step_location_y
                    )

                    current_step_time = (
                        planner_result
                        .step_time
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
            lipm_sample
            .position
            .copy()
        )

        com_velocity_ref = (
            lipm_sample
            .velocity
            .copy()
        )

        # ====================================================
        # ONLINE SWING REFERENCE
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
                swing_sample
                .position
                .copy()
            )

            p_right_ref = (
                right_contact_position
                .copy()
            )

        else:

            p_left_ref = (
                left_contact_position
                .copy()
            )

            p_right_ref = (
                swing_sample
                .position
                .copy()
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
        #
        # P1: support foot
        # P2: CoM
        # P3: swing foot
        # P4: full trunk orientation
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
                    swing_sample
                    .velocity
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

        # ====================================================
        # POINT-FOOT ZMP
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
        # TERMINAL
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
                f" | vCoM="
                f"({com_velocity_actual[0]:+.4f},"
                f"{com_velocity_actual[1]:+.4f}) m/s"
            )

            next_print_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # VIEWER
        # ====================================================

        if (
            kinematic_time
            >=
            next_viewer_sync_time
            -
            TIME_TOLERANCE
        ):

            # =================================================
            # 4-SECOND HISTORY
            # =================================================

            visual_history.record(

                current_time=(
                    kinematic_time
                ),

                robot=(
                    robot
                ),

                zmp_world=(
                    zmp_visualizer
                    .current_zmp_world
                ),
            )

            visual_history.apply_to_walking_visualizer(
                walking_visualizer
            )

            # =================================================
            # GAIT STATE
            # =================================================

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

            # =================================================
            # STANDARD WALKING VISUALIZATION
            # =================================================

            walking_visualizer.update(

                viewer,

                robot,

                visual_state,
            )

            # =================================================
            # RECENT ZMP
            # =================================================

            draw_recent_zmp_trail(

                viewer=(
                    viewer
                ),

                visual_history=(
                    visual_history
                ),
            )

            # =================================================
            # LIPM COM / DCM / uT
            # =================================================

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

            # =================================================
            # NATIVE MUJOCO VELOCITY ARROWS
            #
            # Both originate from the same point above trunk.
            #
            # BLUE:
            #     command
            #
            # RED:
            #     actual robot CoM velocity
            # =================================================

            draw_velocity_arrows(

                viewer=(
                    viewer
                ),

                robot=(
                    robot
                ),

                command_velocity=np.array(
                    [
                        command.x,
                        command.y,
                    ],
                    dtype=float,
                ),

                current_velocity=np.array(
                    [
                        com_velocity_actual[0],
                        com_velocity_actual[1],
                    ],
                    dtype=float,
                ),
            )

            # =================================================
            # CURRENT ZMP
            #
            # This function calls viewer.sync().
            # =================================================

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
    # INITIAL PHYSICAL SETTLING ONLY
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
    # KEYBOARD VELOCITY COMMAND
    # ========================================================

    velocity_command = (
        VelocityCommand(

            initial_x=(
                INITIAL_DESIRED_VELOCITY_X
            ),

            initial_y=(
                INITIAL_DESIRED_VELOCITY_Y
            ),
        )
    )

    # ========================================================
    # VIEWER
    # ========================================================

    if not SHOW_VIEWER:

        raise RuntimeError(
            "SHOW_VIEWER must be True."
        )

    with mujoco.viewer.launch_passive(

        mj_model,

        mj_data,

        key_callback=(
            velocity_command
            .key_callback
        ),

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

            velocity_command=(
                velocity_command
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