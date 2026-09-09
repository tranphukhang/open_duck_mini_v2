from dataclasses import dataclass

import numpy as np
import mujoco


# ============================================================
# VISUAL FOOT GEOMETRY
# ============================================================
#
# These values are used ONLY for visualization.
# They do NOT affect:
#   - footstep planner
#   - IK
#   - MuJoCo contacts
#
# Approximate Open Duck Mini sole footprint.
# +x = walking direction.
# ============================================================

GROUND_Z = 0.0

FOOT_TOE = 0.0645
FOOT_HEEL = 0.0386
FOOT_HALF_WIDTH = 0.02065

FOOT_LENGTH = FOOT_TOE + FOOT_HEEL
FOOT_HALF_LENGTH = 0.5 * FOOT_LENGTH
FOOT_CENTER_X = 0.5 * (FOOT_TOE - FOOT_HEEL)

FOOTPRINT_THICKNESS = 0.0010
FOOTPRINT_Z = 0.0015

SUPPORT_HULL_Z = 0.0040


# ============================================================
# VISUAL COLORS
# ============================================================

PLANNED_FOOT_RGBA = np.array(
    [0.45, 0.08, 0.08, 0.38],
    dtype=np.float32,
)

TOE_EDGE_RGBA = np.array(
    [0.95, 0.80, 0.35, 0.85],
    dtype=np.float32,
)

LEFT_TRAIL_RGBA = np.array(
    [0.10, 0.75, 0.15, 1.00],
    dtype=np.float32,
)

RIGHT_TRAIL_RGBA = np.array(
    [0.75, 0.10, 0.10, 1.00],
    dtype=np.float32,
)

COM_TRAIL_RGBA = np.array(
    [0.10, 0.25, 0.90, 1.00],
    dtype=np.float32,
)

COM_MARKER_RGBA = np.array(
    [0.10, 0.25, 0.90, 0.95],
    dtype=np.float32,
)

SUPPORT_HULL_RGBA = np.array(
    [1.00, 0.55, 0.00, 1.00],
    dtype=np.float32,
)

SUPPORT_VERTEX_RGBA = np.array(
    [1.00, 0.85, 0.10, 1.00],
    dtype=np.float32,
)


# ============================================================
# VISUAL SIZES
# ============================================================

LEFT_TRAIL_WIDTH = 8.0
RIGHT_TRAIL_WIDTH = 8.0
COM_TRAIL_WIDTH = 10.0

TOE_EDGE_WIDTH = 4.0

SUPPORT_HULL_WIDTH = 7.0
SUPPORT_VERTEX_RADIUS = 0.0035

COM_MARKER_RADIUS = 0.007


# ============================================================
# CAMERA
# ============================================================

CAMERA_DISTANCE = 0.85
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -20.0


# ============================================================
# HISTORY LIMITS
# ============================================================

TRAIL_MIN_DISTANCE = 5e-4

TRAIL_MAX_POINTS = 180

MAX_PLANNED_FOOTSTEPS = 120


# ============================================================
# DATA
# ============================================================

@dataclass
class PlannedFootstep:

    side: str

    position: np.ndarray

    rotation: np.ndarray


# ============================================================
# BASIC GEOMETRY
# ============================================================

def yaw_rotation(
    R,
):
    """
    Extract only the yaw component.

    Planned footprints and support polygons are drawn flat
    on the ground plane.
    """

    R = np.asarray(
        R,
        dtype=float,
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0],
    )

    c = np.cos(
        yaw
    )

    s = np.sin(
        yaw
    )

    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


# ============================================================
# MUJOCO USER-SCENE GEOMETRY
# ============================================================

def allocate_geom(
    scene,
):
    if (
        scene.ngeom
        >=
        scene.maxgeom
    ):
        return None

    geom = scene.geoms[
        scene.ngeom
    ]

    scene.ngeom += 1

    return geom


def add_box(
    scene,
    position,
    rotation,
    size,
    rgba,
):
    geom = allocate_geom(
        scene
    )

    if geom is None:
        return

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(
            size,
            dtype=float,
        ),
        pos=np.asarray(
            position,
            dtype=float,
        ),
        mat=np.asarray(
            rotation,
            dtype=float,
        ).reshape(9),
        rgba=np.asarray(
            rgba,
            dtype=np.float32,
        ),
    )


def add_sphere(
    scene,
    position,
    radius,
    rgba,
):
    geom = allocate_geom(
        scene
    )

    if geom is None:
        return

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array(
            [
                radius,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
        pos=np.asarray(
            position,
            dtype=float,
        ),
        mat=np.eye(
            3
        ).reshape(9),
        rgba=np.asarray(
            rgba,
            dtype=np.float32,
        ),
    )


def add_line(
    scene,
    start,
    end,
    width,
    rgba,
):
    start = np.asarray(
        start,
        dtype=float,
    )

    end = np.asarray(
        end,
        dtype=float,
    )

    if (
        np.linalg.norm(
            end - start
        )
        <
        1e-10
    ):
        return

    geom = allocate_geom(
        scene
    )

    if geom is None:
        return

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(
            3
        ),
        pos=np.zeros(
            3
        ),
        mat=np.eye(
            3
        ).reshape(9),
        rgba=np.asarray(
            rgba,
            dtype=np.float32,
        ),
    )

    if hasattr(
        mujoco,
        "mjv_connector",
    ):

        mujoco.mjv_connector(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_LINE,
            width=width,
            from_=start,
            to=end,
        )

    else:

        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            start,
            end,
        )


def draw_polyline(
    scene,
    points,
    width,
    rgba,
):
    if len(
        points
    ) < 2:
        return

    for i in range(
        len(points) - 1
    ):

        add_line(
            scene,
            points[i],
            points[i + 1],
            width,
            rgba,
        )


# ============================================================
# 2D CONVEX HULL
# ============================================================

def convex_hull_2d(
    points,
):
    points = np.asarray(
        points,
        dtype=float,
    )

    unique = {}

    for p in points:

        key = (
            round(
                float(
                    p[0]
                ),
                12,
            ),
            round(
                float(
                    p[1]
                ),
                12,
            ),
        )

        unique[key] = np.array(
            [
                p[0],
                p[1],
                GROUND_Z
                +
                SUPPORT_HULL_Z,
            ],
            dtype=float,
        )

    pts = sorted(
        unique.values(),
        key=lambda p: (
            p[0],
            p[1],
        ),
    )

    if len(
        pts
    ) <= 2:

        return np.asarray(
            pts
        )

    def cross(
        o,
        a,
        b,
    ):
        return (
            (a[0] - o[0])
            *
            (b[1] - o[1])
            -
            (a[1] - o[1])
            *
            (b[0] - o[0])
        )

    lower = []

    for p in pts:

        while (
            len(lower) >= 2
            and
            cross(
                lower[-2],
                lower[-1],
                p,
            )
            <=
            0.0
        ):
            lower.pop()

        lower.append(
            p
        )

    upper = []

    for p in reversed(
        pts
    ):

        while (
            len(upper) >= 2
            and
            cross(
                upper[-2],
                upper[-1],
                p,
            )
            <=
            0.0
        ):
            upper.pop()

        upper.append(
            p
        )

    hull = (
        lower[:-1]
        +
        upper[:-1]
    )

    return np.asarray(
        hull
    )


# ============================================================
# MAIN VISUALIZER
# ============================================================

class WalkingVisualizer:

    def __init__(
        self,
        mj_model,
    ):
        self.mj_model = (
            mj_model
        )

        self.left_trail = []

        self.right_trail = []

        self.com_trail = []

        self.planned_footsteps = []

        self.last_registered_step = (
            None
        )

        self.base_body_id = (
            self.find_base_body_id()
        )


    # ========================================================
    # FLOATING BASE
    # ========================================================

    def find_base_body_id(
        self,
    ):
        for joint_id in range(
            self.mj_model.njnt
        ):

            if (
                self.mj_model.jnt_type[
                    joint_id
                ]
                ==
                mujoco.mjtJoint.mjJNT_FREE
            ):

                return int(
                    self.mj_model.jnt_bodyid[
                        joint_id
                    ]
                )

        raise RuntimeError(
            "Floating-base free joint "
            "was not found."
        )


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(
        self,
        robot,
    ):
        p_left, R_left = (
            robot.get_left_foot_pose()
        )

        p_right, R_right = (
            robot.get_right_foot_pose()
        )

        self.planned_footsteps = [
            PlannedFootstep(
                side="left",
                position=(
                    p_left.copy()
                ),
                rotation=(
                    R_left.copy()
                ),
            ),

            PlannedFootstep(
                side="right",
                position=(
                    p_right.copy()
                ),
                rotation=(
                    R_right.copy()
                ),
            ),
        ]

        self.record_trajectory(
            robot
        )


    # ========================================================
    # PLANNED FOOTSTEPS
    # ========================================================

    def register_step(
        self,
        state,
        robot,
    ):
        if (
            state.phase.value
            !=
            "SINGLE_SUPPORT"
        ):
            return

        if (
            state.step_index
            ==
            self.last_registered_step
        ):
            return

        if (
            state.swing_side
            ==
            "left"
        ):

            _, R = (
                robot.get_left_foot_pose()
            )

        elif (
            state.swing_side
            ==
            "right"
        ):

            _, R = (
                robot.get_right_foot_pose()
            )

        else:

            return

        footstep = PlannedFootstep(
            side=(
                state.swing_side
            ),

            position=(
                state.swing_target.copy()
            ),

            rotation=(
                R.copy()
            ),
        )

        self.planned_footsteps.append(
            footstep
        )

        if (
            len(
                self.planned_footsteps
            )
            >
            MAX_PLANNED_FOOTSTEPS
        ):

            remove_count = (
                len(
                    self.planned_footsteps
                )
                -
                MAX_PLANNED_FOOTSTEPS
            )

            del self.planned_footsteps[
                0:
                remove_count
            ]

        self.last_registered_step = (
            state.step_index
        )


    # ========================================================
    # TRAILS
    # ========================================================

    @staticmethod
    def append_trail_point(
        trail,
        point,
    ):
        point = np.asarray(
            point,
            dtype=float,
        ).copy()

        if len(
            trail
        ) == 0:

            trail.append(
                point
            )

        else:

            distance = np.linalg.norm(
                point
                -
                trail[-1]
            )

            if (
                distance
                >=
                TRAIL_MIN_DISTANCE
            ):

                trail.append(
                    point
                )

        if (
            len(
                trail
            )
            >
            TRAIL_MAX_POINTS
        ):

            remove_count = (
                len(
                    trail
                )
                -
                TRAIL_MAX_POINTS
            )

            del trail[
                0:
                remove_count
            ]


    def record_trajectory(
        self,
        robot,
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

        self.append_trail_point(
            self.left_trail,
            p_left,
        )

        self.append_trail_point(
            self.right_trail,
            p_right,
        )

        self.append_trail_point(
            self.com_trail,
            p_com,
        )

        return p_com


    # ========================================================
    # FOOTPRINT
    # ========================================================

    def draw_planned_footprint(
        self,
        scene,
        footstep,
    ):
        p = (
            footstep.position
        )

        R = yaw_rotation(
            footstep.rotation
        )

        center_local = np.array(
            [
                FOOT_CENTER_X,
                0.0,
                0.0,
            ]
        )

        center_world = (
            p
            +
            R @ center_local
        )

        center_world[2] = (
            GROUND_Z
            +
            FOOTPRINT_Z
        )

        add_box(
            scene,
            center_world,
            R,
            [
                FOOT_HALF_LENGTH,
                FOOT_HALF_WIDTH,
                FOOTPRINT_THICKNESS,
            ],
            PLANNED_FOOT_RGBA,
        )

        toe_left = (
            p
            +
            R
            @
            np.array(
                [
                    FOOT_TOE,
                    FOOT_HALF_WIDTH,
                    0.0,
                ]
            )
        )

        toe_right = (
            p
            +
            R
            @
            np.array(
                [
                    FOOT_TOE,
                    -FOOT_HALF_WIDTH,
                    0.0,
                ]
            )
        )

        toe_left[2] = (
            GROUND_Z
            +
            2.0
            *
            FOOTPRINT_Z
        )

        toe_right[2] = (
            GROUND_Z
            +
            2.0
            *
            FOOTPRINT_Z
        )

        add_line(
            scene,
            toe_left,
            toe_right,
            TOE_EDGE_WIDTH,
            TOE_EDGE_RGBA,
        )


    # ========================================================
    # CURRENT SOLE CORNERS
    # ========================================================

    def sole_corners(
        self,
        position,
        rotation,
    ):
        R = yaw_rotation(
            rotation
        )

        local_corners = np.array(
            [
                [
                    FOOT_TOE,
                    FOOT_HALF_WIDTH,
                    0.0,
                ],
                [
                    FOOT_TOE,
                    -FOOT_HALF_WIDTH,
                    0.0,
                ],
                [
                    -FOOT_HEEL,
                    -FOOT_HALF_WIDTH,
                    0.0,
                ],
                [
                    -FOOT_HEEL,
                    FOOT_HALF_WIDTH,
                    0.0,
                ],
            ]
        )

        corners = np.array(
            [
                position
                +
                R @ corner
                for corner
                in local_corners
            ]
        )

        corners[:, 2] = (
            GROUND_Z
            +
            SUPPORT_HULL_Z
        )

        return corners


    # ========================================================
    # CURRENT SUPPORT POLYGON
    # ========================================================

    def get_support_hull(
        self,
        robot,
        state,
    ):
        p_left, R_left = (
            robot.get_left_foot_pose()
        )

        p_right, R_right = (
            robot.get_right_foot_pose()
        )

        left_corners = (
            self.sole_corners(
                p_left,
                R_left,
            )
        )

        right_corners = (
            self.sole_corners(
                p_right,
                R_right,
            )
        )

        if (
            state.phase.value
            ==
            "SINGLE_SUPPORT"
        ):

            if (
                state.support_side
                ==
                "left"
            ):
                points = (
                    left_corners
                )

            else:
                points = (
                    right_corners
                )

        else:

            points = np.vstack(
                [
                    left_corners,
                    right_corners,
                ]
            )

        return convex_hull_2d(
            points
        )


    def draw_support_hull(
        self,
        scene,
        hull,
    ):
        if (
            hull is None
            or
            len(hull) == 0
        ):
            return

        for p in hull:

            add_sphere(
                scene,
                p,
                SUPPORT_VERTEX_RADIUS,
                SUPPORT_VERTEX_RGBA,
            )

        if len(
            hull
        ) < 2:
            return

        for i in range(
            len(hull)
        ):

            j = (
                i + 1
            ) % len(
                hull
            )

            add_line(
                scene,
                hull[i],
                hull[j],
                SUPPORT_HULL_WIDTH,
                SUPPORT_HULL_RGBA,
            )


    # ========================================================
    # CAMERA
    # ========================================================

    def configure_viewer(
        self,
        viewer,
    ):
        with viewer.lock():

            viewer.cam.type = (
                mujoco.mjtCamera
                .mjCAMERA_TRACKING
            )

            viewer.cam.trackbodyid = (
                self.base_body_id
            )

            viewer.cam.distance = (
                CAMERA_DISTANCE
            )

            viewer.cam.azimuth = (
                CAMERA_AZIMUTH
            )

            viewer.cam.elevation = (
                CAMERA_ELEVATION
            )

            viewer.opt.frame = (
                mujoco.mjtFrame
                .mjFRAME_SITE
            )

        viewer.sync()


    # ========================================================
    # UPDATE USER SCENE
    # ========================================================

    def update(
        self,
        viewer,
        robot,
        state,
    ):
        p_com = (
            self.record_trajectory(
                robot
            )
        )

        support_hull = (
            self.get_support_hull(
                robot,
                state,
            )
        )

        with viewer.lock():

            scene = (
                viewer.user_scn
            )

            scene.ngeom = 0

            for footstep in (
                self.planned_footsteps
            ):

                self.draw_planned_footprint(
                    scene,
                    footstep,
                )

            draw_polyline(
                scene,
                self.left_trail,
                LEFT_TRAIL_WIDTH,
                LEFT_TRAIL_RGBA,
            )

            draw_polyline(
                scene,
                self.right_trail,
                RIGHT_TRAIL_WIDTH,
                RIGHT_TRAIL_RGBA,
            )

            draw_polyline(
                scene,
                self.com_trail,
                COM_TRAIL_WIDTH,
                COM_TRAIL_RGBA,
            )

            self.draw_support_hull(
                scene,
                support_hull,
            )

            add_sphere(
                scene,
                p_com,
                COM_MARKER_RADIUS,
                COM_MARKER_RGBA,
            )

        viewer.sync()