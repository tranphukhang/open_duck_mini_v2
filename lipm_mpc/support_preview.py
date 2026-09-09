# lipm_mpc/support_preview.py

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORT GEOMETRY FROM FOOTSTEP_PLANNING
# ============================================================

from footstep_planning.walking_visualization import (
    convex_hull_2d,
    yaw_rotation,
)


# ============================================================
# DATA
# ============================================================

@dataclass
class SupportPreview:
    """
    2D support-region preview over the MPC horizon.

    For every MPC interval k:

        x_min[k] <= x_Z <= x_max[k]
        y_min[k] <= y_Z <= y_max[k]

    The full support polygon is also stored for future use.
    """

    time: np.ndarray

    x_min: np.ndarray
    x_max: np.ndarray

    y_min: np.ndarray
    y_max: np.ndarray

    phase: tuple[str, ...]
    support_side: tuple[str, ...]

    polygons: tuple[np.ndarray, ...]

    @property
    def horizon_steps(self) -> int:
        return len(self.time)


# ============================================================
# UTILITIES
# ============================================================

def _phase_name(phase) -> str:

    if hasattr(phase, "value"):
        return str(phase.value)

    return str(phase)


def _validate_foot_geometry(
    foot_toe: float,
    foot_heel: float,
    foot_half_width: float,
) -> None:

    if foot_toe <= 0.0:
        raise ValueError(
            "foot_toe must be greater than zero."
        )

    if foot_heel <= 0.0:
        raise ValueError(
            "foot_heel must be greater than zero."
        )

    if foot_half_width <= 0.0:
        raise ValueError(
            "foot_half_width must be greater than zero."
        )


def _validate_pose(
    position,
    rotation,
) -> tuple[np.ndarray, np.ndarray]:

    position = np.asarray(
        position,
        dtype=float,
    )

    rotation = np.asarray(
        rotation,
        dtype=float,
    )

    if position.shape != (3,):
        raise ValueError(
            "position must have shape (3,)."
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            "rotation must have shape (3, 3)."
        )

    if not np.all(
        np.isfinite(position)
    ):
        raise ValueError(
            "position must contain finite values."
        )

    if not np.all(
        np.isfinite(rotation)
    ):
        raise ValueError(
            "rotation must contain finite values."
        )

    return position, rotation


# ============================================================
# SOLE CORNERS
# ============================================================

def compute_sole_corners(
    position,
    rotation,
    foot_toe: float,
    foot_heel: float,
    foot_half_width: float,
) -> np.ndarray:
    """
    Compute the four sole corners in world coordinates.

    Same sole convention as walking_visualization:

        x_local in [-foot_heel, +foot_toe]
        y_local in [-foot_half_width, +foot_half_width]

    Only foot yaw is used when projecting the sole onto
    the ground plane.
    """

    _validate_foot_geometry(
        foot_toe=foot_toe,
        foot_heel=foot_heel,
        foot_half_width=foot_half_width,
    )

    position, rotation = _validate_pose(
        position,
        rotation,
    )

    R_yaw = yaw_rotation(
        rotation
    )

    local_corners = np.array(
        [
            [
                +foot_toe,
                +foot_half_width,
                0.0,
            ],
            [
                +foot_toe,
                -foot_half_width,
                0.0,
            ],
            [
                -foot_heel,
                -foot_half_width,
                0.0,
            ],
            [
                -foot_heel,
                +foot_half_width,
                0.0,
            ],
        ],
        dtype=float,
    )

    corners = np.array(
        [
            position
            +
            R_yaw @ corner

            for corner
            in local_corners
        ],
        dtype=float,
    )

    return corners


# ============================================================
# SUPPORT POLYGON
# ============================================================

def compute_support_polygon(
    left_position,
    left_rotation,
    right_position,
    right_rotation,
    phase,
    support_side,
    foot_toe: float,
    foot_heel: float,
    foot_half_width: float,
) -> np.ndarray:
    """
    Compute the support polygon.

    SINGLE_SUPPORT:
        convex hull of one support foot.

    DOUBLE_SUPPORT:
        convex hull of both feet.

    Returns
    -------
    polygon:
        ndarray with shape (M, 2)
        containing [x, y] hull vertices.
    """

    left_corners = compute_sole_corners(
        position=left_position,
        rotation=left_rotation,
        foot_toe=foot_toe,
        foot_heel=foot_heel,
        foot_half_width=foot_half_width,
    )

    right_corners = compute_sole_corners(
        position=right_position,
        rotation=right_rotation,
        foot_toe=foot_toe,
        foot_heel=foot_heel,
        foot_half_width=foot_half_width,
    )

    phase_name = _phase_name(
        phase
    )

    # --------------------------------------------------------
    # SINGLE SUPPORT
    # --------------------------------------------------------

    if phase_name == "SINGLE_SUPPORT":

        if support_side == "left":

            points = (
                left_corners
            )

        elif support_side == "right":

            points = (
                right_corners
            )

        else:

            raise ValueError(
                "During SINGLE_SUPPORT, "
                "support_side must be "
                "'left' or 'right'."
            )

    # --------------------------------------------------------
    # DOUBLE SUPPORT / STANDING
    # --------------------------------------------------------

    else:

        points = np.vstack(
            [
                left_corners,
                right_corners,
            ]
        )

    hull = convex_hull_2d(
        points
    )

    hull = np.asarray(
        hull,
        dtype=float,
    )

    if (
        hull.ndim != 2
        or
        len(hull) == 0
    ):
        raise RuntimeError(
            "Invalid support polygon."
        )

    return hull[:, :2].copy()


# ============================================================
# SUPPORT BOUNDS
# ============================================================

def compute_support_bounds(
    polygon,
    zmp_scale: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Convert a 2D support polygon into independent x/y bounds.

    Returns:

        x_min, x_max,
        y_min, y_max

    The bounds are contracted by zmp_scale about the
    center of their corresponding interval.

    This is suitable for the current decoupled x/y MPC
    formulation with axis-aligned walking.
    """

    polygon = np.asarray(
        polygon,
        dtype=float,
    )

    if (
        polygon.ndim != 2
        or
        polygon.shape[0] == 0
        or
        polygon.shape[1] < 2
    ):
        raise ValueError(
            "polygon must have shape (N, 2) "
            "with N > 0."
        )

    if not np.all(
        np.isfinite(polygon)
    ):
        raise ValueError(
            "polygon must contain finite values."
        )

    if not (
        0.0 < zmp_scale <= 1.0
    ):
        raise ValueError(
            "zmp_scale must satisfy "
            "0 < zmp_scale <= 1."
        )

    # --------------------------------------------------------
    # Raw bounds
    # --------------------------------------------------------

    raw_x_min = float(
        np.min(
            polygon[:, 0]
        )
    )

    raw_x_max = float(
        np.max(
            polygon[:, 0]
        )
    )

    raw_y_min = float(
        np.min(
            polygon[:, 1]
        )
    )

    raw_y_max = float(
        np.max(
            polygon[:, 1]
        )
    )

    # --------------------------------------------------------
    # Centers
    # --------------------------------------------------------

    center_x = 0.5 * (
        raw_x_min
        +
        raw_x_max
    )

    center_y = 0.5 * (
        raw_y_min
        +
        raw_y_max
    )

    # --------------------------------------------------------
    # Safe scaled bounds
    # --------------------------------------------------------

    x_min = (
        center_x
        +
        zmp_scale
        *
        (
            raw_x_min
            -
            center_x
        )
    )

    x_max = (
        center_x
        +
        zmp_scale
        *
        (
            raw_x_max
            -
            center_x
        )
    )

    y_min = (
        center_y
        +
        zmp_scale
        *
        (
            raw_y_min
            -
            center_y
        )
    )

    y_max = (
        center_y
        +
        zmp_scale
        *
        (
            raw_y_max
            -
            center_y
        )
    )

    return (
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
    )


# ============================================================
# SUPPORT PREVIEW
# ============================================================

def build_support_preview(
    fsm,
    left_rotation,
    right_rotation,
    timestep: float,
    horizon_steps: int,
    foot_toe: float,
    foot_heel: float,
    foot_half_width: float,
    zmp_scale: float,
) -> SupportPreview:
    """
    Build a 2D support preview for the MPC horizon.

    The real WalkingFSM is never modified.

    Instead, a deep copy is propagated into the future
    using the MPC timestep.

    Entry k corresponds to one MPC interval:

        [t0 + k*T, t0 + (k+1)*T)

    Example:

        T_MPC = 0.03 s
        T_DS  = 0.09 s
        T_SS  = 0.18 s

    gives:

        3 DS intervals
        6 SS intervals
    """

    if timestep <= 0.0:
        raise ValueError(
            "timestep must be greater than zero."
        )

    if horizon_steps <= 0:
        raise ValueError(
            "horizon_steps must be greater than zero."
        )

    _validate_foot_geometry(
        foot_toe=foot_toe,
        foot_heel=foot_heel,
        foot_half_width=foot_half_width,
    )

    if not (
        0.0 < zmp_scale <= 1.0
    ):
        raise ValueError(
            "zmp_scale must satisfy "
            "0 < zmp_scale <= 1."
        )

    left_rotation = np.asarray(
        left_rotation,
        dtype=float,
    )

    right_rotation = np.asarray(
        right_rotation,
        dtype=float,
    )

    if left_rotation.shape != (3, 3):
        raise ValueError(
            "left_rotation must have shape (3, 3)."
        )

    if right_rotation.shape != (3, 3):
        raise ValueError(
            "right_rotation must have shape (3, 3)."
        )

    # --------------------------------------------------------
    # Copy FSM for preview
    # --------------------------------------------------------

    preview_fsm = copy.deepcopy(
        fsm
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    times = (
        np.arange(
            horizon_steps,
            dtype=float,
        )
        *
        timestep
    )

    x_min_values = np.zeros(
        horizon_steps,
        dtype=float,
    )

    x_max_values = np.zeros(
        horizon_steps,
        dtype=float,
    )

    y_min_values = np.zeros(
        horizon_steps,
        dtype=float,
    )

    y_max_values = np.zeros(
        horizon_steps,
        dtype=float,
    )

    phase_names = []

    support_sides = []

    polygons = []

    # --------------------------------------------------------
    # Preview loop
    # --------------------------------------------------------

    for k in range(
        horizon_steps
    ):

        state = (
            preview_fsm.get_state()
        )

        polygon = (
            compute_support_polygon(
                left_position=(
                    state.left_contact_position
                ),
                left_rotation=left_rotation,

                right_position=(
                    state.right_contact_position
                ),
                right_rotation=right_rotation,

                phase=state.phase,
                support_side=state.support_side,

                foot_toe=foot_toe,
                foot_heel=foot_heel,
                foot_half_width=(
                    foot_half_width
                ),
            )
        )

        (
            x_min,
            x_max,
            y_min,
            y_max,
        ) = compute_support_bounds(
            polygon=polygon,
            zmp_scale=zmp_scale,
        )

        x_min_values[k] = x_min
        x_max_values[k] = x_max

        y_min_values[k] = y_min
        y_max_values[k] = y_max

        phase_names.append(
            _phase_name(
                state.phase
            )
        )

        support_sides.append(
            str(
                state.support_side
            )
        )

        polygons.append(
            polygon.copy()
        )

        # Advance copied FSM by one MPC interval.
        preview_fsm.update(
            timestep
        )

    return SupportPreview(
        time=times,

        x_min=x_min_values,
        x_max=x_max_values,

        y_min=y_min_values,
        y_max=y_max_values,

        phase=tuple(
            phase_names
        ),

        support_side=tuple(
            support_sides
        ),

        polygons=tuple(
            polygons
        ),
    )