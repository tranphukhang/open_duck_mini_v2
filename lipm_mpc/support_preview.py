# lipm_mpc/support_preview.py

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# IMPORT GEOMETRY FROM FOOTSTEP_PLANNING
# ============================================================

if not __package__:
    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CURRENT_DIR.parent

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

from footstep_planning.walking_visualization import (
    convex_hull_2d,
    yaw_rotation,
)


# ============================================================
# DATA
# ============================================================

@dataclass
class SupportPreviewX:
    """
    Support preview for the sagittal x-axis.

    Each entry k corresponds to the support region over:

        [t0 + k*T, t0 + (k+1)*T)

    where T is the MPC timestep.
    """

    time: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray

    phase: tuple[str, ...]
    support_side: tuple[str, ...]

    polygons: tuple[np.ndarray, ...]

    @property
    def horizon_steps(self) -> int:
        return len(self.x_min)


# ============================================================
# VALIDATION
# ============================================================

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
            "Foot position must have shape (3,)."
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            "Foot rotation must have shape (3, 3)."
        )

    if not np.all(np.isfinite(position)):
        raise ValueError(
            "Foot position must be finite."
        )

    if not np.all(np.isfinite(rotation)):
        raise ValueError(
            "Foot rotation must be finite."
        )

    return position, rotation


def _phase_name(phase) -> str:

    if hasattr(phase, "value"):
        return str(phase.value)

    return str(phase)


# ============================================================
# SOLE GEOMETRY
# ============================================================

def compute_sole_corners(
    position,
    rotation,
    foot_toe: float,
    foot_heel: float,
    foot_half_width: float,
) -> np.ndarray:
    """
    Compute four sole corners in the world frame.

    The geometry follows the same convention as
    footstep_planning.walking_visualization:

          +x
           ^
           |
       toe +---------+
           |         |
           |         |
      heel +---------+

    Foot-site position is used as the local origin.

    Returns
    -------
    corners : ndarray, shape (4, 3)
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

    # Keep only the yaw component, exactly as done
    # for the support polygon visualization.
    R = yaw_rotation(rotation)

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
            position + R @ corner
            for corner in local_corners
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
    Compute the 2D support polygon.

    SINGLE_SUPPORT:
        convex hull of the support foot.

    Other support phases:
        convex hull of both feet.

    Returns
    -------
    polygon : ndarray, shape (M, 2)
        Counter-clockwise convex-hull vertices in the x-y plane.
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

    phase_name = _phase_name(phase)

    if phase_name == "SINGLE_SUPPORT":

        if support_side == "left":
            points = left_corners

        elif support_side == "right":
            points = right_corners

        else:
            raise ValueError(
                "support_side must be 'left' or 'right' "
                "during SINGLE_SUPPORT."
            )

    else:
        # INITIAL_DOUBLE_SUPPORT
        # DOUBLE_SUPPORT
        # FINAL_DOUBLE_SUPPORT
        # FINISHED
        points = np.vstack(
            [
                left_corners,
                right_corners,
            ]
        )

    hull = convex_hull_2d(points)

    hull = np.asarray(
        hull,
        dtype=float,
    )

    if hull.ndim != 2:
        raise RuntimeError(
            "Invalid convex hull."
        )

    if hull.shape[0] == 0:
        raise RuntimeError(
            "Support polygon is empty."
        )

    # convex_hull_2d from walking_visualization returns
    # [x, y, z]. MPC only needs the ground-plane coordinates.
    return hull[:, :2].copy()


# ============================================================
# X-AXIS SUPPORT BOUNDS
# ============================================================

def compute_support_bounds_x(
    polygon,
    zmp_scale: float,
) -> tuple[float, float]:
    """
    Compute safe x-axis ZMP bounds from a support polygon.

    Raw:
        x_min <= x_Z <= x_max

    Scaled:
        x_c + scale * (x - x_c)

    where:
        x_c = (x_min + x_max) / 2
    """

    polygon = np.asarray(
        polygon,
        dtype=float,
    )

    if polygon.ndim != 2:
        raise ValueError(
            "polygon must be a 2D array."
        )

    if polygon.shape[0] == 0:
        raise ValueError(
            "polygon cannot be empty."
        )

    if polygon.shape[1] < 2:
        raise ValueError(
            "polygon must contain x and y coordinates."
        )

    if not np.all(np.isfinite(polygon)):
        raise ValueError(
            "polygon must contain finite values."
        )

    if not (
        0.0 < zmp_scale <= 1.0
    ):
        raise ValueError(
            "zmp_scale must satisfy 0 < zmp_scale <= 1."
        )

    raw_x_min = float(
        np.min(polygon[:, 0])
    )

    raw_x_max = float(
        np.max(polygon[:, 0])
    )

    center_x = 0.5 * (
        raw_x_min
        +
        raw_x_max
    )

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

    return float(x_min), float(x_max)


# ============================================================
# SUPPORT PREVIEW
# ============================================================

def build_support_preview_x(
    fsm,
    left_rotation,
    right_rotation,
    timestep: float,
    horizon_steps: int,
    foot_toe: float,
    foot_heel: float,
    foot_half_width: float,
    zmp_scale: float,
) -> SupportPreviewX:
    """
    Build the support-region preview for x-MPC.

    The real walking FSM is NOT modified.

    A deep copy of the FSM is propagated into the future,
    therefore the preview automatically follows the same:

        - initial double support
        - start half-step
        - single support
        - double support
        - normal steps
        - closing step
        - final double support

    logic as footstep_planning.walking_fsm.

    Preview convention
    ------------------
    Entry k describes the support region during:

        [t0 + k*T, t0 + (k+1)*T)

    Therefore, with:

        T_DS = 0.09 s
        T_SS = 0.18 s
        T_MPC = 0.03 s

    a complete DS produces 3 preview entries and
    a complete SS produces 6 preview entries.
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
            "zmp_scale must satisfy 0 < zmp_scale <= 1."
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
    # IMPORTANT:
    # Never advance the real walking FSM while previewing.
    # --------------------------------------------------------

    preview_fsm = copy.deepcopy(fsm)

    times = np.arange(
        horizon_steps,
        dtype=float,
    ) * timestep

    x_min_values = np.zeros(
        horizon_steps,
        dtype=float,
    )

    x_max_values = np.zeros(
        horizon_steps,
        dtype=float,
    )

    phase_names = []
    support_sides = []
    polygons = []

    # --------------------------------------------------------
    # Build preview
    # --------------------------------------------------------

    for k in range(horizon_steps):

        state = preview_fsm.get_state()

        polygon = compute_support_polygon(
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
            foot_half_width=foot_half_width,
        )

        x_min, x_max = compute_support_bounds_x(
            polygon=polygon,
            zmp_scale=zmp_scale,
        )

        x_min_values[k] = x_min
        x_max_values[k] = x_max

        phase_names.append(
            _phase_name(state.phase)
        )

        support_sides.append(
            str(state.support_side)
        )

        polygons.append(
            polygon.copy()
        )

        # Move the copied FSM one MPC interval forward.
        preview_fsm.update(
            timestep
        )

    return SupportPreviewX(
        time=times,
        x_min=x_min_values,
        x_max=x_max_values,
        phase=tuple(phase_names),
        support_side=tuple(support_sides),
        polygons=tuple(polygons),
    )