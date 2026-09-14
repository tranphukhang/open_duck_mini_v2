# step_timing_adaptation/adaptive_support_preview.py

from __future__ import annotations

import numpy as np

from lipm_mpc.support_preview import (
    SupportPreview,
    compute_support_polygon,
    compute_support_bounds,
)


# ============================================================
# PHASE NAMES
# ============================================================

INITIAL_DOUBLE_SUPPORT = "INITIAL_DOUBLE_SUPPORT"
SINGLE_SUPPORT = "SINGLE_SUPPORT"
DOUBLE_SUPPORT = "DOUBLE_SUPPORT"


# ============================================================
# ADAPTIVE ONE-STEP SUPPORT PREVIEW
# ============================================================

def build_adaptive_support_preview(
    *,
    current_phase: str,
    phase_time: float,

    initial_double_support_duration: float,
    single_support_duration: float,

    stance_side: str,

    left_initial_position,
    right_initial_position,

    landing_position,

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
    Support preview for one adaptive step.

    Architecture:

        INITIAL DOUBLE SUPPORT
                |
                v
        SINGLE SUPPORT
        duration = adaptive T
                |
                v
        DOUBLE SUPPORT after touchdown

    The landing location and single-support duration are supplied
    online by AdaptiveStepPlanner.

    Unlike lipm_mpc.build_support_preview(), this function does
    not depend on WalkingFSM because the current step location
    and timing are optimization variables.
    """

    phase = str(
        current_phase
    )

    stance_side = str(
        stance_side
    ).strip().lower()

    if stance_side not in (
        "left",
        "right",
    ):
        raise ValueError(
            "stance_side must be 'left' or 'right'."
        )

    phase_time = float(
        phase_time
    )

    initial_double_support_duration = float(
        initial_double_support_duration
    )

    single_support_duration = float(
        single_support_duration
    )

    timestep = float(
        timestep
    )

    horizon_steps = int(
        horizon_steps
    )

    if phase_time < 0.0:
        raise ValueError(
            "phase_time must be >= 0."
        )

    if initial_double_support_duration < 0.0:
        raise ValueError(
            "initial_double_support_duration must be >= 0."
        )

    if single_support_duration <= 0.0:
        raise ValueError(
            "single_support_duration must be positive."
        )

    if timestep <= 0.0:
        raise ValueError(
            "timestep must be positive."
        )

    if horizon_steps < 1:
        raise ValueError(
            "horizon_steps must be >= 1."
        )

    left_initial_position = np.asarray(
        left_initial_position,
        dtype=float,
    ).reshape(3)

    right_initial_position = np.asarray(
        right_initial_position,
        dtype=float,
    ).reshape(3)

    landing_position = np.asarray(
        landing_position,
        dtype=float,
    ).reshape(3)

    left_rotation = np.asarray(
        left_rotation,
        dtype=float,
    ).reshape(3, 3)

    right_rotation = np.asarray(
        right_rotation,
        dtype=float,
    ).reshape(3, 3)

    # ========================================================
    # STORAGE
    # ========================================================

    times = (
        (
            np.arange(
                horizon_steps,
                dtype=float,
            )
            +
            1.0
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

    phase_values = []
    support_values = []
    polygons = []

    # ========================================================
    # PREVIEW LOOP
    # ========================================================

    for k in range(
        horizon_steps
    ):

        future_time = (
            (
                k
                +
                1
            )
            *
            timestep
        )

        # ----------------------------------------------------
        # Determine future walking phase.
        # ----------------------------------------------------

        if (
            phase
            ==
            INITIAL_DOUBLE_SUPPORT
        ):

            future_initial_ds_time = (
                phase_time
                +
                future_time
            )

            if (
                future_initial_ds_time
                <
                initial_double_support_duration
            ):

                future_phase = (
                    DOUBLE_SUPPORT
                )

                future_support_side = (
                    "both"
                )

                future_left_position = (
                    left_initial_position
                )

                future_right_position = (
                    right_initial_position
                )

            else:

                future_ss_time = (
                    future_initial_ds_time
                    -
                    initial_double_support_duration
                )

                if (
                    future_ss_time
                    <
                    single_support_duration
                ):

                    future_phase = (
                        SINGLE_SUPPORT
                    )

                    future_support_side = (
                        stance_side
                    )

                    future_left_position = (
                        left_initial_position
                    )

                    future_right_position = (
                        right_initial_position
                    )

                else:

                    future_phase = (
                        DOUBLE_SUPPORT
                    )

                    future_support_side = (
                        "both"
                    )

                    if stance_side == "left":

                        future_left_position = (
                            left_initial_position
                        )

                        future_right_position = (
                            landing_position
                        )

                    else:

                        future_left_position = (
                            landing_position
                        )

                        future_right_position = (
                            right_initial_position
                        )

        elif (
            phase
            ==
            SINGLE_SUPPORT
        ):

            future_ss_time = (
                phase_time
                +
                future_time
            )

            if (
                future_ss_time
                <
                single_support_duration
            ):

                future_phase = (
                    SINGLE_SUPPORT
                )

                future_support_side = (
                    stance_side
                )

                future_left_position = (
                    left_initial_position
                )

                future_right_position = (
                    right_initial_position
                )

            else:

                future_phase = (
                    DOUBLE_SUPPORT
                )

                future_support_side = (
                    "both"
                )

                if stance_side == "left":

                    future_left_position = (
                        left_initial_position
                    )

                    future_right_position = (
                        landing_position
                    )

                else:

                    future_left_position = (
                        landing_position
                    )

                    future_right_position = (
                        right_initial_position
                    )

        else:

            raise ValueError(
                f"Unsupported current_phase: {phase}"
            )

        # ----------------------------------------------------
        # Existing support-polygon implementation.
        # ----------------------------------------------------

        polygon = compute_support_polygon(
            left_position=(
                future_left_position
            ),
            left_rotation=(
                left_rotation
            ),

            right_position=(
                future_right_position
            ),
            right_rotation=(
                right_rotation
            ),

            phase=(
                future_phase
            ),
            support_side=(
                future_support_side
            ),

            foot_toe=(
                foot_toe
            ),
            foot_heel=(
                foot_heel
            ),
            foot_half_width=(
                foot_half_width
            ),
        )

        (
            x_min,
            x_max,
            y_min,
            y_max,
        ) = compute_support_bounds(
            polygon=(
                polygon
            ),
            zmp_scale=(
                zmp_scale
            ),
        )

        x_min_values[k] = (
            x_min
        )

        x_max_values[k] = (
            x_max
        )

        y_min_values[k] = (
            y_min
        )

        y_max_values[k] = (
            y_max
        )

        phase_values.append(
            future_phase
        )

        support_values.append(
            future_support_side
        )

        polygons.append(
            polygon.copy()
        )

    # ========================================================
    # RESULT
    # ========================================================

    return SupportPreview(
        time=(
            times
        ),

        x_min=(
            x_min_values
        ),
        x_max=(
            x_max_values
        ),

        y_min=(
            y_min_values
        ),
        y_max=(
            y_max_values
        ),

        phase=tuple(
            phase_values
        ),

        support_side=tuple(
            support_values
        ),

        polygons=tuple(
            polygons
        ),
    )