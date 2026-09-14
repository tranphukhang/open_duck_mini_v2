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

    single_support_zmp_half_width: float,
) -> SupportPreview:
    """
    Build the future support preview for one adaptive step.

    Phase sequence:

        INITIAL DOUBLE SUPPORT
                |
                v
        SINGLE SUPPORT
        duration = adaptive T
                |
                v
        DOUBLE SUPPORT AFTER TOUCHDOWN


    IMPORTANT MODEL CONSISTENCY
    ---------------------------

    During DOUBLE SUPPORT:

        ZMP is allowed to move inside the support polygon.

    During SINGLE SUPPORT:

        ZMP is constrained to a very small box around
        the stance-foot reference point u0:

            u0_x - eps <= ZMP_x <= u0_x + eps
            u0_y - eps <= ZMP_y <= u0_y + eps

    Therefore approximately:

        p_ZMP = u0

    and the LIPM DCM dynamics become:

        xi_dot = omega * (xi - u0)

    which is consistent with the model used by the
    Step Timing Adaptation planner.

    The true support polygon is still stored in
    SupportPreview.polygons for diagnostics / visualization.
    """

    # ========================================================
    # INPUTS
    # ========================================================

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

    zmp_scale = float(
        zmp_scale
    )

    single_support_zmp_half_width = float(
        single_support_zmp_half_width
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

    if not (
        0.0
        <
        zmp_scale
        <=
        1.0
    ):

        raise ValueError(
            "zmp_scale must satisfy 0 < zmp_scale <= 1."
        )

    if (
        single_support_zmp_half_width
        <=
        0.0
    ):

        raise ValueError(
            "single_support_zmp_half_width "
            "must be positive."
        )

    # ========================================================
    # POSES
    # ========================================================

    left_initial_position = np.asarray(
        left_initial_position,
        dtype=float,
    ).reshape(
        3
    )

    right_initial_position = np.asarray(
        right_initial_position,
        dtype=float,
    ).reshape(
        3
    )

    landing_position = np.asarray(
        landing_position,
        dtype=float,
    ).reshape(
        3
    )

    left_rotation = np.asarray(
        left_rotation,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    right_rotation = np.asarray(
        right_rotation,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    # ========================================================
    # STANCE POINT u0
    #
    # This is the same support point supplied to:
    #
    #     AdaptiveStepPlanner.solve_adaptive_step(...)
    #
    # ========================================================

    if stance_side == "left":

        stance_position = (
            left_initial_position
        )

    else:

        stance_position = (
            right_initial_position
        )

    stance_xy = (
        stance_position[
            0:2
        ].copy()
    )

    # ========================================================
    # TIME VECTOR
    #
    # preview[k] corresponds to:
    #
    #     t_current + (k + 1) * timestep
    #
    # exactly like lipm_mpc.support_preview.
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

    # ========================================================
    # STORAGE
    # ========================================================

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

        # ====================================================
        # FUTURE PHASE
        # ====================================================

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

            # ------------------------------------------------
            # Still in initial DS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Initial DS has ended
            # ------------------------------------------------

            else:

                future_ss_time = (
                    future_initial_ds_time
                    -
                    initial_double_support_duration
                )

                # --------------------------------------------
                # Future sample lies inside SS
                # --------------------------------------------

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

                # --------------------------------------------
                # Future sample lies after touchdown
                # --------------------------------------------

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

        # ====================================================
        # CURRENTLY IN SINGLE SUPPORT
        # ====================================================

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

            # ------------------------------------------------
            # Still in current SS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # After touchdown
            # ------------------------------------------------

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
                f"Unsupported current_phase: "
                f"{phase}"
            )

        # ====================================================
        # TRUE SUPPORT POLYGON
        #
        # Keep this even during SS for diagnostics and future
        # visualization.
        # ====================================================

        polygon = (
            compute_support_polygon(
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
        )

        # ====================================================
        # MPC ZMP BOUNDS
        # ====================================================

        if (
            future_phase
            ==
            SINGLE_SUPPORT
        ):

            # ------------------------------------------------
            # Khadiv-compatible SS model:
            #
            #       p_ZMP ~= u0
            #
            # LIPMMPC1D requires strict:
            #
            #       lower < upper
            #
            # so a tiny interval is used instead of an exact
            # equality constraint.
            # ------------------------------------------------

            epsilon = (
                single_support_zmp_half_width
            )

            x_min = (
                stance_xy[0]
                -
                epsilon
            )

            x_max = (
                stance_xy[0]
                +
                epsilon
            )

            y_min = (
                stance_xy[1]
                -
                epsilon
            )

            y_max = (
                stance_xy[1]
                +
                epsilon
            )

        else:

            # ------------------------------------------------
            # Double support:
            #
            # MPC may exploit the physical support region.
            # ------------------------------------------------

            (
                x_min,
                x_max,
                y_min,
                y_max,
            ) = (
                compute_support_bounds(
                    polygon=(
                        polygon
                    ),

                    zmp_scale=(
                        zmp_scale
                    ),
                )
            )

        # ====================================================
        # STORE
        # ====================================================

        x_min_values[
            k
        ] = (
            x_min
        )

        x_max_values[
            k
        ] = (
            x_max
        )

        y_min_values[
            k
        ] = (
            y_min
        )

        y_max_values[
            k
        ] = (
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