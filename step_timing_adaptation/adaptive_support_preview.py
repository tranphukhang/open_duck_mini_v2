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
# HELPERS
# ============================================================

def _opposite_side(
    side: str,
) -> str:

    if side == "left":
        return "right"

    if side == "right":
        return "left"

    raise ValueError(
        f"Invalid side: {side}"
    )


def _get_contact(
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


def _compute_nominal_target(
    *,
    stance_side,
    left_contact,
    right_contact,
    nominal_left_step_displacement,
    nominal_right_step_displacement,
):

    if stance_side == "left":

        support = (
            left_contact
        )

        swing_contact = (
            right_contact
        )

        displacement = (
            nominal_left_step_displacement
        )

    elif stance_side == "right":

        support = (
            right_contact
        )

        swing_contact = (
            left_contact
        )

        displacement = (
            nominal_right_step_displacement
        )

    else:

        raise ValueError(
            f"Invalid stance_side: {stance_side}"
        )

    target = (
        swing_contact.copy()
    )

    target[0] = (
        support[0]
        +
        displacement[0]
    )

    target[1] = (
        support[1]
        +
        displacement[1]
    )

    # Flat ground:
    # retain the swing-foot contact height.
    target[2] = (
        swing_contact[2]
    )

    return target


def _nominal_step_time(
    stance_side,
    nominal_left_step_time,
    nominal_right_step_time,
):

    if stance_side == "left":

        return float(
            nominal_left_step_time
        )

    if stance_side == "right":

        return float(
            nominal_right_step_time
        )

    raise ValueError(
        f"Invalid stance_side: {stance_side}"
    )


# ============================================================
# MULTI-STEP ADAPTIVE SUPPORT PREVIEW
# ============================================================

def build_adaptive_support_preview(
    *,
    current_phase,
    phase_time,

    initial_double_support_duration,
    double_support_duration,

    current_single_support_duration,

    stance_side,
    swing_side,

    left_contact_position,
    right_contact_position,

    current_landing_position,

    nominal_left_step_displacement,
    nominal_right_step_displacement,

    nominal_left_step_time,
    nominal_right_step_time,

    left_rotation,
    right_rotation,

    timestep,
    horizon_steps,

    foot_toe,
    foot_heel,
    foot_half_width,

    zmp_scale,

    single_support_zmp_half_width,
) -> SupportPreview:
    """
    Multi-step support preview for adaptive walking.

    Current adaptive step:
        uses current_landing_position
        and current_single_support_duration.

    Future steps:
        use nominal step displacement and nominal timing.

    Sequence:

        INITIAL DS
            ->
        SS
            ->
        DS
            ->
        SS
            ->
        DS
            -> ...

    During SS:

        p_ZMP ~= stance foot

    to remain consistent with the DCM model used by the
    Step Timing Adaptation planner.
    """

    # ========================================================
    # INPUT CONVERSION
    # ========================================================

    phase = str(
        current_phase
    )

    elapsed = float(
        phase_time
    )

    stance = str(
        stance_side
    ).lower()

    swing = str(
        swing_side
    ).lower()

    if stance not in (
        "left",
        "right",
    ):

        raise ValueError(
            "stance_side must be left or right."
        )

    if swing != _opposite_side(
        stance
    ):

        raise ValueError(
            "swing_side must be opposite stance_side."
        )

    left_contact = np.asarray(
        left_contact_position,
        dtype=float,
    ).reshape(
        3
    ).copy()

    right_contact = np.asarray(
        right_contact_position,
        dtype=float,
    ).reshape(
        3
    ).copy()

    landing = np.asarray(
        current_landing_position,
        dtype=float,
    ).reshape(
        3
    ).copy()

    nominal_left_disp = np.asarray(
        nominal_left_step_displacement,
        dtype=float,
    ).reshape(
        2
    )

    nominal_right_disp = np.asarray(
        nominal_right_step_displacement,
        dtype=float,
    ).reshape(
        2
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

    dt = float(
        timestep
    )

    N = int(
        horizon_steps
    )

    initial_ds_duration = float(
        initial_double_support_duration
    )

    ds_duration = float(
        double_support_duration
    )

    ss_duration = float(
        current_single_support_duration
    )

    epsilon = float(
        single_support_zmp_half_width
    )

    if phase not in (
        INITIAL_DOUBLE_SUPPORT,
        SINGLE_SUPPORT,
        DOUBLE_SUPPORT,
    ):

        raise ValueError(
            f"Unsupported phase: {phase}"
        )

    if elapsed < 0.0:

        raise ValueError(
            "phase_time must be >= 0."
        )

    if initial_ds_duration <= 0.0:

        raise ValueError(
            "initial_double_support_duration "
            "must be positive."
        )

    if ds_duration <= 0.0:

        raise ValueError(
            "double_support_duration "
            "must be positive."
        )

    if ss_duration <= 0.0:

        raise ValueError(
            "current_single_support_duration "
            "must be positive."
        )

    if dt <= 0.0:

        raise ValueError(
            "timestep must be positive."
        )

    if N < 1:

        raise ValueError(
            "horizon_steps must be >= 1."
        )

    if epsilon <= 0.0:

        raise ValueError(
            "single_support_zmp_half_width "
            "must be positive."
        )

    # ========================================================
    # STORAGE
    # ========================================================

    times = (
        (
            np.arange(
                N,
                dtype=float,
            )
            +
            1.0
        )
        *
        dt
    )

    x_min_values = np.zeros(
        N,
        dtype=float,
    )

    x_max_values = np.zeros(
        N,
        dtype=float,
    )

    y_min_values = np.zeros(
        N,
        dtype=float,
    )

    y_max_values = np.zeros(
        N,
        dtype=float,
    )

    phase_values = []
    support_values = []
    polygons = []

    # ========================================================
    # SIMULATED PREVIEW STATE
    # ========================================================

    preview_phase = (
        phase
    )

    preview_phase_time = (
        elapsed
    )

    preview_stance = (
        stance
    )

    preview_swing = (
        swing
    )

    preview_ss_duration = (
        ss_duration
    )

    preview_landing = (
        landing.copy()
    )

    tolerance = (
        1.0e-12
    )

    # ========================================================
    # PREVIEW LOOP
    # ========================================================

    for k in range(
        N
    ):

        remaining_dt = (
            dt
        )

        # ----------------------------------------------------
        # Advance phase state by one MPC preview timestep.
        # ----------------------------------------------------

        while (
            remaining_dt
            >
            tolerance
        ):

            if (
                preview_phase
                ==
                INITIAL_DOUBLE_SUPPORT
            ):

                phase_duration = (
                    initial_ds_duration
                )

            elif (
                preview_phase
                ==
                SINGLE_SUPPORT
            ):

                phase_duration = (
                    preview_ss_duration
                )

            elif (
                preview_phase
                ==
                DOUBLE_SUPPORT
            ):

                phase_duration = (
                    ds_duration
                )

            else:

                raise RuntimeError(
                    "Invalid preview phase."
                )

            time_to_transition = (
                phase_duration
                -
                preview_phase_time
            )

            # ------------------------------------------------
            # Still inside current phase.
            # ------------------------------------------------

            if (
                remaining_dt
                <
                time_to_transition
                -
                tolerance
            ):

                preview_phase_time += (
                    remaining_dt
                )

                remaining_dt = (
                    0.0
                )

                continue

            # ------------------------------------------------
            # Reach phase boundary.
            # ------------------------------------------------

            remaining_dt -= max(
                time_to_transition,
                0.0,
            )

            preview_phase_time = (
                0.0
            )

            # =================================================
            # INITIAL DS -> FIRST SS
            # =================================================

            if (
                preview_phase
                ==
                INITIAL_DOUBLE_SUPPORT
            ):

                preview_phase = (
                    SINGLE_SUPPORT
                )

                # Keep:
                #
                #   preview_landing
                #   preview_ss_duration
                #
                # supplied by the current runtime state.

            # =================================================
            # SS -> DS
            # =================================================

            elif (
                preview_phase
                ==
                SINGLE_SUPPORT
            ):

                # Commit predicted touchdown.

                if (
                    preview_swing
                    ==
                    "left"
                ):

                    left_contact = (
                        preview_landing.copy()
                    )

                else:

                    right_contact = (
                        preview_landing.copy()
                    )

                # Landed foot becomes next stance.

                preview_stance = (
                    preview_swing
                )

                preview_swing = (
                    _opposite_side(
                        preview_stance
                    )
                )

                # Create nominal NEXT landing target.

                preview_landing = (
                    _compute_nominal_target(
                        stance_side=(
                            preview_stance
                        ),

                        left_contact=(
                            left_contact
                        ),

                        right_contact=(
                            right_contact
                        ),

                        nominal_left_step_displacement=(
                            nominal_left_disp
                        ),

                        nominal_right_step_displacement=(
                            nominal_right_disp
                        ),
                    )
                )

                preview_ss_duration = (
                    _nominal_step_time(
                        preview_stance,
                        nominal_left_step_time,
                        nominal_right_step_time,
                    )
                )

                preview_phase = (
                    DOUBLE_SUPPORT
                )

            # =================================================
            # DS -> NEXT SS
            # =================================================

            elif (
                preview_phase
                ==
                DOUBLE_SUPPORT
            ):

                preview_phase = (
                    SINGLE_SUPPORT
                )

        # ====================================================
        # SUPPORT POLYGON AT THIS PREVIEW SAMPLE
        # ====================================================

        if (
            preview_phase
            ==
            SINGLE_SUPPORT
        ):

            support_side = (
                preview_stance
            )

        else:

            support_side = (
                "both"
            )

        polygon = compute_support_polygon(
            left_position=(
                left_contact
            ),

            left_rotation=(
                left_rotation
            ),

            right_position=(
                right_contact
            ),

            right_rotation=(
                right_rotation
            ),

            phase=(
                preview_phase
                if
                preview_phase
                ==
                SINGLE_SUPPORT
                else
                DOUBLE_SUPPORT
            ),

            support_side=(
                support_side
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

        # ====================================================
        # ZMP CONSTRAINTS
        # ====================================================

        if (
            preview_phase
            ==
            SINGLE_SUPPORT
        ):

            stance_contact = (
                _get_contact(
                    preview_stance,
                    left_contact,
                    right_contact,
                )
            )

            x_min = (
                stance_contact[0]
                -
                epsilon
            )

            x_max = (
                stance_contact[0]
                +
                epsilon
            )

            y_min = (
                stance_contact[1]
                -
                epsilon
            )

            y_max = (
                stance_contact[1]
                +
                epsilon
            )

        else:

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

        # ====================================================
        # STORE
        # ====================================================

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
            preview_phase
        )

        support_values.append(
            support_side
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