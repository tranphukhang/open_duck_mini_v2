import numpy as np


# ============================================================
# DEFAULT RAIBERT PARAMETERS
# ============================================================

DEFAULT_K_X = 0.03
DEFAULT_K_Y = 0.03


# ============================================================
# LONGITUDINAL CORRECTION LIMIT
#
# Feasibility test:
#
#     +/- 2.5 cm : PASS
#
# The test remained feasible up to +/- 3 cm,
# therefore 2.5 cm is kept as a conservative value.
# ============================================================

DEFAULT_MAX_CORRECTION_X = 0.025


# ============================================================
# LATERAL CORRECTION LIMITS
#
# Based on the current Open Duck feasibility test.
#
# LEFT LEG:
#
#     inward  = -Y
#     outward = +Y
#
# RIGHT LEG:
#
#     inward  = +Y
#     outward = -Y
#
# Current conservative values:
#
#     inward  = 0.5 cm
#     outward = 2.0 cm
#
# These values are provisional and should later be validated
# over multiple walking configurations.
# ============================================================

DEFAULT_MAX_CORRECTION_Y_INWARD = 0.005

DEFAULT_MAX_CORRECTION_Y_OUTWARD = 0.020


# ============================================================
# LEG VALIDATION
# ============================================================

def validate_leg(
    leg,
):

    leg = str(
        leg
    ).upper()

    if leg not in (
        "L",
        "R",
    ):

        raise ValueError(
            "leg must be 'L' or 'R'."
        )

    return leg


# ============================================================
# GET LATERAL CLIP LIMITS
#
# World-frame Y convention:
#
# LEFT:
#
#     inward  = -Y
#     outward = +Y
#
# RIGHT:
#
#     inward  = +Y
#     outward = -Y
#
# Returns:
#
#     y_min
#     y_max
# ============================================================

def get_lateral_correction_limits(
    leg,
    max_correction_y_inward=
        DEFAULT_MAX_CORRECTION_Y_INWARD,
    max_correction_y_outward=
        DEFAULT_MAX_CORRECTION_Y_OUTWARD,
):

    leg = validate_leg(
        leg
    )


    max_correction_y_inward = float(
        max_correction_y_inward
    )

    max_correction_y_outward = float(
        max_correction_y_outward
    )


    if max_correction_y_inward < 0.0:

        raise ValueError(
            "max_correction_y_inward must be >= 0."
        )


    if max_correction_y_outward < 0.0:

        raise ValueError(
            "max_correction_y_outward must be >= 0."
        )


    # --------------------------------------------------------
    # LEFT LEG
    #
    # inward  -> negative Y
    # outward -> positive Y
    # --------------------------------------------------------

    if leg == "L":

        y_min = (
            -max_correction_y_inward
        )

        y_max = (
            +max_correction_y_outward
        )


    # --------------------------------------------------------
    # RIGHT LEG
    #
    # outward -> negative Y
    # inward  -> positive Y
    # --------------------------------------------------------

    else:

        y_min = (
            -max_correction_y_outward
        )

        y_max = (
            +max_correction_y_inward
        )


    return (
        float(
            y_min
        ),
        float(
            y_max
        ),
    )


# ============================================================
# RAIBERT VELOCITY CORRECTION
#
# Raw correction:
#
#     delta_px = k_x (v_x - v_x_ref)
#
#     delta_py = k_y (v_y - v_y_ref)
#
# X correction:
#
#     symmetric clipping
#
# Y correction:
#
#     asymmetric clipping according to swing leg geometry
# ============================================================

def compute_raibert_velocity_correction(
    W_v_actual,
    W_v_ref,
    leg,
    k_x=DEFAULT_K_X,
    k_y=DEFAULT_K_Y,
    max_correction_x=
        DEFAULT_MAX_CORRECTION_X,
    max_correction_y_inward=
        DEFAULT_MAX_CORRECTION_Y_INWARD,
    max_correction_y_outward=
        DEFAULT_MAX_CORRECTION_Y_OUTWARD,
):

    # ========================================================
    # INPUT CHECK
    # ========================================================

    leg = validate_leg(
        leg
    )


    W_v_actual = np.asarray(
        W_v_actual,
        dtype=float,
    ).reshape(
        3
    )


    W_v_ref = np.asarray(
        W_v_ref,
        dtype=float,
    ).reshape(
        3
    )


    k_x = float(
        k_x
    )

    k_y = float(
        k_y
    )


    max_correction_x = float(
        max_correction_x
    )


    if max_correction_x < 0.0:

        raise ValueError(
            "max_correction_x must be >= 0."
        )


    # ========================================================
    # VELOCITY ERROR
    # ========================================================

    velocity_error = (
        W_v_actual
        -
        W_v_ref
    )


    # ========================================================
    # RAW RAIBERT CORRECTION
    # ========================================================

    raw_correction = np.zeros(
        3,
        dtype=float,
    )


    raw_correction[
        0
    ] = (
        k_x
        *
        velocity_error[
            0
        ]
    )


    raw_correction[
        1
    ] = (
        k_y
        *
        velocity_error[
            1
        ]
    )


    # No vertical Raibert correction here.
    raw_correction[
        2
    ] = 0.0


    # ========================================================
    # X CLIPPING
    # ========================================================

    x_min = (
        -max_correction_x
    )

    x_max = (
        +max_correction_x
    )


    correction_x = np.clip(
        raw_correction[
            0
        ],
        x_min,
        x_max,
    )


    # ========================================================
    # Y CLIPPING
    # ========================================================

    (
        y_min,
        y_max,

    ) = get_lateral_correction_limits(

        leg=
            leg,

        max_correction_y_inward=
            max_correction_y_inward,

        max_correction_y_outward=
            max_correction_y_outward,
    )


    correction_y = np.clip(
        raw_correction[
            1
        ],
        y_min,
        y_max,
    )


    # ========================================================
    # CLIPPED CORRECTION
    # ========================================================

    clipped_correction = np.array(
        [
            correction_x,
            correction_y,
            0.0,
        ],
        dtype=float,
    )


    # ========================================================
    # CLIPPING FLAGS
    # ========================================================

    x_was_clipped = not np.isclose(
        raw_correction[
            0
        ],
        correction_x,
    )


    y_was_clipped = not np.isclose(
        raw_correction[
            1
        ],
        correction_y,
    )


    # ========================================================
    # IDENTIFY Y DIRECTION
    # ========================================================

    if np.isclose(
        raw_correction[
            1
        ],
        0.0,
    ):

        lateral_direction = (
            "none"
        )


    elif leg == "L":

        if raw_correction[
            1
        ] > 0.0:

            lateral_direction = (
                "outward"
            )

        else:

            lateral_direction = (
                "inward"
            )


    else:

        if raw_correction[
            1
        ] > 0.0:

            lateral_direction = (
                "inward"
            )

        else:

            lateral_direction = (
                "outward"
            )


    # ========================================================
    # RETURN DIAGNOSTICS
    # ========================================================

    return {

        "leg":
            leg,

        "velocity_error":
            velocity_error,

        "raw_correction":
            raw_correction,

        "clipped_correction":
            clipped_correction,

        "x_was_clipped":
            bool(
                x_was_clipped
            ),

        "y_was_clipped":
            bool(
                y_was_clipped
            ),

        "lateral_direction":
            lateral_direction,

        "k_x":
            k_x,

        "k_y":
            k_y,

        "x_min":
            float(
                x_min
            ),

        "x_max":
            float(
                x_max
            ),

        "y_min":
            float(
                y_min
            ),

        "y_max":
            float(
                y_max
            ),

        "max_correction_x":
            max_correction_x,

        "max_correction_y_inward":
            float(
                max_correction_y_inward
            ),

        "max_correction_y_outward":
            float(
                max_correction_y_outward
            ),
    }


# ============================================================
# COMPLETE RAIBERT FOOTHOLD
#
# Paper-style formulation:
#
#     Pf =
#
#     Phip
#     +
#     0.5 * v_actual * T_remaining
#     +
#     correction
#
# Then:
#
#     Pf_z = foothold_z
#
# Important:
#
#     T_remaining is the remaining swing time,
#     NOT the complete swing duration.
# ============================================================

def compute_raibert_foothold(
    W_p_hip,
    W_v_actual,
    W_v_ref,
    T_remaining,
    foothold_z,
    leg,
    k_x=DEFAULT_K_X,
    k_y=DEFAULT_K_Y,
    max_correction_x=
        DEFAULT_MAX_CORRECTION_X,
    max_correction_y_inward=
        DEFAULT_MAX_CORRECTION_Y_INWARD,
    max_correction_y_outward=
        DEFAULT_MAX_CORRECTION_Y_OUTWARD,
):

    # ========================================================
    # INPUTS
    # ========================================================

    leg = validate_leg(
        leg
    )


    W_p_hip = np.asarray(
        W_p_hip,
        dtype=float,
    ).reshape(
        3
    )


    W_v_actual = np.asarray(
        W_v_actual,
        dtype=float,
    ).reshape(
        3
    )


    W_v_ref = np.asarray(
        W_v_ref,
        dtype=float,
    ).reshape(
        3
    )


    T_remaining = float(
        T_remaining
    )


    foothold_z = float(
        foothold_z
    )


    if T_remaining < 0.0:

        raise ValueError(
            "T_remaining must be >= 0."
        )


    # ========================================================
    # NOMINAL RAIBERT TERM
    #
    # Paper:
    #
    #     P_nominal =
    #
    #     P_hip
    #     +
    #     0.5 * v_actual * T_remaining
    # ========================================================

    W_p_nominal = (
        W_p_hip
        +
        0.5
        *
        W_v_actual
        *
        T_remaining
    )


    # ========================================================
    # VELOCITY CORRECTION
    # ========================================================

    correction_result = (
        compute_raibert_velocity_correction(

            W_v_actual=
                W_v_actual,

            W_v_ref=
                W_v_ref,

            leg=
                leg,

            k_x=
                k_x,

            k_y=
                k_y,

            max_correction_x=
                max_correction_x,

            max_correction_y_inward=
                max_correction_y_inward,

            max_correction_y_outward=
                max_correction_y_outward,
        )
    )


    correction = (
        correction_result[
            "clipped_correction"
        ]
    )


    # ========================================================
    # FINAL FOOTHOLD
    # ========================================================

    W_p_foothold = (
        W_p_nominal
        +
        correction
    )


    # Paper explicitly overwrites foothold Z.
    W_p_foothold[
        2
    ] = foothold_z


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "leg":
            leg,

        "W_p_foothold":
            W_p_foothold,

        "W_p_nominal":
            W_p_nominal,

        "W_p_hip":
            W_p_hip.copy(),

        "W_v_actual":
            W_v_actual.copy(),

        "W_v_ref":
            W_v_ref.copy(),

        "T_remaining":
            T_remaining,

        "foothold_z":
            foothold_z,

        "velocity_error":
            correction_result[
                "velocity_error"
            ],

        "raw_correction":
            correction_result[
                "raw_correction"
            ],

        "clipped_correction":
            correction_result[
                "clipped_correction"
            ],

        "x_was_clipped":
            correction_result[
                "x_was_clipped"
            ],

        "y_was_clipped":
            correction_result[
                "y_was_clipped"
            ],

        "lateral_direction":
            correction_result[
                "lateral_direction"
            ],

        "x_min":
            correction_result[
                "x_min"
            ],

        "x_max":
            correction_result[
                "x_max"
            ],

        "y_min":
            correction_result[
                "y_min"
            ],

        "y_max":
            correction_result[
                "y_max"
            ],
    }