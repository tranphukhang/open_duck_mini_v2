import numpy as np


# ============================================================
# RAIBERT FOOT-PLACEMENT PARAMETERS
# ============================================================
#
# Paper baseline:
#
#     k_x = 0.03
#     k_y = 0.03
#
# The original paper implementation uses:
#
#     max correction x = 0.30 m
#     max correction y = 0.30 m
#
# However, Open Duck Mini is much smaller.
#
# For the first implementation we use provisional values:
#
#     x: +/- 2.5 cm
#     y: +/- 2.0 cm
#
# These values are NOT final.
#
# They will later be identified using:
#
#     Raibert
#        ->
#     Bezier
#        ->
#     IK
#        ->
#     joint-limit / singularity / reachability test
#
# ============================================================

DEFAULT_K_X = 0.03
DEFAULT_K_Y = 0.03

DEFAULT_MAX_CORRECTION_X = 0.025
DEFAULT_MAX_CORRECTION_Y = 0.020


# ============================================================
# VALIDATION
# ============================================================

def _validate_vector3(
    value,
    name,
):

    value = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if value.shape != (3,):

        raise ValueError(
            f"{name} phải có shape (3,), "
            f"nhưng nhận được {value.shape}."
        )

    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )

    return value


def _validate_nonnegative_scalar(
    value,
    name,
):

    value = float(
        value
    )

    if (
        not np.isfinite(
            value
        )
        or
        value < 0.0
    ):

        raise ValueError(
            f"{name} phải hữu hạn và >= 0."
        )

    return value


def _validate_finite_scalar(
    value,
    name,
):

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):

        raise ValueError(
            f"{name} phải hữu hạn."
        )

    return value


# ============================================================
# RAIBERT FEEDBACK CORRECTION
# ============================================================

def compute_raibert_velocity_correction(
    W_v_actual,
    W_v_ref,
    k_x=DEFAULT_K_X,
    k_y=DEFAULT_K_Y,
    max_correction_x=DEFAULT_MAX_CORRECTION_X,
    max_correction_y=DEFAULT_MAX_CORRECTION_Y,
):
    """
    Compute the velocity-feedback correction term used by
    Raibert foot placement.

    Parameters
    ----------
    W_v_actual : array_like, shape (3,)
        Actual body/CoM velocity expressed in World frame [m/s].

    W_v_ref : array_like, shape (3,)
        Desired velocity expressed in World frame [m/s].

    k_x : float
        Raibert feedback gain along World x.

    k_y : float
        Raibert feedback gain along World y.

    max_correction_x : float
        Maximum absolute x correction [m].

    max_correction_y : float
        Maximum absolute y correction [m].

    Returns
    -------
    result : dict

        {
            "raw_correction": ...,
            "clipped_correction": ...,
            "velocity_error": ...,
            "clipped_x": ...,
            "clipped_y": ...,
        }

    Notes
    -----
    Paper formula:

        correction_x = k_x * (v_x - v_x_ref)

        correction_y = k_y * (v_y - v_y_ref)

    Only the correction term is clipped.

    The entire foothold position is NOT clipped here.
    """

    # ========================================================
    # INPUTS
    # ========================================================

    W_v_actual = _validate_vector3(
        W_v_actual,
        "W_v_actual",
    )

    W_v_ref = _validate_vector3(
        W_v_ref,
        "W_v_ref",
    )

    k_x = _validate_nonnegative_scalar(
        k_x,
        "k_x",
    )

    k_y = _validate_nonnegative_scalar(
        k_y,
        "k_y",
    )

    max_correction_x = _validate_nonnegative_scalar(
        max_correction_x,
        "max_correction_x",
    )

    max_correction_y = _validate_nonnegative_scalar(
        max_correction_y,
        "max_correction_y",
    )


    # ========================================================
    # VELOCITY ERROR
    #
    # Same sign convention as the paper:
    #
    #     e_v = v_actual - v_ref
    # ========================================================

    velocity_error = (
        W_v_actual
        -
        W_v_ref
    )


    # ========================================================
    # RAW FEEDBACK CORRECTION
    # ========================================================

    correction_raw = np.zeros(
        3,
        dtype=float,
    )

    correction_raw[0] = (
        k_x
        *
        velocity_error[0]
    )

    correction_raw[1] = (
        k_y
        *
        velocity_error[1]
    )

    # No vertical correction from Raibert
    correction_raw[2] = 0.0


    # ========================================================
    # CLIP CORRECTION
    # ========================================================

    correction_clipped = (
        correction_raw.copy()
    )

    correction_clipped[0] = np.clip(
        correction_raw[0],
        -max_correction_x,
        +max_correction_x,
    )

    correction_clipped[1] = np.clip(
        correction_raw[1],
        -max_correction_y,
        +max_correction_y,
    )

    correction_clipped[2] = 0.0


    # ========================================================
    # CLIPPING FLAGS
    # ========================================================

    clipped_x = bool(
        not np.isclose(
            correction_raw[0],
            correction_clipped[0],
        )
    )

    clipped_y = bool(
        not np.isclose(
            correction_raw[1],
            correction_clipped[1],
        )
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "velocity_error":
            velocity_error.copy(),

        "raw_correction":
            correction_raw.copy(),

        "clipped_correction":
            correction_clipped.copy(),

        "clipped_x":
            clipped_x,

        "clipped_y":
            clipped_y,

        "k_x":
            float(
                k_x
            ),

        "k_y":
            float(
                k_y
            ),

        "max_correction_x":
            float(
                max_correction_x
            ),

        "max_correction_y":
            float(
                max_correction_y
            ),
    }


# ============================================================
# RAIBERT FOOT PLACEMENT
# ============================================================

def compute_raibert_foothold(
    W_p_hip,
    W_v_actual,
    W_v_ref,
    T_remaining,
    foothold_z,
    k_x=DEFAULT_K_X,
    k_y=DEFAULT_K_Y,
    max_correction_x=DEFAULT_MAX_CORRECTION_X,
    max_correction_y=DEFAULT_MAX_CORRECTION_Y,
):
    """
    Compute desired foothold using the Raibert heuristic.

    Architecture
    ------------

        W_p_hip
        W_v_actual
        W_v_ref
        T_remaining
               |
               v
           RAIBERT
               |
               v
        W_p_foothold
               |
               v
           BEZIER
               |
               v
              IK


    Formula
    -------

        p_f
          =
        p_hip
          +
        0.5 * v_actual * T_remaining
          +
        k * (v_actual - v_ref)


    with the feedback correction clipped along x and y.


    Parameters
    ----------
    W_p_hip : array_like, shape (3,)
        Current swing-leg hip position expressed in World frame [m].

    W_v_actual : array_like, shape (3,)
        Actual body/CoM velocity expressed in World frame [m/s].

    W_v_ref : array_like, shape (3,)
        Desired body velocity expressed in World frame [m/s].

    T_remaining : float
        Remaining swing time [s].

        IMPORTANT:

        This follows the paper implementation:

            swingTimes[foot]

        Therefore this quantity decreases during the swing.

        At swing start:

            T_remaining = T_swing

        Near touchdown:

            T_remaining -> 0


    foothold_z : float
        Desired foothold height in World frame [m].

        For current flat-ground simulation this will normally
        be a constant ground-related value.


    Returns
    -------
    result : dict

        {
            "W_p_foothold": ...,

            "W_p_nominal": ...,

            "W_p_hip": ...,

            "velocity_error": ...,

            "raw_correction": ...,

            "clipped_correction": ...,

            "clipped_x": ...,

            "clipped_y": ...,

            "T_remaining": ...,
        }
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    W_p_hip = _validate_vector3(
        W_p_hip,
        "W_p_hip",
    )

    W_v_actual = _validate_vector3(
        W_v_actual,
        "W_v_actual",
    )

    W_v_ref = _validate_vector3(
        W_v_ref,
        "W_v_ref",
    )

    T_remaining = _validate_nonnegative_scalar(
        T_remaining,
        "T_remaining",
    )

    foothold_z = _validate_finite_scalar(
        foothold_z,
        "foothold_z",
    )


    # ========================================================
    # 2. NOMINAL RAIBERT LANDING POSITION
    #
    # Paper:
    #
    #     Pf_world
    #
    #       =
    #
    #     hip_pos
    #
    #       +
    #
    #     0.5 * vWorld * swingTimeRemaining
    #
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
    # 3. VELOCITY FEEDBACK CORRECTION
    # ========================================================

    correction_result = (
        compute_raibert_velocity_correction(

            W_v_actual=
                W_v_actual,

            W_v_ref=
                W_v_ref,

            k_x=
                k_x,

            k_y=
                k_y,

            max_correction_x=
                max_correction_x,

            max_correction_y=
                max_correction_y,
        )
    )


    correction = (
        correction_result[
            "clipped_correction"
        ]
    )


    # ========================================================
    # 4. FINAL FOOTHOLD
    # ========================================================

    W_p_foothold = (
        W_p_nominal
        +
        correction
    )


    # Paper explicitly overwrites foothold z.
    #
    # Therefore vertical body velocity does not change the
    # final foothold height.

    W_p_foothold[2] = (
        foothold_z
    )


    # ========================================================
    # 5. OUTPUT VALIDATION
    # ========================================================

    if not np.all(
        np.isfinite(
            W_p_foothold
        )
    ):

        raise RuntimeError(
            "Raibert sinh foothold chứa NaN hoặc Inf."
        )


    # ========================================================
    # 6. RETURN
    # ========================================================

    return {

        "W_p_foothold":
            W_p_foothold.copy(),

        "W_p_nominal":
            W_p_nominal.copy(),

        "W_p_hip":
            W_p_hip.copy(),

        "W_v_actual":
            W_v_actual.copy(),

        "W_v_ref":
            W_v_ref.copy(),

        "velocity_error":
            correction_result[
                "velocity_error"
            ].copy(),

        "raw_correction":
            correction_result[
                "raw_correction"
            ].copy(),

        "clipped_correction":
            correction_result[
                "clipped_correction"
            ].copy(),

        "clipped_x":
            correction_result[
                "clipped_x"
            ],

        "clipped_y":
            correction_result[
                "clipped_y"
            ],

        "T_remaining":
            float(
                T_remaining
            ),

        "foothold_z":
            float(
                foothold_z
            ),

        "k_x":
            correction_result[
                "k_x"
            ],

        "k_y":
            correction_result[
                "k_y"
            ],

        "max_correction_x":
            correction_result[
                "max_correction_x"
            ],

        "max_correction_y":
            correction_result[
                "max_correction_y"
            ],
    }