import numpy as np


# ============================================================
# QUINTIC SMOOTHSTEP
#
# Normalized quintic polynomial:
#
#     s(r)
#
#       =
#
#     10 r^3
#       -
#     15 r^4
#       +
#     6 r^5
#
#
# Boundary conditions:
#
#     s(0)   = 0
#     s(1)   = 1
#
#     s'(0)  = 0
#     s'(1)  = 0
#
#     s''(0) = 0
#     s''(1) = 0
#
# ============================================================


def quintic_smoothstep(
    r
):
    """
    Evaluate normalized quintic smoothstep and its first two
    derivatives with respect to normalized phase r.

    Parameters
    ----------
    r : float
        Normalized phase:

            0 <= r <= 1

    Returns
    -------
    s : float
        Quintic interpolation value.

    ds_dr : float
        First derivative with respect to r.

    d2s_dr2 : float
        Second derivative with respect to r.
    """

    r = float(
        r
    )


    if (
        not np.isfinite(
            r
        )
    ):

        raise ValueError(
            "r phải hữu hạn."
        )


    if (
        r < 0.0
        or
        r > 1.0
    ):

        raise ValueError(
            "r phải nằm trong [0, 1]."
        )


    s = (
        10.0
        *
        r**3

        -

        15.0
        *
        r**4

        +

        6.0
        *
        r**5
    )


    ds_dr = (
        30.0
        *
        r**2

        -

        60.0
        *
        r**3

        +

        30.0
        *
        r**4
    )


    d2s_dr2 = (
        60.0
        *
        r

        -

        180.0
        *
        r**2

        +

        120.0
        *
        r**3
    )


    return (
        float(
            s
        ),
        float(
            ds_dr
        ),
        float(
            d2s_dr2
        )
    )


# ============================================================
# VECTOR VALIDATION
# ============================================================


def _validate_vector3(
    value,
    name
):

    value = np.asarray(
        value,
        dtype=float
    ).reshape(-1)


    if value.shape != (
        3,
    ):

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


# ============================================================
# GENERAL QUINTIC SEGMENT
# ============================================================


def evaluate_quintic_segment(
    p_initial,
    p_final,
    phase,
    duration
):
    """
    Evaluate a quintic trajectory segment between two vectors.

    Boundary conditions:

        p(0) = p_initial
        p(T) = p_final

        v(0) = 0
        v(T) = 0

        a(0) = 0
        a(T) = 0

    Parameters
    ----------
    p_initial : array_like
        Initial vector.

    p_final : array_like
        Final vector.

    phase : float
        Normalized phase:

            0 <= phase <= 1

    duration : float
        Physical duration [s].

    Returns
    -------
    position : np.ndarray

    velocity : np.ndarray

    acceleration : np.ndarray
    """

    p_initial = np.asarray(
        p_initial,
        dtype=float
    ).reshape(-1)


    p_final = np.asarray(
        p_final,
        dtype=float
    ).reshape(-1)


    if (
        p_initial.shape
        !=
        p_final.shape
    ):

        raise ValueError(
            "p_initial và p_final phải có cùng shape."
        )


    if not np.all(
        np.isfinite(
            p_initial
        )
    ):

        raise ValueError(
            "p_initial chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            p_final
        )
    ):

        raise ValueError(
            "p_final chứa NaN hoặc Inf."
        )


    duration = float(
        duration
    )


    if (
        not np.isfinite(
            duration
        )
        or
        duration <= 0.0
    ):

        raise ValueError(
            "duration phải hữu hạn và > 0."
        )


    (
        s,
        ds_dr,
        d2s_dr2

    ) = quintic_smoothstep(
        phase
    )


    delta = (
        p_final
        -
        p_initial
    )


    position = (
        p_initial
        +
        s
        *
        delta
    )


    # phase:
    #
    #     r = t / T
    #
    # therefore:
    #
    #     dr/dt = 1/T
    #

    velocity = (
        ds_dr
        /
        duration
        *
        delta
    )


    acceleration = (
        d2s_dr2
        /
        duration**2
        *
        delta
    )


    return (
        position,
        velocity,
        acceleration
    )


# ============================================================
# QUINTIC SWING-FOOT TRAJECTORY
# ============================================================


def evaluate_quintic_swing_trajectory(
    W_p_liftoff,
    W_p_foothold,
    swing_phase,
    T_swing,
    swing_height
):
    """
    Generate one swing-foot reference using quintic
    interpolation.

    Architecture
    ------------

    Horizontal motion:

        x, y

    use one quintic trajectory over the complete swing:

        liftoff
            ->
        foothold


    Vertical motion:

        z

    uses two quintic segments:

        liftoff
            ->
        apex
            ->
        touchdown


    Therefore:

        vertical velocity = 0

    at:

        liftoff
        apex
        touchdown


    while horizontal motion continues through the apex.


    Parameters
    ----------
    W_p_liftoff : array_like, shape (3,)
        Actual swing-foot position at liftoff in World frame.

    W_p_foothold : array_like, shape (3,)
        Desired touchdown position in World frame.

    swing_phase : float
        Externally synchronized swing phase:

            0 <= swing_phase <= 1

        Typically:

            swing_phase = t_phase / T_SS

    T_swing : float
        Swing duration [s].

        For current walking architecture:

            T_swing = T_SS

    swing_height : float
        Additional swing-foot clearance [m].

    Returns
    -------
    result : dict

        {
            "W_p_swing_des": ...,
            "W_v_swing_des": ...,
            "W_a_swing_des": ...,
            "W_p_apex": ...,
            "swing_phase": ...,
        }
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    W_p_liftoff = _validate_vector3(
        W_p_liftoff,
        "W_p_liftoff"
    )


    W_p_foothold = _validate_vector3(
        W_p_foothold,
        "W_p_foothold"
    )


    swing_phase = float(
        swing_phase
    )


    if not np.isfinite(
        swing_phase
    ):

        raise ValueError(
            "swing_phase phải hữu hạn."
        )


    if (
        swing_phase < 0.0
        or
        swing_phase > 1.0
    ):

        raise ValueError(
            "swing_phase phải nằm trong [0, 1]."
        )


    T_swing = float(
        T_swing
    )


    if (
        not np.isfinite(
            T_swing
        )
        or
        T_swing <= 0.0
    ):

        raise ValueError(
            "T_swing phải hữu hạn và > 0."
        )


    swing_height = float(
        swing_height
    )


    if (
        not np.isfinite(
            swing_height
        )
        or
        swing_height < 0.0
    ):

        raise ValueError(
            "swing_height phải hữu hạn và >= 0."
        )


    # ========================================================
    # 2. HORIZONTAL QUINTIC MOTION
    #
    # Full swing:
    #
    #     phi = 0 -> 1
    # ========================================================

    (
        p_xy,
        v_xy,
        a_xy

    ) = evaluate_quintic_segment(

        p_initial=(
            W_p_liftoff[
                0:2
            ]
        ),

        p_final=(
            W_p_foothold[
                0:2
            ]
        ),

        phase=(
            swing_phase
        ),

        duration=(
            T_swing
        )
    )


    # ========================================================
    # 3. SWING APEX
    #
    # General form also supports different start/end heights.
    # ========================================================

    z_apex = (

        max(
            float(
                W_p_liftoff[
                    2
                ]
            ),
            float(
                W_p_foothold[
                    2
                ]
            )
        )

        +

        swing_height
    )


    W_p_apex = np.array(
        [
            0.5
            *
            (
                W_p_liftoff[
                    0
                ]
                +
                W_p_foothold[
                    0
                ]
            ),

            0.5
            *
            (
                W_p_liftoff[
                    1
                ]
                +
                W_p_foothold[
                    1
                ]
            ),

            z_apex
        ],
        dtype=float
    )


    # ========================================================
    # 4. VERTICAL QUINTIC MOTION
    #
    # First half:
    #
    #     z_liftoff -> z_apex
    #
    # Second half:
    #
    #     z_apex -> z_foothold
    #
    #
    # Each segment duration:
    #
    #     T_swing / 2
    # ========================================================

    T_half = (
        0.5
        *
        T_swing
    )


    if swing_phase <= 0.5:

        local_phase = (
            2.0
            *
            swing_phase
        )


        (
            z_position,
            z_velocity,
            z_acceleration

        ) = evaluate_quintic_segment(

            p_initial=np.array(
                [
                    W_p_liftoff[
                        2
                    ]
                ],
                dtype=float
            ),

            p_final=np.array(
                [
                    z_apex
                ],
                dtype=float
            ),

            phase=(
                local_phase
            ),

            duration=(
                T_half
            )
        )


    else:

        local_phase = (

            2.0
            *
            (
                swing_phase
                -
                0.5
            )
        )


        (
            z_position,
            z_velocity,
            z_acceleration

        ) = evaluate_quintic_segment(

            p_initial=np.array(
                [
                    z_apex
                ],
                dtype=float
            ),

            p_final=np.array(
                [
                    W_p_foothold[
                        2
                    ]
                ],
                dtype=float
            ),

            phase=(
                local_phase
            ),

            duration=(
                T_half
            )
        )


    # ========================================================
    # 5. ASSEMBLE 3D REFERENCE
    # ========================================================

    W_p_swing_des = np.array(
        [
            p_xy[
                0
            ],
            p_xy[
                1
            ],
            z_position[
                0
            ]
        ],
        dtype=float
    )


    W_v_swing_des = np.array(
        [
            v_xy[
                0
            ],
            v_xy[
                1
            ],
            z_velocity[
                0
            ]
        ],
        dtype=float
    )


    W_a_swing_des = np.array(
        [
            a_xy[
                0
            ],
            a_xy[
                1
            ],
            z_acceleration[
                0
            ]
        ],
        dtype=float
    )


    # ========================================================
    # 6. OUTPUT VALIDATION
    # ========================================================

    for value, name in (

        (
            W_p_swing_des,
            "W_p_swing_des"
        ),

        (
            W_v_swing_des,
            "W_v_swing_des"
        ),

        (
            W_a_swing_des,
            "W_a_swing_des"
        ),

        (
            W_p_apex,
            "W_p_apex"
        ),
    ):

        if value.shape != (
            3,
        ):

            raise RuntimeError(
                f"{name} phải có shape (3,)."
            )


        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise RuntimeError(
                f"{name} chứa NaN hoặc Inf."
            )


    # ========================================================
    # 7. RETURN
    # ========================================================

    return {

        "W_p_swing_des":
            W_p_swing_des,

        "W_v_swing_des":
            W_v_swing_des,

        "W_a_swing_des":
            W_a_swing_des,

        "W_p_apex":
            W_p_apex,

        "swing_phase":
            swing_phase,
    }