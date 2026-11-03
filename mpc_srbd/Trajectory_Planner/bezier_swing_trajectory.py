import numpy as np


# ============================================================
# DEFAULT PARAMETERS
#
# Same basic control-point coefficients used by the paper:
#
#     cp1 ~= 1/3
#     cp2 ~= 2/3
#
# Paper code:
#
#     cp1_coef = 0.333
#     cp2_coef = 0.666
#
# ============================================================

DEFAULT_CP1_COEF = 0.333
DEFAULT_CP2_COEF = 0.666


# ============================================================
# VECTOR VALIDATION
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


# ============================================================
# SCALAR VALIDATION
# ============================================================

def _validate_phase(
    phase,
):
    phase = float(
        phase
    )

    if not np.isfinite(
        phase
    ):
        raise ValueError(
            "swing_phase phải hữu hạn."
        )

    if (
        phase < 0.0
        or
        phase > 1.0
    ):
        raise ValueError(
            "swing_phase phải nằm trong [0, 1]."
        )

    return phase


def _validate_positive_scalar(
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
        value <= 0.0
    ):
        raise ValueError(
            f"{name} phải hữu hạn và > 0."
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


# ============================================================
# BEZIER CONTROL POINTS
#
# Flat-ground implementation corresponding to the branch of
# the paper used when:
#
#     elevation = pf_z - p0_z <= 0.05
#
# Current project assumption:
#
#     flat ground
#
# Therefore we intentionally do NOT implement the inclined
# terrain local-frame branch here.
# ============================================================

def compute_bezier_control_points(
    W_p_liftoff,
    W_p_foothold,
    swing_height,
    cp1_coef=DEFAULT_CP1_COEF,
    cp2_coef=DEFAULT_CP2_COEF,
):
    """
    Compute cubic Bezier control points for the swing foot.

    Parameters
    ----------
    W_p_liftoff : array_like, shape (3,)
        Swing-foot position at liftoff in World frame.

    W_p_foothold : array_like, shape (3,)
        Desired touchdown position in World frame.

    swing_height : float
        Desired additional apex height [m].

        Following the paper's flat-ground implementation:

            z_apex = z_liftoff + swing_height

    cp1_coef : float
        First control-point interpolation coefficient.

    cp2_coef : float
        Second control-point interpolation coefficient.

    Returns
    -------
    result : dict
        {
            "P0": ...,
            "P1": ...,
            "P2": ...,
            "Pf": ...,
            "z_apex": ...,
        }
    """

    # ========================================================
    # 1. INPUTS
    # ========================================================

    P0 = _validate_vector3(
        W_p_liftoff,
        "W_p_liftoff",
    )

    Pf = _validate_vector3(
        W_p_foothold,
        "W_p_foothold",
    )

    swing_height = (
        _validate_nonnegative_scalar(
            swing_height,
            "swing_height",
        )
    )

    cp1_coef = float(
        cp1_coef
    )

    cp2_coef = float(
        cp2_coef
    )

    if not np.isfinite(
        cp1_coef
    ):
        raise ValueError(
            "cp1_coef phải hữu hạn."
        )

    if not np.isfinite(
        cp2_coef
    ):
        raise ValueError(
            "cp2_coef phải hữu hạn."
        )

    if (
        cp1_coef < 0.0
        or
        cp1_coef > 1.0
    ):
        raise ValueError(
            "cp1_coef phải nằm trong [0, 1]."
        )

    if (
        cp2_coef < 0.0
        or
        cp2_coef > 1.0
    ):
        raise ValueError(
            "cp2_coef phải nằm trong [0, 1]."
        )

    if cp1_coef > cp2_coef:
        raise ValueError(
            "Yêu cầu cp1_coef <= cp2_coef."
        )

    # ========================================================
    # 2. INITIAL CONTROL POINTS
    #
    # Paper:
    #
    #     P1 = P0 + (Pf - P0) * cp1_coef
    #
    #     P2 = P0 + (Pf - P0) * cp2_coef
    #
    # ========================================================

    delta = (
        Pf
        -
        P0
    )

    P1 = (
        P0
        +
        cp1_coef
        *
        delta
    )

    P2 = (
        P0
        +
        cp2_coef
        *
        delta
    )

    # ========================================================
    # 3. SWING APEX
    #
    # Paper flat-ground branch:
    #
    #     z_apex = P0_z + height
    #
    # The cubic Bezier is constrained so that:
    #
    #     z(phase=0.5) = z_apex
    #
    # With:
    #
    #     P1_z = P2_z
    #
    # resulting in:
    #
    #     P1_z = P2_z
    #
    #          =
    #
    #     (8*z_apex - P0_z - Pf_z) / 6
    #
    # ========================================================

    z_apex = (
        float(
            P0[2]
        )
        +
        swing_height
    )

    z_control = (
        8.0
        *
        z_apex
        -
        P0[2]
        -
        Pf[2]
    ) / 6.0

    P1 = P1.copy()
    P2 = P2.copy()

    P1[2] = (
        z_control
    )

    P2[2] = (
        z_control
    )

    # ========================================================
    # 4. OUTPUT
    # ========================================================

    return {
        "P0": P0.copy(),
        "P1": P1.copy(),
        "P2": P2.copy(),
        "Pf": Pf.copy(),
        "z_apex": float(
            z_apex
        ),
        "cp1_coef": float(
            cp1_coef
        ),
        "cp2_coef": float(
            cp2_coef
        ),
    }


# ============================================================
# CUBIC BEZIER EVALUATION
# ============================================================

def evaluate_cubic_bezier(
    P0,
    P1,
    P2,
    Pf,
    phase,
    duration,
):
    """
    Evaluate a 3D cubic Bezier trajectory.

    Cubic Bezier:

        P(s)
          =
        (1-s)^3 P0
          +
        3(1-s)^2 s P1
          +
        3(1-s) s^2 P2
          +
        s^3 Pf

    with:

        s = t / T

    Velocity:

        dP/dt
          =
        (1/T) dP/ds

    Acceleration:

        d2P/dt2
          =
        (1/T^2) d2P/ds2
    """

    P0 = _validate_vector3(
        P0,
        "P0",
    )

    P1 = _validate_vector3(
        P1,
        "P1",
    )

    P2 = _validate_vector3(
        P2,
        "P2",
    )

    Pf = _validate_vector3(
        Pf,
        "Pf",
    )

    phase = _validate_phase(
        phase
    )

    duration = (
        _validate_positive_scalar(
            duration,
            "duration",
        )
    )

    s = phase

    one_minus_s = (
        1.0
        -
        s
    )

    # ========================================================
    # POSITION
    # ========================================================

    position = (
        one_minus_s**3
        *
        P0

        +

        3.0
        *
        one_minus_s**2
        *
        s
        *
        P1

        +

        3.0
        *
        one_minus_s
        *
        s**2
        *
        P2

        +

        s**3
        *
        Pf
    )

    # ========================================================
    # FIRST DERIVATIVE WITH RESPECT TO PHASE
    # ========================================================

    dP_ds = (
        3.0
        *
        one_minus_s**2
        *
        (
            P1
            -
            P0
        )

        +

        6.0
        *
        one_minus_s
        *
        s
        *
        (
            P2
            -
            P1
        )

        +

        3.0
        *
        s**2
        *
        (
            Pf
            -
            P2
        )
    )

    velocity = (
        dP_ds
        /
        duration
    )

    # ========================================================
    # SECOND DERIVATIVE WITH RESPECT TO PHASE
    #
    # d2P/ds2 =
    #
    #     6(1-s)(P2 - 2P1 + P0)
    #
    #     +
    #
    #     6s(Pf - 2P2 + P1)
    #
    # Paper does not explicitly need acceleration in the
    # FootSwingTrajectory class, but we return it because it
    # is analytically available and useful for debugging.
    # ========================================================

    d2P_ds2 = (
        6.0
        *
        one_minus_s
        *
        (
            P2
            -
            2.0
            *
            P1
            +
            P0
        )

        +

        6.0
        *
        s
        *
        (
            Pf
            -
            2.0
            *
            P2
            +
            P1
        )
    )

    acceleration = (
        d2P_ds2
        /
        duration**2
    )

    return (
        np.asarray(
            position,
            dtype=float,
        ),
        np.asarray(
            velocity,
            dtype=float,
        ),
        np.asarray(
            acceleration,
            dtype=float,
        ),
    )


# ============================================================
# PAPER-STYLE SWING-FOOT BEZIER TRAJECTORY
# ============================================================

def evaluate_bezier_swing_trajectory(
    W_p_liftoff,
    W_p_foothold,
    swing_phase,
    T_swing,
    swing_height,
    cp1_coef=DEFAULT_CP1_COEF,
    cp2_coef=DEFAULT_CP2_COEF,
):
    """
    Generate the swing-foot Cartesian reference using the same
    cubic-Bezier structure as the paper implementation.

    Current implementation assumes flat ground.

    Architecture
    ------------

        W_p_liftoff
             |
             |
             v
        cubic Bezier
             |
             |
             v
        W_p_swing_des
        W_v_swing_des
             |
             |
             v
        ankle-center conversion
             |
             |
             v
        numerical IK

    Parameters
    ----------
    W_p_liftoff : array_like, shape (3,)
        Actual foot position at liftoff [m], World frame.

    W_p_foothold : array_like, shape (3,)
        Desired touchdown/foothold position [m], World frame.

    swing_phase : float
        Normalized swing phase:

            0 <= swing_phase <= 1

    T_swing : float
        Total swing duration [s].

    swing_height : float
        Desired apex height above liftoff height [m].

    cp1_coef : float
        First control-point coefficient.

    cp2_coef : float
        Second control-point coefficient.

    Returns
    -------
    result : dict
        {
            "W_p_swing_des": ...,
            "W_v_swing_des": ...,
            "W_a_swing_des": ...,

            "P0": ...,
            "P1": ...,
            "P2": ...,
            "Pf": ...,

            "z_apex": ...,
            "swing_phase": ...,
            "T_swing": ...,
        }
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    W_p_liftoff = _validate_vector3(
        W_p_liftoff,
        "W_p_liftoff",
    )

    W_p_foothold = _validate_vector3(
        W_p_foothold,
        "W_p_foothold",
    )

    swing_phase = (
        _validate_phase(
            swing_phase
        )
    )

    T_swing = (
        _validate_positive_scalar(
            T_swing,
            "T_swing",
        )
    )

    swing_height = (
        _validate_nonnegative_scalar(
            swing_height,
            "swing_height",
        )
    )

    # ========================================================
    # 2. CONTROL POINTS
    # ========================================================

    control_points = (
        compute_bezier_control_points(
            W_p_liftoff=
                W_p_liftoff,

            W_p_foothold=
                W_p_foothold,

            swing_height=
                swing_height,

            cp1_coef=
                cp1_coef,

            cp2_coef=
                cp2_coef,
        )
    )

    P0 = (
        control_points[
            "P0"
        ]
    )

    P1 = (
        control_points[
            "P1"
        ]
    )

    P2 = (
        control_points[
            "P2"
        ]
    )

    Pf = (
        control_points[
            "Pf"
        ]
    )

    # ========================================================
    # 3. EVALUATE CUBIC BEZIER
    # ========================================================

    (
        W_p_swing_des,
        W_v_swing_des,
        W_a_swing_des,

    ) = evaluate_cubic_bezier(

        P0=P0,

        P1=P1,

        P2=P2,

        Pf=Pf,

        phase=
            swing_phase,

        duration=
            T_swing,
    )

    # ========================================================
    # 4. OUTPUT VALIDATION
    # ========================================================

    for (
        value,
        name,
    ) in (
        (
            W_p_swing_des,
            "W_p_swing_des",
        ),
        (
            W_v_swing_des,
            "W_v_swing_des",
        ),
        (
            W_a_swing_des,
            "W_a_swing_des",
        ),
    ):

        if value.shape != (3,):
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
    # 5. RETURN
    # ========================================================

    return {
        "W_p_swing_des":
            W_p_swing_des,

        "W_v_swing_des":
            W_v_swing_des,

        "W_a_swing_des":
            W_a_swing_des,

        "P0":
            P0.copy(),

        "P1":
            P1.copy(),

        "P2":
            P2.copy(),

        "Pf":
            Pf.copy(),

        "z_apex":
            control_points[
                "z_apex"
            ],

        "cp1_coef":
            control_points[
                "cp1_coef"
            ],

        "cp2_coef":
            control_points[
                "cp2_coef"
            ],

        "swing_phase":
            float(
                swing_phase
            ),

        "T_swing":
            float(
                T_swing
            ),
    }