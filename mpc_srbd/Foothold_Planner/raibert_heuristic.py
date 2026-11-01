import numpy as np


# ============================================================
# RAIBERT HEURISTIC FOOTHOLD PLANNER
# ============================================================

def compute_raibert_foothold(
    W_p_CoM,
    W_R_B,
    B_p_L6_offset,
    B_p_R6_offset,
    W_p_dot_CoM,
    W_p_dot_CoM_ref,
    swing_foot,
    delta_T,
    T_stance,
    k_d=0.05,
    ground_height=0.0
):
    """
    Compute the desired foothold position of the current
    swing foot using a HECTOR-style Raibert heuristic.

    The foothold is expressed in the World frame.


    ==========================================================
    HEURISTIC
    ==========================================================

    For swing foot i:

        W_p_foothold,i
            =
        W_p_CoM
            +
        W_R_B @ B_p_offset,i
            +
        W_v_CoM * T_swing_remaining
            +
        1/2 * W_v_CoM * T_stance
            +
        k_d * (
            W_v_CoM
            -
            W_v_CoM_ref
        )


    Only x-y components are planned using the Raibert
    heuristic.

    The z component is assigned separately:

        z_foothold = ground_height


    ==========================================================
    PHYSICAL INTERPRETATION
    ==========================================================

    1. Nominal foot location:

        W_p_CoM
            +
        W_R_B @ B_p_offset,i

    gives the nominal location of the left/right foot relative
    to the current CoM and body orientation.


    2. Touchdown prediction:

        W_v_CoM * delta_T

    predicts how far the body will move during the remaining
    swing time before touchdown.


    3. Raibert stance placement:

        1/2 * W_v_CoM * T_stance

    places the foot ahead of the predicted body position so
    that the CoM can move through the stance phase.


    4. Velocity feedback:

        k_d * (
            W_v_CoM
            -
            W_v_CoM_ref
        )

    corrects the foothold according to the velocity tracking
    error.


    Parameters
    ----------
    W_p_CoM : array_like, shape (3,)

        Current whole-body CoM position expressed in the
        World frame.

        Unit:

            [m]


    W_R_B : array_like, shape (3, 3)

        Rotation matrix from Body frame to World frame:

            W_v = W_R_B @ B_v


    B_p_L6_offset : array_like, shape (3,)

        Nominal left-foot offset relative to the CoM/body
        origin, expressed in the Body frame.

        At the home configuration this should satisfy
        approximately:

            W_p_L6_home
                =
            W_p_CoM_home
                +
            W_R_B_home @ B_p_L6_offset


    B_p_R6_offset : array_like, shape (3,)

        Nominal right-foot offset relative to the CoM/body
        origin, expressed in the Body frame.


    W_p_dot_CoM : array_like, shape (3,)

        Current CoM linear velocity expressed in the
        World frame.

        Unit:

            [m/s]


    W_p_dot_CoM_ref : array_like, shape (3,)

        Desired / commanded CoM linear velocity expressed
        in the World frame.

        Unit:

            [m/s]


    swing_foot : str or None

        Current swing-foot identity:

            "L"
                -> left foot is swing

            "R"
                -> right foot is swing

            None
                -> Double Support


    delta_T : float or None

        Remaining time until the current swing phase ends.

        This corresponds to:

            T_swing_remaining

        Unit:

            [s]

        During Double Support:

            delta_T = None


    T_stance : float

        Duration of the next single-support / stance phase.

        Current project example:

            T_stance
                =
            N_SS * dt_mpc

        Unit:

            [s]


    k_d : float, optional

        Raibert velocity-feedback gain.

        Default:

            0.05


    ground_height : float, optional

        Desired touchdown height of the controlled foot site
        in the World frame.

        Important:

        This value should correspond to the desired height of
        L6/R6, not necessarily the mathematical terrain-plane
        height if the L6/R6 site has a geometric offset.

        Unit:

            [m]


    Returns
    -------
    W_p_foothold : np.ndarray, shape (3,), or None

        Desired foothold position expressed in the World frame:

            [
                x_foothold,
                y_foothold,
                z_foothold
            ]

        Returns None during Double Support.
    """


    # ========================================================
    # 1. CONVERT INPUTS
    # ========================================================

    W_p_CoM = np.asarray(
        W_p_CoM,
        dtype=float
    ).reshape(-1)


    W_R_B = np.asarray(
        W_R_B,
        dtype=float
    )


    B_p_L6_offset = np.asarray(
        B_p_L6_offset,
        dtype=float
    ).reshape(-1)


    B_p_R6_offset = np.asarray(
        B_p_R6_offset,
        dtype=float
    ).reshape(-1)


    W_p_dot_CoM = np.asarray(
        W_p_dot_CoM,
        dtype=float
    ).reshape(-1)


    W_p_dot_CoM_ref = np.asarray(
        W_p_dot_CoM_ref,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # 2. VECTOR VALIDATION
    # ========================================================

    for name, value in (

        (
            "W_p_CoM",
            W_p_CoM
        ),

        (
            "B_p_L6_offset",
            B_p_L6_offset
        ),

        (
            "B_p_R6_offset",
            B_p_R6_offset
        ),

        (
            "W_p_dot_CoM",
            W_p_dot_CoM
        ),

        (
            "W_p_dot_CoM_ref",
            W_p_dot_CoM_ref
        ),

    ):

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


    # ========================================================
    # 3. ROTATION MATRIX VALIDATION
    # ========================================================

    if W_R_B.shape != (
        3,
        3
    ):

        raise ValueError(
            "W_R_B phải có shape (3, 3), "
            f"nhưng nhận được {W_R_B.shape}."
        )


    if not np.all(
        np.isfinite(
            W_R_B
        )
    ):

        raise ValueError(
            "W_R_B chứa NaN hoặc Inf."
        )


    # --------------------------------------------------------
    # Basic rotation-matrix consistency check
    #
    #     R.T @ R = I
    # --------------------------------------------------------

    if not np.allclose(
        W_R_B.T @ W_R_B,
        np.eye(3),
        atol=1e-5
    ):

        raise ValueError(
            "W_R_B không phải rotation matrix hợp lệ: "
            "R.T @ R khác I."
        )


    # --------------------------------------------------------
    # Proper rotation:
    #
    #     det(R) = +1
    # --------------------------------------------------------

    if not np.isclose(
        np.linalg.det(
            W_R_B
        ),
        1.0,
        atol=1e-5
    ):

        raise ValueError(
            "W_R_B phải có det(R) = +1."
        )


    # ========================================================
    # 4. SCALAR PARAMETER VALIDATION
    # ========================================================

    k_d = float(
        k_d
    )


    if not np.isfinite(
        k_d
    ):

        raise ValueError(
            "k_d phải finite."
        )


    if k_d < 0.0:

        raise ValueError(
            "k_d không được nhỏ hơn 0."
        )


    # --------------------------------------------------------
    # Stance duration
    # --------------------------------------------------------

    T_stance = float(
        T_stance
    )


    if not np.isfinite(
        T_stance
    ):

        raise ValueError(
            "T_stance phải finite."
        )


    if T_stance <= 0.0:

        raise ValueError(
            "T_stance phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # Touchdown height
    # --------------------------------------------------------

    ground_height = float(
        ground_height
    )


    if not np.isfinite(
        ground_height
    ):

        raise ValueError(
            "ground_height phải finite."
        )


    # ========================================================
    # 5. DOUBLE SUPPORT
    # ========================================================

    if swing_foot is None:

        if delta_T is not None:

            raise ValueError(
                "delta_T phải là None khi "
                "swing_foot là None."
            )

        return None


    # ========================================================
    # 6. VALIDATE SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R"
    ):

        raise ValueError(
            "swing_foot chỉ được là "
            "'L', 'R' hoặc None."
        )


    # ========================================================
    # 7. VALIDATE REMAINING SWING TIME
    # ========================================================

    if delta_T is None:

        raise ValueError(
            "delta_T không được là None "
            "khi đang có swing foot."
        )


    delta_T = float(
        delta_T
    )


    if not np.isfinite(
        delta_T
    ):

        raise ValueError(
            "delta_T phải finite."
        )


    if delta_T <= 0.0:

        raise ValueError(
            "delta_T phải lớn hơn 0 "
            "khi đang ở swing phase."
        )


    # ========================================================
    # 8. SELECT NOMINAL FOOT OFFSET
    # ========================================================

    if swing_foot == "L":

        B_p_foot_offset = (
            B_p_L6_offset.copy()
        )

    else:

        B_p_foot_offset = (
            B_p_R6_offset.copy()
        )


    # ========================================================
    # 9. NOMINAL FOOT POSITION IN WORLD FRAME
    #
    #     W_p_foot_nominal
    #
    # =
    #
    #     W_p_CoM
    #
    #     +
    #
    #     W_R_B @ B_p_foot_offset
    #
    #
    # This replaces the old fixed World-frame:
    #
    #     W_p_L6_ref
    #
    # or:
    #
    #     W_p_R6_ref
    #
    # ========================================================

    W_p_foot_nominal = (

        W_p_CoM

        +

        W_R_B
        @
        B_p_foot_offset

    )


    # ========================================================
    # 10. PREDICT BODY MOTION UNTIL TOUCHDOWN
    #
    #     p_touchdown_prediction
    #
    # =
    #
    #     v_CoM * T_swing_remaining
    #
    # ========================================================

    W_p_touchdown_prediction = (

        W_p_dot_CoM

        *

        delta_T

    )


    # ========================================================
    # 11. RAIBERT STANCE PLACEMENT TERM
    #
    #     p_raibert
    #
    # =
    #
    #     1/2 * v_CoM * T_stance
    #
    # ========================================================

    W_p_raibert = (

        0.5

        *

        W_p_dot_CoM

        *

        T_stance

    )


    # ========================================================
    # 12. VELOCITY FEEDBACK CORRECTION
    #
    #     p_velocity_feedback
    #
    # =
    #
    #     k_d * (v_CoM - v_CoM_ref)
    #
    # ========================================================

    W_p_velocity_feedback = (

        k_d

        *

        (
            W_p_dot_CoM

            -

            W_p_dot_CoM_ref
        )

    )


    # ========================================================
    # 13. HE-STYLE RAIBERT FOOTHOLD
    #
    # x-y:
    #
    #     p_foothold
    #
    # =
    #
    #     p_nominal
    #
    #     +
    #
    #     v_CoM * T_swing_remaining
    #
    #     +
    #
    #     1/2 * v_CoM * T_stance
    #
    #     +
    #
    #     k_d * (v_CoM - v_CoM_ref)
    #
    #
    # z:
    #
    #     assigned separately
    # ========================================================

    W_p_foothold = (
        W_p_foot_nominal.copy()
    )


    W_p_foothold[
        0:2
    ] = (

        W_p_foot_nominal[
            0:2
        ]

        +

        W_p_touchdown_prediction[
            0:2
        ]

        +

        W_p_raibert[
            0:2
        ]

        +

        W_p_velocity_feedback[
            0:2
        ]

    )


    # ========================================================
    # 14. TOUCHDOWN HEIGHT
    # ========================================================

    W_p_foothold[
        2
    ] = ground_height


    # ========================================================
    # 15. OUTPUT VALIDATION
    # ========================================================

    if W_p_foothold.shape != (
        3,
    ):

        raise RuntimeError(
            "W_p_foothold phải có shape (3,), "
            f"nhưng nhận được {W_p_foothold.shape}."
        )


    if not np.all(
        np.isfinite(
            W_p_foothold
        )
    ):

        raise RuntimeError(
            "W_p_foothold chứa NaN hoặc Inf."
        )


    # ========================================================
    # 16. RETURN
    # ========================================================

    return W_p_foothold