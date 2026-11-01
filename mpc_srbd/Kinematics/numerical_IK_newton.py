import numpy as np


# ============================================================
# NUMERICAL IK - PSEUDO-NEWTON METHOD
# ============================================================

def compute_newton_step(
    J,
    error,
    alpha_newton=1.0
):
    """
    Compute one numerical inverse-kinematics update step
    using the pseudo-Newton method.

    The method is intended for the swing-leg position IK:

        error
            =
        W_p_i6_des
            -
        W_p_i6

    where:

        W_p_i6_des

    is the desired swing-foot position in the World frame,

    and:

        W_p_i6

    is the current physical foot position in the World frame.


    Since the swing leg has 5 joints while the Cartesian
    position task has 3 dimensions:

        J in R^(3x5)

    the Jacobian is not square.

    Therefore, the Moore-Penrose pseudoinverse is used:

        delta_q
            =
        alpha_newton
        *
        pinv(J)
        *
        error


    Parameters
    ----------
    J : array_like, shape (3, 5)

        Translational Jacobian of the current swing foot
        with respect to the 5 joints of that leg.

        The Jacobian maps joint velocity to Cartesian
        foot velocity:

            W_p_dot_i6 = J q_dot


    error : array_like, shape (3,)

        Cartesian swing-foot position error:

            error
                =
            W_p_i6_des
                -
            W_p_i6

        Unit:

            [m]


    alpha_newton : float, optional

        Newton step scaling factor.

        Default:

            1.0


    Returns
    -------
    delta_q : np.ndarray, shape (5,)

        Joint-position correction:

            [
                delta_q_hip_yaw,
                delta_q_hip_roll,
                delta_q_hip_pitch,
                delta_q_knee,
                delta_q_ankle
            ]

        Unit:

            [rad]
    """


    # ========================================================
    # 1. CONVERT INPUTS
    # ========================================================

    J = np.asarray(
        J,
        dtype=float
    )


    error = np.asarray(
        error,
        dtype=float
    ).reshape(-1)


    alpha_newton = float(
        alpha_newton
    )


    # ========================================================
    # 2. VALIDATE JACOBIAN
    # ========================================================

    if J.shape != (
        3,
        5
    ):

        raise ValueError(
            "J phải có shape (3, 5), "
            f"nhưng nhận được {J.shape}."
        )


    if not np.all(
        np.isfinite(
            J
        )
    ):

        raise ValueError(
            "J chứa NaN hoặc Inf."
        )


    # ========================================================
    # 3. VALIDATE CARTESIAN ERROR
    # ========================================================

    if error.shape != (
        3,
    ):

        raise ValueError(
            "error phải có shape (3,), "
            f"nhưng nhận được {error.shape}."
        )


    if not np.all(
        np.isfinite(
            error
        )
    ):

        raise ValueError(
            "error chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. VALIDATE NEWTON STEP SCALE
    # ========================================================

    if not np.isfinite(
        alpha_newton
    ):

        raise ValueError(
            "alpha_newton phải là số hữu hạn."
        )


    if alpha_newton <= 0.0:

        raise ValueError(
            "alpha_newton phải > 0."
        )


    # ========================================================
    # 5. COMPUTE MOORE-PENROSE PSEUDOINVERSE
    #
    # For:
    #
    #     J in R^(3x5)
    #
    # the pseudoinverse has:
    #
    #     J_pinv in R^(5x3)
    # ========================================================

    J_pinv = np.linalg.pinv(
        J
    )


    # ========================================================
    # 6. NEWTON UPDATE
    #
    #     delta_q
    #
    # =
    #
    #     alpha_newton
    #
    #     *
    #
    #     J^dagger
    #
    #     *
    #
    #     error
    # ========================================================

    delta_q = (
        alpha_newton
        *
        (
            J_pinv
            @
            error
        )
    )


    # ========================================================
    # 7. OUTPUT VALIDATION
    # ========================================================

    if delta_q.shape != (
        5,
    ):

        raise RuntimeError(
            "delta_q phải có shape (5,), "
            f"nhưng nhận được {delta_q.shape}."
        )


    if not np.all(
        np.isfinite(
            delta_q
        )
    ):

        raise RuntimeError(
            "delta_q chứa NaN hoặc Inf."
        )


    # ========================================================
    # 8. RETURN
    # ========================================================

    return delta_q