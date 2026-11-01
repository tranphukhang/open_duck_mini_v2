import numpy as np


# ============================================================
# NUMERICAL IK - GRADIENT DESCENT METHOD
# ============================================================

def compute_gradient_descent_step(
    J,
    error,
    eta_gradient=1.0
):
    """
    Compute one numerical inverse-kinematics update step
    using the Jacobian-transpose gradient descent method.

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


    For the Cartesian position error cost:

        V
            =
        1/2 * error^T * error

    the Jacobian-transpose direction is used:

        delta_q
            =
        eta_gradient
        *
        J^T
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


    eta_gradient : float, optional

        Gradient-descent step size.

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


    eta_gradient = float(
        eta_gradient
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
    # 4. VALIDATE GRADIENT STEP SIZE
    # ========================================================

    if not np.isfinite(
        eta_gradient
    ):

        raise ValueError(
            "eta_gradient phải là số hữu hạn."
        )


    if eta_gradient <= 0.0:

        raise ValueError(
            "eta_gradient phải > 0."
        )


    # ========================================================
    # 5. GRADIENT DESCENT UPDATE
    #
    #     delta_q
    #
    # =
    #
    #     eta_gradient
    #
    #     *
    #
    #     J^T
    #
    #     *
    #
    #     error
    # ========================================================

    delta_q = (
        eta_gradient
        *
        (
            J.T
            @
            error
        )
    )


    # ========================================================
    # 6. OUTPUT VALIDATION
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
    # 7. RETURN
    # ========================================================

    return delta_q