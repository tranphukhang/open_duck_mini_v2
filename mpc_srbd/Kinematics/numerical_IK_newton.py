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

    The method is intended for the 4D swing-foot task:

        error
            =
        [
            position error
            pitch error
        ]

    More explicitly:

        error
            =
        [
            e_x
            e_y
            e_z
            e_pitch
        ]

    where:

        e_x, e_y, e_z

    are the swing-foot Cartesian position errors,

    and:

        e_pitch

    is the swing-foot pitch-orientation error.


    The swing leg has 5 actuated joints:

        q
            =
        [
            q_hip_yaw
            q_hip_roll
            q_hip_pitch
            q_knee
            q_ankle
        ]

    Therefore the 4D task Jacobian has:

        J in R^(4x5)


    The Moore-Penrose pseudoinverse is used:

        delta_q
            =
        alpha_newton
        *
        pinv(J)
        *
        error


    Since:

        J in R^(4x5)

    its pseudoinverse has:

        pinv(J) in R^(5x4)

    and therefore:

        delta_q in R^5


    Parameters
    ----------
    J : array_like, shape (4, 5)

        Swing-leg task Jacobian.

        The first three rows correspond to translational
        motion of the physical swing foot:

            x
            y
            z

        The final row corresponds to the selected
        orientation task:

            pitch


    error : array_like, shape (4,)

        Swing-foot task-space error:

            [
                e_x,
                e_y,
                e_z,
                e_pitch
            ]


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
        4,
        5
    ):

        raise ValueError(
            "J phải có shape (4, 5), "
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
    # 3. VALIDATE TASK ERROR
    # ========================================================

    if error.shape != (
        4,
    ):

        raise ValueError(
            "error phải có shape (4,), "
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
    #     J in R^(4x5)
    #
    # the pseudoinverse has:
    #
    #     J_pinv in R^(5x4)
    #
    # The pseudoinverse is retained exactly as in the original
    # IK structure because it remains numerically robust near
    # singular configurations.
    # ========================================================

    J_pinv = np.linalg.pinv(
        J
    )


    # ========================================================
    # 6. PSEUDO-NEWTON UPDATE
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
