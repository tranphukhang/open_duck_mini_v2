import numpy as np


# ============================================================
# NUMERICAL IK - PSEUDO-NEWTON METHOD
#
# Primary task:
#
#     swing-foot Cartesian position
#
#         [x, y, z]
#
# Swing leg:
#
#     5 actuated joints
#
# Therefore:
#
#     J in R^(3x5)
#
# IMPORTANT:
#
# This module computes ONLY the primary task correction.
#
# Null-space posture regulation belongs to:
#
#     numerical_IK_hybrid.py
#
# ============================================================

def compute_newton_step(
    J,
    error,
    alpha_newton=1.0
):
    """
    Compute one pseudo-Newton IK update for the 3D Cartesian
    swing-foot position task.

    Parameters
    ----------
    J : array_like, shape (3, 5)

        Cartesian position Jacobian of the swing foot with
        respect to the five swing-leg joints.

        Joint order:

            hip_yaw
            hip_roll
            hip_pitch
            knee
            ankle


    error : array_like, shape (3,)

        Cartesian swing-foot position error:

            error =
            [
                e_x
                e_y
                e_z
            ]


    alpha_newton : float

        Newton step scaling factor.

        Must satisfy:

            alpha_newton > 0


    Returns
    -------
    delta_q_task : ndarray, shape (5,)

        Primary-task joint correction:

            delta_q_task
                =
            alpha_newton
            *
            J^dagger
            *
            error

        Null-space posture correction is NOT included here.
    """

    # ========================================================
    # 1. INPUT CONVERSION
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
    # 4. VALIDATE STEP SCALE
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
    # 5. MOORE-PENROSE PSEUDOINVERSE
    #
    #     J in R^(3x5)
    #
    #     J^dagger in R^(5x3)
    # ========================================================

    J_pinv = np.linalg.pinv(
        J
    )


    if J_pinv.shape != (
        5,
        3
    ):

        raise RuntimeError(
            "J_pinv phải có shape (5, 3), "
            f"nhưng nhận được {J_pinv.shape}."
        )


    # ========================================================
    # 6. PRIMARY NEWTON UPDATE
    #
    #     delta_q_task
    #
    #         =
    #
    #     alpha_newton
    #
    #         *
    #
    #     J^dagger
    #
    #         *
    #
    #     error
    #
    # ========================================================

    delta_q_task = (
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

    if delta_q_task.shape != (
        5,
    ):

        raise RuntimeError(
            "delta_q_task phải có shape (5,), "
            f"nhưng nhận được {delta_q_task.shape}."
        )


    if not np.all(
        np.isfinite(
            delta_q_task
        )
    ):

        raise RuntimeError(
            "delta_q_task chứa NaN hoặc Inf."
        )


    # ========================================================
    # 8. RETURN
    # ========================================================

    return delta_q_task