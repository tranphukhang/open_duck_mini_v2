import numpy as np


# ============================================================
# NUMERICAL IK - JACOBIAN-TRANSPOSE GRADIENT DESCENT
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

def compute_gradient_descent_step(
    J,
    error,
    eta_gradient=1.0
):
    """
    Compute one Jacobian-transpose gradient-descent IK update
    for the 3D Cartesian swing-foot position task.

    Parameters
    ----------
    J : array_like, shape (3, 5)

        Cartesian position Jacobian of the swing foot with
        respect to the five swing-leg joints.


    error : array_like, shape (3,)

        Cartesian swing-foot position error:

            error =
            [
                e_x
                e_y
                e_z
            ]


    eta_gradient : float

        Gradient-descent step size.

        Must satisfy:

            eta_gradient > 0


    Returns
    -------
    delta_q_task : ndarray, shape (5,)

        Primary-task joint correction:

            delta_q_task
                =
            eta_gradient
            *
            J.T
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
    # 4. VALIDATE STEP SIZE
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
    # 5. PRIMARY GRADIENT-DESCENT UPDATE
    #
    # Task-space cost:
    #
    #     V = 1/2 * error.T @ error
    #
    # Jacobian-transpose direction:
    #
    #     delta_q_task
    #
    #         =
    #
    #     eta_gradient
    #
    #         *
    #
    #     J.T
    #
    #         *
    #
    #     error
    #
    #
    #     J.T in R^(5x3)
    #
    # therefore:
    #
    #     delta_q_task in R^5
    #
    # ========================================================

    delta_q_task = (
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
    # 7. RETURN
    # ========================================================

    return delta_q_task