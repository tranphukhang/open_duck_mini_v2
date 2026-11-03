import numpy as np


# ============================================================
# NUMERICAL IK - JACOBIAN-TRANSPOSE GRADIENT DESCENT
#
# Active IK variables:
#     q_ik = [hip_roll, hip_pitch, knee]
#
# Primary task:
#     ankle-center Cartesian position [x, y, z]
#
# Therefore:
#     J in R^(3x3)
# ============================================================

def compute_gradient_descent_step(
    J,
    error,
    eta_gradient=1.0,
):
    """
    Compute one Jacobian-transpose gradient-descent update for
    the 3-DOF ankle-center position IK.
    """

    J = np.asarray(J, dtype=float)
    error = np.asarray(error, dtype=float).reshape(-1)
    eta_gradient = float(eta_gradient)

    if J.shape != (3, 3):
        raise ValueError(
            "J phải có shape (3, 3), "
            f"nhưng nhận được {J.shape}."
        )

    if not np.all(np.isfinite(J)):
        raise ValueError("J chứa NaN hoặc Inf.")

    if error.shape != (3,):
        raise ValueError(
            "error phải có shape (3,), "
            f"nhưng nhận được {error.shape}."
        )

    if not np.all(np.isfinite(error)):
        raise ValueError("error chứa NaN hoặc Inf.")

    if not np.isfinite(eta_gradient) or eta_gradient <= 0.0:
        raise ValueError("eta_gradient phải hữu hạn và > 0.")

    delta_q = eta_gradient * (J.T @ error)

    if delta_q.shape != (3,):
        raise RuntimeError(
            "delta_q phải có shape (3,), "
            f"nhưng nhận được {delta_q.shape}."
        )

    if not np.all(np.isfinite(delta_q)):
        raise RuntimeError("delta_q chứa NaN hoặc Inf.")

    return delta_q
