import numpy as np


# ============================================================
# NUMERICAL IK - PSEUDO-NEWTON METHOD
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

def compute_newton_step(
    J,
    error,
    alpha_newton=1.0,
):
    """
    Compute one pseudo-Newton update for the 3-DOF ankle-center
    position IK.
    """

    J = np.asarray(J, dtype=float)
    error = np.asarray(error, dtype=float).reshape(-1)
    alpha_newton = float(alpha_newton)

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

    if not np.isfinite(alpha_newton) or alpha_newton <= 0.0:
        raise ValueError("alpha_newton phải hữu hạn và > 0.")

    J_pinv = np.linalg.pinv(J)

    if J_pinv.shape != (3, 3):
        raise RuntimeError(
            "J_pinv phải có shape (3, 3), "
            f"nhưng nhận được {J_pinv.shape}."
        )

    delta_q = alpha_newton * (J_pinv @ error)

    if delta_q.shape != (3,):
        raise RuntimeError(
            "delta_q phải có shape (3,), "
            f"nhưng nhận được {delta_q.shape}."
        )

    if not np.all(np.isfinite(delta_q)):
        raise RuntimeError("delta_q chứa NaN hoặc Inf.")

    return delta_q
