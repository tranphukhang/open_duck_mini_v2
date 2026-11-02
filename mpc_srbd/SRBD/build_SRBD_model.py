import numpy as np

from continuous_model import build_continuous_model
from discretization import discretize_model
from prediction_model import build_prediction_model


def build_SRBD_model(
    Ew_inverse,
    mass_CoM,
    W_inertia_CoM,
    W_r_L,
    W_r_R,
    dt_mpc,
    horizon,
):
    """
    Build one frozen-linearization SRBD prediction model.

    At each MPC update k, the caller supplies the CURRENT feedback-based:

        Ew_inverse(k)
        W_inertia_CoM(k)
        W_r_L(k)
        W_r_R(k)

    These quantities are frozen over the complete prediction horizon:

        j = 0, ..., N-1

    Therefore:

        x[j+1|k] = Ad(k) x[j|k] + Bd(k) u[j|k]

    and the condensed prediction is:

        X = Phi x[k] + Gamma U

    Notes
    -----
    - No prediction-step-dependent moment-arm sequence is used.
    - No build_SRBD_model_sequence() exists in this baseline.
    - The body-frame inertia is constant outside this function.
      W_inertia_CoM is the CURRENT world-frame inertia obtained from:

          W_I_CoM(k) = W_R_B(k) @ B_I_CoM @ W_R_B(k).T
    """

    Ew_inverse = np.asarray(Ew_inverse, dtype=float)
    W_inertia_CoM = np.asarray(W_inertia_CoM, dtype=float)
    W_r_L = np.asarray(W_r_L, dtype=float).reshape(-1)
    W_r_R = np.asarray(W_r_R, dtype=float).reshape(-1)

    mass_CoM = float(mass_CoM)
    dt_mpc = float(dt_mpc)
    horizon = int(horizon)

    if Ew_inverse.shape != (3, 3):
        raise ValueError("Ew_inverse phải có shape (3, 3).")

    if W_inertia_CoM.shape != (3, 3):
        raise ValueError("W_inertia_CoM phải có shape (3, 3).")

    if W_r_L.shape != (3,):
        raise ValueError("W_r_L phải có shape (3,).")

    if W_r_R.shape != (3,):
        raise ValueError("W_r_R phải có shape (3,).")

    if mass_CoM <= 0.0 or not np.isfinite(mass_CoM):
        raise ValueError("mass_CoM phải hữu hạn và > 0.")

    if dt_mpc <= 0.0 or not np.isfinite(dt_mpc):
        raise ValueError("dt_mpc phải hữu hạn và > 0.")

    if horizon <= 0:
        raise ValueError("horizon phải > 0.")

    for name, value in (
        ("Ew_inverse", Ew_inverse),
        ("W_inertia_CoM", W_inertia_CoM),
        ("W_r_L", W_r_L),
        ("W_r_R", W_r_R),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} chứa NaN hoặc Inf.")

    Ac, Bc = build_continuous_model(
        Ew_inverse=Ew_inverse,
        mass_CoM=mass_CoM,
        W_inertia_CoM=W_inertia_CoM,
        W_r_L=W_r_L,
        W_r_R=W_r_R,
    )

    Ad, Bd = discretize_model(
        Ac=Ac,
        Bc=Bc,
        Ts=dt_mpc,
    )

    Phi, Gamma = build_prediction_model(
        Ad=Ad,
        Bd=Bd,
        N=horizon,
    )

    nx = 13
    nu = 12

    expected_Phi_shape = (horizon * nx, nx)
    expected_Gamma_shape = (horizon * nx, horizon * nu)

    if Phi.shape != expected_Phi_shape:
        raise RuntimeError(
            "Phi sai dimension. "
            f"Expected {expected_Phi_shape}, received {Phi.shape}."
        )

    if Gamma.shape != expected_Gamma_shape:
        raise RuntimeError(
            "Gamma sai dimension. "
            f"Expected {expected_Gamma_shape}, received {Gamma.shape}."
        )

    if not np.all(np.isfinite(Phi)):
        raise RuntimeError("Phi chứa NaN hoặc Inf.")

    if not np.all(np.isfinite(Gamma)):
        raise RuntimeError("Gamma chứa NaN hoặc Inf.")

    return Phi, Gamma
