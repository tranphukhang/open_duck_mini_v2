import numpy as np


# ============================================================
# CONTINUOUS SRBD MODEL
# ============================================================

from continuous_model import (
    build_continuous_model,
)


# ============================================================
# DISCRETIZATION
# ============================================================

from discretization import (
    discretize_model,
)


# ============================================================
# PREDICTION MODEL
# ============================================================

from prediction_model import (
    build_prediction_model,
)


# ============================================================
# BUILD ORIGINAL LTI SRBD MODEL
# ============================================================

def build_SRBD_model(
    Ew_inverse,
    mass_CoM,
    W_inertia_CoM,
    W_r_L,
    W_r_R,
    dt_mpc,
    horizon
):
    """
    Build the original LTI SRBD prediction model.

    This function is preserved for backward compatibility.

    Current formulation
    -------------------

    Continuous model:

        x_dot
            =
        Ac @ x
            +
        Bc @ u


    Discrete model:

        x[k+1]
            =
        Ad @ x[k]
            +
        Bd @ u[k]


    Prediction model:

        X
            =
        Phi @ x[k]
            +
        Gamma @ U


    In this original LTI formulation, the moment arms:

        W_r_L

        W_r_R

    are evaluated at the current MPC update and then frozen
    over the entire prediction horizon.


    Parameters
    ----------
    Ew_inverse : np.ndarray, shape (3, 3)

        Euler-rate transformation matrix:

            euler_dot
                =
            Ew_inverse @ W_omega_base


    mass_CoM : float

        Total robot mass [kg].


    W_inertia_CoM : np.ndarray, shape (3, 3)

        Whole-body inertia tensor about CoM,
        expressed in World frame [kg.m^2].


    W_r_L : np.ndarray, shape (3,)

        Current left moment arm:

            W_r_L
                =
            W_p_L6 - W_p_CoM


    W_r_R : np.ndarray, shape (3,)

        Current right moment arm:

            W_r_R
                =
            W_p_R6 - W_p_CoM


    dt_mpc : float

        Prediction-model discretization step [s].


    horizon : int

        MPC prediction horizon N.


    Returns
    -------
    Phi : np.ndarray, shape (13*N, 13)

        State prediction matrix.


    Gamma : np.ndarray, shape (13*N, 12*N)

        Input prediction matrix.
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    Ew_inverse = np.asarray(
        Ew_inverse,
        dtype=float
    )


    W_inertia_CoM = np.asarray(
        W_inertia_CoM,
        dtype=float
    )


    W_r_L = np.asarray(
        W_r_L,
        dtype=float
    ).reshape(-1)


    W_r_R = np.asarray(
        W_r_R,
        dtype=float
    ).reshape(-1)


    mass_CoM = float(
        mass_CoM
    )


    dt_mpc = float(
        dt_mpc
    )


    horizon = int(
        horizon
    )


    # ========================================================
    # 2. BASIC VALIDATION
    # ========================================================

    if Ew_inverse.shape != (
        3,
        3
    ):

        raise ValueError(
            "Ew_inverse phải có shape (3, 3)."
        )


    if W_inertia_CoM.shape != (
        3,
        3
    ):

        raise ValueError(
            "W_inertia_CoM phải có shape (3, 3)."
        )


    if W_r_L.shape != (
        3,
    ):

        raise ValueError(
            "W_r_L phải có shape (3,)."
        )


    if W_r_R.shape != (
        3,
    ):

        raise ValueError(
            "W_r_R phải có shape (3,)."
        )


    if mass_CoM <= 0.0:

        raise ValueError(
            "mass_CoM phải lớn hơn 0."
        )


    if dt_mpc <= 0.0:

        raise ValueError(
            "dt_mpc phải lớn hơn 0."
        )


    if horizon <= 0:

        raise ValueError(
            "horizon phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # Finite validation
    # --------------------------------------------------------

    for name, value in (
        (
            "Ew_inverse",
            Ew_inverse
        ),
        (
            "W_inertia_CoM",
            W_inertia_CoM
        ),
        (
            "W_r_L",
            W_r_L
        ),
        (
            "W_r_R",
            W_r_R
        )
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


    if not np.isfinite(
        mass_CoM
    ):

        raise ValueError(
            "mass_CoM không finite."
        )


    if not np.isfinite(
        dt_mpc
    ):

        raise ValueError(
            "dt_mpc không finite."
        )


    # ========================================================
    # 3. BUILD CONTINUOUS SRBD MODEL
    #
    #     x_dot
    #
    #       =
    #
    #     Ac x
    #
    #       +
    #
    #     Bc u
    # ========================================================

    (
        Ac,
        Bc

    ) = build_continuous_model(

        Ew_inverse=Ew_inverse,

        mass_CoM=mass_CoM,

        W_inertia_CoM=W_inertia_CoM,

        W_r_L=W_r_L,

        W_r_R=W_r_R
    )


    # ========================================================
    # 4. DISCRETIZE MODEL
    #
    # Forward Euler:
    #
    #     Ad
    #
    #       =
    #
    #     I + Ac * dt_mpc
    #
    #
    #     Bd
    #
    #       =
    #
    #     Bc * dt_mpc
    # ========================================================

    (
        Ad,
        Bd

    ) = discretize_model(

        Ac=Ac,

        Bc=Bc,

        Ts=dt_mpc
    )


    # ========================================================
    # 5. BUILD LTI PREDICTION MODEL
    #
    #     X
    #
    #       =
    #
    #     Phi x_k
    #
    #       +
    #
    #     Gamma U
    # ========================================================

    (
        Phi,
        Gamma

    ) = build_prediction_model(

        Ad=Ad,

        Bd=Bd,

        N=horizon
    )


    # ========================================================
    # 6. FINAL VALIDATION
    # ========================================================

    nx = 13

    nu = 12


    expected_Phi_shape = (
        horizon * nx,
        nx
    )


    expected_Gamma_shape = (
        horizon * nx,
        horizon * nu
    )


    if Phi.shape != expected_Phi_shape:

        raise RuntimeError(
            "Phi sai dimension trong build_SRBD_model(). "
            f"Expected {expected_Phi_shape}, "
            f"received {Phi.shape}."
        )


    if Gamma.shape != expected_Gamma_shape:

        raise RuntimeError(
            "Gamma sai dimension trong build_SRBD_model(). "
            f"Expected {expected_Gamma_shape}, "
            f"received {Gamma.shape}."
        )


    if not np.all(
        np.isfinite(
            Phi
        )
    ):

        raise RuntimeError(
            "Phi chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Gamma
        )
    ):

        raise RuntimeError(
            "Gamma chứa NaN hoặc Inf."
        )


    # ========================================================
    # 7. RETURN
    # ========================================================

    return (
        Phi,
        Gamma
    )

