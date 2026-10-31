# ============================================================
# SRBD MODEL BUILDER
# ============================================================

from continuous_model import (
    build_continuous_model
)

from discretization import (
    discretize_model
)

from prediction_model import (
    build_prediction_model
)


# ============================================================
# BUILD SRBD MODEL
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
    Build the complete SRBD prediction model used by MPC.

    Processing sequence:

        Continuous SRBD model
                ↓
            Ac, Bc

                ↓

        Discrete SRBD model
                ↓
            Ad, Bd

                ↓

        MPC prediction model
                ↓
            Phi, Gamma


    Continuous model
    ----------------

        x_dot = Ac x + Bc u


    Discrete model
    --------------

        x_{k+1}
            =
        Ad x_k
            +
        Bd u_k


    Prediction model
    ----------------

        X
            =
        Phi x_k
            +
        Gamma U


    Parameters
    ----------
    Ew_inverse : ndarray, shape (3, 3)

        Inverse Euler-rate transformation matrix:

            euler_dot
                =
            Ew_inverse @ omega


    mass_CoM : float

        Total robot mass.


    W_inertia_CoM : ndarray, shape (3, 3)

        Robot inertia tensor about CoM,
        expressed in World frame.


    W_r_L : ndarray, shape (3,)

        Moment arm from CoM to left contact point:

            W_r_L
                =
            W_p_L6 - W_p_CoM


    W_r_R : ndarray, shape (3,)

        Moment arm from CoM to right contact point:

            W_r_R
                =
            W_p_R6 - W_p_CoM


    dt_mpc : float

        MPC prediction discretization step.


    horizon : int

        MPC prediction horizon N.


    Returns
    -------
    Phi : ndarray

        State prediction matrix.


    Gamma : ndarray

        Input prediction matrix.


    Prediction equation
    -------------------

        X
            =
        Phi x_k
            +
        Gamma U
    """


    # ========================================================
    # 1. BUILD CONTINUOUS SRBD MODEL
    #
    #     x_dot = Ac x + Bc u
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
    # 2. DISCRETIZE SRBD MODEL
    #
    #     x_{k+1}
    #
    #         =
    #
    #     Ad x_k
    #
    #         +
    #
    #     Bd u_k
    #
    #
    # Current discretization method:
    #
    #     Forward Euler
    #
    # therefore:
    #
    #     Ad = I + Ac * dt_mpc
    #
    #     Bd = Bc * dt_mpc
    # ========================================================

    (
        Ad,
        Bd
    ) = discretize_model(
        Ac,
        Bc,
        dt_mpc
    )


    # ========================================================
    # 3. BUILD MPC PREDICTION MODEL
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
    #
    #
    # where:
    #
    #     X =
    #
    #     [
    #         x_{k+1}
    #         x_{k+2}
    #         ...
    #         x_{k+N}
    #     ]
    #
    #
    # and:
    #
    #     U =
    #
    #     [
    #         u_k
    #         u_{k+1}
    #         ...
    #         u_{k+N-1}
    #     ]
    # ========================================================

    (
        Phi,
        Gamma
    ) = build_prediction_model(
        Ad,
        Bd,
        horizon
    )


    # ========================================================
    # 4. VALIDATION
    # ========================================================

    nx = Ac.shape[0]

    nu = Bc.shape[1]


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
            "Phi sai kích thước. "
            f"Expected {expected_Phi_shape}, "
            f"received {Phi.shape}."
        )


    if Gamma.shape != expected_Gamma_shape:

        raise RuntimeError(
            "Gamma sai kích thước. "
            f"Expected {expected_Gamma_shape}, "
            f"received {Gamma.shape}."
        )


    # ========================================================
    # 5. RETURN MPC PREDICTION MODEL
    # ========================================================

    return (
        Phi,
        Gamma
    )