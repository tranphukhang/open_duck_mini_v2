import numpy as np


# ============================================================
# VALIDATION
# ============================================================

def _validate_vector(
    value,
    name,
    size
):
    value = np.asarray(
        value,
        dtype=float
    ).reshape(-1)


    if value.shape != (
        size,
    ):

        raise ValueError(
            f"{name} phải có shape ({size},), "
            f"nhưng nhận được {value.shape}."
        )


    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    return value


def _validate_positive_scalar(
    value,
    name
):
    value = float(
        value
    )


    if (
        not np.isfinite(
            value
        )
        or
        value <= 0.0
    ):

        raise ValueError(
            f"{name} phải hữu hạn và > 0."
        )


    return value


def _validate_dcm_reference(
    W_xi_ref
):
    W_xi_ref = np.asarray(
        W_xi_ref,
        dtype=float
    )


    if W_xi_ref.ndim != 2:

        raise ValueError(
            "W_xi_ref phải là ma trận 2D."
        )


    if W_xi_ref.shape[
        1
    ] != 2:

        raise ValueError(
            "W_xi_ref phải có shape (N,2)."
        )


    if W_xi_ref.shape[
        0
    ] <= 0:

        raise ValueError(
            "W_xi_ref phải chứa ít nhất một mẫu."
        )


    if not np.all(
        np.isfinite(
            W_xi_ref
        )
    ):

        raise ValueError(
            "W_xi_ref chứa NaN hoặc Inf."
        )


    return W_xi_ref


# ============================================================
# CoM VELOCITY FROM DCM
#
# Flat-ground DCM relation:
#
#     xi
#       =
#     p_CoM
#       +
#     v_CoM / omega_0
#
#
# Therefore:
#
#     v_CoM
#       =
#     omega_0 (xi - p_CoM)
#
#
# Only horizontal x-y motion is considered here.
# ============================================================

def compute_com_velocity_xy(
    W_p_CoM_xy,
    W_xi_xy,
    omega0
):
    """
    Compute horizontal nominal CoM velocity from DCM.

    Parameters
    ----------
    W_p_CoM_xy : array_like, shape (2,)

        Nominal horizontal CoM position.


    W_xi_xy : array_like, shape (2,)

        Nominal horizontal DCM.


    omega0 : float

        DCM natural frequency.


    Returns
    -------
    W_v_CoM_xy : ndarray, shape (2,)
    """

    W_p_CoM_xy = _validate_vector(
        W_p_CoM_xy,
        "W_p_CoM_xy",
        2
    )


    W_xi_xy = _validate_vector(
        W_xi_xy,
        "W_xi_xy",
        2
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    return (

        omega0

        *

        (
            W_xi_xy
            -
            W_p_CoM_xy
        )
    )


# ============================================================
# ONE HEUN INTEGRATION STEP
#
# Nominal stable CoM dynamics:
#
#     p_dot
#       =
#     omega_0 (xi - p)
#
#
# Predictor:
#
#     v_start
#       =
#     omega_0 (xi_start - p_start)
#
#
#     p_predict
#       =
#     p_start
#       +
#     dt * v_start
#
#
# End derivative evaluated at predictor:
#
#     v_end_predict
#       =
#     omega_0 (xi_end - p_predict)
#
#
# Corrector:
#
#     p_end
#       =
#     p_start
#       +
#     dt/2
#     (v_start + v_end_predict)
#
#
# Final velocity:
#
#     v_end
#       =
#     omega_0 (xi_end - p_end)
# ============================================================

def integrate_com_xy_heun_step(
    W_p_CoM_start_xy,
    W_xi_start_xy,
    W_xi_end_xy,
    omega0,
    dt
):
    """
    Integrate one nominal horizontal CoM interval using
    Heun predictor-corrector integration.
    """

    W_p_CoM_start_xy = _validate_vector(
        W_p_CoM_start_xy,
        "W_p_CoM_start_xy",
        2
    )


    W_xi_start_xy = _validate_vector(
        W_xi_start_xy,
        "W_xi_start_xy",
        2
    )


    W_xi_end_xy = _validate_vector(
        W_xi_end_xy,
        "W_xi_end_xy",
        2
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    dt = _validate_positive_scalar(
        dt,
        "dt"
    )


    # ========================================================
    # 1. START VELOCITY
    # ========================================================

    W_v_start_xy = compute_com_velocity_xy(

        W_p_CoM_xy=(
            W_p_CoM_start_xy
        ),

        W_xi_xy=(
            W_xi_start_xy
        ),

        omega0=omega0
    )


    # ========================================================
    # 2. EULER PREDICTOR
    # ========================================================

    W_p_predict_xy = (

        W_p_CoM_start_xy

        +

        dt
        *
        W_v_start_xy
    )


    # ========================================================
    # 3. END VELOCITY AT PREDICTED POSITION
    # ========================================================

    W_v_end_predict_xy = (
        compute_com_velocity_xy(

            W_p_CoM_xy=(
                W_p_predict_xy
            ),

            W_xi_xy=(
                W_xi_end_xy
            ),

            omega0=omega0
        )
    )


    # ========================================================
    # 4. HEUN CORRECTOR
    # ========================================================

    W_p_CoM_end_xy = (

        W_p_CoM_start_xy

        +

        0.5
        *
        dt
        *
        (
            W_v_start_xy
            +
            W_v_end_predict_xy
        )
    )


    # ========================================================
    # 5. DCM-CONSISTENT END VELOCITY
    # ========================================================

    W_v_CoM_end_xy = (
        compute_com_velocity_xy(

            W_p_CoM_xy=(
                W_p_CoM_end_xy
            ),

            W_xi_xy=(
                W_xi_end_xy
            ),

            omega0=omega0
        )
    )


    return (
        W_p_CoM_end_xy,
        W_v_CoM_end_xy
    )


# ============================================================
# BUILD NOMINAL CoM REFERENCE FROM NOMINAL DCM
#
# Input:
#
#     p_CoM,k^nom
#
#     xi_k^nom
#
#     xi_ref =
#
#       [
#           xi_{k+1}^nom
#           xi_{k+2}^nom
#           ...
#           xi_{k+N}^nom
#       ]
#
#
# Output:
#
#     p_CoM_ref =
#
#       [
#           p_{k+1}^nom
#           ...
#           p_{k+N}^nom
#       ]
#
#
#     v_CoM_ref =
#
#       [
#           v_{k+1}^nom
#           ...
#           v_{k+N}^nom
#       ]
#
#
# IMPORTANT:
#
# Actual measured CoM is NOT used here.
#
# The nominal CoM state must be maintained / initialized by
# the reference-generation layer.
#
#
# Flat-ground assumptions:
#
#     z_CoM = constant
#
#     v_z = 0
# ============================================================

def build_com_reference_from_dcm(
    W_p_CoM_nominal_current,
    W_xi_nominal_current_xy,
    W_xi_ref,
    omega0,
    dt_mpc,
    com_height=None
):
    """
    Reconstruct future nominal CoM trajectory from nominal DCM.

    Parameters
    ----------
    W_p_CoM_nominal_current : array_like, shape (3,)

        Nominal CoM position at current time k.

        IMPORTANT:
            This is NOT the measured actual CoM.


    W_xi_nominal_current_xy : array_like, shape (2,)

        Nominal DCM at current time k.


    W_xi_ref : array_like, shape (N,2)

        Future nominal DCM:

            xi_{k+1} ... xi_{k+N}


    omega0 : float

        DCM natural frequency.


    dt_mpc : float

        Prediction-grid interval.


    com_height : float or None

        Constant nominal CoM height.

        If None:

            W_p_CoM_nominal_current[2]


    Returns
    -------
    result : dict

        result["W_p_CoM_ref"]
            shape (N,3)

        result["W_v_CoM_ref"]
            shape (N,3)

        result["W_p_CoM_sequence"]
            shape (N+1,3)

        result["W_v_CoM_sequence"]
            shape (N+1,3)

        result["W_xi_sequence"]
            shape (N+1,2)

        result["W_p_CoM_nominal_current"]
            shape (3,)

        result["W_v_CoM_nominal_current"]
            shape (3,)
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    W_p_CoM_nominal_current = _validate_vector(
        W_p_CoM_nominal_current,
        "W_p_CoM_nominal_current",
        3
    )


    W_xi_nominal_current_xy = _validate_vector(
        W_xi_nominal_current_xy,
        "W_xi_nominal_current_xy",
        2
    )


    W_xi_ref = _validate_dcm_reference(
        W_xi_ref
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    dt_mpc = _validate_positive_scalar(
        dt_mpc,
        "dt_mpc"
    )


    if com_height is None:

        com_height = float(
            W_p_CoM_nominal_current[
                2
            ]
        )


    com_height = _validate_positive_scalar(
        com_height,
        "com_height"
    )


    # --------------------------------------------------------
    # Flat-ground nominal state must already be located at the
    # requested constant CoM height.
    # --------------------------------------------------------

    if not np.isclose(
        W_p_CoM_nominal_current[
            2
        ],
        com_height,
        atol=1e-10,
        rtol=0.0
    ):

        raise ValueError(
            "W_p_CoM_nominal_current[2] phải bằng "
            "com_height trong flat-ground DCM planning."
        )


    N = int(
        W_xi_ref.shape[
            0
        ]
    )


    # ========================================================
    # 2. COMPLETE NOMINAL DCM SEQUENCE
    #
    # index 0:
    #
    #     xi_k^nom
    #
    # index 1...N:
    #
    #     xi_{k+1...k+N}^nom
    # ========================================================

    W_xi_sequence = np.vstack(
        (
            W_xi_nominal_current_xy.reshape(
                1,
                2
            ),

            W_xi_ref
        )
    )


    # ========================================================
    # 3. HORIZONTAL NOMINAL CoM ARRAYS
    # ========================================================

    W_p_CoM_xy = np.zeros(
        (
            N + 1,
            2
        ),
        dtype=float
    )


    W_v_CoM_xy = np.zeros(
        (
            N + 1,
            2
        ),
        dtype=float
    )


    W_p_CoM_xy[
        0
    ] = (
        W_p_CoM_nominal_current[
            0:2
        ]
    )


    W_v_CoM_xy[
        0
    ] = compute_com_velocity_xy(

        W_p_CoM_xy=(
            W_p_CoM_xy[
                0
            ]
        ),

        W_xi_xy=(
            W_xi_sequence[
                0
            ]
        ),

        omega0=omega0
    )


    # ========================================================
    # 4. INTEGRATE NOMINAL CoM THROUGH MPC GRID
    #
    # Interval j:
    #
    #     xi_j
    #       ->
    #     xi_j+1
    #
    #     p_j
    #       ->
    #     p_j+1
    # ========================================================

    for j in range(
        N
    ):

        (
            W_p_CoM_xy[
                j + 1
            ],

            W_v_CoM_xy[
                j + 1
            ]

        ) = integrate_com_xy_heun_step(

            W_p_CoM_start_xy=(
                W_p_CoM_xy[
                    j
                ]
            ),

            W_xi_start_xy=(
                W_xi_sequence[
                    j
                ]
            ),

            W_xi_end_xy=(
                W_xi_sequence[
                    j + 1
                ]
            ),

            omega0=omega0,

            dt=dt_mpc
        )


    # ========================================================
    # 5. 3D FLAT-GROUND NOMINAL SEQUENCE
    # ========================================================

    W_p_CoM_sequence = np.zeros(
        (
            N + 1,
            3
        ),
        dtype=float
    )


    W_v_CoM_sequence = np.zeros(
        (
            N + 1,
            3
        ),
        dtype=float
    )


    W_p_CoM_sequence[
        :,
        0:2
    ] = W_p_CoM_xy


    W_p_CoM_sequence[
        :,
        2
    ] = com_height


    W_v_CoM_sequence[
        :,
        0:2
    ] = W_v_CoM_xy


    # Flat-ground constant-height reference.
    W_v_CoM_sequence[
        :,
        2
    ] = 0.0


    # ========================================================
    # 6. FUTURE MPC REFERENCE
    #
    # Do not include nominal current sample k.
    # ========================================================

    W_p_CoM_ref = (
        W_p_CoM_sequence[
            1:
        ].copy()
    )


    W_v_CoM_ref = (
        W_v_CoM_sequence[
            1:
        ].copy()
    )


    # ========================================================
    # 7. OUTPUT
    # ========================================================

    return {
        "W_p_CoM_ref":
            W_p_CoM_ref,

        "W_v_CoM_ref":
            W_v_CoM_ref,

        "W_p_CoM_sequence":
            W_p_CoM_sequence,

        "W_v_CoM_sequence":
            W_v_CoM_sequence,

        "W_xi_sequence":
            W_xi_sequence,

        "W_p_CoM_nominal_current":
            W_p_CoM_sequence[
                0
            ].copy(),

        "W_v_CoM_nominal_current":
            W_v_CoM_sequence[
                0
            ].copy(),

        "horizon":
            N,

        "dt_mpc":
            dt_mpc,

        "com_height":
            com_height,
    }