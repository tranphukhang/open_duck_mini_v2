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


def _validate_ds_time(
    time,
    T_DS
):
    time = float(
        time
    )


    T_DS = _validate_positive_scalar(
        T_DS,
        "T_DS"
    )


    if not np.isfinite(
        time
    ):

        raise ValueError(
            "time phải hữu hạn."
        )


    if (
        time < 0.0
        or
        time > T_DS
    ):

        raise ValueError(
            "time phải nằm trong khoảng "
            "[0, T_DS]."
        )


    return (
        time,
        T_DS
    )


# ============================================================
# NORMALIZED DOUBLE-SUPPORT PHASE
#
#     s = t / T_DS
#
# with:
#
#     0 <= s <= 1
# ============================================================

def compute_ds_phase(
    time,
    T_DS
):
    """
    Compute normalized Double Support phase.

    Returns
    -------
    phase : float

        phase = time / T_DS

        phase in [0, 1]
    """

    (
        time,
        T_DS

    ) = _validate_ds_time(
        time,
        T_DS
    )


    return float(
        time
        /
        T_DS
    )


# ============================================================
# CUBIC HERMITE BASIS
#
# Position basis:
#
#     h00 =  2s^3 - 3s^2 + 1
#     h10 =    s^3 - 2s^2 + s
#     h01 = -2s^3 + 3s^2
#     h11 =    s^3 -   s^2
#
# ============================================================

def compute_hermite_basis(
    phase
):
    phase = float(
        phase
    )


    if (
        not np.isfinite(
            phase
        )
        or
        phase < 0.0
        or
        phase > 1.0
    ):

        raise ValueError(
            "phase phải hữu hạn và nằm trong [0,1]."
        )


    s = phase

    s2 = (
        s
        *
        s
    )

    s3 = (
        s2
        *
        s
    )


    h00 = (
        2.0 * s3
        -
        3.0 * s2
        +
        1.0
    )


    h10 = (
        s3
        -
        2.0 * s2
        +
        s
    )


    h01 = (
        -2.0 * s3
        +
        3.0 * s2
    )


    h11 = (
        s3
        -
        s2
    )


    return (
        h00,
        h10,
        h01,
        h11
    )


# ============================================================
# DERIVATIVE OF HERMITE BASIS WITH RESPECT TO s
#
#     dh00/ds =  6s^2 - 6s
#     dh10/ds =  3s^2 - 4s + 1
#     dh01/ds = -6s^2 + 6s
#     dh11/ds =  3s^2 - 2s
#
# ============================================================

def compute_hermite_basis_derivative(
    phase
):
    phase = float(
        phase
    )


    if (
        not np.isfinite(
            phase
        )
        or
        phase < 0.0
        or
        phase > 1.0
    ):

        raise ValueError(
            "phase phải hữu hạn và nằm trong [0,1]."
        )


    s = phase

    s2 = (
        s
        *
        s
    )


    dh00 = (
        6.0 * s2
        -
        6.0 * s
    )


    dh10 = (
        3.0 * s2
        -
        4.0 * s
        +
        1.0
    )


    dh01 = (
        -6.0 * s2
        +
        6.0 * s
    )


    dh11 = (
        3.0 * s2
        -
        2.0 * s
    )


    return (
        dh00,
        dh10,
        dh01,
        dh11
    )


# ============================================================
# DOUBLE SUPPORT DCM
#
# Cubic Hermite interpolation:
#
#     xi_DS(t)
#
#       =
#
#     h00 xi_start
#
#       +
#
#     h10 T_DS xi_dot_start
#
#       +
#
#     h01 xi_end
#
#       +
#
#     h11 T_DS xi_dot_end
#
#
# This guarantees:
#
#     xi(0)        = xi_start
#     xi_dot(0)    = xi_dot_start
#
#     xi(T_DS)     = xi_end
#     xi_dot(T_DS) = xi_dot_end
#
# ============================================================

def evaluate_ds_dcm(
    W_xi_start_xy,
    W_xi_dot_start_xy,
    W_xi_end_xy,
    W_xi_dot_end_xy,
    time,
    T_DS
):
    """
    Evaluate the DCM reference during Double Support.

    Parameters
    ----------
    W_xi_start_xy : array_like, shape (2,)

        DCM at the beginning of DS.


    W_xi_dot_start_xy : array_like, shape (2,)

        DCM velocity at the beginning of DS.


    W_xi_end_xy : array_like, shape (2,)

        DCM at the end of DS.


    W_xi_dot_end_xy : array_like, shape (2,)

        DCM velocity at the end of DS.


    time : float

        Elapsed DS time:

            0 <= time <= T_DS


    T_DS : float

        Double Support duration.


    Returns
    -------
    W_xi_xy : ndarray, shape (2,)

        DCM reference during DS.
    """

    W_xi_start_xy = _validate_vector(
        W_xi_start_xy,
        "W_xi_start_xy",
        2
    )


    W_xi_dot_start_xy = _validate_vector(
        W_xi_dot_start_xy,
        "W_xi_dot_start_xy",
        2
    )


    W_xi_end_xy = _validate_vector(
        W_xi_end_xy,
        "W_xi_end_xy",
        2
    )


    W_xi_dot_end_xy = _validate_vector(
        W_xi_dot_end_xy,
        "W_xi_dot_end_xy",
        2
    )


    (
        time,
        T_DS

    ) = _validate_ds_time(
        time,
        T_DS
    )


    phase = (
        time
        /
        T_DS
    )


    (
        h00,
        h10,
        h01,
        h11

    ) = compute_hermite_basis(
        phase
    )


    W_xi_xy = (

        h00
        *
        W_xi_start_xy

        +

        h10
        *
        T_DS
        *
        W_xi_dot_start_xy

        +

        h01
        *
        W_xi_end_xy

        +

        h11
        *
        T_DS
        *
        W_xi_dot_end_xy
    )


    return W_xi_xy


# ============================================================
# DOUBLE SUPPORT DCM VELOCITY
#
# Since:
#
#     s = t / T_DS
#
# therefore:
#
#     d/dt = (1/T_DS) d/ds
#
# ============================================================

def evaluate_ds_dcm_velocity(
    W_xi_start_xy,
    W_xi_dot_start_xy,
    W_xi_end_xy,
    W_xi_dot_end_xy,
    time,
    T_DS
):
    """
    Evaluate DCM velocity during Double Support.
    """

    W_xi_start_xy = _validate_vector(
        W_xi_start_xy,
        "W_xi_start_xy",
        2
    )


    W_xi_dot_start_xy = _validate_vector(
        W_xi_dot_start_xy,
        "W_xi_dot_start_xy",
        2
    )


    W_xi_end_xy = _validate_vector(
        W_xi_end_xy,
        "W_xi_end_xy",
        2
    )


    W_xi_dot_end_xy = _validate_vector(
        W_xi_dot_end_xy,
        "W_xi_dot_end_xy",
        2
    )


    (
        time,
        T_DS

    ) = _validate_ds_time(
        time,
        T_DS
    )


    phase = (
        time
        /
        T_DS
    )


    (
        dh00,
        dh10,
        dh01,
        dh11

    ) = compute_hermite_basis_derivative(
        phase
    )


    W_xi_dot_xy = (

        dh00
        *
        W_xi_start_xy
        /
        T_DS

        +

        dh10
        *
        W_xi_dot_start_xy

        +

        dh01
        *
        W_xi_end_xy
        /
        T_DS

        +

        dh11
        *
        W_xi_dot_end_xy
    )


    return W_xi_dot_xy