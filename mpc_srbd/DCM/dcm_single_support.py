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


def _validate_nonnegative_scalar(
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
        value < 0.0
    ):

        raise ValueError(
            f"{name} phải hữu hạn và >= 0."
        )


    return value


# ============================================================
# SINGLE SUPPORT DCM FORWARD SOLUTION
#
# DCM dynamics:
#
#     xi_dot
#       =
#     omega_0 (xi - p_support)
#
#
# For constant support point during Single Support:
#
#     xi(t)
#       =
#     p_support
#       +
#     exp(omega_0 t)
#     (xi_0 - p_support)
#
#
# where:
#
#     xi_0
#
# is the DCM at the beginning of the considered SS interval.
# ============================================================

def evaluate_ss_dcm(
    W_xi_initial_xy,
    W_p_support_xy,
    time,
    omega0
):
    """
    Evaluate the analytical DCM solution during Single Support.

    Parameters
    ----------
    W_xi_initial_xy : array_like, shape (2,)

        DCM at the beginning of the considered SS interval.


    W_p_support_xy : array_like, shape (2,)

        Constant representative support point during SS.


    time : float

        Elapsed time from the beginning of the considered
        SS interval:

            time >= 0


    omega0 : float

        DCM / LIPM natural frequency.


    Returns
    -------
    W_xi_xy : ndarray, shape (2,)

        DCM position at the requested time:

            xi(t)
                =
            p_support
                +
            exp(omega0*t)
            (xi_initial - p_support)
    """

    W_xi_initial_xy = _validate_vector(
        W_xi_initial_xy,
        "W_xi_initial_xy",
        2
    )


    W_p_support_xy = _validate_vector(
        W_p_support_xy,
        "W_p_support_xy",
        2
    )


    time = _validate_nonnegative_scalar(
        time,
        "time"
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    exponential = np.exp(
        omega0
        *
        time
    )


    W_xi_xy = (

        W_p_support_xy

        +

        exponential

        *

        (
            W_xi_initial_xy
            -
            W_p_support_xy
        )
    )


    return W_xi_xy


# ============================================================
# SINGLE SUPPORT DCM VELOCITY
#
# From the DCM model:
#
#     xi_dot
#       =
#     omega_0 (xi - p_support)
#
# ============================================================

def evaluate_ss_dcm_velocity(
    W_xi_initial_xy,
    W_p_support_xy,
    time,
    omega0
):
    """
    Evaluate DCM velocity during Single Support.
    """

    W_xi_xy = evaluate_ss_dcm(

        W_xi_initial_xy=W_xi_initial_xy,

        W_p_support_xy=W_p_support_xy,

        time=time,

        omega0=omega0
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    W_p_support_xy = _validate_vector(
        W_p_support_xy,
        "W_p_support_xy",
        2
    )


    W_xi_dot_xy = (

        omega0

        *

        (
            W_xi_xy
            -
            W_p_support_xy
        )
    )


    return W_xi_dot_xy


# ============================================================
# SINGLE SUPPORT BACKWARD BOUNDARY CONDITION
#
# Forward relation:
#
#     xi_end
#       =
#     p_support
#       +
#     exp(omega_0 T_SS)
#     (xi_initial - p_support)
#
#
# Rearranging:
#
#     xi_initial
#       =
#     p_support
#       +
#     exp(-omega_0 T_SS)
#     (xi_end - p_support)
#
#
# This relation is important for DCM planning because future
# SS boundary conditions are typically propagated backward
# from a desired end-of-step DCM.
# ============================================================

def compute_ss_initial_dcm_from_end(
    W_p_support_xy,
    W_xi_end_xy,
    T_SS,
    omega0
):
    """
    Compute the required initial DCM so that after a complete
    Single Support phase the DCM reaches W_xi_end_xy.

    Parameters
    ----------
    W_p_support_xy : array_like, shape (2,)

        Constant support point during the SS phase.


    W_xi_end_xy : array_like, shape (2,)

        Desired DCM at the end of the SS phase.


    T_SS : float

        Single Support duration:

            T_SS > 0


    omega0 : float

        DCM natural frequency.


    Returns
    -------
    W_xi_initial_xy : ndarray, shape (2,)

        Required initial DCM:

            xi_initial
                =
            p_support
                +
            exp(-omega0*T_SS)
            (xi_end - p_support)
    """

    W_p_support_xy = _validate_vector(
        W_p_support_xy,
        "W_p_support_xy",
        2
    )


    W_xi_end_xy = _validate_vector(
        W_xi_end_xy,
        "W_xi_end_xy",
        2
    )


    T_SS = _validate_positive_scalar(
        T_SS,
        "T_SS"
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    exponential = np.exp(
        -omega0
        *
        T_SS
    )


    W_xi_initial_xy = (

        W_p_support_xy

        +

        exponential

        *

        (
            W_xi_end_xy
            -
            W_p_support_xy
        )
    )


    return W_xi_initial_xy


# ============================================================
# SPECIAL CASE:
# END DCM = NEXT SUPPORT POINT
#
# A common flat-ground walking choice is:
#
#     xi_end = p_next_support
#
#
# Therefore:
#
#     xi_initial
#       =
#     p_support
#       +
#     exp(-omega_0 T_SS)
#     (p_next_support - p_support)
#
# ============================================================

def compute_ss_initial_dcm_from_next_support(
    W_p_support_xy,
    W_p_next_support_xy,
    T_SS,
    omega0
):
    """
    Compute initial DCM assuming the DCM should reach the next
    support point at the end of the current Single Support.

    This is a convenience wrapper around:

        compute_ss_initial_dcm_from_end(...)
    """

    W_p_next_support_xy = _validate_vector(
        W_p_next_support_xy,
        "W_p_next_support_xy",
        2
    )


    return compute_ss_initial_dcm_from_end(

        W_p_support_xy=W_p_support_xy,

        W_xi_end_xy=W_p_next_support_xy,

        T_SS=T_SS,

        omega0=omega0
    )