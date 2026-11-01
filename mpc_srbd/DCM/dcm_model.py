import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_GRAVITY = 9.81


# ============================================================
# VALIDATION
# ============================================================

def _validate_vector(
    value,
    name,
    size
):
    """
    Validate a finite vector with a required size.
    """

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
    """
    Validate a finite positive scalar.
    """

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


# ============================================================
# DCM NATURAL FREQUENCY
#
# Flat-ground LIPM assumption:
#
#     z_CoM = z_0 = constant
#
# Natural frequency:
#
#     omega_0 = sqrt(g / z_0)
#
# where:
#
#     g   : gravitational acceleration
#     z_0 : nominal CoM height above flat ground
# ============================================================

def compute_omega0(
    com_height,
    gravity=DEFAULT_GRAVITY
):
    """
    Compute the natural frequency of the flat-ground LIPM.

    Parameters
    ----------
    com_height : float

        Constant nominal CoM height:

            z_0 > 0


    gravity : float

        Gravitational acceleration:

            g > 0


    Returns
    -------
    omega0 : float

        Natural frequency:

            omega_0 = sqrt(g / z_0)
    """

    com_height = _validate_positive_scalar(
        com_height,
        "com_height"
    )


    gravity = _validate_positive_scalar(
        gravity,
        "gravity"
    )


    omega0 = np.sqrt(
        gravity
        /
        com_height
    )


    return float(
        omega0
    )


# ============================================================
# DCM FROM CURRENT CoM STATE
#
# For flat-ground walking, only the horizontal x-y motion is
# used by the DCM model.
#
#     xi
#       =
#     p_CoM,xy
#       +
#     v_CoM,xy / omega_0
#
#
# Inputs are kept as 3D World-frame CoM quantities because
# this matches the State/ interface of the current project.
#
# Output:
#
#     xi_xy in World frame
# ============================================================

def compute_dcm_xy(
    W_p_CoM,
    W_v_CoM,
    omega0
):
    """
    Compute the horizontal Divergent Component of Motion.

    Parameters
    ----------
    W_p_CoM : array_like, shape (3,)

        Current CoM position expressed in World frame.


    W_v_CoM : array_like, shape (3,)

        Current CoM linear velocity expressed in World frame.


    omega0 : float

        LIPM natural frequency.


    Returns
    -------
    W_xi_xy : ndarray, shape (2,)

        Current horizontal DCM expressed in World frame:

            xi_xy
                =
            p_CoM,xy
                +
            v_CoM,xy / omega_0
    """

    W_p_CoM = _validate_vector(
        W_p_CoM,
        "W_p_CoM",
        3
    )


    W_v_CoM = _validate_vector(
        W_v_CoM,
        "W_v_CoM",
        3
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
    )


    W_xi_xy = (

        W_p_CoM[
            0:2
        ]

        +

        W_v_CoM[
            0:2
        ]
        /
        omega0
    )


    return W_xi_xy


# ============================================================
# DCM DYNAMICS
#
# Flat-ground DCM dynamics:
#
#     xi_dot
#       =
#     omega_0 (xi - p_support)
#
#
# p_support is the representative support point used by the
# DCM planner:
#
#     LSS -> left foot support point
#
#     RSS -> right foot support point
#
# In the current implementation this point represents the
# planned / realized foot-center location in the x-y plane.
# ============================================================

def compute_dcm_dot_xy(
    W_xi_xy,
    W_p_support_xy,
    omega0
):
    """
    Compute horizontal DCM velocity.

    Parameters
    ----------
    W_xi_xy : array_like, shape (2,)

        Current DCM position in World frame.


    W_p_support_xy : array_like, shape (2,)

        Current representative support point in World frame.


    omega0 : float

        LIPM natural frequency.


    Returns
    -------
    W_xi_dot_xy : ndarray, shape (2,)

        DCM velocity:

            xi_dot
                =
            omega_0 (xi - p_support)
    """

    W_xi_xy = _validate_vector(
        W_xi_xy,
        "W_xi_xy",
        2
    )


    W_p_support_xy = _validate_vector(
        W_p_support_xy,
        "W_p_support_xy",
        2
    )


    omega0 = _validate_positive_scalar(
        omega0,
        "omega0"
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
# CoM VELOCITY FROM DCM
#
# From:
#
#     xi
#       =
#     p_CoM
#       +
#     v_CoM / omega_0
#
# therefore:
#
#     v_CoM
#       =
#     omega_0 (xi - p_CoM)
#
#
# This equation will later be used when reconstructing the
# CoM reference from the desired DCM trajectory.
# ============================================================

def compute_com_velocity_xy_from_dcm(
    W_p_CoM,
    W_xi_xy,
    omega0
):
    """
    Recover horizontal CoM velocity from DCM and CoM position.

    Parameters
    ----------
    W_p_CoM : array_like, shape (3,)

        CoM position expressed in World frame.


    W_xi_xy : array_like, shape (2,)

        Horizontal DCM expressed in World frame.


    omega0 : float

        LIPM natural frequency.


    Returns
    -------
    W_v_CoM_xy : ndarray, shape (2,)

        Horizontal CoM velocity:

            v_CoM,xy
                =
            omega_0
            (xi_xy - p_CoM,xy)
    """

    W_p_CoM = _validate_vector(
        W_p_CoM,
        "W_p_CoM",
        3
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


    W_v_CoM_xy = (

        omega0

        *

        (
            W_xi_xy

            -

            W_p_CoM[
                0:2
            ]
        )
    )


    return W_v_CoM_xy