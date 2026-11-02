from dataclasses import dataclass

import mujoco
import numpy as np

from W_position_CoM import get_W_position_CoM
from W_linear_velocity_CoM import get_W_linear_velocity_CoM
from W_angular_velocity_base import get_W_angular_velocity_base
from W_euler_base import get_W_euler_base, get_Ew_inverse
from W_moment_arm import get_W_moment_arm
from W_R_6 import get_W_R_6
from W_R_6_IK import get_W_R_6_IK
from W_inertia_CoM import get_W_inertia_CoM_from_feedback


@dataclass
class CurrentMPCState:
    """
    Feedback snapshot used to build one MPC problem.
    """

    # x = [p_CoM, euler, v_CoM, omega, 1]
    x: np.ndarray

    # Current feedback-linearized SRBD quantities
    Ew_inverse: np.ndarray
    W_inertia_CoM: np.ndarray
    W_r_L: np.ndarray
    W_r_R: np.ndarray

    # Physical L6 / R6 orientations
    L6_R_W: np.ndarray
    R6_R_W: np.ndarray

    # Corrected contact-frame orientations
    L6_IK_R_W: np.ndarray
    R6_IK_R_W: np.ndarray


def _validate_array(value, expected_shape, name):
    value = np.asarray(value, dtype=float)

    if value.shape != expected_shape:
        raise RuntimeError(
            f"{name} phải có shape {expected_shape}, "
            f"nhưng nhận được {value.shape}."
        )

    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"{name} chứa NaN hoặc Inf.")

    return value


def get_current_state_mpc(
    model,
    data,
    B_inertia_CoM,
):
    """
    Read one synchronized feedback snapshot before the MPC solve.

    At control instant k this returns:

        x_k                       <- feedback
        Ew_inverse(k)             <- Euler feedback
        W_r_L(k), W_r_R(k)        <- foot/CoM feedback
        W_I_CoM(k)                <- orientation feedback

    where the SRBD body-frame inertia is constant:

        B_I_CoM = constant

    and:

        W_I_CoM(k)
            =
        W_R_B(k) @ B_I_CoM @ W_R_B(k).T

    No mj_step() is executed here.
    """

    B_inertia_CoM = _validate_array(
        B_inertia_CoM,
        (3, 3),
        "B_inertia_CoM",
    )

    # Update position-dependent and velocity-dependent MuJoCo quantities
    # without advancing simulation time.
    mujoco.mj_forward(
        model,
        data,
    )

    mujoco.mj_subtreeVel(
        model,
        data,
    )

    W_position_CoM = np.asarray(
        get_W_position_CoM(
            model,
            data,
        ),
        dtype=float,
    ).reshape(3)

    W_euler_base = np.asarray(
        get_W_euler_base(
            model,
            data,
        ),
        dtype=float,
    ).reshape(3)

    W_linear_velocity_CoM = np.asarray(
        get_W_linear_velocity_CoM(
            model,
            data,
        ),
        dtype=float,
    ).reshape(3)

    W_angular_velocity_base = np.asarray(
        get_W_angular_velocity_base(
            model,
            data,
        ),
        dtype=float,
    ).reshape(3)

    x_current = np.concatenate(
        (
            W_position_CoM,
            W_euler_base,
            W_linear_velocity_CoM,
            W_angular_velocity_base,
            np.array([1.0], dtype=float),
        )
    )

    _validate_array(
        x_current,
        (13,),
        "x_current",
    )

    Ew_inverse = np.asarray(
        get_Ew_inverse(
            W_euler_base
        ),
        dtype=float,
    )

    _validate_array(
        Ew_inverse,
        (3, 3),
        "Ew_inverse",
    )

    W_r_L, W_r_R = get_W_moment_arm(
        model,
        data,
    )

    W_r_L = _validate_array(
        np.asarray(W_r_L, dtype=float).reshape(3),
        (3,),
        "W_r_L",
    )

    W_r_R = _validate_array(
        np.asarray(W_r_R, dtype=float).reshape(3),
        (3,),
        "W_r_R",
    )

    W_inertia_CoM = get_W_inertia_CoM_from_feedback(
        model=model,
        data=data,
        B_inertia_CoM=B_inertia_CoM,
    )

    W_inertia_CoM = _validate_array(
        W_inertia_CoM,
        (3, 3),
        "W_inertia_CoM",
    )

    L6_R_W, R6_R_W = get_W_R_6(
        model,
        data,
    )

    L6_R_W = _validate_array(
        L6_R_W,
        (3, 3),
        "L6_R_W",
    )

    R6_R_W = _validate_array(
        R6_R_W,
        (3, 3),
        "R6_R_W",
    )

    L6_IK_R_W, R6_IK_R_W = get_W_R_6_IK(
        model,
        data,
    )

    L6_IK_R_W = _validate_array(
        L6_IK_R_W,
        (3, 3),
        "L6_IK_R_W",
    )

    R6_IK_R_W = _validate_array(
        R6_IK_R_W,
        (3, 3),
        "R6_IK_R_W",
    )

    return CurrentMPCState(
        x=x_current.copy(),
        Ew_inverse=Ew_inverse.copy(),
        W_inertia_CoM=W_inertia_CoM.copy(),
        W_r_L=W_r_L.copy(),
        W_r_R=W_r_R.copy(),
        L6_R_W=L6_R_W.copy(),
        R6_R_W=R6_R_W.copy(),
        L6_IK_R_W=L6_IK_R_W.copy(),
        R6_IK_R_W=R6_IK_R_W.copy(),
    )
