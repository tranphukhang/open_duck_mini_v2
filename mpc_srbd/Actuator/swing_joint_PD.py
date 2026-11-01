import numpy as np


# ============================================================
# SWING LEG JOINT PD CONTROLLER
# ============================================================

def compute_swing_joint_PD(
    q_des,
    q,
    q_dot,
    Kp,
    Kd,
    q_dot_des=None,
):
    """
    Compute joint torque for one swing leg using joint-space PD.

    Control law
    -----------

        tau_PD
            =
        Kp * (q_des - q)
            +
        Kd * (q_dot_des - q_dot)


    Backward compatibility
    ----------------------

    If q_dot_des is omitted:

        q_dot_des = 0

    and the controller becomes the previous implementation:

        tau_PD
            =
        Kp * (q_des - q)
            -
        Kd * q_dot


    Joint order
    -----------

        0. hip yaw
        1. hip roll
        2. hip pitch
        3. knee
        4. ankle


    Parameters
    ----------
    q_des : array_like, shape (5,)
        Desired joint positions [rad].

    q : array_like, shape (5,)
        Current joint positions [rad].

    q_dot : array_like, shape (5,)
        Current joint velocities [rad/s].

    Kp : float or array_like, shape (5,)
        Proportional gains.

    Kd : float or array_like, shape (5,)
        Derivative gains.

    q_dot_des : None or array_like, shape (5,), optional
        Desired joint velocities [rad/s].

        If None:
            q_dot_des = zeros(5)


    Returns
    -------
    tau_PD : np.ndarray, shape (5,)
        Swing-leg joint torques [N.m].
    """

    # ========================================================
    # 1. CONVERT JOINT INPUTS
    # ========================================================

    q_des = _convert_joint_vector(
        q_des,
        "q_des",
    )

    q = _convert_joint_vector(
        q,
        "q",
    )

    q_dot = _convert_joint_vector(
        q_dot,
        "q_dot",
    )


    # ========================================================
    # 2. DESIRED JOINT VELOCITY
    # ========================================================

    if q_dot_des is None:

        q_dot_des = np.zeros(
            5,
            dtype=float,
        )

    else:

        q_dot_des = _convert_joint_vector(
            q_dot_des,
            "q_dot_des",
        )


    # ========================================================
    # 3. CONVERT PD GAINS
    # ========================================================

    Kp = _convert_gain(
        Kp,
        "Kp",
    )

    Kd = _convert_gain(
        Kd,
        "Kd",
    )


    # ========================================================
    # 4. JOINT ERRORS
    #
    # Position:
    #
    #     e_q = q_des - q
    #
    # Velocity:
    #
    #     e_q_dot = q_dot_des - q_dot
    # ========================================================

    q_error = (
        q_des
        -
        q
    )

    q_dot_error = (
        q_dot_des
        -
        q_dot
    )


    # ========================================================
    # 5. PD CONTROL LAW
    #
    #     tau_PD
    #         =
    #     Kp * e_q
    #         +
    #     Kd * e_q_dot
    # ========================================================

    tau_PD = (
        Kp
        *
        q_error

        +

        Kd
        *
        q_dot_error
    )


    # ========================================================
    # 6. OUTPUT VALIDATION
    # ========================================================

    if tau_PD.shape != (
        5,
    ):

        raise RuntimeError(
            "tau_PD phải có shape (5,), "
            f"nhưng nhận được {tau_PD.shape}."
        )


    if not np.all(
        np.isfinite(
            tau_PD
        )
    ):

        raise RuntimeError(
            "tau_PD chứa NaN hoặc Inf."
        )


    # ========================================================
    # 7. RETURN
    # ========================================================

    return tau_PD


# ============================================================
# JOINT VECTOR CONVERSION
# ============================================================

def _convert_joint_vector(
    value,
    value_name,
):

    value = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)


    if value.shape != (
        5,
    ):

        raise ValueError(
            f"{value_name} phải có shape (5,), "
            f"nhưng nhận được {value.shape}."
        )


    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{value_name} chứa NaN hoặc Inf."
        )


    return value.copy()


# ============================================================
# GAIN CONVERSION
# ============================================================

def _convert_gain(
    gain,
    gain_name,
):
    """
    Convert scalar or 5-element gain into a 5-element vector.
    """

    gain = np.asarray(
        gain,
        dtype=float,
    )


    # ========================================================
    # SCALAR GAIN
    # ========================================================

    if gain.ndim == 0:

        gain_value = float(
            gain
        )


        if not np.isfinite(
            gain_value
        ):

            raise ValueError(
                f"{gain_name} phải hữu hạn."
            )


        if gain_value < 0.0:

            raise ValueError(
                f"{gain_name} phải >= 0."
            )


        return np.full(
            5,
            gain_value,
            dtype=float,
        )


    # ========================================================
    # VECTOR GAIN
    # ========================================================

    gain = gain.reshape(-1)


    if gain.shape != (
        5,
    ):

        raise ValueError(
            f"{gain_name} phải là scalar hoặc "
            f"vector shape (5,), "
            f"nhưng nhận được {gain.shape}."
        )


    if not np.all(
        np.isfinite(
            gain
        )
    ):

        raise ValueError(
            f"{gain_name} chứa NaN hoặc Inf."
        )


    if np.any(
        gain < 0.0
    ):

        raise ValueError(
            f"Tất cả phần tử của {gain_name} phải >= 0."
        )


    return gain.copy()
