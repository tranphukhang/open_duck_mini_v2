import numpy as np


# ============================================================
# SWING LEG JOINT PD CONTROLLER
# ============================================================

def compute_swing_joint_PD(
    q_des,
    q,
    q_dot,
    Kp,
    Kd
):
    """
    Compute joint torque for one swing leg using a joint-space
    PD controller.

    Control law
    -----------

        tau_PD
            =
        Kp * (q_des - q)
            -
        Kd * q_dot

    where the desired joint velocity is assumed to be zero:

        q_dot_des = 0


    Parameters
    ----------
    q_des : array_like, shape (5,)

        Desired swing-leg joint positions obtained from IK:

            [
                q_hip_yaw_des,
                q_hip_roll_des,
                q_hip_pitch_des,
                q_knee_des,
                q_ankle_des
            ]

        Unit:

            [rad]


    q : array_like, shape (5,)

        Current swing-leg joint positions:

            [
                q_hip_yaw,
                q_hip_roll,
                q_hip_pitch,
                q_knee,
                q_ankle
            ]

        Unit:

            [rad]


    q_dot : array_like, shape (5,)

        Current swing-leg joint velocities:

            [
                q_dot_hip_yaw,
                q_dot_hip_roll,
                q_dot_hip_pitch,
                q_dot_knee,
                q_dot_ankle
            ]

        Unit:

            [rad/s]


    Kp : float or array_like, shape (5,)

        Proportional gain.

        Can be:

            scalar

        or:

            [
                Kp_hip_yaw,
                Kp_hip_roll,
                Kp_hip_pitch,
                Kp_knee,
                Kp_ankle
            ]


    Kd : float or array_like, shape (5,)

        Derivative gain.

        Can be:

            scalar

        or:

            [
                Kd_hip_yaw,
                Kd_hip_roll,
                Kd_hip_pitch,
                Kd_knee,
                Kd_ankle
            ]


    Returns
    -------
    tau_PD : np.ndarray, shape (5,)

        Swing-leg joint torques:

            [
                tau_hip_yaw,
                tau_hip_roll,
                tau_hip_pitch,
                tau_knee,
                tau_ankle
            ]

        Unit:

            [N.m]
    """


    # ========================================================
    # 1. CONVERT JOINT INPUTS
    # ========================================================

    q_des = np.asarray(
        q_des,
        dtype=float
    ).reshape(-1)


    q = np.asarray(
        q,
        dtype=float
    ).reshape(-1)


    q_dot = np.asarray(
        q_dot,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # 2. VALIDATE JOINT INPUTS
    # ========================================================

    if q_des.shape != (
        5,
    ):

        raise ValueError(
            "q_des phải có shape (5,), "
            f"nhưng nhận được {q_des.shape}."
        )


    if q.shape != (
        5,
    ):

        raise ValueError(
            "q phải có shape (5,), "
            f"nhưng nhận được {q.shape}."
        )


    if q_dot.shape != (
        5,
    ):

        raise ValueError(
            "q_dot phải có shape (5,), "
            f"nhưng nhận được {q_dot.shape}."
        )


    if not np.all(
        np.isfinite(
            q_des
        )
    ):

        raise ValueError(
            "q_des chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            q
        )
    ):

        raise ValueError(
            "q chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            q_dot
        )
    ):

        raise ValueError(
            "q_dot chứa NaN hoặc Inf."
        )


    # ========================================================
    # 3. CONVERT PD GAINS
    # ========================================================

    Kp = _convert_gain(
        Kp,
        "Kp"
    )


    Kd = _convert_gain(
        Kd,
        "Kd"
    )


    # ========================================================
    # 4. JOINT POSITION ERROR
    #
    #     e_q
    #
    # =
    #
    #     q_des - q
    # ========================================================

    q_error = (
        q_des
        -
        q
    )


    # ========================================================
    # 5. PD CONTROL LAW
    #
    # Desired joint velocity:
    #
    #     q_dot_des = 0
    #
    # Therefore:
    #
    #     tau_PD
    #
    # =
    #
    #     Kp * (q_des - q)
    #
    #     -
    #
    #     Kd * q_dot
    # ========================================================

    tau_PD = (
        Kp
        *
        q_error
        -
        Kd
        *
        q_dot
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
# GAIN CONVERSION
# ============================================================

def _convert_gain(
    gain,
    gain_name
):
    """
    Convert scalar or 5-element gain into a 5-element vector.

    Examples
    --------

        Kp = 20

    becomes:

        [20, 20, 20, 20, 20]


    while:

        Kp = [20, 20, 30, 30, 20]

    is kept unchanged.
    """


    gain = np.asarray(
        gain,
        dtype=float
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
            dtype=float
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