import numpy as np

from W_wrench_to_tau_L import get_tau_L_from_W_wrench
from W_wrench_to_tau_R import get_tau_R_from_W_wrench


# ============================================================
# MPC WORLD WRENCH -> FULL JOINT TORQUE
# ============================================================

def get_joint_torque_from_MPC_wrench(
    model,
    data,
    u0_MPC
):
    """
    Mapping first MPC wrench input u0_MPC sang torque
    của toàn bộ 10 khớp chân.

    MPC input ordering:

        u0_MPC =
        [
            F_Lx, F_Ly, F_Lz,
            M_Lx, M_Ly, M_Lz,

            F_Rx, F_Ry, F_Rz,
            M_Rx, M_Ry, M_Rz
        ]

    tức:

        u0_MPC =
        [
            W_wrench_L6,
            W_wrench_R6
        ]

    với:

        W_wrench_L6 in R^6

        W_wrench_R6 in R^6


    Hai wrench được biểu diễn trong World frame.

    Mapping cho từng chân:

        tau_L =
        -(
            W_Jv_L6.T @ W_F_L6
            +
            W_Jw_L6.T @ W_M_L6
        )

        tau_R =
        -(
            W_Jv_R6.T @ W_F_R6
            +
            W_Jw_R6.T @ W_M_R6
        )


    Full joint torque ordering:

        tau =
        [
            tau_L_hip_yaw,
            tau_L_hip_roll,
            tau_L_hip_pitch,
            tau_L_knee,
            tau_L_ankle,

            tau_R_hip_yaw,
            tau_R_hip_roll,
            tau_R_hip_pitch,
            tau_R_knee,
            tau_R_ankle
        ]


    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data at current robot configuration.

    u0_MPC : array_like
        First MPC control input.

        Shape:

            (12,)


    Returns
    -------
    tau : np.ndarray
        Full 10-joint torque command.

        Shape:

            (10,)

    tau_L : np.ndarray
        Left-leg torque.

        Shape:

            (5,)

    tau_R : np.ndarray
        Right-leg torque.

        Shape:

            (5,)
    """

    # ========================================================
    # 1. Convert input
    # ========================================================

    u0_MPC = np.asarray(
        u0_MPC,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # 2. Dimension check
    # ========================================================

    if u0_MPC.shape != (
        12,
    ):

        raise ValueError(
            "u0_MPC phải có shape (12,), "
            f"nhưng nhận được {u0_MPC.shape}."
        )


    # ========================================================
    # 3. Finite check
    # ========================================================

    if not np.all(
        np.isfinite(
            u0_MPC
        )
    ):

        raise ValueError(
            "u0_MPC chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. Split MPC input
    #
    # u0 =
    #
    # [
    #     FLx FLy FLz
    #     MLx MLy MLz
    #
    #     FRx FRy FRz
    #     MRx MRy MRz
    # ]
    # ========================================================

    W_wrench_L6 = u0_MPC[
        0:6
    ].copy()


    W_wrench_R6 = u0_MPC[
        6:12
    ].copy()


    # ========================================================
    # 5. Left-leg torque
    # ========================================================

    tau_L = get_tau_L_from_W_wrench(
        model,
        data,
        W_wrench_L6
    )


    # ========================================================
    # 6. Right-leg torque
    # ========================================================

    tau_R = get_tau_R_from_W_wrench(
        model,
        data,
        W_wrench_R6
    )


    # ========================================================
    # 7. Combine joint torques
    #
    # Actuator order:
    #
    # Left:
    #
    #   L_hip_yaw
    #   L_hip_roll
    #   L_hip_pitch
    #   L_knee
    #   L_ankle
    #
    # Right:
    #
    #   R_hip_yaw
    #   R_hip_roll
    #   R_hip_pitch
    #   R_knee
    #   R_ankle
    # ========================================================

    tau = np.concatenate([
        tau_L,
        tau_R
    ])


    # ========================================================
    # 8. Final checks
    # ========================================================

    if tau.shape != (
        10,
    ):

        raise RuntimeError(
            "tau phải có shape (10,), "
            f"nhưng nhận được {tau.shape}."
        )


    if not np.all(
        np.isfinite(
            tau
        )
    ):

        raise RuntimeError(
            "tau chứa NaN hoặc Inf."
        )


    return (
        tau,
        tau_L,
        tau_R
    )