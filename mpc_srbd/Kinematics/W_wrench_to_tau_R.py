import numpy as np

from W_Jv_R6 import get_W_Jv_R6
from W_Jw_R6 import get_W_Jw_R6


# ============================================================
# WORLD WRENCH AT R6 -> RIGHT LEG JOINT TORQUE
# ============================================================

def get_tau_R_from_W_wrench(
    model,
    data,
    W_wrench_R6
):
    """
    Mapping wrench tại R6 sang torque của 5 khớp chân phải
    bằng nguyên lý công ảo.

    Wrench của MPC được biểu diễn trong World frame:

        W_wrench_R6 =
        [
            W_F_R6
            W_M_R6
        ]

    với:

        W_F_R6 = [Fx, Fy, Fz]^T

        W_M_R6 = [Mx, My, Mz]^T


    Nguyên lý công ảo cho generalized torque do
    external wrench tác dụng lên robot:

        tau_ext =
            W_Jv_R6.T @ W_F_R6
            +
            W_Jw_R6.T @ W_M_R6


    Trong MPC hiện tại:

        W_wrench_R6

    là ground reaction wrench:

        ground -> robot


    Torque actuator cần tạo ra contribution ngược lại:

        tau_R = -tau_ext


    Do đó:

        tau_R =
        -(
            W_Jv_R6.T @ W_F_R6
            +
            W_Jw_R6.T @ W_M_R6
        )


    Joint order:

        tau_R =
        [
            tau_R_hip_yaw
            tau_R_hip_roll
            tau_R_hip_pitch
            tau_R_knee
            tau_R_ankle
        ]


    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data at current robot configuration.

    W_wrench_R6 : array_like
        World-frame wrench at R6.

        Shape:

            (6,)

        Ordering:

            [
                Fx, Fy, Fz,
                Mx, My, Mz
            ]


    Returns
    -------
    tau_R : np.ndarray
        Joint torque command for right leg.

        Shape:

            (5,)
    """

    # ========================================================
    # 1. Convert wrench
    # ========================================================

    W_wrench_R6 = np.asarray(
        W_wrench_R6,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # 2. Dimension check
    # ========================================================

    if W_wrench_R6.shape != (
        6,
    ):

        raise ValueError(
            "W_wrench_R6 phải có shape (6,), "
            f"nhưng nhận được {W_wrench_R6.shape}."
        )


    # ========================================================
    # 3. Finite check
    # ========================================================

    if not np.all(
        np.isfinite(
            W_wrench_R6
        )
    ):

        raise ValueError(
            "W_wrench_R6 chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. Split wrench
    #
    # Wrench ordering:
    #
    #     [Fx, Fy, Fz, Mx, My, Mz]
    # ========================================================

    W_F_R6 = W_wrench_R6[
        0:3
    ]


    W_M_R6 = W_wrench_R6[
        3:6
    ]


    # ========================================================
    # 5. Get Jacobians
    #
    # Both Jacobians are expressed in World frame.
    #
    #     W_Jv_R6 : (3, 5)
    #
    #     W_Jw_R6 : (3, 5)
    # ========================================================

    W_Jv_R6 = get_W_Jv_R6(
        model,
        data
    )


    W_Jw_R6 = get_W_Jw_R6(
        model,
        data
    )


    # ========================================================
    # 6. Virtual-work mapping
    #
    # Generalized torque caused by external GRF:
    #
    #     tau_ext =
    #
    #         Jv^T F
    #
    #         +
    #
    #         Jw^T M
    #
    # ========================================================

    tau_ext_force = (
        W_Jv_R6.T
        @ W_F_R6
    )


    tau_ext_moment = (
        W_Jw_R6.T
        @ W_M_R6
    )


    tau_ext = (
        tau_ext_force
        +
        tau_ext_moment
    )


    # ========================================================
    # 7. Actuator torque command
    #
    # Wrench from MPC:
    #
    #     ground -> robot
    #
    # Actuator/contact action:
    #
    #     robot -> ground
    #
    # Therefore:
    #
    #     tau_R = -tau_ext
    # ========================================================

    tau_R = -tau_ext


    # ========================================================
    # 8. Final checks
    # ========================================================

    if tau_R.shape != (
        5,
    ):

        raise RuntimeError(
            "tau_R phải có shape (5,), "
            f"nhưng nhận được {tau_R.shape}."
        )


    if not np.all(
        np.isfinite(
            tau_R
        )
    ):

        raise RuntimeError(
            "tau_R chứa NaN hoặc Inf."
        )


    return tau_R