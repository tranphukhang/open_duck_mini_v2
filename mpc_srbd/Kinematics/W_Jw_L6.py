import numpy as np
import mujoco


# ============================================================
# LEFT LEG JOINT ORDER
# ============================================================
#
# Column order of W_Jw_L6:
#
#   0 : L_hip_yaw
#   1 : L_hip_roll
#   2 : L_hip_pitch
#   3 : L_knee
#   4 : L_ankle
#
# ============================================================

LEFT_LEG_JOINT_NAMES = (
    "L_hip_yaw",
    "L_hip_roll",
    "L_hip_pitch",
    "L_knee",
    "L_ankle",
)


# ============================================================
# WORLD-FRAME ANGULAR JACOBIAN AT L6
# ============================================================

def get_W_Jw_L6(
    model,
    data,
    site_name="L6"
):
    """
    Tính angular velocity Jacobian của site L6
    đối với 5 khớp chủ động của chân trái.

    Jacobian được biểu diễn trong World frame.

    Quan hệ:

        W_w_L6 = W_Jw_L6 @ dq_L

    với:

        dq_L =
        [
            dq_L_hip_yaw,
            dq_L_hip_roll,
            dq_L_hip_pitch,
            dq_L_knee,
            dq_L_ankle
        ]

    Returns
    -------
    W_Jw_L6 : np.ndarray
        Shape (3, 5)

        Angular velocity Jacobian của L6,
        biểu diễn trong World frame.
    """

    # ========================================================
    # 1. Get site ID of L6
    # ========================================================

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )


    if site_id == -1:

        raise ValueError(
            f"Không tìm thấy site '{site_name}'."
        )


    # ========================================================
    # 2. Compute full site Jacobian
    #
    # MuJoCo returns:
    #
    #     jacp_full : translational Jacobian
    #
    #     jacr_full : rotational / angular Jacobian
    #
    # Shape:
    #
    #     (3, model.nv)
    #
    # ========================================================

    jacp_full = np.zeros(
        (
            3,
            model.nv
        ),
        dtype=float
    )


    jacr_full = np.zeros(
        (
            3,
            model.nv
        ),
        dtype=float
    )


    mujoco.mj_jacSite(
        model,
        data,
        jacp_full,
        jacr_full,
        site_id
    )


    # ========================================================
    # 3. Find DOF indices of left-leg joints
    # ========================================================

    left_dof_indices = []


    for joint_name in LEFT_LEG_JOINT_NAMES:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )


        if joint_id == -1:

            raise ValueError(
                f"Không tìm thấy joint '{joint_name}'."
            )


        # ----------------------------------------------------
        # Each left-leg joint is a hinge joint.
        #
        # Each hinge joint has one DOF.
        #
        # model.jnt_dofadr[joint_id]
        #
        # gives the corresponding Jacobian column.
        # ----------------------------------------------------

        dof_index = int(
            model.jnt_dofadr[
                joint_id
            ]
        )


        left_dof_indices.append(
            dof_index
        )


    # ========================================================
    # 4. Extract only five left-leg columns
    #
    # Here we use jacr_full instead of jacp_full.
    # ========================================================

    W_Jw_L6 = jacr_full[
        :,
        left_dof_indices
    ].copy()


    # ========================================================
    # 5. Dimension check
    # ========================================================

    if W_Jw_L6.shape != (
        3,
        5
    ):

        raise RuntimeError(
            "W_Jw_L6 phải có shape (3, 5), "
            f"nhưng nhận được {W_Jw_L6.shape}."
        )


    # ========================================================
    # 6. Finite check
    # ========================================================

    if not np.all(
        np.isfinite(
            W_Jw_L6
        )
    ):

        raise RuntimeError(
            "W_Jw_L6 chứa NaN hoặc Inf."
        )


    return W_Jw_L6