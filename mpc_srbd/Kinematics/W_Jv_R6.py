import numpy as np
import mujoco


# ============================================================
# RIGHT LEG JOINT ORDER
# ============================================================
#
# Column order of W_Jv_R6:
#
#   0 : R_hip_yaw
#   1 : R_hip_roll
#   2 : R_hip_pitch
#   3 : R_knee
#   4 : R_ankle
#
# ============================================================

RIGHT_LEG_JOINT_NAMES = (
    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "R_ankle",
)


# ============================================================
# WORLD-FRAME LINEAR JACOBIAN AT R6
# ============================================================

def get_W_Jv_R6(
    model,
    data,
    site_name="R6"
):
    """
    Tính linear velocity Jacobian của site R6
    đối với 5 khớp chủ động của chân phải.

    Jacobian được biểu diễn trong World frame.

    Quan hệ:

        W_v_R6 = W_Jv_R6 @ dq_R

    với:

        dq_R =
        [
            dq_R_hip_yaw,
            dq_R_hip_roll,
            dq_R_hip_pitch,
            dq_R_knee,
            dq_R_ankle
        ]

    Returns
    -------
    W_Jv_R6 : np.ndarray
        Shape (3, 5)

        Linear Jacobian của R6,
        biểu diễn trong World frame.
    """

    # ========================================================
    # 1. Get site ID of R6
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
    #     jacr_full : rotational Jacobian
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
    # 3. Find DOF indices of right-leg joints
    # ========================================================

    right_dof_indices = []


    for joint_name in RIGHT_LEG_JOINT_NAMES:

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
        # All right-leg joints are hinge joints.
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


        right_dof_indices.append(
            dof_index
        )


    # ========================================================
    # 4. Extract only five right-leg columns
    # ========================================================

    W_Jv_R6 = jacp_full[
        :,
        right_dof_indices
    ].copy()


    # ========================================================
    # 5. Dimension check
    # ========================================================

    if W_Jv_R6.shape != (
        3,
        5
    ):

        raise RuntimeError(
            "W_Jv_R6 phải có shape (3, 5), "
            f"nhưng nhận được {W_Jv_R6.shape}."
        )


    # ========================================================
    # 6. Finite check
    # ========================================================

    if not np.all(
        np.isfinite(
            W_Jv_R6
        )
    ):

        raise RuntimeError(
            "W_Jv_R6 chứa NaN hoặc Inf."
        )


    return W_Jv_R6