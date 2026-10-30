import mujoco
import numpy as np


def get_W_moment_arm(
    model,
    data,
    left_site_name="L6",
    right_site_name="R6",
    root_body_name="base_link"
):
    """
    Tính moment arm từ whole-body CoM đến hai site L6 và R6,
    biểu diễn trong World frame.

    Định nghĩa:

        W_r_L = W_p_L6 - W_p_CoM
        W_r_R = W_p_R6 - W_p_CoM

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data hiện tại.
        Cần gọi mj_forward(model, data) trước khi dùng hàm.

    left_site_name : str, optional
        Site quy chiếu wrench của chân trái.
        Mặc định: "L6".

    right_site_name : str, optional
        Site quy chiếu wrench của chân phải.
        Mặc định: "R6".

    root_body_name : str, optional
        Root body của toàn robot.
        Mặc định: "base_link".

    Returns
    -------
    W_r_L : np.ndarray, shape (3,)
        Moment arm từ CoM đến L6 trong World frame [m].

    W_r_R : np.ndarray, shape (3,)
        Moment arm từ CoM đến R6 trong World frame [m].
    """

    # ---------------------------------------------------------
    # 1. Root body ID
    # ---------------------------------------------------------
    root_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        root_body_name
    )

    if root_body_id == -1:
        raise ValueError(
            f"Không tìm thấy body '{root_body_name}' trong MuJoCo model."
        )

    # ---------------------------------------------------------
    # 2. Site IDs
    # ---------------------------------------------------------
    left_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        left_site_name
    )

    right_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        right_site_name
    )

    if left_site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{left_site_name}' trong MuJoCo model."
        )

    if right_site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{right_site_name}' trong MuJoCo model."
        )

    # ---------------------------------------------------------
    # 3. Whole-body CoM position in World frame
    # ---------------------------------------------------------
    W_p_CoM = data.subtree_com[root_body_id].copy()

    # ---------------------------------------------------------
    # 4. L6 and R6 positions in World frame
    # ---------------------------------------------------------
    W_p_L6 = data.site_xpos[left_site_id].copy()
    W_p_R6 = data.site_xpos[right_site_id].copy()

    # ---------------------------------------------------------
    # 5. Moment arms
    #
    # Vector direction:
    # CoM ---> L6
    # CoM ---> R6
    # ---------------------------------------------------------
    W_r_L = W_p_L6 - W_p_CoM
    W_r_R = W_p_R6 - W_p_CoM

    return (
        np.asarray(W_r_L, dtype=float),
        np.asarray(W_r_R, dtype=float)
    )