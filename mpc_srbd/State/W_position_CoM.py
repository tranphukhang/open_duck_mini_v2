import mujoco
import numpy as np


def get_W_position_CoM(model, data, root_body_name="base_link"):
    """
    Lấy vị trí Center of Mass của toàn robot biểu diễn trong World frame.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data hiện tại.

    root_body_name : str, optional
        Tên body gốc của toàn robot.
        Mặc định: "base_link".

    Returns
    -------
    W_position_CoM : np.ndarray, shape (3,)
        Vị trí whole-body CoM trong World frame:

        [x_CoM, y_CoM, z_CoM]

        Đơn vị: [m]
    """

    # ---------------------------------------------------------
    # 1. Tìm body ID của root body
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
    # 2. Đọc CoM của toàn bộ subtree
    #
    # data.subtree_com:
    # Center of Mass của toàn bộ kinematic subtree
    # biểu diễn trong World frame.
    # ---------------------------------------------------------
    W_position_CoM = data.subtree_com[root_body_id].copy()

    return np.asarray(W_position_CoM, dtype=float)