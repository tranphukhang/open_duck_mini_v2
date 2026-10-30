import mujoco
import numpy as np


def get_W_linear_velocity_CoM(
    model,
    data,
    root_body_name="base_link"
):
    """
    Lấy vận tốc tuyến tính của Center of Mass toàn robot
    biểu diễn trong World frame.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data hiện tại.

    root_body_name : str, optional
        Body gốc của toàn robot.
        Mặc định: "base_link".

    Returns
    -------
    W_linear_velocity_CoM : np.ndarray, shape (3,)
        Vận tốc tuyến tính whole-body CoM trong World frame:

        [vx_CoM, vy_CoM, vz_CoM]

        Đơn vị: [m/s]
    """

    # ---------------------------------------------------------
    # 1. Tìm ID của root body
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
    # 2. Đọc linear velocity của whole-body CoM
    #
    # data.subtree_linvel:
    # linear velocity của CoM của kinematic subtree,
    # biểu diễn trong World frame.
    #
    # Lưu ý:
    # mj_subtreeVel(model, data) phải được gọi trước đó.
    # ---------------------------------------------------------
    W_linear_velocity_CoM = (
        data.subtree_linvel[root_body_id].copy()
    )

    return np.asarray(
        W_linear_velocity_CoM,
        dtype=float
    )