import mujoco
import numpy as np


def get_W_inertia_CoM(
    model,
    data,
    root_body_name="base_link"
):
    """
    Lấy composite inertia tensor của toàn robot tại whole-body CoM,
    biểu diễn theo World-oriented frame.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data hiện tại.
        Cần gọi mj_forward(model, data) trước khi dùng hàm này.

    root_body_name : str, optional
        Root body của robot.
        Mặc định: "base_link".

    Returns
    -------
    W_inertia_CoM : np.ndarray, shape (3, 3)
        Whole-body inertia tensor tại CoM:

        [ Ixx  Ixy  Ixz ]
        [ Ixy  Iyy  Iyz ]
        [ Ixz  Iyz  Izz ]

        Đơn vị: kg.m^2
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
    # 2. Composite rigid-body inertia
    #
    # data.crb[body_id] có 10 phần tử:
    #
    # [0] Ixx
    # [1] Iyy
    # [2] Izz
    # [3] Ixy
    # [4] Ixz
    # [5] Iyz
    # [6:9] first moment
    # [9] composite mass
    # ---------------------------------------------------------
    crb = data.crb[root_body_id].copy()

    # ---------------------------------------------------------
    # 3. Dựng lại ma trận inertia 3x3
    # ---------------------------------------------------------
    W_inertia_CoM = np.array([
        [crb[0], crb[3], crb[4]],
        [crb[3], crb[1], crb[5]],
        [crb[4], crb[5], crb[2]]
    ], dtype=float)

    return W_inertia_CoM