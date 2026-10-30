import mujoco
import numpy as np


def get_mass_CoM(
    model,
    root_body_name="base_link"
):
    """
    Tính tổng khối lượng của toàn robot.

    Khối lượng SRBD:
        m_CoM = tổng mass của tất cả body
        thuộc subtree bắt đầu từ root_body_name.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    root_body_name : str, optional
        Body gốc của robot.
        Mặc định: "base_link".

    Returns
    -------
    mass_CoM : float
        Tổng khối lượng robot [kg].
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
    # 2. Tìm tất cả body thuộc subtree của root body
    # ---------------------------------------------------------
    subtree_body_ids = []

    for body_id in range(model.nbody):

        current_body_id = body_id

        while current_body_id != 0:

            if current_body_id == root_body_id:
                subtree_body_ids.append(body_id)
                break

            current_body_id = model.body_parentid[current_body_id]

    # ---------------------------------------------------------
    # 3. Cộng mass của tất cả body trong subtree
    # ---------------------------------------------------------
    mass_CoM = np.sum(
        model.body_mass[subtree_body_ids]
    )

    return float(mass_CoM)