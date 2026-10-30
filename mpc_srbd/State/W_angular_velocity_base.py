import mujoco
import numpy as np


def get_W_angular_velocity_base(
    model,
    data,
    site_name="base"
):
    """
    Lấy vận tốc góc của base site biểu diễn trong World frame.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data hiện tại.

    site_name : str, optional
        Site đại diện cho base frame.
        Mặc định: "base".

    Returns
    -------
    W_angular_velocity_base : np.ndarray, shape (3,)
        Angular velocity của base biểu diễn trong World frame:

        [wx, wy, wz]

        Đơn vị: [rad/s]
    """

    # ---------------------------------------------------------
    # 1. Tìm site ID
    # ---------------------------------------------------------
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    if site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{site_name}' trong MuJoCo model."
        )

    # ---------------------------------------------------------
    # 2. Spatial velocity của site
    #
    # Thứ tự:
    # [wx, wy, wz, vx, vy, vz]
    #
    # flg_local = 0:
    # velocity biểu diễn theo World frame
    # ---------------------------------------------------------
    velocity_6D = np.zeros(6)

    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        site_id,
        velocity_6D,
        0
    )

    # ---------------------------------------------------------
    # 3. Angular velocity
    # ---------------------------------------------------------
    W_angular_velocity_base = velocity_6D[0:3].copy()

    return np.asarray(
        W_angular_velocity_base,
        dtype=float
    )