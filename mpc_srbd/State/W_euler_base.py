import mujoco
import numpy as np

from scipy.spatial.transform import Rotation


def get_W_euler_base(
    model,
    data,
    site_name="base"
):
    """
    Lấy Euler angles của base site so với World frame.

    Euler convention:
        [roll, pitch, yaw]
        = [phi, theta, psi]

    Đơn vị:
        rad

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
    W_euler_base : np.ndarray, shape (3,)
        Euler angles của base so với World frame:

        [roll, pitch, yaw]

        Đơn vị: [rad]
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
    # 2. Rotation matrix:
    #
    #       ^W R_base
    #
    # data.site_xmat lưu orientation của site
    # trong World frame.
    # ---------------------------------------------------------
    W_R_base = (
        data.site_xmat[site_id]
        .reshape(3, 3)
        .copy()
    )

    # ---------------------------------------------------------
    # 3. Rotation matrix -> Euler angles
    #
    # Convention:
    # roll  quanh x
    # pitch quanh y
    # yaw   quanh z
    #
    # Output: rad
    # ---------------------------------------------------------
    W_euler_base = (
        Rotation
        .from_matrix(W_R_base)
        .as_euler("xyz", degrees=False)
    )

    return np.asarray(
        W_euler_base,
        dtype=float
    )