import mujoco
import numpy as np


def get_W_position_L6_ref(
    model,
    data,
    site_name="L6_ref"
):
    """
    Lấy vị trí của site L6_ref biểu diễn trong World frame.

    Returns
    -------
    W_position_L6_ref : np.ndarray, shape (3,)
        [x, y, z] của L6_ref trong World frame [m].
    """

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    if site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{site_name}' trong MuJoCo model."
        )

    W_position_L6_ref = data.site_xpos[site_id].copy()

    return np.asarray(
        W_position_L6_ref,
        dtype=float
    )