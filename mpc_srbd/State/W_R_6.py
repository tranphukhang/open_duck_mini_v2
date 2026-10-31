import mujoco
import numpy as np


def get_W_R_6(
    model,
    data,
    left_site_name="L6",
    right_site_name="R6"
):
    """
    Lấy rotation matrix dùng để chuyển vector
    từ World frame về local frame của L6 và R6.

    MuJoCo cung cấp:

        W_R_L6
        W_R_R6

    thông qua:

        data.site_xmat

    với:

        W_v = W_R_L6 @ L6_v

    Do đó, muốn chuyển từ World frame về L6 frame:

        L6_v = L6_R_W @ W_v

    trong đó:

        L6_R_W = W_R_L6.T

    Tương tự:

        R6_R_W = W_R_R6.T

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data.

    left_site_name : str, optional
        Tên site chân trái.
        Mặc định = "L6".

    right_site_name : str, optional
        Tên site chân phải.
        Mặc định = "R6".

    Returns
    -------
    L6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix chuyển vector
        từ World frame về L6 frame.

    R6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix chuyển vector
        từ World frame về R6 frame.
    """

    # ========================================================
    # Get site IDs
    # ========================================================

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


    # ========================================================
    # Validate site IDs
    # ========================================================

    if left_site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{left_site_name}'."
        )

    if right_site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{right_site_name}'."
        )


    # ========================================================
    # MuJoCo site orientation
    #
    # data.site_xmat gives:
    #
    #     W_R_L6
    #     W_R_R6
    #
    # tức orientation của local site frame
    # được biểu diễn trong World frame.
    # ========================================================

    W_R_L6 = np.array(
        data.site_xmat[left_site_id],
        dtype=float
    ).reshape(3, 3)

    W_R_R6 = np.array(
        data.site_xmat[right_site_id],
        dtype=float
    ).reshape(3, 3)


    # ========================================================
    # World -> local rotation
    #
    # Vì rotation matrix là trực giao:
    #
    #     R^(-1) = R^T
    #
    # nên:
    #
    #     L6_R_W = W_R_L6.T
    #
    #     R6_R_W = W_R_R6.T
    # ========================================================

    L6_R_W = W_R_L6.T.copy()

    R6_R_W = W_R_R6.T.copy()


    return (
        L6_R_W,
        R6_R_W
    )