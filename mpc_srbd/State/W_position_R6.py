import mujoco
import numpy as np


# ============================================================
# WORLD POSITION OF RIGHT FOOT SITE R6
# ============================================================

def get_W_position_R6(
    model,
    data,
    site_name="R6"
):
    """
    Return the current position of the physical right-foot site
    R6 expressed in the World frame.

    Parameters
    ----------
    model : mujoco.MjModel

        MuJoCo model.


    data : mujoco.MjData

        MuJoCo simulation data.


    site_name : str, optional

        Name of the physical right-foot site.

        Default:

            "R6"


    Returns
    -------
    W_p_R6 : np.ndarray, shape (3,)

        Current position of site R6 expressed in the
        World frame:

            {}^W p_R6

        Unit:

            [m]
    """


    # ========================================================
    # 1. GET SITE ID
    # ========================================================

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )


    if site_id < 0:

        raise ValueError(
            f"Không tìm thấy site '{site_name}'."
        )


    # ========================================================
    # 2. GET WORLD POSITION
    # ========================================================

    W_p_R6 = (
        data.site_xpos[
            site_id
        ].copy()
    )


    # ========================================================
    # 3. VALIDATION
    # ========================================================

    if W_p_R6.shape != (
        3,
    ):

        raise RuntimeError(
            "W_p_R6 phải có shape (3,), "
            f"nhưng nhận được {W_p_R6.shape}."
        )


    if not np.all(
        np.isfinite(
            W_p_R6
        )
    ):

        raise RuntimeError(
            "W_p_R6 chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. RETURN
    # ========================================================

    return W_p_R6