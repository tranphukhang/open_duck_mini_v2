import mujoco
import numpy as np


# ============================================================
# WORLD POSITION OF LEFT FOOT SITE L6
# ============================================================

def get_W_position_L6(
    model,
    data,
    site_name="L6"
):
    """
    Return the current position of the physical left-foot site
    L6 expressed in the World frame.

    Parameters
    ----------
    model : mujoco.MjModel

        MuJoCo model.


    data : mujoco.MjData

        MuJoCo simulation data.


    site_name : str, optional

        Name of the physical left-foot site.

        Default:

            "L6"


    Returns
    -------
    W_p_L6 : np.ndarray, shape (3,)

        Current position of site L6 expressed in the
        World frame:

            {}^W p_L6

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
    #
    # MuJoCo:
    #
    #     data.site_xpos
    #
    # stores site positions expressed in the World frame.
    # ========================================================

    W_p_L6 = (
        data.site_xpos[
            site_id
        ].copy()
    )


    # ========================================================
    # 3. VALIDATION
    # ========================================================

    if W_p_L6.shape != (
        3,
    ):

        raise RuntimeError(
            "W_p_L6 phải có shape (3,), "
            f"nhưng nhận được {W_p_L6.shape}."
        )


    if not np.all(
        np.isfinite(
            W_p_L6
        )
    ):

        raise RuntimeError(
            "W_p_L6 chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. RETURN
    # ========================================================

    return W_p_L6