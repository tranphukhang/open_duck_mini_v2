import mujoco
import numpy as np


# ============================================================
# WORLD -> L6_IK / R6_IK ROTATION MATRICES
# ============================================================

def get_W_R_6_IK(
    model,
    data,
    left_site_name="L6_IK",
    right_site_name="R6_IK"
):
    """
    Get rotation matrices used to transform vectors from the
    World frame into the local L6_IK / R6_IK frames.

    These IK sites are also used as the desired contact-frame
    orientation for the MPC moment / CoP constraints.

    MuJoCo convention
    -----------------

    data.site_xmat[site_id] gives the orientation:

        W_R_site

    such that:

        W_v = W_R_site @ site_v

    Therefore, the inverse transformation from World to the
    local site frame is:

        site_v = site_R_W @ W_v

    where:

        site_R_W = W_R_site.T


    For the left foot:

        L6_IK_v
            =
        L6_IK_R_W @ W_v


    For the right foot:

        R6_IK_v
            =
        R6_IK_R_W @ W_v


    Parameters
    ----------
    model : mujoco.MjModel

        MuJoCo model.


    data : mujoco.MjData

        MuJoCo simulation data.


    left_site_name : str, optional

        Left contact-orientation site.

        Default:

            "L6_IK"


    right_site_name : str, optional

        Right contact-orientation site.

        Default:

            "R6_IK"


    Returns
    -------
    L6_IK_R_W : np.ndarray, shape (3, 3)

        Rotation matrix:

            World -> L6_IK


    R6_IK_R_W : np.ndarray, shape (3, 3)

        Rotation matrix:

            World -> R6_IK


    Notes
    -----
    This function does NOT modify MuJoCo state.

    It assumes that the caller has already updated MuJoCo
    derived quantities using:

        mujoco.mj_forward(...)

    when necessary.
    """


    # ========================================================
    # 1. VALIDATE SITE NAMES
    # ========================================================

    if not isinstance(
        left_site_name,
        str
    ):

        raise TypeError(
            "left_site_name phải là str."
        )


    if not isinstance(
        right_site_name,
        str
    ):

        raise TypeError(
            "right_site_name phải là str."
        )


    if left_site_name == "":

        raise ValueError(
            "left_site_name không được rỗng."
        )


    if right_site_name == "":

        raise ValueError(
            "right_site_name không được rỗng."
        )


    # ========================================================
    # 2. GET SITE IDs
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
    # 3. VALIDATE SITE IDs
    # ========================================================

    if left_site_id < 0:

        raise ValueError(
            f"Không tìm thấy site "
            f"'{left_site_name}' trong MuJoCo model."
        )


    if right_site_id < 0:

        raise ValueError(
            f"Không tìm thấy site "
            f"'{right_site_name}' trong MuJoCo model."
        )


    # ========================================================
    # 4. READ SITE ORIENTATIONS
    #
    # MuJoCo gives:
    #
    #     W_R_L6_IK
    #
    #     W_R_R6_IK
    #
    # where the columns of each matrix are the local-site
    # axes expressed in the World frame.
    # ========================================================

    W_R_L6_IK = np.asarray(
        data.site_xmat[
            left_site_id
        ],
        dtype=float
    ).reshape(
        3,
        3
    )


    W_R_R6_IK = np.asarray(
        data.site_xmat[
            right_site_id
        ],
        dtype=float
    ).reshape(
        3,
        3
    )


    # ========================================================
    # 5. VALIDATE WORLD-FRAME SITE ROTATIONS
    # ========================================================

    if not np.all(
        np.isfinite(
            W_R_L6_IK
        )
    ):

        raise RuntimeError(
            "W_R_L6_IK chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            W_R_R6_IK
        )
    ):

        raise RuntimeError(
            "W_R_R6_IK chứa NaN hoặc Inf."
        )


    # ========================================================
    # 6. WORLD -> LOCAL ROTATION
    #
    # Because:
    #
    #     R^-1 = R^T
    #
    # for a rotation matrix:
    #
    #     L6_IK_R_W = W_R_L6_IK.T
    #
    #     R6_IK_R_W = W_R_R6_IK.T
    # ========================================================

    L6_IK_R_W = (
        W_R_L6_IK.T.copy()
    )


    R6_IK_R_W = (
        W_R_R6_IK.T.copy()
    )


    # ========================================================
    # 7. OUTPUT VALIDATION
    # ========================================================

    if L6_IK_R_W.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "L6_IK_R_W phải có shape (3, 3)."
        )


    if R6_IK_R_W.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "R6_IK_R_W phải có shape (3, 3)."
        )


    if not np.all(
        np.isfinite(
            L6_IK_R_W
        )
    ):

        raise RuntimeError(
            "L6_IK_R_W chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            R6_IK_R_W
        )
    ):

        raise RuntimeError(
            "R6_IK_R_W chứa NaN hoặc Inf."
        )


    # ========================================================
    # 8. RETURN
    # ========================================================

    return (
        L6_IK_R_W,
        R6_IK_R_W
    )