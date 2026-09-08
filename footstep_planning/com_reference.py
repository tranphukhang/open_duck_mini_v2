import numpy as np


# ============================================================
# COM REFERENCE
# ============================================================

def compute_com_reference(
    left_foot_position,
    right_foot_position,
    left_foot_velocity=None,
    right_foot_velocity=None,
    com_y=0.0,
    com_height=0.205,
):
    """
    Generate the simple kinematic CoM reference.

    The reference follows the kinematic walking prototype:

        x_com_ref =
            0.5 * (x_left_ref + x_right_ref)

        y_com_ref =
            constant

        z_com_ref =
            constant

    The CoM velocity reference is:

        vx_com_ref =
            0.5 * (vx_left_ref + vx_right_ref)

        vy_com_ref = 0
        vz_com_ref = 0

    Parameters
    ----------
    left_foot_position : array-like, shape (3,)
        Left foot reference position in world frame.

    right_foot_position : array-like, shape (3,)
        Right foot reference position in world frame.

    left_foot_velocity : array-like, shape (3,), optional
        Left foot reference linear velocity.

        If None:
            [0, 0, 0] is used.

    right_foot_velocity : array-like, shape (3,), optional
        Right foot reference linear velocity.

        If None:
            [0, 0, 0] is used.

    com_y : float
        Constant lateral CoM reference.

        For straight walking this will normally be
        the settled initial CoM y-coordinate.

    com_height : float
        Constant CoM height.

    Returns
    -------
    p_com_ref : ndarray, shape (3,)
        Desired CoM position.

    v_com_ref : ndarray, shape (3,)
        Desired CoM linear velocity.
    """

    # ========================================================
    # INPUT POSITIONS
    # ========================================================

    p_left = np.asarray(
        left_foot_position,
        dtype=float,
    ).copy()

    p_right = np.asarray(
        right_foot_position,
        dtype=float,
    ).copy()

    if p_left.shape != (3,):
        raise ValueError(
            "left_foot_position must have shape (3,)."
        )

    if p_right.shape != (3,):
        raise ValueError(
            "right_foot_position must have shape (3,)."
        )

    # ========================================================
    # INPUT VELOCITIES
    # ========================================================

    if left_foot_velocity is None:

        v_left = np.zeros(
            3,
            dtype=float,
        )

    else:

        v_left = np.asarray(
            left_foot_velocity,
            dtype=float,
        ).copy()

        if v_left.shape != (3,):
            raise ValueError(
                "left_foot_velocity must have shape (3,)."
            )

    if right_foot_velocity is None:

        v_right = np.zeros(
            3,
            dtype=float,
        )

    else:

        v_right = np.asarray(
            right_foot_velocity,
            dtype=float,
        ).copy()

        if v_right.shape != (3,):
            raise ValueError(
                "right_foot_velocity must have shape (3,)."
            )

    # ========================================================
    # COM POSITION REFERENCE
    # ========================================================

    x_com_ref = 0.5 * (
        p_left[0]
        +
        p_right[0]
    )

    p_com_ref = np.array(
        [
            x_com_ref,
            float(com_y),
            float(com_height),
        ],
        dtype=float,
    )

    # ========================================================
    # COM VELOCITY REFERENCE
    # ========================================================

    vx_com_ref = 0.5 * (
        v_left[0]
        +
        v_right[0]
    )

    v_com_ref = np.array(
        [
            vx_com_ref,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    return (
        p_com_ref,
        v_com_ref,
    )