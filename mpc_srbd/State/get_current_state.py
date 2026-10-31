import numpy as np
import mujoco

from W_position_CoM import (
    get_W_position_CoM
)

from W_linear_velocity_CoM import (
    get_W_linear_velocity_CoM
)

from W_angular_velocity_base import (
    get_W_angular_velocity_base
)

from W_euler_base import (
    get_W_euler_base
)


# ============================================================
# CURRENT SRBD STATE
# ============================================================

def get_current_state(
    model,
    data
):
    """
    Read current SRBD state from MuJoCo.

    State:

        x =
        [
            p_CoM,
            euler_base,
            v_CoM,
            omega_base,
            1
        ]

    State dimension:

        nx = 13


    Returns
    -------
    x_current : ndarray, shape (13,)

        Current SRBD state:

            [
                x_CoM,
                y_CoM,
                z_CoM,

                roll,
                pitch,
                yaw,

                vx_CoM,
                vy_CoM,
                vz_CoM,

                wx_base,
                wy_base,
                wz_base,

                1
            ]
    """

    # ========================================================
    # 1. SYNCHRONIZE MUJOCO DERIVED QUANTITIES
    #
    # mj_forward() updates kinematics and dynamics at the
    # current qpos and qvel.
    #
    # It does NOT advance simulation time.
    # ========================================================

    mujoco.mj_forward(
        model,
        data
    )


    # ========================================================
    # 2. UPDATE SUBTREE CoM VELOCITY
    #
    # Required before reading:
    #
    #     data.subtree_linvel
    # ========================================================

    mujoco.mj_subtreeVel(
        model,
        data
    )


    # ========================================================
    # 3. READ CoM POSITION
    # ========================================================

    W_position_CoM = (
        get_W_position_CoM(
            model,
            data
        )
    )


    # ========================================================
    # 4. READ BASE ORIENTATION
    #
    # Euler convention:
    #
    #     [roll, pitch, yaw]
    #
    # expressed relative to World.
    # ========================================================

    W_euler_base = (
        get_W_euler_base(
            model,
            data
        )
    )


    # ========================================================
    # 5. READ CoM LINEAR VELOCITY
    #
    # Expressed in World frame.
    # ========================================================

    W_linear_velocity_CoM = (
        get_W_linear_velocity_CoM(
            model,
            data
        )
    )


    # ========================================================
    # 6. READ BASE ANGULAR VELOCITY
    #
    # Expressed in World frame.
    # ========================================================

    W_angular_velocity_base = (
        get_W_angular_velocity_base(
            model,
            data
        )
    )


    # ========================================================
    # 7. ASSEMBLE AUGMENTED SRBD STATE
    #
    # x =
    #
    # [
    #     p_CoM
    #     euler_base
    #     v_CoM
    #     omega_base
    #     1
    # ]
    # ========================================================

    x_current = np.concatenate([
        W_position_CoM,
        W_euler_base,
        W_linear_velocity_CoM,
        W_angular_velocity_base,
        np.array(
            [1.0],
            dtype=float
        )
    ])


    # ========================================================
    # 8. VALIDATION
    # ========================================================

    if x_current.shape != (13,):

        raise RuntimeError(
            "x_current phải có shape (13,), "
            f"nhưng nhận được {x_current.shape}."
        )


    if not np.all(
        np.isfinite(
            x_current
        )
    ):

        raise RuntimeError(
            "x_current chứa NaN hoặc Inf."
        )


    return x_current