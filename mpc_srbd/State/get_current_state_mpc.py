from dataclasses import dataclass

import numpy as np
import mujoco


# ============================================================
# STATE COMPONENTS
# ============================================================

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
    get_W_euler_base,
    get_Ew_inverse
)

from W_moment_arm import (
    get_W_moment_arm
)

from W_R_6 import (
    get_W_R_6
)


# ============================================================
# CURRENT MPC STATE
# ============================================================

@dataclass
class CurrentMPCState:
    """
    Snapshot of all state-dependent quantities required
    at the beginning of one MPC control cycle.
    """

    # --------------------------------------------------------
    # SRBD state vector
    # --------------------------------------------------------

    x: np.ndarray


    # --------------------------------------------------------
    # SRBD model quantities
    # --------------------------------------------------------

    Ew_inverse: np.ndarray

    W_r_L: np.ndarray

    W_r_R: np.ndarray


    # --------------------------------------------------------
    # QP constraint quantities
    # --------------------------------------------------------

    L6_R_W: np.ndarray

    R6_R_W: np.ndarray


# ============================================================
# GET CURRENT MPC STATE
# ============================================================

def get_current_state_mpc(
    model,
    data
):
    """
    Read and update all quantities required at the beginning
    of the current MPC control cycle.

    This function performs:

        1. Update MuJoCo derived quantities

        2. Read current SRBD state:

               x_k =
               [
                   p_CoM
                   euler_base
                   v_CoM
                   omega_base
                   1
               ]

        3. Update configuration-dependent SRBD quantities:

               Ew_inverse

               W_r_L

               W_r_R

        4. Update foot orientation matrices used by
           the QP constraints:

               L6_R_W

               R6_R_W


    Parameters
    ----------
    model : mujoco.MjModel

        MuJoCo model.


    data : mujoco.MjData

        Current MuJoCo simulation data.


    Returns
    -------
    current : CurrentMPCState

        Snapshot containing:

            current.x

            current.Ew_inverse

            current.W_r_L

            current.W_r_R

            current.L6_R_W

            current.R6_R_W
    """


    # ========================================================
    # 1. UPDATE MUJOCO DERIVED QUANTITIES
    # ========================================================

    mujoco.mj_forward(
        model,
        data
    )


    mujoco.mj_subtreeVel(
        model,
        data
    )


    # ========================================================
    # 2. SRBD STATE COMPONENTS
    # ========================================================

    W_position_CoM = get_W_position_CoM(
        model,
        data
    )


    W_euler_base = get_W_euler_base(
        model,
        data
    )


    W_linear_velocity_CoM = (
        get_W_linear_velocity_CoM(
            model,
            data
        )
    )


    W_angular_velocity_base = (
        get_W_angular_velocity_base(
            model,
            data
        )
    )


    # ========================================================
    # 3. BUILD CURRENT SRBD STATE VECTOR
    # ========================================================

    x_current = np.concatenate(
        [
            W_position_CoM,

            W_euler_base,

            W_linear_velocity_CoM,

            W_angular_velocity_base,

            np.array(
                [
                    1.0
                ],
                dtype=float
            )
        ]
    )


    # --------------------------------------------------------
    # State validation
    # --------------------------------------------------------

    if x_current.shape != (
        13,
    ):

        raise RuntimeError(
            "SRBD state vector phải có kích thước (13,), "
            f"nhưng nhận được {x_current.shape}."
        )


    if not np.all(
        np.isfinite(
            x_current
        )
    ):

        raise RuntimeError(
            "SRBD state vector chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. UPDATE EULER KINEMATIC MATRIX
    # ========================================================

    Ew_inverse = get_Ew_inverse(
        W_euler_base
    )


    # ========================================================
    # 5. UPDATE MOMENT ARMS
    # ========================================================

    (
        W_r_L,
        W_r_R
    ) = get_W_moment_arm(
        model,
        data
    )


    # ========================================================
    # 6. UPDATE FOOT ORIENTATION MATRICES
    # ========================================================

    (
        L6_R_W,
        R6_R_W
    ) = get_W_R_6(
        model,
        data
    )


    # ========================================================
    # 7. VALIDATION
    # ========================================================

    if Ew_inverse.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "Ew_inverse phải có kích thước (3, 3)."
        )


    if W_r_L.shape != (
        3,
    ):

        raise RuntimeError(
            "W_r_L phải có kích thước (3,)."
        )


    if W_r_R.shape != (
        3,
    ):

        raise RuntimeError(
            "W_r_R phải có kích thước (3,)."
        )


    if L6_R_W.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "L6_R_W phải có kích thước (3, 3)."
        )


    if R6_R_W.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "R6_R_W phải có kích thước (3, 3)."
        )


    for name, value in (
        (
            "Ew_inverse",
            Ew_inverse
        ),
        (
            "W_r_L",
            W_r_L
        ),
        (
            "W_r_R",
            W_r_R
        ),
        (
            "L6_R_W",
            L6_R_W
        ),
        (
            "R6_R_W",
            R6_R_W
        )
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise RuntimeError(
                f"{name} chứa NaN hoặc Inf."
            )


    # ========================================================
    # 8. BUILD CURRENT MPC SNAPSHOT
    # ========================================================

    current = CurrentMPCState(

        x=x_current,

        Ew_inverse=Ew_inverse,

        W_r_L=W_r_L,

        W_r_R=W_r_R,

        L6_R_W=L6_R_W,

        R6_R_W=R6_R_W
    )


    # ========================================================
    # 9. RETURN
    # ========================================================

    return current