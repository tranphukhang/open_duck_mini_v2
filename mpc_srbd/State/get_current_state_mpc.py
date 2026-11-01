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

from W_R_6_IK import (
    get_W_R_6_IK
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
    #
    # x =
    # [
    #     p_CoM
    #     euler_base
    #     v_CoM
    #     omega_base
    #     1
    # ]
    # --------------------------------------------------------

    x: np.ndarray


    # --------------------------------------------------------
    # SRBD model quantities
    # --------------------------------------------------------

    Ew_inverse: np.ndarray

    W_r_L: np.ndarray

    W_r_R: np.ndarray


    # --------------------------------------------------------
    # ORIGINAL L6 / R6 ORIENTATION MATRICES
    #
    # These matrices are preserved for backward compatibility.
    #
    # They represent:
    #
    #     World -> L6
    #
    #     World -> R6
    #
    # IMPORTANT:
    #
    # L6 / R6 are the original physical foot/contact-point
    # sites and their local frames are NOT World-aligned at
    # the home pose.
    # --------------------------------------------------------

    L6_R_W: np.ndarray

    R6_R_W: np.ndarray


    # --------------------------------------------------------
    # L6_IK / R6_IK CONTACT-ORIENTATION MATRICES
    #
    # These matrices represent:
    #
    #     World -> L6_IK
    #
    #     World -> R6_IK
    #
    # At the home configuration:
    #
    #     L6_IK_R_W ~= I
    #
    #     R6_IK_R_W ~= I
    #
    # Therefore these frames are suitable for extracting:
    #
    #     Fz_foot
    #
    #     My_foot
    #
    # for the MPC moment / CoP constraints.
    # --------------------------------------------------------

    L6_IK_R_W: np.ndarray

    R6_IK_R_W: np.ndarray


# ============================================================
# MATRIX VALIDATION
# ============================================================

def _validate_matrix(
    value,
    expected_shape,
    name
):
    """
    Validate shape and finite values of an array.
    """

    value = np.asarray(
        value,
        dtype=float
    )


    if value.shape != expected_shape:

        raise RuntimeError(
            f"{name} phải có kích thước "
            f"{expected_shape}, "
            f"nhưng nhận được {value.shape}."
        )


    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise RuntimeError(
            f"{name} chứa NaN hoặc Inf."
        )


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

    Processing sequence
    -------------------

    1. Update MuJoCo derived quantities.

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

    4. Update original physical-foot orientation matrices:

           L6_R_W

           R6_R_W

    5. Update the new IK/contact orientation matrices:

           L6_IK_R_W

           R6_IK_R_W


    Important
    ---------

    The two orientation-frame sets have different purposes.

    Original physical sites:

        L6
        R6

    are preserved for backward compatibility and for any
    physical-foot quantities that still explicitly depend on
    those original sites.


    New sites:

        L6_IK
        R6_IK

    have their local axes aligned with the desired foot
    contact frame.

    These new matrices are intended for MPC foot-frame
    quantities such as:

        Fz_foot

        My_foot

    used by the moment / CoP constraints.


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

            current.L6_IK_R_W

            current.R6_IK_R_W
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

    W_position_CoM = (
        get_W_position_CoM(
            model,
            data
        )
    )


    W_euler_base = (
        get_W_euler_base(
            model,
            data
        )
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


    _validate_matrix(
        x_current,
        (
            13,
        ),
        "SRBD state vector"
    )


    # ========================================================
    # 4. UPDATE EULER KINEMATIC MATRIX
    # ========================================================

    Ew_inverse = (
        get_Ew_inverse(
            W_euler_base
        )
    )


    # ========================================================
    # 5. UPDATE MOMENT ARMS
    #
    # Physical contact-point positions:
    #
    #     W_r_L
    #
    #       =
    #
    #     W_p_L6 - W_p_CoM
    #
    #
    #     W_r_R
    #
    #       =
    #
    #     W_p_R6 - W_p_CoM
    #
    # IMPORTANT:
    #
    # Moment arms continue to use the physical L6/R6 sites.
    # ========================================================

    (
        W_r_L,
        W_r_R

    ) = get_W_moment_arm(
        model,
        data
    )


    # ========================================================
    # 6. ORIGINAL L6 / R6 ORIENTATION MATRICES
    #
    # Preserve the original quantities:
    #
    #     L6_R_W
    #
    #     R6_R_W
    #
    # No existing behavior is changed here.
    # ========================================================

    (
        L6_R_W,
        R6_R_W

    ) = get_W_R_6(
        model,
        data
    )


    # ========================================================
    # 7. NEW L6_IK / R6_IK CONTACT ORIENTATION MATRICES
    #
    # These are the matrices that will later be supplied to:
    #
    #     moment_constraint.py
    #
    # for:
    #
    #     Fz_foot
    #
    #     My_foot
    #
    # extraction.
    # ========================================================

    (
        L6_IK_R_W,
        R6_IK_R_W

    ) = get_W_R_6_IK(
        model,
        data
    )


    # ========================================================
    # 8. VALIDATION
    # ========================================================

    _validate_matrix(
        Ew_inverse,
        (
            3,
            3
        ),
        "Ew_inverse"
    )


    _validate_matrix(
        W_r_L,
        (
            3,
        ),
        "W_r_L"
    )


    _validate_matrix(
        W_r_R,
        (
            3,
        ),
        "W_r_R"
    )


    # --------------------------------------------------------
    # Original L6 / R6 matrices
    # --------------------------------------------------------

    _validate_matrix(
        L6_R_W,
        (
            3,
            3
        ),
        "L6_R_W"
    )


    _validate_matrix(
        R6_R_W,
        (
            3,
            3
        ),
        "R6_R_W"
    )


    # --------------------------------------------------------
    # New L6_IK / R6_IK matrices
    # --------------------------------------------------------

    _validate_matrix(
        L6_IK_R_W,
        (
            3,
            3
        ),
        "L6_IK_R_W"
    )


    _validate_matrix(
        R6_IK_R_W,
        (
            3,
            3
        ),
        "R6_IK_R_W"
    )


    # ========================================================
    # 9. BUILD CURRENT MPC SNAPSHOT
    # ========================================================

    current = CurrentMPCState(

        x=(
            x_current
        ),

        Ew_inverse=(
            Ew_inverse
        ),

        W_r_L=(
            W_r_L
        ),

        W_r_R=(
            W_r_R
        ),

        L6_R_W=(
            L6_R_W
        ),

        R6_R_W=(
            R6_R_W
        ),

        L6_IK_R_W=(
            L6_IK_R_W
        ),

        R6_IK_R_W=(
            R6_IK_R_W
        )
    )


    # ========================================================
    # 10. RETURN
    # ========================================================

    return current