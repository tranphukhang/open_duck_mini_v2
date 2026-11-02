import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

SRBD_STATE_DIM = 13


# ============================================================
# VALIDATION
# ============================================================

def _validate_reference_matrix(
    value,
    name,
    horizon=None
):
    """
    Validate reference matrix with shape:

        (N, 3)
    """

    value = np.asarray(
        value,
        dtype=float
    )


    if value.ndim != 2:

        raise ValueError(
            f"{name} phải là ma trận 2D."
        )


    if value.shape[1] != 3:

        raise ValueError(
            f"{name} phải có shape (N,3), "
            f"nhưng nhận được {value.shape}."
        )


    if value.shape[0] <= 0:

        raise ValueError(
            f"{name} phải chứa ít nhất một mẫu."
        )


    if (
        horizon is not None
        and
        value.shape[0] != horizon
    ):

        raise ValueError(
            f"{name} phải có horizon = {horizon}, "
            f"nhưng nhận được {value.shape[0]}."
        )


    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    return value


def _expand_reference(
    value,
    horizon,
    name,
    default_zero=False
):
    """
    Expand a reference to shape:

        (N, 3)

    Accepted input:

        None

        shape (3,)

        shape (N,3)
    """

    # ========================================================
    # DEFAULT ZERO REFERENCE
    # ========================================================

    if value is None:

        if not default_zero:

            raise ValueError(
                f"{name} không được None."
            )


        return np.zeros(
            (
                horizon,
                3
            ),
            dtype=float
        )


    value = np.asarray(
        value,
        dtype=float
    )


    # ========================================================
    # CONSTANT REFERENCE
    #
    # Example:
    #
    #     euler_ref =
    #
    #     [0, 0, 0]
    #
    # becomes:
    #
    #     [
    #         [0,0,0]
    #         [0,0,0]
    #         ...
    #     ]
    # ========================================================

    if value.shape == (
        3,
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


        return np.tile(
            value.reshape(
                1,
                3
            ),
            (
                horizon,
                1
            )
        )


    # ========================================================
    # TIME-VARYING REFERENCE
    # ========================================================

    if value.shape == (
        horizon,
        3
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


        return value.copy()


    raise ValueError(
        f"{name} phải có shape (3,) "
        f"hoặc ({horizon},3), "
        f"nhưng nhận được {value.shape}."
    )


# ============================================================
# BUILD ONE SRBD REFERENCE STATE
#
# State convention:
#
#     x =
#
#     [
#         W_p_CoM
#         W_euler_CoM
#         W_v_CoM
#         W_w_CoM
#         1
#     ]
#
#     x in R^13
# ============================================================

def build_srbd_reference_state(
    W_p_CoM_ref,
    W_euler_CoM_ref,
    W_v_CoM_ref,
    W_w_CoM_ref
):
    """
    Build one SRBD reference state.

    Returns
    -------
    x_ref : ndarray, shape (13,)
    """

    W_p_CoM_ref = np.asarray(
        W_p_CoM_ref,
        dtype=float
    ).reshape(-1)


    W_euler_CoM_ref = np.asarray(
        W_euler_CoM_ref,
        dtype=float
    ).reshape(-1)


    W_v_CoM_ref = np.asarray(
        W_v_CoM_ref,
        dtype=float
    ).reshape(-1)


    W_w_CoM_ref = np.asarray(
        W_w_CoM_ref,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # VALIDATION
    # ========================================================

    for value, name in (
        (
            W_p_CoM_ref,
            "W_p_CoM_ref"
        ),
        (
            W_euler_CoM_ref,
            "W_euler_CoM_ref"
        ),
        (
            W_v_CoM_ref,
            "W_v_CoM_ref"
        ),
        (
            W_w_CoM_ref,
            "W_w_CoM_ref"
        ),
    ):

        if value.shape != (
            3,
        ):

            raise ValueError(
                f"{name} phải có shape (3,), "
                f"nhưng nhận được {value.shape}."
            )


        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


    # ========================================================
    # BUILD STATE
    # ========================================================

    x_ref = np.concatenate(
        (
            W_p_CoM_ref,

            W_euler_CoM_ref,

            W_v_CoM_ref,

            W_w_CoM_ref,

            np.array(
                [
                    1.0
                ],
                dtype=float
            )
        )
    )


    if x_ref.shape != (
        SRBD_STATE_DIM,
    ):

        raise RuntimeError(
            "Internal error: SRBD reference state "
            "không có dimension 13."
        )


    return x_ref


# ============================================================
# BUILD COMPLETE SRBD REFERENCE
#
# Prediction convention:
#
#     X_ref =
#
#     [
#         x_ref(k+1)
#         x_ref(k+2)
#         ...
#         x_ref(k+N)
#     ]
#
#
# IMPORTANT:
#
#     x_k hiện tại không nằm trong X_ref.
# ============================================================

def build_srbd_reference(
    W_p_CoM_ref,
    W_v_CoM_ref,
    W_euler_CoM_ref=None,
    W_w_CoM_ref=None
):
    """
    Build complete SRBD-MPC reference trajectory.

    Parameters
    ----------
    W_p_CoM_ref : ndarray, shape (N,3)

        Future nominal CoM position reference.


    W_v_CoM_ref : ndarray, shape (N,3)

        Future nominal CoM linear velocity reference.


    W_euler_CoM_ref : None, shape (3,), or shape (N,3)

        Desired body/CoM Euler orientation in World frame.

        Default:

            [0, 0, 0]


    W_w_CoM_ref : None, shape (3,), or shape (N,3)

        Desired angular velocity in World frame.

        Default:

            [0, 0, 0]


    Returns
    -------
    reference : dict

        reference["X_ref_matrix"]

            shape (N,13)


        reference["X_ref"]

            shape (13*N,)


        reference["W_p_CoM_ref"]

            shape (N,3)


        reference["W_euler_CoM_ref"]

            shape (N,3)


        reference["W_v_CoM_ref"]

            shape (N,3)


        reference["W_w_CoM_ref"]

            shape (N,3)


        reference["horizon"]

            int
    """

    # ========================================================
    # 1. CoM POSITION REFERENCE
    # ========================================================

    W_p_CoM_ref = _validate_reference_matrix(

        W_p_CoM_ref,

        "W_p_CoM_ref"
    )


    horizon = int(
        W_p_CoM_ref.shape[
            0
        ]
    )


    # ========================================================
    # 2. CoM LINEAR VELOCITY REFERENCE
    # ========================================================

    W_v_CoM_ref = _validate_reference_matrix(

        W_v_CoM_ref,

        "W_v_CoM_ref",

        horizon=horizon
    )


    # ========================================================
    # 3. ATTITUDE REFERENCE
    #
    # Flat-ground default:
    #
    #     roll  = 0
    #     pitch = 0
    #     yaw   = 0
    #
    # Later:
    #
    #     yaw reference can be supplied externally.
    # ========================================================

    W_euler_CoM_ref = _expand_reference(

        value=W_euler_CoM_ref,

        horizon=horizon,

        name="W_euler_CoM_ref",

        default_zero=True
    )


    # ========================================================
    # 4. ANGULAR VELOCITY REFERENCE
    #
    # Flat-ground default:
    #
    #     w_ref = [0,0,0]
    # ========================================================

    W_w_CoM_ref = _expand_reference(

        value=W_w_CoM_ref,

        horizon=horizon,

        name="W_w_CoM_ref",

        default_zero=True
    )


    # ========================================================
    # 5. BUILD X_ref MATRIX
    #
    # Each row:
    #
    #     [
    #         p_CoM
    #         euler_CoM
    #         v_CoM
    #         w_CoM
    #         1
    #     ]
    #
    # shape:
    #
    #     (N,13)
    # ========================================================

    X_ref_matrix = np.zeros(
        (
            horizon,
            SRBD_STATE_DIM
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # p_CoM
    #
    # state index:
    #
    #     0:3
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        0:3
    ] = W_p_CoM_ref


    # --------------------------------------------------------
    # euler_CoM
    #
    # state index:
    #
    #     3:6
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        3:6
    ] = W_euler_CoM_ref


    # --------------------------------------------------------
    # v_CoM
    #
    # state index:
    #
    #     6:9
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        6:9
    ] = W_v_CoM_ref


    # --------------------------------------------------------
    # w_CoM
    #
    # state index:
    #
    #     9:12
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        9:12
    ] = W_w_CoM_ref


    # --------------------------------------------------------
    # affine state
    #
    # state index:
    #
    #     12
    # --------------------------------------------------------

    X_ref_matrix[
        :,
        12
    ] = 1.0


    # ========================================================
    # 6. STACK FOR MPC
    #
    # C-order reshape gives:
    #
    #     [
    #         x_ref(k+1)
    #         x_ref(k+2)
    #         ...
    #         x_ref(k+N)
    #     ]
    #
    # shape:
    #
    #     (13*N,)
    # ========================================================

    X_ref = X_ref_matrix.reshape(
        -1
    )


    # ========================================================
    # 7. FINAL VALIDATION
    # ========================================================

    if X_ref.shape != (
        horizon
        *
        SRBD_STATE_DIM,
    ):

        raise RuntimeError(
            "X_ref có dimension không đúng."
        )


    if not np.all(
        np.isfinite(
            X_ref
        )
    ):

        raise RuntimeError(
            "X_ref chứa NaN hoặc Inf."
        )


    # ========================================================
    # 8. RETURN
    # ========================================================

    return {
        "X_ref":
            X_ref,

        "X_ref_matrix":
            X_ref_matrix,

        "W_p_CoM_ref":
            W_p_CoM_ref.copy(),

        "W_euler_CoM_ref":
            W_euler_CoM_ref.copy(),

        "W_v_CoM_ref":
            W_v_CoM_ref.copy(),

        "W_w_CoM_ref":
            W_w_CoM_ref.copy(),

        "horizon":
            horizon,

        "state_dimension":
            SRBD_STATE_DIM,
    }