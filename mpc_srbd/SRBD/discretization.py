import numpy as np


# ============================================================
# DISCRETIZE ONE CONTINUOUS MODEL
# ============================================================

def discretize_model(
    Ac,
    Bc,
    Ts
):
    """
    Rời rạc hóa một mô hình SRBD liên tục bằng Forward Euler.

    Mô hình liên tục:

        x_dot = Ac @ x + Bc @ u

    Mô hình rời rạc:

        x[k+1] = Ad @ x[k] + Bd @ u[k]

    với:

        Ad = I + Ac * Ts

        Bd = Bc * Ts


    Parameters
    ----------
    Ac : np.ndarray, shape (nx, nx)

        Ma trận trạng thái liên tục.


    Bc : np.ndarray, shape (nx, nu)

        Ma trận đầu vào liên tục.


    Ts : float

        Chu kỳ rời rạc hóa [s].


    Returns
    -------
    Ad : np.ndarray, shape (nx, nx)

        Ma trận trạng thái rời rạc.


    Bd : np.ndarray, shape (nx, nu)

        Ma trận đầu vào rời rạc.
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    Ac = np.asarray(
        Ac,
        dtype=float
    )


    Bc = np.asarray(
        Bc,
        dtype=float
    )


    Ts = float(
        Ts
    )


    # ========================================================
    # 2. VALIDATION
    # ========================================================

    if Ac.ndim != 2:

        raise ValueError(
            "Ac phải là ma trận 2D."
        )


    if Bc.ndim != 2:

        raise ValueError(
            "Bc phải là ma trận 2D."
        )


    nx = (
        Ac.shape[0]
    )


    if Ac.shape != (
        nx,
        nx
    ):

        raise ValueError(
            "Ac phải là ma trận vuông."
        )


    if Bc.shape[0] != nx:

        raise ValueError(
            "Số hàng của Bc phải bằng "
            "số state của Ac."
        )


    if Ts <= 0.0:

        raise ValueError(
            "Ts phải lớn hơn 0."
        )


    if not np.isfinite(
        Ts
    ):

        raise ValueError(
            "Ts không finite."
        )


    if not np.all(
        np.isfinite(
            Ac
        )
    ):

        raise ValueError(
            "Ac chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bc
        )
    ):

        raise ValueError(
            "Bc chứa NaN hoặc Inf."
        )


    # ========================================================
    # 3. FORWARD EULER DISCRETIZATION
    #
    #     Ad = I + Ac * Ts
    #
    #     Bd = Bc * Ts
    # ========================================================

    Ad = (
        np.eye(
            nx,
            dtype=float
        )
        +
        Ac
        *
        Ts
    )


    Bd = (
        Bc
        *
        Ts
    )


    # ========================================================
    # 4. FINAL VALIDATION
    # ========================================================

    if Ad.shape != (
        nx,
        nx
    ):

        raise RuntimeError(
            "Ad sai dimension."
        )


    if Bd.shape != Bc.shape:

        raise RuntimeError(
            "Bd sai dimension."
        )


    if not np.all(
        np.isfinite(
            Ad
        )
    ):

        raise RuntimeError(
            "Ad chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bd
        )
    ):

        raise RuntimeError(
            "Bd chứa NaN hoặc Inf."
        )


    return (
        Ad,
        Bd
    )


# ============================================================
# DISCRETIZE CONTINUOUS MODEL SEQUENCE
# ============================================================

def discretize_model_sequence(
    Ac_sequence,
    Bc_sequence,
    Ts
):
    """
    Rời rạc hóa chuỗi mô hình SRBD liên tục trên
    toàn prediction horizon.

    Tại prediction step j:

        x_dot_j
            =
        Ac_j @ x_j
            +
        Bc_j @ u_j


    Sau Forward Euler:

        x[j+1]
            =
        Ad_j @ x[j]
            +
        Bd_j @ u[j]


    với:

        Ad_j
            =
        I + Ac_j * Ts


        Bd_j
            =
        Bc_j * Ts


    Parameters
    ----------
    Ac_sequence : np.ndarray, shape (N, nx, nx)

        Chuỗi continuous state matrices:

            Ac_sequence[j] = Ac_j


    Bc_sequence : np.ndarray, shape (N, nx, nu)

        Chuỗi continuous input matrices:

            Bc_sequence[j] = Bc_j


    Ts : float

        Prediction-model discretization step [s].


    Returns
    -------
    Ad_sequence : np.ndarray, shape (N, nx, nx)

        Chuỗi discrete state matrices:

            Ad_sequence[j] = Ad_j


    Bd_sequence : np.ndarray, shape (N, nx, nu)

        Chuỗi discrete input matrices:

            Bd_sequence[j] = Bd_j
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    Ac_sequence = np.asarray(
        Ac_sequence,
        dtype=float
    )


    Bc_sequence = np.asarray(
        Bc_sequence,
        dtype=float
    )


    Ts = float(
        Ts
    )


    # ========================================================
    # 2. BASIC VALIDATION
    # ========================================================

    if Ac_sequence.ndim != 3:

        raise ValueError(
            "Ac_sequence phải có shape "
            "(N, nx, nx)."
        )


    if Bc_sequence.ndim != 3:

        raise ValueError(
            "Bc_sequence phải có shape "
            "(N, nx, nu)."
        )


    N = (
        Ac_sequence.shape[0]
    )


    if N <= 0:

        raise ValueError(
            "Prediction horizon N phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # Both sequences must have the same horizon
    # --------------------------------------------------------

    if Bc_sequence.shape[0] != N:

        raise ValueError(
            "Ac_sequence và Bc_sequence phải "
            "có cùng prediction horizon."
        )


    # ========================================================
    # 3. STATE / INPUT DIMENSIONS
    # ========================================================

    nx = (
        Ac_sequence.shape[1]
    )


    if Ac_sequence.shape[2] != nx:

        raise ValueError(
            "Mỗi Ac_j phải là ma trận vuông."
        )


    if Bc_sequence.shape[1] != nx:

        raise ValueError(
            "Số hàng của mỗi Bc_j phải bằng nx."
        )


    nu = (
        Bc_sequence.shape[2]
    )


    if nu <= 0:

        raise ValueError(
            "Input dimension nu phải lớn hơn 0."
        )


    # ========================================================
    # 4. FINITE VALIDATION
    # ========================================================

    if Ts <= 0.0:

        raise ValueError(
            "Ts phải lớn hơn 0."
        )


    if not np.isfinite(
        Ts
    ):

        raise ValueError(
            "Ts không finite."
        )


    if not np.all(
        np.isfinite(
            Ac_sequence
        )
    ):

        raise ValueError(
            "Ac_sequence chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bc_sequence
        )
    ):

        raise ValueError(
            "Bc_sequence chứa NaN hoặc Inf."
        )


    # ========================================================
    # 5. ALLOCATE DISCRETE MODEL SEQUENCES
    # ========================================================

    Ad_sequence = np.zeros(
        (
            N,
            nx,
            nx
        ),
        dtype=float
    )


    Bd_sequence = np.zeros(
        (
            N,
            nx,
            nu
        ),
        dtype=float
    )


    # ========================================================
    # 6. DISCRETIZE EACH PREDICTION STEP
    # ========================================================

    for j in range(
        N
    ):

        Ac_j = (
            Ac_sequence[
                j,
                :,
                :
            ]
        )


        Bc_j = (
            Bc_sequence[
                j,
                :,
                :
            ]
        )


        (
            Ad_j,
            Bd_j

        ) = discretize_model(

            Ac=Ac_j,

            Bc=Bc_j,

            Ts=Ts
        )


        Ad_sequence[
            j,
            :,
            :
        ] = Ad_j


        Bd_sequence[
            j,
            :,
            :
        ] = Bd_j


    # ========================================================
    # 7. FINAL VALIDATION
    # ========================================================

    if Ad_sequence.shape != (
        N,
        nx,
        nx
    ):

        raise RuntimeError(
            "Ad_sequence sai dimension."
        )


    if Bd_sequence.shape != (
        N,
        nx,
        nu
    ):

        raise RuntimeError(
            "Bd_sequence sai dimension."
        )


    if not np.all(
        np.isfinite(
            Ad_sequence
        )
    ):

        raise RuntimeError(
            "Ad_sequence chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bd_sequence
        )
    ):

        raise RuntimeError(
            "Bd_sequence chứa NaN hoặc Inf."
        )


    return (
        Ad_sequence,
        Bd_sequence
    )