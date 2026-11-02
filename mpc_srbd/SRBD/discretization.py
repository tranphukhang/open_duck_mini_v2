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
