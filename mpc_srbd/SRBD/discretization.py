import numpy as np


def discretize_model(
    Ac,
    Bc,
    Ts
):
    """
    Rời rạc hóa mô hình SRBD liên tục bằng Forward Euler.

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
        Chu kỳ lấy mẫu [s].

    Returns
    -------
    Ad : np.ndarray, shape (nx, nx)
        Ma trận trạng thái rời rạc.

    Bd : np.ndarray, shape (nx, nu)
        Ma trận đầu vào rời rạc.
    """

    # ========================================================
    # Input conversion
    # ========================================================

    Ac = np.asarray(
        Ac,
        dtype=float
    )

    Bc = np.asarray(
        Bc,
        dtype=float
    )

    Ts = float(Ts)


    # ========================================================
    # Validation
    # ========================================================

    if Ac.ndim != 2:
        raise ValueError(
            "Ac phải là ma trận 2D."
        )

    if Bc.ndim != 2:
        raise ValueError(
            "Bc phải là ma trận 2D."
        )

    nx = Ac.shape[0]

    if Ac.shape[1] != nx:
        raise ValueError(
            "Ac phải là ma trận vuông."
        )

    if Bc.shape[0] != nx:
        raise ValueError(
            "Số hàng của Bc phải bằng số state của Ac."
        )

    if Ts <= 0.0:
        raise ValueError(
            "Ts phải lớn hơn 0."
        )


    # ========================================================
    # Forward Euler discretization
    # ========================================================

    Ad = (
        np.eye(nx)
        + Ac * Ts
    )

    Bd = (
        Bc * Ts
    )


    return Ad, Bd