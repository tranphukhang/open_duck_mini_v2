import numpy as np


def build_prediction_model(
    Ad,
    Bd,
    N
):
    """
    Xây dựng mô hình dự đoán stacked cho MPC.

    Mô hình rời rạc:

        x[k+1] = Ad @ x[k] + Bd @ u[k]

    Prediction horizon:

        x[k+1|k]
        x[k+2|k]
        ...
        x[k+N|k]

    Dạng stacked:

        X = Phi @ x[k] + Gamma @ U

    với:

        X = [
            x[k+1|k]
            x[k+2|k]
            ...
            x[k+N|k]
        ]

        U = [
            u[k|k]
            u[k+1|k]
            ...
            u[k+N-1|k]
        ]

    Parameters
    ----------
    Ad : np.ndarray, shape (nx, nx)
        Ma trận trạng thái rời rạc.

    Bd : np.ndarray, shape (nx, nu)
        Ma trận đầu vào rời rạc.

    N : int
        Prediction horizon.

    Returns
    -------
    Phi : np.ndarray, shape (N*nx, nx)
        Prediction matrix của trạng thái ban đầu.

    Gamma : np.ndarray, shape (N*nx, N*nu)
        Prediction matrix của chuỗi điều khiển.
    """

    # ========================================================
    # Input conversion
    # ========================================================

    Ad = np.asarray(
        Ad,
        dtype=float
    )

    Bd = np.asarray(
        Bd,
        dtype=float
    )

    N = int(N)


    # ========================================================
    # Validation
    # ========================================================

    if Ad.ndim != 2:
        raise ValueError(
            "Ad phải là ma trận 2D."
        )

    if Bd.ndim != 2:
        raise ValueError(
            "Bd phải là ma trận 2D."
        )

    nx = Ad.shape[0]

    if Ad.shape[1] != nx:
        raise ValueError(
            "Ad phải là ma trận vuông."
        )

    if Bd.shape[0] != nx:
        raise ValueError(
            "Số hàng của Bd phải bằng số state của Ad."
        )

    if N <= 0:
        raise ValueError(
            "Prediction horizon N phải lớn hơn 0."
        )

    nu = Bd.shape[1]


    # ========================================================
    # Allocate prediction matrices
    # ========================================================

    Phi = np.zeros(
        (N * nx, nx),
        dtype=float
    )

    Gamma = np.zeros(
        (N * nx, N * nu),
        dtype=float
    )


    # ========================================================
    # Build Phi
    #
    # Phi =
    #
    # [ Ad   ]
    # [ Ad^2 ]
    # [ Ad^3 ]
    # [  ... ]
    # [ Ad^N ]
    # ========================================================

    Ad_power = np.eye(nx)

    for i in range(N):

        Ad_power = Ad_power @ Ad

        row_start = i * nx
        row_end = (i + 1) * nx

        Phi[
            row_start:row_end,
            :
        ] = Ad_power


    # ========================================================
    # Build Gamma
    #
    # Gamma =
    #
    # [ Bd              0       ...   0 ]
    # [ Ad Bd           Bd      ...   0 ]
    # [ Ad^2 Bd         Ad Bd   ...   0 ]
    # [   ...            ...    ... ... ]
    #
    # General block:
    #
    # Gamma(i,j) = Ad^(i-j) Bd
    #
    # với:
    #
    # j <= i
    # ========================================================

    for i in range(N):

        row_start = i * nx
        row_end = (i + 1) * nx

        for j in range(i + 1):

            col_start = j * nu
            col_end = (j + 1) * nu

            power = i - j

            Gamma[
                row_start:row_end,
                col_start:col_end
            ] = (
                np.linalg.matrix_power(
                    Ad,
                    power
                )
                @ Bd
            )


    return Phi, Gamma