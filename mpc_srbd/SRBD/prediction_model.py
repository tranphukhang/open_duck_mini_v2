import numpy as np


# ============================================================
# BUILD LTI PREDICTION MODEL
# ============================================================

def build_prediction_model(
    Ad,
    Bd,
    N
):
    """
    Xây dựng mô hình dự đoán stacked cho MPC
    với mô hình rời rạc time-invariant:

        x[k+1] = Ad @ x[k] + Bd @ u[k]

    Prediction horizon:

        x[k+1|k]
        x[k+2|k]
        ...
        x[k+N|k]

    Dạng stacked:

        X = Phi @ x[k] + Gamma @ U

    với:

        X =
        [
            x[k+1|k]
            x[k+2|k]
            ...
            x[k+N|k]
        ]

        U =
        [
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

    Gamma : np.ndarray, shape (N*nx, N*nu)
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    Ad = np.asarray(
        Ad,
        dtype=float
    )


    Bd = np.asarray(
        Bd,
        dtype=float
    )


    N = int(
        N
    )


    # ========================================================
    # 2. VALIDATION
    # ========================================================

    if Ad.ndim != 2:

        raise ValueError(
            "Ad phải là ma trận 2D."
        )


    if Bd.ndim != 2:

        raise ValueError(
            "Bd phải là ma trận 2D."
        )


    nx = (
        Ad.shape[0]
    )


    if Ad.shape != (
        nx,
        nx
    ):

        raise ValueError(
            "Ad phải là ma trận vuông."
        )


    if Bd.shape[0] != nx:

        raise ValueError(
            "Số hàng của Bd phải bằng nx."
        )


    if N <= 0:

        raise ValueError(
            "Prediction horizon N phải lớn hơn 0."
        )


    if not np.all(
        np.isfinite(
            Ad
        )
    ):

        raise ValueError(
            "Ad chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bd
        )
    ):

        raise ValueError(
            "Bd chứa NaN hoặc Inf."
        )


    nu = (
        Bd.shape[1]
    )


    # ========================================================
    # 3. ALLOCATE MATRICES
    # ========================================================

    Phi = np.zeros(
        (
            N * nx,
            nx
        ),
        dtype=float
    )


    Gamma = np.zeros(
        (
            N * nx,
            N * nu
        ),
        dtype=float
    )


    # ========================================================
    # 4. BUILD Phi
    #
    # Phi =
    #
    # [
    #     Ad
    #     Ad^2
    #     ...
    #     Ad^N
    # ]
    # ========================================================

    Ad_power = np.eye(
        nx,
        dtype=float
    )


    for i in range(
        N
    ):

        Ad_power = (
            Ad
            @
            Ad_power
        )


        row_start = (
            i
            *
            nx
        )


        row_end = (
            (i + 1)
            *
            nx
        )


        Phi[
            row_start:row_end,
            :
        ] = Ad_power


    # ========================================================
    # 5. BUILD Gamma
    #
    # Gamma =
    #
    # [
    #     Bd            0       0     ...
    #
    #     Ad Bd         Bd      0     ...
    #
    #     Ad^2 Bd       Ad Bd   Bd    ...
    #
    #     ...
    # ]
    # ========================================================

    for i in range(
        N
    ):

        row_start = (
            i
            *
            nx
        )


        row_end = (
            (i + 1)
            *
            nx
        )


        for j in range(
            i + 1
        ):

            col_start = (
                j
                *
                nu
            )


            col_end = (
                (j + 1)
                *
                nu
            )


            power = (
                i
                -
                j
            )


            Gamma[
                row_start:row_end,
                col_start:col_end
            ] = (
                np.linalg.matrix_power(
                    Ad,
                    power
                )
                @
                Bd
            )


    # ========================================================
    # 6. FINAL VALIDATION
    # ========================================================

    if Phi.shape != (
        N * nx,
        nx
    ):

        raise RuntimeError(
            "Phi sai dimension."
        )


    if Gamma.shape != (
        N * nx,
        N * nu
    ):

        raise RuntimeError(
            "Gamma sai dimension."
        )


    if not np.all(
        np.isfinite(
            Phi
        )
    ):

        raise RuntimeError(
            "Phi chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Gamma
        )
    ):

        raise RuntimeError(
            "Gamma chứa NaN hoặc Inf."
        )


    return (
        Phi,
        Gamma
    )
