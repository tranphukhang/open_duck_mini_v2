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


# ============================================================
# BUILD LTV PREDICTION MODEL
# ============================================================

def build_prediction_model_sequence(
    Ad_sequence,
    Bd_sequence
):
    """
    Xây dựng mô hình dự đoán stacked cho MPC
    với mô hình rời rạc Linear Time-Varying (LTV).

    Tại prediction step j:

        x[j+1|k]
            =
        Ad_j @ x[j|k]
            +
        Bd_j @ u[j|k]


    với:

        j = 0, 1, ..., N-1


    Prediction states:

        x[1|k]
        x[2|k]
        ...
        x[N|k]


    vẫn có thể viết dưới dạng:

        X
            =
        Phi @ x[k]
            +
        Gamma @ U


    trong đó:

        X =
        [
            x[1|k]
            x[2|k]
            ...
            x[N|k]
        ]


        U =
        [
            u[0|k]
            u[1|k]
            ...
            u[N-1|k]
        ]


    =========================================================
    Ví dụ 3 prediction steps
    =========================================================

    Step 0:

        x1
            =
        A0 x0
            +
        B0 u0


    Step 1:

        x2
            =
        A1 A0 x0

            +
        A1 B0 u0

            +
        B1 u1


    Step 2:

        x3
            =
        A2 A1 A0 x0

            +
        A2 A1 B0 u0

            +
        A2 B1 u1

            +
        B2 u2


    Do đó Gamma có dạng:

        Gamma =

        [
            B0
            0
            0

            A1 B0
            B1
            0

            A2 A1 B0
            A2 B1
            B2

            ...
        ]


    Parameters
    ----------
    Ad_sequence : np.ndarray, shape (N, nx, nx)

        Chuỗi discrete state matrices:

            Ad_sequence[j] = Ad_j


    Bd_sequence : np.ndarray, shape (N, nx, nu)

        Chuỗi discrete input matrices:

            Bd_sequence[j] = Bd_j


    Returns
    -------
    Phi : np.ndarray, shape (N*nx, nx)

        LTV state prediction matrix.


    Gamma : np.ndarray, shape (N*nx, N*nu)

        LTV input prediction matrix.
    """


    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    Ad_sequence = np.asarray(
        Ad_sequence,
        dtype=float
    )


    Bd_sequence = np.asarray(
        Bd_sequence,
        dtype=float
    )


    # ========================================================
    # 2. BASIC VALIDATION
    # ========================================================

    if Ad_sequence.ndim != 3:

        raise ValueError(
            "Ad_sequence phải có shape "
            "(N, nx, nx)."
        )


    if Bd_sequence.ndim != 3:

        raise ValueError(
            "Bd_sequence phải có shape "
            "(N, nx, nu)."
        )


    N = (
        Ad_sequence.shape[0]
    )


    if N <= 0:

        raise ValueError(
            "Prediction horizon N phải lớn hơn 0."
        )


    if Bd_sequence.shape[0] != N:

        raise ValueError(
            "Ad_sequence và Bd_sequence phải "
            "có cùng prediction horizon."
        )


    # ========================================================
    # 3. DIMENSIONS
    # ========================================================

    nx = (
        Ad_sequence.shape[1]
    )


    if Ad_sequence.shape[2] != nx:

        raise ValueError(
            "Mỗi Ad_j phải là ma trận vuông."
        )


    if Bd_sequence.shape[1] != nx:

        raise ValueError(
            "Số hàng của mỗi Bd_j phải bằng nx."
        )


    nu = (
        Bd_sequence.shape[2]
    )


    if nu <= 0:

        raise ValueError(
            "Input dimension nu phải lớn hơn 0."
        )


    # ========================================================
    # 4. FINITE VALIDATION
    # ========================================================

    if not np.all(
        np.isfinite(
            Ad_sequence
        )
    ):

        raise ValueError(
            "Ad_sequence chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            Bd_sequence
        )
    ):

        raise ValueError(
            "Bd_sequence chứa NaN hoặc Inf."
        )


    # ========================================================
    # 5. ALLOCATE STACKED MATRICES
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
    # 6. RECURSIVE LTV PREDICTION
    #
    # Ta xây từng block row bằng recursion.
    #
    # Nếu:
    #
    #     x_j
    #
    #       =
    #
    #     Phi_prev x0
    #
    #       +
    #
    #     Gamma_prev U
    #
    #
    # thì:
    #
    #     x_{j+1}
    #
    #       =
    #
    #     A_j x_j
    #
    #       +
    #
    #     B_j u_j
    #
    #
    # do đó:
    #
    #     Phi_current
    #
    #       =
    #
    #     A_j Phi_prev
    #
    #
    # và:
    #
    #     Gamma_current
    #
    #       =
    #
    #     A_j Gamma_prev
    #
    # sau đó đặt thêm:
    #
    #     B_j
    #
    # vào block tương ứng với u_j.
    # ========================================================


    # --------------------------------------------------------
    # Initial state transition before first prediction step:
    #
    #     x0 = I x0
    # --------------------------------------------------------

    Phi_previous = np.eye(
        nx,
        dtype=float
    )


    # --------------------------------------------------------
    # Coefficient của toàn bộ U đối với x0.
    #
    # Ban đầu chưa có input nào tác động.
    # --------------------------------------------------------

    Gamma_previous = np.zeros(
        (
            nx,
            N * nu
        ),
        dtype=float
    )


    # ========================================================
    # LOOP OVER PREDICTION STEPS
    # ========================================================

    for j in range(
        N
    ):

        Ad_j = (
            Ad_sequence[
                j,
                :,
                :
            ]
        )


        Bd_j = (
            Bd_sequence[
                j,
                :,
                :
            ]
        )


        # ====================================================
        # 6.1 STATE TRANSITION
        #
        # Phi_j
        #
        #     =
        #
        # A_j @ Phi_previous
        # ====================================================

        Phi_current = (
            Ad_j
            @
            Phi_previous
        )


        # ====================================================
        # 6.2 PROPAGATE PREVIOUS INPUT EFFECTS
        #
        # Gamma_current
        #
        #     =
        #
        # A_j @ Gamma_previous
        # ====================================================

        Gamma_current = (
            Ad_j
            @
            Gamma_previous
        )


        # ====================================================
        # 6.3 ADD CURRENT INPUT
        #
        # Current prediction input:
        #
        #     u_j
        #
        # contributes:
        #
        #     Bd_j @ u_j
        #
        # Therefore:
        #
        # Gamma_current[:, j*nu:(j+1)*nu]
        #
        #     =
        #
        # Bd_j
        # ====================================================

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


        Gamma_current[
            :,
            col_start:col_end
        ] = Bd_j


        # ====================================================
        # 6.4 STORE CURRENT BLOCK ROW
        # ====================================================

        row_start = (
            j
            *
            nx
        )


        row_end = (
            (j + 1)
            *
            nx
        )


        Phi[
            row_start:row_end,
            :
        ] = Phi_current


        Gamma[
            row_start:row_end,
            :
        ] = Gamma_current


        # ====================================================
        # 6.5 PREPARE NEXT PREDICTION STEP
        # ====================================================

        Phi_previous = (
            Phi_current
        )


        Gamma_previous = (
            Gamma_current
        )


    # ========================================================
    # 7. FINAL VALIDATION
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