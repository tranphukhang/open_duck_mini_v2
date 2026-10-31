import numpy as np


def build_cost_function(
    Phi,
    Gamma,
    x_current,
    X_ref,
    Q,
    R
):
    """
    Xây dựng hàm mục tiêu QP cho SRBD-MPC.

    Prediction model:

        X = Phi @ x_current + Gamma @ U

    Cost:

        J =
            (X - X_ref)^T Q_bar (X - X_ref)
            + U^T R_bar U

    Dạng QP:

        min  0.5 * U^T H U + f^T U

    Do đó:

        H = 2 * (Gamma^T Q_bar Gamma + R_bar)

        f = 2 * Gamma^T Q_bar
            * (Phi x_current - X_ref)

    Parameters
    ----------
    Phi : np.ndarray, shape (N*nx, nx)
        Prediction matrix của trạng thái hiện tại.

    Gamma : np.ndarray, shape (N*nx, N*nu)
        Prediction matrix của chuỗi điều khiển.

    x_current : np.ndarray, shape (nx,)
        State hiện tại.

    X_ref : np.ndarray
        Reference trajectory.

        Có thể truyền:

            shape (nx,)
                Một reference duy nhất,
                tự động lặp lại N bước.

        hoặc:

            shape (N*nx,)
                Reference trajectory đã được stack.

    Q : np.ndarray, shape (nx, nx)
        State weighting matrix cho một prediction step.

    R : np.ndarray, shape (nu, nu)
        Input weighting matrix cho một prediction step.

    Returns
    -------
    H : np.ndarray, shape (N*nu, N*nu)
        Hessian matrix của bài toán QP.

    f : np.ndarray, shape (N*nu,)
        Linear cost vector của bài toán QP.

    Q_bar : np.ndarray, shape (N*nx, N*nx)
        Stacked state weighting matrix.

    R_bar : np.ndarray, shape (N*nu, N*nu)
        Stacked input weighting matrix.

    X_ref_stack : np.ndarray, shape (N*nx,)
        Reference trajectory sau khi stack.
    """

    # ========================================================
    # Input conversion
    # ========================================================

    Phi = np.asarray(
        Phi,
        dtype=float
    )

    Gamma = np.asarray(
        Gamma,
        dtype=float
    )

    x_current = np.asarray(
        x_current,
        dtype=float
    ).reshape(-1)

    X_ref = np.asarray(
        X_ref,
        dtype=float
    ).reshape(-1)

    Q = np.asarray(
        Q,
        dtype=float
    )

    R = np.asarray(
        R,
        dtype=float
    )


    # ========================================================
    # Dimensions
    # ========================================================

    nx = x_current.size

    if Q.shape != (nx, nx):
        raise ValueError(
            f"Q phải có shape ({nx}, {nx}), "
            f"nhưng hiện tại là {Q.shape}."
        )

    if Phi.shape[1] != nx:
        raise ValueError(
            "Số cột của Phi phải bằng số state nx."
        )

    if Phi.shape[0] % nx != 0:
        raise ValueError(
            "Số hàng của Phi phải chia hết cho nx."
        )

    N = Phi.shape[0] // nx

    if Gamma.shape[0] != N * nx:
        raise ValueError(
            "Gamma và Phi không cùng prediction horizon."
        )

    if Gamma.shape[1] % N != 0:
        raise ValueError(
            "Số cột của Gamma không phù hợp với N."
        )

    nu = Gamma.shape[1] // N

    if R.shape != (nu, nu):
        raise ValueError(
            f"R phải có shape ({nu}, {nu}), "
            f"nhưng hiện tại là {R.shape}."
        )


    # ========================================================
    # Reference trajectory
    # ========================================================

    # Nếu X_ref chỉ là một state reference:
    #
    # X_ref_stack =
    #
    # [
    #   x_ref
    #   x_ref
    #   ...
    #   x_ref
    # ]

    if X_ref.size == nx:

        X_ref_stack = np.tile(
            X_ref,
            N
        )

    elif X_ref.size == N * nx:

        X_ref_stack = X_ref.copy()

    else:

        raise ValueError(
            "X_ref phải có kích thước nx "
            "hoặc N*nx."
        )


    # ========================================================
    # Stacked weighting matrices
    #
    # Q_bar = diag(Q, Q, ..., Q)
    #
    # R_bar = diag(R, R, ..., R)
    # ========================================================

    Q_bar = np.kron(
        np.eye(N),
        Q
    )

    R_bar = np.kron(
        np.eye(N),
        R
    )


    # ========================================================
    # Prediction error independent of U
    #
    # e = Phi x_current - X_ref
    # ========================================================

    prediction_error = (
        Phi @ x_current
        - X_ref_stack
    )


    # ========================================================
    # Hessian matrix
    #
    # H =
    #
    # 2 * (
    #     Gamma^T Q_bar Gamma
    #     + R_bar
    # )
    # ========================================================

    H = 2.0 * (
        Gamma.T
        @ Q_bar
        @ Gamma
        + R_bar
    )


    # ========================================================
    # Linear term
    #
    # f =
    #
    # 2 * Gamma^T Q_bar
    #     (Phi x_current - X_ref)
    # ========================================================

    f = 2.0 * (
        Gamma.T
        @ Q_bar
        @ prediction_error
    )


    # ========================================================
    # Numerical symmetry
    #
    # Về lý thuyết:
    #
    # H = H^T
    #
    # Symmetrize để loại bỏ sai số floating-point nhỏ.
    # ========================================================

    H = 0.5 * (
        H + H.T
    )


    return (
        H,
        f,
        Q_bar,
        R_bar,
        X_ref_stack
    )