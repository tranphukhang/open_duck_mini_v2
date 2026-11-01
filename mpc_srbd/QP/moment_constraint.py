import numpy as np


def build_moment_constraints(
    N,
    L6_R_W,
    R6_R_W,
    l_t,
    l_h
):
    """
    Xây dựng line-contact moment constraints cho SRBD-MPC.

    Decision variable:

        U =
        [
            u_0
            u_1
            ...
            u_{N-1}
        ]

    với mỗi:

        u_k =
        [
            W_F_L6,
            W_M_L6,
            W_F_R6,
            W_M_R6
        ]

    hay theo từng thành phần:

        u_k =
        [
            F_Lx, F_Ly, F_Lz,
            M_Lx, M_Ly, M_Lz,
            F_Rx, F_Ry, F_Rz,
            M_Rx, M_Ry, M_Rz
        ]

    Tất cả decision variables vẫn được biểu diễn trong
    World frame.

    =========================================================
    World -> foot-frame transformation
    =========================================================

    L6_R_W:
        Rotation matrix chuyển vector từ World frame
        về local contact frame của chân trái.

    R6_R_W:
        Rotation matrix chuyển vector từ World frame
        về local contact frame của chân phải.

    Do đó:

        L6_F = L6_R_W @ W_F

        L6_M = L6_R_W @ W_M

        R6_F = R6_R_W @ W_F

        R6_M = R6_R_W @ W_M

    =========================================================
    Line-contact moment constraints
    =========================================================

    Với mỗi chân:

    1. Không cho phép moment quanh local x-axis:

        Mx_foot = 0

    2. Giới hạn moment quanh local y-axis:

        -l_t * Fz_foot
        <=
        My_foot
        <=
        l_h * Fz_foot

    tương đương:

        My_foot - l_h * Fz_foot <= 0

       -My_foot - l_t * Fz_foot <= 0

    =========================================================
    OSQP form
    =========================================================

        l_moment
        <=
        A_moment @ U
        <=
        u_moment

    Parameters
    ----------
    N : int
        Prediction horizon.

    L6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix chuyển từ World frame
        về local contact frame chân trái.

    R6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix chuyển từ World frame
        về local contact frame chân phải.

    l_t : float
        Khoảng cách từ wrench reference point
        đến toe theo local x-axis của foot.

    l_h : float
        Khoảng cách từ wrench reference point
        đến heel theo local x-axis của foot.

    Returns
    -------
    A_moment : np.ndarray
        Moment constraint matrix.
        Shape = (6*N, 12*N)

    l_moment : np.ndarray
        Lower bound vector.
        Shape = (6*N,)

    u_moment : np.ndarray
        Upper bound vector.
        Shape = (6*N,)
    """

    # ========================================================
    # Input conversion
    # ========================================================

    L6_R_W = np.asarray(
        L6_R_W,
        dtype=float
    )

    R6_R_W = np.asarray(
        R6_R_W,
        dtype=float
    )


    # ========================================================
    # Validation
    # ========================================================

    if N <= 0:
        raise ValueError(
            "N phải lớn hơn 0."
        )

    if L6_R_W.shape != (3, 3):
        raise ValueError(
            "L6_R_W phải có shape (3, 3)."
        )

    if R6_R_W.shape != (3, 3):
        raise ValueError(
            "R6_R_W phải có shape (3, 3)."
        )

    if l_t <= 0.0:
        raise ValueError(
            "l_t phải lớn hơn 0."
        )

    if l_h <= 0.0:
        raise ValueError(
            "l_h phải lớn hơn 0."
        )


    # ========================================================
    # Dimensions
    # ========================================================

    nu = 12

    # Mỗi chân có:
    #
    # 1 equality constraint:
    #
    #     Mx = 0
    #
    # 2 inequality constraints:
    #
    #     My - l_h*Fz <= 0
    #
    #    -My - l_t*Fz <= 0
    #
    # => 3 constraints / foot
    #
    # Hai chân:
    #
    # => 6 constraints / prediction step

    constraints_per_step = 6

    total_constraints = (
        N * constraints_per_step
    )


    # ========================================================
    # Allocate matrices
    # ========================================================

    A_moment = np.zeros(
        (
            total_constraints,
            N * nu
        ),
        dtype=float
    )

    # Mặc định các inequality constraint có:
    #
    #     -inf <= A U <= 0
    #
    # Riêng Mx = 0 sẽ được set:
    #
    #     0 <= A U <= 0

    l_moment = np.full(
        total_constraints,
        -np.inf,
        dtype=float
    )

    u_moment = np.zeros(
        total_constraints,
        dtype=float
    )


    # ========================================================
    # Selector vectors
    #
    # S_x extracts Mx from local moment vector
    #
    # S_y extracts My from local moment vector
    #
    # S_z extracts Fz from local force vector
    # ========================================================

    S_x = np.array([
        1.0,
        0.0,
        0.0
    ])

    S_y = np.array([
        0.0,
        1.0,
        0.0
    ])

    S_z = np.array([
        0.0,
        0.0,
        1.0
    ])


    # ========================================================
    # Local foot components written directly as
    # coefficients acting on World-frame wrench
    #
    # LEFT:
    #
    # L6_Fz =
    # S_z @ L6_R_W @ W_F
    #
    # L6_Mx =
    # S_x @ L6_R_W @ W_M
    #
    # L6_My =
    # S_y @ L6_R_W @ W_M
    #
    #
    # RIGHT tương tự.
    # ========================================================

    L_Fz_row = (
        S_z
        @ L6_R_W
    )

    L_Mx_row = (
        S_x
        @ L6_R_W
    )

    L_My_row = (
        S_y
        @ L6_R_W
    )


    R_Fz_row = (
        S_z
        @ R6_R_W
    )

    R_Mx_row = (
        S_x
        @ R6_R_W
    )

    R_My_row = (
        S_y
        @ R6_R_W
    )


    # ========================================================
    # Build constraints
    # ========================================================

    row = 0

    for k in range(N):

        step_col = k * nu


        # ====================================================
        # LEFT FOOT
        #
        # u_L =
        #
        # [Fx Fy Fz Mx My Mz]
        # ====================================================

        L_force_cols = slice(
            step_col,
            step_col + 3
        )

        L_moment_cols = slice(
            step_col + 3,
            step_col + 6
        )


        # ----------------------------------------------------
        # 1.
        #
        # Mx_L6 = 0
        #
        # OSQP:
        #
        #     0 <= Mx_L6 <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_moment_cols
        ] = (
            L_Mx_row
        )

        l_moment[
            row
        ] = 0.0

        u_moment[
            row
        ] = 0.0

        row += 1


        # ----------------------------------------------------
        # 2.
        #
        # My_L6 - l_h * Fz_L6 <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            -l_h
            * L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            L_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 3.
        #
        # -My_L6 - l_t * Fz_L6 <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            -l_t
            * L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            -L_My_row
        )

        row += 1


        # ====================================================
        # RIGHT FOOT
        #
        # u_R =
        #
        # [Fx Fy Fz Mx My Mz]
        # ====================================================

        R_force_cols = slice(
            step_col + 6,
            step_col + 9
        )

        R_moment_cols = slice(
            step_col + 9,
            step_col + 12
        )


        # ----------------------------------------------------
        # 4.
        #
        # Mx_R6 = 0
        #
        # OSQP:
        #
        #     0 <= Mx_R6 <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_moment_cols
        ] = (
            R_Mx_row
        )

        l_moment[
            row
        ] = 0.0

        u_moment[
            row
        ] = 0.0

        row += 1


        # ----------------------------------------------------
        # 5.
        #
        # My_R6 - l_h * Fz_R6 <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            -l_h
            * R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            R_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 6.
        #
        # -My_R6 - l_t * Fz_R6 <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            -l_t
            * R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            -R_My_row
        )

        row += 1


    # ========================================================
    # Final validation
    # ========================================================

    if row != total_constraints:
        raise RuntimeError(
            "Số moment constraints được xây dựng "
            "không khớp với kích thước dự kiến."
        )


    return (
        A_moment,
        l_moment,
        u_moment
    )