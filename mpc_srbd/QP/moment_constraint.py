import numpy as np


def build_moment_constraints(
    N,
    L6_R_W,
    R6_R_W,
    l_t,
    l_h
):
    """
    Xây dựng moment / ZMP-CoP constraints cho SRBD-MPC.

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
    World -> L6/R6 transformation
    =========================================================

    L6_R_W:
        Rotation matrix chuyển vector từ World frame
        về local frame L6.

    R6_R_W:
        Rotation matrix chuyển vector từ World frame
        về local frame R6.

    Do đó:

        L6_F = L6_R_W @ W_F

        L6_M = L6_R_W @ W_M

        R6_F = R6_R_W @ W_F

        R6_M = R6_R_W @ W_M

    =========================================================
    Moment / ZMP-CoP constraint
    =========================================================

    Với mỗi chân:

        -l_t * Fz_foot
        <=
        My_foot
        <=
        l_h * Fz_foot

    Tách thành hai bất đẳng thức:

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
        về L6 frame.

    R6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix chuyển từ World frame
        về R6 frame.

    l_t : float
        Khoảng cách từ wrench reference point
        đến toe theo trục x của foot.

    l_h : float
        Khoảng cách từ wrench reference point
        đến heel theo trục x của foot.

    Returns
    -------
    A_moment : np.ndarray
        Moment constraint matrix.
        Shape = (4*N, 12*N)

    l_moment : np.ndarray
        Lower bound vector.
        Shape = (4*N,)

    u_moment : np.ndarray
        Upper bound vector.
        Shape = (4*N,)
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
    # 2 moment constraints
    #
    # Hai chân:
    #
    # 4 constraints / prediction step

    constraints_per_step = 4

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
    # S_y extracts My from local moment vector
    #
    # S_z extracts Fz from local force vector
    # ========================================================

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
    # L6_Fz =
    # S_z @ L6_R_W @ W_F
    #
    # L6_My =
    # S_y @ L6_R_W @ W_M
    #
    # tương tự cho R6
    # ========================================================

    L_Fz_row = (
        S_z
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
        # 2.
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
        # 3.
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
        # 4.
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


    return (
        A_moment,
        l_moment,
        u_moment
    )