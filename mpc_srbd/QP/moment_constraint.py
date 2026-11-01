import numpy as np


def build_moment_constraints(
    N,
    L6_R_W,
    R6_R_W,
    l_t,
    l_h,
    mu
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
            F_Lx, F_Ly, F_Lz,
            M_Lx, M_Ly, M_Lz,
            F_Rx, F_Ry, F_Rz,
            M_Rx, M_Ry, M_Rz
        ]

    Tất cả decision variables được biểu diễn trong World frame.

    =========================================================
    World -> foot-frame transformation
    =========================================================

        F_foot = R_foot_W @ F_W

        M_foot = R_foot_W @ M_W

    =========================================================
    Line-contact constraints
    =========================================================

    Với mỗi chân:

    1. Line contact không sinh moment quanh local x-axis:

        Mx_foot = 0

    2. Toe / heel normal-contact constraint:

        -l_t * Fz_foot
        <=
        My_foot
        <=
        l_h * Fz_foot

    tương đương:

        My_foot - l_h * Fz_foot <= 0

       -My_foot - l_t * Fz_foot <= 0

    3. Toe / heel lateral friction pyramid:

        |f_ty| <= mu * f_tz

        |f_hy| <= mu * f_hz

    với:

        f_tz =
            (l_h * Fz - My)
            /
            (l_t + l_h)

        f_hz =
            (l_t * Fz + My)
            /
            (l_t + l_h)

        f_ty =
            (l_h * Fy + Mz)
            /
            (l_t + l_h)

        f_hy =
            (l_t * Fy - Mz)
            /
            (l_t + l_h)

    Sau khi khử mẫu số:

    Toe:

        l_h*Fy
        - mu*l_h*Fz
        + mu*My
        + Mz
        <= 0

       -l_h*Fy
        - mu*l_h*Fz
        + mu*My
        - Mz
        <= 0

    Heel:

        l_t*Fy
        - mu*l_t*Fz
        - mu*My
        - Mz
        <= 0

       -l_t*Fy
        - mu*l_t*Fz
        - mu*My
        + Mz
        <= 0

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
        Rotation matrix:
            World -> left foot contact frame.

    R6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix:
            World -> right foot contact frame.

    l_t : float
        Khoảng cách từ wrench reference point
        đến toe theo local x-axis.

    l_h : float
        Khoảng cách từ wrench reference point
        đến heel theo local x-axis.

    mu : float
        Hệ số ma sát.

    Returns
    -------
    A_moment : np.ndarray
        Shape = (14*N, 12*N)

    l_moment : np.ndarray
        Shape = (14*N,)

    u_moment : np.ndarray
        Shape = (14*N,)
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

    if mu < 0.0:
        raise ValueError(
            "mu phải >= 0."
        )


    # ========================================================
    # Dimensions
    # ========================================================

    nu = 12

    # Mỗi chân:
    #
    # 1 constraint:
    #
    #     Mx = 0
    #
    # 2 constraints:
    #
    #     toe / heel normal contact
    #
    # 4 constraints:
    #
    #     toe / heel lateral friction pyramid
    #
    # => 7 constraints / foot
    #
    # Hai chân:
    #
    # => 14 constraints / prediction step

    constraints_per_step = 14

    total_constraints = (
        N
        *
        constraints_per_step
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

    # Mặc định:
    #
    #     -inf <= A U <= 0
    #
    # Riêng equality Mx = 0:
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
    # LEFT FOOT local component rows
    # ========================================================

    L_Fy_row = (
        S_y
        @ L6_R_W
    )

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

    L_Mz_row = (
        S_z
        @ L6_R_W
    )


    # ========================================================
    # RIGHT FOOT local component rows
    # ========================================================

    R_Fy_row = (
        S_y
        @ R6_R_W
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

    R_Mz_row = (
        S_z
        @ R6_R_W
    )


    # ========================================================
    # Build constraints
    # ========================================================

    row = 0

    for k in range(N):

        step_col = (
            k
            *
            nu
        )


        # ====================================================
        # LEFT FOOT
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
        # 1. Mx_L = 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_moment_cols
        ] = (
            L_Mx_row
        )

        l_moment[row] = 0.0
        u_moment[row] = 0.0

        row += 1


        # ----------------------------------------------------
        # 2. My_L - l_h * Fz_L <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            -l_h
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            L_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 3. -My_L - l_t * Fz_L <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            -l_t
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            -L_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 4. TOE friction:
        #
        # l_h*Fy
        # - mu*l_h*Fz
        # + mu*My
        # + Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            l_h
            *
            L_Fy_row
            -
            mu
            *
            l_h
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            mu
            *
            L_My_row
            +
            L_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 5. TOE friction:
        #
        # -l_h*Fy
        # - mu*l_h*Fz
        # + mu*My
        # - Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            -l_h
            *
            L_Fy_row
            -
            mu
            *
            l_h
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            mu
            *
            L_My_row
            -
            L_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 6. HEEL friction:
        #
        # l_t*Fy
        # - mu*l_t*Fz
        # - mu*My
        # - Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            l_t
            *
            L_Fy_row
            -
            mu
            *
            l_t
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            -mu
            *
            L_My_row
            -
            L_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 7. HEEL friction:
        #
        # -l_t*Fy
        # - mu*l_t*Fz
        # - mu*My
        # + Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols
        ] = (
            -l_t
            *
            L_Fy_row
            -
            mu
            *
            l_t
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols
        ] = (
            -mu
            *
            L_My_row
            +
            L_Mz_row
        )

        row += 1


        # ====================================================
        # RIGHT FOOT
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
        # 8. Mx_R = 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_moment_cols
        ] = (
            R_Mx_row
        )

        l_moment[row] = 0.0
        u_moment[row] = 0.0

        row += 1


        # ----------------------------------------------------
        # 9. My_R - l_h * Fz_R <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            -l_h
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            R_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 10. -My_R - l_t * Fz_R <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            -l_t
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            -R_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 11. TOE friction
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            l_h
            *
            R_Fy_row
            -
            mu
            *
            l_h
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            mu
            *
            R_My_row
            +
            R_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 12. TOE friction
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            -l_h
            *
            R_Fy_row
            -
            mu
            *
            l_h
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            mu
            *
            R_My_row
            -
            R_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 13. HEEL friction
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            l_t
            *
            R_Fy_row
            -
            mu
            *
            l_t
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            -mu
            *
            R_My_row
            -
            R_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 14. HEEL friction
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols
        ] = (
            -l_t
            *
            R_Fy_row
            -
            mu
            *
            l_t
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols
        ] = (
            -mu
            *
            R_My_row
            +
            R_Mz_row
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


    if A_moment.shape != (
        14 * N,
        12 * N
    ):
        raise RuntimeError(
            "A_moment có dimension không đúng."
        )


    if l_moment.shape != (
        14 * N,
    ):
        raise RuntimeError(
            "l_moment có dimension không đúng."
        )


    if u_moment.shape != (
        14 * N,
    ):
        raise RuntimeError(
            "u_moment có dimension không đúng."
        )


    if not np.all(
        np.isfinite(
            A_moment
        )
    ):
        raise RuntimeError(
            "A_moment chứa NaN hoặc Inf."
        )


    return (
        A_moment,
        l_moment,
        u_moment
    )