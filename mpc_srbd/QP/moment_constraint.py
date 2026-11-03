import numpy as np


def build_moment_constraints(
    N,
    L6_R_W,
    R6_R_W,
    l_t,
    l_h,
    mu,
    Mx_tolerance,
):
    """
    Build line-contact moment constraints for SRBD-MPC.

    Decision variable
    -----------------

    U =
    [
        u_0
        u_1
        ...
        u_{N-1}
    ]

    with

    u_k =
    [
        F_Lx, F_Ly, F_Lz,
        M_Lx, M_Ly, M_Lz,

        F_Rx, F_Ry, F_Rz,
        M_Rx, M_Ry, M_Rz
    ]

    All MPC decision variables are expressed in World frame.


    ===========================================================
    WORLD -> FOOT FRAME
    ===========================================================

    F_foot = R_foot_W @ F_W

    M_foot = R_foot_W @ M_W


    ===========================================================
    MOMENT CONSTRAINTS
    ===========================================================

    For each foot:


    1. Relaxed line-contact roll moment

        Ideal line-contact model:

            Mx_foot = 0

        Implementation:

            -Mx_tolerance
            <=
            Mx_foot
            <=
            Mx_tolerance


    2. Toe / heel normal-contact constraint

        -l_t * Fz_foot
        <=
        My_foot
        <=
        l_h * Fz_foot

    Equivalent to:

        My_foot - l_h * Fz_foot <= 0

       -My_foot - l_t * Fz_foot <= 0


    3. Toe / heel lateral friction constraints

        |f_ty| <= mu * f_tz

        |f_hy| <= mu * f_hz

    where

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

    After eliminating the denominator:

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


    ===========================================================
    OSQP FORM
    ===========================================================

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
            World -> left contact frame.

    R6_R_W : np.ndarray, shape (3, 3)
        Rotation matrix:
            World -> right contact frame.

    l_t : float
        Distance from L6/R6 to toe along local x-axis [m].

    l_h : float
        Distance from L6/R6 to heel along local x-axis [m].

    mu : float
        Friction coefficient.

    Mx_tolerance : float
        Symmetric roll-moment relaxation [N.m]:

            -Mx_tolerance <= Mx_foot <= Mx_tolerance


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
    # INPUT CONVERSION
    # ========================================================

    L6_R_W = np.asarray(
        L6_R_W,
        dtype=float,
    )

    R6_R_W = np.asarray(
        R6_R_W,
        dtype=float,
    )

    Mx_tolerance = float(
        Mx_tolerance
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if not isinstance(
        N,
        (int, np.integer),
    ):
        raise TypeError(
            "N phải là số nguyên."
        )

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

    if not np.all(
        np.isfinite(
            L6_R_W
        )
    ):
        raise ValueError(
            "L6_R_W chứa NaN hoặc Inf."
        )

    if not np.all(
        np.isfinite(
            R6_R_W
        )
    ):
        raise ValueError(
            "R6_R_W chứa NaN hoặc Inf."
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

    if not np.isfinite(
        Mx_tolerance
    ):
        raise ValueError(
            "Mx_tolerance phải là số hữu hạn."
        )

    if Mx_tolerance < 0.0:
        raise ValueError(
            "Mx_tolerance phải >= 0."
        )


    # ========================================================
    # DIMENSIONS
    # ========================================================

    nu = 12

    # Per foot:
    #
    #   1 relaxed Mx constraint
    #   2 toe/heel normal-contact constraints
    #   4 toe/heel lateral-friction constraints
    #
    # => 7 constraints / foot
    #
    # Two feet:
    #
    # => 14 constraints / prediction step

    constraints_per_step = 14

    total_constraints = (
        N
        *
        constraints_per_step
    )


    # ========================================================
    # ALLOCATE MATRICES
    # ========================================================

    A_moment = np.zeros(
        (
            total_constraints,
            N * nu,
        ),
        dtype=float,
    )

    # Default inequality:
    #
    #     -inf <= A U <= 0
    #
    # The Mx rows are overwritten by:
    #
    #     -Mx_tolerance
    #     <=
    #     Mx_foot
    #     <=
    #     Mx_tolerance

    l_moment = np.full(
        total_constraints,
        -np.inf,
        dtype=float,
    )

    u_moment = np.zeros(
        total_constraints,
        dtype=float,
    )


    # ========================================================
    # SELECTOR VECTORS
    # ========================================================

    S_x = np.array(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    S_y = np.array(
        [
            0.0,
            1.0,
            0.0,
        ],
        dtype=float,
    )

    S_z = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )


    # ========================================================
    # LEFT FOOT LOCAL COMPONENT ROWS
    # ========================================================

    L_Fy_row = (
        S_y
        @
        L6_R_W
    )

    L_Fz_row = (
        S_z
        @
        L6_R_W
    )

    L_Mx_row = (
        S_x
        @
        L6_R_W
    )

    L_My_row = (
        S_y
        @
        L6_R_W
    )

    L_Mz_row = (
        S_z
        @
        L6_R_W
    )


    # ========================================================
    # RIGHT FOOT LOCAL COMPONENT ROWS
    # ========================================================

    R_Fy_row = (
        S_y
        @
        R6_R_W
    )

    R_Fz_row = (
        S_z
        @
        R6_R_W
    )

    R_Mx_row = (
        S_x
        @
        R6_R_W
    )

    R_My_row = (
        S_y
        @
        R6_R_W
    )

    R_Mz_row = (
        S_z
        @
        R6_R_W
    )


    # ========================================================
    # BUILD CONSTRAINTS
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
            step_col + 3,
        )

        L_moment_cols = slice(
            step_col + 3,
            step_col + 6,
        )


        # ----------------------------------------------------
        # 1. RELAXED LEFT-FOOT ROLL MOMENT
        #
        # -Mx_tolerance
        # <=
        # Mx_L
        # <=
        # Mx_tolerance
        # ----------------------------------------------------

        A_moment[
            row,
            L_moment_cols,
        ] = (
            L_Mx_row
        )

        l_moment[row] = (
            -Mx_tolerance
        )

        u_moment[row] = (
            Mx_tolerance
        )

        row += 1


        # ----------------------------------------------------
        # 2. My_L - l_h * Fz_L <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols,
        ] = (
            -l_h
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols,
        ] = (
            L_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 3. -My_L - l_t * Fz_L <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols,
        ] = (
            -l_t
            *
            L_Fz_row
        )

        A_moment[
            row,
            L_moment_cols,
        ] = (
            -L_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 4. LEFT TOE FRICTION
        #
        # l_h*Fy
        # - mu*l_h*Fz
        # + mu*My
        # + Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols,
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
            L_moment_cols,
        ] = (
            mu
            *
            L_My_row
            +
            L_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 5. LEFT TOE FRICTION
        #
        # -l_h*Fy
        # - mu*l_h*Fz
        # + mu*My
        # - Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols,
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
            L_moment_cols,
        ] = (
            mu
            *
            L_My_row
            -
            L_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 6. LEFT HEEL FRICTION
        #
        # l_t*Fy
        # - mu*l_t*Fz
        # - mu*My
        # - Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols,
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
            L_moment_cols,
        ] = (
            -mu
            *
            L_My_row
            -
            L_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 7. LEFT HEEL FRICTION
        #
        # -l_t*Fy
        # - mu*l_t*Fz
        # - mu*My
        # + Mz
        # <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            L_force_cols,
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
            L_moment_cols,
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
            step_col + 9,
        )

        R_moment_cols = slice(
            step_col + 9,
            step_col + 12,
        )


        # ----------------------------------------------------
        # 8. RELAXED RIGHT-FOOT ROLL MOMENT
        #
        # -Mx_tolerance
        # <=
        # Mx_R
        # <=
        # Mx_tolerance
        # ----------------------------------------------------

        A_moment[
            row,
            R_moment_cols,
        ] = (
            R_Mx_row
        )

        l_moment[row] = (
            -Mx_tolerance
        )

        u_moment[row] = (
            Mx_tolerance
        )

        row += 1


        # ----------------------------------------------------
        # 9. My_R - l_h * Fz_R <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols,
        ] = (
            -l_h
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols,
        ] = (
            R_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 10. -My_R - l_t * Fz_R <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols,
        ] = (
            -l_t
            *
            R_Fz_row
        )

        A_moment[
            row,
            R_moment_cols,
        ] = (
            -R_My_row
        )

        row += 1


        # ----------------------------------------------------
        # 11. RIGHT TOE FRICTION
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols,
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
            R_moment_cols,
        ] = (
            mu
            *
            R_My_row
            +
            R_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 12. RIGHT TOE FRICTION
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols,
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
            R_moment_cols,
        ] = (
            mu
            *
            R_My_row
            -
            R_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 13. RIGHT HEEL FRICTION
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols,
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
            R_moment_cols,
        ] = (
            -mu
            *
            R_My_row
            -
            R_Mz_row
        )

        row += 1


        # ----------------------------------------------------
        # 14. RIGHT HEEL FRICTION
        # ----------------------------------------------------

        A_moment[
            row,
            R_force_cols,
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
            R_moment_cols,
        ] = (
            -mu
            *
            R_My_row
            +
            R_Mz_row
        )

        row += 1


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if row != total_constraints:
        raise RuntimeError(
            "Số moment constraints được xây dựng "
            "không khớp với kích thước dự kiến."
        )

    if A_moment.shape != (
        14 * N,
        12 * N,
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

    if np.any(
        l_moment
        >
        u_moment
    ):
        raise RuntimeError(
            "Moment constraints có lower bound > upper bound."
        )


    return (
        A_moment,
        l_moment,
        u_moment,
    )