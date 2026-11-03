import numpy as np


def build_moment_constraints(
    N,
    L6_R_W,
    R6_R_W,
    l_t,
    l_h,
    mu,
    Mx_tolerance,
    enable_toe_heel_lateral_friction=False,
):
    """
    Build contact-moment constraints for SRBD-MPC.

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

    All decision variables are expressed in World frame.


    ===========================================================
    WORLD -> FOOT FRAME
    ===========================================================

        F_foot = R_foot_W @ F_W

        M_foot = R_foot_W @ M_W


    ===========================================================
    BASELINE MOMENT CONSTRAINTS
    ===========================================================

    For each foot:


    1. Relaxed line-contact roll moment

        -Mx_tolerance
        <=
        Mx_foot
        <=
        Mx_tolerance


    2. Toe / heel CoP constraints

        -l_t * Fz_foot
        <=
        My_foot
        <=
        l_h * Fz_foot

    Equivalent to:

        My_foot - l_h * Fz_foot <= 0

       -My_foot - l_t * Fz_foot <= 0


    Baseline:

        3 constraints / foot

        6 constraints / prediction step


    ===========================================================
    OPTIONAL EXTENDED TOE/HEEL LATERAL-FRICTION CONSTRAINTS
    ===========================================================

    If:

        enable_toe_heel_lateral_friction = True

    additionally enforce:

        |f_ty| <= mu * f_tz

        |f_hy| <= mu * f_hz

    where:

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


    Extended mode:

        7 constraints / foot

        14 constraints / prediction step


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
        World -> left contact-frame rotation.

    R6_R_W : np.ndarray, shape (3, 3)
        World -> right contact-frame rotation.

    l_t : float
        Distance from L6/R6 to toe [m].

    l_h : float
        Distance from L6/R6 to heel [m].

    mu : float
        Friction coefficient.

    Mx_tolerance : float
        Symmetric Mx relaxation [N.m].

    enable_toe_heel_lateral_friction : bool
        False:
            baseline moment constraints.

        True:
            add 4 toe/heel lateral-friction constraints
            per foot.

    Returns
    -------
    A_moment : np.ndarray

    l_moment : np.ndarray

    u_moment : np.ndarray
    """

    # ========================================================
    # 1. INPUT CONVERSION
    # ========================================================

    L6_R_W = np.asarray(
        L6_R_W,
        dtype=float,
    )

    R6_R_W = np.asarray(
        R6_R_W,
        dtype=float,
    )

    l_t = float(
        l_t
    )

    l_h = float(
        l_h
    )

    mu = float(
        mu
    )

    Mx_tolerance = float(
        Mx_tolerance
    )

    enable_toe_heel_lateral_friction = bool(
        enable_toe_heel_lateral_friction
    )


    # ========================================================
    # 2. VALIDATION
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

    if L6_R_W.shape != (
        3,
        3,
    ):
        raise ValueError(
            "L6_R_W phải có shape (3, 3)."
        )

    if R6_R_W.shape != (
        3,
        3,
    ):
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

    if (
        not np.isfinite(l_t)
        or
        l_t <= 0.0
    ):
        raise ValueError(
            "l_t phải hữu hạn và > 0."
        )

    if (
        not np.isfinite(l_h)
        or
        l_h <= 0.0
    ):
        raise ValueError(
            "l_h phải hữu hạn và > 0."
        )

    if (
        not np.isfinite(mu)
        or
        mu < 0.0
    ):
        raise ValueError(
            "mu phải hữu hạn và >= 0."
        )

    if (
        not np.isfinite(Mx_tolerance)
        or
        Mx_tolerance < 0.0
    ):
        raise ValueError(
            "Mx_tolerance phải hữu hạn và >= 0."
        )


    # ========================================================
    # 3. DIMENSIONS
    # ========================================================

    nu = 12


    # --------------------------------------------------------
    # Baseline:
    #
    #     3 constraints / foot
    #
    # Extended:
    #
    #     7 constraints / foot
    # --------------------------------------------------------

    if enable_toe_heel_lateral_friction:

        constraints_per_foot = 7

    else:

        constraints_per_foot = 3


    constraints_per_step = (
        2
        *
        constraints_per_foot
    )


    total_constraints = (
        N
        *
        constraints_per_step
    )


    # ========================================================
    # 4. ALLOCATE MATRICES
    # ========================================================

    A_moment = np.zeros(
        (
            total_constraints,
            N * nu,
        ),
        dtype=float,
    )


    # Default:
    #
    #     -inf <= A U <= 0
    #
    # Mx rows are overwritten by:
    #
    #     -Mx_tolerance <= Mx <= Mx_tolerance

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
    # 5. SELECTOR VECTORS
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
    # 6. LEFT FOOT LOCAL COMPONENT ROWS
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
    # 7. RIGHT FOOT LOCAL COMPONENT ROWS
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
    # 8. INTERNAL FOOT-CONSTRAINT BUILDER
    # ========================================================

    def add_foot_constraints(
        row,
        force_cols,
        moment_cols,
        Fy_row,
        Fz_row,
        Mx_row,
        My_row,
        Mz_row,
    ):

        # ----------------------------------------------------
        # 1. RELAXED Mx
        #
        #     -Mx_tolerance <= Mx <= Mx_tolerance
        # ----------------------------------------------------

        A_moment[
            row,
            moment_cols,
        ] = Mx_row

        l_moment[row] = (
            -Mx_tolerance
        )

        u_moment[row] = (
            Mx_tolerance
        )

        row += 1


        # ----------------------------------------------------
        # 2. TOE CoP
        #
        #     My - l_h*Fz <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            force_cols,
        ] = (
            -l_h
            *
            Fz_row
        )

        A_moment[
            row,
            moment_cols,
        ] = My_row

        row += 1


        # ----------------------------------------------------
        # 3. HEEL CoP
        #
        #     -My - l_t*Fz <= 0
        # ----------------------------------------------------

        A_moment[
            row,
            force_cols,
        ] = (
            -l_t
            *
            Fz_row
        )

        A_moment[
            row,
            moment_cols,
        ] = (
            -My_row
        )

        row += 1


        # ====================================================
        # OPTIONAL EXTENDED CONSTRAINTS
        # ====================================================

        if enable_toe_heel_lateral_friction:

            # ------------------------------------------------
            # 4. TOE lateral friction
            #
            # l_h*Fy
            # - mu*l_h*Fz
            # + mu*My
            # + Mz
            # <= 0
            # ------------------------------------------------

            A_moment[
                row,
                force_cols,
            ] = (
                l_h
                *
                Fy_row
                -
                mu
                *
                l_h
                *
                Fz_row
            )

            A_moment[
                row,
                moment_cols,
            ] = (
                mu
                *
                My_row
                +
                Mz_row
            )

            row += 1


            # ------------------------------------------------
            # 5. TOE lateral friction
            #
            # -l_h*Fy
            # - mu*l_h*Fz
            # + mu*My
            # - Mz
            # <= 0
            # ------------------------------------------------

            A_moment[
                row,
                force_cols,
            ] = (
                -l_h
                *
                Fy_row
                -
                mu
                *
                l_h
                *
                Fz_row
            )

            A_moment[
                row,
                moment_cols,
            ] = (
                mu
                *
                My_row
                -
                Mz_row
            )

            row += 1


            # ------------------------------------------------
            # 6. HEEL lateral friction
            #
            # l_t*Fy
            # - mu*l_t*Fz
            # - mu*My
            # - Mz
            # <= 0
            # ------------------------------------------------

            A_moment[
                row,
                force_cols,
            ] = (
                l_t
                *
                Fy_row
                -
                mu
                *
                l_t
                *
                Fz_row
            )

            A_moment[
                row,
                moment_cols,
            ] = (
                -mu
                *
                My_row
                -
                Mz_row
            )

            row += 1


            # ------------------------------------------------
            # 7. HEEL lateral friction
            #
            # -l_t*Fy
            # - mu*l_t*Fz
            # - mu*My
            # + Mz
            # <= 0
            # ------------------------------------------------

            A_moment[
                row,
                force_cols,
            ] = (
                -l_t
                *
                Fy_row
                -
                mu
                *
                l_t
                *
                Fz_row
            )

            A_moment[
                row,
                moment_cols,
            ] = (
                -mu
                *
                My_row
                +
                Mz_row
            )

            row += 1


        return row


    # ========================================================
    # 9. BUILD CONSTRAINTS OVER HORIZON
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


        row = add_foot_constraints(
            row=row,

            force_cols=L_force_cols,
            moment_cols=L_moment_cols,

            Fy_row=L_Fy_row,
            Fz_row=L_Fz_row,

            Mx_row=L_Mx_row,
            My_row=L_My_row,
            Mz_row=L_Mz_row,
        )


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


        row = add_foot_constraints(
            row=row,

            force_cols=R_force_cols,
            moment_cols=R_moment_cols,

            Fy_row=R_Fy_row,
            Fz_row=R_Fz_row,

            Mx_row=R_Mx_row,
            My_row=R_My_row,
            Mz_row=R_Mz_row,
        )


    # ========================================================
    # 10. FINAL VALIDATION
    # ========================================================

    if row != total_constraints:

        raise RuntimeError(
            "Số moment constraints được xây dựng "
            "không khớp với kích thước dự kiến."
        )


    expected_shape = (
        total_constraints,
        N * nu,
    )


    if A_moment.shape != expected_shape:

        raise RuntimeError(
            "A_moment có dimension không đúng. "
            f"Expected {expected_shape}, "
            f"received {A_moment.shape}."
        )


    if l_moment.shape != (
        total_constraints,
    ):

        raise RuntimeError(
            "l_moment có dimension không đúng."
        )


    if u_moment.shape != (
        total_constraints,
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
            "Moment constraints có "
            "lower bound > upper bound."
        )


    return (
        A_moment,
        l_moment,
        u_moment,
    )