import numpy as np


# ============================================================
# QP DIAGNOSTICS
# ============================================================
#
# QP formulation:
#
#     minimize
#
#         1/2 * U.T @ H @ U
#         +
#         f.T @ U
#
#     subject to
#
#         l <= A @ U <= u
#
#
# Diagnostics được chia thành:
#
#     1. Pre-solve diagnostics
#
#     2. Post-solve diagnostics
#
#
# Trong SRBD-MPC hiện tại, Hessian được thiết kế
# để thỏa:
#
#     H > 0
#
# tức H positive definite.
#
# Điều này giúp:
#
#     - QP strictly convex
#
#     - nghiệm tối ưu duy nhất
#
#     - conditioning tốt hơn
#
#     - tránh các hướng không bị penalize
#
# ============================================================


# ============================================================
# PRE-SOLVE DIAGNOSTICS
# ============================================================

def check_QP_before_solve(
    H,
    f,
    A,
    l,
    u,
    symmetry_tolerance=1e-10,
    pd_tolerance=1e-12
):
    """
    Kiểm tra bài toán QP trước khi đưa vào solver.

    Tiêu chí chính của SRBD-MPC hiện tại:

        H phải positive definite.

    Parameters
    ----------
    H : np.ndarray
        Hessian matrix.

    f : np.ndarray
        Linear cost vector.

    A : np.ndarray
        Constraint matrix.

    l : np.ndarray
        Lower constraint bound.

    u : np.ndarray
        Upper constraint bound.

    symmetry_tolerance : float
        Tolerance kiểm tra tính đối xứng của H.

    pd_tolerance : float
        Tolerance dùng khi kiểm tra eigenvalue nhỏ nhất
        của H.

        Yêu cầu:

            min_eigenvalue(H) > pd_tolerance

    Returns
    -------
    diagnostics : dict
        Kết quả kiểm tra QP trước khi solve.
    """

    # ========================================================
    # Input conversion
    # ========================================================

    H = np.asarray(
        H,
        dtype=float
    )

    f = np.asarray(
        f,
        dtype=float
    ).reshape(-1)

    A = np.asarray(
        A,
        dtype=float
    )

    l = np.asarray(
        l,
        dtype=float
    ).reshape(-1)

    u = np.asarray(
        u,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # Initialize diagnostics
    # ========================================================

    diagnostics = {

        "stage": "pre_solve",

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        "dimensions_ok": False,

        # ----------------------------------------------------
        # Objective
        # ----------------------------------------------------

        "bad_objective": True,

        # ----------------------------------------------------
        # Constraints
        # ----------------------------------------------------

        "constraint_matrix_finite": False,

        "bounds_no_nan": False,

        "bounds_ok": False,

        # ----------------------------------------------------
        # Hessian
        # ----------------------------------------------------

        "hessian_symmetric": False,

        "symmetry_error": np.inf,

        "chol_flag": 1,

        "hessian_pd": False,

        "hessian_psd": False,

        "min_eigenvalue_H": np.nan,

        "max_eigenvalue_H": np.nan,

        "condition_H": np.inf,

        "rcond_H": 0.0,

        # ----------------------------------------------------
        # Final flag
        # ----------------------------------------------------

        "ready_to_solve": False,

        # ----------------------------------------------------
        # Messages
        # ----------------------------------------------------

        "errors": [],

        "warnings": []
    }


    # ========================================================
    # 1. DIMENSION CHECK
    # ========================================================

    dimensions_ok = True


    # --------------------------------------------------------
    # H must be square
    # --------------------------------------------------------

    if H.ndim != 2:

        diagnostics["errors"].append(
            "H không phải ma trận 2 chiều."
        )

        dimensions_ok = False


    elif H.shape[0] != H.shape[1]:

        diagnostics["errors"].append(
            "H không phải ma trận vuông."
        )

        dimensions_ok = False


    # --------------------------------------------------------
    # Remaining dimensions
    # --------------------------------------------------------

    if (
        H.ndim == 2
        and
        H.shape[0] == H.shape[1]
    ):

        nU = H.shape[0]


        if f.shape != (nU,):

            diagnostics["errors"].append(
                f"f phải có shape ({nU},), "
                f"nhưng nhận được {f.shape}."
            )

            dimensions_ok = False


        if A.ndim != 2:

            diagnostics["errors"].append(
                "A không phải ma trận 2 chiều."
            )

            dimensions_ok = False


        else:

            if A.shape[1] != nU:

                diagnostics["errors"].append(
                    f"A phải có {nU} cột, "
                    f"nhưng nhận được "
                    f"{A.shape[1]}."
                )

                dimensions_ok = False


            n_constraints = A.shape[0]


            if l.shape != (n_constraints,):

                diagnostics["errors"].append(
                    f"l phải có shape "
                    f"({n_constraints},), "
                    f"nhưng nhận được "
                    f"{l.shape}."
                )

                dimensions_ok = False


            if u.shape != (n_constraints,):

                diagnostics["errors"].append(
                    f"u phải có shape "
                    f"({n_constraints},), "
                    f"nhưng nhận được "
                    f"{u.shape}."
                )

                dimensions_ok = False


    diagnostics[
        "dimensions_ok"
    ] = bool(
        dimensions_ok
    )


    # ========================================================
    # Stop if dimensions are invalid
    # ========================================================

    if not dimensions_ok:

        diagnostics[
            "ready_to_solve"
        ] = False

        return diagnostics


    # ========================================================
    # 2. OBJECTIVE FINITE CHECK
    #
    # Equivalent to badObj in old Simulink code.
    # ========================================================

    bad_objective = (
        np.any(
            ~np.isfinite(H)
        )
        or
        np.any(
            ~np.isfinite(f)
        )
    )


    diagnostics[
        "bad_objective"
    ] = bool(
        bad_objective
    )


    if bad_objective:

        diagnostics["errors"].append(
            "H hoặc f chứa NaN/Inf."
        )


    # ========================================================
    # 3. CONSTRAINT MATRIX CHECK
    # ========================================================

    constraint_matrix_finite = np.all(
        np.isfinite(A)
    )


    diagnostics[
        "constraint_matrix_finite"
    ] = bool(
        constraint_matrix_finite
    )


    if not constraint_matrix_finite:

        diagnostics["errors"].append(
            "A chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. BOUND CHECK
    #
    # OSQP cho phép:
    #
    #     +Inf
    #     -Inf
    #
    # trong l và u.
    #
    # Nhưng không cho phép NaN.
    # ========================================================

    bounds_no_nan = (
        not np.any(
            np.isnan(l)
        )
        and
        not np.any(
            np.isnan(u)
        )
    )


    diagnostics[
        "bounds_no_nan"
    ] = bool(
        bounds_no_nan
    )


    if not bounds_no_nan:

        diagnostics["errors"].append(
            "l hoặc u chứa NaN."
        )


    # ========================================================
    # Check:
    #
    #     l <= u
    # ========================================================

    if bounds_no_nan:

        bounds_ok = np.all(
            l <= u
        )

    else:

        bounds_ok = False


    diagnostics[
        "bounds_ok"
    ] = bool(
        bounds_ok
    )


    if not bounds_ok:

        diagnostics["errors"].append(
            "Tồn tại lower bound lớn hơn upper bound."
        )


    # ========================================================
    # Stop Hessian analysis if objective is invalid
    # ========================================================

    if bad_objective:

        diagnostics[
            "ready_to_solve"
        ] = False

        return diagnostics


    # ========================================================
    # 5. HESSIAN SYMMETRY
    # ========================================================

    symmetry_error = np.max(
        np.abs(
            H - H.T
        )
    )


    hessian_symmetric = (
        symmetry_error
        <=
        symmetry_tolerance
    )


    diagnostics[
        "symmetry_error"
    ] = float(
        symmetry_error
    )


    diagnostics[
        "hessian_symmetric"
    ] = bool(
        hessian_symmetric
    )


    if not hessian_symmetric:

        diagnostics["errors"].append(
            "H không đối xứng trong tolerance đã chọn."
        )


    # ========================================================
    # Numerical symmetric Hessian
    #
    # Same idea as:
    #
    #     H = 0.5 * (H + H')
    #
    # ========================================================

    H_sym = 0.5 * (
        H + H.T
    )


    # ========================================================
    # 6. CHOLESKY CHECK
    #
    # Matlab:
    #
    #     [~, cholFlag] = chol(H)
    #
    #
    # Here:
    #
    #     chol_flag = 0
    #
    #         H positive definite
    #
    #
    #     chol_flag = 1
    #
    #         Cholesky failed
    #
    # ========================================================

    try:

        np.linalg.cholesky(
            H_sym
        )

        chol_flag = 0


    except np.linalg.LinAlgError:

        chol_flag = 1


    diagnostics[
        "chol_flag"
    ] = int(
        chol_flag
    )


    # ========================================================
    # 7. EIGENVALUE ANALYSIS
    # ========================================================

    eigenvalues = np.linalg.eigvalsh(
        H_sym
    )


    min_eig = np.min(
        eigenvalues
    )

    max_eig = np.max(
        eigenvalues
    )


    diagnostics[
        "min_eigenvalue_H"
    ] = float(
        min_eig
    )


    diagnostics[
        "max_eigenvalue_H"
    ] = float(
        max_eig
    )


    # ========================================================
    # PSD check
    #
    # Kept only for diagnostics.
    # ========================================================

    hessian_psd = (
        min_eig
        >=
        -pd_tolerance
    )


    diagnostics[
        "hessian_psd"
    ] = bool(
        hessian_psd
    )


    # ========================================================
    # PD check
    #
    # Main requirement for current SRBD-MPC.
    #
    # Require:
    #
    #     lambda_min(H) > tolerance
    #
    # and Cholesky must succeed.
    # ========================================================

    hessian_pd = (
        chol_flag == 0
        and
        min_eig > pd_tolerance
    )


    diagnostics[
        "hessian_pd"
    ] = bool(
        hessian_pd
    )


    if not hessian_pd:

        diagnostics["errors"].append(
            "H không positive definite."
        )


    # ========================================================
    # 8. CONDITION NUMBER
    # ========================================================

    try:

        condition_H = np.linalg.cond(
            H_sym
        )


        if (
            np.isfinite(
                condition_H
            )
            and
            condition_H > 0.0
        ):

            rcond_H = (
                1.0
                /
                condition_H
            )

        else:

            rcond_H = 0.0


    except np.linalg.LinAlgError:

        condition_H = np.inf

        rcond_H = 0.0


    diagnostics[
        "condition_H"
    ] = float(
        condition_H
    )


    diagnostics[
        "rcond_H"
    ] = float(
        rcond_H
    )


    # ========================================================
    # Conditioning warnings
    # ========================================================

    if rcond_H < 1e-12:

        diagnostics["warnings"].append(
            "H có conditioning rất kém "
            "(rcond_H < 1e-12)."
        )


    elif rcond_H < 1e-8:

        diagnostics["warnings"].append(
            "H có conditioning tương đối kém "
            "(rcond_H < 1e-8)."
        )


    # ========================================================
    # 9. READY TO SOLVE
    #
    # Current project policy:
    #
    #     H must be positive definite.
    #
    # ========================================================

    ready_to_solve = (
        dimensions_ok
        and
        not bad_objective
        and
        constraint_matrix_finite
        and
        bounds_no_nan
        and
        bounds_ok
        and
        hessian_symmetric
        and
        hessian_pd
    )


    diagnostics[
        "ready_to_solve"
    ] = bool(
        ready_to_solve
    )


    return diagnostics


# ============================================================
# POST-SOLVE DIAGNOSTICS
# ============================================================

def check_QP_after_solve(
    U_opt,
    H,
    f,
    A,
    l,
    u,
    solver_info=None,
    constraint_tolerance=1e-5
):
    """
    Kiểm tra nghiệm QP sau khi solver hoàn tất.

    Parameters
    ----------
    U_opt : np.ndarray
        Optimal control sequence.

    H : np.ndarray
        Hessian matrix.

    f : np.ndarray
        Linear cost vector.

    A : np.ndarray
        Constraint matrix.

    l : np.ndarray
        Lower constraint bound.

    u : np.ndarray
        Upper constraint bound.

    solver_info : dict, optional
        Thông tin từ solver.

        Có thể gồm:

            status
            status_val
            iterations
            primal_residual
            dual_residual
            solve_time

    constraint_tolerance : float
        Constraint violation tolerance.

    Returns
    -------
    diagnostics : dict
        Kết quả kiểm tra sau khi solve.
    """

    # ========================================================
    # Input conversion
    # ========================================================

    U_opt = np.asarray(
        U_opt,
        dtype=float
    ).reshape(-1)

    H = np.asarray(
        H,
        dtype=float
    )

    f = np.asarray(
        f,
        dtype=float
    ).reshape(-1)

    A = np.asarray(
        A,
        dtype=float
    )

    l = np.asarray(
        l,
        dtype=float
    ).reshape(-1)

    u = np.asarray(
        u,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # Initialize
    # ========================================================

    diagnostics = {

        "stage": "post_solve",

        # ----------------------------------------------------
        # Solution
        # ----------------------------------------------------

        "solution_dimension_ok": False,

        "solution_finite": False,

        # ----------------------------------------------------
        # Objective
        # ----------------------------------------------------

        "objective_value": np.nan,

        "objective_finite": False,

        # ----------------------------------------------------
        # Constraints
        # ----------------------------------------------------

        "max_lower_violation": np.inf,

        "max_upper_violation": np.inf,

        "max_violation": np.inf,

        "constraint_ok": False,

        # ----------------------------------------------------
        # Solver
        # ----------------------------------------------------

        "solver_status": None,

        "solver_status_val": None,

        "iterations": None,

        "primal_residual": None,

        "dual_residual": None,

        "solve_time": None,

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        "solution_ok": False,

        "errors": [],

        "warnings": []
    }


    # ========================================================
    # 1. DIMENSION CHECK
    # ========================================================

    if (
        H.ndim != 2
        or
        H.shape[0] != H.shape[1]
    ):

        diagnostics["errors"].append(
            "H không phải ma trận vuông."
        )

        return diagnostics


    nU = H.shape[0]


    if U_opt.shape != (nU,):

        diagnostics["errors"].append(
            f"U_opt phải có shape ({nU},), "
            f"nhưng nhận được {U_opt.shape}."
        )

        return diagnostics


    if f.shape != (nU,):

        diagnostics["errors"].append(
            "Dimension của f không hợp lệ."
        )

        return diagnostics


    if (
        A.ndim != 2
        or
        A.shape[1] != nU
    ):

        diagnostics["errors"].append(
            "Dimension của A không hợp lệ."
        )

        return diagnostics


    n_constraints = A.shape[0]


    if (
        l.shape != (n_constraints,)
        or
        u.shape != (n_constraints,)
    ):

        diagnostics["errors"].append(
            "Dimension của l hoặc u không hợp lệ."
        )

        return diagnostics


    diagnostics[
        "solution_dimension_ok"
    ] = True


    # ========================================================
    # 2. SOLUTION FINITE CHECK
    # ========================================================

    solution_finite = np.all(
        np.isfinite(
            U_opt
        )
    )


    diagnostics[
        "solution_finite"
    ] = bool(
        solution_finite
    )


    if not solution_finite:

        diagnostics["errors"].append(
            "U_opt chứa NaN hoặc Inf."
        )

        return diagnostics


    # ========================================================
    # 3. OBJECTIVE VALUE
    #
    #     J =
    #
    #     1/2 U^T H U
    #
    #     +
    #
    #     f^T U
    # ========================================================

    H_sym = 0.5 * (
        H + H.T
    )


    objective_value = (
        0.5
        *
        U_opt.T
        @ H_sym
        @ U_opt
        +
        f.T
        @ U_opt
    )


    objective_finite = np.isfinite(
        objective_value
    )


    diagnostics[
        "objective_value"
    ] = float(
        objective_value
    )


    diagnostics[
        "objective_finite"
    ] = bool(
        objective_finite
    )


    if not objective_finite:

        diagnostics["errors"].append(
            "Objective value không finite."
        )


    # ========================================================
    # 4. CONSTRAINT VIOLATION
    #
    # OSQP form:
    #
    #     l <= A U <= u
    # ========================================================

    AU = (
        A
        @ U_opt
    )


    if not np.all(
        np.isfinite(
            AU
        )
    ):

        diagnostics["errors"].append(
            "A @ U_opt chứa NaN hoặc Inf."
        )

        return diagnostics


    # ========================================================
    # Lower violation
    #
    #     l - AU
    # ========================================================

    lower_violation = np.maximum(
        l - AU,
        0.0
    )


    # ========================================================
    # Upper violation
    #
    #     AU - u
    # ========================================================

    upper_violation = np.maximum(
        AU - u,
        0.0
    )


    # ========================================================
    # Maximum violation
    # ========================================================

    if n_constraints > 0:

        max_lower_violation = np.max(
            lower_violation
        )

        max_upper_violation = np.max(
            upper_violation
        )

    else:

        max_lower_violation = 0.0

        max_upper_violation = 0.0


    max_violation = max(
        max_lower_violation,
        max_upper_violation
    )


    diagnostics[
        "max_lower_violation"
    ] = float(
        max_lower_violation
    )


    diagnostics[
        "max_upper_violation"
    ] = float(
        max_upper_violation
    )


    diagnostics[
        "max_violation"
    ] = float(
        max_violation
    )


    # ========================================================
    # Constraint status
    # ========================================================

    constraint_ok = (
        max_violation
        <=
        constraint_tolerance
    )


    diagnostics[
        "constraint_ok"
    ] = bool(
        constraint_ok
    )


    if not constraint_ok:

        diagnostics["errors"].append(
            "Nghiệm vi phạm constraint vượt quá tolerance."
        )


    # ========================================================
    # 5. SOLVER INFORMATION
    # ========================================================

    solver_status_ok = True


    if solver_info is not None:

        diagnostics[
            "solver_status"
        ] = solver_info.get(
            "status"
        )


        diagnostics[
            "solver_status_val"
        ] = solver_info.get(
            "status_val"
        )


        diagnostics[
            "iterations"
        ] = solver_info.get(
            "iterations"
        )


        diagnostics[
            "primal_residual"
        ] = solver_info.get(
            "primal_residual"
        )


        diagnostics[
            "dual_residual"
        ] = solver_info.get(
            "dual_residual"
        )


        diagnostics[
            "solve_time"
        ] = solver_info.get(
            "solve_time"
        )


        # ====================================================
        # OSQP status
        #
        # 1 = solved
        #
        # 2 = solved inaccurate
        # ====================================================

        status_val = solver_info.get(
            "status_val"
        )


        if status_val is not None:

            solver_status_ok = (
                status_val
                in (
                    1,
                    2
                )
            )


            if not solver_status_ok:

                diagnostics[
                    "errors"
                ].append(
                    "Solver status không hợp lệ."
                )


            if status_val == 2:

                diagnostics[
                    "warnings"
                ].append(
                    "OSQP trả về solved inaccurate."
                )


    # ========================================================
    # 6. FINAL SOLUTION FLAG
    # ========================================================

    solution_ok = (
        diagnostics[
            "solution_dimension_ok"
        ]
        and
        solution_finite
        and
        objective_finite
        and
        constraint_ok
        and
        solver_status_ok
    )


    diagnostics[
        "solution_ok"
    ] = bool(
        solution_ok
    )


    return diagnostics


# ============================================================
# PRINT DIAGNOSTICS
# ============================================================

def print_QP_diagnostics(
    diagnostics
):
    """
    In diagnostics ra terminal theo dạng dễ đọc.
    """

    stage = diagnostics.get(
        "stage",
        "unknown"
    )


    print(
        "\n========================================"
    )

    print(
        f"QP diagnostics: {stage}"
    )

    print(
        "========================================"
    )


    # ========================================================
    # PRE-SOLVE
    # ========================================================

    if stage == "pre_solve":

        print(
            "dimensions_ok       :",
            diagnostics[
                "dimensions_ok"
            ]
        )

        print(
            "bad_objective       :",
            diagnostics[
                "bad_objective"
            ]
        )

        print(
            "A finite            :",
            diagnostics[
                "constraint_matrix_finite"
            ]
        )

        print(
            "bounds_no_nan       :",
            diagnostics[
                "bounds_no_nan"
            ]
        )

        print(
            "bounds_ok           :",
            diagnostics[
                "bounds_ok"
            ]
        )

        print(
            "H symmetric         :",
            diagnostics[
                "hessian_symmetric"
            ]
        )

        print(
            "symmetry error      :",
            diagnostics[
                "symmetry_error"
            ]
        )

        print(
            "chol_flag           :",
            diagnostics[
                "chol_flag"
            ]
        )

        print(
            "H positive definite :",
            diagnostics[
                "hessian_pd"
            ]
        )

        print(
            "H positive semidef. :",
            diagnostics[
                "hessian_psd"
            ]
        )

        print(
            "min eig(H)          :",
            diagnostics[
                "min_eigenvalue_H"
            ]
        )

        print(
            "max eig(H)          :",
            diagnostics[
                "max_eigenvalue_H"
            ]
        )

        print(
            "condition(H)        :",
            diagnostics[
                "condition_H"
            ]
        )

        print(
            "rcond(H)            :",
            diagnostics[
                "rcond_H"
            ]
        )

        print(
            "ready_to_solve      :",
            diagnostics[
                "ready_to_solve"
            ]
        )


    # ========================================================
    # POST-SOLVE
    # ========================================================

    elif stage == "post_solve":

        print(
            "solution finite     :",
            diagnostics[
                "solution_finite"
            ]
        )

        print(
            "objective           :",
            diagnostics[
                "objective_value"
            ]
        )

        print(
            "max lower violation :",
            diagnostics[
                "max_lower_violation"
            ]
        )

        print(
            "max upper violation :",
            diagnostics[
                "max_upper_violation"
            ]
        )

        print(
            "max violation       :",
            diagnostics[
                "max_violation"
            ]
        )

        print(
            "constraint_ok       :",
            diagnostics[
                "constraint_ok"
            ]
        )

        print(
            "solver status       :",
            diagnostics[
                "solver_status"
            ]
        )

        print(
            "iterations          :",
            diagnostics[
                "iterations"
            ]
        )

        print(
            "primal residual     :",
            diagnostics[
                "primal_residual"
            ]
        )

        print(
            "dual residual       :",
            diagnostics[
                "dual_residual"
            ]
        )

        print(
            "solve time          :",
            diagnostics[
                "solve_time"
            ]
        )

        print(
            "solution_ok         :",
            diagnostics[
                "solution_ok"
            ]
        )


    # ========================================================
    # ERRORS
    # ========================================================

    errors = diagnostics.get(
        "errors",
        []
    )


    if len(errors) > 0:

        print(
            "\nErrors:"
        )

        for error in errors:

            print(
                "  -",
                error
            )


    # ========================================================
    # WARNINGS
    # ========================================================

    warnings = diagnostics.get(
        "warnings",
        []
    )


    if len(warnings) > 0:

        print(
            "\nWarnings:"
        )

        for warning in warnings:

            print(
                "  -",
                warning
            )


    print(
        "========================================"
    )