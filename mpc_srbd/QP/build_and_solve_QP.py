# ============================================================
# QP PROBLEM
# ============================================================

from QP_problem import (
    build_QP_problem
)


# ============================================================
# BUILD AND SOLVE QP
# ============================================================

def build_and_solve_QP(
    qp_solver,
    Phi,
    Gamma,
    x_current,
    X_ref,
    N,
    L6_R_W,
    R6_R_W,
    print_diagnostics=False
):
    """
    Build and solve the MPC quadratic programming problem.

    Processing sequence
    -------------------

        Phi, Gamma
        x_current
        X_ref
        N
        L6_R_W
        R6_R_W
            |
            v
    build_QP_problem()
            |
            v
      H, f, A, l, u
            |
            v
      qp_solver.setup()
            |
            v
       qp_solver.solve()
            |
            v
      U_opt, u0_opt
      solver_info
      post_diagnostics


    Parameters
    ----------
    qp_solver : QPSolver

        Initialized QP solver object.


    Phi : np.ndarray

        MPC state prediction matrix.


    Gamma : np.ndarray

        MPC input prediction matrix.


    x_current : np.ndarray

        Current SRBD state vector:

            x_k


    X_ref : np.ndarray

        Reference trajectory over the prediction horizon.


    N : int

        MPC prediction horizon.


    L6_R_W : np.ndarray

        Rotation matrix associated with the left foot
        configuration used by the QP constraints.


    R6_R_W : np.ndarray

        Rotation matrix associated with the right foot
        configuration used by the QP constraints.


    print_diagnostics : bool

        Enable QP diagnostic printing.

        Default:

            False


    Returns
    -------
    U_opt : np.ndarray or None

        Optimal control sequence over the prediction horizon.


    u0_opt : np.ndarray or None

        First MPC control input:

            u_k


    solver_info : dict

        Information returned by the QP solver.


    post_diagnostics : dict or None

        Post-solve QP diagnostics.
    """


    # ========================================================
    # 1. BUILD QP PROBLEM
    # ========================================================

    (
        H,
        f,
        A,
        l,
        u
    ) = build_QP_problem(

        Phi=Phi,

        Gamma=Gamma,

        x_current=x_current,

        X_ref=X_ref,

        N=N,

        L6_R_W=L6_R_W,

        R6_R_W=R6_R_W
    )


    # ========================================================
    # 2. SETUP QP SOLVER
    # ========================================================

    qp_solver.setup(
        H,
        f,
        A,
        l,
        u,
        print_diagnostics=print_diagnostics
    )


    # ========================================================
    # 3. SOLVE QP
    # ========================================================

    (
        U_opt,
        u0_opt,
        solver_info,
        post_diagnostics
    ) = qp_solver.solve(
        print_diagnostics=print_diagnostics
    )


    # ========================================================
    # 4. RETURN
    # ========================================================

    return (
        U_opt,
        u0_opt,
        solver_info,
        post_diagnostics
    )