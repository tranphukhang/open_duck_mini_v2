import numpy as np
import scipy.sparse as sp
import osqp

from QP_diagnostics import (
    check_QP_before_solve,
    check_QP_after_solve,
    print_QP_diagnostics
)


class QPSolver:
    """
    OSQP solver cho SRBD-MPC.

    QP formulation:

        minimize

            1/2 * U.T @ H @ U
            +
            f.T @ U

        subject to

            l <= A @ U <= u


    Decision variable:

        U =
        [
            u_0
            u_1
            ...
            u_{N-1}
        ]

    với:

        u_k =
        [
            F_Lx, F_Ly, F_Lz,
            M_Lx, M_Ly, M_Lz,
            F_Rx, F_Ry, F_Rz,
            M_Rx, M_Ry, M_Rz
        ]
    """


    def __init__(
        self,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=4000,
        polishing=True,
        enable_warm_start=True
    ):
        """
        Parameters
        ----------
        verbose : bool
            Hiển thị log của OSQP.

        eps_abs : float
            Absolute convergence tolerance.

        eps_rel : float
            Relative convergence tolerance.

        max_iter : int
            Số iteration tối đa.

        polishing : bool
            Bật OSQP polishing.

        enable_warm_start : bool
            Cho phép OSQP warm start.
        """

        self.verbose = verbose

        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        self.max_iter = max_iter

        self.polishing = polishing

        self.enable_warm_start = (
            enable_warm_start
        )


        # ====================================================
        # Solver object
        # ====================================================

        self.solver = None

        self.is_initialized = False


        # ====================================================
        # Current QP data
        # ====================================================

        self.H = None
        self.f = None

        self.A = None
        self.l = None
        self.u = None


        # ====================================================
        # Dimensions
        # ====================================================

        self.nU = None
        self.n_constraints = None


        # ====================================================
        # Diagnostics
        # ====================================================

        self.pre_diagnostics = None
        self.post_diagnostics = None


        # ====================================================
        # Previous OSQP solution
        #
        # Useful for explicit warm start later.
        # ====================================================

        self.last_U = None
        self.last_y = None


    # ========================================================
    # SETUP QP
    # ========================================================

    def setup(
        self,
        H,
        f,
        A,
        l,
        u,
        print_diagnostics=False
    ):
        """
        Kiểm tra và khởi tạo bài toán OSQP.

        Parameters
        ----------
        H, f, A, l, u
            QP matrices/vectors.

        print_diagnostics : bool
            In pre-solve diagnostics.

        Returns
        -------
        pre_diagnostics : dict
            Kết quả kiểm tra trước khi solve.
        """

        # ====================================================
        # Convert inputs
        # ====================================================

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


        # ====================================================
        # PRE-SOLVE DIAGNOSTICS
        # ====================================================

        self.pre_diagnostics = (
            check_QP_before_solve(
                H=H,
                f=f,
                A=A,
                l=l,
                u=u
            )
        )


        if print_diagnostics:

            print_QP_diagnostics(
                self.pre_diagnostics
            )


        # ====================================================
        # Do not send invalid QP to OSQP
        # ====================================================

        if not self.pre_diagnostics[
            "ready_to_solve"
        ]:

            raise RuntimeError(
                "QP không đạt kiểm tra pre-solve. "
                "Không gọi OSQP."
            )


        # ====================================================
        # Numerical symmetry
        # ====================================================

        H = 0.5 * (
            H + H.T
        )


        # ====================================================
        # Store QP
        # ====================================================

        self.H = H.copy()

        self.f = f.copy()

        self.A = A.copy()

        self.l = l.copy()

        self.u = u.copy()


        self.nU = H.shape[0]

        self.n_constraints = A.shape[0]


        # ====================================================
        # Convert to sparse CSC
        #
        # OSQP notation:
        #
        #     P = H
        #     q = f
        #
        # OSQP only uses upper triangular part of P.
        # ====================================================

        P_sparse = sp.triu(
            sp.csc_matrix(
                H
            ),
            format="csc"
        )


        A_sparse = sp.csc_matrix(
            A
        )


        # ====================================================
        # Create solver
        # ====================================================

        self.solver = osqp.OSQP()


        # ====================================================
        # Setup OSQP
        # ====================================================

        self.solver.setup(
            P=P_sparse,
            q=f,
            A=A_sparse,
            l=l,
            u=u,

            verbose=self.verbose,

            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,

            max_iter=self.max_iter,

            polishing=self.polishing,

            warm_starting=(
                self.enable_warm_start
            )
        )


        self.is_initialized = True


        # ====================================================
        # Reset previous solution
        # ====================================================

        self.last_U = None
        self.last_y = None


        return self.pre_diagnostics


    # ========================================================
    # SOLVE QP
    # ========================================================

    def solve(
        self,
        print_diagnostics=False
    ):
        """
        Giải QP hiện tại.

        Returns
        -------
        U_opt : np.ndarray or None
            Full optimal MPC input sequence.

        u0_opt : np.ndarray or None
            First optimal control input.

            Shape = (12,)

        solver_info : dict
            Thông tin từ OSQP.

        post_diagnostics : dict or None
            Kết quả kiểm tra nghiệm sau solve.
        """

        # ====================================================
        # Check setup
        # ====================================================

        if not self.is_initialized:

            raise RuntimeError(
                "QPSolver chưa được setup."
            )


        # ====================================================
        # Solve
        # ====================================================

        result = self.solver.solve()


        # ====================================================
        # Solver information
        # ====================================================

        solver_info = {

            "status":
                result.info.status,

            "status_val":
                result.info.status_val,

            "objective":
                result.info.obj_val,

            "iterations":
                result.info.iter,

            "primal_residual":
                result.info.prim_res,

            "dual_residual":
                result.info.dual_res,

            "setup_time":
                result.info.setup_time,

            "solve_time":
                result.info.solve_time,

            "update_time":
                result.info.update_time,

            "polish_time":
                result.info.polish_time,

            "run_time":
                result.info.run_time
        }


        # ====================================================
        # OSQP status:
        #
        # 1 = solved
        # 2 = solved inaccurate
        # ====================================================

        solved_status = (
            result.info.status_val
            in (
                1,
                2
            )
        )


        # ====================================================
        # No valid primal solution
        # ====================================================

        if (
            not solved_status
            or
            result.x is None
        ):

            self.post_diagnostics = None


            if self.verbose:

                print(
                    "\nOSQP solve failed:"
                )

                print(
                    "status     =",
                    result.info.status
                )

                print(
                    "status_val =",
                    result.info.status_val
                )


            return (
                None,
                None,
                solver_info,
                None
            )


        # ====================================================
        # Full optimal sequence
        # ====================================================

        U_opt = np.asarray(
            result.x,
            dtype=float
        ).reshape(-1)


        # ====================================================
        # Dimension check
        # ====================================================

        if U_opt.shape != (
            self.nU,
        ):

            raise RuntimeError(
                "Dimension của nghiệm OSQP "
                "không khớp decision variable."
            )


        # ====================================================
        # POST-SOLVE DIAGNOSTICS
        # ====================================================

        self.post_diagnostics = (
            check_QP_after_solve(
                U_opt=U_opt,

                H=self.H,
                f=self.f,

                A=self.A,
                l=self.l,
                u=self.u,

                solver_info=solver_info,

                constraint_tolerance=max(
                    self.eps_abs,
                    1e-5
                )
            )
        )


        if print_diagnostics:

            print_QP_diagnostics(
                self.post_diagnostics
            )


        # ====================================================
        # Do not apply invalid solution
        # ====================================================

        if not self.post_diagnostics[
            "solution_ok"
        ]:

            return (
                U_opt,
                None,
                solver_info,
                self.post_diagnostics
            )


        # ====================================================
        # First MPC input
        #
        # u0 =
        #
        # [
        #     F_Lx, F_Ly, F_Lz,
        #     M_Lx, M_Ly, M_Lz,
        #     F_Rx, F_Ry, F_Rz,
        #     M_Rx, M_Ry, M_Rz
        # ]
        # ====================================================

        if self.nU < 12:

            raise RuntimeError(
                "Decision variable có ít hơn "
                "12 phần tử."
            )


        u0_opt = U_opt[
            0:12
        ].copy()


        # ====================================================
        # Save solution for warm start / monitoring
        # ====================================================

        self.last_U = U_opt.copy()


        if result.y is not None:

            self.last_y = np.asarray(
                result.y,
                dtype=float
            ).reshape(-1).copy()


        # ====================================================
        # Return
        # ====================================================

        return (
            U_opt,
            u0_opt,
            solver_info,
            self.post_diagnostics
        )


    # ========================================================
    # MANUAL WARM START
    # ========================================================

    def warm_start(
        self,
        U_init=None,
        y_init=None
    ):
        """
        Warm start thủ công cho OSQP.

        Parameters
        ----------
        U_init : np.ndarray, optional
            Initial primal variable.

        y_init : np.ndarray, optional
            Initial dual variable.
        """

        if not self.is_initialized:

            raise RuntimeError(
                "QPSolver chưa được setup."
            )


        # ====================================================
        # Build warm-start data
        # ====================================================

        warm_data = {}


        # ----------------------------------------------------
        # Primal
        # ----------------------------------------------------

        if U_init is not None:

            U_init = np.asarray(
                U_init,
                dtype=float
            ).reshape(-1)


            if U_init.shape != (
                self.nU,
            ):

                raise ValueError(
                    f"U_init phải có shape "
                    f"({self.nU},)."
                )


            if not np.all(
                np.isfinite(
                    U_init
                )
            ):

                raise ValueError(
                    "U_init chứa NaN hoặc Inf."
                )


            warm_data[
                "x"
            ] = U_init


        # ----------------------------------------------------
        # Dual
        # ----------------------------------------------------

        if y_init is not None:

            y_init = np.asarray(
                y_init,
                dtype=float
            ).reshape(-1)


            if y_init.shape != (
                self.n_constraints,
            ):

                raise ValueError(
                    f"y_init phải có shape "
                    f"({self.n_constraints},)."
                )


            if not np.all(
                np.isfinite(
                    y_init
                )
            ):

                raise ValueError(
                    "y_init chứa NaN hoặc Inf."
                )


            warm_data[
                "y"
            ] = y_init


        # ====================================================
        # Warm start OSQP
        # ====================================================

        if warm_data:

            self.solver.warm_start(
                **warm_data
            )


    # ========================================================
    # SHIFTED MPC WARM START
    # ========================================================

    def get_shifted_warm_start(
        self,
        nu=12
    ):
        """
        Tạo shifted warm-start sequence từ nghiệm trước.

        Nếu:

            U_prev =
            [
                u0
                u1
                ...
                u_{N-1}
            ]

        thì:

            U_shift =
            [
                u1
                u2
                ...
                u_{N-1}
                u_{N-1}
            ]

        Parameters
        ----------
        nu : int
            Input dimension per MPC step.
            Mặc định = 12.

        Returns
        -------
        U_shift : np.ndarray or None
            Shifted warm-start sequence.
        """

        if self.last_U is None:

            return None


        if self.last_U.size < nu:

            raise RuntimeError(
                "last_U không đủ kích thước "
                "để thực hiện shifted warm start."
            )


        if (
            self.last_U.size
            %
            nu
            != 0
        ):

            raise RuntimeError(
                "Dimension của last_U không chia hết "
                "cho nu."
            )


        U_shift = np.empty_like(
            self.last_U
        )


        # ====================================================
        # Shift horizon
        # ====================================================

        U_shift[
            :-nu
        ] = self.last_U[
            nu:
        ]


        # ====================================================
        # Repeat final control input
        # ====================================================

        U_shift[
            -nu:
        ] = self.last_U[
            -nu:
        ]


        return U_shift


    # ========================================================
    # RESET SOLVER
    # ========================================================

    def reset(self):
        """
        Reset toàn bộ solver state.
        """

        self.solver = None

        self.is_initialized = False


        self.H = None
        self.f = None

        self.A = None
        self.l = None
        self.u = None


        self.nU = None
        self.n_constraints = None


        self.pre_diagnostics = None
        self.post_diagnostics = None


        self.last_U = None
        self.last_y = None