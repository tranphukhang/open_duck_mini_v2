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


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=4000,
        polishing=True,
        enable_warm_start=True,
        constraint_tolerance=1e-4
    ):
        """
        Parameters
        ----------
        verbose : bool
            Hiển thị log OSQP.

        eps_abs : float
            Absolute convergence tolerance của OSQP.

        eps_rel : float
            Relative convergence tolerance của OSQP.

        max_iter : int
            Số iteration tối đa của OSQP.

        polishing : bool
            Bật OSQP polishing.

        enable_warm_start : bool
            Cho phép OSQP warm start.

        constraint_tolerance : float
            Tolerance dùng để kiểm tra constraint violation
            sau khi OSQP solve.

            Đây là tolerance của post-solve diagnostics,
            độc lập với eps_abs và eps_rel của OSQP.

            Default:
                1e-4
        """

        self.verbose = bool(
            verbose
        )


        self.eps_abs = float(
            eps_abs
        )


        self.eps_rel = float(
            eps_rel
        )


        self.max_iter = int(
            max_iter
        )


        self.polishing = bool(
            polishing
        )


        self.enable_warm_start = bool(
            enable_warm_start
        )


        self.constraint_tolerance = float(
            constraint_tolerance
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if self.eps_abs <= 0.0:

            raise ValueError(
                "eps_abs phải lớn hơn 0."
            )


        if self.eps_rel <= 0.0:

            raise ValueError(
                "eps_rel phải lớn hơn 0."
            )


        if self.max_iter <= 0:

            raise ValueError(
                "max_iter phải lớn hơn 0."
            )


        if self.constraint_tolerance <= 0.0:

            raise ValueError(
                "constraint_tolerance phải lớn hơn 0."
            )


        # ====================================================
        # SOLVER OBJECT
        # ====================================================

        self.solver = None

        self.is_initialized = False


        # ====================================================
        # CURRENT QP DATA
        # ====================================================

        self.H = None
        self.f = None

        self.A = None
        self.l = None
        self.u = None


        # ====================================================
        # DIMENSIONS
        # ====================================================

        self.nU = None

        self.n_constraints = None


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        self.pre_diagnostics = None

        self.post_diagnostics = None


        # ====================================================
        # PREVIOUS SOLUTION
        #
        # Dùng cho warm start sau này.
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

        Returns
        -------
        pre_diagnostics : dict
            Kết quả kiểm tra QP trước solve.
        """

        # ====================================================
        # INPUT CONVERSION
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
        # DO NOT SEND INVALID QP TO OSQP
        # ====================================================

        if not self.pre_diagnostics[
            "ready_to_solve"
        ]:

            raise RuntimeError(
                "QP không đạt kiểm tra pre-solve. "
                "Không gọi OSQP."
            )


        # ====================================================
        # NUMERICAL SYMMETRY
        # ====================================================

        H = 0.5 * (
            H
            +
            H.T
        )


        # ====================================================
        # STORE CURRENT QP
        # ====================================================

        self.H = H.copy()

        self.f = f.copy()

        self.A = A.copy()

        self.l = l.copy()

        self.u = u.copy()


        self.nU = (
            H.shape[0]
        )


        self.n_constraints = (
            A.shape[0]
        )


        # ====================================================
        # SPARSE MATRICES
        #
        # OSQP:
        #
        #     P = H
        #     q = f
        #
        # OSQP chỉ cần upper triangular part của P.
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
        # CREATE OSQP SOLVER
        # ====================================================

        self.solver = osqp.OSQP()


        # ====================================================
        # SETUP OSQP
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
        # RESET PREVIOUS SOLUTION
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
            First optimal input.

            Shape:
                (12,)

        solver_info : dict
            Thông tin OSQP.

        post_diagnostics : dict or None
            Kết quả kiểm tra nghiệm sau solve.
        """

        # ====================================================
        # CHECK SETUP
        # ====================================================

        if not self.is_initialized:

            raise RuntimeError(
                "QPSolver chưa được setup."
            )


        # ====================================================
        # SOLVE
        # ====================================================

        result = self.solver.solve()


        # ====================================================
        # SOLVER INFORMATION
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
        # OSQP STATUS
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
        # NO VALID SOLUTION
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
                    "status =",
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
        # FULL OPTIMAL INPUT SEQUENCE
        # ====================================================

        U_opt = np.asarray(
            result.x,
            dtype=float
        ).reshape(-1)


        # ====================================================
        # DIMENSION CHECK
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
        #
        # IMPORTANT:
        #
        # constraint_tolerance được tách riêng khỏi:
        #
        #     eps_abs
        #     eps_rel
        #
        # OSQP solver tolerance:
        #
        #     eps_abs = 1e-5
        #     eps_rel = 1e-5
        #
        # Acceptance tolerance:
        #
        #     constraint_tolerance = 1e-4
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

                constraint_tolerance=(
                    self.constraint_tolerance
                )
            )
        )


        if print_diagnostics:

            print_QP_diagnostics(
                self.post_diagnostics
            )


        # ====================================================
        # DO NOT APPLY INVALID SOLUTION
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
        # FIRST MPC INPUT
        #
        # u0 =
        #
        # [
        #     F_Lx, F_Ly, F_Lz,
        #     M_Lx, M_Ly, M_Lz,
        #
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
        # SAVE SOLUTION
        # ====================================================

        self.last_U = (
            U_opt.copy()
        )


        if result.y is not None:

            self.last_y = np.asarray(
                result.y,
                dtype=float
            ).reshape(-1).copy()


        # ====================================================
        # RETURN
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
        Warm start OSQP thủ công.

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
        # BUILD WARM START DATA
        # ====================================================

        warm_data = {}


        # ====================================================
        # PRIMAL
        # ====================================================

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


        # ====================================================
        # DUAL
        # ====================================================

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
        # APPLY WARM START
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
        Shift nghiệm MPC trước:

            U_prev =
            [
                u0
                u1
                ...
                u_N-1
            ]

        thành:

            U_shift =
            [
                u1
                u2
                ...
                u_N-1
                u_N-1
            ]

        Parameters
        ----------
        nu : int
            Input dimension per MPC step.

            Default:
                12

        Returns
        -------
        U_shift : np.ndarray or None
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
                "Dimension của last_U "
                "không chia hết cho nu."
            )


        U_shift = np.empty_like(
            self.last_U
        )


        # ====================================================
        # SHIFT HORIZON
        # ====================================================

        U_shift[
            :-nu
        ] = self.last_U[
            nu:
        ]


        # ====================================================
        # REPEAT FINAL INPUT
        # ====================================================

        U_shift[
            -nu:
        ] = self.last_U[
            -nu:
        ]


        return U_shift


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self
    ):
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