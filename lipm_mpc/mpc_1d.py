# lipm_mpc/mpc_1d.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.optimize import (
    LinearConstraint,
    minimize,
)


# ============================================================
# LOCAL IMPORT
# ============================================================

if __package__:

    from .lipm_model import (
        LIPMModel1D,
    )

else:

    from lipm_model import (
        LIPMModel1D,
    )


# ============================================================
# RESULT
# ============================================================

@dataclass
class MPC1DResult:

    # Optimal jerk sequence:
    #
    # control[k] = u_k
    control: np.ndarray

    # State trajectory:
    #
    # state[0] = current state x_0
    # state[1] = predicted state x_1
    # ...
    # state[N] = terminal state x_N
    state: np.ndarray

    # Predicted ZMP trajectory:
    #
    # zmp[0]     = ZMP of x_1
    # ...
    # zmp[N - 1] = ZMP of x_N
    zmp: np.ndarray

    objective: float

    success: bool

    message: str

    iterations: int


    @property
    def first_control(
        self,
    ) -> float:
        """
        First jerk command of the optimal sequence.

        In receding-horizon MPC, this is the only command
        applied before solving the optimization problem again.
        """

        return float(
            self.control[0]
        )


# ============================================================
# LIPM MPC 1D
# ============================================================

class LIPMMPC1D:
    """
    One-dimensional LIPM Model Predictive Controller.

    The same class is instantiated independently for:

        x-axis:
            [x_G, xdot_G, xddot_G]

        y-axis:
            [y_G, ydot_G, yddot_G]

    State
    -----
        x_k = [p_G, v_G, a_G]^T

    Control
    -------
        u_k = jerk

    Dynamics
    --------
        x_{k+1} = A x_k + B u_k

    ZMP
    ---
        p_Z,k = C_zmp x_k

    Cost
    ----
        J =
            w_terminal * ||x_N - x_goal||^2
            +
            w_control * sum(u_k^2)

    ZMP constraints
    ---------------
        lower_k
        <=
        C_zmp x_{k+1}
        <=
        upper_k

        k = 0, ..., N-1

    Therefore the N support-preview entries correspond to:

        x_1, x_2, ..., x_N

    rather than:

        x_0, ..., x_{N-1}.
    """

    def __init__(
        self,
        model: LIPMModel1D,
        horizon_steps: int,
        terminal_weight: float,
        control_weight: float,
    ) -> None:

        # ====================================================
        # CHECK PARAMETERS
        # ====================================================

        if not isinstance(
            horizon_steps,
            (int, np.integer),
        ):

            raise ValueError(
                "horizon_steps must be an integer."
            )

        if horizon_steps <= 0:

            raise ValueError(
                "horizon_steps must be positive."
            )

        if terminal_weight < 0.0:

            raise ValueError(
                "terminal_weight cannot be negative."
            )

        if control_weight <= 0.0:

            raise ValueError(
                "control_weight must be positive."
            )

        # ====================================================
        # STORE MODEL
        # ====================================================

        self.model = model

        self.horizon_steps = int(
            horizon_steps
        )

        self.terminal_weight = float(
            terminal_weight
        )

        self.control_weight = float(
            control_weight
        )

        self.A = np.asarray(
            model.A,
            dtype=float,
        )

        self.B = np.asarray(
            model.B,
            dtype=float,
        ).reshape(3)

        self.C_zmp = np.asarray(
            model.C_zmp,
            dtype=float,
        ).reshape(3)

        # ====================================================
        # BUILD CONSTANT PREDICTION MATRICES
        # ====================================================

        self._build_prediction_matrices()


    # ========================================================
    # PREDICTION MATRICES
    # ========================================================

    def _build_prediction_matrices(
        self,
    ) -> None:
        """
        Build constant prediction matrices.

        Terminal state:

            x_N =
                F x_0
                +
                G_t U

        Predicted ZMP:

            Z =
                Z_x x_0
                +
                Z_u U

        where:

            Z =
            [
                z_1
                z_2
                ...
                z_N
            ]^T

        and:

            U =
            [
                u_0
                u_1
                ...
                u_{N-1}
            ]^T.
        """

        N = self.horizon_steps

        # ----------------------------------------------------
        # Powers of A
        #
        # A_powers[k] = A^k
        # ----------------------------------------------------

        A_powers = [
            np.eye(
                3,
                dtype=float,
            )
        ]

        for _ in range(N):

            A_powers.append(
                A_powers[-1]
                @
                self.A
            )

        # ====================================================
        # TERMINAL STATE
        # ====================================================
        #
        # x_N =
        # A^N x_0
        #
        # + A^(N-1) B u_0
        # + A^(N-2) B u_1
        # ...
        # + B u_(N-1)
        # ====================================================

        self.terminal_state_matrix = (
            A_powers[N].copy()
        )

        self.terminal_control_matrix = (
            np.zeros(
                (
                    3,
                    N,
                ),
                dtype=float,
            )
        )

        for j in range(N):

            self.terminal_control_matrix[
                :,
                j,
            ] = (
                A_powers[
                    N - 1 - j
                ]
                @
                self.B
            )

        # ====================================================
        # ZMP PREDICTION
        # ====================================================
        #
        # Row k corresponds to predicted state:
        #
        #     x_(k+1)
        #
        # Therefore:
        #
        # z_(k+1)
        # =
        # C A^(k+1) x_0
        #
        # +
        # sum_{j=0}^{k}
        # C A^(k-j) B u_j
        #
        # ====================================================

        self.zmp_state_matrix = (
            np.zeros(
                (
                    N,
                    3,
                ),
                dtype=float,
            )
        )

        self.zmp_control_matrix = (
            np.zeros(
                (
                    N,
                    N,
                ),
                dtype=float,
            )
        )

        for k in range(N):

            # ------------------------------------------------
            # Free response:
            #
            # C A^(k+1) x0
            # ------------------------------------------------

            self.zmp_state_matrix[
                k,
                :,
            ] = (
                self.C_zmp
                @
                A_powers[
                    k + 1
                ]
            )

            # ------------------------------------------------
            # Controlled response:
            #
            # j = 0, ..., k
            # ------------------------------------------------

            for j in range(
                k + 1
            ):

                self.zmp_control_matrix[
                    k,
                    j,
                ] = (
                    self.C_zmp
                    @
                    (
                        A_powers[
                            k - j
                        ]
                        @
                        self.B
                    )
                )


    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_state(
        state,
        name: str,
    ) -> np.ndarray:

        state = np.asarray(
            state,
            dtype=float,
        )

        if state.shape != (3,):

            raise ValueError(
                f"{name} must have shape (3,)."
            )

        if not np.all(
            np.isfinite(
                state
            )
        ):

            raise ValueError(
                f"{name} must contain finite values."
            )

        return state


    def _validate_bounds(
        self,
        lower_bounds,
        upper_bounds,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        lower_bounds = np.asarray(
            lower_bounds,
            dtype=float,
        )

        upper_bounds = np.asarray(
            upper_bounds,
            dtype=float,
        )

        expected_shape = (
            self.horizon_steps,
        )

        if (
            lower_bounds.shape
            !=
            expected_shape
        ):

            raise ValueError(
                "lower_bounds must have shape "
                f"{expected_shape}."
            )

        if (
            upper_bounds.shape
            !=
            expected_shape
        ):

            raise ValueError(
                "upper_bounds must have shape "
                f"{expected_shape}."
            )

        if not np.all(
            np.isfinite(
                lower_bounds
            )
        ):

            raise ValueError(
                "lower_bounds must be finite."
            )

        if not np.all(
            np.isfinite(
                upper_bounds
            )
        ):

            raise ValueError(
                "upper_bounds must be finite."
            )

        if np.any(
            lower_bounds
            >=
            upper_bounds
        ):

            raise ValueError(
                "Every lower bound must be "
                "smaller than its upper bound."
            )

        return (
            lower_bounds,
            upper_bounds,
        )


    # ========================================================
    # BUILD QP
    # ========================================================

    def build_qp(
        self,
        current_state,
        goal_state,
        lower_bounds,
        upper_bounds,
    ):
        """
        Build:

            min
                1/2 U^T H U
                +
                f^T U

        subject to:

            lb
            <=
            G U
            <=
            ub

        where:

            U =
            [u_0, ..., u_(N-1)]^T.
        """

        x0 = self._validate_state(
            current_state,
            "current_state",
        )

        x_goal = self._validate_state(
            goal_state,
            "goal_state",
        )

        (
            lower_bounds,
            upper_bounds,
        ) = self._validate_bounds(
            lower_bounds,
            upper_bounds,
        )

        # ====================================================
        # TERMINAL STATE
        # ====================================================

        F = (
            self.terminal_state_matrix
        )

        G_t = (
            self.terminal_control_matrix
        )

        terminal_offset = (
            F @ x0
            -
            x_goal
        )

        # ====================================================
        # COST
        # ====================================================
        #
        # J =
        #
        # wt ||F x0 + G_t U - x_goal||^2
        #
        # +
        #
        # wu ||U||^2
        #
        # Standard QP:
        #
        # J =
        # 1/2 U^T H U
        # +
        # f^T U
        # +
        # constant
        #
        # ====================================================

        H = 2.0 * (
            self.terminal_weight
            *
            (
                G_t.T
                @
                G_t
            )
            +
            self.control_weight
            *
            np.eye(
                self.horizon_steps,
                dtype=float,
            )
        )

        f = (
            2.0
            *
            self.terminal_weight
            *
            (
                G_t.T
                @
                terminal_offset
            )
        )

        # ====================================================
        # ZMP CONSTRAINTS
        # ====================================================
        #
        # Z =
        #
        # Z_x x0
        #
        # +
        #
        # Z_u U
        #
        # where:
        #
        # Z =
        # [z_1, ..., z_N]^T
        #
        # ====================================================

        zmp_free = (
            self.zmp_state_matrix
            @
            x0
        )

        G = (
            self.zmp_control_matrix
        )

        # ----------------------------------------------------
        # lower <= Z <= upper
        #
        # lower - Z_free
        #
        # <=
        #
        # Z_u U
        #
        # <=
        #
        # upper - Z_free
        # ----------------------------------------------------

        lb = (
            lower_bounds
            -
            zmp_free
        )

        ub = (
            upper_bounds
            -
            zmp_free
        )

        return (
            H,
            f,
            G,
            lb,
            ub,
        )


    # ========================================================
    # SOLVE
    # ========================================================

    def solve(
        self,
        current_state,
        goal_state,
        lower_bounds,
        upper_bounds,
        initial_control=None,
        solver_options=None,
    ) -> MPC1DResult:
        """
        Solve the 1D LIPM-MPC optimization problem.

        Note
        ----
        lower_bounds[0] and upper_bounds[0] correspond to
        predicted state x_1, NOT current state x_0.

        Current-state ZMP feasibility should therefore be
        checked externally using the current support region
        if required.
        """

        x0 = self._validate_state(
            current_state,
            "current_state",
        )

        x_goal = self._validate_state(
            goal_state,
            "goal_state",
        )

        (
            lower_bounds,
            upper_bounds,
        ) = self._validate_bounds(
            lower_bounds,
            upper_bounds,
        )

        # ====================================================
        # BUILD QP
        # ====================================================

        (
            H,
            f,
            G,
            lb,
            ub,
        ) = self.build_qp(
            current_state=x0,
            goal_state=x_goal,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        # ====================================================
        # INITIAL GUESS
        # ====================================================

        if initial_control is None:

            U0 = np.zeros(
                self.horizon_steps,
                dtype=float,
            )

        else:

            U0 = np.asarray(
                initial_control,
                dtype=float,
            )

            if U0.shape != (
                self.horizon_steps,
            ):

                raise ValueError(
                    "initial_control has invalid shape."
                )

            if not np.all(
                np.isfinite(
                    U0
                )
            ):

                raise ValueError(
                    "initial_control must be finite."
                )

        # ====================================================
        # OBJECTIVE
        # ====================================================

        def objective(
            U,
        ):

            return float(
                0.5
                *
                U
                @
                H
                @
                U
                +
                f
                @
                U
            )


        def gradient(
            U,
        ):

            return (
                H @ U
                +
                f
            )

        # ====================================================
        # LINEAR ZMP CONSTRAINT
        # ====================================================

        linear_constraint = (
            LinearConstraint(
                G,
                lb,
                ub,
            )
        )

        # ====================================================
        # SOLVE
        # ====================================================

        result = minimize(
            objective,
            U0,
            jac=gradient,
            constraints=[
                linear_constraint
            ],
            method="SLSQP",
            options=solver_options,
        )

        if not result.success:

            raise RuntimeError(
                "MPC QP solve failed: "
                f"{result.message}"
            )

        U = np.asarray(
            result.x,
            dtype=float,
        )

        # ====================================================
        # RECONSTRUCT STATE TRAJECTORY
        # ====================================================

        X = np.zeros(
            (
                self.horizon_steps
                +
                1,
                3,
            ),
            dtype=float,
        )

        X[0] = x0

        for k in range(
            self.horizon_steps
        ):

            X[k + 1] = (
                self.model.propagate(
                    state=X[k],
                    jerk=U[k],
                )
            )

        # ====================================================
        # RECONSTRUCT ZMP TRAJECTORY
        # ====================================================
        #
        # Constraints correspond to:
        #
        # x_1, x_2, ..., x_N
        #
        # Therefore:
        #
        # zmp[k] = C_zmp x_(k+1)
        #
        # ====================================================

        zmp = np.array(
            [
                self.model.compute_zmp(
                    X[k + 1]
                )

                for k in range(
                    self.horizon_steps
                )
            ],
            dtype=float,
        )

        # ====================================================
        # ORIGINAL COST VALUE
        # ====================================================

        terminal_error = (
            X[-1]
            -
            x_goal
        )

        objective_value = (
            self.terminal_weight
            *
            float(
                terminal_error
                @
                terminal_error
            )
            +
            self.control_weight
            *
            float(
                U @ U
            )
        )

        return MPC1DResult(
            control=U,

            state=X,

            zmp=zmp,

            objective=objective_value,

            success=bool(
                result.success
            ),

            message=str(
                result.message
            ),

            iterations=int(
                getattr(
                    result,
                    "nit",
                    0,
                )
            ),
        )