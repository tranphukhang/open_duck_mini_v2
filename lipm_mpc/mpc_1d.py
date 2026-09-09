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

    # Optimal jerk sequence
    control: np.ndarray

    # Predicted states:
    #
    # state[0] = current state
    # state[N] = terminal state
    state: np.ndarray

    # Predicted ZMP values corresponding to:
    #
    # state[0], ..., state[N-1]
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

        This is the command applied by receding-horizon MPC.
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

    The same class is used independently for:

        x-axis:
            [x_G, xdot_G, xddot_G]

        y-axis:
            [y_G, ydot_G, yddot_G]

    State:
        x_k = [p_G, v_G, a_G]^T

    Control:
        u_k = jerk

    Dynamics:
        x_{k+1} = A x_k + B u_k

    ZMP:
        p_Z = C_zmp x

    Cost:
        J =
            w_terminal * ||x_N - x_goal||^2
            +
            w_control * sum(u_k^2)

    Constraint:
        lower_k <= p_Z,k <= upper_k
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
        # BUILD CONSTANT PREVIEW MATRICES
        # ====================================================

        self._build_prediction_matrices()


    # ========================================================
    # PREDICTION MATRICES
    # ========================================================

    def _build_prediction_matrices(
        self,
    ) -> None:
        """
        Build matrices that map:

            current state
            +
            future jerk sequence

        to:

            terminal state
            +
            ZMP trajectory.
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

        # ----------------------------------------------------
        # TERMINAL STATE
        #
        # x_N =
        #
        # F x_0 + G U
        #
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ZMP PREVIEW
        #
        # Z =
        #
        # Z_x x_0
        # +
        # Z_u U
        #
        #
        # Important:
        #
        # ZMP constraints are applied to:
        #
        # x_0, x_1, ..., x_{N-1}
        #
        # Terminal state x_N is used in the cost.
        # ----------------------------------------------------

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

            # Free response:
            #
            # C A^k x0

            self.zmp_state_matrix[
                k,
                :,
            ] = (
                self.C_zmp
                @
                A_powers[k]
            )

            # Controlled response

            for j in range(k):

                self.zmp_control_matrix[
                    k,
                    j,
                ] = (
                    self.C_zmp
                    @
                    (
                        A_powers[
                            k - 1 - j
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
            np.isfinite(state)
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
        Build the quadratic program:

            min
                1/2 U^T H U + f^T U

            subject to

                lb <= G U <= ub

        where:

            U =
                [u_0, ..., u_{N-1}]^T
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

        # ----------------------------------------------------
        # TERMINAL STATE
        #
        # x_N =
        # F x0 + G_t U
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # COST
        #
        # J =
        #
        # wt ||F x0 + G_t U - x_goal||^2
        # +
        # wu ||U||^2
        #
        #
        # Standard QP:
        #
        # 1/2 U^T H U + f^T U
        # ----------------------------------------------------

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
                self.horizon_steps
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

        # ----------------------------------------------------
        # ZMP
        #
        # Z =
        # Z_x x0 + Z_u U
        # ----------------------------------------------------

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
        # <=
        # Z_u U
        # <=
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
        Solve the 1D LIPM-MPC problem.
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

        # ----------------------------------------------------
        # CURRENT ZMP
        #
        # The first support constraint corresponds to x0.
        #
        # u0 cannot modify x0 instantaneously.
        # Therefore, if the current ZMP is already outside
        # the first support interval, the QP is infeasible.
        # ----------------------------------------------------

        current_zmp = (
            self.model.compute_zmp(
                x0
            )
        )

        if not (
            lower_bounds[0]
            <=
            current_zmp
            <=
            upper_bounds[0]
        ):
            raise ValueError(
                "Current ZMP lies outside the first "
                "support bound. The first MPC control "
                "cannot modify the current state."
            )

        # ----------------------------------------------------
        # BUILD QP
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # INITIAL GUESS
        # ----------------------------------------------------

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
                np.isfinite(U0)
            ):
                raise ValueError(
                    "initial_control must be finite."
                )

        # ----------------------------------------------------
        # OBJECTIVE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LINEAR ZMP CONSTRAINT
        # ----------------------------------------------------

        linear_constraint = (
            LinearConstraint(
                G,
                lb,
                ub,
            )
        )

        # ----------------------------------------------------
        # SOLVE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RECONSTRUCT STATE TRAJECTORY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RECONSTRUCT ZMP TRAJECTORY
        #
        # Constraints correspond to:
        #
        # x0 ... x_{N-1}
        # ----------------------------------------------------

        zmp = np.array(
            [
                self.model.compute_zmp(
                    X[k]
                )
                for k in range(
                    self.horizon_steps
                )
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # ORIGINAL COST VALUE
        # ----------------------------------------------------

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