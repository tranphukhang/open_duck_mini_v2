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

    control: np.ndarray

    # state[0] = x_0
    # ...
    # state[N] = x_N
    state: np.ndarray

    # zmp[0] = ZMP(x_1)
    # ...
    # zmp[N-1] = ZMP(x_N)
    zmp: np.ndarray

    objective: float

    success: bool

    message: str

    iterations: int


    @property
    def first_control(
        self,
    ) -> float:

        return float(
            self.control[0]
        )


# ============================================================
# LIPM MPC 1D
# ============================================================

class LIPMMPC1D:
    """
    Generic 1D LIPM-MPC.

    State:
        x_k = [p_G, v_G, a_G]^T

    Control:
        u_k = jerk

    Dynamics:
        x_(k+1) = A x_k + B u_k

    ZMP:
        p_Z = C_zmp x

    Terminal cost:
        J_terminal =
            (x_N - x_goal)^T
            Q_f
            (x_N - x_goal)

    where:

        Q_f =
            diag(q_p, q_v, q_a)

    Control cost:
        J_control =
            w_u * U^T U

    ZMP constraints are imposed on:

        x_1, ..., x_N.
    """

    def __init__(
        self,
        model: LIPMModel1D,
        horizon_steps: int,
        terminal_weights,
        control_weight: float,
    ) -> None:

        # ====================================================
        # VALIDATE BASIC PARAMETERS
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

        if control_weight <= 0.0:

            raise ValueError(
                "control_weight must be positive."
            )

        # ====================================================
        # TERMINAL WEIGHTS
        # ====================================================

        terminal_weights = np.asarray(
            terminal_weights,
            dtype=float,
        )

        if terminal_weights.shape != (
            3,
        ):

            raise ValueError(
                "terminal_weights must have shape (3,)."
            )

        if not np.all(
            np.isfinite(
                terminal_weights
            )
        ):

            raise ValueError(
                "terminal_weights must contain finite values."
            )

        if np.any(
            terminal_weights
            <
            0.0
        ):

            raise ValueError(
                "terminal_weights cannot be negative."
            )

        # ====================================================
        # STORE PARAMETERS
        # ====================================================

        self.model = model

        self.horizon_steps = int(
            horizon_steps
        )

        self.terminal_weights = (
            terminal_weights.copy()
        )

        self.Q_terminal = np.diag(
            self.terminal_weights
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
        ).reshape(
            3
        )

        self.C_zmp = np.asarray(
            model.C_zmp,
            dtype=float,
        ).reshape(
            3
        )

        # ====================================================
        # CONSTANT PREDICTION MATRICES
        # ====================================================

        self._build_prediction_matrices()


    # ========================================================
    # PREDICTION MATRICES
    # ========================================================

    def _build_prediction_matrices(
        self,
    ) -> None:

        N = (
            self.horizon_steps
        )

        # ----------------------------------------------------
        # A^k
        # ----------------------------------------------------

        A_powers = [
            np.eye(
                3,
                dtype=float,
            )
        ]

        for _ in range(
            N
        ):

            A_powers.append(
                A_powers[-1]
                @
                self.A
            )

        # ====================================================
        # TERMINAL STATE
        #
        # x_N =
        #
        # F x_0
        # +
        # G_t U
        # ====================================================

        self.terminal_state_matrix = (
            A_powers[N].copy()
        )

        self.terminal_control_matrix = np.zeros(
            (
                3,
                N,
            ),
            dtype=float,
        )

        for j in range(
            N
        ):

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
        #
        # Z =
        # [z_1 ... z_N]^T
        #
        # Z =
        # Z_x x_0
        # +
        # Z_u U
        # ====================================================

        self.zmp_state_matrix = np.zeros(
            (
                N,
                3,
            ),
            dtype=float,
        )

        self.zmp_control_matrix = np.zeros(
            (
                N,
                N,
            ),
            dtype=float,
        )

        for k in range(
            N
        ):

            # ------------------------------------------------
            # Free response
            #
            # z_(k+1) =
            # C A^(k+1) x_0
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
            # Controlled response
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
    # STATE VALIDATION
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

        if state.shape != (
            3,
        ):

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


    # ========================================================
    # BOUNDS VALIDATION
    # ========================================================

    def _validate_bounds(
        self,
        lower_bounds,
        upper_bounds,
    ):

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

        if lower_bounds.shape != (
            expected_shape
        ):

            raise ValueError(
                "lower_bounds must have shape "
                f"{expected_shape}."
            )

        if upper_bounds.shape != (
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
        #
        # x_N =
        # F x0 + G_t U
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

        Q = (
            self.Q_terminal
        )

        # ====================================================
        # COST
        #
        # J =
        #
        # (F x0 + G_t U - x_goal)^T
        # Q
        # (F x0 + G_t U - x_goal)
        #
        # +
        #
        # w_u U^T U
        #
        # Standard QP:
        #
        # 1/2 U^T H U + f^T U
        # ====================================================

        H = 2.0 * (
            G_t.T
            @
            Q
            @
            G_t
            +
            self.control_weight
            *
            np.eye(
                self.horizon_steps,
                dtype=float,
            )
        )

        f = 2.0 * (
            G_t.T
            @
            Q
            @
            terminal_offset
        )

        # ====================================================
        # ZMP CONSTRAINT
        # ====================================================

        zmp_free = (
            self.zmp_state_matrix
            @
            x0
        )

        G = (
            self.zmp_control_matrix
        )

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
            current_state=(
                x0
            ),

            goal_state=(
                x_goal
            ),

            lower_bounds=(
                lower_bounds
            ),

            upper_bounds=(
                upper_bounds
            ),
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

            jac=(
                gradient
            ),

            constraints=[
                linear_constraint
            ],

            method=(
                "SLSQP"
            ),

            options=(
                solver_options
            ),
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

        X[0] = (
            x0
        )

        for k in range(
            self.horizon_steps
        ):

            X[k + 1] = (
                self.model.propagate(
                    state=(
                        X[k]
                    ),

                    jerk=(
                        U[k]
                    ),
                )
            )

        # ====================================================
        # RECONSTRUCT ZMP
        #
        # zmp[k] corresponds to X[k+1]
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

        terminal_cost = float(
            terminal_error
            @
            self.Q_terminal
            @
            terminal_error
        )

        control_cost = (
            self.control_weight
            *
            float(
                U
                @
                U
            )
        )

        objective_value = (
            terminal_cost
            +
            control_cost
        )

        return MPC1DResult(
            control=(
                U
            ),

            state=(
                X
            ),

            zmp=(
                zmp
            ),

            objective=(
                objective_value
            ),

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