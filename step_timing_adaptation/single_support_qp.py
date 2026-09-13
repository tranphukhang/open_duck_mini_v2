# step_timing_adaptation/single_support_qp.py

from __future__ import annotations

from dataclasses import dataclass
import time

import casadi as ca
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SingleSupportHQPConfig:

    friction_coefficient: float = 0.6

    # Numerical regularization in reduced QPs.
    numerical_regularization: float = 1.0e-4

    # SVD / pseudoinverse tolerance.
    svd_tolerance: float = 1.0e-9

    # Physical-constraint / hierarchy validation tolerance.
    constraint_tolerance: float = 1.0e-6

    solver_name: str = "qrqp"


# ============================================================
# SOLUTION
# ============================================================

@dataclass(frozen=True)
class SingleSupportHQPSolution:

    qacc: np.ndarray

    stance_wrench: np.ndarray

    torque: np.ndarray

    rank2_residual: float
    rank3_residual: float
    rank4_residual: float
    rank5_residual: float

    rank5_used: bool

    max_constraint_violation: float

    solve_time_rank2: float
    solve_time_rank3: float
    solve_time_rank4: float
    solve_time_rank5: float


# ============================================================
# HIERARCHICAL INVERSE DYNAMICS
# ============================================================

class SingleSupportHierarchicalInverseDynamics:
    """
    Single-support hierarchical inverse dynamics.

    Decision variable:

        y = [qddot(nv), lambda_stance(6)]

    Hierarchy:

        Rank 1:
            floating-base Newton-Euler
            actuator torque limits
            stance unilateral-contact constraint
            stance friction feasibility
            stance CoP feasibility

        Rank 2:
            stance-foot 6D acceleration constraint
            CoM vertical acceleration task

        Rank 3:
            swing-foot translational acceleration task

        Rank 4:
            actuated-joint posture task

        Rank 5:
            stance-wrench regularization

    The swing-foot orientation is intentionally NOT controlled
    here. For this project, the WBC is only the execution layer;
    the main contribution remains step-location / step-timing
    adaptation.
    """

    def __init__(
        self,
        nv: int,
        nu: int,
        actuated_dof_indices,
        config: SingleSupportHQPConfig,
    ) -> None:

        self.nv = int(nv)
        self.nu = int(nu)

        self.config = config

        self.actuated_dof_indices = np.asarray(
            actuated_dof_indices,
            dtype=int,
        )

        if self.actuated_dof_indices.shape != (self.nu,):

            raise ValueError(
                "actuated_dof_indices has invalid shape."
            )

        actuated_set = set(
            self.actuated_dof_indices.tolist()
        )

        self.unactuated_dof_indices = np.asarray(
            [
                index
                for index in range(self.nv)
                if index not in actuated_set
            ],
            dtype=int,
        )

        if self.unactuated_dof_indices.shape != (6,):

            raise RuntimeError(
                "Controller assumes a 6-DoF floating base."
            )

        # ----------------------------------------------------
        # Decision-vector layout
        # ----------------------------------------------------

        self.qacc_slice = slice(
            0,
            self.nv,
        )

        self.stance_wrench_start = (
            self.nv
        )

        self.stance_wrench_slice = slice(
            self.stance_wrench_start,
            self.stance_wrench_start + 6,
        )

        self.nvar = (
            self.nv
            +
            6
        )

        self._solver_cache = {}


    # ========================================================
    # LINEAR-ALGEBRA HELPERS
    # ========================================================

    def _nullspace(
        self,
        A,
    ) -> np.ndarray:

        A = np.asarray(
            A,
            dtype=float,
        )

        if A.ndim != 2:

            raise ValueError(
                "Null-space input must be a matrix."
            )

        if A.shape[0] == 0:

            return np.eye(
                A.shape[1],
                dtype=float,
            )

        _, singular_values, Vh = np.linalg.svd(
            A,
            full_matrices=True,
        )

        if singular_values.size == 0:

            rank = 0

        else:

            numerical_tolerance = (
                max(A.shape)
                *
                np.max(singular_values)
                *
                np.finfo(float).eps
            )

            tolerance = max(
                float(
                    self.config.svd_tolerance
                ),
                numerical_tolerance,
            )

            rank = int(
                np.sum(
                    singular_values
                    >
                    tolerance
                )
            )

        return (
            Vh[
                rank:,
                :
            ]
            .T
            .copy()
        )


    def _particular_solution(
        self,
        B,
        desired,
    ) -> np.ndarray:

        B = np.asarray(
            B,
            dtype=float,
        )

        desired = np.asarray(
            desired,
            dtype=float,
        )

        y = (
            np.linalg.pinv(
                B,
                rcond=(
                    self.config.svd_tolerance
                ),
            )
            @
            desired
        )

        residual = (
            B
            @
            y
            -
            desired
        )

        residual_inf = float(
            np.linalg.norm(
                residual,
                ord=np.inf,
            )
        )

        if (
            residual_inf
            >
            self.config.constraint_tolerance
        ):

            raise RuntimeError(
                "Could not construct Rank-1 "
                "particular solution.\n"
                f"residual = {residual_inf:.6e}"
            )

        return y


    # ========================================================
    # CASADI QP
    # ========================================================

    def _get_solver(
        self,
        name,
        number_variables,
        number_constraints,
    ):

        key = (
            str(name),
            int(number_variables),
            int(number_constraints),
        )

        if key in self._solver_cache:

            return self._solver_cache[
                key
            ]

        qp_structure = {
            "h": ca.Sparsity.dense(
                number_variables,
                number_variables,
            ),

            "a": ca.Sparsity.dense(
                number_constraints,
                number_variables,
            ),
        }

        options = {
            "print_header": False,
            "print_iter": False,
            "print_info": False,

            "error_on_fail": False,

            "constr_viol_tol": 1.0e-9,
            "dual_inf_tol": 1.0e-9,

            "max_iter": 1000,
        }

        solver = ca.conic(
            (
                f"{name}_"
                f"{number_variables}_"
                f"{number_constraints}"
            ),
            self.config.solver_name,
            qp_structure,
            options,
        )

        self._solver_cache[
            key
        ] = solver

        return solver


    def _build_reduced_objective(
        self,
        y_base,
        Z,
        B,
        desired,
    ):

        y_base = np.asarray(
            y_base,
            dtype=float,
        )

        Z = np.asarray(
            Z,
            dtype=float,
        )

        B = np.asarray(
            B,
            dtype=float,
        )

        desired = np.asarray(
            desired,
            dtype=float,
        )

        C = (
            B
            @
            Z
        )

        error_target = (
            desired
            -
            B
            @
            y_base
        )

        epsilon = float(
            self.config.numerical_regularization
        )

        H = (
            2.0
            *
            (
                C.T
                @
                C
                +
                epsilon
                *
                (
                    Z.T
                    @
                    Z
                )
            )
        )

        g = (
            -2.0
            *
            C.T
            @
            error_target
            +
            2.0
            *
            epsilon
            *
            Z.T
            @
            y_base
        )

        H = (
            0.5
            *
            (
                H
                +
                H.T
            )
        )

        return (
            H,
            g,
        )


    def _reduce_constraints(
        self,
        A,
        lower,
        upper,
        y_base,
        Z,
    ):

        A = np.asarray(
            A,
            dtype=float,
        )

        lower = np.asarray(
            lower,
            dtype=float,
        )

        upper = np.asarray(
            upper,
            dtype=float,
        )

        y_base = np.asarray(
            y_base,
            dtype=float,
        )

        Z = np.asarray(
            Z,
            dtype=float,
        )

        A_reduced = (
            A
            @
            Z
        )

        Ay_base = (
            A
            @
            y_base
        )

        lower_reduced = (
            lower
            -
            Ay_base
        )

        upper_reduced = (
            upper
            -
            Ay_base
        )

        keep_rows = []

        zero_row_tolerance = 1.0e-12

        for row_index in range(
            A_reduced.shape[0]
        ):

            row_norm = float(
                np.linalg.norm(
                    A_reduced[
                        row_index,
                        :
                    ]
                )
            )

            if row_norm > zero_row_tolerance:

                keep_rows.append(
                    row_index
                )

                continue

            lower_value = (
                lower_reduced[
                    row_index
                ]
            )

            upper_value = (
                upper_reduced[
                    row_index
                ]
            )

            if (
                np.isfinite(
                    lower_value
                )
                and
                0.0
                <
                lower_value
                -
                self.config.constraint_tolerance
            ):

                raise RuntimeError(
                    "Locked higher-priority solution "
                    "violates a physical lower bound."
                )

            if (
                np.isfinite(
                    upper_value
                )
                and
                0.0
                >
                upper_value
                +
                self.config.constraint_tolerance
            ):

                raise RuntimeError(
                    "Locked higher-priority solution "
                    "violates a physical upper bound."
                )

        if keep_rows:

            indices = np.asarray(
                keep_rows,
                dtype=int,
            )

            A_reduced = (
                A_reduced[
                    indices,
                    :
                ]
            )

            lower_reduced = (
                lower_reduced[
                    indices
                ]
            )

            upper_reduced = (
                upper_reduced[
                    indices
                ]
            )

        else:

            A_reduced = np.zeros(
                (
                    0,
                    Z.shape[1],
                ),
                dtype=float,
            )

            lower_reduced = np.zeros(
                0,
                dtype=float,
            )

            upper_reduced = np.zeros(
                0,
                dtype=float,
            )

        return (
            A_reduced,
            lower_reduced,
            upper_reduced,
        )


    def _solve_reduced_qp(
        self,
        name,
        y_base,
        Z,
        B,
        desired,
        A_ineq,
        lower_ineq,
        upper_ineq,
    ):

        number_variables = int(
            Z.shape[1]
        )

        if number_variables == 0:

            return (
                y_base.copy(),
                0.0,
            )

        H, g = (
            self._build_reduced_objective(
                y_base=y_base,
                Z=Z,
                B=B,
                desired=desired,
            )
        )

        (
            A_reduced,
            lower_reduced,
            upper_reduced,
        ) = (
            self._reduce_constraints(
                A=A_ineq,
                lower=lower_ineq,
                upper=upper_ineq,
                y_base=y_base,
                Z=Z,
            )
        )

        # QRQP expects at least one row.

        if A_reduced.shape[0] == 0:

            A_reduced = np.zeros(
                (
                    1,
                    number_variables,
                ),
                dtype=float,
            )

            lower_reduced = np.array(
                [0.0],
                dtype=float,
            )

            upper_reduced = np.array(
                [0.0],
                dtype=float,
            )

        solver = self._get_solver(
            name=name,
            number_variables=(
                number_variables
            ),
            number_constraints=(
                A_reduced.shape[0]
            ),
        )

        result = solver(
            h=ca.DM(
                H
            ),

            g=ca.DM(
                g
            ),

            a=ca.DM(
                A_reduced
            ),

            lba=ca.DM(
                lower_reduced
            ),

            uba=ca.DM(
                upper_reduced
            ),

            lbx=ca.DM(
                np.full(
                    number_variables,
                    -np.inf,
                    dtype=float,
                )
            ),

            ubx=ca.DM(
                np.full(
                    number_variables,
                    +np.inf,
                    dtype=float,
                )
            ),
        )

        stats = solver.stats()

        success = bool(
            stats.get(
                "success",
                False,
            )
        )

        if not success:

            raise RuntimeError(
                f"{name} solver failed.\n"
                f"status = "
                f"{stats.get('return_status', 'unknown')}"
            )

        u = np.asarray(
            result[
                "x"
            ]
        ).reshape(-1)

        if not np.all(
            np.isfinite(
                u
            )
        ):

            raise RuntimeError(
                f"{name} returned NaN/Inf."
            )

        y = (
            y_base
            +
            Z
            @
            u
        )

        # ----------------------------------------------------
        # Validate original physical inequalities.
        # ----------------------------------------------------

        values = (
            A_ineq
            @
            y
        )

        lower_violation = np.where(
            np.isfinite(
                lower_ineq
            ),
            np.maximum(
                lower_ineq
                -
                values,
                0.0,
            ),
            0.0,
        )

        upper_violation = np.where(
            np.isfinite(
                upper_ineq
            ),
            np.maximum(
                values
                -
                upper_ineq,
                0.0,
            ),
            0.0,
        )

        max_violation = float(
            max(
                np.max(
                    lower_violation
                ),
                np.max(
                    upper_violation
                ),
            )
        )

        if (
            max_violation
            >
            self.config.constraint_tolerance
        ):

            raise RuntimeError(
                f"{name} physical-constraint violation.\n"
                f"max violation = "
                f"{max_violation:.6e}"
            )

        return (
            y,
            max_violation,
        )


    # ========================================================
    # PHYSICAL CONSTRAINTS
    # ========================================================

    def _append_friction_constraints(
        self,
        rows,
        lower,
        upper,
    ) -> None:

        mu = float(
            self.config.friction_coefficient
        )

        fx = (
            self.stance_wrench_start
            +
            0
        )

        fy = (
            self.stance_wrench_start
            +
            1
        )

        fz = (
            self.stance_wrench_start
            +
            2
        )

        # Conservative friction pyramid:
        #
        #     |Fx| + |Fy| <= mu Fz

        for sx, sy in (
            (+1.0, +1.0),
            (+1.0, -1.0),
            (-1.0, +1.0),
            (-1.0, -1.0),
        ):

            row = np.zeros(
                self.nvar,
                dtype=float,
            )

            row[
                fx
            ] = sx

            row[
                fy
            ] = sy

            row[
                fz
            ] = -mu

            rows.append(
                row
            )

            lower.append(
                -np.inf
            )

            upper.append(
                0.0
            )

        # Fz >= 0.

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fz
        ] = 1.0

        rows.append(
            row
        )

        lower.append(
            0.0
        )

        upper.append(
            np.inf
        )


    def _append_cop_constraints(
        self,
        rows,
        lower,
        upper,
        support_bounds,
        contact_height,
    ) -> None:

        bounds = np.asarray(
            support_bounds,
            dtype=float,
        )

        if bounds.shape != (4,):

            raise ValueError(
                "support_bounds must be "
                "[xmin, xmax, ymin, ymax]."
            )

        (
            xmin,
            xmax,
            ymin,
            ymax,
        ) = bounds

        h = float(
            contact_height
        )

        fx = (
            self.stance_wrench_start
            +
            0
        )

        fy = (
            self.stance_wrench_start
            +
            1
        )

        fz = (
            self.stance_wrench_start
            +
            2
        )

        mx = (
            self.stance_wrench_start
            +
            3
        )

        my = (
            self.stance_wrench_start
            +
            4
        )

        # x_cop >= xmin:
        #
        #     My + h Fx + xmin Fz <= 0

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fx
        ] = h

        row[
            fz
        ] = xmin

        row[
            my
        ] = 1.0

        rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )

        # x_cop <= xmax:
        #
        #     -My - h Fx - xmax Fz <= 0

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fx
        ] = -h

        row[
            fz
        ] = -xmax

        row[
            my
        ] = -1.0

        rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )

        # y_cop >= ymin:
        #
        #     -Mx + h Fy + ymin Fz <= 0

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fy
        ] = h

        row[
            fz
        ] = ymin

        row[
            mx
        ] = -1.0

        rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )

        # y_cop <= ymax:
        #
        #     Mx - h Fy - ymax Fz <= 0

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fy
        ] = -h

        row[
            fz
        ] = -ymax

        row[
            mx
        ] = 1.0

        rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )


    # ========================================================
    # RANK 1 — RIGID-BODY DYNAMICS + PHYSICAL FEASIBILITY
    # ========================================================

    def _build_physical_model(
        self,
        mass_matrix,
        effective_bias,
        selection_matrix,
        stance_jacobian,
        support_bounds,
        contact_height,
        torque_lower,
        torque_upper,
    ):

        M = np.asarray(
            mass_matrix,
            dtype=float,
        )

        h = np.asarray(
            effective_bias,
            dtype=float,
        )

        S = np.asarray(
            selection_matrix,
            dtype=float,
        )

        J = np.asarray(
            stance_jacobian,
            dtype=float,
        )

        if M.shape != (
            self.nv,
            self.nv,
        ):

            raise ValueError(
                "mass_matrix has invalid shape."
            )

        if J.shape != (
            6,
            self.nv,
        ):

            raise ValueError(
                "stance_jacobian has invalid shape."
            )

        # ----------------------------------------------------
        # Floating-base Newton-Euler equations:
        #
        #     M qdd + h = S.T tau + J.T lambda
        #
        # Use only the six unactuated rows.
        # ----------------------------------------------------

        B1 = np.zeros(
            (
                6,
                self.nvar,
            ),
            dtype=float,
        )

        base_indices = (
            self.unactuated_dof_indices
        )

        B1[
            :,
            self.qacc_slice
        ] = (
            M[
                base_indices,
                :
            ]
        )

        B1[
            :,
            self.stance_wrench_slice
        ] = (
            -J[
                :,
                base_indices
            ].T
        )

        d1 = (
            -h[
                base_indices
            ]
        )

        # ----------------------------------------------------
        # Recover actuator torque:
        #
        #     tau = A_tau y + b_tau
        # ----------------------------------------------------

        actuated_indices = (
            self.actuated_dof_indices
        )

        E = (
            S.T[
                actuated_indices,
                :
            ]
        )

        if E.shape != (
            self.nu,
            self.nu,
        ):

            raise RuntimeError(
                "Unexpected actuated selection-matrix shape."
            )

        if (
            np.linalg.matrix_rank(
                E
            )
            !=
            self.nu
        ):

            raise RuntimeError(
                "Actuated selection matrix is singular."
            )

        E_inv = np.linalg.inv(
            E
        )

        A_tau = np.zeros(
            (
                self.nu,
                self.nvar,
            ),
            dtype=float,
        )

        A_tau[
            :,
            self.qacc_slice
        ] = (
            M[
                actuated_indices,
                :
            ]
        )

        A_tau[
            :,
            self.stance_wrench_slice
        ] = (
            -J[
                :,
                actuated_indices
            ].T
        )

        A_tau = (
            E_inv
            @
            A_tau
        )

        b_tau = (
            E_inv
            @
            h[
                actuated_indices
            ]
        )

        tau_lower = np.asarray(
            torque_lower,
            dtype=float,
        )

        tau_upper = np.asarray(
            torque_upper,
            dtype=float,
        )

        if (
            tau_lower.shape != (self.nu,)
            or
            tau_upper.shape != (self.nu,)
        ):

            raise ValueError(
                "Torque limits have invalid shape."
            )

        # ----------------------------------------------------
        # Inequalities
        # ----------------------------------------------------

        rows = []
        lower = []
        upper = []

        for actuator_id in range(
            self.nu
        ):

            rows.append(
                A_tau[
                    actuator_id,
                    :
                ]
            )

            lower.append(
                tau_lower[
                    actuator_id
                ]
                -
                b_tau[
                    actuator_id
                ]
            )

            upper.append(
                tau_upper[
                    actuator_id
                ]
                -
                b_tau[
                    actuator_id
                ]
            )

        self._append_friction_constraints(
            rows=rows,
            lower=lower,
            upper=upper,
        )

        self._append_cop_constraints(
            rows=rows,
            lower=lower,
            upper=upper,
            support_bounds=(
                support_bounds
            ),
            contact_height=(
                contact_height
            ),
        )

        A_ineq = np.vstack(
            rows
        )

        lower_ineq = np.asarray(
            lower,
            dtype=float,
        )

        upper_ineq = np.asarray(
            upper,
            dtype=float,
        )

        return (
            B1,
            d1,

            A_ineq,
            lower_ineq,
            upper_ineq,

            A_tau,
            b_tau,
        )


    # ========================================================
    # RANK 2 — STANCE FOOT + COM HEIGHT
    # ========================================================

    def _build_rank2(
        self,
        stance_jacobian,
        stance_jdot_v,
        com_jacobian,
        com_jdot_v_z,
        desired_com_acceleration_z,
    ):

        J_stance = np.asarray(
            stance_jacobian,
            dtype=float,
        )

        Jdot_stance_v = np.asarray(
            stance_jdot_v,
            dtype=float,
        )

        J_com = np.asarray(
            com_jacobian,
            dtype=float,
        )

        if J_stance.shape != (
            6,
            self.nv,
        ):

            raise ValueError(
                "stance_jacobian has invalid shape."
            )

        if Jdot_stance_v.shape != (6,):

            raise ValueError(
                "stance_jdot_v has invalid shape."
            )

        if J_com.shape != (
            3,
            self.nv,
        ):

            raise ValueError(
                "com_jacobian has invalid shape."
            )

        B2 = np.zeros(
            (
                7,
                self.nvar,
            ),
            dtype=float,
        )

        d2 = np.zeros(
            7,
            dtype=float,
        )

        # 6D stance-foot acceleration = 0.

        B2[
            0:6,
            self.qacc_slice
        ] = (
            J_stance
        )

        d2[
            0:6
        ] = (
            -Jdot_stance_v
        )

        # CoM vertical task.

        B2[
            6,
            self.qacc_slice
        ] = (
            J_com[
                2,
                :
            ]
        )

        d2[
            6
        ] = (
            float(
                desired_com_acceleration_z
            )
            -
            float(
                com_jdot_v_z
            )
        )

        return (
            B2,
            d2,
        )


    # ========================================================
    # RANK 3 — SWING-FOOT TRANSLATION
    # ========================================================

    def _build_rank3(
        self,
        swing_jacobian,
        swing_jdot_v,
        desired_swing_linear_acceleration,
    ):

        J_swing = np.asarray(
            swing_jacobian,
            dtype=float,
        )

        Jdot_swing_v = np.asarray(
            swing_jdot_v,
            dtype=float,
        )

        desired = np.asarray(
            desired_swing_linear_acceleration,
            dtype=float,
        )

        if J_swing.shape != (
            6,
            self.nv,
        ):

            raise ValueError(
                "swing_jacobian has invalid shape."
            )

        if Jdot_swing_v.shape != (6,):

            raise ValueError(
                "swing_jdot_v has invalid shape."
            )

        if desired.shape != (3,):

            raise ValueError(
                "desired_swing_linear_acceleration "
                "must have shape (3,)."
            )

        B3 = np.zeros(
            (
                3,
                self.nvar,
            ),
            dtype=float,
        )

        B3[
            :,
            self.qacc_slice
        ] = (
            J_swing[
                0:3,
                :
            ]
        )

        d3 = (
            desired
            -
            Jdot_swing_v[
                0:3
            ]
        )

        return (
            B3,
            d3,
        )


    # ========================================================
    # RANK 4 — POSTURE
    # ========================================================

    def _build_rank4(
        self,
        desired_posture_acceleration,
    ):

        desired = np.asarray(
            desired_posture_acceleration,
            dtype=float,
        )

        if desired.shape != (
            self.nu,
        ):

            raise ValueError(
                "desired_posture_acceleration "
                "has invalid shape."
            )

        B4 = np.zeros(
            (
                self.nu,
                self.nvar,
            ),
            dtype=float,
        )

        for actuator_id, dof_index in enumerate(
            self.actuated_dof_indices
        ):

            B4[
                actuator_id,
                dof_index
            ] = 1.0

        return (
            B4,
            desired,
        )


    # ========================================================
    # RANK 5 — STANCE-WRENCH REGULARIZATION
    # ========================================================

    def _build_rank5(
        self,
        stance_wrench_reference,
    ):

        reference = np.asarray(
            stance_wrench_reference,
            dtype=float,
        )

        if reference.shape != (6,):

            raise ValueError(
                "stance_wrench_reference "
                "must have shape (6,)."
            )

        B5 = np.zeros(
            (
                6,
                self.nvar,
            ),
            dtype=float,
        )

        B5[
            :,
            self.stance_wrench_slice
        ] = np.eye(
            6
        )

        return (
            B5,
            reference,
        )


    # ========================================================
    # MAIN SOLVE
    # ========================================================

    def solve(
        self,

        mass_matrix,
        effective_bias,
        selection_matrix,

        stance_jacobian,
        stance_jdot_v,

        swing_jacobian,
        swing_jdot_v,

        com_jacobian,
        com_jdot_v_z,

        desired_com_acceleration_z,

        desired_swing_linear_acceleration,

        desired_posture_acceleration,

        stance_wrench_reference,

        stance_support_bounds,
        stance_contact_height,

        torque_lower,
        torque_upper,
    ) -> SingleSupportHQPSolution:

        # ====================================================
        # RANK 1
        # ====================================================

        (
            B1,
            d1,

            A_ineq,
            lower_ineq,
            upper_ineq,

            A_tau,
            b_tau,
        ) = (
            self._build_physical_model(
                mass_matrix=(
                    mass_matrix
                ),
                effective_bias=(
                    effective_bias
                ),
                selection_matrix=(
                    selection_matrix
                ),
                stance_jacobian=(
                    stance_jacobian
                ),
                support_bounds=(
                    stance_support_bounds
                ),
                contact_height=(
                    stance_contact_height
                ),
                torque_lower=(
                    torque_lower
                ),
                torque_upper=(
                    torque_upper
                ),
            )
        )

        y1 = (
            self._particular_solution(
                B1,
                d1,
            )
        )

        Z1 = (
            self._nullspace(
                B1
            )
        )

        # ====================================================
        # RANK 2
        # ====================================================

        B2, d2 = (
            self._build_rank2(
                stance_jacobian=(
                    stance_jacobian
                ),
                stance_jdot_v=(
                    stance_jdot_v
                ),
                com_jacobian=(
                    com_jacobian
                ),
                com_jdot_v_z=(
                    com_jdot_v_z
                ),
                desired_com_acceleration_z=(
                    desired_com_acceleration_z
                ),
            )
        )

        start = (
            time.perf_counter()
        )

        y2, violation2 = (
            self._solve_reduced_qp(
                name="single_rank2",
                y_base=y1,
                Z=Z1,
                B=B2,
                desired=d2,
                A_ineq=A_ineq,
                lower_ineq=(
                    lower_ineq
                ),
                upper_ineq=(
                    upper_ineq
                ),
            )
        )

        solve_time_rank2 = (
            time.perf_counter()
            -
            start
        )

        rank2_residual = float(
            np.linalg.norm(
                B2
                @
                y2
                -
                d2
            )
        )

        B12 = np.vstack(
            (
                B1,
                B2,
            )
        )

        Z2 = (
            self._nullspace(
                B12
            )
        )

        # ====================================================
        # RANK 3
        # ====================================================

        B3, d3 = (
            self._build_rank3(
                swing_jacobian=(
                    swing_jacobian
                ),
                swing_jdot_v=(
                    swing_jdot_v
                ),
                desired_swing_linear_acceleration=(
                    desired_swing_linear_acceleration
                ),
            )
        )

        start = (
            time.perf_counter()
        )

        y3, violation3 = (
            self._solve_reduced_qp(
                name="single_rank3",
                y_base=y2,
                Z=Z2,
                B=B3,
                desired=d3,
                A_ineq=A_ineq,
                lower_ineq=(
                    lower_ineq
                ),
                upper_ineq=(
                    upper_ineq
                ),
            )
        )

        solve_time_rank3 = (
            time.perf_counter()
            -
            start
        )

        rank3_residual = float(
            np.linalg.norm(
                B3
                @
                y3
                -
                d3
            )
        )

        B123 = np.vstack(
            (
                B1,
                B2,
                B3,
            )
        )

        Z3 = (
            self._nullspace(
                B123
            )
        )

        # ====================================================
        # RANK 4
        # ====================================================

        B4, d4 = (
            self._build_rank4(
                desired_posture_acceleration
            )
        )

        start = (
            time.perf_counter()
        )

        y4, violation4 = (
            self._solve_reduced_qp(
                name="single_rank4",
                y_base=y3,
                Z=Z3,
                B=B4,
                desired=d4,
                A_ineq=A_ineq,
                lower_ineq=(
                    lower_ineq
                ),
                upper_ineq=(
                    upper_ineq
                ),
            )
        )

        solve_time_rank4 = (
            time.perf_counter()
            -
            start
        )

        rank4_residual = float(
            np.linalg.norm(
                B4
                @
                y4
                -
                d4
            )
        )

        B1234 = np.vstack(
            (
                B1,
                B2,
                B3,
                B4,
            )
        )

        Z4 = (
            self._nullspace(
                B1234
            )
        )

        # ====================================================
        # RANK 5
        # ====================================================

        B5, d5 = (
            self._build_rank5(
                stance_wrench_reference
            )
        )

        rank5_used = False

        if Z4.shape[1] > 0:

            start = (
                time.perf_counter()
            )

            try:

                y5, violation5 = (
                    self._solve_reduced_qp(
                        name="single_rank5",
                        y_base=y4,
                        Z=Z4,
                        B=B5,
                        desired=d5,
                        A_ineq=A_ineq,
                        lower_ineq=(
                            lower_ineq
                        ),
                        upper_ineq=(
                            upper_ineq
                        ),
                    )
                )

                rank5_used = True

            except RuntimeError:

                y5 = (
                    y4.copy()
                )

                violation5 = (
                    violation4
                )

            solve_time_rank5 = (
                time.perf_counter()
                -
                start
            )

        else:

            y5 = (
                y4.copy()
            )

            violation5 = (
                violation4
            )

            solve_time_rank5 = 0.0

        rank5_residual = float(
            np.linalg.norm(
                B5
                @
                y5
                -
                d5
            )
        )

        # ====================================================
        # EXTRACT + TORQUE
        # ====================================================

        qacc = (
            y5[
                self.qacc_slice
            ].copy()
        )

        stance_wrench = (
            y5[
                self.stance_wrench_slice
            ].copy()
        )

        torque = (
            A_tau
            @
            y5
            +
            b_tau
        )

        tau_lower = np.asarray(
            torque_lower,
            dtype=float,
        )

        tau_upper = np.asarray(
            torque_upper,
            dtype=float,
        )

        torque_violation = max(
            float(
                np.max(
                    np.maximum(
                        tau_lower
                        -
                        torque,
                        0.0,
                    )
                )
            ),
            float(
                np.max(
                    np.maximum(
                        torque
                        -
                        tau_upper,
                        0.0,
                    )
                )
            ),
        )

        if (
            torque_violation
            >
            self.config.constraint_tolerance
        ):

            raise RuntimeError(
                "Recovered torque violates limits.\n"
                f"max violation = "
                f"{torque_violation:.6e}"
            )

        # ====================================================
        # STRICT-HIERARCHY CHECKS
        # ====================================================

        rank1_lock_error = float(
            np.linalg.norm(
                B1
                @
                y5
                -
                d1,
                ord=np.inf,
            )
        )

        rank2_lock_error = float(
            np.linalg.norm(
                B2
                @
                y5
                -
                B2
                @
                y2,
                ord=np.inf,
            )
        )

        rank3_lock_error = float(
            np.linalg.norm(
                B3
                @
                y5
                -
                B3
                @
                y3,
                ord=np.inf,
            )
        )

        rank4_lock_error = float(
            np.linalg.norm(
                B4
                @
                y5
                -
                B4
                @
                y4,
                ord=np.inf,
            )
        )

        for name, error in (
            (
                "Rank-1",
                rank1_lock_error,
            ),
            (
                "Rank-2",
                rank2_lock_error,
            ),
            (
                "Rank-3",
                rank3_lock_error,
            ),
            (
                "Rank-4",
                rank4_lock_error,
            ),
        ):

            if (
                error
                >
                self.config.constraint_tolerance
            ):

                raise RuntimeError(
                    f"{name} hierarchy violation.\n"
                    f"error = {error:.6e}"
                )

        max_violation = max(
            violation2,
            violation3,
            violation4,
            violation5,
            torque_violation,
            rank1_lock_error,
            rank2_lock_error,
            rank3_lock_error,
            rank4_lock_error,
        )

        return SingleSupportHQPSolution(
            qacc=(
                qacc
            ),
            stance_wrench=(
                stance_wrench
            ),
            torque=(
                torque
            ),
            rank2_residual=(
                rank2_residual
            ),
            rank3_residual=(
                rank3_residual
            ),
            rank4_residual=(
                rank4_residual
            ),
            rank5_residual=(
                rank5_residual
            ),
            rank5_used=(
                rank5_used
            ),
            max_constraint_violation=(
                max_violation
            ),
            solve_time_rank2=(
                solve_time_rank2
            ),
            solve_time_rank3=(
                solve_time_rank3
            ),
            solve_time_rank4=(
                solve_time_rank4
            ),
            solve_time_rank5=(
                solve_time_rank5
            ),
        )
