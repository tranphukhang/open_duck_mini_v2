# step_timing_adaptation/whole_body_qp.py

from __future__ import annotations

from dataclasses import dataclass
import time

import casadi as ca
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DoubleSupportHQPConfig:

    friction_coefficient: float = 0.6

    # Numerical regularization used in reduced QPs.
    numerical_regularization: float = 1.0e-4

    # SVD / pseudoinverse tolerance.
    svd_tolerance: float = 1.0e-9

    # Validation tolerance.
    constraint_tolerance: float = 1.0e-6

    solver_name: str = "qrqp"


# ============================================================
# SOLUTION
# ============================================================

@dataclass
class DoubleSupportHQPSolution:

    qacc: np.ndarray

    left_wrench: np.ndarray
    right_wrench: np.ndarray

    torque: np.ndarray

    rank2_residual: float
    rank4_residual: float
    rank5_residual: float

    rank5_used: bool

    max_constraint_violation: float

    solve_time_rank2: float
    solve_time_rank4: float
    solve_time_rank5: float


# ============================================================
# HIERARCHICAL INVERSE DYNAMICS
# ============================================================

class WholeBodyHierarchicalInverseDynamics:
    """
    Double-support hierarchical inverse dynamics.

    Decision variable:

        y =
        [
            qddot       nv
            lambda_L      6
            lambda_R      6
        ]

    Torque is recovered from actuated rigid-body dynamics.

    Hierarchy:

        Rank 1:
            floating-base Newton-Euler
            torque limits
            unilateral contact
            friction feasibility
            CoP feasibility

        Rank 2:
            left stance foot 6D
            right stance foot 6D
            CoM height

        Rank 3:
            none in double support

        Rank 4:
            actuated-joint posture

        Rank 5:
            contact-wrench regularization

    No horizontal CoM control.
    No CoP tracking.
    No ZMP tracking.
    """

    def __init__(
        self,
        nv: int,
        nu: int,
        actuated_dof_indices,
        config: DoubleSupportHQPConfig,
    ) -> None:

        self.nv = int(nv)
        self.nu = int(nu)

        self.config = config

        self.actuated_dof_indices = np.asarray(
            actuated_dof_indices,
            dtype=int,
        )

        if self.actuated_dof_indices.shape != (
            self.nu,
        ):

            raise ValueError(
                "actuated_dof_indices has invalid shape."
            )

        actuated_set = set(
            self.actuated_dof_indices.tolist()
        )

        self.unactuated_dof_indices = np.asarray(
            [
                i
                for i in range(self.nv)
                if i not in actuated_set
            ],
            dtype=int,
        )

        if self.unactuated_dof_indices.shape != (
            6,
        ):

            raise RuntimeError(
                "Controller assumes a 6-DoF floating base."
            )

        # ====================================================
        # VARIABLE LAYOUT
        # ====================================================

        self.qacc_slice = slice(
            0,
            self.nv,
        )

        self.left_wrench_start = (
            self.nv
        )

        self.right_wrench_start = (
            self.left_wrench_start
            +
            6
        )

        self.left_wrench_slice = slice(
            self.left_wrench_start,
            self.right_wrench_start,
        )

        self.right_wrench_slice = slice(
            self.right_wrench_start,
            self.right_wrench_start + 6,
        )

        self.nvar = (
            self.nv
            +
            12
        )

        # CasADi solvers indexed by reduced problem size.
        self._solver_cache = {}


    # ========================================================
    # NULL SPACE
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

            numerical_tol = (
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
                numerical_tol,
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


    # ========================================================
    # PARTICULAR SOLUTION
    # ========================================================

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
    # CASADI SOLVER
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

            return (
                self._solver_cache[
                    key
                ]
            )

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

        solver_options = {
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
            solver_options,
        )

        self._solver_cache[
            key
        ] = solver

        return solver


    # ========================================================
    # REDUCED OBJECTIVE
    # ========================================================

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

        # y = y_base + Z u
        #
        # min || B y - desired ||^2
        #
        #     + epsilon ||y||^2

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


    # ========================================================
    # REDUCE INEQUALITIES
    # ========================================================

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

            if (
                row_norm
                >
                zero_row_tolerance
            ):

                keep_rows.append(
                    row_index
                )

                continue

            # No remaining freedom along this row.
            # y_base itself must already satisfy it.

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


    # ========================================================
    # SOLVE REDUCED QP
    # ========================================================

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

        # QRQP expects a constraint matrix.
        #
        # If no physical inequalities remain, add one
        # harmless equality:
        #
        #     0 = 0

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

        stats = (
            solver.stats()
        )

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

        # ====================================================
        # VALIDATE ORIGINAL PHYSICAL INEQUALITIES
        # ====================================================

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
    # FRICTION
    # ========================================================

    def _append_friction_constraints(
        self,
        rows,
        lower,
        upper,
        wrench_start,
    ):

        """
        Conservative inner approximation:

            |Fx| + |Fy| <= mu Fz

            Fz >= 0

        Flat horizontal terrain -> forces are represented
        directly in WORLD coordinates.
        """

        mu = float(
            self.config.friction_coefficient
        )

        fx = wrench_start + 0
        fy = wrench_start + 1
        fz = wrench_start + 2

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

            row[fx] = sx
            row[fy] = sy
            row[fz] = -mu

            rows.append(
                row
            )

            lower.append(
                -np.inf
            )

            upper.append(
                0.0
            )

        # Fz >= 0

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[fz] = 1.0

        rows.append(
            row
        )

        lower.append(
            0.0
        )

        upper.append(
            np.inf
        )


    # ========================================================
    # COP FEASIBILITY
    # ========================================================

    def _append_cop_constraints(
        self,
        rows,
        lower,
        upper,
        wrench_start,
        support_bounds,
        contact_height,
    ):

        """
        Contact wrench is represented at the foot site:

            lambda =
            [Fx, Fy, Fz, Mx, My, Mz]

        in WORLD coordinates.

        Flat ground:

            n = world +Z.

        If foot-site origin is h meters above the ground plane:

            r = [x_cop, y_cop, -h]

        and:

            M = r x F + [0, 0, Mz_free]

        Therefore:

            x_cop =
                -(My + h Fx) / Fz

            y_cop =
                (Mx - h Fy) / Fz

        Enforce:

            xmin <= x_cop <= xmax
            ymin <= y_cop <= ymax

        This is CoP FEASIBILITY only.

        There is no desired CoP and no CoP tracking task.
        """

        bounds = np.asarray(
            support_bounds,
            dtype=float,
        )

        if bounds.shape != (
            4,
        ):

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

        fx = wrench_start + 0
        fy = wrench_start + 1
        fz = wrench_start + 2

        mx = wrench_start + 3
        my = wrench_start + 4

        # ====================================================
        # x_cop >= xmin
        #
        # -(My + h Fx)/Fz >= xmin
        #
        # My + h Fx + xmin Fz <= 0
        # ====================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[fx] = h
        row[fz] = xmin
        row[my] = 1.0

        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        # ====================================================
        # x_cop <= xmax
        #
        # -(My + h Fx)/Fz <= xmax
        #
        # -My - h Fx - xmax Fz <= 0
        # ====================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[fx] = -h
        row[fz] = -xmax
        row[my] = -1.0

        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        # ====================================================
        # y_cop >= ymin
        #
        # (Mx - h Fy)/Fz >= ymin
        #
        # -Mx + h Fy + ymin Fz <= 0
        # ====================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[fy] = h
        row[fz] = ymin
        row[mx] = -1.0

        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        # ====================================================
        # y_cop <= ymax
        #
        # Mx - h Fy - ymax Fz <= 0
        # ====================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[fy] = -h
        row[fz] = -ymax
        row[mx] = 1.0

        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)


    # ========================================================
    # PHYSICAL MODEL — RANK 1
    # ========================================================

    def _build_physical_model(
        self,
        mass_matrix,
        effective_bias,
        selection_matrix,

        left_jacobian,
        right_jacobian,

        left_support_bounds,
        right_support_bounds,

        left_contact_height,
        right_contact_height,

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

        JL = np.asarray(
            left_jacobian,
            dtype=float,
        )

        JR = np.asarray(
            right_jacobian,
            dtype=float,
        )

        # ====================================================
        # CONTACT JACOBIAN
        # ====================================================

        Jc = np.vstack(
            (
                JL,
                JR,
            )
        )

        # ====================================================
        # RANK-1 FLOATING-BASE DYNAMICS
        # ====================================================

        base_indices = (
            self.unactuated_dof_indices
        )

        B1 = np.zeros(
            (
                6,
                self.nvar,
            ),
            dtype=float,
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

        Jc_base_transpose = (
            Jc[
                :,
                base_indices
            ]
            .T
        )

        B1[
            :,
            self.left_wrench_slice
        ] = (
            -Jc_base_transpose[
                :,
                0:6
            ]
        )

        B1[
            :,
            self.right_wrench_slice
        ] = (
            -Jc_base_transpose[
                :,
                6:12
            ]
        )

        d1 = (
            -h[
                base_indices
            ]
        )

        # ====================================================
        # TORQUE RECOVERY MAP
        # ====================================================

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
                "Unexpected actuated selection matrix shape."
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

        E_inv = (
            np.linalg.inv(
                E
            )
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

        Jc_act_transpose = (
            Jc[
                :,
                actuated_indices
            ]
            .T
        )

        A_tau[
            :,
            self.left_wrench_slice
        ] = (
            -Jc_act_transpose[
                :,
                0:6
            ]
        )

        A_tau[
            :,
            self.right_wrench_slice
        ] = (
            -Jc_act_transpose[
                :,
                6:12
            ]
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

        # ====================================================
        # PHYSICAL INEQUALITIES
        # ====================================================

        rows = []
        lower = []
        upper = []

        # ----------------------------------------------------
        # Torque limits
        #
        # tau = A_tau y + b_tau
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Friction + unilateral
        # ----------------------------------------------------

        self._append_friction_constraints(
            rows=rows,
            lower=lower,
            upper=upper,
            wrench_start=(
                self.left_wrench_start
            ),
        )

        self._append_friction_constraints(
            rows=rows,
            lower=lower,
            upper=upper,
            wrench_start=(
                self.right_wrench_start
            ),
        )

        # ----------------------------------------------------
        # CoP feasibility
        # ----------------------------------------------------

        self._append_cop_constraints(
            rows=rows,
            lower=lower,
            upper=upper,
            wrench_start=(
                self.left_wrench_start
            ),
            support_bounds=(
                left_support_bounds
            ),
            contact_height=(
                left_contact_height
            ),
        )

        self._append_cop_constraints(
            rows=rows,
            lower=lower,
            upper=upper,
            wrench_start=(
                self.right_wrench_start
            ),
            support_bounds=(
                right_support_bounds
            ),
            contact_height=(
                right_contact_height
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
    # RANK 2
    # ========================================================

    def _build_rank2(
        self,

        left_jacobian,
        left_jdot_v,

        right_jacobian,
        right_jdot_v,

        com_jacobian,
        com_jdot_v_z,

        desired_com_acceleration_z,
    ):

        JL = np.asarray(
            left_jacobian,
            dtype=float,
        )

        JR = np.asarray(
            right_jacobian,
            dtype=float,
        )

        JdotL_v = np.asarray(
            left_jdot_v,
            dtype=float,
        )

        JdotR_v = np.asarray(
            right_jdot_v,
            dtype=float,
        )

        Jcom = np.asarray(
            com_jacobian,
            dtype=float,
        )

        # Two 6D stance-foot constraints + CoM-z.
        B2 = np.zeros(
            (
                13,
                self.nvar,
            ),
            dtype=float,
        )

        d2 = np.zeros(
            13,
            dtype=float,
        )

        # ----------------------------------------------------
        # Left stance 6D
        # ----------------------------------------------------

        B2[
            0:6,
            self.qacc_slice
        ] = (
            JL
        )

        d2[
            0:6
        ] = (
            -JdotL_v
        )

        # ----------------------------------------------------
        # Right stance 6D
        # ----------------------------------------------------

        B2[
            6:12,
            self.qacc_slice
        ] = (
            JR
        )

        d2[
            6:12
        ] = (
            -JdotR_v
        )

        # ----------------------------------------------------
        # CoM height only
        # ----------------------------------------------------

        B2[
            12,
            self.qacc_slice
        ] = (
            Jcom[
                2,
                :
            ]
        )

        d2[
            12
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
    # RANK 5 — WRENCH REGULARIZATION
    # ========================================================

    def _build_rank5(
        self,
        left_wrench_reference,
        right_wrench_reference,
    ):

        left_reference = np.asarray(
            left_wrench_reference,
            dtype=float,
        )

        right_reference = np.asarray(
            right_wrench_reference,
            dtype=float,
        )

        B5 = np.zeros(
            (
                12,
                self.nvar,
            ),
            dtype=float,
        )

        B5[
            0:6,
            self.left_wrench_slice
        ] = np.eye(
            6
        )

        B5[
            6:12,
            self.right_wrench_slice
        ] = np.eye(
            6
        )

        d5 = np.concatenate(
            (
                left_reference,
                right_reference,
            )
        )

        return (
            B5,
            d5,
        )


    # ========================================================
    # MAIN SOLVE
    # ========================================================

    def solve_double_support(
        self,

        mass_matrix,
        effective_bias,
        selection_matrix,

        left_jacobian,
        left_jdot_v,

        right_jacobian,
        right_jdot_v,

        com_jacobian,
        com_jdot_v_z,

        desired_com_acceleration_z,

        desired_posture_acceleration,

        left_wrench_reference,
        right_wrench_reference,

        left_support_bounds,
        right_support_bounds,

        left_contact_height,
        right_contact_height,

        torque_lower,
        torque_upper,
    ) -> DoubleSupportHQPSolution:

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

                left_jacobian=(
                    left_jacobian
                ),

                right_jacobian=(
                    right_jacobian
                ),

                left_support_bounds=(
                    left_support_bounds
                ),

                right_support_bounds=(
                    right_support_bounds
                ),

                left_contact_height=(
                    left_contact_height
                ),

                right_contact_height=(
                    right_contact_height
                ),

                torque_lower=(
                    torque_lower
                ),

                torque_upper=(
                    torque_upper
                ),
            )
        )

        # Particular solution satisfying the 6D
        # floating-base dynamics.

        y1 = (
            self._particular_solution(
                B1,
                d1,
            )
        )

        # Remaining null space for lower priorities.

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

                left_jacobian=(
                    left_jacobian
                ),

                left_jdot_v=(
                    left_jdot_v
                ),

                right_jacobian=(
                    right_jacobian
                ),

                right_jdot_v=(
                    right_jdot_v
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

                name="rank2",

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

        # Lock the achieved Rank-2 task.

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

                name="rank4",

                y_base=y2,

                Z=Z2,

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

        # Lock achieved Rank 4.

        B124 = np.vstack(
            (
                B1,
                B2,
                B4,
            )
        )

        Z4 = (
            self._nullspace(
                B124
            )
        )

        # ====================================================
        # RANK 5
        # ====================================================

        B5, d5 = (
            self._build_rank5(
                left_wrench_reference,
                right_wrench_reference,
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

                        name="rank5",

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

                # Rank 5 is only redundancy resolution.
                #
                # Never sacrifice the higher-priority
                # physically valid solution because the
                # lowest-priority regularizer becomes
                # numerically singular.

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

            solve_time_rank5 = (
                0.0
            )

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
        # EXTRACT VARIABLES
        # ====================================================

        qacc = (
            y5[
                self.qacc_slice
            ].copy()
        )

        left_wrench = (
            y5[
                self.left_wrench_slice
            ].copy()
        )

        right_wrench = (
            y5[
                self.right_wrench_slice
            ].copy()
        )

        # ====================================================
        # RECOVER TORQUE
        # ====================================================

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
                f"{torque_violation:.6e}\n"
                f"tau = {torque}"
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

        if (
            rank1_lock_error
            >
            self.config.constraint_tolerance
        ):

            raise RuntimeError(
                "Rank-1 hierarchy violation.\n"
                f"error = "
                f"{rank1_lock_error:.6e}"
            )

        if (
            rank2_lock_error
            >
            self.config.constraint_tolerance
        ):

            raise RuntimeError(
                "Rank-2 hierarchy violation.\n"
                f"error = "
                f"{rank2_lock_error:.6e}"
            )

        if (
            rank4_lock_error
            >
            self.config.constraint_tolerance
        ):

            raise RuntimeError(
                "Rank-4 hierarchy violation.\n"
                f"error = "
                f"{rank4_lock_error:.6e}"
            )

        max_violation = max(
            violation2,
            violation4,
            violation5,

            torque_violation,

            rank1_lock_error,
            rank2_lock_error,
            rank4_lock_error,
        )

        return DoubleSupportHQPSolution(

            qacc=(
                qacc
            ),

            left_wrench=(
                left_wrench
            ),

            right_wrench=(
                right_wrench
            ),

            torque=(
                torque
            ),

            rank2_residual=(
                rank2_residual
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

            solve_time_rank4=(
                solve_time_rank4
            ),

            solve_time_rank5=(
                solve_time_rank5
            ),
        )