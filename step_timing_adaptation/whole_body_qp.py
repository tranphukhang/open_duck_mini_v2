# step_timing_adaptation/whole_body_qp.py

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DoubleSupportQPConfig:

    friction_coefficient: float = 0.6

    # Rank 4
    posture_weight: float = 1.0e4

    # Rank 5
    force_regularization_weight: float = 1.0
    moment_regularization_weight: float = 10.0

    # Numerical regularization
    qacc_regularization_weight: float = 1.0e-6
    torque_regularization_weight: float = 1.0e-8
    global_regularization_weight: float = 1.0e-10

    solver_name: str = "qrqp"


# ============================================================
# SOLUTION
# ============================================================

@dataclass
class DoubleSupportQPSolution:

    qacc: np.ndarray

    left_wrench: np.ndarray
    right_wrench: np.ndarray

    torque: np.ndarray

    cost: float

    success: bool


# ============================================================
# WHOLE-BODY INVERSE DYNAMICS QP
# ============================================================

class WholeBodyInverseDynamicsQP:
    """
    Double-support inverse-dynamics QP.

    Decision vector:

        x =
        [
            qacc          nv
            lambda_L      6
            lambda_R      6
            tau           nu
        ]

    Contact wrench convention:

        lambda =
        [Fx, Fy, Fz, Mx, My, Mz]

    expressed in the WORLD frame at each foot site.

    HARD constraints:

        M qacc + h
            =
        S.T tau
            +
        JL.T lambda_L
            +
        JR.T lambda_R

        JL qacc + JdotL v = 0

        JR qacc + JdotR v = 0

        Jcom_z qacc + Jdot_com_z v
            =
        a_com_z_des

        tau_min <= tau <= tau_max

        Fz >= 0

        |Fx| <= mu Fz

        |Fy| <= mu Fz

    No CoP constraint.
    No ZMP task.
    No horizontal CoM task.
    """

    def __init__(
        self,
        nv: int,
        nu: int,
        actuated_dof_indices,
        config: DoubleSupportQPConfig,
    ) -> None:

        self.nv = int(
            nv
        )

        self.nu = int(
            nu
        )

        self.config = config

        self.actuated_dof_indices = np.asarray(
            actuated_dof_indices,
            dtype=int,
        )

        if self.actuated_dof_indices.shape != (
            self.nu,
        ):

            raise ValueError(
                "actuated_dof_indices has wrong shape."
            )

        # ====================================================
        # VARIABLE LAYOUT
        # ====================================================

        self.qacc_start = 0

        self.left_wrench_start = (
            self.nv
        )

        self.right_wrench_start = (
            self.left_wrench_start
            +
            6
        )

        self.torque_start = (
            self.right_wrench_start
            +
            6
        )

        self.nvar = (
            self.torque_start
            +
            self.nu
        )

        self.qacc_slice = slice(
            self.qacc_start,
            self.left_wrench_start,
        )

        self.left_wrench_slice = slice(
            self.left_wrench_start,
            self.right_wrench_start,
        )

        self.right_wrench_slice = slice(
            self.right_wrench_start,
            self.torque_start,
        )

        self.torque_slice = slice(
            self.torque_start,
            self.nvar,
        )

        # ====================================================
        # CONSTRAINT COUNT
        # ====================================================

        # Dynamics        : nv
        # Left stance     : 6
        # Right stance    : 6
        # CoM height      : 1
        # Friction        : 5 x 2

        self.ncon = (
            self.nv
            +
            6
            +
            6
            +
            1
            +
            10
        )

        # ====================================================
        # CASADI QP
        # ====================================================

        qp_structure = {
            "h": ca.Sparsity.dense(
                self.nvar,
                self.nvar,
            ),

            "a": ca.Sparsity.dense(
                self.ncon,
                self.nvar,
            ),
        }

        # Important for dynamic control:
        # do not print QRQP iterations every 1 ms.

        solver_options = {
            "print_header": False,
            "print_iter": False,
            "print_info": False,
            "error_on_fail": True,
        }

        self.solver = ca.conic(
            "whole_body_inverse_dynamics_qp",
            self.config.solver_name,
            qp_structure,
            solver_options,
        )


    # ========================================================
    # TRACKING COST
    # ========================================================

    @staticmethod
    def _add_tracking_cost(
        H,
        g,
        indices,
        target,
        weight,
    ):

        indices = np.asarray(
            indices,
            dtype=int,
        )

        target = np.asarray(
            target,
            dtype=float,
        )

        weight = float(
            weight
        )

        if target.shape != (
            len(
                indices
            ),
        ):

            raise ValueError(
                "Tracking target has wrong shape."
            )

        for k, index in enumerate(
            indices
        ):

            # w * (x-d)^2
            #
            # =
            #
            # 1/2 x' H x + g' x + constant

            H[
                index,
                index,
            ] += (
                2.0
                *
                weight
            )

            g[
                index
            ] += (
                -2.0
                *
                weight
                *
                target[
                    k
                ]
            )


    # ========================================================
    # FRICTION inner approximation CONE
    # ========================================================

    def _append_friction_constraints(
        self,
        A_rows,
        lower,
        upper,
        wrench_start,
    ):

        """
        Conservative linear inner approximation of the
        MuJoCo elliptic friction cone.

        MuJoCo contact model:

            sqrt(Fx^2 + Fy^2) <= mu * Fz

        QP approximation:

            |Fx| + |Fy| <= mu * Fz

        Therefore every force accepted by this QP is also
        inside the circular / elliptic friction cone.

        No CoP constraint is introduced here.
        """

        mu = float(
            self.config.friction_coefficient
        )

        fx = (
            wrench_start
            +
            0
        )

        fy = (
            wrench_start
            +
            1
        )

        fz = (
            wrench_start
            +
            2
        )

        # ========================================================
        # Fz >= 0
        # ========================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fz
        ] = 1.0

        A_rows.append(
            row
        )

        lower.append(
            0.0
        )

        upper.append(
            np.inf
        )

        # ========================================================
        # Fx + Fy <= mu Fz
        # ========================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fx
        ] = 1.0

        row[
            fy
        ] = 1.0

        row[
            fz
        ] = -mu

        A_rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )

        # ========================================================
        # Fx - Fy <= mu Fz
        # ========================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fx
        ] = 1.0

        row[
            fy
        ] = -1.0

        row[
            fz
        ] = -mu

        A_rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )

        # ========================================================
        # -Fx + Fy <= mu Fz
        # ========================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fx
        ] = -1.0

        row[
            fy
        ] = 1.0

        row[
            fz
        ] = -mu

        A_rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )

        # ========================================================
        # -Fx - Fy <= mu Fz
        # ========================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            fx
        ] = -1.0

        row[
            fy
        ] = -1.0

        row[
            fz
        ] = -mu

        A_rows.append(
            row
        )

        lower.append(
            -np.inf
        )

        upper.append(
            0.0
        )


    # ========================================================
    # SOLVE
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
        torque_lower,
        torque_upper,
    ) -> DoubleSupportQPSolution:

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

        qacc_posture_des = np.asarray(
            desired_posture_acceleration,
            dtype=float,
        )

        lambda_left_des = np.asarray(
            left_wrench_reference,
            dtype=float,
        )

        lambda_right_des = np.asarray(
            right_wrench_reference,
            dtype=float,
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
        # COST
        # ====================================================

        H = np.zeros(
            (
                self.nvar,
                self.nvar,
            ),
            dtype=float,
        )

        g = np.zeros(
            self.nvar,
            dtype=float,
        )

        H += (
            2.0
            *
            self.config.global_regularization_weight
            *
            np.eye(
                self.nvar
            )
        )

        # ----------------------------------------------------
        # qacc regularization
        # ----------------------------------------------------

        for i in range(
            self.nv
        ):

            H[
                i,
                i,
            ] += (
                2.0
                *
                self.config.qacc_regularization_weight
            )

        # ----------------------------------------------------
        # Rank 4: posture
        # ----------------------------------------------------

        posture_indices = (
            self.actuated_dof_indices
        )

        self._add_tracking_cost(
            H,
            g,
            posture_indices,
            qacc_posture_des,
            self.config.posture_weight,
        )

        # ----------------------------------------------------
        # Rank 5: contact force regularization
        # ----------------------------------------------------

        left_force_indices = np.arange(
            self.left_wrench_start,
            self.left_wrench_start + 3,
        )

        right_force_indices = np.arange(
            self.right_wrench_start,
            self.right_wrench_start + 3,
        )

        self._add_tracking_cost(
            H,
            g,
            left_force_indices,
            lambda_left_des[
                0:3
            ],
            self.config.force_regularization_weight,
        )

        self._add_tracking_cost(
            H,
            g,
            right_force_indices,
            lambda_right_des[
                0:3
            ],
            self.config.force_regularization_weight,
        )

        # ----------------------------------------------------
        # Rank 5: contact moment regularization
        # ----------------------------------------------------

        left_moment_indices = np.arange(
            self.left_wrench_start + 3,
            self.left_wrench_start + 6,
        )

        right_moment_indices = np.arange(
            self.right_wrench_start + 3,
            self.right_wrench_start + 6,
        )

        self._add_tracking_cost(
            H,
            g,
            left_moment_indices,
            lambda_left_des[
                3:6
            ],
            self.config.moment_regularization_weight,
        )

        self._add_tracking_cost(
            H,
            g,
            right_moment_indices,
            lambda_right_des[
                3:6
            ],
            self.config.moment_regularization_weight,
        )

        # ----------------------------------------------------
        # torque numerical regularization
        # ----------------------------------------------------

        for index in range(
            self.torque_start,
            self.nvar,
        ):

            H[
                index,
                index,
            ] += (
                2.0
                *
                self.config.torque_regularization_weight
            )

        # ====================================================
        # HARD CONSTRAINT MATRIX
        # ====================================================

        A_rows = []

        lower = []

        upper = []

        # ====================================================
        # RANK 1 - WHOLE-BODY DYNAMICS
        # ====================================================

        A_dyn = np.zeros(
            (
                self.nv,
                self.nvar,
            ),
            dtype=float,
        )

        A_dyn[
            :,
            self.qacc_slice,
        ] = M

        A_dyn[
            :,
            self.left_wrench_slice,
        ] = (
            -JL.T
        )

        A_dyn[
            :,
            self.right_wrench_slice,
        ] = (
            -JR.T
        )

        A_dyn[
            :,
            self.torque_slice,
        ] = (
            -S.T
        )

        b_dyn = (
            -h
        )

        for i in range(
            self.nv
        ):

            A_rows.append(
                A_dyn[
                    i,
                    :
                ]
            )

            lower.append(
                b_dyn[
                    i
                ]
            )

            upper.append(
                b_dyn[
                    i
                ]
            )

        # ====================================================
        # RANK 2 - LEFT STANCE
        # ====================================================

        for i in range(
            6
        ):

            row = np.zeros(
                self.nvar,
                dtype=float,
            )

            row[
                self.qacc_slice
            ] = (
                JL[
                    i,
                    :
                ]
            )

            target = (
                -JdotL_v[
                    i
                ]
            )

            A_rows.append(
                row
            )

            lower.append(
                target
            )

            upper.append(
                target
            )

        # ====================================================
        # RANK 2 - RIGHT STANCE
        # ====================================================

        for i in range(
            6
        ):

            row = np.zeros(
                self.nvar,
                dtype=float,
            )

            row[
                self.qacc_slice
            ] = (
                JR[
                    i,
                    :
                ]
            )

            target = (
                -JdotR_v[
                    i
                ]
            )

            A_rows.append(
                row
            )

            lower.append(
                target
            )

            upper.append(
                target
            )

        # ====================================================
        # RANK 2 - COM HEIGHT
        # ====================================================

        row = np.zeros(
            self.nvar,
            dtype=float,
        )

        row[
            self.qacc_slice
        ] = (
            Jcom[
                2,
                :
            ]
        )

        target_com = (
            float(
                desired_com_acceleration_z
            )
            -
            float(
                com_jdot_v_z
            )
        )

        A_rows.append(
            row
        )

        lower.append(
            target_com
        )

        upper.append(
            target_com
        )

        # ====================================================
        # RANK 1 - CONTACT FEASIBILITY
        # ====================================================

        self._append_friction_constraints(
            A_rows,
            lower,
            upper,
            self.left_wrench_start,
        )

        self._append_friction_constraints(
            A_rows,
            lower,
            upper,
            self.right_wrench_start,
        )

        # ====================================================
        # STACK
        # ====================================================

        A = np.vstack(
            A_rows
        )

        lba = np.asarray(
            lower,
            dtype=float,
        )

        uba = np.asarray(
            upper,
            dtype=float,
        )

        if A.shape != (
            self.ncon,
            self.nvar,
        ):

            raise RuntimeError(
                f"Unexpected A shape: {A.shape}"
            )

        # ====================================================
        # VARIABLE BOUNDS
        # ====================================================

        lbx = np.full(
            self.nvar,
            -np.inf,
            dtype=float,
        )

        ubx = np.full(
            self.nvar,
            +np.inf,
            dtype=float,
        )

        lbx[
            self.torque_slice
        ] = (
            tau_lower
        )

        ubx[
            self.torque_slice
        ] = (
            tau_upper
        )

        # ====================================================
        # SOLVE
        # ====================================================

        result = self.solver(
            h=ca.DM(
                H
            ),

            g=ca.DM(
                g
            ),

            a=ca.DM(
                A
            ),

            lba=ca.DM(
                lba
            ),

            uba=ca.DM(
                uba
            ),

            lbx=ca.DM(
                lbx
            ),

            ubx=ca.DM(
                ubx
            ),
        )

        x = np.asarray(
            result[
                "x"
            ]
        ).reshape(
            -1
        )

        stats = (
            self.solver.stats()
        )

        success = bool(
            stats.get(
                "success",
                True,
            )
        )

        if not success:

            raise RuntimeError(
                f"QP failed: {stats}"
            )

        cost = float(
            0.5
            *
            x
            @
            H
            @
            x
            +
            g
            @
            x
        )

        return DoubleSupportQPSolution(
            qacc=(
                x[
                    self.qacc_slice
                ].copy()
            ),

            left_wrench=(
                x[
                    self.left_wrench_slice
                ].copy()
            ),

            right_wrench=(
                x[
                    self.right_wrench_slice
                ].copy()
            ),

            torque=(
                x[
                    self.torque_slice
                ].copy()
            ),

            cost=(
                cost
            ),

            success=(
                success
            ),
        )