# step_timing_adaptation/adaptive_step_planner.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import casadi as ca
import numpy as np


# ============================================================
# STANCE LEG
# ============================================================

class StanceLeg(Enum):

    LEFT = "left"
    RIGHT = "right"


# ============================================================
# PARAMETERS
# ============================================================

@dataclass(frozen=True)
class StepPlannerParameters:
    """
    Parameters of the LIPM stepping planner.

    Conventions:

        x : forward
        y : left
        z : upward

    W is the lateral deviation from the default step width l_p,
    following the convention used in the current implementation.

    Therefore W = 0 means straight walking with nominal
    left-right foot spacing l_p.
    """

    gravity: float

    com_height: float

    default_step_width: float

    step_length_min: float
    step_length_max: float

    step_width_min: float
    step_width_max: float

    step_time_min: float
    step_time_max: float


# ============================================================
# VIABILITY LIMITS
# ============================================================

@dataclass(frozen=True)
class ViabilityBounds:

    bx_min: float
    bx_max: float

    by_max_out: float
    by_max_in: float

    tau_min: float
    tau_max: float


# ============================================================
# NOMINAL STEP
# ============================================================

@dataclass(frozen=True)
class NominalStep:

    stance_leg: StanceLeg

    desired_velocity_x: float
    desired_velocity_y: float

    lower_time_bound: float
    upper_time_bound: float

    step_time: float

    step_length: float

    # W_nom from the paper:
    # lateral deviation from default step width.
    step_width_deviation: float

    # Actual displacement of next foot relative to stance foot.
    step_displacement_x: float
    step_displacement_y: float

    tau: float

    dcm_offset_x: float
    dcm_offset_y: float


# ============================================================
# ADAPTED STEP
# ============================================================

@dataclass(frozen=True)
class AdaptedStep:

    stance_leg: StanceLeg

    # Absolute landing position in the same horizontal frame
    # used by stance_position.
    step_location_x: float
    step_location_y: float

    # Landing displacement relative to current stance position.
    step_displacement_x: float
    step_displacement_y: float

    tau: float
    step_time: float

    dcm_offset_x: float
    dcm_offset_y: float

    objective: float
    max_equality_residual: float


# ============================================================
# PLANNER
# ============================================================

class AdaptiveStepPlanner:
    """
    LIPM / DCM step planner based on Khadiv et al.,
    "Walking Control Based on Step Timing Adaptation".

    Implemented here:

        Stage 1
            - LIPM natural frequency
            - viability bounds
            - nominal stepping values
              Eq. (7), Eq. (16), Eq. (17)

        Stage 2
            - timing change of variable tau = exp(omega*T)
            - linear DCM equality
            - online step location / timing QP
              Eq. (18), Eq. (19), Eq. (20)

    All numerical walking parameters and test settings are
    intentionally supplied from run.py.
    """

    def __init__(
        self,
        parameters: StepPlannerParameters,
    ) -> None:

        self.parameters = parameters

        self._validate_parameters()

        self.omega = math.sqrt(
            self.parameters.gravity
            /
            self.parameters.com_height
        )

        self.viability_bounds = (
            self._compute_viability_bounds()
        )

        self._stage2_solver = (
            self._create_stage2_solver()
        )


    # ========================================================
    # PARAMETER VALIDATION
    # ========================================================

    def _validate_parameters(
        self,
    ) -> None:

        p = self.parameters

        values = {
            "gravity": p.gravity,
            "com_height": p.com_height,
            "default_step_width": p.default_step_width,
            "step_length_min": p.step_length_min,
            "step_length_max": p.step_length_max,
            "step_width_min": p.step_width_min,
            "step_width_max": p.step_width_max,
            "step_time_min": p.step_time_min,
            "step_time_max": p.step_time_max,
        }

        for name, value in values.items():

            if not math.isfinite(
                float(value)
            ):

                raise ValueError(
                    f"{name} must be finite."
                )

        if p.gravity <= 0.0:

            raise ValueError(
                "gravity must be positive."
            )

        if p.com_height <= 0.0:

            raise ValueError(
                "com_height must be positive."
            )

        if p.default_step_width <= 0.0:

            raise ValueError(
                "default_step_width must be positive."
            )

        if (
            p.step_length_min
            >
            p.step_length_max
        ):

            raise ValueError(
                "step_length_min must be <= step_length_max."
            )

        if (
            p.step_width_min
            >
            p.step_width_max
        ):

            raise ValueError(
                "step_width_min must be <= step_width_max."
            )

        if p.step_time_min <= 0.0:

            raise ValueError(
                "step_time_min must be positive."
            )

        if (
            p.step_time_min
            >
            p.step_time_max
        ):

            raise ValueError(
                "step_time_min must be <= step_time_max."
            )


    # ========================================================
    # VIABILITY BOUNDS
    # ========================================================

    def _compute_viability_bounds(
        self,
    ) -> ViabilityBounds:
        """
        Sagittal:
            Eq. (9), extended to the configured backward bound.

        Lateral:
            Appendix B, Eq. (32a)-(32b).
        """

        p = self.parameters

        tau_min = math.exp(
            self.omega
            *
            p.step_time_min
        )

        tau_max = math.exp(
            self.omega
            *
            p.step_time_max
        )

        denominator_x = (
            tau_min
            -
            1.0
        )

        bx_min = (
            p.step_length_min
            /
            denominator_x
        )

        bx_max = (
            p.step_length_max
            /
            denominator_x
        )

        denominator_y = (
            1.0
            -
            tau_min**2
        )

        by_max_out = (
            p.default_step_width
            /
            (
                1.0
                +
                tau_min
            )
            +
            (
                p.step_width_max
                -
                p.step_width_min
                *
                tau_min
            )
            /
            denominator_y
        )

        by_max_in = (
            p.default_step_width
            /
            (
                1.0
                +
                tau_min
            )
            +
            (
                p.step_width_min
                -
                p.step_width_max
                *
                tau_min
            )
            /
            denominator_y
        )

        return ViabilityBounds(
            bx_min=float(
                bx_min
            ),
            bx_max=float(
                bx_max
            ),
            by_max_out=float(
                by_max_out
            ),
            by_max_in=float(
                by_max_in
            ),
            tau_min=float(
                tau_min
            ),
            tau_max=float(
                tau_max
            ),
        )


    # ========================================================
    # STAGE 1 — NOMINAL STEPPING
    # ========================================================

    def compute_nominal_step(
        self,
        desired_velocity_x: float,
        desired_velocity_y: float,
        stance_leg: StanceLeg,
    ) -> NominalStep:
        """
        Compute Stage-1 nominal values.

        Paper:
            Eq. (16)
            Eq. (17)
            Eq. (7)

        T_nom is selected at the center of the feasible timing
        interval [B_l, B_u].
        """

        vx = float(
            desired_velocity_x
        )

        vy = float(
            desired_velocity_y
        )

        if not math.isfinite(vx):

            raise ValueError(
                "desired_velocity_x must be finite."
            )

        if not math.isfinite(vy):

            raise ValueError(
                "desired_velocity_y must be finite."
            )

        p = self.parameters

        # ====================================================
        # FEASIBLE TIME INTERVAL
        # ====================================================

        lower_candidates = [
            p.step_time_min,
        ]

        upper_candidates = [
            p.step_time_max,
        ]

        # ----------------------------------------------------
        # Sagittal velocity
        # ----------------------------------------------------

        if abs(vx) > 1.0e-12:

            if vx > 0.0:

                if p.step_length_max <= 0.0:

                    raise ValueError(
                        "Positive desired_velocity_x is not "
                        "feasible because step_length_max <= 0."
                    )

                upper_candidates.append(
                    p.step_length_max
                    /
                    vx
                )

                if p.step_length_min > 0.0:

                    lower_candidates.append(
                        p.step_length_min
                        /
                        vx
                    )

            else:

                if p.step_length_min >= 0.0:

                    raise ValueError(
                        "Negative desired_velocity_x is not "
                        "feasible because step_length_min >= 0."
                    )

                speed = abs(
                    vx
                )

                upper_candidates.append(
                    abs(
                        p.step_length_min
                    )
                    /
                    speed
                )

                if p.step_length_max < 0.0:

                    lower_candidates.append(
                        abs(
                            p.step_length_max
                        )
                        /
                        speed
                    )

        # ----------------------------------------------------
        # Lateral velocity / W deviation
        # ----------------------------------------------------

        if abs(vy) > 1.0e-12:

            if vy > 0.0:

                if p.step_width_max <= 0.0:

                    raise ValueError(
                        "Positive desired_velocity_y is not "
                        "feasible because step_width_max <= 0."
                    )

                upper_candidates.append(
                    p.step_width_max
                    /
                    vy
                )

                if p.step_width_min > 0.0:

                    lower_candidates.append(
                        p.step_width_min
                        /
                        vy
                    )

            else:

                if p.step_width_min >= 0.0:

                    raise ValueError(
                        "Negative desired_velocity_y is not "
                        "feasible because step_width_min >= 0."
                    )

                speed = abs(
                    vy
                )

                upper_candidates.append(
                    abs(
                        p.step_width_min
                    )
                    /
                    speed
                )

                if p.step_width_max < 0.0:

                    lower_candidates.append(
                        abs(
                            p.step_width_max
                        )
                        /
                        speed
                    )

        B_l = max(
            lower_candidates
        )

        B_u = min(
            upper_candidates
        )

        if (
            B_l
            >
            B_u
            +
            1.0e-12
        ):

            raise ValueError(
                "Desired walking velocity is not feasible "
                "with the current step bounds.\n"
                f"B_l = {B_l:.6f} s\n"
                f"B_u = {B_u:.6f} s"
            )

        # ====================================================
        # EQ. (17)
        # ====================================================

        T_nom = (
            0.5
            *
            (
                B_l
                +
                B_u
            )
        )

        L_nom = (
            vx
            *
            T_nom
        )

        W_nom = (
            vy
            *
            T_nom
        )

        # ====================================================
        # ACTUAL NEXT-FOOT DISPLACEMENT
        # ====================================================

        if stance_leg is StanceLeg.LEFT:

            delta_y_nom = (
                -p.default_step_width
                +
                W_nom
            )

            stance_sign = +1.0

        elif stance_leg is StanceLeg.RIGHT:

            delta_y_nom = (
                +p.default_step_width
                +
                W_nom
            )

            stance_sign = -1.0

        else:

            raise ValueError(
                f"Unsupported stance leg: {stance_leg}"
            )

        # ====================================================
        # TAU
        # ====================================================

        tau_nom = math.exp(
            self.omega
            *
            T_nom
        )

        # ====================================================
        # EQ. (7a)
        # ====================================================

        bx_nom = (
            L_nom
            /
            (
                tau_nom
                -
                1.0
            )
        )

        # ====================================================
        # EQ. (7b)
        # ====================================================

        by_nom = (
            stance_sign
            *
            p.default_step_width
            /
            (
                1.0
                +
                tau_nom
            )
            -
            W_nom
            /
            (
                1.0
                -
                tau_nom
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        tolerance = 1.0e-10

        if not (
            p.step_length_min
            -
            tolerance
            <=
            L_nom
            <=
            p.step_length_max
            +
            tolerance
        ):

            raise RuntimeError(
                "Computed nominal step length violates bounds."
            )

        if not (
            p.step_width_min
            -
            tolerance
            <=
            W_nom
            <=
            p.step_width_max
            +
            tolerance
        ):

            raise RuntimeError(
                "Computed nominal step-width deviation "
                "violates bounds."
            )

        if not (
            p.step_time_min
            -
            tolerance
            <=
            T_nom
            <=
            p.step_time_max
            +
            tolerance
        ):

            raise RuntimeError(
                "Computed nominal step time violates bounds."
            )

        return NominalStep(
            stance_leg=stance_leg,
            desired_velocity_x=vx,
            desired_velocity_y=vy,
            lower_time_bound=float(
                B_l
            ),
            upper_time_bound=float(
                B_u
            ),
            step_time=float(
                T_nom
            ),
            step_length=float(
                L_nom
            ),
            step_width_deviation=float(
                W_nom
            ),
            step_displacement_x=float(
                L_nom
            ),
            step_displacement_y=float(
                delta_y_nom
            ),
            tau=float(
                tau_nom
            ),
            dcm_offset_x=float(
                bx_nom
            ),
            dcm_offset_y=float(
                by_nom
            ),
        )


    # ========================================================
    # STAGE 2 — QP CREATION
    # ========================================================

    def _create_stage2_solver(
        self,
    ):

        # Decision vector:
        #
        # z = [u_Tx, u_Ty, tau, b_x, b_y]

        number_variables = 5
        number_equalities = 2

        qp_structure = {
            "h": ca.Sparsity.dense(
                number_variables,
                number_variables,
            ),
            "a": ca.Sparsity.dense(
                number_equalities,
                number_variables,
            ),
        }

        options = {
            "print_header": False,
            "print_iter": False,
            "print_info": False,
            "error_on_fail": False,
            "max_iter": 1000,
        }

        return ca.conic(
            "adaptive_step_stage2",
            "qrqp",
            qp_structure,
            options,
        )


    # ========================================================
    # STAGE 2 — LATERAL LANDING BOUNDS
    # ========================================================

    def _get_lateral_step_displacement_bounds(
        self,
        stance_leg: StanceLeg,
    ) -> tuple[float, float]:

        p = self.parameters

        if stance_leg is StanceLeg.LEFT:

            # Left stance -> right foot swings.
            # Delta_y = -l_p + W.

            lower = (
                -p.default_step_width
                +
                p.step_width_min
            )

            upper = (
                -p.default_step_width
                +
                p.step_width_max
            )

        elif stance_leg is StanceLeg.RIGHT:

            # Right stance -> left foot swings.
            # Delta_y = +l_p + W.

            lower = (
                +p.default_step_width
                +
                p.step_width_min
            )

            upper = (
                +p.default_step_width
                +
                p.step_width_max
            )

        else:

            raise ValueError(
                f"Unsupported stance leg: {stance_leg}"
            )

        return (
            float(lower),
            float(upper),
        )


    # ========================================================
    # STAGE 2 — LATERAL VIABILITY BOUNDS
    # ========================================================

    def _get_lateral_dcm_bounds(
        self,
        stance_leg: StanceLeg,
    ) -> tuple[float, float]:

        vb = self.viability_bounds

        positive_lower = min(
            vb.by_max_out,
            vb.by_max_in,
        )

        positive_upper = max(
            vb.by_max_out,
            vb.by_max_in,
        )

        if stance_leg is StanceLeg.LEFT:

            # Current convention gives positive b_y for
            # straight walking during left stance.

            lower = positive_lower
            upper = positive_upper

        elif stance_leg is StanceLeg.RIGHT:

            # Mirror of the left-stance bounds.

            lower = -positive_upper
            upper = -positive_lower

        else:

            raise ValueError(
                f"Unsupported stance leg: {stance_leg}"
            )

        return (
            float(lower),
            float(upper),
        )


    # ========================================================
    # STAGE 2 — ONLINE STEP LOCATION / TIMING ADAPTATION
    # ========================================================

    def solve_adaptive_step(
        self,
        nominal_step: NominalStep,
        dcm_measured,
        stance_position,
        elapsed_time: float,
        alpha_location: float,
        alpha_timing: float,
        alpha_dcm: float,
    ) -> AdaptedStep:
        """
        Solve the Stage-2 QP.

        Decision vector:

            z = [u_Tx, u_Ty, tau, b_x, b_y]

        with:

            tau = exp(omega*T)

        and the linear DCM equality:

            u_T
            - (xi_mea - u_0) exp(-omega*t) tau
            + b
            = u_0

        This first implementation uses hard viability bounds.
        The high-penalty soft viability formulation described
        by the paper can be introduced after the basic QP has
        been validated independently.
        """

        xi = np.asarray(
            dcm_measured,
            dtype=float,
        ).reshape(-1)

        u0 = np.asarray(
            stance_position,
            dtype=float,
        ).reshape(-1)

        if xi.shape != (2,):

            raise ValueError(
                "dcm_measured must contain exactly 2 values."
            )

        if u0.shape != (2,):

            raise ValueError(
                "stance_position must contain exactly 2 values."
            )

        if not np.all(
            np.isfinite(
                xi
            )
        ):

            raise ValueError(
                "dcm_measured must be finite."
            )

        if not np.all(
            np.isfinite(
                u0
            )
        ):

            raise ValueError(
                "stance_position must be finite."
            )

        t = float(
            elapsed_time
        )

        if (
            not math.isfinite(t)
            or
            t < 0.0
        ):

            raise ValueError(
                "elapsed_time must be finite and >= 0."
            )

        if (
            t
            >
            self.parameters.step_time_max
            +
            1.0e-12
        ):

            raise ValueError(
                "elapsed_time exceeds step_time_max."
            )

        alpha_location = float(
            alpha_location
        )

        alpha_timing = float(
            alpha_timing
        )

        alpha_dcm = float(
            alpha_dcm
        )

        for name, value in (
            ("alpha_location", alpha_location),
            ("alpha_timing", alpha_timing),
            ("alpha_dcm", alpha_dcm),
        ):

            if (
                not math.isfinite(value)
                or
                value <= 0.0
            ):

                raise ValueError(
                    f"{name} must be finite and positive."
                )

        # ====================================================
        # NOMINAL REFERENCES
        # ====================================================

        uT_nom = np.array(
            [
                u0[0]
                +
                nominal_step.step_displacement_x,

                u0[1]
                +
                nominal_step.step_displacement_y,
            ],
            dtype=float,
        )

        z_reference = np.array(
            [
                uT_nom[0],
                uT_nom[1],
                nominal_step.tau,
                nominal_step.dcm_offset_x,
                nominal_step.dcm_offset_y,
            ],
            dtype=float,
        )

        # ====================================================
        # OBJECTIVE — EQ. (20)
        # ====================================================

        weights = np.array(
            [
                alpha_location,
                alpha_location,
                alpha_timing,
                alpha_dcm,
                alpha_dcm,
            ],
            dtype=float,
        )

        # CasADi conic solves:
        #
        #     0.5 z^T H z + g^T z
        #
        # The constant part of ||z-z_ref||_W^2 is omitted.

        H = (
            2.0
            *
            np.diag(
                weights
            )
        )

        g = (
            -2.0
            *
            weights
            *
            z_reference
        )

        # ====================================================
        # LINEAR DCM EQUALITY — EQ. (19)
        # ====================================================

        dcm_factor = (
            (
                xi
                -
                u0
            )
            *
            math.exp(
                -self.omega
                *
                t
            )
        )

        Aeq = np.array(
            [
                [
                    1.0,
                    0.0,
                    -dcm_factor[0],
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    -dcm_factor[1],
                    0.0,
                    1.0,
                ],
            ],
            dtype=float,
        )

        beq = (
            u0.copy()
        )

        # ====================================================
        # VARIABLE BOUNDS
        # ====================================================

        p = self.parameters
        vb = self.viability_bounds

        lateral_step_lower, lateral_step_upper = (
            self._get_lateral_step_displacement_bounds(
                nominal_step.stance_leg
            )
        )

        by_lower, by_upper = (
            self._get_lateral_dcm_bounds(
                nominal_step.stance_leg
            )
        )

        lower_bounds = np.array(
            [
                u0[0]
                +
                p.step_length_min,

                u0[1]
                +
                lateral_step_lower,

                vb.tau_min,

                vb.bx_min,

                by_lower,
            ],
            dtype=float,
        )

        upper_bounds = np.array(
            [
                u0[0]
                +
                p.step_length_max,

                u0[1]
                +
                lateral_step_upper,

                vb.tau_max,

                vb.bx_max,

                by_upper,
            ],
            dtype=float,
        )

        if np.any(
            lower_bounds
            >
            upper_bounds
        ):

            raise RuntimeError(
                "Invalid Stage-2 variable bounds."
            )

        # ====================================================
        # SOLVE QP
        # ====================================================

        result = self._stage2_solver(
            h=ca.DM(
                H
            ),
            g=ca.DM(
                g
            ),
            a=ca.DM(
                Aeq
            ),
            lba=ca.DM(
                beq
            ),
            uba=ca.DM(
                beq
            ),
            lbx=ca.DM(
                lower_bounds
            ),
            ubx=ca.DM(
                upper_bounds
            ),
        )

        stats = (
            self._stage2_solver.stats()
        )

        if not bool(
            stats.get(
                "success",
                False,
            )
        ):

            raise RuntimeError(
                "Stage-2 adaptive-step QP failed.\n"
                f"status = "
                f"{stats.get('return_status', 'unknown')}"
            )

        solution = np.asarray(
            result[
                "x"
            ],
            dtype=float,
        ).reshape(
            5
        )

        uTx = float(
            solution[0]
        )

        uTy = float(
            solution[1]
        )

        tau = float(
            solution[2]
        )

        bx = float(
            solution[3]
        )

        by = float(
            solution[4]
        )

        if tau <= 0.0:

            raise RuntimeError(
                "QP returned non-positive tau."
            )

        T = (
            math.log(
                tau
            )
            /
            self.omega
        )

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        equality_residual = (
            Aeq
            @
            solution
            -
            beq
        )

        max_equality_residual = float(
            np.max(
                np.abs(
                    equality_residual
                )
            )
        )

        error = (
            solution
            -
            z_reference
        )

        objective = float(
            alpha_location
            *
            (
                error[0]**2
                +
                error[1]**2
            )
            +
            alpha_timing
            *
            error[2]**2
            +
            alpha_dcm
            *
            (
                error[3]**2
                +
                error[4]**2
            )
        )

        return AdaptedStep(
            stance_leg=nominal_step.stance_leg,
            step_location_x=uTx,
            step_location_y=uTy,
            step_displacement_x=float(
                uTx
                -
                u0[0]
            ),
            step_displacement_y=float(
                uTy
                -
                u0[1]
            ),
            tau=tau,
            step_time=float(
                T
            ),
            dcm_offset_x=bx,
            dcm_offset_y=by,
            objective=objective,
            max_equality_residual=max_equality_residual,
        )
