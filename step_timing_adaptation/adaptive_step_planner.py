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

    W is the lateral deviation from the default step width l_p.

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

    # Soft viability slack.
    viability_slack_x: float
    viability_slack_y: float

    # Effective lower bound imposed on total step time:
    #
    # T >= max(T_min, elapsed_time + timing_gap)
    timing_lower_bound: float

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
            - linear DCM equality, Eq. (19)
            - online step location / timing QP, Eq. (20)
            - soft viability constraints
            - online timing causality:
                  T >= max(T_min, t + T_gap)

    All numerical walking parameters are supplied from run.py.
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

        # ----------------------------------------------------
        # Sagittal viability
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Lateral viability
        # Appendix B, Eq. (32a)-(32b)
        # ----------------------------------------------------

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

            # Left stance -> right swing foot.
            delta_y_nom = (
                -p.default_step_width
                +
                W_nom
            )

            stance_sign = +1.0

        elif stance_leg is StanceLeg.RIGHT:

            # Right stance -> left swing foot.
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
        # z = [
        #     u_Tx,
        #     u_Ty,
        #     tau,
        #     b_x,
        #     b_y,
        #     s_x,
        #     s_y,
        # ]
        #
        # s_x, s_y >= 0 are soft-viability slack variables.

        number_variables = 7

        # Constraints:
        #
        # 0: Eq. (19), x
        # 1: Eq. (19), y
        # 2: b_x + s_x >= b_x,min
        # 3: b_x - s_x <= b_x,max
        # 4: b_y + s_y >= b_y,min
        # 5: b_y - s_y <= b_y,max

        number_constraints = 6

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

            # Current convention gives positive b_y during
            # left stance.

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
        alpha_viability: float,
        timing_gap: float,
    ) -> AdaptedStep:
        """
        Solve the Stage-2 QP.

        Decision vector:

            z = [u_Tx, u_Ty, tau, b_x, b_y, s_x, s_y]

        with:

            tau = exp(omega*T)

        DCM equality, Eq. (19):

            u_T
            - (xi_mea - u_0) exp(-omega*t) tau
            + b
            = u_0

        Hard constraints:
            - landing position bounds
            - total step-time bounds
            - online timing causality
            - Eq. (19)

        Soft constraints:
            - sagittal viability bound on b_x
            - lateral viability bound on b_y

        Online timing causality:

            T >= max(T_min, elapsed_time + timing_gap)

        If elapsed_time + timing_gap exceeds T_max, the timing
        adaptation window is considered closed. The caller
        should freeze the current landing target instead of
        solving a new adaptation.
        """

        # ====================================================
        # INPUTS
        # ====================================================

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

        p = self.parameters
        vb = self.viability_bounds

        if (
            t
            >
            p.step_time_max
            +
            1.0e-12
        ):

            raise ValueError(
                "elapsed_time exceeds step_time_max."
            )

        # ====================================================
        # QP WEIGHTS / ONLINE TIMING GAP
        # ====================================================

        alpha_location = float(
            alpha_location
        )

        alpha_timing = float(
            alpha_timing
        )

        alpha_dcm = float(
            alpha_dcm
        )

        alpha_viability = float(
            alpha_viability
        )

        timing_gap = float(
            timing_gap
        )

        for name, value in (
            ("alpha_location", alpha_location),
            ("alpha_timing", alpha_timing),
            ("alpha_dcm", alpha_dcm),
            ("alpha_viability", alpha_viability),
        ):

            if (
                not math.isfinite(value)
                or
                value <= 0.0
            ):

                raise ValueError(
                    f"{name} must be finite and positive."
                )

        if (
            not math.isfinite(timing_gap)
            or
            timing_gap < 0.0
        ):

            raise ValueError(
                "timing_gap must be finite and >= 0."
            )

        # ====================================================
        # ONLINE TIMING CAUSALITY
        # ====================================================

        timing_lower_bound = max(
            p.step_time_min,
            t
            +
            timing_gap,
        )

        if (
            timing_lower_bound
            >
            p.step_time_max
            +
            1.0e-12
        ):

            raise RuntimeError(
                "Stage-2 timing adaptation window is closed. "
                "The current landing target should be frozen."
            )

        # Remove tiny floating-point overshoot at T_max.
        timing_lower_bound = min(
            timing_lower_bound,
            p.step_time_max,
        )

        tau_lower = math.exp(
            self.omega
            *
            timing_lower_bound
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

                0.0,
                0.0,
            ],
            dtype=float,
        )

        # ====================================================
        # OBJECTIVE — EQ. (20) + SOFT-VIABILITY PENALTY
        # ====================================================

        weights = np.array(
            [
                alpha_location,
                alpha_location,

                alpha_timing,

                alpha_dcm,
                alpha_dcm,

                alpha_viability,
                alpha_viability,
            ],
            dtype=float,
        )

        # CasADi conic solves:
        #
        #     0.5 z^T H z + g^T z
        #
        # Therefore, for
        #
        #     sum_i w_i (z_i - z_ref_i)^2,
        #
        # use:
        #
        #     H = 2 diag(w)
        #     g = -2 w .* z_ref
        #
        # The constant term is irrelevant to optimization.

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

        # ====================================================
        # LATERAL BOUNDS
        # ====================================================

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

        # ====================================================
        # FULL LINEAR CONSTRAINT MATRIX
        # ====================================================

        A = np.array(
            [
                # --------------------------------------------
                # Eq. (19), x
                #
                # u_Tx - dcm_factor_x*tau + b_x = u_0x
                # --------------------------------------------
                [
                    1.0,
                    0.0,
                    -dcm_factor[0],
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],

                # --------------------------------------------
                # Eq. (19), y
                # --------------------------------------------
                [
                    0.0,
                    1.0,
                    -dcm_factor[1],
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],

                # --------------------------------------------
                # Soft sagittal lower viability:
                #
                # b_x + s_x >= b_x,min
                # --------------------------------------------
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                ],

                # --------------------------------------------
                # Soft sagittal upper viability:
                #
                # b_x - s_x <= b_x,max
                # --------------------------------------------
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    0.0,
                ],

                # --------------------------------------------
                # Soft lateral lower viability:
                #
                # b_y + s_y >= b_y,min
                # --------------------------------------------
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ],

                # --------------------------------------------
                # Soft lateral upper viability:
                #
                # b_y - s_y <= b_y,max
                # --------------------------------------------
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                ],
            ],
            dtype=float,
        )

        constraint_lower = np.array(
            [
                # Eq. (19), x
                u0[0],

                # Eq. (19), y
                u0[1],

                # b_x + s_x >= b_x,min
                vb.bx_min,

                # b_x - s_x <= b_x,max
                -np.inf,

                # b_y + s_y >= b_y,min
                by_lower,

                # b_y - s_y <= b_y,max
                -np.inf,
            ],
            dtype=float,
        )

        constraint_upper = np.array(
            [
                # Eq. (19), x
                u0[0],

                # Eq. (19), y
                u0[1],

                # b_x + s_x >= b_x,min
                +np.inf,

                # b_x - s_x <= b_x,max
                vb.bx_max,

                # b_y + s_y >= b_y,min
                +np.inf,

                # b_y - s_y <= b_y,max
                by_upper,
            ],
            dtype=float,
        )

        # ====================================================
        # VARIABLE BOUNDS
        # ====================================================
        #
        # Hard:
        #   landing position
        #   timing
        #   non-negative slack
        #
        # Soft:
        #   b_x, b_y viability
        # ====================================================

        lower_bounds = np.array(
            [
                # u_Tx
                u0[0]
                +
                p.step_length_min,

                # u_Ty
                u0[1]
                +
                lateral_step_lower,

                # tau
                tau_lower,

                # b_x
                -np.inf,

                # b_y
                -np.inf,

                # s_x
                0.0,

                # s_y
                0.0,
            ],
            dtype=float,
        )

        upper_bounds = np.array(
            [
                # u_Tx
                u0[0]
                +
                p.step_length_max,

                # u_Ty
                u0[1]
                +
                lateral_step_upper,

                # tau
                vb.tau_max,

                # b_x
                +np.inf,

                # b_y
                +np.inf,

                # s_x
                +np.inf,

                # s_y
                +np.inf,
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
                A
            ),
            lba=ca.DM(
                constraint_lower
            ),
            uba=ca.DM(
                constraint_upper
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

        # ====================================================
        # EXTRACT SOLUTION
        # ====================================================

        solution = np.asarray(
            result[
                "x"
            ],
            dtype=float,
        ).reshape(
            7
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

        slack_x = float(
            solution[5]
        )

        slack_y = float(
            solution[6]
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

        # Only the first two rows are equality constraints.
        equality_residual = (
            A[
                0:2,
                :
            ]
            @
            solution
            -
            u0
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
            np.sum(
                weights
                *
                error**2
            )
        )

        # Small numerical negative slack should not be exposed.
        slack_x = max(
            0.0,
            slack_x,
        )

        slack_y = max(
            0.0,
            slack_y,
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

            viability_slack_x=slack_x,
            viability_slack_y=slack_y,

            timing_lower_bound=float(
                timing_lower_bound
            ),

            objective=objective,

            max_equality_residual=(
                max_equality_residual
            ),
        )
