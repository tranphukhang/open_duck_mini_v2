# step_timing_adaptation/adaptive_step_planner.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import casadi as ca


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
    following Khadiv et al.

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
    #
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

    step_location_x: float
    step_location_y: float

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
    Stage 1 of:

        Khadiv et al.,
        "Walking Control Based on Step Timing Adaptation"

    This class currently implements:

        - LIPM natural frequency
        - viability bounds
        - nominal stepping values, Eq. (7), (16), (17)

    Stage 2 QP will be added after Stage 1 validation.
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

        

    # ========================================================
    # PARAMETER VALIDATION
    # ========================================================

    def _validate_parameters(
        self,
    ) -> None:

        p = self.parameters

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
            Eq. (9), extended symmetrically to backward walking.

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

            bx_min=(
                bx_min
            ),

            bx_max=(
                bx_max
            ),

            by_max_out=(
                by_max_out
            ),

            by_max_in=(
                by_max_in
            ),

            tau_min=(
                tau_min
            ),

            tau_max=(
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

        For zero vx or vy, the corresponding term is removed
        from B_l and B_u exactly as stated in the paper.
        """

        vx = float(
            desired_velocity_x
        )

        vy = float(
            desired_velocity_y
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

                # Forward walking:
                #
                # L_nom = vx T
                #
                # L_nom <= L_max

                upper_candidates.append(
                    p.step_length_max
                    /
                    vx
                )

                # If positive lower forward bound existed,
                # this would impose a lower timing bound.
                #
                # Our L_min is negative, so it does not.

                if (
                    p.step_length_min
                    >
                    0.0
                ):

                    lower_candidates.append(
                        p.step_length_min
                        /
                        vx
                    )

            else:

                # Backward walking:
                #
                # L_nom = vx T < 0

                speed = abs(
                    vx
                )

                # Magnitude allowed backward:
                backward_limit = abs(
                    p.step_length_min
                )

                upper_candidates.append(
                    backward_limit
                    /
                    speed
                )

                if (
                    p.step_length_max
                    <
                    0.0
                ):

                    lower_candidates.append(
                        abs(
                            p.step_length_max
                        )
                        /
                        speed
                    )

        # ----------------------------------------------------
        # Lateral velocity
        # ----------------------------------------------------

        if abs(vy) > 1.0e-12:

            if vy > 0.0:

                upper_candidates.append(
                    p.step_width_max
                    /
                    vy
                )

                if (
                    p.step_width_min
                    >
                    0.0
                ):

                    lower_candidates.append(
                        p.step_width_min
                        /
                        vy
                    )

            else:

                speed = abs(
                    vy
                )

                lateral_limit = abs(
                    p.step_width_min
                )

                upper_candidates.append(
                    lateral_limit
                    /
                    speed
                )

                if (
                    p.step_width_max
                    <
                    0.0
                ):

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
        #
        # W_nom is only the lateral deviation around l_p.
        #
        # MuJoCo convention:
        #
        #     +y = left
        #
        # Left stance:
        #     next foot = right
        #     Delta y ~= -l_p
        #
        # Right stance:
        #     next foot = left
        #     Delta y ~= +l_p
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
                "Computed nominal step length "
                "violates bounds."
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
                "Computed nominal step time "
                "violates bounds."
            )

        return NominalStep(

            stance_leg=(
                stance_leg
            ),

            desired_velocity_x=(
                vx
            ),

            desired_velocity_y=(
                vy
            ),

            lower_time_bound=(
                B_l
            ),

            upper_time_bound=(
                B_u
            ),

            step_time=(
                T_nom
            ),

            step_length=(
                L_nom
            ),

            step_width_deviation=(
                W_nom
            ),

            step_displacement_x=(
                L_nom
            ),

            step_displacement_y=(
                delta_y_nom
            ),

            tau=(
                tau_nom
            ),

            dcm_offset_x=(
                bx_nom
            ),

            dcm_offset_y=(
                by_nom
            ),
        )