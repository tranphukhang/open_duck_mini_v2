# step_timing_adaptation/swing_trajectory.py

from __future__ import annotations

from dataclasses import dataclass
import math

import casadi as ca
import numpy as np


# ============================================================
# HORIZONTAL SWING SAMPLE
# ============================================================

@dataclass(frozen=True)
class HorizontalSwingSample:
    """
    Desired horizontal swing-foot state at one sample.

    All vectors use the world-frame horizontal coordinates:

        x : forward
        y : left
    """

    time: float

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    landing_time: float
    landing_position: np.ndarray

    # Polynomial coefficients for x and y.
    #
    # coefficients[axis, i] corresponds to:
    #
    #     X(s) = sum_{i=0}^5 c_i s^i
    #
    # where:
    #
    #     s = t - t_previous
    coefficients: np.ndarray

    max_boundary_residual: float



# ============================================================
# VERTICAL SWING QP PARAMETERS
# ============================================================

@dataclass(frozen=True)
class VerticalSwingQPParameters:
    """
    Runtime parameters for the vertical swing-foot QP.

    The numerical values are intentionally supplied from run.py.
    """

    desired_height: float
    maximum_height: float

    constraint_samples: int

    coefficient_regularization: float

    bound_tolerance: float
    max_refinements: int


# ============================================================
# VERTICAL SWING SAMPLE
# ============================================================

@dataclass(frozen=True)
class VerticalSwingSample:
    """
    Desired vertical swing-foot state at one sample.

    Height is measured relative to the current ground/contact
    plane of the step:

        z = 0 at lift-off and touchdown.
    """

    time: float

    height: float
    velocity: float
    acceleration: float

    landing_time: float

    coefficients: np.ndarray

    midpoint_height: float

    continuous_min_height: float
    continuous_max_height: float

    max_boundary_residual: float

    refinement_iterations: int


# ============================================================
# ONLINE SWING TRAJECTORY
# ============================================================

class OnlineSwingFootTrajectory:
    """
    Online horizontal swing-foot trajectory regeneration.

    This implements the horizontal part of the swing-foot
    adaptation strategy used by Khadiv et al.

    At each update, a fifth-order polynomial is regenerated
    from the desired swing-foot state at the previous sample

        X(t_{k-1})
        X_dot(t_{k-1})
        X_ddot(t_{k-1})

    to the latest planner landing target

        X(T)      = u_T
        X_dot(T)  = 0
        X_ddot(T) = 0

    The polynomial is represented in local time

        s = t - t_{k-1}

    for improved numerical conditioning.

    This class does not contain walking parameters.
    Runtime configuration remains in run.py.
    """

    def __init__(
        self,
    ) -> None:

        self._initialized = False

        self._previous_time = 0.0

        self._position = np.zeros(
            2,
            dtype=float,
        )

        self._velocity = np.zeros(
            2,
            dtype=float,
        )

        self._acceleration = np.zeros(
            2,
            dtype=float,
        )

        # ----------------------------------------------------
        # Vertical trajectory state.
        # ----------------------------------------------------

        self._vertical_initialized = False

        self._vertical_previous_time = 0.0

        self._vertical_height = 0.0
        self._vertical_velocity = 0.0
        self._vertical_acceleration = 0.0

        self._vertical_solver_counter = 0


    # ========================================================
    # STATE
    # ========================================================

    @property
    def initialized(
        self,
    ) -> bool:

        return bool(
            self._initialized
        )


    @property
    def previous_time(
        self,
    ) -> float:

        self._require_initialized()

        return float(
            self._previous_time
        )


    @property
    def position(
        self,
    ) -> np.ndarray:

        self._require_initialized()

        return (
            self._position.copy()
        )


    @property
    def velocity(
        self,
    ) -> np.ndarray:

        self._require_initialized()

        return (
            self._velocity.copy()
        )


    @property
    def acceleration(
        self,
    ) -> np.ndarray:

        self._require_initialized()

        return (
            self._acceleration.copy()
        )


    # ========================================================
    # RESET AT LIFT-OFF
    # ========================================================

    def reset_horizontal(
        self,
        initial_position,
        initial_velocity=None,
        initial_acceleration=None,
        start_time: float = 0.0,
    ) -> None:
        """
        Initialize a new swing phase.

        initial_position:
            [x, y] desired/actual swing-foot position.

        initial_velocity:
            [vx, vy]. Defaults to zero.

        initial_acceleration:
            [ax, ay]. Defaults to zero.

        start_time:
            Time measured from the beginning of the current
            walking step. Normally zero.
        """

        position = self._as_vector2(
            initial_position,
            "initial_position",
        )

        if initial_velocity is None:

            velocity = np.zeros(
                2,
                dtype=float,
            )

        else:

            velocity = self._as_vector2(
                initial_velocity,
                "initial_velocity",
            )

        if initial_acceleration is None:

            acceleration = np.zeros(
                2,
                dtype=float,
            )

        else:

            acceleration = self._as_vector2(
                initial_acceleration,
                "initial_acceleration",
            )

        t0 = float(
            start_time
        )

        if (
            not math.isfinite(t0)
            or
            t0 < 0.0
        ):

            raise ValueError(
                "start_time must be finite and >= 0."
            )

        self._previous_time = t0

        self._position = (
            position.copy()
        )

        self._velocity = (
            velocity.copy()
        )

        self._acceleration = (
            acceleration.copy()
        )

        self._initialized = True


    # ========================================================
    # ONLINE UPDATE
    # ========================================================

    def update_horizontal(
        self,
        current_time: float,
        landing_time: float,
        landing_position,
    ) -> HorizontalSwingSample:
        """
        Regenerate and evaluate the fifth-order trajectory.

        current_time:
            Current elapsed time from the start of the step.

        landing_time:
            Adapted total step duration T from the planner.

        landing_position:
            Adapted landing location [u_Tx, u_Ty].

        The generated polynomial connects the previous desired
        state continuously up to acceleration level to the new
        landing target.
        """

        self._require_initialized()

        t_current = float(
            current_time
        )

        T = float(
            landing_time
        )

        target = self._as_vector2(
            landing_position,
            "landing_position",
        )

        if (
            not math.isfinite(t_current)
            or
            t_current < 0.0
        ):

            raise ValueError(
                "current_time must be finite and >= 0."
            )

        if (
            not math.isfinite(T)
            or
            T <= 0.0
        ):

            raise ValueError(
                "landing_time must be finite and positive."
            )

        tolerance = 1.0e-12

        if (
            t_current
            <
            self._previous_time
            -
            tolerance
        ):

            raise ValueError(
                "current_time must be >= previous sample time."
            )

        if (
            T
            <
            t_current
            -
            tolerance
        ):

            raise ValueError(
                "landing_time must be >= current_time."
            )

        # If current_time is numerically equal to T, the swing
        # has reached touchdown. Return the exact terminal state.
        if (
            T
            -
            t_current
            <=
            tolerance
        ):

            coefficients = np.zeros(
                (
                    2,
                    6,
                ),
                dtype=float,
            )

            coefficients[
                :,
                0
            ] = target

            position = (
                target.copy()
            )

            velocity = np.zeros(
                2,
                dtype=float,
            )

            acceleration = np.zeros(
                2,
                dtype=float,
            )

            self._previous_time = t_current

            self._position = (
                position.copy()
            )

            self._velocity = (
                velocity.copy()
            )

            self._acceleration = (
                acceleration.copy()
            )

            return HorizontalSwingSample(
                time=t_current,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                landing_time=T,
                landing_position=(
                    target.copy()
                ),
                coefficients=coefficients,
                max_boundary_residual=0.0,
            )

        t_previous = float(
            self._previous_time
        )

        horizon = (
            T
            -
            t_previous
        )

        if horizon <= tolerance:

            raise RuntimeError(
                "Remaining swing horizon is too small "
                "to regenerate a fifth-order polynomial."
            )

        evaluation_time = (
            t_current
            -
            t_previous
        )

        if (
            evaluation_time
            >
            horizon
            +
            tolerance
        ):

            raise RuntimeError(
                "Current sample lies after the landing time."
            )

        # Clamp only numerical roundoff.
        evaluation_time = min(
            max(
                evaluation_time,
                0.0,
            ),
            horizon,
        )

        previous_position = (
            self._position.copy()
        )

        previous_velocity = (
            self._velocity.copy()
        )

        previous_acceleration = (
            self._acceleration.copy()
        )

        coefficients = np.zeros(
            (
                2,
                6,
            ),
            dtype=float,
        )

        max_boundary_residual = 0.0

        for axis in range(
            2
        ):

            axis_coefficients = (
                self._solve_quintic(
                    initial_position=(
                        previous_position[
                            axis
                        ]
                    ),
                    initial_velocity=(
                        previous_velocity[
                            axis
                        ]
                    ),
                    initial_acceleration=(
                        previous_acceleration[
                            axis
                        ]
                    ),
                    final_position=(
                        target[
                            axis
                        ]
                    ),
                    horizon=(
                        horizon
                    ),
                )
            )

            coefficients[
                axis,
                :
            ] = axis_coefficients

            boundary_residual = (
                self._compute_boundary_residual(
                    coefficients=(
                        axis_coefficients
                    ),
                    horizon=(
                        horizon
                    ),
                    initial_position=(
                        previous_position[
                            axis
                        ]
                    ),
                    initial_velocity=(
                        previous_velocity[
                            axis
                        ]
                    ),
                    initial_acceleration=(
                        previous_acceleration[
                            axis
                        ]
                    ),
                    final_position=(
                        target[
                            axis
                        ]
                    ),
                )
            )

            max_boundary_residual = max(
                max_boundary_residual,
                boundary_residual,
            )

        position = np.zeros(
            2,
            dtype=float,
        )

        velocity = np.zeros(
            2,
            dtype=float,
        )

        acceleration = np.zeros(
            2,
            dtype=float,
        )

        for axis in range(
            2
        ):

            (
                position[
                    axis
                ],
                velocity[
                    axis
                ],
                acceleration[
                    axis
                ],
            ) = self._evaluate_quintic(
                coefficients=(
                    coefficients[
                        axis,
                        :
                    ]
                ),
                local_time=(
                    evaluation_time
                ),
            )

        if not np.all(
            np.isfinite(
                position
            )
        ):

            raise RuntimeError(
                "Non-finite swing position generated."
            )

        if not np.all(
            np.isfinite(
                velocity
            )
        ):

            raise RuntimeError(
                "Non-finite swing velocity generated."
            )

        if not np.all(
            np.isfinite(
                acceleration
            )
        ):

            raise RuntimeError(
                "Non-finite swing acceleration generated."
            )

        # The next online regeneration starts exactly from this
        # desired state. This implements the paper's continuity
        # condition at t_{k-1}.
        self._previous_time = t_current

        self._position = (
            position.copy()
        )

        self._velocity = (
            velocity.copy()
        )

        self._acceleration = (
            acceleration.copy()
        )

        return HorizontalSwingSample(
            time=t_current,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            landing_time=T,
            landing_position=(
                target.copy()
            ),
            coefficients=(
                coefficients.copy()
            ),
            max_boundary_residual=float(
                max_boundary_residual
            ),
        )



    # ========================================================
    # VERTICAL SWING — RESET
    # ========================================================

    def reset_vertical(
        self,
    ) -> None:
        """
        Initialize a new vertical swing phase.

        The paper uses the boundary conditions

            z(0)     = 0
            z_dot(0) = 0
            z_ddot(0)= 0

        for flat-ground walking. Therefore the desired vertical
        swing state is reset exactly to zero at lift-off.
        """

        self._vertical_previous_time = 0.0

        self._vertical_height = 0.0
        self._vertical_velocity = 0.0
        self._vertical_acceleration = 0.0

        self._vertical_initialized = True


    # ========================================================
    # VERTICAL SWING — ONLINE 9TH-ORDER QP
    # ========================================================

    def update_vertical(
        self,
        current_time: float,
        landing_time: float,
        parameters: VerticalSwingQPParameters,
    ) -> VerticalSwingSample:
        """
        Regenerate the vertical swing-foot polynomial.

        Paper Eq. (21):

            minimize
                || z(T/2) - z_des ||^2

            subject to
                0 <= z(t) <= z_max

                z(0)       = 0
                z(t_k-1)   = z_k-1
                z(T)       = 0

                z_dot(0)     = 0
                z_dot(t_k-1) = z_dot_k-1
                z_dot(T)     = 0

                z_ddot(0)     = 0
                z_ddot(t_k-1) = z_ddot_k-1
                z_ddot(T)     = 0

        A ninth-order polynomial is used:

            z(r) = sum_{i=0}^9 c_i r^i

        with normalized phase

            r = t / T.

        This normalized representation is mathematically
        equivalent to a ninth-order polynomial in physical time
        but has much better numerical conditioning.

        The paper writes the height bound continuously in time.
        The QP starts with uniformly sampled linear inequalities.
        After each solve, all real extrema of the polynomial are
        checked. Any violating extremum is added as a cutting
        constraint and the QP is solved again. This refinement
        enforces the continuous bound to the configured numerical
        tolerance.

        A very small coefficient regularization is added only to
        remove numerical non-uniqueness, especially at the first
        sample where t_(k-1) = 0 duplicates the initial boundary
        conditions.
        """

        self._require_vertical_initialized()

        self._validate_vertical_parameters(
            parameters
        )

        t_current = float(
            current_time
        )

        T = float(
            landing_time
        )

        if (
            not math.isfinite(t_current)
            or
            t_current < 0.0
        ):

            raise ValueError(
                "current_time must be finite and >= 0."
            )

        if (
            not math.isfinite(T)
            or
            T <= 0.0
        ):

            raise ValueError(
                "landing_time must be finite and positive."
            )

        tolerance = 1.0e-12

        if (
            t_current
            <
            self._vertical_previous_time
            -
            tolerance
        ):

            raise ValueError(
                "current_time must be >= previous vertical "
                "sample time."
            )

        if (
            T
            <
            t_current
            -
            tolerance
        ):

            raise ValueError(
                "landing_time must be >= current_time."
            )

        # ----------------------------------------------------
        # Exact touchdown state.
        # ----------------------------------------------------

        if (
            T
            -
            t_current
            <=
            tolerance
        ):

            coefficients = np.zeros(
                10,
                dtype=float,
            )

            self._vertical_previous_time = t_current

            self._vertical_height = 0.0
            self._vertical_velocity = 0.0
            self._vertical_acceleration = 0.0

            return VerticalSwingSample(
                time=t_current,
                height=0.0,
                velocity=0.0,
                acceleration=0.0,
                landing_time=T,
                coefficients=coefficients,
                midpoint_height=0.0,
                continuous_min_height=0.0,
                continuous_max_height=0.0,
                max_boundary_residual=0.0,
                refinement_iterations=0,
            )

        t_previous = float(
            self._vertical_previous_time
        )

        if (
            T
            <=
            t_previous
            +
            tolerance
        ):

            raise RuntimeError(
                "Remaining vertical swing horizon is too small."
            )

        previous_state = np.array(
            [
                self._vertical_height,
                self._vertical_velocity,
                self._vertical_acceleration,
            ],
            dtype=float,
        )

        # ====================================================
        # EQ. (21) EQUALITY CONSTRAINTS
        # ====================================================

        equality_rows = []
        equality_values = []

        # ----------------------------------------------------
        # Start of the complete step: t = 0.
        # ----------------------------------------------------

        for derivative_order in range(
            3
        ):

            equality_rows.append(
                self._vertical_basis(
                    normalized_time=0.0,
                    derivative_order=derivative_order,
                    landing_time=T,
                )
            )

            equality_values.append(
                0.0
            )

        # ----------------------------------------------------
        # Previous desired sample: t = t_(k-1).
        # ----------------------------------------------------

        normalized_previous_time = (
            t_previous
            /
            T
        )

        for derivative_order in range(
            3
        ):

            equality_rows.append(
                self._vertical_basis(
                    normalized_time=(
                        normalized_previous_time
                    ),
                    derivative_order=(
                        derivative_order
                    ),
                    landing_time=T,
                )
            )

            equality_values.append(
                previous_state[
                    derivative_order
                ]
            )

        # ----------------------------------------------------
        # Touchdown: t = T.
        # ----------------------------------------------------

        for derivative_order in range(
            3
        ):

            equality_rows.append(
                self._vertical_basis(
                    normalized_time=1.0,
                    derivative_order=derivative_order,
                    landing_time=T,
                )
            )

            equality_values.append(
                0.0
            )

        equality_matrix_full = np.asarray(
            equality_rows,
            dtype=float,
        )

        equality_vector_full = np.asarray(
            equality_values,
            dtype=float,
        )

        (
            equality_matrix,
            equality_vector,
        ) = self._remove_redundant_equalities(
            matrix=(
                equality_matrix_full
            ),
            vector=(
                equality_vector_full
            ),
        )

        # ====================================================
        # OBJECTIVE
        # ====================================================

        midpoint_basis = self._vertical_basis(
            normalized_time=0.5,
            derivative_order=0,
            landing_time=T,
        )

        regularization = float(
            parameters.coefficient_regularization
        )

        H = (
            2.0
            *
            (
                np.outer(
                    midpoint_basis,
                    midpoint_basis,
                )
                +
                regularization
                *
                np.eye(
                    10,
                    dtype=float,
                )
            )
        )

        g = (
            -2.0
            *
            float(
                parameters.desired_height
            )
            *
            midpoint_basis
        )

        # ====================================================
        # INITIAL HEIGHT-BOUND SAMPLE LOCATIONS
        # ====================================================

        height_constraint_locations = list(
            np.linspace(
                0.0,
                1.0,
                int(
                    parameters.constraint_samples
                ),
                dtype=float,
            )
        )

        coefficients = None

        continuous_min_height = math.nan
        continuous_max_height = math.nan

        refinement_iterations = 0

        # ====================================================
        # CUTTING-PLANE REFINEMENT OF CONTINUOUS HEIGHT BOUNDS
        # ====================================================

        for refinement_iteration in range(
            int(
                parameters.max_refinements
            )
            +
            1
        ):

            refinement_iterations = (
                refinement_iteration
            )

            height_matrix = np.vstack(
                [
                    self._vertical_basis(
                        normalized_time=(
                            location
                        ),
                        derivative_order=0,
                        landing_time=T,
                    )
                    for location
                    in height_constraint_locations
                ]
            )

            constraint_matrix = np.vstack(
                [
                    equality_matrix,
                    height_matrix,
                ]
            )

            number_equalities = (
                equality_matrix.shape[
                    0
                ]
            )

            number_height_constraints = (
                height_matrix.shape[
                    0
                ]
            )

            constraint_lower = np.concatenate(
                [
                    equality_vector,
                    np.zeros(
                        number_height_constraints,
                        dtype=float,
                    ),
                ]
            )

            constraint_upper = np.concatenate(
                [
                    equality_vector,
                    np.full(
                        number_height_constraints,
                        float(
                            parameters.maximum_height
                        ),
                        dtype=float,
                    ),
                ]
            )

            coefficients = self._solve_vertical_qp(
                H=H,
                g=g,
                constraint_matrix=(
                    constraint_matrix
                ),
                constraint_lower=(
                    constraint_lower
                ),
                constraint_upper=(
                    constraint_upper
                ),
            )

            (
                continuous_min_height,
                continuous_max_height,
                minimum_location,
                maximum_location,
            ) = self._continuous_vertical_extrema(
                coefficients=(
                    coefficients
                ),
            )

            lower_violation = (
                continuous_min_height
                <
                -float(
                    parameters.bound_tolerance
                )
            )

            upper_violation = (
                continuous_max_height
                >
                float(
                    parameters.maximum_height
                )
                +
                float(
                    parameters.bound_tolerance
                )
            )

            if (
                not lower_violation
                and
                not upper_violation
            ):

                break

            if (
                refinement_iteration
                >=
                int(
                    parameters.max_refinements
                )
            ):

                raise RuntimeError(
                    "Vertical swing QP could not enforce the "
                    "continuous height bounds within the "
                    "configured refinement limit."
                )

            if lower_violation:

                self._append_unique_location(
                    height_constraint_locations,
                    minimum_location,
                )

            if upper_violation:

                self._append_unique_location(
                    height_constraint_locations,
                    maximum_location,
                )

        if coefficients is None:

            raise RuntimeError(
                "Vertical swing QP did not return coefficients."
            )

        # ====================================================
        # CURRENT DESIRED STATE
        # ====================================================

        normalized_current_time = (
            t_current
            /
            T
        )

        current_height = float(
            self._vertical_basis(
                normalized_time=(
                    normalized_current_time
                ),
                derivative_order=0,
                landing_time=T,
            )
            @
            coefficients
        )

        current_velocity = float(
            self._vertical_basis(
                normalized_time=(
                    normalized_current_time
                ),
                derivative_order=1,
                landing_time=T,
            )
            @
            coefficients
        )

        current_acceleration = float(
            self._vertical_basis(
                normalized_time=(
                    normalized_current_time
                ),
                derivative_order=2,
                landing_time=T,
            )
            @
            coefficients
        )

        midpoint_height = float(
            midpoint_basis
            @
            coefficients
        )

        if not all(
            math.isfinite(
                value
            )
            for value in (
                current_height,
                current_velocity,
                current_acceleration,
                midpoint_height,
                continuous_min_height,
                continuous_max_height,
            )
        ):

            raise RuntimeError(
                "Non-finite vertical swing state generated."
            )

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        boundary_residual = (
            equality_matrix_full
            @
            coefficients
            -
            equality_vector_full
        )

        max_boundary_residual = float(
            np.max(
                np.abs(
                    boundary_residual
                )
            )
        )

        # ====================================================
        # STORE DESIRED STATE FOR NEXT REGENERATION
        # ====================================================

        self._vertical_previous_time = (
            t_current
        )

        self._vertical_height = (
            current_height
        )

        self._vertical_velocity = (
            current_velocity
        )

        self._vertical_acceleration = (
            current_acceleration
        )

        return VerticalSwingSample(
            time=t_current,
            height=current_height,
            velocity=current_velocity,
            acceleration=current_acceleration,
            landing_time=T,
            coefficients=(
                coefficients.copy()
            ),
            midpoint_height=midpoint_height,
            continuous_min_height=float(
                continuous_min_height
            ),
            continuous_max_height=float(
                continuous_max_height
            ),
            max_boundary_residual=(
                max_boundary_residual
            ),
            refinement_iterations=int(
                refinement_iterations
            ),
        )


    # ========================================================
    # VERTICAL QP SOLVER
    # ========================================================

    def _solve_vertical_qp(
        self,
        H,
        g,
        constraint_matrix,
        constraint_lower,
        constraint_upper,
    ) -> np.ndarray:

        H = np.asarray(
            H,
            dtype=float,
        )

        g = np.asarray(
            g,
            dtype=float,
        ).reshape(-1)

        A = np.asarray(
            constraint_matrix,
            dtype=float,
        )

        lba = np.asarray(
            constraint_lower,
            dtype=float,
        ).reshape(-1)

        uba = np.asarray(
            constraint_upper,
            dtype=float,
        ).reshape(-1)

        if H.shape != (
            10,
            10,
        ):

            raise ValueError(
                "Vertical QP Hessian must be 10x10."
            )

        if g.shape != (
            10,
        ):

            raise ValueError(
                "Vertical QP gradient must contain 10 values."
            )

        if (
            A.ndim != 2
            or
            A.shape[
                1
            ]
            !=
            10
        ):

            raise ValueError(
                "Vertical QP constraint matrix must have "
                "10 columns."
            )

        if (
            lba.shape[
                0
            ]
            !=
            A.shape[
                0
            ]
            or
            uba.shape[
                0
            ]
            !=
            A.shape[
                0
            ]
        ):

            raise ValueError(
                "Vertical QP constraint bounds have "
                "inconsistent dimensions."
            )

        self._vertical_solver_counter += 1

        solver_name = (
            "vertical_swing_qp_"
            f"{id(self)}_"
            f"{self._vertical_solver_counter}"
        )

        qp_structure = {
            "h": ca.Sparsity.dense(
                10,
                10,
            ),
            "a": ca.Sparsity.dense(
                A.shape[
                    0
                ],
                10,
            ),
        }

        solver = ca.conic(
            solver_name,
            "qrqp",
            qp_structure,
            {
                "print_header": False,
                "print_iter": False,
                "print_info": False,
                "error_on_fail": False,
                "max_iter": 1000,
            },
        )

        result = solver(
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
                np.full(
                    10,
                    -np.inf,
                    dtype=float,
                )
            ),
            ubx=ca.DM(
                np.full(
                    10,
                    +np.inf,
                    dtype=float,
                )
            ),
        )

        stats = (
            solver.stats()
        )

        if not bool(
            stats.get(
                "success",
                False,
            )
        ):

            raise RuntimeError(
                "Vertical swing QP failed.\n"
                f"status = "
                f"{stats.get('return_status', 'unknown')}"
            )

        coefficients = np.asarray(
            result[
                "x"
            ],
            dtype=float,
        ).reshape(
            10
        )

        if not np.all(
            np.isfinite(
                coefficients
            )
        ):

            raise RuntimeError(
                "Vertical swing QP returned non-finite "
                "coefficients."
            )

        return coefficients


    # ========================================================
    # VERTICAL POLYNOMIAL BASIS
    # ========================================================

    @staticmethod
    def _vertical_basis(
        normalized_time: float,
        derivative_order: int,
        landing_time: float,
    ) -> np.ndarray:

        r = float(
            normalized_time
        )

        T = float(
            landing_time
        )

        if (
            not math.isfinite(r)
            or
            not math.isfinite(T)
            or
            T <= 0.0
        ):

            raise ValueError(
                "Invalid normalized time or landing time."
            )

        if derivative_order not in (
            0,
            1,
            2,
        ):

            raise ValueError(
                "derivative_order must be 0, 1, or 2."
            )

        basis = np.zeros(
            10,
            dtype=float,
        )

        for power in range(
            10
        ):

            if derivative_order == 0:

                basis[
                    power
                ] = (
                    r**power
                )

            elif (
                derivative_order == 1
                and
                power >= 1
            ):

                basis[
                    power
                ] = (
                    power
                    *
                    r**(
                        power
                        -
                        1
                    )
                    /
                    T
                )

            elif (
                derivative_order == 2
                and
                power >= 2
            ):

                basis[
                    power
                ] = (
                    power
                    *
                    (
                        power
                        -
                        1
                    )
                    *
                    r**(
                        power
                        -
                        2
                    )
                    /
                    (
                        T**2
                    )
                )

        return basis


    # ========================================================
    # VERTICAL CONTINUOUS EXTREMA
    # ========================================================

    @staticmethod
    def _continuous_vertical_extrema(
        coefficients,
    ) -> tuple[float, float, float, float]:

        c = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(-1)

        if c.shape != (
            10,
        ):

            raise ValueError(
                "Vertical coefficients must contain 10 values."
            )

        # dz/dr has degree <= 8.
        derivative_coefficients_ascending = np.array(
            [
                power
                *
                c[
                    power
                ]
                for power in range(
                    1,
                    10,
                )
            ],
            dtype=float,
        )

        # np.roots expects descending powers.
        derivative_coefficients_descending = (
            derivative_coefficients_ascending[
                ::-1
            ]
        )

        # Remove numerically zero leading coefficients.
        while (
            derivative_coefficients_descending.size
            >
            1
            and
            abs(
                derivative_coefficients_descending[
                    0
                ]
            )
            <
            1.0e-14
        ):

            derivative_coefficients_descending = (
                derivative_coefficients_descending[
                    1:
                ]
            )

        candidate_locations = [
            0.0,
            1.0,
        ]

        if np.any(
            np.abs(
                derivative_coefficients_descending
            )
            >
            1.0e-14
        ):

            roots = np.roots(
                derivative_coefficients_descending
            )

            for root in roots:

                if (
                    abs(
                        float(
                            np.imag(
                                root
                            )
                        )
                    )
                    <=
                    1.0e-9
                ):

                    location = float(
                        np.real(
                            root
                        )
                    )

                    if (
                        location
                        >
                        0.0
                        and
                        location
                        <
                        1.0
                    ):

                        candidate_locations.append(
                            location
                        )

        candidate_locations = np.asarray(
            candidate_locations,
            dtype=float,
        )

        heights = np.array(
            [
                np.polynomial.polynomial.polyval(
                    location,
                    c,
                )
                for location
                in candidate_locations
            ],
            dtype=float,
        )

        minimum_index = int(
            np.argmin(
                heights
            )
        )

        maximum_index = int(
            np.argmax(
                heights
            )
        )

        return (
            float(
                heights[
                    minimum_index
                ]
            ),
            float(
                heights[
                    maximum_index
                ]
            ),
            float(
                candidate_locations[
                    minimum_index
                ]
            ),
            float(
                candidate_locations[
                    maximum_index
                ]
            ),
        )


    # ========================================================
    # VERTICAL EQUALITY REDUCTION
    # ========================================================

    @staticmethod
    def _remove_redundant_equalities(
        matrix,
        vector,
    ) -> tuple[np.ndarray, np.ndarray]:

        A = np.asarray(
            matrix,
            dtype=float,
        )

        b = np.asarray(
            vector,
            dtype=float,
        ).reshape(-1)

        if (
            A.ndim != 2
            or
            A.shape[
                0
            ]
            !=
            b.shape[
                0
            ]
        ):

            raise ValueError(
                "Invalid equality system."
            )

        selected_indices = []

        current_rank = 0

        for row_index in range(
            A.shape[
                0
            ]
        ):

            candidate_indices = (
                selected_indices
                +
                [
                    row_index
                ]
            )

            candidate_matrix = (
                A[
                    candidate_indices,
                    :
                ]
            )

            candidate_rank = int(
                np.linalg.matrix_rank(
                    candidate_matrix,
                    tol=1.0e-10,
                )
            )

            if (
                candidate_rank
                >
                current_rank
            ):

                selected_indices.append(
                    row_index
                )

                current_rank = (
                    candidate_rank
                )

                continue

            # Redundant row: verify that its RHS is consistent
            # with the already selected equalities.
            if selected_indices:

                selected_matrix = (
                    A[
                        selected_indices,
                        :
                    ]
                )

                selected_vector = (
                    b[
                        selected_indices
                    ]
                )

                least_squares_solution = np.linalg.lstsq(
                    selected_matrix,
                    selected_vector,
                    rcond=None,
                )[
                    0
                ]

                redundant_residual = abs(
                    float(
                        A[
                            row_index,
                            :
                        ]
                        @
                        least_squares_solution
                        -
                        b[
                            row_index
                        ]
                    )
                )

                if (
                    redundant_residual
                    >
                    1.0e-8
                ):

                    raise RuntimeError(
                        "Vertical swing equality constraints "
                        "are inconsistent."
                    )

        if not selected_indices:

            raise RuntimeError(
                "No independent vertical equality constraints."
            )

        return (
            A[
                selected_indices,
                :
            ],
            b[
                selected_indices
            ],
        )


    # ========================================================
    # VERTICAL PARAMETER VALIDATION
    # ========================================================

    @staticmethod
    def _validate_vertical_parameters(
        parameters: VerticalSwingQPParameters,
    ) -> None:

        desired_height = float(
            parameters.desired_height
        )

        maximum_height = float(
            parameters.maximum_height
        )

        regularization = float(
            parameters.coefficient_regularization
        )

        bound_tolerance = float(
            parameters.bound_tolerance
        )

        constraint_samples = int(
            parameters.constraint_samples
        )

        max_refinements = int(
            parameters.max_refinements
        )

        for name, value in (
            ("desired_height", desired_height),
            ("maximum_height", maximum_height),
            ("coefficient_regularization", regularization),
            ("bound_tolerance", bound_tolerance),
        ):

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    f"{name} must be finite."
                )

        if desired_height <= 0.0:

            raise ValueError(
                "desired_height must be positive."
            )

        if maximum_height <= 0.0:

            raise ValueError(
                "maximum_height must be positive."
            )

        if (
            desired_height
            >
            maximum_height
        ):

            raise ValueError(
                "desired_height must be <= maximum_height."
            )

        if constraint_samples < 3:

            raise ValueError(
                "constraint_samples must be >= 3."
            )

        if regularization <= 0.0:

            raise ValueError(
                "coefficient_regularization must be positive."
            )

        if bound_tolerance <= 0.0:

            raise ValueError(
                "bound_tolerance must be positive."
            )

        if max_refinements < 0:

            raise ValueError(
                "max_refinements must be >= 0."
            )


    # ========================================================
    # VERTICAL HELPERS
    # ========================================================

    def _require_vertical_initialized(
        self,
    ) -> None:

        if not self._vertical_initialized:

            raise RuntimeError(
                "Vertical swing trajectory has not been "
                "initialized. Call reset_vertical() first."
            )


    @staticmethod
    def _append_unique_location(
        locations,
        new_location: float,
    ) -> None:

        value = float(
            np.clip(
                new_location,
                0.0,
                1.0,
            )
        )

        for existing in locations:

            if (
                abs(
                    float(
                        existing
                    )
                    -
                    value
                )
                <=
                1.0e-10
            ):

                return

        locations.append(
            value
        )


    # ========================================================
    # QUINTIC SOLVER
    # ========================================================

    @staticmethod
    def _solve_quintic(
        initial_position: float,
        initial_velocity: float,
        initial_acceleration: float,
        final_position: float,
        horizon: float,
    ) -> np.ndarray:
        """
        Solve:

            X(0)   = X0
            X'(0)  = V0
            X''(0) = A0

            X(H)   = Xf
            X'(H)  = 0
            X''(H) = 0

        for:

            X(s) = c0 + c1*s + ... + c5*s^5.
        """

        H = float(
            horizon
        )

        if (
            not math.isfinite(H)
            or
            H <= 0.0
        ):

            raise ValueError(
                "horizon must be finite and positive."
            )

        p0 = float(
            initial_position
        )

        v0 = float(
            initial_velocity
        )

        a0 = float(
            initial_acceleration
        )

        pf = float(
            final_position
        )

        for name, value in (
            ("initial_position", p0),
            ("initial_velocity", v0),
            ("initial_acceleration", a0),
            ("final_position", pf),
        ):

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    f"{name} must be finite."
                )

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0

        matrix = np.array(
            [
                [
                    H**3,
                    H**4,
                    H**5,
                ],
                [
                    3.0 * H**2,
                    4.0 * H**3,
                    5.0 * H**4,
                ],
                [
                    6.0 * H,
                    12.0 * H**2,
                    20.0 * H**3,
                ],
            ],
            dtype=float,
        )

        rhs = np.array(
            [
                pf
                -
                (
                    c0
                    +
                    c1 * H
                    +
                    c2 * H**2
                ),

                -(
                    c1
                    +
                    2.0 * c2 * H
                ),

                -(
                    2.0 * c2
                ),
            ],
            dtype=float,
        )

        try:

            c3_c4_c5 = np.linalg.solve(
                matrix,
                rhs,
            )

        except np.linalg.LinAlgError as exc:

            raise RuntimeError(
                "Failed to solve horizontal swing "
                "quintic polynomial."
            ) from exc

        coefficients = np.array(
            [
                c0,
                c1,
                c2,
                c3_c4_c5[
                    0
                ],
                c3_c4_c5[
                    1
                ],
                c3_c4_c5[
                    2
                ],
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                coefficients
            )
        ):

            raise RuntimeError(
                "Non-finite quintic coefficients generated."
            )

        return coefficients


    # ========================================================
    # QUINTIC EVALUATION
    # ========================================================

    @staticmethod
    def _evaluate_quintic(
        coefficients,
        local_time: float,
    ) -> tuple[float, float, float]:

        c = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(-1)

        if c.shape != (6,):

            raise ValueError(
                "Quintic coefficients must contain 6 values."
            )

        s = float(
            local_time
        )

        if not math.isfinite(
            s
        ):

            raise ValueError(
                "local_time must be finite."
            )

        position = (
            c[0]
            +
            c[1] * s
            +
            c[2] * s**2
            +
            c[3] * s**3
            +
            c[4] * s**4
            +
            c[5] * s**5
        )

        velocity = (
            c[1]
            +
            2.0 * c[2] * s
            +
            3.0 * c[3] * s**2
            +
            4.0 * c[4] * s**3
            +
            5.0 * c[5] * s**4
        )

        acceleration = (
            2.0 * c[2]
            +
            6.0 * c[3] * s
            +
            12.0 * c[4] * s**2
            +
            20.0 * c[5] * s**3
        )

        return (
            float(position),
            float(velocity),
            float(acceleration),
        )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    @classmethod
    def _compute_boundary_residual(
        cls,
        coefficients,
        horizon: float,
        initial_position: float,
        initial_velocity: float,
        initial_acceleration: float,
        final_position: float,
    ) -> float:

        (
            p_start,
            v_start,
            a_start,
        ) = cls._evaluate_quintic(
            coefficients=(
                coefficients
            ),
            local_time=0.0,
        )

        (
            p_end,
            v_end,
            a_end,
        ) = cls._evaluate_quintic(
            coefficients=(
                coefficients
            ),
            local_time=(
                horizon
            ),
        )

        residuals = np.array(
            [
                p_start
                -
                initial_position,

                v_start
                -
                initial_velocity,

                a_start
                -
                initial_acceleration,

                p_end
                -
                final_position,

                v_end,

                a_end,
            ],
            dtype=float,
        )

        return float(
            np.max(
                np.abs(
                    residuals
                )
            )
        )


    # ========================================================
    # HELPERS
    # ========================================================

    def _require_initialized(
        self,
    ) -> None:

        if not self._initialized:

            raise RuntimeError(
                "Swing trajectory has not been initialized. "
                "Call reset_horizontal() first."
            )


    @staticmethod
    def _as_vector2(
        value,
        name: str,
    ) -> np.ndarray:

        vector = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if vector.shape != (2,):

            raise ValueError(
                f"{name} must contain exactly 2 values."
            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):

            raise ValueError(
                f"{name} must be finite."
            )

        return vector
