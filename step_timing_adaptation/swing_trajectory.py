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

    time: float

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    landing_time: float
    landing_position: np.ndarray

    coefficients: np.ndarray

    max_boundary_residual: float


# ============================================================
# VERTICAL SWING PARAMETERS
# ============================================================

@dataclass(frozen=True)
class VerticalSwingQPParameters:

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
# COMBINED SWING SAMPLE
# ============================================================

@dataclass(frozen=True)
class SwingFootSample:

    time: float

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    landing_time: float
    landing_position: np.ndarray

    horizontal: HorizontalSwingSample
    vertical: VerticalSwingSample

    max_boundary_residual: float


# ============================================================
# ONLINE SWING TRAJECTORY
# ============================================================

class OnlineSwingFootTrajectory:
    """
    Online swing-foot trajectory used together with the
    Step Timing Adaptation planner.

    Horizontal:
        fifth-order polynomial regenerated online.

    Vertical:
        ninth-order polynomial based on Eq. (21) of
        Khadiv et al.

    At every update the trajectory starts from the desired
    state generated at the previous sample and connects to
    the latest adapted landing time and landing location.

    Therefore changes in:

        u_T
        T

    can be incorporated online while maintaining continuity
    of position, velocity and acceleration.
    """

    def __init__(
        self,
    ) -> None:

        # ====================================================
        # HORIZONTAL STATE
        # ====================================================

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

        # ====================================================
        # VERTICAL STATE
        # ====================================================

        self._vertical_initialized = False

        self._vertical_previous_time = 0.0

        self._vertical_height = 0.0
        self._vertical_velocity = 0.0
        self._vertical_acceleration = 0.0

        # Last feasible solution is retained and used as a
        # numerical warm-start by the fallback solver.
        self._vertical_coefficients = None

        self._vertical_solver_counter = 0

        # ====================================================
        # COMBINED 3D
        # ====================================================

        self._vertical_reference_height = 0.0

        self._combined_initialized = False


    # ========================================================
    # BASIC PROPERTIES
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
    # RESET COMPLETE 3D SWING
    # ========================================================

    def reset_3d(
        self,
        initial_position,
        initial_velocity=None,
        initial_acceleration=None,
        start_time: float = 0.0,
    ) -> None:

        position = self._as_vector3(
            initial_position,
            "initial_position",
        )

        if initial_velocity is None:

            velocity = np.zeros(
                3,
                dtype=float,
            )

        else:

            velocity = self._as_vector3(
                initial_velocity,
                "initial_velocity",
            )

        if initial_acceleration is None:

            acceleration = np.zeros(
                3,
                dtype=float,
            )

        else:

            acceleration = self._as_vector3(
                initial_acceleration,
                "initial_acceleration",
            )

        t0 = float(
            start_time
        )

        if (
            not math.isfinite(t0)
            or
            abs(t0) > 1.0e-12
        ):

            raise ValueError(
                "reset_3d currently requires start_time = 0."
            )

        if (
            abs(
                velocity[2]
            )
            >
            1.0e-10
        ):

            raise ValueError(
                "reset_3d requires zero initial "
                "vertical velocity."
            )

        if (
            abs(
                acceleration[2]
            )
            >
            1.0e-10
        ):

            raise ValueError(
                "reset_3d requires zero initial "
                "vertical acceleration."
            )

        # World-z reference of flat terrain.
        self._vertical_reference_height = float(
            position[2]
        )

        self.reset_horizontal(
            initial_position=(
                position[
                    0:2
                ]
            ),

            initial_velocity=(
                velocity[
                    0:2
                ]
            ),

            initial_acceleration=(
                acceleration[
                    0:2
                ]
            ),

            start_time=0.0,
        )

        self.reset_vertical()

        self._combined_initialized = True


    # ========================================================
    # UPDATE COMPLETE 3D SWING
    # ========================================================

    def update_3d(
        self,
        current_time: float,
        landing_time: float,
        landing_position_xy,
        vertical_parameters: VerticalSwingQPParameters,
    ) -> SwingFootSample:

        if not self._combined_initialized:

            raise RuntimeError(
                "3D swing trajectory has not been initialized. "
                "Call reset_3d() first."
            )

        landing_xy = self._as_vector2(
            landing_position_xy,
            "landing_position_xy",
        )

        horizontal = self.update_horizontal(
            current_time=(
                current_time
            ),

            landing_time=(
                landing_time
            ),

            landing_position=(
                landing_xy
            ),
        )

        vertical = self.update_vertical(
            current_time=(
                current_time
            ),

            landing_time=(
                landing_time
            ),

            parameters=(
                vertical_parameters
            ),
        )

        position = np.array(
            [
                horizontal.position[0],
                horizontal.position[1],

                self._vertical_reference_height
                +
                vertical.height,
            ],
            dtype=float,
        )

        velocity = np.array(
            [
                horizontal.velocity[0],
                horizontal.velocity[1],
                vertical.velocity,
            ],
            dtype=float,
        )

        acceleration = np.array(
            [
                horizontal.acceleration[0],
                horizontal.acceleration[1],
                vertical.acceleration,
            ],
            dtype=float,
        )

        landing_position = np.array(
            [
                landing_xy[0],
                landing_xy[1],
                self._vertical_reference_height,
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                position
            )
        ):

            raise RuntimeError(
                "Non-finite combined swing position."
            )

        if not np.all(
            np.isfinite(
                velocity
            )
        ):

            raise RuntimeError(
                "Non-finite combined swing velocity."
            )

        if not np.all(
            np.isfinite(
                acceleration
            )
        ):

            raise RuntimeError(
                "Non-finite combined swing acceleration."
            )

        return SwingFootSample(
            time=float(
                current_time
            ),

            position=(
                position
            ),

            velocity=(
                velocity
            ),

            acceleration=(
                acceleration
            ),

            landing_time=float(
                landing_time
            ),

            landing_position=(
                landing_position
            ),

            horizontal=(
                horizontal
            ),

            vertical=(
                vertical
            ),

            max_boundary_residual=float(
                max(
                    horizontal.max_boundary_residual,
                    vertical.max_boundary_residual,
                )
            ),
        )


    # ========================================================
    # RESET HORIZONTAL SWING
    # ========================================================

    def reset_horizontal(
        self,
        initial_position,
        initial_velocity=None,
        initial_acceleration=None,
        start_time: float = 0.0,
    ) -> None:

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
            not math.isfinite(
                t0
            )
            or
            t0 < 0.0
        ):

            raise ValueError(
                "start_time must be finite and >= 0."
            )

        self._previous_time = (
            t0
        )

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
    # UPDATE HORIZONTAL SWING
    # ========================================================

    def update_horizontal(
        self,
        current_time: float,
        landing_time: float,
        landing_position,
    ) -> HorizontalSwingSample:
        """
        Regenerate a quintic trajectory:

            p(t_prev)
            v(t_prev)
            a(t_prev)

        ->

            p(T) = landing target
            v(T) = 0
            a(T) = 0
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

        tolerance = (
            1.0e-12
        )

        if (
            not math.isfinite(
                t_current
            )
            or
            t_current < 0.0
        ):

            raise ValueError(
                "current_time must be finite and >= 0."
            )

        if (
            not math.isfinite(
                T
            )
            or
            T <= 0.0
        ):

            raise ValueError(
                "landing_time must be finite and positive."
            )

        if (
            t_current
            <
            self._previous_time
            -
            tolerance
        ):

            raise ValueError(
                "current_time must not move backwards."
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

        # ====================================================
        # TOUCHDOWN
        # ====================================================

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
            ] = (
                target
            )

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

            self._previous_time = (
                t_current
            )

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
                time=(
                    t_current
                ),

                position=(
                    position
                ),

                velocity=(
                    velocity
                ),

                acceleration=(
                    acceleration
                ),

                landing_time=(
                    T
                ),

                landing_position=(
                    target.copy()
                ),

                coefficients=(
                    coefficients
                ),

                max_boundary_residual=0.0,
            )

        # ====================================================
        # ONLINE REGENERATION
        # ====================================================

        t_previous = float(
            self._previous_time
        )

        horizon = (
            T
            -
            t_previous
        )

        if (
            horizon
            <=
            tolerance
        ):

            raise RuntimeError(
                "Remaining horizontal swing horizon "
                "is too small."
            )

        local_time = (
            t_current
            -
            t_previous
        )

        local_time = min(
            max(
                local_time,
                0.0,
            ),
            horizon,
        )

        p0 = (
            self._position.copy()
        )

        v0 = (
            self._velocity.copy()
        )

        a0 = (
            self._acceleration.copy()
        )

        coefficients = np.zeros(
            (
                2,
                6,
            ),
            dtype=float,
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

        max_boundary_residual = 0.0

        for axis in range(
            2
        ):

            coefficients[
                axis,
                :
            ] = self._solve_quintic(
                initial_position=(
                    p0[
                        axis
                    ]
                ),

                initial_velocity=(
                    v0[
                        axis
                    ]
                ),

                initial_acceleration=(
                    a0[
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
                    local_time
                ),
            )

            residual = (
                self._compute_quintic_boundary_residual(
                    coefficients=(
                        coefficients[
                            axis,
                            :
                        ]
                    ),

                    horizon=(
                        horizon
                    ),

                    initial_position=(
                        p0[
                            axis
                        ]
                    ),

                    initial_velocity=(
                        v0[
                            axis
                        ]
                    ),

                    initial_acceleration=(
                        a0[
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
                residual,
            )

        if not np.all(
            np.isfinite(
                position
            )
        ):

            raise RuntimeError(
                "Horizontal swing position became non-finite."
            )

        if not np.all(
            np.isfinite(
                velocity
            )
        ):

            raise RuntimeError(
                "Horizontal swing velocity became non-finite."
            )

        if not np.all(
            np.isfinite(
                acceleration
            )
        ):

            raise RuntimeError(
                "Horizontal swing acceleration became non-finite."
            )

        # ====================================================
        # STORE CURRENT DESIRED STATE
        # ====================================================

        self._previous_time = (
            t_current
        )

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
            time=(
                t_current
            ),

            position=(
                position
            ),

            velocity=(
                velocity
            ),

            acceleration=(
                acceleration
            ),

            landing_time=(
                T
            ),

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
    # RESET VERTICAL SWING
    # ========================================================

    def reset_vertical(
        self,
    ) -> None:

        self._vertical_previous_time = (
            0.0
        )

        self._vertical_height = (
            0.0
        )

        self._vertical_velocity = (
            0.0
        )

        self._vertical_acceleration = (
            0.0
        )

        self._vertical_coefficients = (
            None
        )

        self._vertical_initialized = (
            True
        )


    # ========================================================
    # UPDATE VERTICAL SWING
    # ========================================================

    def update_vertical(
        self,
        current_time: float,
        landing_time: float,
        parameters: VerticalSwingQPParameters,
    ) -> VerticalSwingSample:
        """
        Online ninth-order vertical trajectory.

        Paper Eq. (21):

            minimize
                (z(T/2) - z_des)^2

        subject to

            0 <= z(t) <= z_max

            z(0)       = 0
            zdot(0)    = 0
            zddot(0)   = 0

            z(t_prev)       = z_prev
            zdot(t_prev)    = zdot_prev
            zddot(t_prev)   = zddot_prev

            z(T)       = 0
            zdot(T)    = 0
            zddot(T)   = 0
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

        tolerance = (
            1.0e-12
        )

        if (
            not math.isfinite(
                t_current
            )
            or
            t_current < 0.0
        ):

            raise ValueError(
                "current_time must be finite and >= 0."
            )

        if (
            not math.isfinite(
                T
            )
            or
            T <= 0.0
        ):

            raise ValueError(
                "landing_time must be finite and positive."
            )

        if (
            t_current
            <
            self._vertical_previous_time
            -
            tolerance
        ):

            raise ValueError(
                "current_time must not move backwards."
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

        # ====================================================
        # TOUCHDOWN
        # ====================================================

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

            self._vertical_previous_time = (
                t_current
            )

            self._vertical_height = (
                0.0
            )

            self._vertical_velocity = (
                0.0
            )

            self._vertical_acceleration = (
                0.0
            )

            self._vertical_coefficients = (
                coefficients.copy()
            )

            return VerticalSwingSample(
                time=(
                    t_current
                ),

                height=0.0,
                velocity=0.0,
                acceleration=0.0,

                landing_time=(
                    T
                ),

                coefficients=(
                    coefficients
                ),

                midpoint_height=0.0,

                continuous_min_height=0.0,
                continuous_max_height=0.0,

                max_boundary_residual=0.0,

                refinement_iterations=0,
            )

        # ====================================================
        # PREVIOUS DESIRED STATE
        # ====================================================

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
                "Remaining vertical swing horizon "
                "is too small."
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
        # EQUALITY CONSTRAINTS
        # ====================================================

        equality_rows = []
        equality_values = []

        # ----------------------------------------------------
        # Lift-off: t = 0
        # ----------------------------------------------------

        for derivative_order in range(
            3
        ):

            equality_rows.append(
                self._vertical_basis(
                    normalized_time=0.0,
                    derivative_order=(
                        derivative_order
                    ),
                    landing_time=(
                        T
                    ),
                )
            )

            equality_values.append(
                0.0
            )

        # ----------------------------------------------------
        # Previous online sample
        # ----------------------------------------------------

        previous_phase = (
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
                        previous_phase
                    ),

                    derivative_order=(
                        derivative_order
                    ),

                    landing_time=(
                        T
                    ),
                )
            )

            equality_values.append(
                previous_state[
                    derivative_order
                ]
            )

        # ----------------------------------------------------
        # Touchdown
        # ----------------------------------------------------

        for derivative_order in range(
            3
        ):

            equality_rows.append(
                self._vertical_basis(
                    normalized_time=1.0,
                    derivative_order=(
                        derivative_order
                    ),
                    landing_time=(
                        T
                    ),
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
            landing_time=(
                T
            ),
        )

        # A small strictly-positive regularization makes
        # the Hessian numerically better conditioned.
        regularization = max(
            float(
                parameters.coefficient_regularization
            ),
            1.0e-10,
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
        # HEIGHT CONSTRAINT LOCATIONS
        # ====================================================

        height_locations = list(
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

        continuous_min_height = (
            math.nan
        )

        continuous_max_height = (
            math.nan
        )

        refinement_iterations = (
            0
        )

        # ====================================================
        # CUTTING-PLANE LOOP
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

                        landing_time=(
                            T
                        ),
                    )

                    for location
                    in height_locations
                ]
            )

            constraint_matrix = np.vstack(
                [
                    equality_matrix,
                    height_matrix,
                ]
            )

            number_equalities = int(
                equality_matrix.shape[
                    0
                ]
            )

            number_height_constraints = int(
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
                H=(
                    H
                ),

                g=(
                    g
                ),

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
                coefficients
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
                    "Vertical swing QP could not enforce "
                    "continuous height bounds."
                )

            if lower_violation:

                self._append_unique_location(
                    height_locations,
                    minimum_location,
                )

            if upper_violation:

                self._append_unique_location(
                    height_locations,
                    maximum_location,
                )

        if coefficients is None:

            raise RuntimeError(
                "Vertical swing QP returned no solution."
            )

        # ====================================================
        # EVALUATE CURRENT SAMPLE
        # ====================================================

        current_phase = (
            t_current
            /
            T
        )

        current_height = float(
            self._vertical_basis(
                normalized_time=(
                    current_phase
                ),

                derivative_order=0,

                landing_time=(
                    T
                ),
            )
            @
            coefficients
        )

        current_velocity = float(
            self._vertical_basis(
                normalized_time=(
                    current_phase
                ),

                derivative_order=1,

                landing_time=(
                    T
                ),
            )
            @
            coefficients
        )

        current_acceleration = float(
            self._vertical_basis(
                normalized_time=(
                    current_phase
                ),

                derivative_order=2,

                landing_time=(
                    T
                ),
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
                "Non-finite vertical swing trajectory."
            )

        # ====================================================
        # RESIDUAL OF FULL PAPER BOUNDARY CONDITIONS
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
        # STORE FOR NEXT ONLINE REGENERATION
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

        self._vertical_coefficients = (
            coefficients.copy()
        )

        return VerticalSwingSample(
            time=(
                t_current
            ),

            height=(
                current_height
            ),

            velocity=(
                current_velocity
            ),

            acceleration=(
                current_acceleration
            ),

            landing_time=(
                T
            ),

            coefficients=(
                coefficients.copy()
            ),

            midpoint_height=(
                midpoint_height
            ),

            continuous_min_height=float(
                continuous_min_height
            ),

            continuous_max_height=float(
                continuous_max_height
            ),

            max_boundary_residual=(
                max_boundary_residual
            ),

            refinement_iterations=(
                refinement_iterations
            ),
        )


    # ========================================================
    # ROBUST VERTICAL QP
    # ========================================================

    def _solve_vertical_qp(
        self,
        H,
        g,
        constraint_matrix,
        constraint_lower,
        constraint_upper,
    ) -> np.ndarray:
        """
        Solve the convex vertical swing QP.

        Numerical strategy:

            1. CasADi QRQP
            2. CasADi qpOASES
            3. SciPy SLSQP

        The mathematical QP remains unchanged.
        """

        H = np.asarray(
            H,
            dtype=float,
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

        g = np.asarray(
            g,
            dtype=float,
        ).reshape(
            -1
        )

        A = np.asarray(
            constraint_matrix,
            dtype=float,
        )

        lba = np.asarray(
            constraint_lower,
            dtype=float,
        ).reshape(
            -1
        )

        uba = np.asarray(
            constraint_upper,
            dtype=float,
        ).reshape(
            -1
        )

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
            A.shape[1] != 10
        ):

            raise ValueError(
                "Vertical QP constraint matrix must "
                "have 10 columns."
            )

        if (
            lba.shape[0]
            !=
            A.shape[0]
            or
            uba.shape[0]
            !=
            A.shape[0]
        ):

            raise ValueError(
                "Vertical QP constraint dimensions "
                "are inconsistent."
            )

        qp_structure = {
            "h": ca.Sparsity.dense(
                10,
                10,
            ),

            "a": ca.Sparsity.dense(
                A.shape[0],
                10,
            ),
        }

        lbx = np.full(
            10,
            -np.inf,
            dtype=float,
        )

        ubx = np.full(
            10,
            +np.inf,
            dtype=float,
        )

        solver_messages = []

        # ====================================================
        # SOLVER HELPER
        # ====================================================

        def run_casadi_solver(
            plugin,
            options,
        ):

            self._vertical_solver_counter += (
                1
            )

            solver_name = (
                "vertical_swing_"
                f"{plugin}_"
                f"{id(self)}_"
                f"{self._vertical_solver_counter}"
            )

            solver = ca.conic(
                solver_name,
                plugin,
                qp_structure,
                options,
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
                    lbx
                ),

                ubx=ca.DM(
                    ubx
                ),
            )

            stats = (
                solver.stats()
            )

            return (
                result,
                stats,
            )

        # ====================================================
        # 1. QRQP
        # ====================================================

        try:

            (
                result,
                stats,
            ) = run_casadi_solver(
                "qrqp",
                {
                    "print_header": False,
                    "print_iter": False,
                    "print_info": False,
                    "error_on_fail": False,
                    "max_iter": 1000,
                },
            )

            if bool(
                stats.get(
                    "success",
                    False,
                )
            ):

                coefficients = np.asarray(
                    result[
                        "x"
                    ],
                    dtype=float,
                ).reshape(
                    10
                )

                if self._qp_solution_is_valid(
                    coefficients=(
                        coefficients
                    ),

                    A=(
                        A
                    ),

                    lba=(
                        lba
                    ),

                    uba=(
                        uba
                    ),
                ):

                    return (
                        coefficients
                    )

            solver_messages.append(
                "QRQP: "
                f"{stats.get('return_status', 'unknown')}"
            )

        except Exception as error:

            solver_messages.append(
                "QRQP exception: "
                f"{error}"
            )

        # ====================================================
        # 2. qpOASES
        # ====================================================

        try:

            qpoases_available = bool(
                ca.has_conic(
                    "qpoases"
                )
            )

        except Exception:

            qpoases_available = (
                False
            )

        if qpoases_available:

            try:

                (
                    result,
                    stats,
                ) = run_casadi_solver(
                    "qpoases",
                    {
                        "error_on_fail": False,
                        "print_time": False,
                    },
                )

                if bool(
                    stats.get(
                        "success",
                        False,
                    )
                ):

                    coefficients = np.asarray(
                        result[
                            "x"
                        ],
                        dtype=float,
                    ).reshape(
                        10
                    )

                    if self._qp_solution_is_valid(
                        coefficients=(
                            coefficients
                        ),

                        A=(
                            A
                        ),

                        lba=(
                            lba
                        ),

                        uba=(
                            uba
                        ),
                    ):

                        return (
                            coefficients
                        )

                solver_messages.append(
                    "qpOASES: "
                    f"{stats.get('return_status', 'unknown')}"
                )

            except Exception as error:

                solver_messages.append(
                    "qpOASES exception: "
                    f"{error}"
                )

        else:

            solver_messages.append(
                "qpOASES unavailable"
            )

        # ====================================================
        # 3. SCIPY SLSQP FALLBACK
        # ====================================================

        try:

            from scipy.optimize import (
                LinearConstraint,
                minimize,
            )

            # -----------------------------------------------
            # Warm start
            # -----------------------------------------------

            if (
                self._vertical_coefficients
                is not None
                and
                np.all(
                    np.isfinite(
                        self._vertical_coefficients
                    )
                )
            ):

                x0 = (
                    self._vertical_coefficients
                    .copy()
                )

            else:

                # Use least-squares solution of equality rows.

                equality_mask = np.isclose(
                    lba,
                    uba,
                    rtol=0.0,
                    atol=1.0e-10,
                )

                if np.any(
                    equality_mask
                ):

                    Aeq = (
                        A[
                            equality_mask,
                            :
                        ]
                    )

                    beq = (
                        0.5
                        *
                        (
                            lba[
                                equality_mask
                            ]
                            +
                            uba[
                                equality_mask
                            ]
                        )
                    )

                    x0 = np.linalg.lstsq(
                        Aeq,
                        beq,
                        rcond=None,
                    )[0]

                else:

                    x0 = np.zeros(
                        10,
                        dtype=float,
                    )

            linear_constraint = (
                LinearConstraint(
                    A,
                    lba,
                    uba,
                )
            )

            def objective(
                x,
            ):

                return float(
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

            def gradient(
                x,
            ):

                return (
                    H
                    @
                    x
                    +
                    g
                )

            scipy_result = minimize(
                objective,
                x0,
                jac=(
                    gradient
                ),

                constraints=[
                    linear_constraint
                ],

                method="SLSQP",

                options={
                    "ftol": 1.0e-12,
                    "maxiter": 1000,
                    "disp": False,
                },
            )

            coefficients = np.asarray(
                scipy_result.x,
                dtype=float,
            ).reshape(
                10
            )

            if (
                scipy_result.success
                and
                self._qp_solution_is_valid(
                    coefficients=(
                        coefficients
                    ),

                    A=(
                        A
                    ),

                    lba=(
                        lba
                    ),

                    uba=(
                        uba
                    ),

                    tolerance=1.0e-6,
                )
            ):

                return (
                    coefficients
                )

            solver_messages.append(
                "SLSQP: "
                f"{scipy_result.message}"
            )

        except Exception as error:

            solver_messages.append(
                "SLSQP exception: "
                f"{error}"
            )

        # ====================================================
        # ALL SOLVERS FAILED
        # ====================================================

        raise RuntimeError(
            "Vertical swing QP failed with all solvers.\n"
            +
            "\n".join(
                solver_messages
            )
        )


    # ========================================================
    # QP SOLUTION VALIDATION
    # ========================================================

    @staticmethod
    def _qp_solution_is_valid(
        coefficients,
        A,
        lba,
        uba,
        tolerance: float = 1.0e-7,
    ) -> bool:

        x = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(
            -1
        )

        if not np.all(
            np.isfinite(
                x
            )
        ):

            return False

        values = (
            A
            @
            x
        )

        lower_violation = np.max(
            np.maximum(
                lba
                -
                values,
                0.0,
            )
        )

        upper_violation = np.max(
            np.maximum(
                values
                -
                uba,
                0.0,
            )
        )

        max_violation = max(
            float(
                lower_violation
            ),
            float(
                upper_violation
            ),
        )

        return bool(
            max_violation
            <=
            tolerance
        )


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
            not math.isfinite(
                r
            )
            or
            not math.isfinite(
                T
            )
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
                "derivative_order must be 0, 1 or 2."
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

        return (
            basis
        )


    # ========================================================
    # CONTINUOUS VERTICAL EXTREMA
    # ========================================================

    @staticmethod
    def _continuous_vertical_extrema(
        coefficients,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:

        c = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(
            10
        )

        derivative_ascending = np.array(
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

        derivative_descending = (
            derivative_ascending[
                ::-1
            ]
        )

        # Remove insignificant leading terms.
        while (
            derivative_descending.size
            >
            1
            and
            abs(
                derivative_descending[
                    0
                ]
            )
            <
            1.0e-14
        ):

            derivative_descending = (
                derivative_descending[
                    1:
                ]
            )

        candidates = [
            0.0,
            1.0,
        ]

        if np.any(
            np.abs(
                derivative_descending
            )
            >
            1.0e-14
        ):

            roots = np.roots(
                derivative_descending
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
                        0.0
                        <
                        location
                        <
                        1.0
                    ):

                        candidates.append(
                            location
                        )

        candidates = np.asarray(
            candidates,
            dtype=float,
        )

        heights = np.array(
            [
                np.polynomial.polynomial.polyval(
                    location,
                    c,
                )

                for location
                in candidates
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
                candidates[
                    minimum_index
                ]
            ),

            float(
                candidates[
                    maximum_index
                ]
            ),
        )


    # ========================================================
    # REMOVE REDUNDANT EQUALITIES
    # ========================================================

    @staticmethod
    def _remove_redundant_equalities(
        matrix,
        vector,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        A = np.asarray(
            matrix,
            dtype=float,
        )

        b = np.asarray(
            vector,
            dtype=float,
        ).reshape(
            -1
        )

        if (
            A.ndim != 2
            or
            A.shape[0]
            !=
            b.shape[0]
        ):

            raise ValueError(
                "Invalid equality system."
            )

        selected_indices = []

        current_rank = (
            0
        )

        for row_index in range(
            A.shape[0]
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

        if len(
            selected_indices
        ) == 0:

            return (
                np.zeros(
                    (
                        0,
                        A.shape[1],
                    ),
                    dtype=float,
                ),

                np.zeros(
                    0,
                    dtype=float,
                ),
            )

        return (
            A[
                selected_indices,
                :
            ].copy(),

            b[
                selected_indices
            ].copy(),
        )


    # ========================================================
    # ADD CUTTING-PLANE LOCATION
    # ========================================================

    @staticmethod
    def _append_unique_location(
        locations,
        new_location,
    ) -> None:

        value = float(
            new_location
        )

        if not (
            0.0
            <=
            value
            <=
            1.0
        ):

            return

        for old_location in locations:

            if (
                abs(
                    float(
                        old_location
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

        H = float(
            horizon
        )

        if (
            not math.isfinite(
                H
            )
            or
            H <= 0.0
        ):

            raise ValueError(
                "Quintic horizon must be positive."
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

        c0 = (
            p0
        )

        c1 = (
            v0
        )

        c2 = (
            0.5
            *
            a0
        )

        M = np.array(
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
                    2.0
                    *
                    c2
                    *
                    H
                ),

                -(
                    2.0
                    *
                    c2
                ),
            ],
            dtype=float,
        )

        try:

            c3_to_c5 = np.linalg.solve(
                M,
                rhs,
            )

        except np.linalg.LinAlgError as error:

            raise RuntimeError(
                "Horizontal quintic solve failed."
            ) from error

        coefficients = np.array(
            [
                c0,
                c1,
                c2,
                c3_to_c5[0],
                c3_to_c5[1],
                c3_to_c5[2],
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                coefficients
            )
        ):

            raise RuntimeError(
                "Horizontal quintic coefficients "
                "became non-finite."
            )

        return (
            coefficients
        )


    # ========================================================
    # QUINTIC EVALUATION
    # ========================================================

    @staticmethod
    def _evaluate_quintic(
        coefficients,
        local_time: float,
    ) -> tuple[
        float,
        float,
        float,
    ]:

        c = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(
            6
        )

        t = float(
            local_time
        )

        position = (
            c[0]
            +
            c[1] * t
            +
            c[2] * t**2
            +
            c[3] * t**3
            +
            c[4] * t**4
            +
            c[5] * t**5
        )

        velocity = (
            c[1]
            +
            2.0 * c[2] * t
            +
            3.0 * c[3] * t**2
            +
            4.0 * c[4] * t**3
            +
            5.0 * c[5] * t**4
        )

        acceleration = (
            2.0 * c[2]
            +
            6.0 * c[3] * t
            +
            12.0 * c[4] * t**2
            +
            20.0 * c[5] * t**3
        )

        return (
            float(
                position
            ),
            float(
                velocity
            ),
            float(
                acceleration
            ),
        )


    # ========================================================
    # QUINTIC BOUNDARY CHECK
    # ========================================================

    @classmethod
    def _compute_quintic_boundary_residual(
        cls,
        coefficients,
        horizon: float,
        initial_position: float,
        initial_velocity: float,
        initial_acceleration: float,
        final_position: float,
    ) -> float:

        (
            p_initial,
            v_initial,
            a_initial,
        ) = cls._evaluate_quintic(
            coefficients=(
                coefficients
            ),

            local_time=0.0,
        )

        (
            p_final,
            v_final,
            a_final,
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
                p_initial
                -
                initial_position,

                v_initial
                -
                initial_velocity,

                a_initial
                -
                initial_acceleration,

                p_final
                -
                final_position,

                v_final,

                a_final,
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
    # PARAMETER VALIDATION
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

        if (
            not math.isfinite(
                desired_height
            )
            or
            desired_height < 0.0
        ):

            raise ValueError(
                "desired_height must be finite and >= 0."
            )

        if (
            not math.isfinite(
                maximum_height
            )
            or
            maximum_height <= 0.0
        ):

            raise ValueError(
                "maximum_height must be finite and positive."
            )

        if (
            desired_height
            >
            maximum_height
        ):

            raise ValueError(
                "desired_height must not exceed maximum_height."
            )

        if (
            not math.isfinite(
                regularization
            )
            or
            regularization < 0.0
        ):

            raise ValueError(
                "coefficient_regularization must be >= 0."
            )

        if (
            not math.isfinite(
                bound_tolerance
            )
            or
            bound_tolerance < 0.0
        ):

            raise ValueError(
                "bound_tolerance must be >= 0."
            )

        if constraint_samples < 3:

            raise ValueError(
                "constraint_samples must be >= 3."
            )

        if max_refinements < 0:

            raise ValueError(
                "max_refinements must be >= 0."
            )


    # ========================================================
    # VECTOR UTILITIES
    # ========================================================

    @staticmethod
    def _as_vector2(
        value,
        name,
    ) -> np.ndarray:

        result = np.asarray(
            value,
            dtype=float,
        ).reshape(
            -1
        )

        if result.shape != (
            2,
        ):

            raise ValueError(
                f"{name} must contain exactly 2 values."
            )

        if not np.all(
            np.isfinite(
                result
            )
        ):

            raise ValueError(
                f"{name} must contain finite values."
            )

        return (
            result.copy()
        )


    @staticmethod
    def _as_vector3(
        value,
        name,
    ) -> np.ndarray:

        result = np.asarray(
            value,
            dtype=float,
        ).reshape(
            -1
        )

        if result.shape != (
            3,
        ):

            raise ValueError(
                f"{name} must contain exactly 3 values."
            )

        if not np.all(
            np.isfinite(
                result
            )
        ):

            raise ValueError(
                f"{name} must contain finite values."
            )

        return (
            result.copy()
        )


    # ========================================================
    # STATE CHECKS
    # ========================================================

    def _require_initialized(
        self,
    ) -> None:

        if not self._initialized:

            raise RuntimeError(
                "Horizontal swing trajectory is not initialized."
            )


    def _require_vertical_initialized(
        self,
    ) -> None:

        if not self._vertical_initialized:

            raise RuntimeError(
                "Vertical swing trajectory is not initialized."
            )