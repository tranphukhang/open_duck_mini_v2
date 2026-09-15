# step_timing_adaptation/online_swing_trajectory.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# OUTPUT
# ============================================================

@dataclass(frozen=True)
class SwingSample:

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


# ============================================================
# ONLINE QUINTIC SWING TRAJECTORY
# ============================================================

class OnlineQuinticSwingTrajectory:
    """
    Online C2-continuous swing-foot trajectory.

    Each update:

        1. Evaluate the PREVIOUS polynomial at the current time
           to obtain the current desired:

               p
               v
               a

        2. Replan a new quintic from that exact state toward
           the latest adaptive landing target.

    Therefore changes in:

        landing position uT
        landing time T

    do not create position / velocity / acceleration jumps.

    Horizontal:
        current state -> landing target

    Vertical:
        before apex:
            current state -> fixed apex

        after apex:
            current state -> landing height

    No QP.
    No CasADi.
    No generic optimizer.
    """

    TIME_TOLERANCE = 1.0e-12


    def __init__(
        self,
    ) -> None:

        self._initialized = False

        self._start_time = 0.0
        self._previous_time = 0.0

        self._position = np.zeros(
            3,
            dtype=float,
        )

        self._velocity = np.zeros(
            3,
            dtype=float,
        )

        self._acceleration = np.zeros(
            3,
            dtype=float,
        )

        # Current local quintic plan for x, y, z.
        #
        # coefficients[axis] =
        #
        #     c0 ... c5
        #
        # such that:
        #
        #     p(t) =
        #         c0
        #       + c1 t
        #       + c2 t^2
        #       + c3 t^3
        #       + c4 t^4
        #       + c5 t^5

        self._coefficients = np.zeros(
            (
                3,
                6,
            ),
            dtype=float,
        )

        # Individual local duration for each axis.
        #
        # X/Y:
        #     until touchdown
        #
        # Z:
        #     until apex or touchdown

        self._plan_duration = np.zeros(
            3,
            dtype=float,
        )

        self._plan_start_time = 0.0

        self._plan_available = False

        # Vertical state.

        self._apex_initialized = False

        self._apex_time = 0.0
        self._apex_height = 0.0

        self._apex_reached = False


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        initial_position,
        start_time: float = 0.0,
        initial_velocity=None,
        initial_acceleration=None,
    ) -> None:

        position = self._vector3(
            initial_position,
            "initial_position",
        )

        if initial_velocity is None:

            velocity = np.zeros(
                3,
                dtype=float,
            )

        else:

            velocity = self._vector3(
                initial_velocity,
                "initial_velocity",
            )

        if initial_acceleration is None:

            acceleration = np.zeros(
                3,
                dtype=float,
            )

        else:

            acceleration = self._vector3(
                initial_acceleration,
                "initial_acceleration",
            )

        t0 = float(
            start_time
        )

        if not np.isfinite(
            t0
        ):

            raise ValueError(
                "start_time must be finite."
            )

        self._start_time = t0
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

        self._coefficients[:] = 0.0
        self._plan_duration[:] = 0.0

        self._plan_start_time = t0
        self._plan_available = False

        self._apex_initialized = False
        self._apex_reached = False

        self._apex_time = t0
        self._apex_height = float(
            position[2]
        )

        self._initialized = True


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        *,
        current_time: float,
        landing_time: float,
        target_position,
        swing_height: float,
    ) -> SwingSample:

        self._require_initialized()

        t = float(
            current_time
        )

        T = float(
            landing_time
        )

        target = self._vector3(
            target_position,
            "target_position",
        )

        height = float(
            swing_height
        )

        if not np.isfinite(
            t
        ):

            raise ValueError(
                "current_time must be finite."
            )

        if not np.isfinite(
            T
        ):

            raise ValueError(
                "landing_time must be finite."
            )

        if height < 0.0:

            raise ValueError(
                "swing_height must be >= 0."
            )

        if (
            t
            <
            self._previous_time
            -
            self.TIME_TOLERANCE
        ):

            raise ValueError(
                "current_time moved backwards."
            )

        if (
            T
            <
            t
            -
            self.TIME_TOLERANCE
        ):

            raise ValueError(
                "landing_time is earlier than current_time."
            )

        # ====================================================
        # ADVANCE OLD PLAN TO CURRENT TIME
        #
        # This is the key continuity step.
        #
        # We DO NOT recompute the current desired state from
        # lift-off.
        #
        # Instead we first propagate the trajectory that was
        # already active.
        # ====================================================

        if self._plan_available:

            local_time = (
                t
                -
                self._plan_start_time
            )

            for axis in range(
                3
            ):

                evaluation_time = np.clip(
                    local_time,
                    0.0,
                    self._plan_duration[
                        axis
                    ],
                )

                (
                    self._position[
                        axis
                    ],
                    self._velocity[
                        axis
                    ],
                    self._acceleration[
                        axis
                    ],
                ) = self._evaluate_quintic(
                    self._coefficients[
                        axis
                    ],
                    evaluation_time,
                )

        # ====================================================
        # TOUCHDOWN
        # ====================================================

        remaining_time = (
            T
            -
            t
        )

        if (
            remaining_time
            <=
            self.TIME_TOLERANCE
        ):

            self._position = (
                target.copy()
            )

            self._velocity[:] = 0.0
            self._acceleration[:] = 0.0

            self._previous_time = t

            self._plan_available = False

            return SwingSample(
                position=(
                    self._position.copy()
                ),

                velocity=(
                    self._velocity.copy()
                ),

                acceleration=(
                    self._acceleration.copy()
                ),
            )

        # ====================================================
        # INITIALIZE VERTICAL APEX
        #
        # Apex time is determined once from the step duration
        # available when swing begins.
        #
        # This avoids continuously moving the apex toward the
        # future as a receding-horizon artifact.
        # ====================================================

        if not self._apex_initialized:

            self._apex_time = (
                self._start_time
                +
                0.5
                *
                (
                    T
                    -
                    self._start_time
                )
            )

            self._apex_height = (
                max(
                    self._position[2],
                    target[2],
                )
                +
                height
            )

            self._apex_initialized = True

        # ====================================================
        # IF TIMING SHRINKS SO MUCH THAT THE OLD APEX WOULD
        # OCCUR AFTER TOUCHDOWN
        #
        # Move the apex between now and touchdown.
        #
        # C2 continuity is still preserved because replanning
        # starts from the current p/v/a.
        # ====================================================

        if (
            not self._apex_reached
            and
            self._apex_time
            >=
            T
            -
            self.TIME_TOLERANCE
        ):

            self._apex_time = (
                t
                +
                0.5
                *
                remaining_time
            )

        # ====================================================
        # APEX SWITCH
        # ====================================================

        if (
            not self._apex_reached
            and
            t
            >=
            self._apex_time
            -
            self.TIME_TOLERANCE
        ):

            self._apex_reached = True

        # ====================================================
        # STORE CURRENT OUTPUT
        #
        # This is the reference used by IK at time t.
        # ====================================================

        output = SwingSample(
            position=(
                self._position.copy()
            ),

            velocity=(
                self._velocity.copy()
            ),

            acceleration=(
                self._acceleration.copy()
            ),
        )

        # ====================================================
        # BUILD NEW PLAN FOR FUTURE SAMPLES
        # ====================================================

        # ----------------------------------------------------
        # X/Y:
        #
        # current p/v/a
        #       ->
        # latest adaptive landing position
        #
        # over:
        #
        #       T - t
        # ----------------------------------------------------

        for axis in (
            0,
            1,
        ):

            self._coefficients[
                axis
            ] = self._solve_quintic(
                initial_position=(
                    self._position[
                        axis
                    ]
                ),

                initial_velocity=(
                    self._velocity[
                        axis
                    ]
                ),

                initial_acceleration=(
                    self._acceleration[
                        axis
                    ]
                ),

                final_position=(
                    target[
                        axis
                    ]
                ),

                duration=(
                    remaining_time
                ),
            )

            self._plan_duration[
                axis
            ] = (
                remaining_time
            )

        # ----------------------------------------------------
        # Z before apex:
        #
        # current p/v/a
        #       ->
        # apex height, zero v/a
        # ----------------------------------------------------

        if not self._apex_reached:

            vertical_duration = (
                self._apex_time
                -
                t
            )

            if (
                vertical_duration
                <=
                self.TIME_TOLERANCE
            ):

                self._apex_reached = True

            else:

                self._coefficients[
                    2
                ] = self._solve_quintic(
                    initial_position=(
                        self._position[2]
                    ),

                    initial_velocity=(
                        self._velocity[2]
                    ),

                    initial_acceleration=(
                        self._acceleration[2]
                    ),

                    final_position=(
                        self._apex_height
                    ),

                    duration=(
                        vertical_duration
                    ),
                )

                self._plan_duration[
                    2
                ] = (
                    vertical_duration
                )

        # ----------------------------------------------------
        # Z after apex:
        #
        # current p/v/a
        #       ->
        # latest touchdown height
        # ----------------------------------------------------

        if self._apex_reached:

            self._coefficients[
                2
            ] = self._solve_quintic(
                initial_position=(
                    self._position[2]
                ),

                initial_velocity=(
                    self._velocity[2]
                ),

                initial_acceleration=(
                    self._acceleration[2]
                ),

                final_position=(
                    target[2]
                ),

                duration=(
                    remaining_time
                ),
            )

            self._plan_duration[
                2
            ] = (
                remaining_time
            )

        self._plan_start_time = t
        self._plan_available = True

        self._previous_time = t

        return output


    # ========================================================
    # QUINTIC
    #
    # Boundary conditions:
    #
    # p(0)   = p0
    # v(0)   = v0
    # a(0)   = a0
    #
    # p(H)   = pf
    # v(H)   = 0
    # a(H)   = 0
    #
    # Closed-form solution avoids a numerical matrix solve at
    # 2 kHz.
    # ========================================================

    @staticmethod
    def _solve_quintic(
        *,
        initial_position: float,
        initial_velocity: float,
        initial_acceleration: float,
        final_position: float,
        duration: float,
    ) -> np.ndarray:

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

        H = float(
            duration
        )

        if (
            not np.isfinite(H)
            or
            H <= 0.0
        ):

            raise ValueError(
                "Quintic duration must be positive."
            )

        delta_p = (
            pf
            -
            p0
        )

        c0 = p0
        c1 = v0
        c2 = 0.5 * a0

        c3 = (
            10.0
            *
            delta_p
            /
            H**3
            -
            6.0
            *
            v0
            /
            H**2
            -
            1.5
            *
            a0
            /
            H
        )

        c4 = (
            -15.0
            *
            delta_p
            /
            H**4
            +
            8.0
            *
            v0
            /
            H**3
            +
            1.5
            *
            a0
            /
            H**2
        )

        c5 = (
            6.0
            *
            delta_p
            /
            H**5
            -
            3.0
            *
            v0
            /
            H**4
            -
            0.5
            *
            a0
            /
            H**3
        )

        coefficients = np.array(
            [
                c0,
                c1,
                c2,
                c3,
                c4,
                c5,
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                coefficients
            )
        ):

            raise RuntimeError(
                "Quintic coefficients became NaN/Inf."
            )

        return coefficients


    # ========================================================
    # QUINTIC EVALUATION
    # ========================================================

    @staticmethod
    def _evaluate_quintic(
        coefficients,
        time: float,
    ):

        c = np.asarray(
            coefficients,
            dtype=float,
        )

        t = float(
            time
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
            float(position),
            float(velocity),
            float(acceleration),
        )


    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _vector3(
        value,
        name,
    ) -> np.ndarray:

        vector = np.asarray(
            value,
            dtype=float,
        ).reshape(
            -1
        )

        if vector.shape != (
            3,
        ):

            raise ValueError(
                f"{name} must contain exactly 3 values."
            )

        if not np.all(
            np.isfinite(
                vector
            )
        ):

            raise ValueError(
                f"{name} contains NaN/Inf."
            )

        return vector.copy()


    def _require_initialized(
        self,
    ) -> None:

        if not self._initialized:

            raise RuntimeError(
                "Swing trajectory is not initialized. "
                "Call reset() at lift-off."
            )