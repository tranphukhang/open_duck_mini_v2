# lipm_mpc/com_trajectory.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# REFERENCE RESULT
# ============================================================

@dataclass
class CoMReference:

    # 3D CoM position [m]
    position: np.ndarray

    # 3D CoM velocity [m/s]
    velocity: np.ndarray

    # 3D CoM acceleration [m/s^2]
    acceleration: np.ndarray

    # 3D CoM jerk [m/s^3]
    jerk: np.ndarray


# ============================================================
# CONSTANT-JERK COM SEGMENT
# ============================================================

class ConstantJerkCoMSegment:
    """
    Continuous CoM reference over one MPC interval.

    MPC state for each horizontal axis:

        state = [position, velocity, acceleration]

    MPC control:

        jerk = constant

    During one MPC interval:

        0 <= tau <= duration

    the continuous trajectory is:

        p(tau) =
            p0
            + v0 * tau
            + 1/2 * a0 * tau^2
            + 1/6 * j * tau^3

        v(tau) =
            v0
            + a0 * tau
            + 1/2 * j * tau^2

        a(tau) =
            a0
            + j * tau

    The z-direction is held constant.

    This class does NOT solve MPC.

    It only converts one MPC state + first jerk command
    into a smooth continuous reference that can be sampled
    at the faster executor / IK timestep.
    """

    def __init__(
        self,
        x_state,
        y_state,
        x_jerk: float,
        y_jerk: float,
        com_height: float,
        duration: float,
    ) -> None:

        # ====================================================
        # VALIDATE STATES
        # ====================================================

        self.x_state = self._validate_state(
            x_state,
            "x_state",
        )

        self.y_state = self._validate_state(
            y_state,
            "y_state",
        )

        # ====================================================
        # VALIDATE JERK
        # ====================================================

        x_jerk = float(
            x_jerk
        )

        y_jerk = float(
            y_jerk
        )

        if not np.isfinite(
            x_jerk
        ):

            raise ValueError(
                "x_jerk must be finite."
            )

        if not np.isfinite(
            y_jerk
        ):

            raise ValueError(
                "y_jerk must be finite."
            )

        # ====================================================
        # VALIDATE HEIGHT
        # ====================================================

        com_height = float(
            com_height
        )

        if (
            not np.isfinite(
                com_height
            )
            or
            com_height <= 0.0
        ):

            raise ValueError(
                "com_height must be finite and positive."
            )

        # ====================================================
        # VALIDATE DURATION
        # ====================================================

        duration = float(
            duration
        )

        if (
            not np.isfinite(
                duration
            )
            or
            duration <= 0.0
        ):

            raise ValueError(
                "duration must be finite and positive."
            )

        # ====================================================
        # STORE
        # ====================================================

        self.x_jerk = (
            x_jerk
        )

        self.y_jerk = (
            y_jerk
        )

        self.com_height = (
            com_height
        )

        self.duration = (
            duration
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

        return (
            state.copy()
        )


    # ========================================================
    # TIME VALIDATION
    # ========================================================

    def _validate_time(
        self,
        tau: float,
    ) -> float:

        tau = float(
            tau
        )

        if not np.isfinite(
            tau
        ):

            raise ValueError(
                "tau must be finite."
            )

        tolerance = (
            1e-12
        )

        if (
            tau
            <
            -tolerance
            or
            tau
            >
            self.duration
            +
            tolerance
        ):

            raise ValueError(
                "tau must satisfy "
                "0 <= tau <= duration."
            )

        # Remove tiny floating-point overshoots
        tau = min(
            max(
                tau,
                0.0,
            ),
            self.duration,
        )

        return (
            tau
        )


    # ========================================================
    # EVALUATE ONE HORIZONTAL AXIS
    # ========================================================

    @staticmethod
    def _evaluate_axis(
        state,
        jerk: float,
        tau: float,
    ) -> np.ndarray:

        position_0 = (
            state[0]
        )

        velocity_0 = (
            state[1]
        )

        acceleration_0 = (
            state[2]
        )

        position = (
            position_0
            +
            velocity_0
            *
            tau
            +
            0.5
            *
            acceleration_0
            *
            tau**2
            +
            (
                1.0
                /
                6.0
            )
            *
            jerk
            *
            tau**3
        )

        velocity = (
            velocity_0
            +
            acceleration_0
            *
            tau
            +
            0.5
            *
            jerk
            *
            tau**2
        )

        acceleration = (
            acceleration_0
            +
            jerk
            *
            tau
        )

        return np.array(
            [
                position,
                velocity,
                acceleration,
            ],
            dtype=float,
        )


    # ========================================================
    # GET X STATE
    # ========================================================

    def get_x_state(
        self,
        tau: float,
    ) -> np.ndarray:

        tau = self._validate_time(
            tau
        )

        return self._evaluate_axis(
            state=(
                self.x_state
            ),

            jerk=(
                self.x_jerk
            ),

            tau=(
                tau
            ),
        )


    # ========================================================
    # GET Y STATE
    # ========================================================

    def get_y_state(
        self,
        tau: float,
    ) -> np.ndarray:

        tau = self._validate_time(
            tau
        )

        return self._evaluate_axis(
            state=(
                self.y_state
            ),

            jerk=(
                self.y_jerk
            ),

            tau=(
                tau
            ),
        )


    # ========================================================
    # EVALUATE FULL 3D COM REFERENCE
    # ========================================================

    def evaluate(
        self,
        tau: float,
    ) -> CoMReference:

        tau = self._validate_time(
            tau
        )

        x = self._evaluate_axis(
            state=(
                self.x_state
            ),

            jerk=(
                self.x_jerk
            ),

            tau=(
                tau
            ),
        )

        y = self._evaluate_axis(
            state=(
                self.y_state
            ),

            jerk=(
                self.y_jerk
            ),

            tau=(
                tau
            ),
        )

        position = np.array(
            [
                x[0],
                y[0],
                self.com_height,
            ],
            dtype=float,
        )

        velocity = np.array(
            [
                x[1],
                y[1],
                0.0,
            ],
            dtype=float,
        )

        acceleration = np.array(
            [
                x[2],
                y[2],
                0.0,
            ],
            dtype=float,
        )

        jerk = np.array(
            [
                self.x_jerk,
                self.y_jerk,
                0.0,
            ],
            dtype=float,
        )

        return CoMReference(
            position=(
                position
            ),

            velocity=(
                velocity
            ),

            acceleration=(
                acceleration
            ),

            jerk=(
                jerk
            ),
        )


    # ========================================================
    # TERMINAL STATES
    # ========================================================

    def get_terminal_x_state(
        self,
    ) -> np.ndarray:

        return self.get_x_state(
            self.duration
        )


    def get_terminal_y_state(
        self,
    ) -> np.ndarray:

        return self.get_y_state(
            self.duration
        )