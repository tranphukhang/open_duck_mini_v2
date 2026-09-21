# step_timing_adaptation/lipm_state_generator.py

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# ============================================================
# LIPM SAMPLE
# ============================================================

@dataclass(frozen=True)
class LIPMSample:
    """
    State of the point-foot LIPM at one instant.

    Conventions:
        x : forward
        y : left
        z : upward

    The horizontal LIPM dynamics are

        x_ddot = omega^2 (x - u)

    with

        omega = sqrt(g / z0)

    where u is the current stance/contact point.

    The DCM is

        xi = x + x_dot / omega
    """

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    dcm: np.ndarray
    support_position: np.ndarray

    omega: float
    com_height: float


# ============================================================
# POINT-FOOT LIPM
# ============================================================

class PointFootLIPM:
    """
    Point-foot Linear Inverted Pendulum Model.

    This class reproduces the CoM evolution used in the LIPM
    simulation of Khadiv et al., "Walking Control Based on
    Step Timing Adaptation":

        1) During one stance phase, the support point u0 is
           fixed.

        2) The horizontal CoM state is propagated forward
           from

               x_ddot = omega^2 (x - u0)

        3) At touchdown, the support point is reset

               u0 <- uT

           while CoM position and velocity remain continuous.

    No CoM MPC is solved here.

    The integration below uses the exact closed-form solution
    over each dt. This is mathematically equivalent to
    integrating the LIPM ODE forward, but avoids numerical
    drift associated with explicit Euler integration.

    An optional constant horizontal external force can be
    applied during one integration interval. It enters as

        x_ddot = omega^2 (x - u0) + F_ext / m
    """

    def __init__(
        self,
        *,
        gravity: float,
        com_height: float,
        initial_position,
        initial_velocity,
        support_position,
    ) -> None:

        self.gravity = float(gravity)
        self.com_height = float(com_height)

        if (
            not math.isfinite(self.gravity)
            or
            self.gravity <= 0.0
        ):
            raise ValueError(
                "gravity must be finite and positive."
            )

        if (
            not math.isfinite(self.com_height)
            or
            self.com_height <= 0.0
        ):
            raise ValueError(
                "com_height must be finite and positive."
            )

        self.omega = math.sqrt(
            self.gravity
            /
            self.com_height
        )

        self._position = self._as_vector3(
            initial_position,
            name="initial_position",
        )

        self._velocity = self._as_vector3(
            initial_velocity,
            name="initial_velocity",
        )

        self._support_position = self._as_vector3(
            support_position,
            name="support_position",
        )

        # LIPM assumes constant CoM height relative to the
        # support plane.
        self._position[2] = (
            self._support_position[2]
            +
            self.com_height
        )

        self._velocity[2] = 0.0

        self._last_external_acceleration_xy = np.zeros(
            2,
            dtype=float,
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _as_vector3(
        value,
        *,
        name: str,
    ) -> np.ndarray:

        vector = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if vector.size < 3:

            raise ValueError(
                f"{name} must contain at least 3 values."
            )

        vector = vector[:3].copy()

        if not np.all(
            np.isfinite(
                vector
            )
        ):

            raise ValueError(
                f"{name} contains NaN/Inf."
            )

        return vector


    @staticmethod
    def _as_vector2(
        value,
        *,
        name: str,
    ) -> np.ndarray:

        vector = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        if vector.size < 2:

            raise ValueError(
                f"{name} must contain at least 2 values."
            )

        vector = vector[:2].copy()

        if not np.all(
            np.isfinite(
                vector
            )
        ):

            raise ValueError(
                f"{name} contains NaN/Inf."
            )

        return vector


    # ========================================================
    # SUPPORT POINT
    # ========================================================

    def set_support_position(
        self,
        support_position,
    ) -> None:
        """
        Reset the LIPM support/contact point at touchdown.

        This implements the paper's operation

            u0 <- uT

        at the optimized landing time T.

        Horizontal CoM position and velocity are NOT reset.
        Therefore the LIPM state remains continuous across
        the support switch.

        For flat terrain this only changes u_x and u_y.
        z is kept consistent with the fixed CoM-height
        assumption.
        """

        new_support = self._as_vector3(
            support_position,
            name="support_position",
        )

        self._support_position[:] = (
            new_support
        )

        self._position[2] = (
            self._support_position[2]
            +
            self.com_height
        )

        self._velocity[2] = 0.0


    # ========================================================
    # CURRENT STATE
    # ========================================================

    def sample(
        self,
    ) -> LIPMSample:
        """
        Return the current LIPM state without propagating it.
        """

        acceleration = np.zeros(
            3,
            dtype=float,
        )

        acceleration[0:2] = (
            self.omega**2
            *
            (
                self._position[0:2]
                -
                self._support_position[0:2]
            )
            +
            self._last_external_acceleration_xy
        )

        dcm = (
            self._position[0:2]
            +
            self._velocity[0:2]
            /
            self.omega
        )

        return LIPMSample(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            acceleration=acceleration,
            dcm=dcm.copy(),
            support_position=self._support_position.copy(),
            omega=float(self.omega),
            com_height=float(self.com_height),
        )


    # ========================================================
    # FORWARD PROPAGATION
    # ========================================================

    def advance(
        self,
        dt: float,
        *,
        external_force_xy=None,
        mass: float | None = None,
    ) -> LIPMSample:
        """
        Propagate the LIPM forward by dt.

        Nominal dynamics:

            x_ddot = omega^2 (x - u0)

        With an optional constant horizontal external force
        over this interval:

            x_ddot
                = omega^2 (x - u0)
                + F_ext / m

        The exact solution is used over the interval dt.

        Parameters
        ----------
        dt
            Integration interval [s].

        external_force_xy
            Optional [Fx, Fy] force [N]. The force is assumed
            constant during this call.

        mass
            Robot mass [kg], required when external_force_xy
            is supplied.

        Returns
        -------
        LIPMSample
            State after propagation.
        """

        dt = float(dt)

        if (
            not math.isfinite(dt)
            or
            dt < 0.0
        ):
            raise ValueError(
                "dt must be finite and non-negative."
            )

        if dt == 0.0:

            self._last_external_acceleration_xy[:] = 0.0

            return self.sample()

        # ----------------------------------------------------
        # External acceleration
        # ----------------------------------------------------

        external_acceleration_xy = np.zeros(
            2,
            dtype=float,
        )

        if external_force_xy is not None:

            force_xy = self._as_vector2(
                external_force_xy,
                name="external_force_xy",
            )

            if mass is None:

                raise ValueError(
                    "mass is required when "
                    "external_force_xy is supplied."
                )

            mass = float(mass)

            if (
                not math.isfinite(mass)
                or
                mass <= 0.0
            ):
                raise ValueError(
                    "mass must be finite and positive."
                )

            external_acceleration_xy = (
                force_xy
                /
                mass
            )

        self._last_external_acceleration_xy[:] = (
            external_acceleration_xy
        )

        # ----------------------------------------------------
        # Exact LIPM integration
        # ----------------------------------------------------
        #
        # x_ddot = omega^2 (x - u) + a_ext
        #
        # Rewrite as
        #
        # x_ddot = omega^2 (x - u_eff)
        #
        # with
        #
        # u_eff = u - a_ext / omega^2
        #
        # Then use the exact homogeneous LIPM solution.
        # ----------------------------------------------------

        omega = self.omega

        support_xy = (
            self._support_position[0:2]
        )

        effective_support_xy = (
            support_xy
            -
            external_acceleration_xy
            /
            (
                omega**2
            )
        )

        position_xy = (
            self._position[0:2]
        )

        velocity_xy = (
            self._velocity[0:2]
        )

        relative_position = (
            position_xy
            -
            effective_support_xy
        )

        omega_dt = (
            omega
            *
            dt
        )

        cosh_term = math.cosh(
            omega_dt
        )

        sinh_term = math.sinh(
            omega_dt
        )

        new_position_xy = (
            effective_support_xy
            +
            relative_position
            *
            cosh_term
            +
            velocity_xy
            /
            omega
            *
            sinh_term
        )

        new_velocity_xy = (
            omega
            *
            relative_position
            *
            sinh_term
            +
            velocity_xy
            *
            cosh_term
        )

        self._position[0:2] = (
            new_position_xy
        )

        self._velocity[0:2] = (
            new_velocity_xy
        )

        # Fixed CoM height.
        self._position[2] = (
            self._support_position[2]
            +
            self.com_height
        )

        self._velocity[2] = 0.0

        return self.sample()


    # ========================================================
    # OPTIONAL DIRECT STATE RESET
    # ========================================================

    def set_horizontal_state(
        self,
        *,
        position_xy,
        velocity_xy,
    ) -> None:
        """
        Explicitly overwrite the horizontal LIPM state.

        This is not needed in the paper-style nominal loop,
        but is useful for initialization or controlled tests.
        """

        position_xy = self._as_vector2(
            position_xy,
            name="position_xy",
        )

        velocity_xy = self._as_vector2(
            velocity_xy,
            name="velocity_xy",
        )

        self._position[0:2] = (
            position_xy
        )

        self._velocity[0:2] = (
            velocity_xy
        )

        self._last_external_acceleration_xy[:] = 0.0
