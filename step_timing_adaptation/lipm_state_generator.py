# step_timing_adaptation/lipm_state_generator.py

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# ============================================================
# OUTPUT
# ============================================================

@dataclass(frozen=True)
class LIPMSample:

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    dcm: np.ndarray
    support_position: np.ndarray


# ============================================================
# POINT-FOOT LIPM
# ============================================================

class PointFootLIPM:
    """
    Exact horizontal point-foot LIPM state generator.

    Nominal model:

        c_ddot = omega^2 (c - u0)

    where:

        c  : horizontal CoM position
        u0 : stance-foot CoP / ZMP

    With an external horizontal force applied at the CoM:

        c_ddot
            =
            omega^2 (c - u0)
            +
            F_ext / m

    The external force is handled directly in the reduced-order
    LIPM dynamics.

    During one support phase:

        u0 = constant

    At touchdown:

        u0(new) = uT

    Horizontal propagation is analytical.
    Vertical CoM height is constant.
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

        self.gravity = float(
            gravity
        )

        self.com_height = float(
            com_height
        )

        if self.gravity <= 0.0:

            raise ValueError(
                "gravity must be positive."
            )

        if self.com_height <= 0.0:

            raise ValueError(
                "com_height must be positive."
            )

        self.omega = math.sqrt(
            self.gravity
            /
            self.com_height
        )

        position = self._vector3(
            initial_position,
            "initial_position",
        )

        velocity = self._vector3(
            initial_velocity,
            "initial_velocity",
        )

        support = self._vector3(
            support_position,
            "support_position",
        )

        self._position_xy = (
            position[0:2].copy()
        )

        self._velocity_xy = (
            velocity[0:2].copy()
        )

        self._com_z = float(
            position[2]
        )

        self._support_position = (
            support.copy()
        )


    # ========================================================
    # SUPPORT SWITCH
    # ========================================================

    def set_support_position(
        self,
        support_position,
    ) -> None:
        """
        Switch point-foot support without changing CoM
        position or velocity.

        At touchdown:

            u0 <- uT
        """

        support = self._vector3(
            support_position,
            "support_position",
        )

        self._support_position = (
            support.copy()
        )


    # ========================================================
    # CURRENT SAMPLE
    # ========================================================

    def sample(
        self,
    ) -> LIPMSample:

        support_xy = (
            self._support_position[
                0:2
            ]
        )

        acceleration_xy = (
            self.omega**2
            *
            (
                self._position_xy
                -
                support_xy
            )
        )

        dcm = (
            self._position_xy
            +
            self._velocity_xy
            /
            self.omega
        )

        position = np.array(
            [
                self._position_xy[0],
                self._position_xy[1],
                self._com_z,
            ],
            dtype=float,
        )

        velocity = np.array(
            [
                self._velocity_xy[0],
                self._velocity_xy[1],
                0.0,
            ],
            dtype=float,
        )

        acceleration = np.array(
            [
                acceleration_xy[0],
                acceleration_xy[1],
                0.0,
            ],
            dtype=float,
        )

        return LIPMSample(

            position=(
                position
            ),

            velocity=(
                velocity
            ),

            acceleration=(
                acceleration
            ),

            dcm=(
                dcm.copy()
            ),

            support_position=(
                self._support_position.copy()
            ),
        )


    # ========================================================
    # EXACT PROPAGATION
    # ========================================================

    def advance(
        self,
        dt: float,
        *,
        external_force_xy=None,
        mass=None,
    ) -> LIPMSample:
        """
        Exact propagation over dt.

        Without external force:

            c_ddot
                =
                omega^2 (c - u0)

        With a constant external force during this interval:

            c_ddot
                =
                omega^2 (c - u0)
                +
                F / m

        Define:

            a_ext = F / m

        then:

            u_eff
                =
                u0
                -
                a_ext / omega^2

        and the system becomes:

            c_ddot
                =
                omega^2 (c - u_eff)

        Therefore the same analytical LIPM solution can be
        used exactly over the interval.
        """

        dt = float(
            dt
        )

        if (
            not math.isfinite(
                dt
            )
            or
            dt <= 0.0
        ):

            raise ValueError(
                "dt must be positive and finite."
            )

        support_xy = (
            self._support_position[
                0:2
            ]
        )

        # ====================================================
        # EXTERNAL HORIZONTAL FORCE
        # ====================================================

        external_acceleration_xy = np.zeros(
            2,
            dtype=float,
        )

        if external_force_xy is not None:

            force_xy = (
                self._vector2(
                    external_force_xy,
                    "external_force_xy",
                )
            )

            if mass is None:

                raise ValueError(
                    "mass is required when "
                    "external_force_xy is supplied."
                )

            mass = float(
                mass
            )

            if (
                not math.isfinite(
                    mass
                )
                or
                mass <= 0.0
            ):

                raise ValueError(
                    "mass must be positive and finite."
                )

            external_acceleration_xy = (
                force_xy
                /
                mass
            )

        # ====================================================
        # EFFECTIVE SUPPORT POINT
        #
        # c_ddot
        #   = omega^2(c-u0) + a_ext
        #
        #   = omega^2(c-u_eff)
        #
        # u_eff
        #   = u0 - a_ext/omega^2
        # ====================================================

        effective_support_xy = (
            support_xy
            -
            external_acceleration_xy
            /
            self.omega**2
        )

        relative_position = (
            self._position_xy
            -
            effective_support_xy
        )

        omega_dt = (
            self.omega
            *
            dt
        )

        ch = math.cosh(
            omega_dt
        )

        sh = math.sinh(
            omega_dt
        )

        new_position = (
            effective_support_xy
            +
            relative_position
            *
            ch
            +
            self._velocity_xy
            /
            self.omega
            *
            sh
        )

        new_velocity = (
            self.omega
            *
            relative_position
            *
            sh
            +
            self._velocity_xy
            *
            ch
        )

        self._position_xy = (
            new_position
        )

        self._velocity_xy = (
            new_velocity
        )

        return self.sample()


    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _vector2(
        value,
        name: str,
    ) -> np.ndarray:

        value = np.asarray(
            value,
            dtype=float,
        )

        if value.shape != (
            2,
        ):

            raise ValueError(
                f"{name} must have shape (2,)."
            )

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} must contain finite values."
            )

        return value


    @staticmethod
    def _vector3(
        value,
        name: str,
    ) -> np.ndarray:

        value = np.asarray(
            value,
            dtype=float,
        )

        if value.shape != (
            3,
        ):

            raise ValueError(
                f"{name} must have shape (3,)."
            )

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} must contain finite values."
            )

        return value