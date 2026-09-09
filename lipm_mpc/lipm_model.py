# lipm_mpc/lipm_model.py

from __future__ import annotations

import numpy as np


class LIPMModel1D:
    """
    One-dimensional CoM model used by the LIPM-MPC.

    State:
        x = [p_G, v_G, a_G]^T

    Control:
        u = jerk = d(a_G)/dt

    Discrete state equation:
        x_{k+1} = A x_k + B u_k

    LIPM ZMP relation:
        p_Z = p_G - (h / g) * a_G

    The same model can be used independently for the x-axis
    and the y-axis.
    """

    STATE_SIZE = 3

    def __init__(
        self,
        timestep: float,
        com_height: float,
        gravity: float,
    ) -> None:
        self._validate_parameters(
            timestep=timestep,
            com_height=com_height,
            gravity=gravity,
        )

        self.timestep = float(timestep)
        self.com_height = float(com_height)
        self.gravity = float(gravity)

        self.A = self._build_state_matrix()
        self.B = self._build_input_matrix()
        self.C_zmp = self._build_zmp_matrix()

    # ========================================================
    # Model matrices
    # ========================================================

    def _build_state_matrix(self) -> np.ndarray:
        """
        Build the discrete-time state matrix A.

        State:
            [position, velocity, acceleration]^T
        """
        T = self.timestep

        return np.array(
            [
                [1.0, T, 0.5 * T**2],
                [0.0, 1.0, T],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _build_input_matrix(self) -> np.ndarray:
        """
        Build the discrete-time input matrix B.

        Control input:
            jerk [m/s^3]
        """
        T = self.timestep

        return np.array(
            [
                T**3 / 6.0,
                T**2 / 2.0,
                T,
            ],
            dtype=float,
        )

    def _build_zmp_matrix(self) -> np.ndarray:
        """
        Build the LIPM ZMP output matrix.

        p_Z = C_zmp @ x

        where:
            C_zmp = [1, 0, -h/g]
        """
        return np.array(
            [
                1.0,
                0.0,
                -self.com_height / self.gravity,
            ],
            dtype=float,
        )

    # ========================================================
    # State propagation
    # ========================================================

    def propagate(
        self,
        state: np.ndarray,
        jerk: float,
    ) -> np.ndarray:
        """
        Propagate the CoM state by one MPC timestep.

        Parameters
        ----------
        state:
            Current state:
                [position, velocity, acceleration]

        jerk:
            Constant jerk applied during one MPC interval.

        Returns
        -------
        np.ndarray
            Next state:
                [position, velocity, acceleration]
        """
        state = self._validate_state(state)
        jerk = float(jerk)

        if not np.isfinite(jerk):
            raise ValueError("jerk must be finite.")

        return self.A @ state + self.B * jerk

    # ========================================================
    # LIPM / ZMP
    # ========================================================

    def compute_zmp(self, state: np.ndarray) -> float:
        """
        Compute the ZMP position from the CoM state.

        LIPM relation:
            p_Z = p_G - (h/g) * a_G
        """
        state = self._validate_state(state)

        return float(self.C_zmp @ state)

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_parameters(
        timestep: float,
        com_height: float,
        gravity: float,
    ) -> None:
        if timestep <= 0.0:
            raise ValueError("timestep must be greater than zero.")

        if com_height <= 0.0:
            raise ValueError("com_height must be greater than zero.")

        if gravity <= 0.0:
            raise ValueError("gravity must be greater than zero.")

    @classmethod
    def _validate_state(cls, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)

        if state.shape != (cls.STATE_SIZE,):
            raise ValueError(
                "state must have shape (3,), "
                "[position, velocity, acceleration]."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError("state must contain only finite values.")

        return state