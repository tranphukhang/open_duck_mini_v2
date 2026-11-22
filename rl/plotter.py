"""Plot linear velocity and base orientation at each policy step.

This module stores samples in RAM and displays Matplotlib plots.
No CSV or figure files are written.
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt


class Plotter:
    """Plot velocity tracking and base roll/pitch/yaw."""

    def __init__(
        self,
        policy_dt: float = 0.02,
        refresh_interval: float = 0.10,
    ):
        if policy_dt <= 0.0:
            raise ValueError("policy_dt must be positive")

        if refresh_interval < 0.0:
            raise ValueError("refresh_interval must be nonnegative")

        self.policy_dt = float(policy_dt)
        self.refresh_interval = float(refresh_interval)

        # ====================================================
        # Data buffers
        # ====================================================

        self.time_s: list[float] = []

        # Linear velocity [m/s]
        self.vx_desired: list[float] = []
        self.vx_actual: list[float] = []

        self.vy_desired: list[float] = []
        self.vy_actual: list[float] = []

        # Base orientation relative to World [deg]
        self.roll_deg: list[float] = []
        self.pitch_deg: list[float] = []
        self.yaw_deg: list[float] = []

        # ====================================================
        # Matplotlib objects
        # ====================================================

        self._fig = None

        self._ax_x = None
        self._ax_y = None
        self._ax_rpy = None

        self._line_vx_desired = None
        self._line_vx_actual = None

        self._line_vy_desired = None
        self._line_vy_actual = None

        self._line_roll = None
        self._line_pitch = None
        self._line_yaw = None

        self._last_refresh = 0.0

    # ========================================================
    # Add Sample
    # ========================================================

    def add_sample(
        self,
        vx_desired: float,
        vx_actual: float,
        vy_desired: float,
        vy_actual: float,
        roll_deg: float | None = None,
        pitch_deg: float | None = None,
        yaw_deg: float | None = None,
    ) -> None:
        """Add exactly one sample per policy step."""

        angles = (roll_deg, pitch_deg, yaw_deg)

        if (
            any(value is None for value in angles)
            and not all(value is None for value in angles)
        ):
            raise ValueError(
                "Provide roll_deg, pitch_deg, yaw_deg together"
            )

        # Simulation time [s]
        t = len(self.time_s) * self.policy_dt
        self.time_s.append(t)

        # Linear velocity [m/s]
        self.vx_desired.append(float(vx_desired))
        self.vx_actual.append(float(vx_actual))

        self.vy_desired.append(float(vy_desired))
        self.vy_actual.append(float(vy_actual))

        # Base orientation [deg]
        self.roll_deg.append(
            math.nan if roll_deg is None else float(roll_deg)
        )

        self.pitch_deg.append(
            math.nan if pitch_deg is None else float(pitch_deg)
        )

        self.yaw_deg.append(
            math.nan if yaw_deg is None else float(yaw_deg)
        )

    # ========================================================
    # Initialize Plots
    # ========================================================

    def start(self) -> None:
        """Open one window containing three plots."""

        if self._fig is not None:
            return

        plt.ion()

        self._fig, (
            self._ax_x,
            self._ax_y,
            self._ax_rpy,
        ) = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            figsize=(11, 9),
        )

        # ====================================================
        # X Velocity
        # ====================================================

        (self._line_vx_desired,) = self._ax_x.plot(
            [], [],
            label="x_velocity_desired",
        )

        (self._line_vx_actual,) = self._ax_x.plot(
            [], [],
            label="x_velocity_actual",
        )

        # ====================================================
        # Y Velocity
        # ====================================================

        (self._line_vy_desired,) = self._ax_y.plot(
            [], [],
            label="y_velocity_desired",
        )

        (self._line_vy_actual,) = self._ax_y.plot(
            [], [],
            label="y_velocity_actual",
        )

        # ====================================================
        # Base Orientation
        # ====================================================

        (self._line_roll,) = self._ax_rpy.plot(
            [], [],
            label="roll",
        )

        (self._line_pitch,) = self._ax_rpy.plot(
            [], [],
            label="pitch",
        )

        (self._line_yaw,) = self._ax_rpy.plot(
            [], [],
            label="yaw",
        )

        # ====================================================
        # Axes
        # ====================================================

        for ax, title, ylabel in (
            (
                self._ax_x,
                "Linear velocity along body X",
                "Velocity (m/s)",
            ),
            (
                self._ax_y,
                "Linear velocity along body Y",
                "Velocity (m/s)",
            ),
            (
                self._ax_rpy,
                "Base orientation relative to World",
                "Angle (deg)",
            ),
        ):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc="upper right")

        self._ax_rpy.set_xlabel("Simulation time (s)")

        self._fig.tight_layout()

        plt.show(block=False)

    # ========================================================
    # Refresh Plots
    # ========================================================

    def refresh(self, force: bool = False) -> None:
        """Update all plots without changing the sampling rate."""

        if not self.time_s:
            return

        if self._fig is None:
            self.start()

        if not plt.fignum_exists(self._fig.number):
            return

        now = time.perf_counter()

        if not force:
            elapsed = now - self._last_refresh

            if elapsed < self.refresh_interval:
                return

        self._last_refresh = now

        # X velocity
        self._line_vx_desired.set_data(
            self.time_s, self.vx_desired
        )

        self._line_vx_actual.set_data(
            self.time_s, self.vx_actual
        )

        # Y velocity
        self._line_vy_desired.set_data(
            self.time_s, self.vy_desired
        )

        self._line_vy_actual.set_data(
            self.time_s, self.vy_actual
        )

        # Roll, Pitch, Yaw
        self._line_roll.set_data(
            self.time_s, self.roll_deg
        )

        self._line_pitch.set_data(
            self.time_s, self.pitch_deg
        )

        self._line_yaw.set_data(
            self.time_s, self.yaw_deg
        )

        # Update axes
        for ax in (
            self._ax_x,
            self._ax_y,
            self._ax_rpy,
        ):
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    # ========================================================
    # Show Final Plots
    # ========================================================

    def show(self) -> None:
        """Display the completed plots."""

        if not self.time_s:
            print("[Plotter] No policy samples to plot.")
            return

        self.refresh(force=True)

        plt.ioff()
        plt.show()