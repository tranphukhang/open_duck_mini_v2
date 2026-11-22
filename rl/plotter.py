"""Plot commanded and actual planar base velocities at each policy step.

This module only stores samples in RAM and displays Matplotlib plots.
No CSV or figure files are written.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt


class Plotter:
    """Plot vx/vy desired versus actual using simulation time."""

    def __init__(self, policy_dt: float = 0.02, refresh_interval: float = 0.10):
        if policy_dt <= 0.0:
            raise ValueError("policy_dt must be positive")
        if refresh_interval < 0.0:
            raise ValueError("refresh_interval must be nonnegative")

        self.policy_dt = float(policy_dt)
        self.refresh_interval = float(refresh_interval)

        self.time_s: list[float] = []
        self.vx_desired: list[float] = []
        self.vx_actual: list[float] = []
        self.vy_desired: list[float] = []
        self.vy_actual: list[float] = []

        self._fig = None
        self._ax_x = None
        self._ax_y = None
        self._line_vx_desired = None
        self._line_vx_actual = None
        self._line_vy_desired = None
        self._line_vy_actual = None
        self._last_refresh = 0.0

    def add_sample(
        self,
        vx_desired: float,
        vx_actual: float,
        vy_desired: float,
        vy_actual: float,
    ) -> None:
        """Add exactly one sample per policy step (not per physics substep)."""
        self.time_s.append(len(self.time_s) * self.policy_dt)
        self.vx_desired.append(float(vx_desired))
        self.vx_actual.append(float(vx_actual))
        self.vy_desired.append(float(vy_desired))
        self.vy_actual.append(float(vy_actual))

    def start(self) -> None:
        """Open a single window containing separate x and y velocity plots."""
        if self._fig is not None:
            return

        plt.ion()
        self._fig, (self._ax_x, self._ax_y) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(10, 7)
        )
        (self._line_vx_desired,) = self._ax_x.plot([], [], label="x_velocity_desired")
        (self._line_vx_actual,) = self._ax_x.plot([], [], label="x_velocity_actual")
        (self._line_vy_desired,) = self._ax_y.plot([], [], label="y_velocity_desired")
        (self._line_vy_actual,) = self._ax_y.plot([], [], label="y_velocity_actual")

        for ax, title in (
            (self._ax_x, "Linear velocity along body X"),
            (self._ax_y, "Linear velocity along body Y"),
        ):
            ax.set_title(title)
            ax.set_ylabel("Velocity (m/s)")
            ax.grid(True)
            ax.legend(loc="upper right")
        self._ax_y.set_xlabel("Simulation time (s)")
        self._fig.tight_layout()
        plt.show(block=False)

    def refresh(self, force: bool = False) -> None:
        """Update both plots; redraw rate is limited to save CPU time."""
        if not self.time_s:
            return
        if self._fig is None:
            self.start()
        if not plt.fignum_exists(self._fig.number):
            return

        now = time.perf_counter()
        if not force and now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now

        self._line_vx_desired.set_data(self.time_s, self.vx_desired)
        self._line_vx_actual.set_data(self.time_s, self.vx_actual)
        self._line_vy_desired.set_data(self.time_s, self.vy_desired)
        self._line_vy_actual.set_data(self.time_s, self.vy_actual)

        for ax in (self._ax_x, self._ax_y):
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def show(self) -> None:
        """Display the completed plots and wait until the window is closed."""
        if not self.time_s:
            print("[VelocityPlotter] No policy samples to plot.")
            return
        self.refresh(force=True)
        plt.ioff()
        plt.show()
