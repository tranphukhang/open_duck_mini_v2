"""Plot planar velocity, base orientation, and body angular velocity.

Samples are kept in RAM and displayed with Matplotlib.
No CSV or figure files are written.
"""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt


class Plotter:
    """Plot the base motion at each policy step (simulation time)."""

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

        # Data buffers (one entry per policy step, plus t=0).
        self.time_s: list[float] = []

        # Linear velocities, expressed in the robot body frame [m/s].
        self.vx_desired: list[float] = []
        self.vx_actual: list[float] = []
        self.vy_desired: list[float] = []
        self.vy_actual: list[float] = []

        # Base orientation relative to World, ZYX Euler angles [deg].
        self.roll_deg: list[float] = []
        self.pitch_deg: list[float] = []
        self.yaw_deg: list[float] = []

        # Base angular velocity in the robot body frame [rad/s].
        self.wx_rad_s: list[float] = []
        self.wy_rad_s: list[float] = []
        self.wz_rad_s: list[float] = []

        self._fig = None
        self._ax_x = None
        self._ax_y = None
        self._ax_rpy = None
        self._ax_ang_vel = None

        self._line_vx_desired = None
        self._line_vx_actual = None
        self._line_vy_desired = None
        self._line_vy_actual = None
        self._line_roll = None
        self._line_pitch = None
        self._line_yaw = None
        self._line_wx = None
        self._line_wy = None
        self._line_wz = None

        self._last_refresh = 0.0

    def add_sample(
        self,
        vx_desired: float,
        vx_actual: float,
        vy_desired: float,
        vy_actual: float,
        roll_deg: float | None = None,
        pitch_deg: float | None = None,
        yaw_deg: float | None = None,
        wx_rad_s: float | None = None,
        wy_rad_s: float | None = None,
        wz_rad_s: float | None = None,
    ) -> None:
        """Record one sample at the current policy step.

        Orientation and angular-velocity triplets are optional to remain
        compatible with older callers; an omitted triplet is stored as NaN.
        """
        angles = (roll_deg, pitch_deg, yaw_deg)
        ang_vel = (wx_rad_s, wy_rad_s, wz_rad_s)
        if any(v is None for v in angles) and not all(v is None for v in angles):
            raise ValueError("Provide roll_deg, pitch_deg, yaw_deg together")
        if any(v is None for v in ang_vel) and not all(v is None for v in ang_vel):
            raise ValueError("Provide wx_rad_s, wy_rad_s, wz_rad_s together")

        self.time_s.append(len(self.time_s) * self.policy_dt)
        self.vx_desired.append(float(vx_desired))
        self.vx_actual.append(float(vx_actual))
        self.vy_desired.append(float(vy_desired))
        self.vy_actual.append(float(vy_actual))

        self.roll_deg.append(math.nan if roll_deg is None else float(roll_deg))
        self.pitch_deg.append(math.nan if pitch_deg is None else float(pitch_deg))
        self.yaw_deg.append(math.nan if yaw_deg is None else float(yaw_deg))

        self.wx_rad_s.append(math.nan if wx_rad_s is None else float(wx_rad_s))
        self.wy_rad_s.append(math.nan if wy_rad_s is None else float(wy_rad_s))
        self.wz_rad_s.append(math.nan if wz_rad_s is None else float(wz_rad_s))

    def start(self) -> None:
        """Open a window with four vertically stacked plots."""
        if self._fig is not None:
            return

        plt.ion()
        self._fig, (
            self._ax_x,
            self._ax_y,
            self._ax_rpy,
            self._ax_ang_vel,
        ) = plt.subplots(
            nrows=4,
            ncols=1,
            sharex=True,
            figsize=(11, 11),
        )

        (self._line_vx_desired,) = self._ax_x.plot([], [], label="x_velocity_desired")
        (self._line_vx_actual,) = self._ax_x.plot([], [], label="x_velocity_actual")
        (self._line_vy_desired,) = self._ax_y.plot([], [], label="y_velocity_desired")
        (self._line_vy_actual,) = self._ax_y.plot([], [], label="y_velocity_actual")
        (self._line_roll,) = self._ax_rpy.plot([], [], label="roll")
        (self._line_pitch,) = self._ax_rpy.plot([], [], label="pitch")
        (self._line_yaw,) = self._ax_rpy.plot([], [], label="yaw")
        (self._line_wx,) = self._ax_ang_vel.plot([], [], label="omega_x")
        (self._line_wy,) = self._ax_ang_vel.plot([], [], label="omega_y")
        (self._line_wz,) = self._ax_ang_vel.plot([], [], label="omega_z")

        for ax, title, ylabel in (
            (self._ax_x, "Linear velocity along body X", "Velocity (m/s)"),
            (self._ax_y, "Linear velocity along body Y", "Velocity (m/s)"),
            (self._ax_rpy, "Base orientation relative to World", "Angle (deg)"),
            (self._ax_ang_vel, "Base angular velocity in body frame", "Angular velocity (rad/s)"),
        ):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc="upper right")
        self._ax_ang_vel.set_xlabel("Simulation time (s)")

        self._fig.tight_layout()
        plt.show(block=False)

    def refresh(self, force: bool = False) -> None:
        """Update all plots; redraw throttle does not alter sampling rate."""
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

        for line, values in (
            (self._line_vx_desired, self.vx_desired),
            (self._line_vx_actual, self.vx_actual),
            (self._line_vy_desired, self.vy_desired),
            (self._line_vy_actual, self.vy_actual),
            (self._line_roll, self.roll_deg),
            (self._line_pitch, self.pitch_deg),
            (self._line_yaw, self.yaw_deg),
            (self._line_wx, self.wx_rad_s),
            (self._line_wy, self.wy_rad_s),
            (self._line_wz, self.wz_rad_s),
        ):
            line.set_data(self.time_s, values)

        for ax in (self._ax_x, self._ax_y, self._ax_rpy, self._ax_ang_vel):
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def show(self) -> None:
        """Show the completed plots, keeping all data in memory only."""
        if not self.time_s:
            print("[Plotter] No policy samples to plot.")
            return
        self.refresh(force=True)
        plt.ioff()
        plt.show()
