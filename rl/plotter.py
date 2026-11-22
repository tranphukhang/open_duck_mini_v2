"""Plot base motion live; show left/right joint positions after simulation.

All signals are sampled at each policy step and kept in RAM.
Figure 1 refreshes during simulation; Figure 2 is created only in show().
No CSV or figure files are written.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import matplotlib.pyplot as plt


LEFT_JOINT_NAMES = (
    "L_hip_yaw",
    "L_hip_roll",
    "L_hip_pitch",
    "L_knee",
    "L_ankle",
)
RIGHT_JOINT_NAMES = (
    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "R_ankle",
)


class Plotter:
    """Plot base motion live and joint positions after the simulation."""

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

        # One sample for t=0, then one per policy step.
        self.time_s: list[float] = []

        # Figure 1: base linear velocities, expressed in body frame [m/s].
        self.vx_desired: list[float] = []
        self.vx_actual: list[float] = []
        self.vy_desired: list[float] = []
        self.vy_actual: list[float] = []

        # Figure 1: ZYX Euler orientation of base relative to World [deg].
        self.roll_deg: list[float] = []
        self.pitch_deg: list[float] = []
        self.yaw_deg: list[float] = []

        # Figure 1: angular velocity of base expressed in body frame [rad/s].
        self.wx_rad_s: list[float] = []
        self.wy_rad_s: list[float] = []
        self.wz_rad_s: list[float] = []

        # Figure 2: actual, absolute joint angles [rad], not joint_pos_rel.
        self.left_joint_pos: dict[str, list[float]] = {
            name: [] for name in LEFT_JOINT_NAMES
        }
        self.right_joint_pos: dict[str, list[float]] = {
            name: [] for name in RIGHT_JOINT_NAMES
        }

        # Keep Figure 1's existing names/behavior.
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

        # Figure 2 is created only when simulation finishes (show()).
        self._fig_joints = None
        self._ax_left = None
        self._ax_right = None
        self._left_lines: dict[str, object] = {}
        self._right_lines: dict[str, object] = {}

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
        joint_pos_left: Sequence[float] | None = None,
        joint_pos_right: Sequence[float] | None = None,
    ) -> None:
        """Add one complete sample; all buffers share the same time axis.

        The joint sequences must follow LEFT_JOINT_NAMES and RIGHT_JOINT_NAMES.
        If both are omitted, NaNs are recorded for backward compatibility.
        """
        angles = (roll_deg, pitch_deg, yaw_deg)
        ang_vel = (wx_rad_s, wy_rad_s, wz_rad_s)
        if any(v is None for v in angles) and not all(v is None for v in angles):
            raise ValueError("Provide roll_deg, pitch_deg, yaw_deg together")
        if any(v is None for v in ang_vel) and not all(v is None for v in ang_vel):
            raise ValueError("Provide wx_rad_s, wy_rad_s, wz_rad_s together")
        if (joint_pos_left is None) != (joint_pos_right is None):
            raise ValueError("Provide joint_pos_left and joint_pos_right together")

        if joint_pos_left is None:
            left_values = (math.nan,) * len(LEFT_JOINT_NAMES)
            right_values = (math.nan,) * len(RIGHT_JOINT_NAMES)
        else:
            if (len(joint_pos_left) != len(LEFT_JOINT_NAMES)
                    or len(joint_pos_right) != len(RIGHT_JOINT_NAMES)):
                raise ValueError("Each leg must provide exactly five joint angles")
            left_values = tuple(float(x) for x in joint_pos_left)
            right_values = tuple(float(x) for x in joint_pos_right)

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

        for name, angle in zip(LEFT_JOINT_NAMES, left_values, strict=True):
            self.left_joint_pos[name].append(angle)
        for name, angle in zip(RIGHT_JOINT_NAMES, right_values, strict=True):
            self.right_joint_pos[name].append(angle)

    def start(self) -> None:
        """Open ONLY Figure 1; defer Figure 2 until simulation is finished."""
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
            num="Base motion",
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

        # Do not create Figure 2 while the policy is running.
        plt.show(block=False)

    def refresh(self, force: bool = False) -> None:
        """Refresh ONLY Figure 1; sampling remains one sample/policy step."""
        if not self.time_s:
            return
        if self._fig is None:
            self.start()

        now = time.perf_counter()
        if not force and now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now

        # Only Figure 1 is redrawn during the simulation.
        if not plt.fignum_exists(self._fig.number):
            return

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

    def _create_joint_figure(self) -> None:
        """Create Figure 2 once, after the policy simulation has completed."""
        if self._fig_joints is not None and plt.fignum_exists(self._fig_joints.number):
            return

        self._fig_joints, (self._ax_left, self._ax_right) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(11, 8),
            num="Joint positions",
        )

        # Plot complete histories only once, not at every policy step.
        self._left_lines = {}
        self._right_lines = {}
        for name in LEFT_JOINT_NAMES:
            (line,) = self._ax_left.plot(
                self.time_s, self.left_joint_pos[name], label=name
            )
            self._left_lines[name] = line
        for name in RIGHT_JOINT_NAMES:
            (line,) = self._ax_right.plot(
                self.time_s, self.right_joint_pos[name], label=name
            )
            self._right_lines[name] = line

        for ax, title in (
            (self._ax_left, "Left leg joint positions"),
            (self._ax_right, "Right leg joint positions"),
        ):
            ax.set_title(title)
            ax.set_ylabel("Joint position (rad)")
            ax.grid(True)
            ax.legend(loc="upper right", ncol=2, fontsize="small")

        self._ax_right.set_xlabel("Simulation time (s)")
        self._fig_joints.tight_layout()

    def show(self) -> None:
        """Finalize Figure 1 and show Figure 2 only after simulation ends."""
        if not self.time_s:
            print("[Plotter] No policy samples to plot.")
            return
        self.refresh(force=True)
        self._create_joint_figure()
        plt.ioff()
        plt.show()
