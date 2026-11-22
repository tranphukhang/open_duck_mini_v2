"""Plot Open Duck Mini V2 states sampled at each policy step.

Figures 1 (base motion) and 3 (foot contacts) update live in GUI mode.
Figure 2 (joint positions) is created only after the simulation. In no-GUI
mode, all figures are deferred until the evaluation has finished.
Nothing is saved to CSV or image files.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import matplotlib.pyplot as plt


LEFT_JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
)
RIGHT_JOINT_NAMES = (
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)


class Plotter:
    """Collect states in RAM and display three Matplotlib figures."""

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
        self.time_s: list[float] = []

        # Figure 1: planar base velocities [m/s].
        self.vx_desired: list[float] = []
        self.vx_actual: list[float] = []
        self.vy_desired: list[float] = []
        self.vy_actual: list[float] = []

        # Figure 1: ZYX Euler orientation of base relative to World [deg].
        self.roll_deg: list[float] = []
        self.pitch_deg: list[float] = []
        self.yaw_deg: list[float] = []

        # Figure 1: angular velocity in the body frame [rad/s].
        self.wx_rad_s: list[float] = []
        self.wy_rad_s: list[float] = []
        self.wz_rad_s: list[float] = []

        # Figure 2: actual absolute joint positions [rad].
        self.left_joint_pos: dict[str, list[float]] = {
            name: [] for name in LEFT_JOINT_NAMES
        }
        self.right_joint_pos: dict[str, list[float]] = {
            name: [] for name in RIGHT_JOINT_NAMES
        }

        # Figure 3: contact = 1 iff ground reaction normal force > 5 N.
        self.contact_left: list[float] = []
        self.contact_right: list[float] = []

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

        self._fig_joints = None
        self._fig_contacts = None
        self._ax_contact_left = None
        self._ax_contact_right = None
        self._line_contact_left = None
        self._line_contact_right = None
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
        contact_left: int | float | None = None,
        contact_right: int | float | None = None,
    ) -> None:
        """Append one sample. All buffers use the same simulation-time axis."""
        angles = (roll_deg, pitch_deg, yaw_deg)
        rates = (wx_rad_s, wy_rad_s, wz_rad_s)
        if any(v is None for v in angles) and not all(v is None for v in angles):
            raise ValueError("Provide roll_deg, pitch_deg, yaw_deg together")
        if any(v is None for v in rates) and not all(v is None for v in rates):
            raise ValueError("Provide wx_rad_s, wy_rad_s, wz_rad_s together")
        if (joint_pos_left is None) != (joint_pos_right is None):
            raise ValueError("Provide joint_pos_left and joint_pos_right together")
        if (contact_left is None) != (contact_right is None):
            raise ValueError("Provide contact_left and contact_right together")

        if joint_pos_left is None:
            left = (math.nan,) * 5
            right = (math.nan,) * 5
        else:
            if len(joint_pos_left) != 5 or len(joint_pos_right) != 5:
                raise ValueError("Each leg must provide exactly five joint positions")
            left = tuple(map(float, joint_pos_left))
            right = tuple(map(float, joint_pos_right))

        if contact_left is None:
            contacts = (math.nan, math.nan)
        else:
            contacts = (float(contact_left), float(contact_right))
            if any(not (v in (0.0, 1.0) or math.isnan(v)) for v in contacts):
                raise ValueError("Foot contacts must be 0, 1 or NaN (reset marker)")

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

        for name, angle in zip(LEFT_JOINT_NAMES, left, strict=True):
            self.left_joint_pos[name].append(angle)
        for name, angle in zip(RIGHT_JOINT_NAMES, right, strict=True):
            self.right_joint_pos[name].append(angle)
        self.contact_left.append(contacts[0])
        self.contact_right.append(contacts[1])

    def start(self) -> None:
        """Create Figures 1 and 3 for live display (GUI mode only)."""
        if self._fig is not None and plt.fignum_exists(self._fig.number):
            return

        plt.ion()
        self._fig, (self._ax_x, self._ax_y, self._ax_rpy, self._ax_ang_vel) = (
            plt.subplots(
                nrows=4, ncols=1, sharex=True, figsize=(11, 11),
                num="Base motion",
            )
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

        # Figure 3 also updates live when --gui is enabled.
        # Figure 2 remains deferred until show().
        self._create_contact_figure()
        plt.show(block=False)

    def refresh(self, force: bool = False) -> None:
        """Refresh live Figures 1 and 3, but never redraw Figure 2."""
        if not self.time_s:
            return

        # With --no-gui, run.py never calls refresh() during simulation.
        # At show() time, start() creates Figures 1 and 3 from buffered data.
        if self._fig is None and self._fig_contacts is None:
            self.start()

        now = time.perf_counter()
        if not force and now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now

        if self._fig is not None and plt.fignum_exists(self._fig.number):
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

        if self._fig_contacts is not None and plt.fignum_exists(self._fig_contacts.number):
            self._line_contact_left.set_data(self.time_s, self.contact_left)
            self._line_contact_right.set_data(self.time_s, self.contact_right)
            for ax in (self._ax_contact_left, self._ax_contact_right):
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=False)
            self._fig_contacts.canvas.draw_idle()

        # Process Matplotlib events once for both live windows.
        plt.pause(0.001)

    def _create_joint_figure(self) -> None:
        """Create Figure 2 only after simulation ends."""
        if self._fig_joints is not None and plt.fignum_exists(self._fig_joints.number):
            return
        self._fig_joints, (ax_left, ax_right) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(11, 8),
            num="Joint positions",
        )
        for name in LEFT_JOINT_NAMES:
            ax_left.plot(self.time_s, self.left_joint_pos[name], label=name)
        for name in RIGHT_JOINT_NAMES:
            ax_right.plot(self.time_s, self.right_joint_pos[name], label=name)
        for ax, title in (
            (ax_left, "Left leg joint positions"),
            (ax_right, "Right leg joint positions"),
        ):
            ax.set_title(title)
            ax.set_ylabel("Joint position (rad)")
            ax.grid(True)
            ax.legend(loc="upper right", ncol=2, fontsize="small")
        ax_right.set_xlabel("Simulation time (s)")
        self._fig_joints.tight_layout()

    def _create_contact_figure(self) -> None:
        """Create Figure 3 for live binary contact histories (GUI mode)."""
        if self._fig_contacts is not None and plt.fignum_exists(self._fig_contacts.number):
            return
        self._fig_contacts, (self._ax_contact_left, self._ax_contact_right) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(11, 6),
            num="Foot contacts (> 5 N)",
        )
        (self._line_contact_left,) = self._ax_contact_left.step(
            [], [], where="post", label="left_contact",
        )
        (self._line_contact_right,) = self._ax_contact_right.step(
            [], [], where="post", label="right_contact",
        )
        for ax, title in (
            (self._ax_contact_left, "Left foot ground contact (normal force > 5 N)"),
            (self._ax_contact_right, "Right foot ground contact (normal force > 5 N)"),
        ):
            ax.set_title(title)
            ax.set_ylabel("Contact (0/1)")
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True)
            ax.legend(loc="upper right")
        self._ax_contact_right.set_xlabel("Simulation time (s)")
        self._fig_contacts.tight_layout()

    def show(self) -> None:
        """Finalize live Figures 1/3 and create Figure 2 after simulation."""
        if not self.time_s:
            print("[Plotter] No policy samples to plot.")
            return
        # In --no-gui mode, no figures were opened during evaluation.
        if self._fig is None:
            self.start()
        self._create_contact_figure()
        self.refresh(force=True)
        self._create_joint_figure()
        plt.ioff()
        plt.show()
