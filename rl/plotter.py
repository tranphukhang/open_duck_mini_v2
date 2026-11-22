"""Plot Open Duck Mini V2 states sampled at each policy step.

Figure 1: base motion. Figure 2: joint positions (deferred).
Figure 3: foot contacts. Figure 4: support-boundary history and ZMP
trajectory in World XY. Each support boundary is a separate, closed
convex-hull outline (never joined to the next time step). ALL figures are
created only after the rollout, regardless of MuJoCo GUI settings.
Nothing is saved to disk.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


LEFT_JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
)
RIGHT_JOINT_NAMES = (
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)


class Plotter:
    """Record policy-rate data in RAM and display four Matplotlib figures."""

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

        # Figure 1: base linear velocity [m/s], expressed in body frame.
        self.vx_desired: list[float] = []
        self.vx_actual: list[float] = []
        self.vy_desired: list[float] = []
        self.vy_actual: list[float] = []

        # Figure 1: base orientation wrt World (ZYX Euler) [deg].
        self.roll_deg: list[float] = []
        self.pitch_deg: list[float] = []
        self.yaw_deg: list[float] = []

        # Figure 1: base angular rate expressed in body frame [rad/s].
        self.wx_rad_s: list[float] = []
        self.wy_rad_s: list[float] = []
        self.wz_rad_s: list[float] = []

        # Figure 2: actual joint positions stored in radians.
        # Conversion to degrees is performed only when plotting.
        self.left_joint_pos: dict[str, list[float]] = {
            name: [] for name in LEFT_JOINT_NAMES
        }
        self.right_joint_pos: dict[str, list[float]] = {
            name: [] for name in RIGHT_JOINT_NAMES
        }

        # Figure 3: foot contact thresholded at normal ground reaction > 5 N.
        self.contact_left: list[float] = []
        self.contact_right: list[float] = []

        # Figure 4: each active support polygon in World XY [m] and net ZMP [m].
        # The polygon is a convex sequence of (x,y) vertices, unclosed.
        # Empty polygon = no active support or a reset marker.
        self.support_polygon_xy: list[tuple[tuple[float, float], ...]] = []
        self.zmp_x_w: list[float] = []
        self.zmp_y_w: list[float] = []

        # Figure 1 handles.
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

        # Figure 2 is created only in show().
        self._fig_joints = None

        # Figure 3 handles.
        self._fig_contacts = None
        self._ax_contact_left = None
        self._ax_contact_right = None
        self._line_contact_left = None
        self._line_contact_right = None

        # Figure 4 handles.
        self._fig_support = None
        self._ax_support = None
        self._support_history = None
        self._support_outline = None
        self._zmp_trail = None
        self._zmp_current = None

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
        support_polygon_xy: Sequence[Sequence[float]] | None = None,
        zmp_x_w: float | None = None,
        zmp_y_w: float | None = None,
    ) -> None:
        """Append exactly one policy-step sample to all data buffers."""
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
        if (zmp_x_w is None) != (zmp_y_w is None):
            raise ValueError("Provide zmp_x_w and zmp_y_w together")

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

        if support_polygon_xy is None:
            polygon: tuple[tuple[float, float], ...] = ()
        else:
            polygon = tuple((float(p[0]), float(p[1])) for p in support_polygon_xy)
            if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in polygon):
                raise ValueError("Support polygon coordinates must be finite")

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
        self.support_polygon_xy.append(polygon)
        self.zmp_x_w.append(math.nan if zmp_x_w is None else float(zmp_x_w))
        self.zmp_y_w.append(math.nan if zmp_y_w is None else float(zmp_y_w))

    def start(self) -> None:
        """Create Figures 1, 3 and 4 after the simulation finishes."""
        if self._fig is not None and plt.fignum_exists(self._fig.number):
            return

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

        self._create_contact_figure()
        self._create_support_figure()

    def refresh(self, force: bool = False) -> None:
        """Populate buffered line artists once, when the rollout ends."""
        if not self.time_s:
            return
        if self._fig is None and self._fig_contacts is None and self._fig_support is None:
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

        if self._fig_support is not None and plt.fignum_exists(self._fig_support.number):
            self._refresh_support_figure()
            self._fig_support.canvas.draw_idle()

    def _create_joint_figure(self) -> None:
        """Create Figure 2 with joint positions converted to degrees."""
        if self._fig_joints is not None and plt.fignum_exists(self._fig_joints.number):
            return

        self._fig_joints, (ax_left, ax_right) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(11, 8),
            num="Joint positions",
        )

        # Left leg joint positions (rad -> deg)
        for name in LEFT_JOINT_NAMES:
            ax_left.plot(
                self.time_s,
                np.rad2deg(self.left_joint_pos[name]),
                label=name,
            )

        # Right leg joint positions (rad -> deg)
        for name in RIGHT_JOINT_NAMES:
            ax_right.plot(
                self.time_s,
                np.rad2deg(self.right_joint_pos[name]),
                label=name,
            )

        for ax, title in (
            (ax_left, "Left leg joint positions"),
            (ax_right, "Right leg joint positions"),
        ):
            ax.set_title(title)
            ax.set_ylabel("Joint position (deg)")
            ax.grid(True)
            ax.legend(loc="upper right", ncol=2, fontsize="small")

        ax_right.set_xlabel("Simulation time (s)")
        self._fig_joints.tight_layout()

    def _create_contact_figure(self) -> None:
        """Create Figure 3 for live binary foot contacts."""
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

    def _create_support_figure(self) -> None:
        """Create Figure 4: past/current support OUTLINES and measured ZMP.

        Each polygon is already the convex hull of active foot-patch corners.
        In DSP, we draw ONLY the exterior hull, never the two overlapping
        rectangles or artificial lines between successive polygons.
        """
        if self._fig_support is not None and plt.fignum_exists(self._fig_support.number):
            return

        self._fig_support, self._ax_support = plt.subplots(
            figsize=(10, 8), num="Support polygon history and ZMP (World XY)"
        )
        ax = self._ax_support

        # One independent CLOSED path per policy step, color-coded by its
        # simulation timestamp. LineCollection avoids hundreds of artists.
        self._support_history = LineCollection(
            [],
            cmap="viridis",
            norm=Normalize(vmin=0.0, vmax=max(self.policy_dt, 10.0)),
            linewidths=0.9,
            alpha=0.32,
            zorder=2,
            label="Past support boundaries",
        )
        ax.add_collection(self._support_history)
        colorbar = self._fig_support.colorbar(
            self._support_history, ax=ax, pad=0.02, fraction=0.045
        )
        colorbar.set_label("Simulation time (s)")

        (self._support_outline,) = ax.plot(
            [], [], color="C0", linewidth=2.5, zorder=5,
            label="Current support boundary"
        )
        (self._zmp_trail,) = ax.plot(
            [], [], color="C3", linewidth=1.2, alpha=0.85,
            zorder=3, label="ZMP trajectory"
        )
        (self._zmp_current,) = ax.plot(
            [], [], marker="o", linestyle="None", color="C3",
            markersize=8, zorder=6, label="Current ZMP"
        )

        ax.set_title("Support boundary history and measured ZMP")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend(loc="best", fontsize="small")
        self._fig_support.tight_layout()

    def _refresh_support_figure(self) -> None:
        """Update all past hull outlines, the latest hull and the ZMP path.

        Boundary shapes are NOT joined between time steps: every stored
        polygon is rendered as its own closed contour. No fill is used.
        """
        # A NaN contact marker at the final auto-reset step must not erase
        # the last physical support polygon from the completed plot.
        current_idx = len(self.support_polygon_xy) - 1
        while current_idx >= 0 and (
            math.isnan(self.contact_left[current_idx])
            or math.isnan(self.contact_right[current_idx])
        ):
            current_idx -= 1

        history_paths = []
        history_times = []

        # Keep every support boundary in RAM at 50 Hz, but display an outline
        # approximately every 0.1 s (plus every contact-mode transition).
        # This avoids dense, overlapping contours looking like a filled area.
        display_stride = max(1, round(0.10 / self.policy_dt))
        previous_contact = None

        for k, polygon in enumerate(self.support_polygon_xy):
            if len(polygon) < 3:
                previous_contact = None
                continue

            contact = (self.contact_left[k], self.contact_right[k])
            changed_mode = contact != previous_contact
            previous_contact = contact

            if k == current_idx or not (k % display_stride == 0 or changed_mode):
                continue

            # Closed individual contour. This does not connect one time
            # step to another or draw internal edges when both feet support.
            points = np.asarray(polygon, dtype=float)
            history_paths.append(np.vstack((points, points[0])))
            history_times.append(self.time_s[k])

        self._support_history.set_segments(history_paths)
        self._support_history.set_array(np.asarray(history_times, dtype=float))
        self._support_history.set_clim(
            0.0, max(self.policy_dt, self.time_s[-1])
        )

        current_polygon = (
            self.support_polygon_xy[current_idx] if current_idx >= 0 else ()
        )

        if len(current_polygon) >= 3:
            points = np.asarray(current_polygon, dtype=float)
            closed = np.vstack((points, points[0]))
            self._support_outline.set_data(closed[:, 0], closed[:, 1])
        else:
            self._support_outline.set_data([], [])

        # NaN values retained in the ZMP trajectory break the line at resets.
        self._zmp_trail.set_data(self.zmp_x_w, self.zmp_y_w)

        zx = self.zmp_x_w[current_idx] if current_idx >= 0 else math.nan
        zy = self.zmp_y_w[current_idx] if current_idx >= 0 else math.nan

        if math.isfinite(zx) and math.isfinite(zy):
            self._zmp_current.set_data([zx], [zy])
        else:
            self._zmp_current.set_data([], [])

        # Matplotlib relim() ignores LineCollection: determine limits using
        # ALL historical support vertices as well as the complete ZMP trail.
        coords = [
            point
            for polygon in self.support_polygon_xy
            for point in polygon
        ]
        coords.extend(
            (x, y) for x, y in zip(self.zmp_x_w, self.zmp_y_w, strict=True)
            if math.isfinite(x) and math.isfinite(y)
        )

        ax = self._ax_support

        if coords:
            bounds = np.asarray(coords, dtype=float)
            x_min, y_min = bounds.min(axis=0)
            x_max, y_max = bounds.max(axis=0)

            width = max(x_max - x_min, 0.05)
            height = max(y_max - y_min, 0.05)
            pad = 0.08 * max(width, height)

            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
        else:
            ax.set_xlim(-0.15, 0.15)
            ax.set_ylim(-0.15, 0.15)

    def show(self) -> None:
        """Create and render all four figures after the simulation finishes."""
        if not self.time_s:
            print("[Plotter] No policy samples to plot.")
            return

        plt.ioff()

        if self._fig is None:
            self.start()

        self._create_contact_figure()
        self._create_support_figure()
        self.refresh(force=True)
        self._create_joint_figure()

        plt.ioff()
        plt.show()