"""Plot Open Duck Mini V2 states sampled at each policy step.

Figure 1: base motion. Figure 2: joint positions (deferred).
Figure 3: foot contacts. Figure 4: support-boundary history and ZMP
trajectory in World XY. Figures 5 and 6: joint walking limit cycles
for the left and right legs (q [rad] versus q_dot [rad/s]). Each support boundary is a separate, closed
convex-hull outline (never joined to the next time step). ALL figures are
created only after the rollout, regardless of MuJoCo GUI settings.
After the rollout, data and a MuJoCo video can be saved under rl/log/.
"""

from __future__ import annotations

import copy
import csv
import math
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

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
    """Record policy-rate data and display six Matplotlib figures."""

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

        # Figures 5 and 6: measured joint velocities in radians/second.
        # Use MJLab joint_vel directly, not a numerical derivative of q.
        self.left_joint_vel: dict[str, list[float]] = {
            name: [] for name in LEFT_JOINT_NAMES
        }
        self.right_joint_vel: dict[str, list[float]] = {
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

        # Figures 2, 5 and 6 are created only in show().
        self._fig_joints = None
        self._fig_cycle_left = None
        self._fig_cycle_right = None

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

        # Evaluation artifacts: CSV and optional video share one run directory.
        self._output_dir: Path | None = None
        self._video: MuJoCoVideoRecorder | None = None
        self._video_stride = 1

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
        joint_vel_left: Sequence[float] | None = None,
        joint_vel_right: Sequence[float] | None = None,
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
        if (joint_vel_left is None) != (joint_vel_right is None):
            raise ValueError("Provide joint_vel_left and joint_vel_right together")
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

        if joint_vel_left is None:
            left_vel = (math.nan,) * 5
            right_vel = (math.nan,) * 5
        else:
            if len(joint_vel_left) != 5 or len(joint_vel_right) != 5:
                raise ValueError("Each leg must provide exactly five joint velocities")
            left_vel = tuple(map(float, joint_vel_left))
            right_vel = tuple(map(float, joint_vel_right))

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
        for name, rate in zip(LEFT_JOINT_NAMES, left_vel, strict=True):
            self.left_joint_vel[name].append(rate)
        for name, rate in zip(RIGHT_JOINT_NAMES, right_vel, strict=True):
            self.right_joint_vel[name].append(rate)
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

    def _create_walking_limit_cycle_figures(self) -> None:
        """Plot five joint phase portraits for each leg, preserving time order.

        Each subplot contains one physical joint: x=q [rad], y=q_dot [rad/s].
        NaNs from episode resets break the curves instead of joining cycles.
        The curve is not forced closed; periodicity is an observed property.
        """
        for side, names, positions, velocities, attr in (
            ("Left", LEFT_JOINT_NAMES, self.left_joint_pos,
             self.left_joint_vel, "_fig_cycle_left"),
            ("Right", RIGHT_JOINT_NAMES, self.right_joint_pos,
             self.right_joint_vel, "_fig_cycle_right"),
        ):
            fig = getattr(self, attr)
            if fig is not None and plt.fignum_exists(fig.number):
                continue

            fig, axes = plt.subplots(
                nrows=2, ncols=3, figsize=(15, 8),
                num=f"{side} leg walking limit cycles",
            )
            setattr(self, attr, fig)

            for ax, joint in zip(axes.flat, names, strict=False):
                ax.plot(
                    positions[joint], velocities[joint],
                    linewidth=1.2,
                )
                ax.set_title(joint)
                ax.set_xlabel("Joint position (rad)")
                ax.set_ylabel("Joint velocity (rad/s)")
                ax.grid(True)

            # Only five joints per leg: hide the spare sixth subplot.
            axes.flat[-1].set_visible(False)
            fig.suptitle(f"{side} leg - walking limit cycles", fontsize=14)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

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

    def get_output_dir(self, output_root: str | Path | None = None) -> Path:
        """Create one unique rl/log/eval_<timestamp> directory per evaluation."""
        if self._output_dir is None:
            root = (
                Path(output_root).expanduser()
                if output_root is not None
                else Path(__file__).resolve().parent / "log"
            )
            root = root.resolve()
            self._output_dir = root / (
                "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            self._output_dir.mkdir(parents=True, exist_ok=False)
        elif output_root is not None:
            expected_root = Path(output_root).expanduser().resolve()
            if self._output_dir.parent != expected_root:
                raise ValueError(
                    "Output root differs from the directory used for this evaluation"
                )
        return self._output_dir

    def save_csv(self, output_root: str | Path | None = None) -> Path:
        """Export all recorded data for MATLAB, without downsampling.

        states.csv: one row per policy sample, with raw joint rad and plot deg.
        support_polygon.csv: one row per hull vertex, no duplicated closing
        vertex. walking_limit_cycle.csv: synchronized joint positions (rad)
        and joint velocities (rad/s) for MATLAB phase portraits.
        A common sample_index (1-based) connects all three tables.
        """
        if not self.time_s:
            raise ValueError("Cannot export CSV: no samples were recorded")

        n = len(self.time_s)
        series = {
            "vx_desired": self.vx_desired,
            "vx_actual": self.vx_actual,
            "vy_desired": self.vy_desired,
            "vy_actual": self.vy_actual,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "wx_rad_s": self.wx_rad_s,
            "wy_rad_s": self.wy_rad_s,
            "wz_rad_s": self.wz_rad_s,
            "contact_left": self.contact_left,
            "contact_right": self.contact_right,
            "zmp_x_w": self.zmp_x_w,
            "zmp_y_w": self.zmp_y_w,
            "support_polygon_xy": self.support_polygon_xy,
            **self.left_joint_pos,
            **self.right_joint_pos,
        }
        # Validate the extra phase-portrait buffers independently.
        for joint in LEFT_JOINT_NAMES:
            series[f"{joint}_velocity"] = self.left_joint_vel[joint]
        for joint in RIGHT_JOINT_NAMES:
            series[f"{joint}_velocity"] = self.right_joint_vel[joint]
        for name, values in series.items():
            if len(values) != n:
                raise ValueError(f"Buffer {name}: {len(values)} samples; expected {n}")

        output_dir = self.get_output_dir(output_root)
        states_path = output_dir / "states.csv"
        polygon_path = output_dir / "support_polygon.csv"
        cycle_path = output_dir / "walking_limit_cycle.csv"
        joints = (*LEFT_JOINT_NAMES, *RIGHT_JOINT_NAMES)

        header = [
            "sample_index", "time_s",
            "vx_desired_m_s", "vx_actual_m_s",
            "vy_desired_m_s", "vy_actual_m_s",
            "roll_deg", "pitch_deg", "yaw_deg",
            "wx_rad_s", "wy_rad_s", "wz_rad_s",
        ]
        header += [f"{name}_rad" for name in joints]
        header += [f"{name}_deg" for name in joints]
        header += [
            "contact_left", "contact_right", "zmp_x_w_m", "zmp_y_w_m",
            "polygon_vertex_count",
        ]

        with (
            states_path.open("w", newline="", encoding="utf-8") as fs,
            polygon_path.open("w", newline="", encoding="utf-8") as fp,
            cycle_path.open("w", newline="", encoding="utf-8") as fc,
        ):
            writer = csv.writer(fs)
            vertices_writer = csv.writer(fp)
            cycle_writer = csv.writer(fc)
            writer.writerow(header)
            vertices_writer.writerow([
                "sample_index", "time_s", "vertex_index", "x_w_m", "y_w_m"
            ])
            cycle_writer.writerow(
                ["sample_index", "time_s"]
                + [f"{name}_rad" for name in joints]
                + [f"{name}_rad_s" for name in joints]
            )

            for k, t in enumerate(self.time_s):
                joint_rad = [
                    *[self.left_joint_pos[name][k] for name in LEFT_JOINT_NAMES],
                    *[self.right_joint_pos[name][k] for name in RIGHT_JOINT_NAMES],
                ]
                joint_vel = [
                    *[self.left_joint_vel[name][k] for name in LEFT_JOINT_NAMES],
                    *[self.right_joint_vel[name][k] for name in RIGHT_JOINT_NAMES],
                ]
                polygon = self.support_polygon_xy[k]
                cycle_writer.writerow([k + 1, t, *joint_rad, *joint_vel])
                writer.writerow([
                    k + 1, t,
                    self.vx_desired[k], self.vx_actual[k],
                    self.vy_desired[k], self.vy_actual[k],
                    self.roll_deg[k], self.pitch_deg[k], self.yaw_deg[k],
                    self.wx_rad_s[k], self.wy_rad_s[k], self.wz_rad_s[k],
                    *joint_rad,
                    *[math.degrees(value) for value in joint_rad],
                    self.contact_left[k], self.contact_right[k],
                    self.zmp_x_w[k], self.zmp_y_w[k], len(polygon),
                ])
                for j, (x, y) in enumerate(polygon, start=1):
                    vertices_writer.writerow([k + 1, t, j, x, y])

        print(f"[CSV] Saved: {states_path}")
        print(f"[CSV] Saved: {polygon_path}")
        print(f"[CSV] Saved: {cycle_path}")
        return output_dir

    def start_video(
        self, env, *, fps: int = 25, width: int = 1280, height: int = 720,
        output_root: str | Path | None = None,
    ) -> None:
        """Start offscreen MP4 recording (works with --gui and --no-gui)."""
        if self._video is not None:
            raise RuntimeError("Video recording has already started")
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video fps, width and height must all be positive")
        # FPS is a divisor of the policy rate, so no samples need interpolation.
        frames_per_policy = 1.0 / (fps * self.policy_dt)
        stride = round(frames_per_policy)
        if stride < 1 or not math.isclose(
            frames_per_policy, stride, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"Video fps must divide the policy rate ({1 / self.policy_dt:g} Hz)"
            )
        if width % 2 or height % 2:
            raise ValueError("Video width and height must be even for H.264")

        output_dir = self.get_output_dir(output_root)
        self._video = MuJoCoVideoRecorder(
            env, output_dir / "simulation.mp4", fps=fps,
            width=width, height=height,
        )
        self._video_stride = stride
        print(f"[VIDEO] Recording at {fps} FPS, {width}x{height}")

    def capture_video(self, env) -> None:
        """Capture the most recent valid policy sample at a fixed sim-time rate."""
        if self._video is None or not self.time_s:
            return
        k = len(self.time_s) - 1
        if k % self._video_stride != 0:
            return
        # On a reset boundary the sampler logs NaNs: do not record reset pose.
        if (
            not math.isfinite(self.contact_left[k])
            or not math.isfinite(self.contact_right[k])
        ):
            return
        self._video.capture(env, self)

    def close_video(self) -> None:
        """Flush MP4 even if the GUI was closed before the episode finished."""
        if self._video is None:
            return
        video = self._video
        self._video = None
        video.close()

    def show(self) -> None:
        """Create and render all six figures after the simulation finishes."""
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
        self._create_walking_limit_cycle_figures()

        plt.ioff()
        plt.show()

class MuJoCoVideoRecorder:
    """Write a simulation-time MP4 stream using MJLab's offscreen renderer.

    This renders the MuJoCo scene (not the GUI desktop/window chrome) from a
    tracking camera, and overlays the actual ZMP and support polygon recorded
    by EvaluationSampler. Only one video frame is held in RAM at a time.
    """

    def __init__(self, env, path: Path, *, fps: int, width: int, height: int):
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Missing video encoder: pip install imageio-ffmpeg"
            ) from exc
        import mujoco
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig

        cfg = copy.copy(env.cfg.viewer)
        cfg.height = height
        cfg.width = width
        cfg.env_idx = 0
        cfg.max_extra_envs = 0
        cfg.origin_type = ViewerConfig.OriginType.ASSET_ROOT
        cfg.entity_name = "robot"
        cfg.distance = 1.1
        cfg.azimuth = 135.0
        cfg.elevation = -22.0

        self.path = Path(path)
        self._mujoco = mujoco
        self._renderer = OffscreenRenderer(
            model=env.sim.mj_model,
            cfg=cfg,
            scene=env.scene,
            sim_model=env.sim.model,
            expanded_fields=env.sim.expanded_fields,
        )
        self._writer = None
        self.frames = 0
        try:
            self._renderer.initialize()
            # Visualization-only flags; simulated forces and model untouched.
            flags = self._renderer._opt.flags
            flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            self._writer = imageio_ffmpeg.write_frames(
                str(self.path), (width, height), fps=fps,
                codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                quality=7, macro_block_size=16,
                output_params=["-movflags", "+faststart"],
            )
            self._writer.send(None)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _segment(mujoco, scene, a, b, color, width):
        if scene.ngeom >= scene.maxgeom:
            return False
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return False
        if np.linalg.norm(b - a) < 1.0e-8:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
            size=np.zeros(3), pos=np.zeros(3), mat=np.eye(3).ravel(),
            rgba=np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom=geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
            width=width, from_=a, to=b,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
        return True

    def _overlay(self, plotter: Plotter) -> None:
        """Draw the current hull and recent measured ZMP above the ground."""
        mujoco = self._mujoco
        scene = self._renderer.renderer.scene
        k = len(plotter.time_s) - 1
        polygon = plotter.support_polygon_xy[k]
        for a, b in zip(polygon, (*polygon[1:], polygon[0])) if len(polygon) >= 3 else ():
            self._segment(
                mujoco, scene, (a[0], a[1], 0.008), (b[0], b[1], 0.008),
                (0.02, 0.85, 0.98, 1.0), 0.003,
            )

        # Recent ZMP trail (last 2 seconds) plus a bright current ZMP marker.
        first = max(0, k - round(2.0 / plotter.policy_dt))
        prev = None
        for j in range(first, k + 1):
            x, y = plotter.zmp_x_w[j], plotter.zmp_y_w[j]
            if not (math.isfinite(x) and math.isfinite(y)):
                prev = None
                continue
            cur = (x, y, 0.010)
            if prev is not None:
                self._segment(
                    mujoco, scene, prev, cur,
                    (0.90, 0.12, 0.18, 0.85), 0.0015,
                )
            prev = cur

        x, y = plotter.zmp_x_w[k], plotter.zmp_y_w[k]
        if math.isfinite(x) and math.isfinite(y) and scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom=geom, type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
                size=np.array([0.008, 0.008, 0.008]),
                pos=np.array([x, y, 0.014]), mat=np.eye(3).ravel(),
                rgba=np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32),
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1

    def capture(self, env, plotter: Plotter) -> None:
        """GPU state -> MuJoCo CPU scene -> RGB frame -> MP4 encoder."""
        if self._writer is None:
            raise RuntimeError("MP4 writer is closed")
        self._renderer.update(env.sim.data)
        self._overlay(plotter)
        frame = self._renderer.render()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Unexpected MuJoCo video frame format: {frame.shape}")
        self._writer.send(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        """Close FFmpeg first (flush final MP4 metadata), then OpenGL context."""
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
        finally:
            self._renderer.close()
        if self.frames:
            print(f"[VIDEO] Saved {self.frames} frames: {self.path}")
        else:
            print(f"[VIDEO] No frames captured: {self.path}")
