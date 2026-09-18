# step_timing_adaptation/simulation_plot.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SIMULATION LOG
# ============================================================

@dataclass
class SimulationLog:

    time: list = field(default_factory=list)

    command_vx: list = field(default_factory=list)
    command_vy: list = field(default_factory=list)

    left_foot_x: list = field(default_factory=list)
    left_foot_y: list = field(default_factory=list)
    left_foot_z: list = field(default_factory=list)

    right_foot_x: list = field(default_factory=list)
    right_foot_y: list = field(default_factory=list)
    right_foot_z: list = field(default_factory=list)

    com_x: list = field(default_factory=list)
    com_y: list = field(default_factory=list)

    dcm_x: list = field(default_factory=list)
    dcm_y: list = field(default_factory=list)

    landing_x: list = field(default_factory=list)
    landing_y: list = field(default_factory=list)

    step_time: list = field(default_factory=list)

    push_active: list = field(default_factory=list)


    def append(
        self,
        *,
        time,
        command_vx,
        command_vy,
        left_foot_position,
        right_foot_position,
        com_position,
        dcm,
        landing_position,
        step_time,
        push_active,
    ):

        self.time.append(
            float(time)
        )

        self.command_vx.append(
            float(command_vx)
        )

        self.command_vy.append(
            float(command_vy)
        )

        self.left_foot_x.append(
            float(left_foot_position[0])
        )

        self.left_foot_y.append(
            float(left_foot_position[1])
        )

        self.left_foot_z.append(
            float(left_foot_position[2])
        )

        self.right_foot_x.append(
            float(right_foot_position[0])
        )

        self.right_foot_y.append(
            float(right_foot_position[1])
        )

        self.right_foot_z.append(
            float(right_foot_position[2])
        )

        self.com_x.append(
            float(com_position[0])
        )

        self.com_y.append(
            float(com_position[1])
        )

        self.dcm_x.append(
            float(dcm[0])
        )

        self.dcm_y.append(
            float(dcm[1])
        )

        self.landing_x.append(
            float(landing_position[0])
        )

        self.landing_y.append(
            float(landing_position[1])
        )

        self.step_time.append(
            float(step_time)
        )

        self.push_active.append(
            1.0
            if bool(push_active)
            else 0.0
        )


    def as_numpy(
        self,
    ):

        return {
            name: np.asarray(
                values,
                dtype=float,
            )
            for name, values
            in vars(self).items()
        }


# ============================================================
# COMMON MARKERS
# ============================================================

def _draw_event_lines(
    axis,
    *,
    command_change_times,
    push_time,
):

    for event_time in command_change_times:

        axis.axvline(
            float(event_time),
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )

    if push_time is not None:

        axis.axvline(
            float(push_time),
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label="Push",
        )


# ============================================================
# PLOT
# ============================================================

def plot_simulation_results(
    simulation_log: SimulationLog,
    *,
    save_directory=None,
    push_time=None,
    command_change_times=(),
    show=True,
):

    data = (
        simulation_log
        .as_numpy()
    )

    if data["time"].size == 0:

        print(
            "No simulation data available for plotting."
        )

        return

    time = (
        data["time"]
    )

    # ========================================================
    # FIGURE 1
    # X trajectory + foot height
    # ========================================================

    figure_trajectory, axes = (
        plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
        )
    )

    axes[0].plot(
        time,
        data["right_foot_x"],
        label="Right foot",
    )

    axes[0].plot(
        time,
        data["left_foot_x"],
        label="Left foot",
    )

    axes[0].plot(
        time,
        data["com_x"],
        "--",
        label="CoM",
    )

    axes[0].plot(
        time,
        data["dcm_x"],
        ":",
        label="DCM",
    )

    _draw_event_lines(
        axes[0],
        command_change_times=(
            command_change_times
        ),
        push_time=(
            push_time
        ),
    )

    axes[0].set_ylabel(
        "x position (m)"
    )

    axes[0].set_title(
        "Step Timing Adaptation - Sagittal Motion"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend(
        loc="best"
    )

    axes[1].plot(
        time,
        data["right_foot_z"],
        label="Right foot",
    )

    axes[1].plot(
        time,
        data["left_foot_z"],
        label="Left foot",
    )

    _draw_event_lines(
        axes[1],
        command_change_times=(
            command_change_times
        ),
        push_time=(
            push_time
        ),
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].set_ylabel(
        "Foot height (m)"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].legend(
        loc="best"
    )

    figure_trajectory.tight_layout()

    # ========================================================
    # FIGURE 2
    # Y trajectory + velocity command
    # ========================================================

    figure_lateral, axes_lateral = (
        plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
        )
    )

    axes_lateral[0].plot(
        time,
        data["right_foot_y"],
        label="Right foot",
    )

    axes_lateral[0].plot(
        time,
        data["left_foot_y"],
        label="Left foot",
    )

    axes_lateral[0].plot(
        time,
        data["com_y"],
        "--",
        label="CoM",
    )

    axes_lateral[0].plot(
        time,
        data["dcm_y"],
        ":",
        label="DCM",
    )

    axes_lateral[0].plot(
        time,
        data["landing_y"],
        "-.",
        label="Adaptive landing y",
    )

    _draw_event_lines(
        axes_lateral[0],
        command_change_times=(
            command_change_times
        ),
        push_time=(
            push_time
        ),
    )

    axes_lateral[0].set_ylabel(
        "y position (m)"
    )

    axes_lateral[0].set_title(
        "Lateral Motion and Velocity Command"
    )

    axes_lateral[0].grid(
        True,
        alpha=0.3,
    )

    axes_lateral[0].legend(
        loc="best"
    )

    axes_lateral[1].step(
        time,
        data["command_vx"],
        where="post",
        label=r"$v_x^d$",
    )

    axes_lateral[1].step(
        time,
        data["command_vy"],
        where="post",
        label=r"$v_y^d$",
    )

    _draw_event_lines(
        axes_lateral[1],
        command_change_times=(
            command_change_times
        ),
        push_time=(
            push_time
        ),
    )

    axes_lateral[1].set_xlabel(
        "Time (s)"
    )

    axes_lateral[1].set_ylabel(
        "Velocity (m/s)"
    )

    axes_lateral[1].grid(
        True,
        alpha=0.3,
    )

    axes_lateral[1].legend(
        loc="best"
    )

    figure_lateral.tight_layout()

    # ========================================================
    # FIGURE 3
    # QP outputs
    # ========================================================

    figure_qp, axes_qp = (
        plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
        )
    )

    axes_qp[0].plot(
        time,
        data["landing_x"],
        label=r"$u_{T,x}$",
    )

    axes_qp[0].plot(
        time,
        data["landing_y"],
        label=r"$u_{T,y}$",
    )

    _draw_event_lines(
        axes_qp[0],
        command_change_times=(
            command_change_times
        ),
        push_time=(
            push_time
        ),
    )

    axes_qp[0].set_ylabel(
        "Landing position (m)"
    )

    axes_qp[0].set_title(
        "Adaptive Step Planner Outputs"
    )

    axes_qp[0].grid(
        True,
        alpha=0.3,
    )

    axes_qp[0].legend(
        loc="best"
    )

    axes_qp[1].plot(
        time,
        data["step_time"],
        label=r"$T^*$",
    )

    _draw_event_lines(
        axes_qp[1],
        command_change_times=(
            command_change_times
        ),
        push_time=(
            push_time
        ),
    )

    axes_qp[1].set_xlabel(
        "Time (s)"
    )

    axes_qp[1].set_ylabel(
        "Step time (s)"
    )

    axes_qp[1].grid(
        True,
        alpha=0.3,
    )

    axes_qp[1].legend(
        loc="best"
    )

    figure_qp.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    if save_directory is not None:

        save_directory = Path(
            save_directory
        )

        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_files = (
            (
                figure_trajectory,
                "01_sagittal_trajectory.png",
            ),
            (
                figure_lateral,
                "02_lateral_motion_command.png",
            ),
            (
                figure_qp,
                "03_adaptive_qp_outputs.png",
            ),
        )

        for figure, filename in output_files:

            output_path = (
                save_directory
                /
                filename
            )

            figure.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Saved plot: {output_path}"
            )

    if show:

        plt.show()

    else:

        plt.close(
            figure_trajectory
        )

        plt.close(
            figure_lateral
        )

        plt.close(
            figure_qp
        )
