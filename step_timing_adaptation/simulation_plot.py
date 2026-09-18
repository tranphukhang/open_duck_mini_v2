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

    # Velocity command
    command_vx: list = field(default_factory=list)
    command_vy: list = field(default_factory=list)

    # Foot positions
    left_foot_x: list = field(default_factory=list)
    right_foot_x: list = field(default_factory=list)

    left_foot_z: list = field(default_factory=list)
    right_foot_z: list = field(default_factory=list)

    # Reduced-order model
    com_x: list = field(default_factory=list)
    com_y: list = field(default_factory=list)

    dcm_x: list = field(default_factory=list)
    dcm_y: list = field(default_factory=list)

    # Adaptive step planner
    landing_x: list = field(default_factory=list)
    landing_y: list = field(default_factory=list)

    step_time: list = field(default_factory=list)


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

        self.right_foot_x.append(
            float(right_foot_position[0])
        )

        self.left_foot_z.append(
            float(left_foot_position[2])
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


    def as_numpy(
        self,
    ):

        return {
            name: np.asarray(
                value,
                dtype=float,
            )
            for name, value
            in vars(self).items()
        }


# ============================================================
# PLOT
# ============================================================

def plot_simulation_results(
    simulation_log: SimulationLog,
    *,
    save_directory=None,
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
    #
    # Similar to the figure in the paper:
    #
    #   top:
    #       left/right foot
    #       CoM
    #       DCM
    #
    #   bottom:
    #       left/right swing-foot height
    # ========================================================

    figure_trajectory, axes = (
        plt.subplots(
            2,
            1,
            figsize=(10, 7),
            sharex=True,
        )
    )

    # --------------------------------------------------------
    # Horizontal walking trajectory
    # --------------------------------------------------------

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

    axes[0].set_ylabel(
        "x position (m)"
    )

    axes[0].set_title(
        "Step Timing Adaptation"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend(
        loc="best"
    )

    # --------------------------------------------------------
    # Swing-foot height
    # --------------------------------------------------------

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
    #
    # Command and adaptive timing
    # ========================================================

    figure_command, axes_command = (
        plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=True,
        )
    )

    # --------------------------------------------------------
    # Velocity command
    # --------------------------------------------------------

    axes_command[0].step(
        time,
        data["command_vx"],
        where="post",
        label=r"$v_x^d$",
    )

    axes_command[0].step(
        time,
        data["command_vy"],
        where="post",
        label=r"$v_y^d$",
    )

    axes_command[0].set_ylabel(
        "Velocity (m/s)"
    )

    axes_command[0].set_title(
        "Velocity Command"
    )

    axes_command[0].grid(
        True,
        alpha=0.3,
    )

    axes_command[0].legend(
        loc="best"
    )

    # --------------------------------------------------------
    # Adaptive step time
    # --------------------------------------------------------

    axes_command[1].plot(
        time,
        data["step_time"],
        label="Adapted step time",
    )

    axes_command[1].set_xlabel(
        "Time (s)"
    )

    axes_command[1].set_ylabel(
        "Step time (s)"
    )

    axes_command[1].grid(
        True,
        alpha=0.3,
    )

    axes_command[1].legend(
        loc="best"
    )

    figure_command.tight_layout()

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

        trajectory_path = (
            save_directory
            /
            "step_timing_trajectory.png"
        )

        command_path = (
            save_directory
            /
            "velocity_and_step_time.png"
        )

        figure_trajectory.savefig(
            trajectory_path,
            dpi=200,
            bbox_inches="tight",
        )

        figure_command.savefig(
            command_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved plot: {trajectory_path}"
        )

        print(
            f"Saved plot: {command_path}"
        )

    # ========================================================
    # SHOW
    # ========================================================

    if show:

        plt.show()

    else:

        plt.close(
            figure_trajectory
        )

        plt.close(
            figure_command
        )