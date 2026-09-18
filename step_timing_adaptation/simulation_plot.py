# step_timing_adaptation/simulation_plot.py

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SIMULATION LOG
# ============================================================

@dataclass
class SimulationLog:

    time: list = field(default_factory=list)

    # --------------------------------------------------------
    # FOOT POSITION
    # --------------------------------------------------------

    left_foot_x: list = field(default_factory=list)
    left_foot_y: list = field(default_factory=list)
    left_foot_z: list = field(default_factory=list)

    right_foot_x: list = field(default_factory=list)
    right_foot_y: list = field(default_factory=list)
    right_foot_z: list = field(default_factory=list)

    # --------------------------------------------------------
    # LIPM COM
    # --------------------------------------------------------

    com_x: list = field(default_factory=list)
    com_y: list = field(default_factory=list)

    # --------------------------------------------------------
    # DCM
    # --------------------------------------------------------

    dcm_x: list = field(default_factory=list)
    dcm_y: list = field(default_factory=list)


    # ========================================================
    # APPEND
    # ========================================================

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

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        self.time.append(
            float(time)
        )

        # ----------------------------------------------------
        # Left foot
        # ----------------------------------------------------

        self.left_foot_x.append(
            float(
                left_foot_position[0]
            )
        )

        self.left_foot_y.append(
            float(
                left_foot_position[1]
            )
        )

        self.left_foot_z.append(
            float(
                left_foot_position[2]
            )
        )

        # ----------------------------------------------------
        # Right foot
        # ----------------------------------------------------

        self.right_foot_x.append(
            float(
                right_foot_position[0]
            )
        )

        self.right_foot_y.append(
            float(
                right_foot_position[1]
            )
        )

        self.right_foot_z.append(
            float(
                right_foot_position[2]
            )
        )

        # ----------------------------------------------------
        # CoM
        # ----------------------------------------------------

        self.com_x.append(
            float(
                com_position[0]
            )
        )

        self.com_y.append(
            float(
                com_position[1]
            )
        )

        # ----------------------------------------------------
        # DCM
        # ----------------------------------------------------

        self.dcm_x.append(
            float(
                dcm[0]
            )
        )

        self.dcm_y.append(
            float(
                dcm[1]
            )
        )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

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
# PLOT SIMULATION RESULTS
# ============================================================

def plot_simulation_results(
    simulation_log: SimulationLog,
):

    data = (
        simulation_log
        .as_numpy()
    )

    if (
        data["time"].size
        ==
        0
    ):

        print(
            "No simulation data available for plotting."
        )

        return

    time = (
        data["time"]
    )

    # ========================================================
    # CREATE FIGURE
    #
    # 3 plots:
    #
    #   1. X:
    #       Right foot
    #       Left foot
    #       CoM
    #       DCM
    #
    #   2. Y:
    #       Right foot
    #       Left foot
    #       CoM
    #       DCM
    #
    #   3. Z:
    #       Right foot
    #       Left foot
    # ========================================================

    figure, axes = (
        plt.subplots(
            3,
            1,
            figsize=(11, 10),
            sharex=True,
        )
    )


    # ========================================================
    # PLOT 1 — X DIRECTION
    # ========================================================

    axes[0].plot(
        time,
        data["right_foot_x"],
        color="red",
        linewidth=1.5,
        label="Right",
    )

    axes[0].plot(
        time,
        data["left_foot_x"],
        color="blue",
        linewidth=1.5,
        label="Left",
    )

    axes[0].plot(
        time,
        data["com_x"],
        color="magenta",
        linestyle="--",
        linewidth=1.8,
        label="CoM",
    )

    axes[0].plot(
        time,
        data["dcm_x"],
        color="cyan",
        linestyle=":",
        linewidth=2.0,
        label="DCM",
    )

    axes[0].set_ylabel(
        "X (m)"
    )

    axes[0].set_title(
        "Sagittal Motion"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend(
        loc="best",
        ncol=4,
    )


    # ========================================================
    # PLOT 2 — Y DIRECTION
    # ========================================================

    axes[1].plot(
        time,
        data["right_foot_y"],
        color="red",
        linewidth=1.5,
        label="Right",
    )

    axes[1].plot(
        time,
        data["left_foot_y"],
        color="blue",
        linewidth=1.5,
        label="Left",
    )

    axes[1].plot(
        time,
        data["com_y"],
        color="magenta",
        linestyle="--",
        linewidth=1.8,
        label="CoM",
    )

    axes[1].plot(
        time,
        data["dcm_y"],
        color="cyan",
        linestyle=":",
        linewidth=2.0,
        label="DCM",
    )

    axes[1].set_ylabel(
        "Y (m)"
    )

    axes[1].set_title(
        "Lateral Motion"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].legend(
        loc="best",
        ncol=4,
    )


    # ========================================================
    # PLOT 3 — SWING FOOT HEIGHT
    # ========================================================

    axes[2].plot(
        time,
        data["right_foot_z"],
        color="red",
        linewidth=1.5,
        label="Right",
    )

    axes[2].plot(
        time,
        data["left_foot_z"],
        color="blue",
        linewidth=1.5,
        label="Left",
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].set_ylabel(
        "Z (m)"
    )

    axes[2].set_title(
        "Swing Foot Height"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    axes[2].legend(
        loc="best",
    )


    # ========================================================
    # FINAL
    # ========================================================

    figure.tight_layout()

    plt.show()