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

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # VELOCITY COMMAND
    # --------------------------------------------------------

    command_vx: list = field(
        default_factory=list
    )

    command_vy: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # LEFT FOOT
    # --------------------------------------------------------

    left_foot_x: list = field(
        default_factory=list
    )

    left_foot_y: list = field(
        default_factory=list
    )

    left_foot_z: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # RIGHT FOOT
    # --------------------------------------------------------

    right_foot_x: list = field(
        default_factory=list
    )

    right_foot_y: list = field(
        default_factory=list
    )

    right_foot_z: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # LIPM COM
    # --------------------------------------------------------

    com_x: list = field(
        default_factory=list
    )

    com_y: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # DCM
    # --------------------------------------------------------

    dcm_x: list = field(
        default_factory=list
    )

    dcm_y: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # ADAPTIVE LANDING POSITION
    # --------------------------------------------------------

    landing_x: list = field(
        default_factory=list
    )

    landing_y: list = field(
        default_factory=list
    )

    landing_z: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # STEP TIME
    # --------------------------------------------------------

    step_time: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # PUSH STATUS
    # --------------------------------------------------------

    push_active: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # WHOLE-BODY COM FROM MUJOCO
    # --------------------------------------------------------

    whole_body_com_x: list = field(
        default_factory=list
    )

    whole_body_com_y: list = field(
        default_factory=list
    )

    whole_body_com_z: list = field(
        default_factory=list
    )

    whole_body_com_vx: list = field(
        default_factory=list
    )

    whole_body_com_vy: list = field(
        default_factory=list
    )

    whole_body_com_vz: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # WHOLE-BODY CENTROIDAL ANGULAR MOMENTUM
    # --------------------------------------------------------

    angular_momentum_x: list = field(
        default_factory=list
    )

    angular_momentum_y: list = field(
        default_factory=list
    )

    angular_momentum_z: list = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # LEG JOINT ANGLES
    # --------------------------------------------------------

    leg_joint_angles: dict = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # LEG JOINT LIMITS
    # --------------------------------------------------------

    leg_joint_limits: dict = field(
        default_factory=dict
    )


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
        whole_body_com_position,
        whole_body_com_velocity,
        angular_momentum,
        leg_joint_angles,
    ):

        self.time.append(
            float(
                time
            )
        )

        self.command_vx.append(
            float(
                command_vx
            )
        )

        self.command_vy.append(
            float(
                command_vy
            )
        )

        self.left_foot_x.append(
            float(
                left_foot_position[
                    0
                ]
            )
        )

        self.left_foot_y.append(
            float(
                left_foot_position[
                    1
                ]
            )
        )

        self.left_foot_z.append(
            float(
                left_foot_position[
                    2
                ]
            )
        )

        self.right_foot_x.append(
            float(
                right_foot_position[
                    0
                ]
            )
        )

        self.right_foot_y.append(
            float(
                right_foot_position[
                    1
                ]
            )
        )

        self.right_foot_z.append(
            float(
                right_foot_position[
                    2
                ]
            )
        )

        self.com_x.append(
            float(
                com_position[
                    0
                ]
            )
        )

        self.com_y.append(
            float(
                com_position[
                    1
                ]
            )
        )

        self.dcm_x.append(
            float(
                dcm[
                    0
                ]
            )
        )

        self.dcm_y.append(
            float(
                dcm[
                    1
                ]
            )
        )

        self.landing_x.append(
            float(
                landing_position[
                    0
                ]
            )
        )

        self.landing_y.append(
            float(
                landing_position[
                    1
                ]
            )
        )

        self.landing_z.append(
            float(
                landing_position[
                    2
                ]
            )
        )

        self.step_time.append(
            float(
                step_time
            )
        )

        self.push_active.append(
            bool(
                push_active
            )
        )

        self.whole_body_com_x.append(
            float(
                whole_body_com_position[
                    0
                ]
            )
        )

        self.whole_body_com_y.append(
            float(
                whole_body_com_position[
                    1
                ]
            )
        )

        self.whole_body_com_z.append(
            float(
                whole_body_com_position[
                    2
                ]
            )
        )

        self.whole_body_com_vx.append(
            float(
                whole_body_com_velocity[
                    0
                ]
            )
        )

        self.whole_body_com_vy.append(
            float(
                whole_body_com_velocity[
                    1
                ]
            )
        )

        self.whole_body_com_vz.append(
            float(
                whole_body_com_velocity[
                    2
                ]
            )
        )

        self.angular_momentum_x.append(
            float(
                angular_momentum[
                    0
                ]
            )
        )

        self.angular_momentum_y.append(
            float(
                angular_momentum[
                    1
                ]
            )
        )

        self.angular_momentum_z.append(
            float(
                angular_momentum[
                    2
                ]
            )
        )

        for joint_name in (
            self.leg_joint_angles
        ):

            if joint_name not in (
                leg_joint_angles
            ):

                raise KeyError(
                    f"Missing logged joint angle: "
                    f"{joint_name}"
                )

            self.leg_joint_angles[
                joint_name
            ].append(
                float(
                    leg_joint_angles[
                        joint_name
                    ]
                )
            )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    def as_numpy(
        self,
    ):

        data = {}

        for name, values in vars(
            self
        ).items():

            if name == "leg_joint_angles":

                data[
                    name
                ] = {

                    joint_name: np.asarray(
                        joint_values,
                        dtype=float,
                    )

                    for (
                        joint_name,
                        joint_values,
                    )
                    in values.items()
                }

            elif name == "leg_joint_limits":

                data[
                    name
                ] = dict(
                    values
                )

            else:

                data[
                    name
                ] = np.asarray(
                    values
                )

        return data


# ============================================================
# FOOT / COM / DCM PLOT
# ============================================================

def plot_motion_results(
    data,
):

    time = (
        data[
            "time"
        ]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(
            11,
            10,
        ),
        sharex=True,
    )

    # ========================================================
    # SAGITTAL
    # ========================================================

    axes[
        0
    ].plot(
        time,
        data[
            "right_foot_x"
        ],
        linewidth=1.5,
        label="Right",
    )

    axes[
        0
    ].plot(
        time,
        data[
            "left_foot_x"
        ],
        linewidth=1.5,
        label="Left",
    )

    axes[
        0
    ].plot(
        time,
        data[
            "com_x"
        ],
        linestyle="--",
        linewidth=1.8,
        label="CoM",
    )

    axes[
        0
    ].plot(
        time,
        data[
            "dcm_x"
        ],
        linestyle=":",
        linewidth=2.0,
        label="DCM",
    )

    axes[
        0
    ].set_ylabel(
        "X (m)"
    )

    axes[
        0
    ].set_title(
        "Sagittal Motion"
    )

    axes[
        0
    ].grid(
        True,
        alpha=0.3,
    )

    axes[
        0
    ].legend(
        loc="best",
        ncol=4,
    )

    # ========================================================
    # LATERAL
    # ========================================================

    axes[
        1
    ].plot(
        time,
        data[
            "right_foot_y"
        ],
        linewidth=1.5,
        label="Right",
    )

    axes[
        1
    ].plot(
        time,
        data[
            "left_foot_y"
        ],
        linewidth=1.5,
        label="Left",
    )

    axes[
        1
    ].plot(
        time,
        data[
            "com_y"
        ],
        linestyle="--",
        linewidth=1.8,
        label="CoM",
    )

    axes[
        1
    ].plot(
        time,
        data[
            "dcm_y"
        ],
        linestyle=":",
        linewidth=2.0,
        label="DCM",
    )

    axes[
        1
    ].set_ylabel(
        "Y (m)"
    )

    axes[
        1
    ].set_title(
        "Lateral Motion"
    )

    axes[
        1
    ].grid(
        True,
        alpha=0.3,
    )

    axes[
        1
    ].legend(
        loc="best",
        ncol=4,
    )

    # ========================================================
    # FOOT HEIGHT
    # ========================================================

    axes[
        2
    ].plot(
        time,
        data[
            "right_foot_z"
        ],
        linewidth=1.5,
        label="Right",
    )

    axes[
        2
    ].plot(
        time,
        data[
            "left_foot_z"
        ],
        linewidth=1.5,
        label="Left",
    )

    axes[
        2
    ].set_xlabel(
        "Time (s)"
    )

    axes[
        2
    ].set_ylabel(
        "Z (m)"
    )

    axes[
        2
    ].set_title(
        "Swing Foot Height"
    )

    axes[
        2
    ].grid(
        True,
        alpha=0.3,
    )

    axes[
        2
    ].legend(
        loc="best"
    )

    figure.tight_layout()


# ============================================================
# DESIRED / ACTUAL ROBOT VELOCITY
# ============================================================

def plot_velocity_tracking(
    data,
):
    """
    Compare the commanded horizontal walking velocity against the
    actual whole-body CoM velocity computed from MuJoCo.

    Desired:
        command_vx, command_vy

    Actual:
        whole_body_com_vx, whole_body_com_vy
    """

    time = np.asarray(
        data[
            "time"
        ],
        dtype=float,
    )

    desired_vx = np.asarray(
        data[
            "command_vx"
        ],
        dtype=float,
    )

    desired_vy = np.asarray(
        data[
            "command_vy"
        ],
        dtype=float,
    )

    actual_vx = np.asarray(
        data[
            "whole_body_com_vx"
        ],
        dtype=float,
    )

    actual_vy = np.asarray(
        data[
            "whole_body_com_vy"
        ],
        dtype=float,
    )

    if not (
        time.size
        ==
        desired_vx.size
        ==
        desired_vy.size
        ==
        actual_vx.size
        ==
        actual_vy.size
    ):

        raise RuntimeError(
            "Velocity plot data have inconsistent lengths."
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            11,
            7,
        ),
        sharex=True,
    )

    # ========================================================
    # X VELOCITY
    # ========================================================

    axes[
        0
    ].plot(
        time,
        desired_vx,
        linestyle="--",
        linewidth=1.8,
        label="Desired Vx",
    )

    axes[
        0
    ].plot(
        time,
        actual_vx,
        linewidth=1.4,
        label="Actual Vx",
    )

    axes[
        0
    ].set_ylabel(
        "Velocity (m/s)"
    )

    axes[
        0
    ].set_title(
        "Sagittal Velocity Tracking"
    )

    axes[
        0
    ].grid(
        True,
        alpha=0.3,
    )

    axes[
        0
    ].legend(
        loc="best"
    )

    # ========================================================
    # Y VELOCITY
    # ========================================================

    axes[
        1
    ].plot(
        time,
        desired_vy,
        linestyle="--",
        linewidth=1.8,
        label="Desired Vy",
    )

    axes[
        1
    ].plot(
        time,
        actual_vy,
        linewidth=1.4,
        label="Actual Vy",
    )

    axes[
        1
    ].set_xlabel(
        "Time (s)"
    )

    axes[
        1
    ].set_ylabel(
        "Velocity (m/s)"
    )

    axes[
        1
    ].set_title(
        "Lateral Velocity Tracking"
    )

    axes[
        1
    ].grid(
        True,
        alpha=0.3,
    )

    axes[
        1
    ].legend(
        loc="best"
    )

    figure.suptitle(
        "Desired and Actual Robot Velocity",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.96,
        )
    )


# ============================================================
# LEG JOINT ANGLE PLOT
# ============================================================

def plot_leg_joint_angles(
    data,
):

    time = (
        data[
            "time"
        ]
    )

    joint_angles = (
        data[
            "leg_joint_angles"
        ]
    )

    joint_limits = (
        data[
            "leg_joint_limits"
        ]
    )

    joint_pairs = (
        (
            "left_hip_yaw",
            "right_hip_yaw",
            "Hip Yaw",
        ),
        (
            "left_hip_roll",
            "right_hip_roll",
            "Hip Roll",
        ),
        (
            "left_hip_pitch",
            "right_hip_pitch",
            "Hip Pitch",
        ),
        (
            "left_knee",
            "right_knee",
            "Knee",
        ),
        (
            "left_ankle",
            "right_ankle",
            "Ankle",
        ),
    )

    figure, axes = plt.subplots(
        5,
        2,
        figsize=(
            14,
            15,
        ),
        sharex=True,
    )

    for row, (
        left_joint,
        right_joint,
        joint_label,
    ) in enumerate(
        joint_pairs
    ):

        for column, (
            joint_name,
            side_name,
        ) in enumerate(
            (
                (
                    left_joint,
                    "Left",
                ),
                (
                    right_joint,
                    "Right",
                ),
            )
        ):

            axis = (
                axes[
                    row,
                    column,
                ]
            )

            if joint_name not in joint_angles:

                raise KeyError(
                    f"Joint angle data for "
                    f"'{joint_name}' were not found."
                )

            if joint_name not in joint_limits:

                raise KeyError(
                    f"Joint limits for "
                    f"'{joint_name}' were not found."
                )

            angle_deg = np.rad2deg(
                joint_angles[
                    joint_name
                ]
            )

            axis.plot(
                time,
                angle_deg,
                linewidth=1.5,
                label="Joint angle",
            )

            (
                lower_limit_rad,
                upper_limit_rad,
            ) = (
                joint_limits[
                    joint_name
                ]
            )

            if np.isfinite(
                lower_limit_rad
            ):

                axis.axhline(
                    y=np.rad2deg(
                        lower_limit_rad
                    ),
                    linestyle="--",
                    linewidth=1.2,
                    label="Lower limit",
                )

            if np.isfinite(
                upper_limit_rad
            ):

                axis.axhline(
                    y=np.rad2deg(
                        upper_limit_rad
                    ),
                    linestyle="--",
                    linewidth=1.2,
                    label="Upper limit",
                )

            axis.set_title(
                f"{side_name} {joint_label}"
            )

            axis.set_ylabel(
                "Angle (deg)"
            )

            axis.grid(
                True,
                alpha=0.3,
            )

            axis.legend(
                loc="best",
                fontsize=8,
            )

    axes[
        -1,
        0,
    ].set_xlabel(
        "Time (s)"
    )

    axes[
        -1,
        1,
    ].set_xlabel(
        "Time (s)"
    )

    figure.suptitle(
        "Leg Joint Angles and Joint Limits",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.97,
        )
    )


# ============================================================
# MAIN PLOT FUNCTION
# ============================================================

def plot_simulation_results(
    simulation_log: SimulationLog,
):

    data = (
        simulation_log.as_numpy()
    )

    if (
        data[
            "time"
        ].size
        ==
        0
    ):

        print(
            "No simulation data available "
            "for plotting."
        )

        return

    # ========================================================
    # MOTION
    # ========================================================

    plot_motion_results(
        data
    )

    # ========================================================
    # DESIRED / ACTUAL HORIZONTAL VELOCITY
    # ========================================================

    plot_velocity_tracking(
        data
    )

    # ========================================================
    # LEG JOINT ANGLES + POSITION LIMITS
    # ========================================================

    if (
        len(
            data[
                "leg_joint_angles"
            ]
        )
        >
        0
    ):

        plot_leg_joint_angles(
            data
        )

    else:

        print(
            "No leg joint angle data available "
            "for plotting."
        )

    # ========================================================
    # SHOW ALL FIGURES
    # ========================================================

    plt.show()
