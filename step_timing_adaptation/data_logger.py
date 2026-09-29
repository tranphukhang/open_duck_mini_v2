# step_timing_adaptation/data_logger.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# ============================================================
# JOINT ANGLE LOGGER
# ============================================================

class JointAngleDataLogger:
    """
    Store joint-angle samples during the simulation and save them to CSV.

    Each row contains:
        time_s, joint_1_rad, joint_2_rad, ...

    The logger is intentionally independent of MuJoCo/Pinocchio.
    run.py is responsible for obtaining the joint angles and passing
    them to this class.
    """

    def __init__(
        self,
        *,
        joint_names,
        output_file,
    ) -> None:

        self.joint_names = tuple(
            str(name)
            for name in joint_names
        )

        if len(
            self.joint_names
        ) == 0:

            raise ValueError(
                "joint_names must not be empty."
            )

        if len(
            set(
                self.joint_names
            )
        ) != len(
            self.joint_names
        ):

            raise ValueError(
                "joint_names contains duplicate names."
            )

        self.output_file = Path(
            output_file
        )

        self.time = []

        self.joint_angles = {
            joint_name: []
            for joint_name
            in self.joint_names
        }


    def append(
        self,
        *,
        time,
        joint_angles,
    ) -> None:
        """
        Append one joint-angle sample.

        Parameters
        ----------
        time
            Simulation time [s].

        joint_angles
            Dictionary:
                {
                    "joint_name": angle_in_radians,
                    ...
                }
        """

        sample_time = float(
            time
        )

        if not np.isfinite(
            sample_time
        ):

            raise ValueError(
                "time must be finite."
            )

        if (
            len(
                self.time
            )
            >
            0
            and
            sample_time
            <=
            self.time[-1]
        ):

            raise ValueError(
                "Logged time must be strictly increasing."
            )

        sample = {}

        for joint_name in self.joint_names:

            if joint_name not in joint_angles:

                raise KeyError(
                    f"Missing joint angle for '{joint_name}'."
                )

            angle = float(
                joint_angles[
                    joint_name
                ]
            )

            if not np.isfinite(
                angle
            ):

                raise ValueError(
                    f"Joint angle '{joint_name}' "
                    "contains NaN/Inf."
                )

            sample[
                joint_name
            ] = angle

        self.time.append(
            sample_time
        )

        for joint_name in self.joint_names:

            self.joint_angles[
                joint_name
            ].append(
                sample[
                    joint_name
                ]
            )


    def save_csv(
        self,
    ) -> Path:
        """
        Save all recorded samples to CSV.

        Joint angles are stored in radians.
        """

        if len(
            self.time
        ) == 0:

            raise RuntimeError(
                "No joint-angle samples were recorded."
            )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        header = [
            "time_s",
            *[
                f"{joint_name}_rad"
                for joint_name
                in self.joint_names
            ],
        ]

        with self.output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow(
                header
            )

            for sample_index, sample_time in enumerate(
                self.time
            ):

                row = [
                    sample_time
                ]

                for joint_name in self.joint_names:

                    row.append(
                        self.joint_angles[
                            joint_name
                        ][
                            sample_index
                        ]
                    )

                writer.writerow(
                    row
                )

        return self.output_file.resolve()


    def number_samples(
        self,
    ) -> int:

        return len(
            self.time
        )


# ============================================================
# EXPERIMENT CSV LOGGER
# ============================================================

class ExperimentCSVDataLogger:
    """
    Save the processed experiment results into individual CSV files.

    The following files are generated:

        velocity_x.csv
        velocity_y.csv
        left_swing_height.csv
        right_swing_height.csv
        sagittal_motion.csv
        lateral_motion.csv
    """

    def __init__(
        self,
        *,
        output_dir,
    ) -> None:

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # BASIC CSV WRITER
    # ========================================================

    def _write_columns(
        self,
        *,
        filename,
        columns,
    ) -> Path:
        """
        Write a dictionary of equally-sized 1-D columns to CSV.

        columns must preserve the desired column order.
        """

        if len(
            columns
        ) == 0:

            raise ValueError(
                "columns must not be empty."
            )

        prepared = {}

        number_rows = None

        for column_name, values in columns.items():

            array = np.asarray(
                values
            )

            if array.ndim != 1:

                raise ValueError(
                    f"Column '{column_name}' must be 1-D."
                )

            if number_rows is None:

                number_rows = int(
                    array.size
                )

            elif (
                int(
                    array.size
                )
                !=
                number_rows
            ):

                raise ValueError(
                    "CSV columns have inconsistent lengths."
                )

            prepared[
                str(
                    column_name
                )
            ] = array

        if (
            number_rows
            is None
            or
            number_rows
            <=
            0
        ):

            raise RuntimeError(
                f"No data available for '{filename}'."
            )

        output_file = (
            self.output_dir
            /
            filename
        )

        with output_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow(
                list(
                    prepared.keys()
                )
            )

            for row_index in range(
                number_rows
            ):

                writer.writerow(
                    [
                        prepared[
                            column_name
                        ][
                            row_index
                        ]
                        for column_name
                        in prepared
                    ]
                )

        return output_file.resolve()


    # ========================================================
    # SIMULATION LOG DATA
    # ========================================================

    @staticmethod
    def _simulation_data(
        simulation_log,
    ):

        if simulation_log is None:

            raise RuntimeError(
                "Simulation log is unavailable."
            )

        data = (
            simulation_log.as_numpy()
        )

        if (
            "time"
            not in data
            or
            np.asarray(
                data[
                    "time"
                ]
            ).size
            ==
            0
        ):

            raise RuntimeError(
                "Simulation log is empty."
            )

        return data


    # ========================================================
    # VELOCITY X
    # ========================================================

    def _save_velocity_x(
        self,
        *,
        simulation_log,
    ) -> Path:

        data = (
            self._simulation_data(
                simulation_log
            )
        )

        return self._write_columns(
            filename=(
                "velocity_x.csv"
            ),
            columns={
                "time_s": data[
                    "time"
                ],
                "desired_vx_m_s": data[
                    "command_vx"
                ],
                "actual_com_vx_m_s": data[
                    "whole_body_com_vx"
                ],
            },
        )


    # ========================================================
    # VELOCITY Y
    # ========================================================

    def _save_velocity_y(
        self,
        *,
        simulation_log,
    ) -> Path:

        data = (
            self._simulation_data(
                simulation_log
            )
        )

        return self._write_columns(
            filename=(
                "velocity_y.csv"
            ),
            columns={
                "time_s": data[
                    "time"
                ],
                "desired_vy_m_s": data[
                    "command_vy"
                ],
                "actual_com_vy_m_s": data[
                    "whole_body_com_vy"
                ],
            },
        )


    # ========================================================
    # LEFT SWING HEIGHT
    # ========================================================

    def _save_left_swing_height(
        self,
        *,
        simulation_log,
    ) -> Path:

        data = (
            self._simulation_data(
                simulation_log
            )
        )

        return self._write_columns(
            filename=(
                "left_swing_height.csv"
            ),
            columns={
                "time_s": data[
                    "time"
                ],
                "left_foot_z_m": data[
                    "left_foot_z"
                ],
            },
        )


    # ========================================================
    # RIGHT SWING HEIGHT
    # ========================================================

    def _save_right_swing_height(
        self,
        *,
        simulation_log,
    ) -> Path:

        data = (
            self._simulation_data(
                simulation_log
            )
        )

        return self._write_columns(
            filename=(
                "right_swing_height.csv"
            ),
            columns={
                "time_s": data[
                    "time"
                ],
                "right_foot_z_m": data[
                    "right_foot_z"
                ],
            },
        )


    # ========================================================
    # SAGITTAL MOTION
    # ========================================================

    def _save_sagittal_motion(
        self,
        *,
        simulation_log,
    ) -> Path:

        data = (
            self._simulation_data(
                simulation_log
            )
        )

        return self._write_columns(
            filename=(
                "sagittal_motion.csv"
            ),
            columns={
                "time_s": data[
                    "time"
                ],
                "right_foot_x_m": data[
                    "right_foot_x"
                ],
                "left_foot_x_m": data[
                    "left_foot_x"
                ],
                "com_x_m": data[
                    "com_x"
                ],
                "dcm_x_m": data[
                    "dcm_x"
                ],
            },
        )


    # ========================================================
    # LATERAL MOTION
    # ========================================================

    def _save_lateral_motion(
        self,
        *,
        simulation_log,
    ) -> Path:

        data = (
            self._simulation_data(
                simulation_log
            )
        )

        return self._write_columns(
            filename=(
                "lateral_motion.csv"
            ),
            columns={
                "time_s": data[
                    "time"
                ],
                "right_foot_y_m": data[
                    "right_foot_y"
                ],
                "left_foot_y_m": data[
                    "left_foot_y"
                ],
                "com_y_m": data[
                    "com_y"
                ],
                "dcm_y_m": data[
                    "dcm_y"
                ],
            },
        )


    # ========================================================
    # SAVE ALL REQUESTED CSV FILES
    # ========================================================

    def save_all(
        self,
        *,
        simulation_log,
    ) -> dict[str, Path]:

        saved_files = {}

        # ----------------------------------------------------
        # Desired / actual velocity
        # ----------------------------------------------------

        saved_files[
            "velocity_x"
        ] = (
            self._save_velocity_x(
                simulation_log=(
                    simulation_log
                )
            )
        )

        saved_files[
            "velocity_y"
        ] = (
            self._save_velocity_y(
                simulation_log=(
                    simulation_log
                )
            )
        )

        # ----------------------------------------------------
        # Swing height
        # ----------------------------------------------------

        saved_files[
            "left_swing_height"
        ] = (
            self._save_left_swing_height(
                simulation_log=(
                    simulation_log
                )
            )
        )

        saved_files[
            "right_swing_height"
        ] = (
            self._save_right_swing_height(
                simulation_log=(
                    simulation_log
                )
            )
        )

        # ----------------------------------------------------
        # Sagittal / lateral motion
        # ----------------------------------------------------

        saved_files[
            "sagittal_motion"
        ] = (
            self._save_sagittal_motion(
                simulation_log=(
                    simulation_log
                )
            )
        )

        saved_files[
            "lateral_motion"
        ] = (
            self._save_lateral_motion(
                simulation_log=(
                    simulation_log
                )
            )
        )

        return saved_files
