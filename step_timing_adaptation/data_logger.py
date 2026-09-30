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
        wrench_distance.csv
        qacc_norm.csv
        required_wrench_norm.csv
        left_contact_force.csv
        right_contact_force.csv
        joint_torque.csv
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
        wrench_distance_results=None,
    ) -> dict[str, Path]:

        saved_files = {}

        # ----------------------------------------------------
        # Contact-wrench feasibility / force / torque
        # ----------------------------------------------------

        if wrench_distance_results is not None:

            saved_files[
                "wrench_distance"
            ] = (
                self._save_wrench_distance(
                    wrench_distance_results=(
                        wrench_distance_results
                    )
                )
            )

            saved_files[
                "qacc_norm"
            ] = (
                self._save_qacc_norm(
                    wrench_distance_results=(
                        wrench_distance_results
                    )
                )
            )

            saved_files[
                "required_wrench_norm"
            ] = (
                self._save_required_wrench_norm(
                    wrench_distance_results=(
                        wrench_distance_results
                    )
                )
            )

            saved_files[
                "left_contact_force"
            ] = (
                self._save_left_contact_force(
                    wrench_distance_results=(
                        wrench_distance_results
                    )
                )
            )

            saved_files[
                "right_contact_force"
            ] = (
                self._save_right_contact_force(
                    wrench_distance_results=(
                        wrench_distance_results
                    )
                )
            )

            saved_files[
                "joint_torque"
            ] = (
                self._save_joint_torque(
                    wrench_distance_results=(
                        wrench_distance_results
                    )
                )
            )

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

    # ========================================================
    # CONTACT WRENCH DISTANCE
    # ========================================================

    def _save_wrench_distance(
        self,
        *,
        wrench_distance_results,
    ) -> Path:

        if wrench_distance_results is None:

            raise RuntimeError(
                "Wrench-distance results are unavailable."
            )

        results = list(
            wrench_distance_results
        )

        if len(
            results
        ) == 0:

            raise RuntimeError(
                "Wrench-distance result list is empty."
            )

        time = np.asarray(
            [
                result.time
                for result
                in results
            ],
            dtype=float,
        )

        distance = np.asarray(
            [
                result.distance
                for result
                in results
            ],
            dtype=float,
        )

        feasible = np.asarray(
            [
                int(
                    result.feasible
                )
                for result
                in results
            ],
            dtype=int,
        )

        converged = np.asarray(
            [
                int(
                    result.converged
                )
                for result
                in results
            ],
            dtype=int,
        )

        iterations = np.asarray(
            [
                result.iterations
                for result
                in results
            ],
            dtype=int,
        )

        generator_count = np.asarray(
            [
                result.active_generator_count
                for result
                in results
            ],
            dtype=int,
        )

        required = np.vstack(
            [
                result.required_wrench
                for result
                in results
            ]
        )

        closest = np.vstack(
            [
                result.closest_wrench
                for result
                in results
            ]
        )

        residual = np.vstack(
            [
                result.residual_wrench
                for result
                in results
            ]
        )

        return self._write_columns(
            filename=(
                "wrench_distance.csv"
            ),
            columns={
                "time_s":
                    time,

                "distance":
                    distance,

                "feasible":
                    feasible,

                "converged":
                    converged,

                "iterations":
                    iterations,

                "active_generators":
                    generator_count,

                "required_0":
                    required[:, 0],

                "required_1":
                    required[:, 1],

                "required_2":
                    required[:, 2],

                "required_3":
                    required[:, 3],

                "required_4":
                    required[:, 4],

                "required_5":
                    required[:, 5],

                "closest_0":
                    closest[:, 0],

                "closest_1":
                    closest[:, 1],

                "closest_2":
                    closest[:, 2],

                "closest_3":
                    closest[:, 3],

                "closest_4":
                    closest[:, 4],

                "closest_5":
                    closest[:, 5],

                "residual_0":
                    residual[:, 0],

                "residual_1":
                    residual[:, 1],

                "residual_2":
                    residual[:, 2],

                "residual_3":
                    residual[:, 3],

                "residual_4":
                    residual[:, 4],

                "residual_5":
                    residual[:, 5],
            },
        )

    # ========================================================
    # LEFT FOOT TOTAL CONTACT FORCE
    # ========================================================

    def _save_left_contact_force(
        self,
        *,
        wrench_distance_results,
    ) -> Path:

        if wrench_distance_results is None:

            raise RuntimeError(
                "Wrench-distance results are unavailable."
            )

        results = list(
            wrench_distance_results
        )

        if len(
            results
        ) == 0:

            raise RuntimeError(
                "Wrench-distance result list is empty."
            )

        time = np.asarray(
            [
                result.time
                for result
                in results
            ],
            dtype=float,
        )

        force = np.vstack(
            [
                result.left_total_force
                for result
                in results
            ]
        )

        if (
            force.ndim != 2
            or
            force.shape[1] != 3
        ):

            raise RuntimeError(
                "left_total_force must have shape (N, 3)."
            )

        return self._write_columns(
            filename=(
                "left_contact_force.csv"
            ),
            columns={
                "time_s":
                    time,

                "force_x_N":
                    force[:, 0],

                "force_y_N":
                    force[:, 1],

                "force_z_N":
                    force[:, 2],
            },
        )


    # ========================================================
    # RIGHT FOOT TOTAL CONTACT FORCE
    # ========================================================

    def _save_right_contact_force(
        self,
        *,
        wrench_distance_results,
    ) -> Path:

        if wrench_distance_results is None:

            raise RuntimeError(
                "Wrench-distance results are unavailable."
            )

        results = list(
            wrench_distance_results
        )

        if len(
            results
        ) == 0:

            raise RuntimeError(
                "Wrench-distance result list is empty."
            )

        time = np.asarray(
            [
                result.time
                for result
                in results
            ],
            dtype=float,
        )

        force = np.vstack(
            [
                result.right_total_force
                for result
                in results
            ]
        )

        if (
            force.ndim != 2
            or
            force.shape[1] != 3
        ):

            raise RuntimeError(
                "right_total_force must have shape (N, 3)."
            )

        return self._write_columns(
            filename=(
                "right_contact_force.csv"
            ),
            columns={
                "time_s":
                    time,

                "force_x_N":
                    force[:, 0],

                "force_y_N":
                    force[:, 1],

                "force_z_N":
                    force[:, 2],
            },
        )


    # ========================================================
    # LEG JOINT TORQUE
    # ========================================================

    def _save_joint_torque(
        self,
        *,
        wrench_distance_results,
    ) -> Path:
        """
        Save the required generalized torques of the leg joints.

        The torque values are reconstructed in
        contact_wrench_distance.py from the actuated components of

            M qdd + qfrc_bias - qfrc_passive
            - sum_i Jv_i^T f_i.

        For the current model all listed leg joints are 1-DoF
        hinge joints, so the generalized torque unit is N m.
        """

        if wrench_distance_results is None:

            raise RuntimeError(
                "Wrench-distance results are unavailable."
            )

        results = list(
            wrench_distance_results
        )

        if len(
            results
        ) == 0:

            raise RuntimeError(
                "Wrench-distance result list is empty."
            )

        joint_names = tuple(
            str(name)
            for name
            in results[0].joint_torque_names
        )

        if len(
            joint_names
        ) == 0:

            raise RuntimeError(
                "joint_torque_names is empty."
            )

        if len(
            set(
                joint_names
            )
        ) != len(
            joint_names
        ):

            raise RuntimeError(
                "joint_torque_names contains duplicates."
            )

        for result in results:

            current_names = tuple(
                str(name)
                for name
                in result.joint_torque_names
            )

            if current_names != joint_names:

                raise RuntimeError(
                    "Joint-torque name ordering changes "
                    "between samples."
                )

        time = np.asarray(
            [
                result.time
                for result
                in results
            ],
            dtype=float,
        )

        torque = np.vstack(
            [
                np.asarray(
                    result.joint_torques,
                    dtype=float,
                ).reshape(-1)
                for result
                in results
            ]
        )

        expected_shape = (
            len(
                results
            ),
            len(
                joint_names
            ),
        )

        if torque.shape != expected_shape:

            raise RuntimeError(
                "joint_torques has unexpected shape: "
                f"{torque.shape}, expected {expected_shape}."
            )

        if not np.all(
            np.isfinite(
                torque
            )
        ):

            raise RuntimeError(
                "joint_torques contains NaN/Inf."
            )

        columns = {
            "time_s":
                time,
        }

        for joint_index, joint_name in enumerate(
            joint_names
        ):

            columns[
                f"{joint_name}_Nm"
            ] = (
                torque[
                    :,
                    joint_index
                ]
            )

        return self._write_columns(
            filename=(
                "joint_torque.csv"
            ),
            columns=columns,
        )


    # ========================================================
    # GENERALIZED ACCELERATION NORM
    # ========================================================

    def _save_qacc_norm(
        self,
        *,
        wrench_distance_results,
    ) -> Path:

        if wrench_distance_results is None:

            raise RuntimeError(
                "Wrench-distance results are unavailable."
            )

        results = list(
            wrench_distance_results
        )

        if len(
            results
        ) == 0:

            raise RuntimeError(
                "Wrench-distance result list is empty."
            )

        time = np.asarray(
            [
                result.time
                for result
                in results
            ],
            dtype=float,
        )

        qacc_norm = np.asarray(
            [
                np.linalg.norm(
                    result.qacc
                )
                for result
                in results
            ],
            dtype=float,
        )

        stance_side = np.asarray(
            [
                0
                if result.stance_side == "left"
                else 1
                for result
                in results
            ],
            dtype=int,
        )

        stance_switch = np.zeros(
            len(
                results
            ),
            dtype=int,
        )

        for index in range(
            1,
            len(
                results
            ),
        ):

            if (
                results[
                    index
                ].stance_side
                !=
                results[
                    index - 1
                ].stance_side
            ):

                stance_switch[
                    index
                ] = 1

        return self._write_columns(
            filename=(
                "qacc_norm.csv"
            ),
            columns={
                "time_s":
                    time,

                "qacc_norm":
                    qacc_norm,

                "stance_side":
                    stance_side,

                "stance_switch":
                    stance_switch,
            },
        )

    # ========================================================
    # REQUIRED WRENCH NORM
    # ========================================================

    def _save_required_wrench_norm(
        self,
        *,
        wrench_distance_results,
    ) -> Path:

        if wrench_distance_results is None:

            raise RuntimeError(
                "Wrench-distance results are unavailable."
            )

        results = list(
            wrench_distance_results
        )

        if len(
            results
        ) == 0:

            raise RuntimeError(
                "Wrench-distance result list is empty."
            )

        time = np.asarray(
            [
                result.time
                for result
                in results
            ],
            dtype=float,
        )

        required_wrench_norm = np.asarray(
            [
                np.linalg.norm(
                    result.required_wrench
                )
                for result
                in results
            ],
            dtype=float,
        )

        stance_side = np.asarray(
            [
                0
                if result.stance_side == "left"
                else 1
                for result
                in results
            ],
            dtype=int,
        )

        stance_switch = np.zeros(
            len(
                results
            ),
            dtype=int,
        )

        for index in range(
            1,
            len(
                results
            ),
        ):

            if (
                results[
                    index
                ].stance_side
                !=
                results[
                    index - 1
                ].stance_side
            ):

                stance_switch[
                    index
                ] = 1

        return self._write_columns(
            filename=(
                "required_wrench_norm.csv"
            ),
            columns={
                "time_s":
                    time,

                "required_wrench_norm":
                    required_wrench_norm,

                "stance_side":
                    stance_side,

                "stance_switch":
                    stance_switch,
            },
        )
