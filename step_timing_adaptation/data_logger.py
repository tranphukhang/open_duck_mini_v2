# step_timing_adaptation/data_logger.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


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
