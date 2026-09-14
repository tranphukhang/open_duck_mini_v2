# step_timing_adaptation/weight_shift_qp.py

from __future__ import annotations

import math
import numpy as np

if __package__:
    from .whole_body_qp import (
        DoubleSupportHQPConfig,
        DoubleSupportHQPSolution,
        WholeBodyHierarchicalInverseDynamics,
    )
else:
    from whole_body_qp import (
        DoubleSupportHQPConfig,
        DoubleSupportHQPSolution,
        WholeBodyHierarchicalInverseDynamics,
    )


class DoubleSupportWeightShiftController(
    WholeBodyHierarchicalInverseDynamics
):
    """
    Double-support preparation controller used only before the
    first adaptive step.

    It reuses the existing stable double-support HQP and changes
    only Rank 2 by adding a lateral CoM acceleration task:

        Rank 2:
            left foot 6D
            right foot 6D
            CoM-y
            CoM-z

    The original whole_body_qp.py is NOT modified.

    Runtime references and gains remain in run.py.
    """

    def __init__(
        self,
        nv: int,
        nu: int,
        actuated_dof_indices,
        config: DoubleSupportHQPConfig,
    ) -> None:

        super().__init__(
            nv=nv,
            nu=nu,
            actuated_dof_indices=actuated_dof_indices,
            config=config,
        )

        self._desired_com_acceleration_y = 0.0
        self._com_jdot_v_y = 0.0

    def _build_rank2(
        self,
        left_jacobian,
        left_jdot_v,
        right_jacobian,
        right_jdot_v,
        com_jacobian,
        com_jdot_v_z,
        desired_com_acceleration_z,
    ):

        JL = np.asarray(left_jacobian, dtype=float)
        JR = np.asarray(right_jacobian, dtype=float)

        JdotL_v = np.asarray(left_jdot_v, dtype=float)
        JdotR_v = np.asarray(right_jdot_v, dtype=float)

        Jcom = np.asarray(com_jacobian, dtype=float)

        if JL.shape != (6, self.nv):
            raise ValueError("left_jacobian has invalid shape.")

        if JR.shape != (6, self.nv):
            raise ValueError("right_jacobian has invalid shape.")

        if JdotL_v.shape != (6,):
            raise ValueError("left_jdot_v has invalid shape.")

        if JdotR_v.shape != (6,):
            raise ValueError("right_jdot_v has invalid shape.")

        if Jcom.shape != (3, self.nv):
            raise ValueError("com_jacobian has invalid shape.")

        desired_com_acceleration_y = float(
            self._desired_com_acceleration_y
        )

        com_jdot_v_y = float(
            self._com_jdot_v_y
        )

        if not math.isfinite(desired_com_acceleration_y):
            raise ValueError(
                "Non-finite desired CoM-y acceleration."
            )

        if not math.isfinite(com_jdot_v_y):
            raise ValueError(
                "Non-finite CoM Jdot*v in y."
            )

        # 6D left stance + 6D right stance + CoM-y + CoM-z.
        B2 = np.zeros(
            (14, self.nvar),
            dtype=float,
        )

        d2 = np.zeros(
            14,
            dtype=float,
        )

        # Left foot 6D rigid contact.
        B2[0:6, self.qacc_slice] = JL
        d2[0:6] = -JdotL_v

        # Right foot 6D rigid contact.
        B2[6:12, self.qacc_slice] = JR
        d2[6:12] = -JdotR_v

        # CoM-y acceleration.
        B2[12, self.qacc_slice] = Jcom[1, :]
        d2[12] = (
            desired_com_acceleration_y
            -
            com_jdot_v_y
        )

        # CoM-z acceleration.
        B2[13, self.qacc_slice] = Jcom[2, :]
        d2[13] = (
            float(desired_com_acceleration_z)
            -
            float(com_jdot_v_z)
        )

        return B2, d2

    def solve_weight_shift(
        self,
        *,
        com_jdot_v_y,
        desired_com_acceleration_y,
        **double_support_arguments,
    ) -> DoubleSupportHQPSolution:
        """
        Solve one double-support HQP update with an additional
        CoM-y Rank-2 task.
        """

        self._com_jdot_v_y = float(
            com_jdot_v_y
        )

        self._desired_com_acceleration_y = float(
            desired_com_acceleration_y
        )

        return super().solve_double_support(
            **double_support_arguments
        )
