# step_timing_adaptation/joint_torque_reconstruction.py

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MATRIX_RCOND = 1.0e-10
TIME_TOLERANCE = 1.0e-9

LEFT_FOOT_GEOM_NAME = "left_foot_bottom_tpu"
RIGHT_FOOT_GEOM_NAME = "right_foot_bottom_tpu"

FLOATING_BASE_JOINT_NAME = "floating_base"

BASE_DOF = 6


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class JointTorqueTrajectory:
    """
    Reconstructed joint torques over the complete recorded trajectory.

    time:
        Shape (N,)

    valid:
        Shape (N,)
        True only when the corresponding contact-force reconstruction
        produced a valid point-contact force distribution.

    joint_names:
        Actuated single-DoF joints, kept in actuator order.

    joint_torques:
        Dictionary:
            joint_name -> np.ndarray shape (N,)

        Invalid samples are stored as NaN.

    joint_torque_limits:
        Dictionary:
            joint_name -> (lower_limit, upper_limit)

        Limits are expressed as generalized joint torque [N m].
        They are obtained from MuJoCo actuator forcerange and gear.

    base_residual_inf:
        ||tau_generalized[0:6]||_inf after subtracting reconstructed
        contact generalized force. For a consistent floating-base
        reconstruction this should remain close to zero.
    """

    time: np.ndarray
    valid: np.ndarray

    joint_names: tuple[str, ...]

    joint_torques: dict[str, np.ndarray]
    joint_torque_limits: dict[str, tuple[float, float]]

    base_residual_inf: np.ndarray


# ============================================================
# RECONSTRUCTOR
# ============================================================

class JointTorqueReconstructor:
    """
    Reconstruct actuator-side joint-space torque demand from the
    remaining equations of rigid-body dynamics after contact forces
    have already been reconstructed.

    The same dynamics convention used by
    contact_force_reconstruction.py is preserved:

        M(q) qdd + qfrc_bias
            =
            qfrc_passive
            + tau
            + J_c(q)^T f_c

    Therefore:

        tau
            =
            M(q) qdd
            + qfrc_bias
            - qfrc_passive
            - J_c(q)^T f_c

    The first six floating-base components of tau must be zero because
    the floating base is unactuated. Those six entries are retained
    only as a consistency residual.

    The remaining single-DoF actuated joint entries are reported as
    reconstructed joint torques.

    IMPORTANT
    ---------
    This is consistent with the dynamics convention already used by
    the contact-force reconstruction. In particular, qfrc_passive is
    subtracted exactly as in that module.

    Dry joint frictionloss is not added as a separate manually modeled
    term here; the purpose is to remain consistent with the existing
    inverse-dynamics formulation used in the project.
    """

    def __init__(
        self,
        *,
        mj_model,
    ) -> None:

        self.mj_model = mj_model

        # ----------------------------------------------------
        # Floating base validation
        # ----------------------------------------------------

        floating_base_joint_id = int(
            mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                FLOATING_BASE_JOINT_NAME,
            )
        )

        if floating_base_joint_id < 0:

            raise RuntimeError(
                f"MuJoCo joint '{FLOATING_BASE_JOINT_NAME}' "
                "was not found."
            )

        floating_base_dof_address = int(
            self.mj_model.jnt_dofadr[
                floating_base_joint_id
            ]
        )

        if floating_base_dof_address != 0:

            raise RuntimeError(
                "Joint torque reconstruction assumes that the "
                "floating-base DoFs occupy qvel indices 0:6."
            )

        # ----------------------------------------------------
        # Foot bodies used for point-force Jacobians
        # ----------------------------------------------------

        self.left_foot_body_id = (
            self._body_id_from_geom(
                LEFT_FOOT_GEOM_NAME
            )
        )

        self.right_foot_body_id = (
            self._body_id_from_geom(
                RIGHT_FOOT_GEOM_NAME
            )
        )

        # ----------------------------------------------------
        # Actuated single-DoF joints + torque limits
        # ----------------------------------------------------

        (
            self.joint_names,
            self.joint_dof_addresses,
            self.joint_torque_limits,
        ) = (
            self._build_actuated_joint_map()
        )


    # ========================================================
    # GEOM -> BODY
    # ========================================================

    def _body_id_from_geom(
        self,
        geom_name: str,
    ) -> int:

        geom_id = int(
            mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
        )

        if geom_id < 0:

            raise RuntimeError(
                f"MuJoCo geom '{geom_name}' was not found."
            )

        return int(
            self.mj_model.geom_bodyid[
                geom_id
            ]
        )


    # ========================================================
    # ACTUATED JOINT MAP / LIMITS
    # ========================================================

    def _build_actuated_joint_map(
        self,
    ):
        """
        Build one joint-space torque channel per actuated single-DoF
        hinge/slide joint.

        For a joint transmission:

            tau_joint = gear * force_actuator

        Therefore an actuator force range:

            [f_min, f_max]

        maps to the joint-space torque interval:

            [min(gear*f_min, gear*f_max),
             max(gear*f_min, gear*f_max)]

        If multiple actuators target the same joint, their attainable
        torque intervals are summed.
        """

        joint_order = []
        actuator_ids_by_joint = {}

        for actuator_id in range(
            self.mj_model.nu
        ):

            joint_id = int(
                self.mj_model.actuator_trnid[
                    actuator_id,
                    0,
                ]
            )

            if joint_id < 0:

                continue

            joint_type = int(
                self.mj_model.jnt_type[
                    joint_id
                ]
            )

            if joint_type not in (
                int(
                    mujoco.mjtJoint.mjJNT_HINGE
                ),
                int(
                    mujoco.mjtJoint.mjJNT_SLIDE
                ),
            ):

                continue

            joint_name = mujoco.mj_id2name(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if joint_name is None:

                continue

            if joint_name not in actuator_ids_by_joint:

                joint_order.append(
                    joint_name
                )

                actuator_ids_by_joint[
                    joint_name
                ] = []

            actuator_ids_by_joint[
                joint_name
            ].append(
                actuator_id
            )

        if len(
            joint_order
        ) == 0:

            raise RuntimeError(
                "No actuated single-DoF joints were found."
            )

        joint_dof_addresses = {}
        torque_limits = {}

        for joint_name in joint_order:

            joint_id = int(
                mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
            )

            dof_address = int(
                self.mj_model.jnt_dofadr[
                    joint_id
                ]
            )

            joint_dof_addresses[
                joint_name
            ] = (
                dof_address
            )

            lower_total = 0.0
            upper_total = 0.0

            finite_limit_available = True

            for actuator_id in (
                actuator_ids_by_joint[
                    joint_name
                ]
            ):

                gear = float(
                    self.mj_model.actuator_gear[
                        actuator_id,
                        0,
                    ]
                )

                force_limited = bool(
                    self.mj_model.actuator_forcelimited[
                        actuator_id
                    ]
                )

                if not np.isfinite(
                    gear
                ):

                    finite_limit_available = False
                    break

                force_lower = float(
                    self.mj_model.actuator_forcerange[
                        actuator_id,
                        0,
                    ]
                )

                force_upper = float(
                    self.mj_model.actuator_forcerange[
                        actuator_id,
                        1,
                    ]
                )

                # MuJoCo normally marks an actuator as force-limited
                # automatically when a finite forcerange is specified.
                #
                # For robustness, also accept an explicitly populated
                # nonzero forcerange even if the model flag is false.
                # This preserves the design limit from the MJCF
                # (current STS3215: -3.23 ... +3.23).
                has_explicit_range = (
                    np.isfinite(
                        force_lower
                    )
                    and
                    np.isfinite(
                        force_upper
                    )
                    and
                    force_upper
                    >
                    force_lower
                    and
                    (
                        abs(
                            force_lower
                        )
                        >
                        0.0
                        or
                        abs(
                            force_upper
                        )
                        >
                        0.0
                    )
                )

                if (
                    not force_limited
                    and
                    not has_explicit_range
                ):

                    finite_limit_available = False
                    break

                torque_0 = (
                    gear
                    *
                    force_lower
                )

                torque_1 = (
                    gear
                    *
                    force_upper
                )

                lower_total += min(
                    torque_0,
                    torque_1,
                )

                upper_total += max(
                    torque_0,
                    torque_1,
                )

            if finite_limit_available:

                torque_limits[
                    joint_name
                ] = (
                    float(
                        lower_total
                    ),
                    float(
                        upper_total
                    ),
                )

            else:

                torque_limits[
                    joint_name
                ] = (
                    -np.inf,
                    +np.inf,
                )

        return (
            tuple(
                joint_order
            ),
            joint_dof_addresses,
            torque_limits,
        )


    # ========================================================
    # FULL MASS MATRIX
    # ========================================================

    def _full_mass_matrix(
        self,
        *,
        mj_data,
    ) -> np.ndarray:

        mass_matrix = np.zeros(
            (
                self.mj_model.nv,
                self.mj_model.nv,
            ),
            dtype=float,
        )

        if hasattr(
            mj_data,
            "qM",
        ):

            mujoco.mj_fullM(
                self.mj_model,
                mass_matrix,
                mj_data.qM,
            )

        else:

            mujoco.mj_fullM(
                self.mj_model,
                mj_data,
                mass_matrix,
            )

        if not np.all(
            np.isfinite(
                mass_matrix
            )
        ):

            raise RuntimeError(
                "MuJoCo mass matrix contains NaN/Inf."
            )

        return mass_matrix


    # ========================================================
    # POINT JACOBIAN
    # ========================================================

    def _point_translation_jacobian(
        self,
        *,
        mj_data,
        position_world,
        body_id,
    ) -> np.ndarray:

        jacobian = np.zeros(
            (
                3,
                self.mj_model.nv,
            ),
            dtype=float,
        )

        mujoco.mj_jac(
            self.mj_model,
            mj_data,
            jacobian,
            None,
            np.asarray(
                position_world,
                dtype=float,
            ),
            int(
                body_id
            ),
        )

        return jacobian


    # ========================================================
    # CONTACT GENERALIZED FORCE
    # ========================================================

    def _contact_generalized_force(
        self,
        *,
        mj_data,
        point_contact_forces,
    ) -> np.ndarray:

        generalized_force = np.zeros(
            self.mj_model.nv,
            dtype=float,
        )

        for point_force in point_contact_forces:

            if point_force.foot_side == "left":

                body_id = (
                    self.left_foot_body_id
                )

            elif point_force.foot_side == "right":

                body_id = (
                    self.right_foot_body_id
                )

            else:

                raise RuntimeError(
                    "Unknown foot side in reconstructed "
                    f"contact force: {point_force.foot_side}"
                )

            force_world = np.asarray(
                point_force.force_world,
                dtype=float,
            ).reshape(
                3
            )

            position_world = np.asarray(
                point_force.position_world,
                dtype=float,
            ).reshape(
                3
            )

            if (
                not np.all(
                    np.isfinite(
                        force_world
                    )
                )
                or
                not np.all(
                    np.isfinite(
                        position_world
                    )
                )
            ):

                raise RuntimeError(
                    "Contact force/position contains NaN/Inf."
                )

            J = (
                self._point_translation_jacobian(
                    mj_data=(
                        mj_data
                    ),
                    position_world=(
                        position_world
                    ),
                    body_id=(
                        body_id
                    ),
                )
            )

            generalized_force += (
                J.T
                @
                force_world
            )

        return generalized_force


    # ========================================================
    # FORCE-RESULT VALIDITY
    # ========================================================

    @staticmethod
    def _force_result_is_valid(
        result,
    ) -> bool:

        if (
            result.number_contacts
            <=
            0
        ):

            return False

        if not result.unilateral_feasible:

            return False

        if (
            result.friction_cone_enabled
            and
            not result.friction_cone_feasible
        ):

            return False

        if (
            len(
                result.point_contact_forces
            )
            !=
            result.number_contacts
        ):

            return False

        return True


    # ========================================================
    # SOLVE COMPLETE TRAJECTORY
    # ========================================================

    def solve_all(
        self,
        *,
        trajectory_samples,
        contact_force_results,
    ) -> JointTorqueTrajectory:
        """
        Compute joint torques for all samples for which contact forces
        are valid.

        trajectory_samples must be the exact samples recorded by the
        same ContactForceReconstructor that produced
        contact_force_results.
        """

        samples = list(
            trajectory_samples
        )

        force_results = list(
            contact_force_results
        )

        if len(
            samples
        ) != len(
            force_results
        ):

            raise ValueError(
                "Trajectory sample count and contact-force result "
                "count do not match."
            )

        if len(
            samples
        ) == 0:

            raise RuntimeError(
                "No trajectory samples are available for "
                "joint-torque reconstruction."
            )

        number_samples = len(
            samples
        )

        time = np.zeros(
            number_samples,
            dtype=float,
        )

        valid = np.zeros(
            number_samples,
            dtype=bool,
        )

        base_residual_inf = np.full(
            number_samples,
            np.nan,
            dtype=float,
        )

        joint_torques = {

            joint_name: np.full(
                number_samples,
                np.nan,
                dtype=float,
            )

            for joint_name
            in self.joint_names
        }

        scratch_data = mujoco.MjData(
            self.mj_model
        )

        for sample_index, (
            sample,
            force_result,
        ) in enumerate(
            zip(
                samples,
                force_results,
            )
        ):

            sample_time = float(
                sample.time
            )

            result_time = float(
                force_result.time
            )

            if not np.isclose(
                sample_time,
                result_time,
                rtol=0.0,
                atol=TIME_TOLERANCE,
            ):

                raise RuntimeError(
                    "Trajectory/contact-force time mismatch at "
                    f"sample {sample_index}: "
                    f"{sample_time} vs {result_time}"
                )

            time[
                sample_index
            ] = sample_time

            if not (
                self._force_result_is_valid(
                    force_result
                )
            ):

                continue

            qacc = np.asarray(
                force_result.qacc,
                dtype=float,
            ).reshape(
                self.mj_model.nv
            )

            if not np.all(
                np.isfinite(
                    qacc
                )
            ):

                continue

            # -----------------------------------------------
            # Restore the exact recorded state.
            # -----------------------------------------------

            scratch_data.qpos[:] = (
                sample.qpos
            )

            scratch_data.qvel[:] = (
                sample.qvel
            )

            scratch_data.time = (
                sample_time
            )

            scratch_data.qfrc_applied[:] = 0.0
            scratch_data.xfrc_applied[:] = 0.0

            mujoco.mj_forward(
                self.mj_model,
                scratch_data,
            )

            # -----------------------------------------------
            # Required generalized force before contacts:
            #
            # r = M qdd + qfrc_bias - qfrc_passive
            # -----------------------------------------------

            mass_matrix = (
                self._full_mass_matrix(
                    mj_data=(
                        scratch_data
                    )
                )
            )

            required_generalized_force = (
                mass_matrix
                @
                qacc
                +
                np.asarray(
                    scratch_data.qfrc_bias,
                    dtype=float,
                )
                -
                np.asarray(
                    scratch_data.qfrc_passive,
                    dtype=float,
                )
            )

            # -----------------------------------------------
            # Contact generalized force:
            #
            # tau_contact = J_c^T f_c
            # -----------------------------------------------

            contact_generalized_force = (
                self._contact_generalized_force(
                    mj_data=(
                        scratch_data
                    ),
                    point_contact_forces=(
                        force_result
                        .point_contact_forces
                    ),
                )
            )

            # -----------------------------------------------
            # Remaining generalized force:
            #
            # tau = r - J_c^T f_c
            # -----------------------------------------------

            generalized_joint_torque = (
                required_generalized_force
                -
                contact_generalized_force
            )

            if not np.all(
                np.isfinite(
                    generalized_joint_torque
                )
            ):

                continue

            base_residual_inf[
                sample_index
            ] = float(
                np.linalg.norm(
                    generalized_joint_torque[
                        0:
                        BASE_DOF
                    ],
                    ord=np.inf,
                )
            )

            for joint_name in (
                self.joint_names
            ):

                dof_address = (
                    self.joint_dof_addresses[
                        joint_name
                    ]
                )

                joint_torques[
                    joint_name
                ][
                    sample_index
                ] = float(
                    generalized_joint_torque[
                        dof_address
                    ]
                )

            valid[
                sample_index
            ] = True

        return JointTorqueTrajectory(
            time=time,
            valid=valid,
            joint_names=(
                self.joint_names
            ),
            joint_torques=(
                joint_torques
            ),
            joint_torque_limits=dict(
                self.joint_torque_limits
            ),
            base_residual_inf=(
                base_residual_inf
            ),
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def print_summary(
        result: JointTorqueTrajectory,
    ) -> None:

        valid_count = int(
            np.count_nonzero(
                result.valid
            )
        )

        invalid_count = int(
            result.valid.size
            -
            valid_count
        )

        finite_base_residual = (
            result.base_residual_inf[
                np.isfinite(
                    result.base_residual_inf
                )
            ]
        )

        print()
        print(
            "================================================"
        )
        print(
            "JOINT TORQUE RECONSTRUCTION"
        )
        print(
            "================================================"
        )

        print(
            f"Recorded samples       : {result.time.size}"
        )

        print(
            f"Torque-valid samples   : {valid_count}"
        )

        print(
            f"Torque-invalid samples : {invalid_count}"
        )

        if finite_base_residual.size > 0:

            print(
                "Base residual "
                "|tau_base|_inf:"
            )

            print(
                f"  p50 : "
                f"{np.percentile(finite_base_residual, 50.0):.6e}"
            )

            print(
                f"  p95 : "
                f"{np.percentile(finite_base_residual, 95.0):.6e}"
            )

            print(
                f"  max : "
                f"{np.max(finite_base_residual):.6e}"
            )

        print()
        print(
            "JOINT TORQUE SUMMARY [N m]"
        )
        print(
            "------------------------------------------------"
        )

        for joint_name in (
            result.joint_names
        ):

            torque = (
                result.joint_torques[
                    joint_name
                ]
            )

            finite_torque = (
                torque[
                    np.isfinite(
                        torque
                    )
                ]
            )

            lower_limit, upper_limit = (
                result.joint_torque_limits[
                    joint_name
                ]
            )

            if finite_torque.size == 0:

                print(
                    f"{joint_name:>18s}"
                    " | no valid samples"
                )

                continue

            abs_torque = np.abs(
                finite_torque
            )

            violations = np.zeros(
                finite_torque.shape,
                dtype=bool,
            )

            if np.isfinite(
                lower_limit
            ):

                violations |= (
                    finite_torque
                    <
                    lower_limit
                )

            if np.isfinite(
                upper_limit
            ):

                violations |= (
                    finite_torque
                    >
                    upper_limit
                )

            if (
                np.isfinite(
                    lower_limit
                )
                and
                np.isfinite(
                    upper_limit
                )
            ):

                magnitude_limit = max(
                    abs(
                        lower_limit
                    ),
                    abs(
                        upper_limit
                    ),
                )

                if magnitude_limit > 0.0:

                    max_utilization = (
                        np.max(
                            abs_torque
                        )
                        /
                        magnitude_limit
                    )

                else:

                    max_utilization = float(
                        "inf"
                    )

            else:

                max_utilization = float(
                    "nan"
                )

            print(
                f"{joint_name:>18s}"
                f" | p95|tau|={np.percentile(abs_torque, 95.0):.4f}"
                f" | max|tau|={np.max(abs_torque):.4f}"
                f" | limit=[{lower_limit:+.3f}, {upper_limit:+.3f}]"
                f" | max util={max_utilization:.3f}"
                f" | violations={np.count_nonzero(violations)}"
            )

        print(
            "================================================"
        )
        print()
