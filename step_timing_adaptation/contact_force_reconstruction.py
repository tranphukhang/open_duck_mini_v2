# step_timing_adaptation/contact_force_reconstruction.py

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MATRIX_RCOND = 1.0e-10
CONTACT_DISTANCE_TOLERANCE = 1.0e-10

FLOOR_GEOM_NAME = "floor"

LEFT_FOOT_GEOM_NAME = "left_foot_bottom_tpu"
RIGHT_FOOT_GEOM_NAME = "right_foot_bottom_tpu"

LEFT_FOOT_SITE_NAME = "left_foot"
RIGHT_FOOT_SITE_NAME = "right_foot"

FLOATING_BASE_JOINT_NAME = "floating_base"

BASE_DOF = 6


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class TrajectorySample:
    """
    State recorded from the kinematically imposed MuJoCo trajectory.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class PointContactForce:
    """
    One reconstructed 3-D point-contact force.

    force_world is the force applied by the ground to the robot,
    expressed in the world frame.
    """

    foot_side: str
    position_world: np.ndarray
    force_world: np.ndarray


@dataclass(frozen=True)
class ContactForceResult:
    """
    Result of the floating-base inverse-dynamics reconstruction
    at one time sample.
    """

    time: float

    number_contacts: int

    dynamics_rank: int
    dynamics_condition_number: float

    qacc: np.ndarray

    # Required generalized wrench corresponding to the first
    # six floating-base equations:
    #
    #     M_b(q) qdd + h_b(q, qdot)
    #
    base_required_generalized_force: np.ndarray

    # Residual of:
    #
    #     J_c,b^T f_c = M_b qdd + h_b
    #
    dynamics_residual_inf: float
    dynamics_residual_l2: float

    point_contact_forces: tuple[PointContactForce, ...]

    left_resultant_force_world: np.ndarray
    left_resultant_moment_world: np.ndarray

    right_resultant_force_world: np.ndarray
    right_resultant_moment_world: np.ndarray


# ============================================================
# RECONSTRUCTOR
# ============================================================

class ContactForceReconstructor:
    """
    Reconstruct point-contact forces from the first six equations of
    the floating-base rigid-body dynamics.

    ------------------------------------------------------------
    MODEL
    ------------------------------------------------------------

        M(q) qdd + h(q, qdot)
            = S^T tau + J_c(q)^T f_c

    The robot has a 6-DoF free floating base. Therefore the first six
    generalized coordinates in velocity space are unactuated:

        [S^T tau]_base = 0

    so:

        M_b(q) qdd + h_b(q, qdot)
            = J_c,b(q)^T f_c

    In the current MuJoCo model, foot-ground contact uses condim = 3.
    Therefore each contact point contributes a 3-D force:

        f_i = [f_x, f_y, f_z]^T

    No direct contact torque is assigned to one point contact.

    ------------------------------------------------------------
    IMPORTANT
    ------------------------------------------------------------

    This module intentionally DOES NOT impose:

        - f_n >= 0
        - friction cone / friction pyramid
        - CoP constraints
        - torque limits
        - contact-stability classification

    It only asks:

        "What point-contact force distribution is required by the
         current q, qdot, qdd trajectory?"

    When there are more force unknowns than the six floating-base
    equations, the solution is not unique. We choose the minimum
    Euclidean-norm solution:

        min ||f_c||_2

        subject to:

            J_c,b^T f_c
                = M_b qdd + h_b

    np.linalg.lstsq() returns this minimum-norm solution for the
    underdetermined system.
    """

    def __init__(
        self,
        *,
        mj_model,
    ) -> None:

        self.mj_model = mj_model

        self.samples: list[TrajectorySample] = []

        # ----------------------------------------------------
        # Required MuJoCo object IDs
        # ----------------------------------------------------

        self.floor_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            FLOOR_GEOM_NAME,
        )

        self.left_foot_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            LEFT_FOOT_GEOM_NAME,
        )

        self.right_foot_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            RIGHT_FOOT_GEOM_NAME,
        )

        self.left_foot_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE,
            LEFT_FOOT_SITE_NAME,
        )

        self.right_foot_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE,
            RIGHT_FOOT_SITE_NAME,
        )

        self.floating_base_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            FLOATING_BASE_JOINT_NAME,
        )

        # ----------------------------------------------------
        # Validate floating-base ordering.
        #
        # MuJoCo free joint has 6 DoF in qvel:
        #   3 translational + 3 rotational.
        #
        # This implementation explicitly uses the first six rows of
        # the dynamics, so the free joint must start at dof address 0.
        # ----------------------------------------------------

        joint_type = int(
            self.mj_model.jnt_type[
                self.floating_base_joint_id
            ]
        )

        if joint_type != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):

            raise RuntimeError(
                f"Joint '{FLOATING_BASE_JOINT_NAME}' "
                "is not a MuJoCo free joint."
            )

        floating_base_dof_address = int(
            self.mj_model.jnt_dofadr[
                self.floating_base_joint_id
            ]
        )

        if floating_base_dof_address != 0:

            raise RuntimeError(
                "The floating-base joint does not start at qvel "
                "index 0. This module assumes that the first six "
                "velocity-space equations are the floating-base "
                "equations."
            )

        if self.mj_model.nv < BASE_DOF:

            raise RuntimeError(
                "MuJoCo model has fewer than 6 generalized DoF."
            )


    # ========================================================
    # MUJOCO OBJECT LOOKUP
    # ========================================================

    def _require_id(
        self,
        object_type,
        name: str,
    ) -> int:

        object_id = int(
            mujoco.mj_name2id(
                self.mj_model,
                object_type,
                name,
            )
        )

        if object_id < 0:

            raise RuntimeError(
                f"MuJoCo object '{name}' was not found."
            )

        return object_id


    # ========================================================
    # RECORD TRAJECTORY
    # ========================================================

    def record_sample(
        self,
        *,
        time,
        mj_data,
    ) -> None:
        """
        Record only qpos and qvel.

        Contact geometry, M(q), h(q,qdot), and Jacobians are rebuilt
        offline using MuJoCo when solve_all() is called.
        """

        sample_time = float(
            time
        )

        if not np.isfinite(
            sample_time
        ):

            raise ValueError(
                "Sample time must be finite."
            )

        if (
            len(
                self.samples
            )
            >
            0
            and
            sample_time
            <=
            self.samples[
                -1
            ].time
        ):

            raise ValueError(
                "Sample time must be strictly increasing."
            )

        qpos = np.asarray(
            mj_data.qpos,
            dtype=float,
        ).copy()

        qvel = np.asarray(
            mj_data.qvel,
            dtype=float,
        ).copy()

        if qpos.shape != (
            self.mj_model.nq,
        ):

            raise RuntimeError(
                "Unexpected MuJoCo qpos shape."
            )

        if qvel.shape != (
            self.mj_model.nv,
        ):

            raise RuntimeError(
                "Unexpected MuJoCo qvel shape."
            )

        if (
            not np.all(
                np.isfinite(
                    qpos
                )
            )
            or
            not np.all(
                np.isfinite(
                    qvel
                )
            )
        ):

            raise RuntimeError(
                "qpos/qvel contains NaN or Inf."
            )

        self.samples.append(
            TrajectorySample(
                time=sample_time,
                qpos=qpos,
                qvel=qvel,
            )
        )


    # ========================================================
    # CONTACT HELPERS
    # ========================================================

    @staticmethod
    def _contact_geom_ids(
        contact,
    ) -> tuple[int, int]:

        if hasattr(
            contact,
            "geom",
        ):

            geom = np.asarray(
                contact.geom,
                dtype=int,
            ).reshape(
                -1
            )

            if geom.size >= 2:

                return (
                    int(
                        geom[0]
                    ),
                    int(
                        geom[1]
                    ),
                )

        return (
            int(
                contact.geom1
            ),
            int(
                contact.geom2
            ),
        )


    def _extract_robot_contacts(
        self,
        *,
        mj_data,
    ):
        """
        Return active foot-ground contacts.

        For each contact we store:
            foot_side,
            contact position in world,
            body id of the foot body.

        The floor is static, so the point Jacobian needed for a force
        acting on the robot is simply the Jacobian of the point
        considered attached to the corresponding foot body.
        """

        contacts = []

        for contact_index in range(
            int(
                mj_data.ncon
            )
        ):

            contact = (
                mj_data.contact[
                    contact_index
                ]
            )

            geom_0, geom_1 = (
                self._contact_geom_ids(
                    contact
                )
            )

            pair = {
                geom_0,
                geom_1,
            }

            if (
                self.floor_geom_id
                not in
                pair
            ):

                continue

            if (
                self.left_foot_geom_id
                in
                pair
            ):

                foot_side = "left"

                foot_geom_id = (
                    self.left_foot_geom_id
                )

            elif (
                self.right_foot_geom_id
                in
                pair
            ):

                foot_side = "right"

                foot_geom_id = (
                    self.right_foot_geom_id
                )

            else:

                continue

            # MuJoCo contact is considered within the active contact
            # margin when:
            #
            #     dist <= includemargin
            #
            contact_distance = float(
                contact.dist
            )

            include_margin = float(
                contact.includemargin
            )

            if (
                contact_distance
                >
                include_margin
                +
                CONTACT_DISTANCE_TOLERANCE
            ):

                continue

            # Current foot-ground model is expected to be condim=3.
            # We do not use the cone here, but requiring at least 3
            # contact dimensions is consistent with a 3-D point force.
            if int(
                contact.dim
            ) < 3:

                continue

            foot_body_id = int(
                self.mj_model.geom_bodyid[
                    foot_geom_id
                ]
            )

            contact_position = np.asarray(
                contact.pos,
                dtype=float,
            ).reshape(
                3
            ).copy()

            contacts.append(
                (
                    foot_side,
                    contact_position,
                    foot_body_id,
                )
            )

        return contacts


    # ========================================================
    # FULL MASS MATRIX
    # ========================================================

    def _full_mass_matrix(
        self,
        *,
        mj_data,
    ) -> np.ndarray:
        """
        Return the dense joint-space inertia matrix M(q).

        MuJoCo changed the mj_fullM API in newer releases:

        Older API:
            mj_fullM(model, dst, data.qM)

        Newer API:
            mj_fullM(model, data, dst)

        New MuJoCo versions also replaced mjData.qM with mjData.M.
        This compatibility block supports both API generations.
        """

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
            # MuJoCo legacy API
            mujoco.mj_fullM(
                self.mj_model,
                mass_matrix,
                mj_data.qM,
            )

        else:
            # MuJoCo current API
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
    # CONTACT JACOBIAN
    # ========================================================

    def _build_contact_jacobian(
        self,
        *,
        mj_data,
        contacts,
    ) -> np.ndarray:
        """
        Stack translational point Jacobians:

            J_c =
                [ J_1 ]
                [ J_2 ]
                [ ... ]

        where:

            J_i in R^(3 x nv)

        and therefore:

            J_c^T f_c

        maps stacked world-frame point forces into generalized force.
        """

        number_contacts = len(
            contacts
        )

        J_c = np.zeros(
            (
                3
                *
                number_contacts,
                self.mj_model.nv,
            ),
            dtype=float,
        )

        for contact_index, (
            _,
            position_world,
            foot_body_id,
        ) in enumerate(
            contacts
        ):

            jacobian_translation = np.zeros(
                (
                    3,
                    self.mj_model.nv,
                ),
                dtype=float,
            )

            mujoco.mj_jac(
                self.mj_model,
                mj_data,
                jacobian_translation,
                None,
                position_world,
                foot_body_id,
            )

            row = (
                3
                *
                contact_index
            )

            J_c[
                row:
                row + 3,
                :,
            ] = (
                jacobian_translation
            )

        return J_c


    # ========================================================
    # MATRIX CONDITION NUMBER
    # ========================================================

    @staticmethod
    def _matrix_condition_number(
        A,
    ) -> float:

        singular_values = np.linalg.svd(
            A,
            compute_uv=False,
        )

        if singular_values.size == 0:

            return float(
                "inf"
            )

        sigma_max = float(
            np.max(
                singular_values
            )
        )

        sigma_min = float(
            np.min(
                singular_values
            )
        )

        if (
            sigma_min
            <=
            MATRIX_RCOND
            *
            max(
                1.0,
                sigma_max,
            )
        ):

            return float(
                "inf"
            )

        return float(
            sigma_max
            /
            sigma_min
        )


    # ========================================================
    # SOLVE ONE SAMPLE
    # ========================================================

    def _solve_one(
        self,
        *,
        sample: TrajectorySample,
        qacc,
        scratch_data,
    ) -> ContactForceResult:

        qacc = np.asarray(
            qacc,
            dtype=float,
        ).reshape(
            self.mj_model.nv
        )

        # ----------------------------------------------------
        # Restore recorded trajectory state.
        # ----------------------------------------------------

        scratch_data.qpos[:] = (
            sample.qpos
        )

        scratch_data.qvel[:] = (
            sample.qvel
        )

        scratch_data.time = float(
            sample.time
        )

        # No explicitly applied generalized/spatial external force is
        # used in the current walking experiment.
        scratch_data.qfrc_applied[:] = 0.0
        scratch_data.xfrc_applied[:] = 0.0

        # Recompute MuJoCo kinematics, inertia quantities, bias forces,
        # passive forces and collision contacts at this recorded state.
        mujoco.mj_forward(
            self.mj_model,
            scratch_data,
        )

        # ----------------------------------------------------
        # Rigid-body dynamics:
        #
        #     M qdd + qfrc_bias
        #       = qfrc_passive + tau_act + J_c^T f_c
        #
        # Therefore, before contact and actuator generalized forces:
        #
        #     r = M qdd + qfrc_bias - qfrc_passive
        #
        # For the free base, tau_act_base = 0.
        # ----------------------------------------------------

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

        base_required = (
            required_generalized_force[
                0:
                BASE_DOF
            ].copy()
        )

        # ----------------------------------------------------
        # Current MuJoCo foot-ground contacts.
        # ----------------------------------------------------

        contacts = (
            self._extract_robot_contacts(
                mj_data=(
                    scratch_data
                )
            )
        )

        number_contacts = len(
            contacts
        )

        zero = np.zeros(
            3,
            dtype=float,
        )

        if (
            number_contacts
            ==
            0
        ):

            return ContactForceResult(
                time=float(
                    sample.time
                ),
                number_contacts=0,
                dynamics_rank=0,
                dynamics_condition_number=float(
                    "inf"
                ),
                qacc=qacc.copy(),
                base_required_generalized_force=(
                    base_required
                ),
                dynamics_residual_inf=float(
                    "nan"
                ),
                dynamics_residual_l2=float(
                    "nan"
                ),
                point_contact_forces=tuple(),
                left_resultant_force_world=(
                    zero.copy()
                ),
                left_resultant_moment_world=(
                    zero.copy()
                ),
                right_resultant_force_world=(
                    zero.copy()
                ),
                right_resultant_moment_world=(
                    zero.copy()
                ),
            )

        # ----------------------------------------------------
        # Build:
        #
        #     J_c,b^T f_c = base_required
        #
        # J_c:
        #     (3*Nc) x nv
        #
        # J_c^T:
        #     nv x (3*Nc)
        #
        # First six rows:
        #     6 x (3*Nc)
        # ----------------------------------------------------

        J_c = (
            self._build_contact_jacobian(
                mj_data=(
                    scratch_data
                ),
                contacts=(
                    contacts
                ),
            )
        )

        A_base = (
            J_c.T[
                0:
                BASE_DOF,
                :
            ]
        )

        dynamics_rank = int(
            np.linalg.matrix_rank(
                A_base,
                tol=(
                    MATRIX_RCOND
                    *
                    max(
                        A_base.shape
                    )
                    *
                    np.linalg.norm(
                        A_base,
                        ord=2,
                    )
                ),
            )
        )

        condition_number = (
            self._matrix_condition_number(
                A_base
            )
        )

        # ----------------------------------------------------
        # Minimum-norm force solution.
        #
        # For the usual single-support case with 3 contact points:
        #
        #     A_base in R^(6 x 9)
        #     f_c    in R^9
        #
        # so the force distribution is underdetermined.
        #
        # np.linalg.lstsq() returns the minimum-norm solution.
        # ----------------------------------------------------

        contact_force_vector, _, _, _ = (
            np.linalg.lstsq(
                A_base,
                base_required,
                rcond=MATRIX_RCOND,
            )
        )

        residual = (
            A_base
            @
            contact_force_vector
            -
            base_required
        )

        residual_inf = float(
            np.max(
                np.abs(
                    residual
                )
            )
        )

        residual_l2 = float(
            np.linalg.norm(
                residual
            )
        )

        # ----------------------------------------------------
        # Resultant force / moment for each foot.
        # Reference point for each resultant moment:
        # MuJoCo foot site position.
        # ----------------------------------------------------

        left_foot_reference = np.asarray(
            scratch_data.site_xpos[
                self.left_foot_site_id
            ],
            dtype=float,
        ).copy()

        right_foot_reference = np.asarray(
            scratch_data.site_xpos[
                self.right_foot_site_id
            ],
            dtype=float,
        ).copy()

        left_force = np.zeros(
            3,
            dtype=float,
        )

        left_moment = np.zeros(
            3,
            dtype=float,
        )

        right_force = np.zeros(
            3,
            dtype=float,
        )

        right_moment = np.zeros(
            3,
            dtype=float,
        )

        point_forces = []

        for contact_index, (
            foot_side,
            position_world,
            _,
        ) in enumerate(
            contacts
        ):

            column = (
                3
                *
                contact_index
            )

            force_world = (
                contact_force_vector[
                    column:
                    column + 3
                ].copy()
            )

            point_forces.append(
                PointContactForce(
                    foot_side=(
                        foot_side
                    ),
                    position_world=(
                        position_world.copy()
                    ),
                    force_world=(
                        force_world.copy()
                    ),
                )
            )

            if foot_side == "left":

                left_force += (
                    force_world
                )

                left_moment += np.cross(
                    (
                        position_world
                        -
                        left_foot_reference
                    ),
                    force_world,
                )

            elif foot_side == "right":

                right_force += (
                    force_world
                )

                right_moment += np.cross(
                    (
                        position_world
                        -
                        right_foot_reference
                    ),
                    force_world,
                )

        return ContactForceResult(
            time=float(
                sample.time
            ),
            number_contacts=int(
                number_contacts
            ),
            dynamics_rank=int(
                dynamics_rank
            ),
            dynamics_condition_number=float(
                condition_number
            ),
            qacc=qacc.copy(),
            base_required_generalized_force=(
                base_required.copy()
            ),
            dynamics_residual_inf=float(
                residual_inf
            ),
            dynamics_residual_l2=float(
                residual_l2
            ),
            point_contact_forces=tuple(
                point_forces
            ),
            left_resultant_force_world=(
                left_force
            ),
            left_resultant_moment_world=(
                left_moment
            ),
            right_resultant_force_world=(
                right_force
            ),
            right_resultant_moment_world=(
                right_moment
            ),
        )


    # ========================================================
    # SOLVE COMPLETE TRAJECTORY
    # ========================================================

    def solve_all(
        self,
    ) -> list[ContactForceResult]:
        """
        Reconstruct qdd offline and solve contact forces at every
        recorded sample.

        qdd is obtained from the complete qvel history:

            qdd = d(qvel)/dt

        np.gradient() uses centered finite differences for interior
        samples and second-order one-sided differences at the two
        boundaries.
        """

        number_samples = len(
            self.samples
        )

        if (
            number_samples
            <
            3
        ):

            raise RuntimeError(
                "At least 3 trajectory samples are required."
            )

        time = np.asarray(
            [
                sample.time
                for sample
                in self.samples
            ],
            dtype=float,
        )

        if not np.all(
            np.diff(
                time
            )
            >
            0.0
        ):

            raise RuntimeError(
                "Recorded sample time is not strictly increasing."
            )

        qvel = np.vstack(
            [
                sample.qvel
                for sample
                in self.samples
            ]
        )

        qacc = np.gradient(
            qvel,
            time,
            axis=0,
            edge_order=2,
        )

        scratch_data = mujoco.MjData(
            self.mj_model
        )

        results = []

        for sample_index, sample in enumerate(
            self.samples
        ):

            result = (
                self._solve_one(
                    sample=(
                        sample
                    ),
                    qacc=(
                        qacc[
                            sample_index
                        ]
                    ),
                    scratch_data=(
                        scratch_data
                    ),
                )
            )

            results.append(
                result
            )

        return results


    # ========================================================
    # SUMMARY HELPERS
    # ========================================================

    @staticmethod
    def _percentiles(
        values,
    ):

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if values.size == 0:

            return (
                float(
                    "nan"
                ),
                float(
                    "nan"
                ),
                float(
                    "nan"
                ),
                float(
                    "nan"
                ),
            )

        return (
            float(
                np.percentile(
                    values,
                    50.0,
                )
            ),
            float(
                np.percentile(
                    values,
                    95.0,
                )
            ),
            float(
                np.percentile(
                    values,
                    99.0,
                )
            ),
            float(
                np.max(
                    values
                )
            ),
        )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    @staticmethod
    def print_summary(
        results,
    ) -> None:

        results = list(
            results
        )

        if len(
            results
        ) == 0:

            print(
                "No reconstructed contact-force results."
            )

            return

        contact_count = Counter(
            result.number_contacts
            for result
            in results
        )

        rank_count = Counter(
            result.dynamics_rank
            for result
            in results
        )

        with_contact = [
            result
            for result
            in results
            if result.number_contacts
            >
            0
        ]

        no_contact_count = (
            len(
                results
            )
            -
            len(
                with_contact
            )
        )

        residual_inf = [
            result.dynamics_residual_inf
            for result
            in with_contact
        ]

        condition_number = [
            result.dynamics_condition_number
            for result
            in with_contact
        ]

        left_force_norm = [
            np.linalg.norm(
                result.left_resultant_force_world
            )
            for result
            in with_contact
        ]

        right_force_norm = [
            np.linalg.norm(
                result.right_resultant_force_world
            )
            for result
            in with_contact
        ]

        (
            residual_p50,
            residual_p95,
            residual_p99,
            residual_max,
        ) = (
            ContactForceReconstructor
            ._percentiles(
                residual_inf
            )
        )

        (
            cond_p50,
            cond_p95,
            cond_p99,
            cond_max,
        ) = (
            ContactForceReconstructor
            ._percentiles(
                condition_number
            )
        )

        (
            left_f_p50,
            left_f_p95,
            left_f_p99,
            left_f_max,
        ) = (
            ContactForceReconstructor
            ._percentiles(
                left_force_norm
            )
        )

        (
            right_f_p50,
            right_f_p95,
            right_f_p99,
            right_f_max,
        ) = (
            ContactForceReconstructor
            ._percentiles(
                right_force_norm
            )
        )

        print()
        print(
            "================================================"
        )
        print(
            "FLOATING-BASE CONTACT FORCE RECONSTRUCTION"
        )
        print(
            "================================================"
        )

        print(
            f"Recorded samples       : {len(results)}"
        )

        print(
            f"Samples with contact   : {len(with_contact)}"
        )

        print(
            f"No-contact samples     : {no_contact_count}"
        )

        print()

        print(
            "CONTACT COUNT HISTOGRAM"
        )
        print(
            "------------------------------------------------"
        )

        for number_contacts in sorted(
            contact_count
        ):

            print(
                f"  {number_contacts}: "
                f"{contact_count[number_contacts]}"
            )

        print()

        print(
            "DYNAMICS MATRIX RANK HISTOGRAM"
        )
        print(
            "------------------------------------------------"
        )

        for rank in sorted(
            rank_count
        ):

            print(
                f"  {rank}: "
                f"{rank_count[rank]}"
            )

        print()

        print(
            "DYNAMICS RESIDUAL |A f - b|_inf"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {residual_p50:.6e}"
        )

        print(
            f"  p95 : {residual_p95:.6e}"
        )

        print(
            f"  p99 : {residual_p99:.6e}"
        )

        print(
            f"  max : {residual_max:.6e}"
        )

        print()

        print(
            "cond(J_c,b^T)"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {cond_p50:.6e}"
        )

        print(
            f"  p95 : {cond_p95:.6e}"
        )

        print(
            f"  p99 : {cond_p99:.6e}"
        )

        print(
            f"  max : {cond_max:.6e}"
        )

        print()

        print(
            "LEFT RESULTANT FORCE NORM [N]"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {left_f_p50:.6f}"
        )

        print(
            f"  p95 : {left_f_p95:.6f}"
        )

        print(
            f"  p99 : {left_f_p99:.6f}"
        )

        print(
            f"  max : {left_f_max:.6f}"
        )

        print()

        print(
            "RIGHT RESULTANT FORCE NORM [N]"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {right_f_p50:.6f}"
        )

        print(
            f"  p95 : {right_f_p95:.6f}"
        )

        print(
            f"  p99 : {right_f_p99:.6f}"
        )

        print(
            f"  max : {right_f_max:.6f}"
        )

        print(
            "================================================"
        )
        print()


# ============================================================
# CONTACT FORCE PLOTTING
# ============================================================

def plot_contact_force_results(
    results,
    *,
    mj_model=None,
    show=False,
):
    """
    Plot reconstructed resultant contact forces.

    The forces are expressed in the MuJoCo world frame.

    Figures:
        1) Fx: left foot, right foot, total
        2) Fy: left foot, right foot, total
        3) Fz: left foot, right foot, total
        4) Resultant force norm: left, right, total
        5) Resultant contact moments about each foot site

    Important:
        The six floating-base equations determine the required
        net contact wrench, but they do not uniquely determine how
        that wrench is distributed among multiple simultaneous
        contact points.

        Therefore, during multi-contact / double-support samples,
        left-vs-right force sharing is the minimum-norm distribution
        selected by np.linalg.lstsq(). The total force/wrench is the
        quantity directly constrained by the six floating-base
        equations.
    """

    import matplotlib.pyplot as plt

    results = list(
        results
    )

    if len(
        results
    ) == 0:

        print(
            "No contact-force reconstruction results "
            "available for plotting."
        )

        return

    time = np.asarray(
        [
            result.time
            for result
            in results
        ],
        dtype=float,
    )

    left_force = np.vstack(
        [
            result.left_resultant_force_world
            for result
            in results
        ]
    )

    right_force = np.vstack(
        [
            result.right_resultant_force_world
            for result
            in results
        ]
    )

    total_force = (
        left_force
        +
        right_force
    )

    left_moment = np.vstack(
        [
            result.left_resultant_moment_world
            for result
            in results
        ]
    )

    right_moment = np.vstack(
        [
            result.right_resultant_moment_world
            for result
            in results
        ]
    )

    left_force_norm = np.linalg.norm(
        left_force,
        axis=1,
    )

    right_force_norm = np.linalg.norm(
        right_force,
        axis=1,
    )

    total_force_norm = np.linalg.norm(
        total_force,
        axis=1,
    )

    # ========================================================
    # BASIC SANITY CHECK
    # ========================================================

    for name, value in (
        (
            "time",
            time,
        ),
        (
            "left_force",
            left_force,
        ),
        (
            "right_force",
            right_force,
        ),
        (
            "total_force",
            total_force,
        ),
        (
            "left_moment",
            left_moment,
        ),
        (
            "right_moment",
            right_moment,
        ),
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise RuntimeError(
                f"Plot data '{name}' contains NaN/Inf."
            )

    # ========================================================
    # OPTIONAL ROBOT WEIGHT
    # ========================================================

    robot_weight = None

    if mj_model is not None:

        robot_mass = float(
            np.sum(
                mj_model.body_mass
            )
        )

        gravity_vector = np.asarray(
            mj_model.opt.gravity,
            dtype=float,
        ).reshape(
            3
        )

        gravity_magnitude = float(
            np.linalg.norm(
                gravity_vector
            )
        )

        if (
            np.isfinite(
                robot_mass
            )
            and
            robot_mass > 0.0
            and
            np.isfinite(
                gravity_magnitude
            )
            and
            gravity_magnitude > 0.0
        ):

            robot_weight = (
                robot_mass
                *
                gravity_magnitude
            )

    # ========================================================
    # FORCE COMPONENT FIGURES
    # ========================================================

    component_names = (
        "Fx",
        "Fy",
        "Fz",
    )

    for component_index, component_name in enumerate(
        component_names
    ):

        figure, axis = plt.subplots(
            figsize=(
                11,
                5,
            )
        )

        axis.plot(
            time,
            left_force[
                :,
                component_index
            ],
            linewidth=1.3,
            label="Left foot",
        )

        axis.plot(
            time,
            right_force[
                :,
                component_index
            ],
            linewidth=1.3,
            label="Right foot",
        )

        axis.plot(
            time,
            total_force[
                :,
                component_index
            ],
            linewidth=1.8,
            label="Total",
        )

        if (
            component_name == "Fz"
            and
            robot_weight is not None
        ):

            axis.axhline(
                y=robot_weight,
                linestyle="--",
                linewidth=1.2,
                label=(
                    f"Robot weight mg = "
                    f"{robot_weight:.3f} N"
                ),
            )

        axis.set_title(
            f"Reconstructed Contact Force - {component_name}"
        )

        axis.set_xlabel(
            "Time (s)"
        )

        axis.set_ylabel(
            f"{component_name} (N)"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.legend(
            loc="best"
        )

        figure.tight_layout()

    # ========================================================
    # FORCE NORM FIGURE
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    axis.plot(
        time,
        left_force_norm,
        linewidth=1.3,
        label="||F_left||",
    )

    axis.plot(
        time,
        right_force_norm,
        linewidth=1.3,
        label="||F_right||",
    )

    axis.plot(
        time,
        total_force_norm,
        linewidth=1.8,
        label="||F_total||",
    )

    axis.set_title(
        "Reconstructed Resultant Contact Force Norm"
    )

    axis.set_xlabel(
        "Time (s)"
    )

    axis.set_ylabel(
        "Force norm (N)"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend(
        loc="best"
    )

    figure.tight_layout()

    # ========================================================
    # RESULTANT MOMENT FIGURES
    # ========================================================

    moment_names = (
        "Mx",
        "My",
        "Mz",
    )

    for component_index, component_name in enumerate(
        moment_names
    ):

        figure, axis = plt.subplots(
            figsize=(
                11,
                5,
            )
        )

        axis.plot(
            time,
            left_moment[
                :,
                component_index
            ],
            linewidth=1.3,
            label="Left foot",
        )

        axis.plot(
            time,
            right_moment[
                :,
                component_index
            ],
            linewidth=1.3,
            label="Right foot",
        )

        axis.set_title(
            f"Resultant Contact Moment about Foot Site - "
            f"{component_name}"
        )

        axis.set_xlabel(
            "Time (s)"
        )

        axis.set_ylabel(
            f"{component_name} (N m)"
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        axis.legend(
            loc="best"
        )

        figure.tight_layout()

    # ========================================================
    # PRINT COMPONENT STATISTICS
    # ========================================================

    print()

    print(
        "================================================"
    )

    print(
        "CONTACT FORCE COMPONENT SUMMARY"
    )

    print(
        "================================================"
    )

    if robot_weight is not None:

        print(
            f"Robot weight mg        : "
            f"{robot_weight:.6f} N"
        )

    for component_index, component_name in enumerate(
        component_names
    ):

        component = (
            total_force[
                :,
                component_index
            ]
        )

        print(
            f"Total {component_name:>2s}"
            f" | p50={np.percentile(component, 50.0):+.6f} N"
            f" | p05={np.percentile(component, 5.0):+.6f} N"
            f" | p95={np.percentile(component, 95.0):+.6f} N"
            f" | min={np.min(component):+.6f} N"
            f" | max={np.max(component):+.6f} N"
        )

    print(
        "================================================"
    )

    print()

    if show:

        plt.show()
