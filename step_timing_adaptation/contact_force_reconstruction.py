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

# Numerical tolerances for the unilateral contact-force solve.
#
# The constrained problem is:
#
#     min  1/2 ||f_c||^2
#
#     s.t. A f_c = b
#          Fz_i >= 0
#
# where each point-contact force is stored as:
#
#     [Fx_i, Fy_i, Fz_i].
#
# The friction cone is optional; see ENABLE_FRICTION_CONE below.
UNILATERAL_FORCE_TOLERANCE = 1.0e-10
DYNAMICS_EQUALITY_TOLERANCE = 1.0e-9


# ============================================================
# OPTIONAL FRICTION-CONE CONSTRAINT
# ============================================================
#
# False:
#   Keep the original solver behavior:
#
#       min 1/2 ||f_c||^2
#
#       s.t. A f_c = b
#            Fz_i >= 0
#
# True:
#   After finding a unilateral-feasible solution, additionally solve:
#
#       min 1/2 ||f_c||^2
#
#       s.t. A f_c = b
#            Fz_i >= 0
#            sqrt(Fx_i^2 + Fy_i^2) <= mu_i Fz_i
#
# for every active point contact.
#
# mu_i is read directly from MuJoCo's effective contact friction.
#
# Because the current ground is horizontal, world +z is the contact
# normal and world x/y span the tangential plane.
#
# IMPORTANT:
#   When this flag is False, friction ratios are still computed for
#   diagnostic purposes, but the friction cone is NOT enforced.
# ============================================================

ENABLE_FRICTION_CONE = True

FRICTION_CONE_TOLERANCE = 1.0e-8
FRICTION_TANGENTIAL_EPS = 1.0e-12

FRICTION_SOLVER_MAX_ITERATIONS = 300
FRICTION_SOLVER_FTOL = 1.0e-12


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

    # Effective sliding-friction coefficient used by the optional
    # isotropic Coulomb friction cone.
    friction_coefficient: float

    # rho = sqrt(Fx^2 + Fy^2) / (mu * Fz)
    #
    # rho <= 1:
    #     inside/on the friction cone
    #
    # rho > 1:
    #     outside the friction cone
    friction_ratio: float


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

    # True only when a force distribution exists that satisfies
    # both the six floating-base dynamic equations and:
    #
    #     Fz_i >= 0
    #
    # for every active point contact.
    unilateral_feasible: bool

    # True when the friction-cone constraint was enabled for this
    # reconstruction run.
    friction_cone_enabled: bool

    # When friction_cone_enabled is:
    #
    #   False -> None
    #   True  -> True/False depending on feasibility.
    friction_cone_feasible: bool | None

    # Maximum point-contact friction ratio for the accepted solution.
    #
    # If the friction cone is disabled, this is only a diagnostic
    # value computed from the unilateral minimum-norm solution.
    #
    # NaN is used when no valid force solution exists.
    max_friction_ratio: float

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

    This module imposes the unilateral normal-force constraint:

        Fz_i >= 0

    for every active foot-ground point contact. Because the current
    ground is horizontal in the MuJoCo world frame, world +z is the
    contact-normal direction.

    This module always imposes the unilateral normal-force constraint.

    The Coulomb friction cone can be enabled with:

        ENABLE_FRICTION_CONE = True

    When enabled, each active point contact additionally satisfies:

        sqrt(Fx_i^2 + Fy_i^2) <= mu_i Fz_i

    The module still DOES NOT impose:

        - CoP constraints
        - torque limits

    The reconstructed force distribution is obtained from:

        min  1/2 ||f_c||_2^2

        subject to:

            J_c,b^T f_c
                = M_b qdd + h_b

            Fz_i >= 0

    When the unconstrained minimum-norm solution already satisfies
    Fz_i >= 0, it is kept directly.

    Otherwise, the code solves the same convex problem with an exact
    active-set enumeration over the normal-force bounds Fz_i = 0.
    With the current MuJoCo contact set (typically only a few point
    contacts), this avoids adding an external QP-solver dependency.

    If no force distribution can satisfy both the floating-base
    dynamics and Fz_i >= 0 at one sample, that sample is explicitly
    marked unilateral-infeasible rather than clipping a negative
    normal force after the solve.
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

            # ------------------------------------------------
            # Effective sliding friction coefficient.
            #
            # MuJoCo stores contact friction coefficients in:
            #
            #     contact.friction
            #
            # For condim >= 3, the first two entries correspond
            # to the two tangential directions. The current model
            # uses isotropic sliding friction, but taking the smaller
            # of the two values also gives a conservative isotropic
            # cone if they ever differ.
            # ------------------------------------------------

            contact_friction = np.asarray(
                contact.friction,
                dtype=float,
            ).reshape(
                -1
            )

            if contact_friction.size < 2:

                raise RuntimeError(
                    "MuJoCo contact does not provide two "
                    "tangential friction coefficients."
                )

            friction_coefficient = float(
                min(
                    contact_friction[0],
                    contact_friction[1],
                )
            )

            if (
                not np.isfinite(
                    friction_coefficient
                )
                or
                friction_coefficient
                <
                0.0
            ):

                raise RuntimeError(
                    "Invalid MuJoCo contact friction coefficient: "
                    f"{friction_coefficient}"
                )

            contacts.append(
                (
                    foot_side,
                    contact_position,
                    foot_body_id,
                    friction_coefficient,
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
            _,
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
    # UNILATERAL MINIMUM-NORM FORCE SOLVER
    # ========================================================

    @staticmethod
    def _solve_unilateral_minimum_norm(
        *,
        A,
        b,
        number_contacts,
    ):
        """
        Solve the convex force-distribution problem:

            min  1/2 ||f||^2

            s.t. A f = b
                 Fz_i >= 0

        where:

            f =
                [Fx_1, Fy_1, Fz_1,
                 Fx_2, Fy_2, Fz_2,
                 ...]

        The unconstrained minimum-norm solution is checked first.

        If it violates Fz_i >= 0, the solver enumerates active sets
        of the unilateral constraints. For one active constraint,
        Fz_i is fixed exactly to zero. On each resulting affine face,
        np.linalg.lstsq() gives the minimum-norm force vector.

        Because the current contact set is small, this exact active-set
        enumeration is practical and keeps this module independent of
        external QP packages.

        Returns
        -------
        contact_force_vector : np.ndarray | None
            Minimum-norm unilateral-feasible force vector. None means
            the equality dynamics and Fz >= 0 are jointly infeasible
            within numerical tolerance.

        residual_inf : float
            Infinity norm of A f - b for the returned solution.

        residual_l2 : float
            Euclidean norm of A f - b for the returned solution.
        """

        A = np.asarray(
            A,
            dtype=float,
        )

        b = np.asarray(
            b,
            dtype=float,
        ).reshape(
            A.shape[0]
        )

        number_contacts = int(
            number_contacts
        )

        number_force_variables = (
            3
            *
            number_contacts
        )

        if A.shape[1] != number_force_variables:

            raise ValueError(
                "A has an unexpected number of force columns."
            )

        normal_indices = np.arange(
            2,
            number_force_variables,
            3,
            dtype=int,
        )

        equality_tolerance = (
            DYNAMICS_EQUALITY_TOLERANCE
            *
            max(
                1.0,
                float(
                    np.linalg.norm(
                        b,
                        ord=np.inf,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # First try the ordinary minimum-norm solution.
        # ----------------------------------------------------

        unconstrained_force, _, _, _ = (
            np.linalg.lstsq(
                A,
                b,
                rcond=MATRIX_RCOND,
            )
        )

        unconstrained_residual = (
            A
            @
            unconstrained_force
            -
            b
        )

        unconstrained_residual_inf = float(
            np.linalg.norm(
                unconstrained_residual,
                ord=np.inf,
            )
        )

        if (
            unconstrained_residual_inf
            <=
            equality_tolerance
            and
            np.all(
                unconstrained_force[
                    normal_indices
                ]
                >=
                0.0
            )
        ):

            return (
                unconstrained_force,
                unconstrained_residual_inf,
                float(
                    np.linalg.norm(
                        unconstrained_residual
                    )
                ),
            )

        # ----------------------------------------------------
        # Exact active-set enumeration.
        #
        # If a normal-force inequality is active:
        #
        #     Fz_i = 0.
        #
        # All remaining variables are free. We solve the
        # minimum-norm equality problem on that affine face.
        # ----------------------------------------------------

        best_force = None
        best_norm_squared = float(
            "inf"
        )
        best_residual_inf = float(
            "nan"
        )
        best_residual_l2 = float(
            "nan"
        )

        all_variable_indices = np.arange(
            number_force_variables,
            dtype=int,
        )

        number_active_set_combinations = (
            1
            <<
            number_contacts
        )

        # mask = 0 was already tested by the unconstrained solve.
        for active_mask in range(
            1,
            number_active_set_combinations,
        ):

            active_normal_indices = []

            for contact_index in range(
                number_contacts
            ):

                if (
                    active_mask
                    &
                    (
                        1
                        <<
                        contact_index
                    )
                ):

                    active_normal_indices.append(
                        int(
                            normal_indices[
                                contact_index
                            ]
                        )
                    )

            active_normal_indices = np.asarray(
                active_normal_indices,
                dtype=int,
            )

            free_variable_mask = np.ones(
                number_force_variables,
                dtype=bool,
            )

            free_variable_mask[
                active_normal_indices
            ] = False

            free_variable_indices = (
                all_variable_indices[
                    free_variable_mask
                ]
            )

            if free_variable_indices.size == 0:

                continue

            A_free = (
                A[
                    :,
                    free_variable_indices
                ]
            )

            free_force, _, _, _ = (
                np.linalg.lstsq(
                    A_free,
                    b,
                    rcond=MATRIX_RCOND,
                )
            )

            candidate_force = np.zeros(
                number_force_variables,
                dtype=float,
            )

            candidate_force[
                free_variable_indices
            ] = (
                free_force
            )

            # ------------------------------------------------
            # Unilateral feasibility.
            # ------------------------------------------------

            candidate_normal_force = (
                candidate_force[
                    normal_indices
                ]
            )

            if np.any(
                candidate_normal_force
                <
                -UNILATERAL_FORCE_TOLERANCE
            ):

                continue

            # Remove tiny negative round-off values only.
            tiny_negative_normal = (
                (
                    candidate_normal_force
                    <
                    0.0
                )
                &
                (
                    candidate_normal_force
                    >=
                    -UNILATERAL_FORCE_TOLERANCE
                )
            )

            if np.any(
                tiny_negative_normal
            ):

                candidate_force[
                    normal_indices[
                        tiny_negative_normal
                    ]
                ] = 0.0

            candidate_residual = (
                A
                @
                candidate_force
                -
                b
            )

            candidate_residual_inf = float(
                np.linalg.norm(
                    candidate_residual,
                    ord=np.inf,
                )
            )

            if (
                candidate_residual_inf
                >
                equality_tolerance
            ):

                continue

            candidate_norm_squared = float(
                candidate_force
                @
                candidate_force
            )

            if (
                candidate_norm_squared
                <
                best_norm_squared
            ):

                best_force = (
                    candidate_force.copy()
                )

                best_norm_squared = (
                    candidate_norm_squared
                )

                best_residual_inf = (
                    candidate_residual_inf
                )

                best_residual_l2 = float(
                    np.linalg.norm(
                        candidate_residual
                    )
                )

        return (
            best_force,
            best_residual_inf,
            best_residual_l2,
        )


    # ========================================================
    # FRICTION RATIO
    # ========================================================

    @staticmethod
    def _friction_ratios(
        *,
        contact_force_vector,
        friction_coefficients,
    ) -> np.ndarray:
        """
        Compute one Coulomb friction ratio for each point contact:

            rho_i =
                sqrt(Fx_i^2 + Fy_i^2)
                ---------------------
                    mu_i * Fz_i

        Interpretation:

            rho_i < 1  -> inside the cone
            rho_i = 1  -> on the cone boundary
            rho_i > 1  -> outside the cone

        If both tangential force and mu*Fz are effectively zero,
        rho is defined as zero.

        If mu*Fz is zero but tangential force is nonzero, rho is +inf.
        """

        force = np.asarray(
            contact_force_vector,
            dtype=float,
        ).reshape(
            -1
        )

        mu = np.asarray(
            friction_coefficients,
            dtype=float,
        ).reshape(
            -1
        )

        number_contacts = (
            mu.size
        )

        if force.size != 3 * number_contacts:

            raise ValueError(
                "Unexpected force-vector size in friction-ratio "
                "calculation."
            )

        ratios = np.zeros(
            number_contacts,
            dtype=float,
        )

        for contact_index in range(
            number_contacts
        ):

            column = (
                3
                *
                contact_index
            )

            fx = float(
                force[
                    column
                ]
            )

            fy = float(
                force[
                    column + 1
                ]
            )

            fz = float(
                force[
                    column + 2
                ]
            )

            tangential = float(
                np.hypot(
                    fx,
                    fy,
                )
            )

            capacity = float(
                mu[
                    contact_index
                ]
                *
                max(
                    0.0,
                    fz,
                )
            )

            if (
                capacity
                <=
                FRICTION_TANGENTIAL_EPS
            ):

                if (
                    tangential
                    <=
                    FRICTION_TANGENTIAL_EPS
                ):

                    ratios[
                        contact_index
                    ] = 0.0

                else:

                    ratios[
                        contact_index
                    ] = float(
                        "inf"
                    )

            else:

                ratios[
                    contact_index
                ] = (
                    tangential
                    /
                    capacity
                )

        return ratios


    # ========================================================
    # FRICTION-CONE MINIMUM-NORM FORCE SOLVER
    # ========================================================

    @staticmethod
    def _solve_friction_cone_minimum_norm(
        *,
        A,
        b,
        number_contacts,
        friction_coefficients,
        initial_force,
    ):
        """
        Solve:

            min  1/2 ||f||^2

            s.t. A f = b
                 Fz_i >= 0
                 sqrt(Fx_i^2 + Fy_i^2) <= mu_i Fz_i

        for every active point contact.

        The problem is a convex second-order-cone force-distribution
        problem. Here it is solved numerically with SciPy SLSQP to
        avoid introducing an additional dedicated SOCP package.

        The already computed unilateral minimum-norm force is used as
        the initial guess. If that force already satisfies the cone,
        it is returned directly and SLSQP is skipped.

        Returns
        -------
        contact_force_vector : np.ndarray | None

        residual_inf : float

        residual_l2 : float
        """

        try:

            from scipy.optimize import (
                minimize,
            )

        except ImportError as error:

            raise RuntimeError(
                "ENABLE_FRICTION_CONE=True requires SciPy. "
                "Install it with: pip install scipy"
            ) from error

        A = np.asarray(
            A,
            dtype=float,
        )

        b = np.asarray(
            b,
            dtype=float,
        ).reshape(
            A.shape[0]
        )

        number_contacts = int(
            number_contacts
        )

        mu = np.asarray(
            friction_coefficients,
            dtype=float,
        ).reshape(
            number_contacts
        )

        number_force_variables = (
            3
            *
            number_contacts
        )

        if A.shape[1] != number_force_variables:

            raise ValueError(
                "A has an unexpected number of force columns."
            )

        if (
            np.any(
                ~np.isfinite(
                    mu
                )
            )
            or
            np.any(
                mu
                <
                0.0
            )
        ):

            raise ValueError(
                "Friction coefficients must be finite and nonnegative."
            )

        initial_force = np.asarray(
            initial_force,
            dtype=float,
        ).reshape(
            number_force_variables
        )

        normal_indices = np.arange(
            2,
            number_force_variables,
            3,
            dtype=int,
        )

        equality_tolerance = (
            DYNAMICS_EQUALITY_TOLERANCE
            *
            max(
                1.0,
                float(
                    np.linalg.norm(
                        b,
                        ord=np.inf,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # Helper: equality residual.
        # ----------------------------------------------------

        def equality_function(
            force,
        ):

            return (
                A
                @
                force
                -
                b
            )


        def equality_jacobian(
            force,
        ):

            _ = force

            return A


        # ----------------------------------------------------
        # Helper: exact circular Coulomb friction cone.
        #
        # For each contact:
        #
        #     g_i(f) =
        #         mu_i Fz_i
        #         -
        #         sqrt(Fx_i^2 + Fy_i^2)
        #
        # Feasible:
        #
        #     g_i(f) >= 0
        # ----------------------------------------------------

        def friction_function(
            force,
        ):

            values = np.zeros(
                number_contacts,
                dtype=float,
            )

            for contact_index in range(
                number_contacts
            ):

                column = (
                    3
                    *
                    contact_index
                )

                fx = float(
                    force[
                        column
                    ]
                )

                fy = float(
                    force[
                        column + 1
                    ]
                )

                fz = float(
                    force[
                        column + 2
                    ]
                )

                values[
                    contact_index
                ] = (
                    mu[
                        contact_index
                    ]
                    *
                    fz
                    -
                    np.hypot(
                        fx,
                        fy,
                    )
                )

            return values


        def friction_jacobian(
            force,
        ):

            jacobian = np.zeros(
                (
                    number_contacts,
                    number_force_variables,
                ),
                dtype=float,
            )

            for contact_index in range(
                number_contacts
            ):

                column = (
                    3
                    *
                    contact_index
                )

                fx = float(
                    force[
                        column
                    ]
                )

                fy = float(
                    force[
                        column + 1
                    ]
                )

                tangential = float(
                    np.hypot(
                        fx,
                        fy,
                    )
                )

                if (
                    tangential
                    >
                    FRICTION_TANGENTIAL_EPS
                ):

                    jacobian[
                        contact_index,
                        column,
                    ] = (
                        -fx
                        /
                        tangential
                    )

                    jacobian[
                        contact_index,
                        column + 1,
                    ] = (
                        -fy
                        /
                        tangential
                    )

                else:

                    # At zero tangential force the norm gradient is
                    # not unique. Zero is a valid subgradient and is
                    # sufficient for this numerical solve.
                    jacobian[
                        contact_index,
                        column,
                    ] = 0.0

                    jacobian[
                        contact_index,
                        column + 1,
                    ] = 0.0

                jacobian[
                    contact_index,
                    column + 2,
                ] = (
                    mu[
                        contact_index
                    ]
                )

            return jacobian


        # ----------------------------------------------------
        # First test the unilateral minimum-norm force.
        # ----------------------------------------------------

        initial_residual = (
            equality_function(
                initial_force
            )
        )

        initial_residual_inf = float(
            np.linalg.norm(
                initial_residual,
                ord=np.inf,
            )
        )

        initial_friction_margin = (
            friction_function(
                initial_force
            )
        )

        if (
            initial_residual_inf
            <=
            equality_tolerance
            and
            np.all(
                initial_force[
                    normal_indices
                ]
                >=
                -UNILATERAL_FORCE_TOLERANCE
            )
            and
            np.all(
                initial_friction_margin
                >=
                -FRICTION_CONE_TOLERANCE
            )
        ):

            return (
                initial_force.copy(),
                initial_residual_inf,
                float(
                    np.linalg.norm(
                        initial_residual
                    )
                ),
            )

        # ----------------------------------------------------
        # Objective and gradient.
        # ----------------------------------------------------

        def objective(
            force,
        ):

            return float(
                0.5
                *
                force
                @
                force
            )


        def objective_jacobian(
            force,
        ):

            return np.asarray(
                force,
                dtype=float,
            )


        # ----------------------------------------------------
        # Bounds:
        #
        #     Fx, Fy -> unbounded
        #     Fz     -> [0, +inf)
        # ----------------------------------------------------

        bounds = []

        for variable_index in range(
            number_force_variables
        ):

            if (
                variable_index
                %
                3
                ==
                2
            ):

                bounds.append(
                    (
                        0.0,
                        None,
                    )
                )

            else:

                bounds.append(
                    (
                        None,
                        None,
                    )
                )

        constraints = [
            {
                "type":
                    "eq",

                "fun":
                    equality_function,

                "jac":
                    equality_jacobian,
            },
            {
                "type":
                    "ineq",

                "fun":
                    friction_function,

                "jac":
                    friction_jacobian,
            },
        ]

        optimization_result = minimize(
            fun=(
                objective
            ),
            x0=(
                initial_force
            ),
            jac=(
                objective_jacobian
            ),
            bounds=(
                bounds
            ),
            constraints=(
                constraints
            ),
            method=(
                "SLSQP"
            ),
            options={
                "maxiter":
                    int(
                        FRICTION_SOLVER_MAX_ITERATIONS
                    ),

                "ftol":
                    float(
                        FRICTION_SOLVER_FTOL
                    ),

                "disp":
                    False,
            },
        )

        candidate_force = np.asarray(
            optimization_result.x,
            dtype=float,
        ).reshape(
            number_force_variables
        )

        if not np.all(
            np.isfinite(
                candidate_force
            )
        ):

            return (
                None,
                float(
                    "nan"
                ),
                float(
                    "nan"
                ),
            )

        # ----------------------------------------------------
        # Post-validation.
        #
        # Do not accept the solver's success flag blindly.
        # Explicitly re-check:
        #
        #   A f = b
        #   Fz >= 0
        #   ||Ft|| <= mu Fz
        # ----------------------------------------------------

        candidate_residual = (
            A
            @
            candidate_force
            -
            b
        )

        candidate_residual_inf = float(
            np.linalg.norm(
                candidate_residual,
                ord=np.inf,
            )
        )

        candidate_friction_margin = (
            friction_function(
                candidate_force
            )
        )

        normal_force = (
            candidate_force[
                normal_indices
            ]
        )

        equality_ok = (
            candidate_residual_inf
            <=
            equality_tolerance
        )

        unilateral_ok = bool(
            np.all(
                normal_force
                >=
                -UNILATERAL_FORCE_TOLERANCE
            )
        )

        friction_ok = bool(
            np.all(
                candidate_friction_margin
                >=
                -FRICTION_CONE_TOLERANCE
            )
        )

        if not (
            equality_ok
            and
            unilateral_ok
            and
            friction_ok
        ):

            return (
                None,
                float(
                    "nan"
                ),
                float(
                    "nan"
                ),
            )

        return (
            candidate_force,
            candidate_residual_inf,
            float(
                np.linalg.norm(
                    candidate_residual
                )
            ),
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

        friction_coefficients = np.asarray(
            [
                contact[
                    3
                ]
                for contact
                in contacts
            ],
            dtype=float,
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
                unilateral_feasible=False,
                friction_cone_enabled=(
                    ENABLE_FRICTION_CONE
                ),
                friction_cone_feasible=(
                    False
                    if
                    ENABLE_FRICTION_CONE
                    else
                    None
                ),
                max_friction_ratio=float(
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
        # Stage 1:
        # unilateral minimum-norm force solution
        #
        #     min  1/2 ||f_c||^2
        #
        #     s.t. A_base f_c = base_required
        #          Fz_i >= 0
        # ----------------------------------------------------

        (
            unilateral_force_vector,
            unilateral_residual_inf,
            unilateral_residual_l2,
        ) = (
            self._solve_unilateral_minimum_norm(
                A=(
                    A_base
                ),
                b=(
                    base_required
                ),
                number_contacts=(
                    number_contacts
                ),
            )
        )

        unilateral_feasible = (
            unilateral_force_vector
            is not None
        )

        # ----------------------------------------------------
        # If no unilateral-feasible force distribution exists,
        # do not clip a negative Fz after the solve. That would
        # destroy the dynamic equality A f = b.
        #
        # Instead mark this sample as infeasible. NaN resultants
        # make the invalid interval appear as a gap in the plots.
        # ----------------------------------------------------

        if not unilateral_feasible:

            nan3 = np.full(
                3,
                np.nan,
                dtype=float,
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
                    "nan"
                ),
                dynamics_residual_l2=float(
                    "nan"
                ),
                unilateral_feasible=False,
                friction_cone_enabled=(
                    ENABLE_FRICTION_CONE
                ),
                friction_cone_feasible=(
                    False
                    if
                    ENABLE_FRICTION_CONE
                    else
                    None
                ),
                max_friction_ratio=float(
                    "nan"
                ),
                point_contact_forces=tuple(),
                left_resultant_force_world=(
                    nan3.copy()
                ),
                left_resultant_moment_world=(
                    nan3.copy()
                ),
                right_resultant_force_world=(
                    nan3.copy()
                ),
                right_resultant_moment_world=(
                    nan3.copy()
                ),
            )

        # ----------------------------------------------------
        # Stage 2:
        # optional Coulomb friction cone.
        #
        # When disabled, preserve the original unilateral solution.
        #
        # When enabled, solve:
        #
        #     min  1/2 ||f_c||^2
        #
        #     s.t. A_base f_c = base_required
        #          Fz_i >= 0
        #          sqrt(Fx_i^2 + Fy_i^2) <= mu_i Fz_i
        # ----------------------------------------------------

        if ENABLE_FRICTION_CONE:

            (
                contact_force_vector,
                residual_inf,
                residual_l2,
            ) = (
                self._solve_friction_cone_minimum_norm(
                    A=(
                        A_base
                    ),
                    b=(
                        base_required
                    ),
                    number_contacts=(
                        number_contacts
                    ),
                    friction_coefficients=(
                        friction_coefficients
                    ),
                    initial_force=(
                        unilateral_force_vector
                    ),
                )
            )

            friction_cone_feasible = (
                contact_force_vector
                is not None
            )

            if not friction_cone_feasible:

                nan3 = np.full(
                    3,
                    np.nan,
                    dtype=float,
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
                        "nan"
                    ),
                    dynamics_residual_l2=float(
                        "nan"
                    ),
                    unilateral_feasible=True,
                    friction_cone_enabled=True,
                    friction_cone_feasible=False,
                    max_friction_ratio=float(
                        "nan"
                    ),
                    point_contact_forces=tuple(),
                    left_resultant_force_world=(
                        nan3.copy()
                    ),
                    left_resultant_moment_world=(
                        nan3.copy()
                    ),
                    right_resultant_force_world=(
                        nan3.copy()
                    ),
                    right_resultant_moment_world=(
                        nan3.copy()
                    ),
                )

        else:

            contact_force_vector = (
                unilateral_force_vector
            )

            residual_inf = (
                unilateral_residual_inf
            )

            residual_l2 = (
                unilateral_residual_l2
            )

            friction_cone_feasible = None

        # ----------------------------------------------------
        # Friction-ratio diagnostic.
        # ----------------------------------------------------

        friction_ratios = (
            self._friction_ratios(
                contact_force_vector=(
                    contact_force_vector
                ),
                friction_coefficients=(
                    friction_coefficients
                ),
            )
        )

        if friction_ratios.size > 0:

            max_friction_ratio = float(
                np.max(
                    friction_ratios
                )
            )

        else:

            max_friction_ratio = float(
                "nan"
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
            friction_coefficient,
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
                    friction_coefficient=float(
                        friction_coefficient
                    ),
                    friction_ratio=float(
                        friction_ratios[
                            contact_index
                        ]
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
            unilateral_feasible=True,
            friction_cone_enabled=(
                ENABLE_FRICTION_CONE
            ),
            friction_cone_feasible=(
                friction_cone_feasible
            ),
            max_friction_ratio=float(
                max_friction_ratio
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

        unilateral_feasible = [
            result
            for result
            in with_contact
            if result.unilateral_feasible
        ]

        unilateral_infeasible = [
            result
            for result
            in with_contact
            if not result.unilateral_feasible
        ]

        friction_cone_enabled = any(
            result.friction_cone_enabled
            for result
            in results
        )

        if friction_cone_enabled:

            friction_feasible = [
                result
                for result
                in unilateral_feasible
                if result.friction_cone_feasible
            ]

            friction_infeasible = [
                result
                for result
                in unilateral_feasible
                if not result.friction_cone_feasible
            ]

            valid_force_results = (
                friction_feasible
            )

        else:

            friction_feasible = []
            friction_infeasible = []

            valid_force_results = (
                unilateral_feasible
            )

        residual_inf = [
            result.dynamics_residual_inf
            for result
            in valid_force_results
        ]

        condition_number = [
            result.dynamics_condition_number
            for result
            in valid_force_results
        ]

        left_force_norm = [
            np.linalg.norm(
                result.left_resultant_force_world
            )
            for result
            in valid_force_results
        ]

        right_force_norm = [
            np.linalg.norm(
                result.right_resultant_force_world
            )
            for result
            in valid_force_results
        ]

        friction_ratio_values = [
            result.max_friction_ratio
            for result
            in valid_force_results
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

        (
            friction_ratio_p50,
            friction_ratio_p95,
            friction_ratio_p99,
            friction_ratio_max,
        ) = (
            ContactForceReconstructor
            ._percentiles(
                friction_ratio_values
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

        print(
            f"Unilateral feasible    : {len(unilateral_feasible)}"
        )

        print(
            f"Unilateral infeasible  : {len(unilateral_infeasible)}"
        )

        if friction_cone_enabled:

            print(
                "Friction cone          : ENABLED"
            )

            print(
                f"Friction feasible      : {len(friction_feasible)}"
            )

            print(
                f"Friction infeasible    : {len(friction_infeasible)}"
            )

        else:

            print(
                "Friction cone          : DISABLED"
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

        print()

        if friction_cone_enabled:

            print(
                "MAX POINT-CONTACT FRICTION RATIO"
            )

        else:

            print(
                "MAX POINT-CONTACT FRICTION RATIO "
                "(DIAGNOSTIC ONLY)"
            )

        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {friction_ratio_p50:.6f}"
        )

        print(
            f"  p95 : {friction_ratio_p95:.6f}"
        )

        print(
            f"  p99 : {friction_ratio_p99:.6f}"
        )

        print(
            f"  max : {friction_ratio_max:.6f}"
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
    Plot reconstructed resultant contact forces for each foot separately.

    The forces are expressed in the MuJoCo world frame.

    Figures:
        1) Fx: left foot, right foot
        2) Fy: left foot, right foot
        3) Fz: left foot, right foot
        4) Resultant force norm: left foot, right foot

    No total force of the two feet is calculated or plotted here.
    Mx/My/Mz are also intentionally not plotted at this stage.

    Samples where the active force constraints cannot be satisfied
    are stored as NaN and therefore appear as gaps in the force plots.

    If ENABLE_FRICTION_CONE is False:
        active constraints are dynamics equality + Fz_i >= 0.

    If ENABLE_FRICTION_CONE is True:
        the Coulomb friction cone is also enforced.
    """

    import matplotlib.pyplot as plt

    # Kept in the function signature for compatibility with run.py.
    _ = mj_model

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

    unilateral_infeasible_count = sum(
        1
        for result
        in results
        if (
            result.number_contacts
            >
            0
            and
            not result.unilateral_feasible
        )
    )

    if unilateral_infeasible_count > 0:

        print(
            f"Contact-force plot: "
            f"{unilateral_infeasible_count} samples are "
            "unilateral-infeasible and will appear as gaps."
        )

    friction_infeasible_count = sum(
        1
        for result
        in results
        if (
            result.number_contacts
            >
            0
            and
            result.unilateral_feasible
            and
            result.friction_cone_enabled
            and
            not result.friction_cone_feasible
        )
    )

    if friction_infeasible_count > 0:

        print(
            f"Contact-force plot: "
            f"{friction_infeasible_count} samples are "
            "friction-cone-infeasible and will appear as gaps."
        )

    # ========================================================
    # DATA
    # ========================================================

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

    left_force_norm = np.linalg.norm(
        left_force,
        axis=1,
    )

    right_force_norm = np.linalg.norm(
        right_force,
        axis=1,
    )

    max_friction_ratio = np.asarray(
        [
            result.max_friction_ratio
            for result
            in results
        ],
        dtype=float,
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
    ):

        if np.any(
            np.isinf(
                value
            )
        ):

            raise RuntimeError(
                f"Plot data '{name}' contains Inf."
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
    # FRICTION RATIO FIGURE
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    axis.plot(
        time,
        max_friction_ratio,
        linewidth=1.3,
        label="max contact friction ratio",
    )

    axis.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
        label="friction-cone boundary",
    )

    axis.set_title(
        "Maximum Point-Contact Friction Ratio"
    )

    axis.set_xlabel(
        "Time (s)"
    )

    axis.set_ylabel(
        "rho = ||Ft|| / (mu Fz)"
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
    # PRINT FORCE COMPONENT STATISTICS
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

    for foot_name, foot_force in (
        (
            "LEFT FOOT",
            left_force,
        ),
        (
            "RIGHT FOOT",
            right_force,
        ),
    ):

        print()
        print(
            foot_name
        )
        print(
            "------------------------------------------------"
        )

        for component_index, component_name in enumerate(
            component_names
        ):

            component = (
                foot_force[
                    :,
                    component_index
                ]
            )

            finite_component = (
                component[
                    np.isfinite(
                        component
                    )
                ]
            )

            if finite_component.size == 0:

                print(
                    f"{component_name:>2s}"
                    " | no force-feasible samples"
                )

                continue

            print(
                f"{component_name:>2s}"
                f" | p50={np.percentile(finite_component, 50.0):+.6f} N"
                f" | p05={np.percentile(finite_component, 5.0):+.6f} N"
                f" | p95={np.percentile(finite_component, 95.0):+.6f} N"
                f" | min={np.min(finite_component):+.6f} N"
                f" | max={np.max(finite_component):+.6f} N"
            )

    print(
        "================================================"
    )
    print()

    if show:

        plt.show()
