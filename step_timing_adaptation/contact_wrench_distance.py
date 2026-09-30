# step_timing_adaptation/contact_wrench_distance.py

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BASE_DOF = 6

# Zheng & Chew (2009), Distance Algorithm Step 2.
SUPPORT_TOLERANCE = 1.0e-8

# Numerical tolerance used only to classify d ~= 0 in software.
DISTANCE_TOLERANCE = 1.0e-6

# Numerical interpretation of negative / zero / positive
# coefficients in the paper subalgorithm.
COEFFICIENT_TOLERANCE = 1.0e-10

# Numerical interpretation of Theorem 4 condition 2.
SUPPORT_PLANE_TOLERANCE = 1.0e-10

# Programming guard only; not an additional paper step.
MAX_DISTANCE_ITERATIONS = 100


# ============================================================
# FIXED FOOT CONTACT MODEL
# ============================================================

FOOT_CONTACT_X_MAX = 0.06286
FOOT_CONTACT_X_MIN = -0.03810
FOOT_CONTACT_Y_MAX = 0.01901
FOOT_CONTACT_Y_MIN = -0.01822

FOOT_CONTACT_POINTS_LOCAL = np.array(
    [
        [FOOT_CONTACT_X_MAX, FOOT_CONTACT_Y_MAX, 0.0],
        [FOOT_CONTACT_X_MAX, FOOT_CONTACT_Y_MIN, 0.0],
        [FOOT_CONTACT_X_MIN, FOOT_CONTACT_Y_MAX, 0.0],
        [FOOT_CONTACT_X_MIN, FOOT_CONTACT_Y_MIN, 0.0],
    ],
    dtype=float,
)

NUMBER_CONTACT_POINTS_PER_FOOT = 4


# ============================================================
# MUJOCO OBJECT NAMES
# ============================================================

FLOOR_GEOM_NAME = "floor"
LEFT_FOOT_GEOM_NAME = "left_foot_bottom_tpu"
RIGHT_FOOT_GEOM_NAME = "right_foot_bottom_tpu"
LEFT_FOOT_SITE_NAME = "left_foot"
RIGHT_FOOT_SITE_NAME = "right_foot"
FLOATING_BASE_JOINT_NAME = "floating_base"

# Leg joints whose required generalized torques are reconstructed.
# Neck/head joints are intentionally excluded.
LEG_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class TrajectorySample:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    stance_side: str


@dataclass(frozen=True)
class FixedContact:
    position_world: np.ndarray
    body_id: int
    mu: float
    side: str
    point_index: int


@dataclass(frozen=True)
class PrimitiveGenerator:
    """
    Primitive generator selected by the support mapping.

    wrench:
        w_j = G_i s_j

    force:
        s_j in U_i

    contact_index:
        Index of the active fixed contact in the current sample.
    """

    wrench: np.ndarray
    force: np.ndarray
    contact_index: int
    side: str
    point_index: int


@dataclass(frozen=True)
class WrenchDistanceResult:
    time: float
    stance_side: str

    contact_mode: str
    left_collision_detected: bool
    right_collision_detected: bool
    collision_detected: bool

    number_contacts: int

    dynamics_rank: int
    dynamics_condition_number: float

    qacc: np.ndarray

    required_wrench: np.ndarray
    closest_wrench: np.ndarray
    residual_wrench: np.ndarray

    distance: float
    feasible: bool

    converged: bool
    iterations: int
    final_support_value: float
    active_generator_count: int

    # Contact forces at the four predefined points of each foot.
    # Shape: (4, 3), world ordering [Fx, Fy, Fz].
    left_contact_forces: np.ndarray
    right_contact_forces: np.ndarray

    # Total force of each foot, world ordering [Fx, Fy, Fz].
    left_total_force: np.ndarray
    right_total_force: np.ndarray

    # Diagnostics for the 6 floating-base equations.
    reconstructed_contact_wrench: np.ndarray
    force_reconstruction_error: float
    force_balance_error: float

    # Full generalized-force quantities.
    # required_generalized_force =
    #     M @ qacc + qfrc_bias - qfrc_passive
    required_generalized_force: np.ndarray

    # Sum_i Jv_i^T f_i over all active fixed contact points.
    generalized_contact_force: np.ndarray

    # Residual of the six unactuated floating-base equations.
    base_dynamics_residual: np.ndarray

    # Required generalized torques at the 10 leg joints.
    joint_torque_names: tuple[str, ...]
    joint_torques: np.ndarray


# ============================================================
# CONTACT WRENCH DISTANCE EVALUATOR
# ============================================================

class ContactWrenchDistanceEvaluator:
    """
    Evaluate contact-wrench feasibility and recover contact forces.

    Contact mode:
        left only  -> left single support  -> 4 fixed points
        right only -> right single support -> 4 fixed points
        both feet  -> double support       -> 8 fixed points
        neither    -> no support

    MuJoCo collision detection only decides which foot is active.
    Actual MuJoCo collision locations are not used to construct V.

    Distance:
        Zheng & Chew (2009) convex-cone distance algorithm.

    Force distribution:
        From the final active primitive set W_hat,

            w_j = G_{I(j)} s_j,

        the coefficients c_j are recovered from the paper's
        projection relation and

            f_i = sum_{j : I(j)=i} c_j s_j,

        corresponding to Eq. (50)-(51) of Zheng & Chew (2009).

    Joint-torque reconstruction:
        This is a rigid-body inverse-dynamics post-processing step,
        not an additional step of the Zheng-Chew distance algorithm.

        After the point-contact forces f_i are recovered,

            Q_c = sum_i Jv_i^T f_i

        and the required leg-joint generalized torques are obtained
        from the actuated components of

            M qdd + qfrc_bias - qfrc_passive - Q_c.
    """

    def __init__(self, *, mj_model) -> None:
        self.mj_model = mj_model
        self.samples: list[TrajectorySample] = []

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

        self.left_foot_body_id = int(
            self.mj_model.site_bodyid[self.left_foot_site_id]
        )
        self.right_foot_body_id = int(
            self.mj_model.site_bodyid[self.right_foot_site_id]
        )

        floating_base_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            FLOATING_BASE_JOINT_NAME,
        )

        if int(self.mj_model.jnt_type[floating_base_joint_id]) != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError(
                f"Joint '{FLOATING_BASE_JOINT_NAME}' is not a free joint."
            )

        if int(self.mj_model.jnt_dofadr[floating_base_joint_id]) != 0:
            raise RuntimeError(
                "This implementation assumes that the first six qvel "
                "coordinates belong to the floating base."
            )

        if self.mj_model.nv < BASE_DOF:
            raise RuntimeError(
                "MuJoCo model has fewer than six generalized DoFs."
            )

        # ----------------------------------------------------
        # LEG JOINT GENERALIZED-VELOCITY INDICES
        # ----------------------------------------------------
        #
        # For each 1-DoF hinge joint, jnt_dofadr gives the row
        # index of its generalized velocity/force in MuJoCo.
        # These indices are used later to extract the required
        # leg-joint generalized torques from
        #
        #     M qdd + qfrc_bias - qfrc_passive - Jv^T f.
        #
        self.leg_joint_names = tuple(
            LEG_JOINT_NAMES
        )

        leg_dof_indices = []

        for joint_name in self.leg_joint_names:
            joint_id = self._require_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            joint_type = int(
                self.mj_model.jnt_type[
                    joint_id
                ]
            )

            if joint_type != int(
                mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise RuntimeError(
                    f"Leg joint '{joint_name}' is not a hinge joint."
                )

            dof_index = int(
                self.mj_model.jnt_dofadr[
                    joint_id
                ]
            )

            if dof_index < BASE_DOF:
                raise RuntimeError(
                    f"Leg joint '{joint_name}' overlaps the floating-base DoFs."
                )

            leg_dof_indices.append(
                dof_index
            )

        if len(set(leg_dof_indices)) != len(leg_dof_indices):
            raise RuntimeError(
                "Duplicate generalized-velocity index detected "
                "among the leg joints."
            )

        self.leg_dof_indices = np.asarray(
            leg_dof_indices,
            dtype=int,
        )

        floor_mu = float(
            self.mj_model.geom_friction[self.floor_geom_id, 0]
        )
        left_mu = float(
            self.mj_model.geom_friction[self.left_foot_geom_id, 0]
        )
        right_mu = float(
            self.mj_model.geom_friction[self.right_foot_geom_id, 0]
        )

        self.left_mu = min(floor_mu, left_mu)
        self.right_mu = min(floor_mu, right_mu)

        contact_length_mm = (
            FOOT_CONTACT_X_MAX - FOOT_CONTACT_X_MIN
        ) * 1000.0
        contact_width_mm = (
            FOOT_CONTACT_Y_MAX - FOOT_CONTACT_Y_MIN
        ) * 1000.0

        print()
        print("================================================")
        print("CONTACT WRENCH DISTANCE / FORCE DISTRIBUTION")
        print("================================================")
        print(
            "Contact rectangle: "
            f"{contact_length_mm:.2f} x "
            f"{contact_width_mm:.2f} mm"
        )
        print(
            "Contact points / active foot: "
            f"{NUMBER_CONTACT_POINTS_PER_FOOT}"
        )
        print(f"Left friction coefficient : {self.left_mu:.4f}")
        print(f"Right friction coefficient: {self.right_mu:.4f}")
        print(
            "Contact mode source: "
            "MuJoCo foot-floor collision detection"
        )
        print(
            "Force distribution: "
            "Zheng & Chew (2009), Eq. (50)-(51)"
        )
        print(
            "Joint torque reconstruction: "
            f"{len(self.leg_joint_names)} leg joints"
        )
        print("================================================")
        print()

    # ========================================================
    # BASIC MODEL HELPERS
    # ========================================================

    def _require_id(self, object_type, name: str) -> int:
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

    def record_sample(
        self,
        *,
        time,
        mj_data,
        stance_side,
    ) -> None:
        sample_time = float(time)
        stance_side = str(stance_side).lower()

        if stance_side not in ("left", "right"):
            raise ValueError(
                f"Invalid stance_side: {stance_side}"
            )
        if not np.isfinite(sample_time):
            raise ValueError(
                "Sample time must be finite."
            )
        if self.samples and sample_time <= self.samples[-1].time:
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

        if qpos.shape != (self.mj_model.nq,):
            raise RuntimeError(
                "Unexpected qpos shape."
            )
        if qvel.shape != (self.mj_model.nv,):
            raise RuntimeError(
                "Unexpected qvel shape."
            )
        if (
            not np.all(np.isfinite(qpos))
            or
            not np.all(np.isfinite(qvel))
        ):
            raise RuntimeError(
                "Recorded qpos/qvel contains NaN or Inf."
            )

        self.samples.append(
            TrajectorySample(
                time=sample_time,
                qpos=qpos,
                qvel=qvel,
                stance_side=stance_side,
            )
        )

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

        try:
            mujoco.mj_fullM(
                self.mj_model,
                mj_data,
                mass_matrix,
            )
        except TypeError:
            if not hasattr(
                mj_data,
                "qM",
            ):
                raise RuntimeError(
                    "Unsupported MuJoCo mj_fullM API: "
                    "new API failed and mjData.qM is unavailable."
                )

            mujoco.mj_fullM(
                self.mj_model,
                mass_matrix,
                mj_data.qM,
            )

        if not np.all(np.isfinite(mass_matrix)):
            raise RuntimeError(
                "Mass matrix contains NaN/Inf."
            )

        return mass_matrix

    # ========================================================
    # COLLISION DETECTION
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
            ).reshape(-1)
            if geom.size >= 2:
                return (
                    int(geom[0]),
                    int(geom[1]),
                )

        return (
            int(contact.geom1),
            int(contact.geom2),
        )

    def _foot_collision_detected(
        self,
        *,
        mj_data,
        foot_geom_id,
    ) -> bool:
        foot_geom_id = int(
            foot_geom_id
        )

        for contact_index in range(
            int(mj_data.ncon)
        ):
            contact = mj_data.contact[
                contact_index
            ]

            geom_0, geom_1 = self._contact_geom_ids(
                contact
            )

            pair = {
                geom_0,
                geom_1,
            }

            if (
                self.floor_geom_id in pair
                and
                foot_geom_id in pair
            ):
                if (
                    float(contact.dist)
                    <=
                    float(contact.includemargin)
                    +
                    1.0e-9
                ):
                    return True

        return False

    def _detect_contact_mode(
        self,
        *,
        mj_data,
    ) -> tuple[bool, bool, str]:
        left_contact = self._foot_collision_detected(
            mj_data=mj_data,
            foot_geom_id=self.left_foot_geom_id,
        )
        right_contact = self._foot_collision_detected(
            mj_data=mj_data,
            foot_geom_id=self.right_foot_geom_id,
        )

        if left_contact and right_contact:
            contact_mode = "double_support"
        elif left_contact:
            contact_mode = "left_single_support"
        elif right_contact:
            contact_mode = "right_single_support"
        else:
            contact_mode = "no_support"

        return (
            bool(left_contact),
            bool(right_contact),
            contact_mode,
        )

    # ========================================================
    # FIXED CONTACT POINTS
    # ========================================================

    def _fixed_contacts_for_side(
        self,
        *,
        mj_data,
        side,
    ) -> list[FixedContact]:
        side = str(
            side
        ).lower()

        if side == "left":
            site_id = self.left_foot_site_id
            body_id = self.left_foot_body_id
            mu = self.left_mu
        elif side == "right":
            site_id = self.right_foot_site_id
            body_id = self.right_foot_body_id
            mu = self.right_mu
        else:
            raise ValueError(
                f"Invalid foot side: {side}"
            )

        site_position_world = np.asarray(
            mj_data.site_xpos[
                site_id
            ],
            dtype=float,
        ).reshape(3)

        R_world_site = np.asarray(
            mj_data.site_xmat[
                site_id
            ],
            dtype=float,
        ).reshape(3, 3)

        contacts = []

        for point_index, point_local in enumerate(
            FOOT_CONTACT_POINTS_LOCAL
        ):
            point_world = (
                site_position_world
                +
                R_world_site
                @
                point_local
            )

            contacts.append(
                FixedContact(
                    position_world=(
                        point_world.copy()
                    ),
                    body_id=int(
                        body_id
                    ),
                    mu=float(
                        mu
                    ),
                    side=side,
                    point_index=int(
                        point_index
                    ),
                )
            )

        return contacts

    def _active_fixed_contacts(
        self,
        *,
        mj_data,
        left_contact,
        right_contact,
    ) -> list[FixedContact]:
        contacts = []

        # Deterministic ordering:
        # left P1..P4, then right P1..P4.
        if left_contact:
            contacts.extend(
                self._fixed_contacts_for_side(
                    mj_data=mj_data,
                    side="left",
                )
            )

        if right_contact:
            contacts.extend(
                self._fixed_contacts_for_side(
                    mj_data=mj_data,
                    side="right",
                )
            )

        return contacts

    # ========================================================
    # POINT JACOBIAN / CONTACT WRENCH MAP
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
            int(body_id),
        )

        if not np.all(np.isfinite(jacobian)):
            raise RuntimeError(
                "Point Jacobian contains NaN/Inf."
            )

        return jacobian

    def _build_contact_wrench_maps(
        self,
        *,
        mj_data,
        contacts,
    ):
        """
        Build both forms needed later:

        Jv_i:
            Full translational point Jacobian,
            shape (3, nv).

        G_i:
            Floating-base wrench map,
            G_i = Jv_i.T[0:6, :],
            shape (6, 3).

        G_i is used by the Zheng-Chew wrench-distance algorithm.
        The full Jv_i is retained so that, after f_i has been
        reconstructed, Jv_i.T @ f_i can be used in the complete
        rigid-body dynamics.
        """

        G_blocks = []
        Jv_blocks = []
        friction_coefficients = []

        for contact in contacts:
            Jv_i = self._point_translation_jacobian(
                mj_data=mj_data,
                position_world=(
                    contact.position_world
                ),
                body_id=(
                    contact.body_id
                ),
            )

            if Jv_i.shape != (
                3,
                self.mj_model.nv,
            ):
                raise RuntimeError(
                    f"Unexpected Jv_i shape: {Jv_i.shape}"
                )

            G_i = Jv_i.T[
                0:BASE_DOF,
                :
            ].copy()

            if G_i.shape != (
                BASE_DOF,
                3,
            ):
                raise RuntimeError(
                    f"Unexpected G_i shape: {G_i.shape}"
                )

            Jv_blocks.append(
                Jv_i.copy()
            )
            G_blocks.append(
                G_i
            )
            friction_coefficients.append(
                float(contact.mu)
            )

        return (
            G_blocks,
            Jv_blocks,
            np.asarray(
                friction_coefficients,
                dtype=float,
            ),
        )

    @staticmethod
    def _matrix_condition_number(
        matrix,
    ) -> float:
        singular_values = np.linalg.svd(
            np.asarray(
                matrix,
                dtype=float,
            ),
            compute_uv=False,
        )

        if singular_values.size == 0:
            return float(
                "inf"
            )

        largest = float(
            singular_values[0]
        )
        smallest = float(
            singular_values[-1]
        )

        if (
            smallest <= 0.0
            or
            not np.isfinite(smallest)
        ):
            return float(
                "inf"
            )

        return (
            largest
            /
            smallest
        )

    # ========================================================
    # SUPPORT FUNCTION / SUPPORT MAPPING
    # ========================================================

    @staticmethod
    def _support_mapping(
        *,
        residual,
        G_blocks,
        friction_coefficients,
        contacts,
    ):
        """
        PCwF support mapping, Eq. (40)-(44).

        Paper ordering:
            [normal, tangent1, tangent2]

        Current force ordering:
            [Fx, Fy, Fz]

        Flat ground:
            normal = Fz.
        """

        r = np.asarray(
            residual,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        best_support_value = -float(
            "inf"
        )
        best_generator = None

        for contact_index, (
            G_i,
            mu_i,
            contact,
        ) in enumerate(
            zip(
                G_blocks,
                friction_coefficients,
                contacts,
            )
        ):
            # u_i = G_i^T r.
            u_i = (
                G_i.T
                @
                r
            )

            u_t1 = float(
                u_i[0]
            )
            u_t2 = float(
                u_i[1]
            )
            u_n = float(
                u_i[2]
            )
            mu_i = float(
                mu_i
            )

            rho = float(
                np.hypot(
                    u_t1,
                    u_t2,
                )
            )

            h_t = (
                mu_i
                *
                rho
            )

            support_value = (
                u_n
                +
                h_t
            )

            if h_t != 0.0:
                # Eq. (44), reordered to [Fx,Fy,Fz].
                primitive_force = np.array(
                    [
                        (
                            mu_i
                            *
                            mu_i
                            *
                            u_t1
                        )
                        /
                        h_t,

                        (
                            mu_i
                            *
                            mu_i
                            *
                            u_t2
                        )
                        /
                        h_t,

                        1.0,
                    ],
                    dtype=float,
                )
            else:
                # If h_Ti=0, paper allows any point of U_i.
                primitive_force = np.array(
                    [
                        mu_i,
                        0.0,
                        1.0,
                    ],
                    dtype=float,
                )

            primitive_wrench = (
                G_i
                @
                primitive_force
            )

            if (
                support_value
                >
                best_support_value
            ):
                best_support_value = float(
                    support_value
                )

                best_generator = PrimitiveGenerator(
                    wrench=(
                        primitive_wrench.copy()
                    ),
                    force=(
                        primitive_force.copy()
                    ),
                    contact_index=int(
                        contact_index
                    ),
                    side=(
                        contact.side
                    ),
                    point_index=int(
                        contact.point_index
                    ),
                )

        if best_generator is None:
            raise RuntimeError(
                "Support mapping failed."
            )

        return (
            best_support_value,
            best_generator,
        )

    # ========================================================
    # PAPER EQ. (7)-(9)
    # ========================================================

    @staticmethod
    def _projection_quantities(
        *,
        required_wrench,
        generators,
    ):
        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        if len(generators) == 0:
            raise ValueError(
                "Projection set must not be empty."
            )

        A = np.column_stack(
            [
                np.asarray(
                    generator,
                    dtype=float,
                ).reshape(
                    BASE_DOF
                )
                for generator
                in generators
            ]
        )

        gram = (
            A.T
            @
            A
        )
        rhs = (
            A.T
            @
            b
        )

        try:
            coefficients = np.linalg.solve(
                gram,
                rhs,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "The current generator set is not "
                "numerically linearly independent, "
                "so Eq. (7) cannot be evaluated."
            ) from exc

        projection = (
            A
            @
            coefficients
        )
        residual = (
            b
            -
            projection
        )

        return (
            coefficients,
            projection,
            residual,
        )

    @staticmethod
    def _coefficient_sets(
        coefficients,
    ):
        c = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(-1)

        negative = [
            i
            for i, value
            in enumerate(c)
            if (
                value
                <
                -COEFFICIENT_TOLERANCE
            )
        ]

        zero = [
            i
            for i, value
            in enumerate(c)
            if (
                abs(value)
                <=
                COEFFICIENT_TOLERANCE
            )
        ]

        positive = [
            i
            for i, value
            in enumerate(c)
            if (
                value
                >
                COEFFICIENT_TOLERANCE
            )
        ]

        return (
            negative,
            zero,
            positive,
        )

    # ========================================================
    # PAPER SUBALGORITHM, SECTION IV-B
    # ========================================================

    def _subalgorithm(
        self,
        *,
        required_wrench,
        A_k_plus_1,
        new_generator,
    ):
        """
        Same paper subalgorithm as before. PrimitiveGenerator
        only carries additional metadata; all geometric tests
        use generator.wrench.
        """

        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        working = []

        for index, generator in enumerate(
            A_k_plus_1
        ):
            working.append(
                {
                    "generator":
                        generator,
                    "is_new":
                        (
                            index
                            ==
                            len(
                                A_k_plus_1
                            )
                            -
                            1
                        ),
                }
            )

        if not working:
            raise ValueError(
                "A_{k+1} must not be empty."
            )

        if not np.allclose(
            working[-1][
                "generator"
            ].wrench,
            new_generator.wrench,
            rtol=0.0,
            atol=(
                COEFFICIENT_TOLERANCE
            ),
        ):
            raise RuntimeError(
                "The last generator of "
                "A_{k+1} must be s_A(r_k)."
            )

        # Step 1 / Step 2.
        while True:
            wrenches = [
                item[
                    "generator"
                ].wrench
                for item
                in working
            ]

            (
                coefficients,
                projection,
                residual,
            ) = (
                self._projection_quantities(
                    required_wrench=b,
                    generators=(
                        wrenches
                    ),
                )
            )

            (
                negative,
                _,
                positive,
            ) = (
                self._coefficient_sets(
                    coefficients
                )
            )

            if len(negative) == 0:
                A_hat = [
                    working[i][
                        "generator"
                    ]
                    for i
                    in positive
                ]

                return (
                    projection,
                    A_hat,
                )

            if len(negative) == 1:
                remove_index = (
                    negative[0]
                )

                if working[
                    remove_index
                ][
                    "is_new"
                ]:
                    raise RuntimeError(
                        "Numerical inconsistency in "
                        "the paper subalgorithm: "
                        "s_A(r_k) became the unique "
                        "negative-coefficient point."
                    )

                del working[
                    remove_index
                ]
                continue

            break

        # Current A- after singleton removals.
        wrenches = [
            item[
                "generator"
            ].wrench
            for item
            in working
        ]

        (
            coefficients,
            _,
            _,
        ) = (
            self._projection_quantities(
                required_wrench=b,
                generators=(
                    wrenches
                ),
            )
        )

        (
            negative,
            _,
            _,
        ) = (
            self._coefficient_sets(
                coefficients
            )
        )

        negative_set = set(
            negative
        )

        new_indices = [
            i
            for i, item
            in enumerate(
                working
            )
            if item[
                "is_new"
            ]
        ]

        if len(new_indices) != 1:
            raise RuntimeError(
                "Could not uniquely identify "
                "s_A(r_k)."
            )

        new_index = (
            new_indices[0]
        )
        number_points = len(
            working
        )

        # Steps 3-5.
        for candidate_size in range(
            number_points - 1,
            0,
            -1,
        ):
            for subset_tuple in combinations(
                range(
                    number_points
                ),
                candidate_size,
            ):
                subset = set(
                    subset_tuple
                )

                # Theorem 5(1).
                if new_index not in subset:
                    continue

                # Theorem 5(2).
                if negative_set.issubset(
                    subset
                ):
                    continue

                candidate_indices = sorted(
                    subset
                )

                candidate_generators = [
                    working[i][
                        "generator"
                    ]
                    for i
                    in candidate_indices
                ]

                candidate_wrenches = [
                    generator.wrench
                    for generator
                    in candidate_generators
                ]

                (
                    candidate_coefficients,
                    candidate_projection,
                    candidate_residual,
                ) = (
                    self._projection_quantities(
                        required_wrench=b,
                        generators=(
                            candidate_wrenches
                        ),
                    )
                )

                # Theorem 4 condition 1.
                if not np.all(
                    candidate_coefficients
                    >
                    COEFFICIENT_TOLERANCE
                ):
                    continue

                # Theorem 4 condition 2.
                condition_2_satisfied = True

                for index, item in enumerate(
                    working
                ):
                    if index in subset:
                        continue

                    value = float(
                        item[
                            "generator"
                        ].wrench
                        @
                        candidate_residual
                    )

                    if (
                        value
                        >
                        SUPPORT_PLANE_TOLERANCE
                    ):
                        condition_2_satisfied = False
                        break

                if not condition_2_satisfied:
                    continue

                return (
                    candidate_projection,
                    candidate_generators,
                )

        raise RuntimeError(
            "The paper subalgorithm could not find "
            "Ahat_{k+1}. This indicates a numerical "
            "inconsistency with the assumed linearly "
            "independent generator set."
        )

    # ========================================================
    # PAPER DISTANCE ALGORITHM, SECTION IV-A
    # ========================================================

    def _distance_to_feasible_wrench_cone(
        self,
        *,
        required_wrench,
        G_blocks,
        friction_coefficients,
        contacts,
    ):
        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        if not np.all(
            np.isfinite(b)
        ):
            raise RuntimeError(
                "Required wrench contains NaN/Inf."
            )

        if len(G_blocks) == 0:
            raise ValueError(
                "Distance algorithm requires at least "
                "one active contact point."
            )

        # Step 1.
        v_k = np.zeros(
            BASE_DOF,
            dtype=float,
        )
        r_k = (
            b.copy()
        )
        A_hat_k = []
        k = 0
        final_support_value = float(
            "nan"
        )

        while (
            k
            <
            MAX_DISTANCE_ITERATIONS
        ):
            (
                support_value,
                support_generator,
            ) = (
                self._support_mapping(
                    residual=r_k,
                    G_blocks=(
                        G_blocks
                    ),
                    friction_coefficients=(
                        friction_coefficients
                    ),
                    contacts=(
                        contacts
                    ),
                )
            )

            final_support_value = float(
                support_value
            )

            # Step 2.
            if (
                support_value
                <
                SUPPORT_TOLERANCE
            ):
                distance = float(
                    np.linalg.norm(
                        r_k
                    )
                )

                return (
                    distance,
                    v_k,
                    r_k,
                    True,
                    k,
                    final_support_value,
                    A_hat_k,
                )

            # Step 3.
            A_k_plus_1 = list(
                A_hat_k
            )
            A_k_plus_1.append(
                support_generator
            )

            # Step 4.
            (
                v_k_plus_1,
                A_hat_k_plus_1,
            ) = (
                self._subalgorithm(
                    required_wrench=b,
                    A_k_plus_1=(
                        A_k_plus_1
                    ),
                    new_generator=(
                        support_generator
                    ),
                )
            )

            # Step 5.
            r_k_plus_1 = (
                b
                -
                v_k_plus_1
            )

            v_k = (
                v_k_plus_1
            )
            r_k = (
                r_k_plus_1
            )
            A_hat_k = (
                A_hat_k_plus_1
            )
            k += 1

        # Software guard only.
        distance = float(
            np.linalg.norm(
                r_k
            )
        )

        return (
            distance,
            v_k,
            r_k,
            False,
            k,
            final_support_value,
            A_hat_k,
        )

    # ========================================================
    # FORCE DISTRIBUTION, EQ. (50)-(51)
    # ========================================================

    def _distribute_contact_forces(
        self,
        *,
        required_wrench,
        closest_wrench,
        active_generators,
        contacts,
        G_blocks,
    ):
        """
        Recover contact forces from the final active primitive set.

            closest_wrench
                = sum_j c_j w_j
                = sum_j c_j G_{I(j)} s_j

            f_i
                = sum_{j : I(j)=i} c_j s_j

        For a feasible sample:
            closest_wrench ~= required_wrench.
        """

        left_contact_forces = np.zeros(
            (
                NUMBER_CONTACT_POINTS_PER_FOOT,
                3,
            ),
            dtype=float,
        )

        right_contact_forces = np.zeros(
            (
                NUMBER_CONTACT_POINTS_PER_FOOT,
                3,
            ),
            dtype=float,
        )

        reconstructed_contact_wrench = np.zeros(
            BASE_DOF,
            dtype=float,
        )

        if len(active_generators) == 0:
            left_total_force = np.zeros(
                3,
                dtype=float,
            )
            right_total_force = np.zeros(
                3,
                dtype=float,
            )

            force_reconstruction_error = float(
                np.linalg.norm(
                    closest_wrench
                )
            )
            force_balance_error = float(
                np.linalg.norm(
                    required_wrench
                )
            )

            return (
                left_contact_forces,
                right_contact_forces,
                left_total_force,
                right_total_force,
                reconstructed_contact_wrench,
                force_reconstruction_error,
                force_balance_error,
            )

        active_wrenches = [
            generator.wrench
            for generator
            in active_generators
        ]

        (
            coefficients,
            projection,
            _,
        ) = (
            self._projection_quantities(
                required_wrench=(
                    required_wrench
                ),
                generators=(
                    active_wrenches
                ),
            )
        )

        if np.any(
            coefficients
            <
            -COEFFICIENT_TOLERANCE
        ):
            raise RuntimeError(
                "Negative coefficient encountered while "
                "recovering contact forces from A_hat."
            )

        for coefficient, generator in zip(
            coefficients,
            active_generators,
        ):
            force_contribution = (
                float(
                    coefficient
                )
                *
                generator.force
            )

            if generator.side == "left":
                left_contact_forces[
                    generator.point_index
                ] += (
                    force_contribution
                )
            elif generator.side == "right":
                right_contact_forces[
                    generator.point_index
                ] += (
                    force_contribution
                )
            else:
                raise RuntimeError(
                    "Invalid generator foot side: "
                    f"{generator.side}"
                )

        # Reconstruct the generalized contact wrench after
        # grouping the primitive contributions by contact point.
        for contact_index, contact in enumerate(
            contacts
        ):
            if contact.side == "left":
                contact_force = (
                    left_contact_forces[
                        contact.point_index
                    ]
                )
            elif contact.side == "right":
                contact_force = (
                    right_contact_forces[
                        contact.point_index
                    ]
                )
            else:
                raise RuntimeError(
                    "Invalid contact foot side: "
                    f"{contact.side}"
                )

            reconstructed_contact_wrench += (
                G_blocks[
                    contact_index
                ]
                @
                contact_force
            )

        left_total_force = np.sum(
            left_contact_forces,
            axis=0,
        )
        right_total_force = np.sum(
            right_contact_forces,
            axis=0,
        )

        force_reconstruction_error = float(
            np.linalg.norm(
                np.asarray(
                    closest_wrench,
                    dtype=float,
                ).reshape(
                    BASE_DOF
                )
                -
                reconstructed_contact_wrench
            )
        )

        force_balance_error = float(
            np.linalg.norm(
                np.asarray(
                    required_wrench,
                    dtype=float,
                ).reshape(
                    BASE_DOF
                )
                -
                reconstructed_contact_wrench
            )
        )

        projection_mismatch = float(
            np.linalg.norm(
                projection
                -
                np.asarray(
                    closest_wrench,
                    dtype=float,
                ).reshape(
                    BASE_DOF
                )
            )
        )

        if (
            projection_mismatch
            >
            1.0e-7
        ):
            raise RuntimeError(
                "Force-distribution projection is inconsistent "
                "with the closest wrench: "
                f"{projection_mismatch:.6e}"
            )

        return (
            left_contact_forces,
            right_contact_forces,
            left_total_force,
            right_total_force,
            reconstructed_contact_wrench,
            force_reconstruction_error,
            force_balance_error,
        )

    # ========================================================
    # FULL GENERALIZED CONTACT FORCE / JOINT TORQUES
    # ========================================================

    def _generalized_contact_force(
        self,
        *,
        contacts,
        Jv_blocks,
        left_contact_forces,
        right_contact_forces,
    ) -> np.ndarray:
        """
        Compute

            Q_c = sum_i Jv_i^T f_i

        using the same fixed contact points and reconstructed
        point-contact forces used by the wrench-distance method.

        This is the generalized contact-force contribution in the
        complete floating-base rigid-body dynamics.
        """

        if len(contacts) != len(Jv_blocks):
            raise RuntimeError(
                "contacts and Jv_blocks have inconsistent lengths."
            )

        generalized_contact_force = np.zeros(
            self.mj_model.nv,
            dtype=float,
        )

        for contact, Jv_i in zip(
            contacts,
            Jv_blocks,
        ):
            if contact.side == "left":
                force_i = left_contact_forces[
                    contact.point_index
                ]
            elif contact.side == "right":
                force_i = right_contact_forces[
                    contact.point_index
                ]
            else:
                raise RuntimeError(
                    f"Invalid contact side: {contact.side}"
                )

            force_i = np.asarray(
                force_i,
                dtype=float,
            ).reshape(3)

            Jv_i = np.asarray(
                Jv_i,
                dtype=float,
            )

            if Jv_i.shape != (
                3,
                self.mj_model.nv,
            ):
                raise RuntimeError(
                    f"Unexpected Jv_i shape: {Jv_i.shape}"
                )

            generalized_contact_force += (
                Jv_i.T
                @
                force_i
            )

        if not np.all(
            np.isfinite(
                generalized_contact_force
            )
        ):
            raise RuntimeError(
                "Generalized contact force contains NaN/Inf."
            )

        return generalized_contact_force

    # ========================================================
    # NO SUPPORT SPECIAL CASE
    # ========================================================

    @staticmethod
    def _distance_without_support(
        *,
        required_wrench,
    ):
        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        closest_wrench = np.zeros(
            BASE_DOF,
            dtype=float,
        )
        residual_wrench = (
            b.copy()
        )
        distance = float(
            np.linalg.norm(
                residual_wrench
            )
        )

        return (
            distance,
            closest_wrench,
            residual_wrench,
            True,
            0,
            float("nan"),
            [],
        )

    # ========================================================
    # SOLVE ONE SAMPLE
    # ========================================================

    def _solve_one(
        self,
        *,
        sample,
        qacc,
        scratch_data,
    ) -> WrenchDistanceResult:
        qacc = np.asarray(
            qacc,
            dtype=float,
        ).reshape(
            self.mj_model.nv
        )

        if not np.all(
            np.isfinite(qacc)
        ):
            raise RuntimeError(
                "qacc contains NaN/Inf."
            )

        scratch_data.qpos[:] = (
            sample.qpos
        )
        scratch_data.qvel[:] = (
            sample.qvel
        )
        scratch_data.time = float(
            sample.time
        )
        scratch_data.qfrc_applied[:] = 0.0
        scratch_data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(
            self.mj_model,
            scratch_data,
        )

        # M qdd + qfrc_bias
        # = qfrc_passive + S^T tau + J_c^T f_c
        mass_matrix = self._full_mass_matrix(
            mj_data=scratch_data
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

        required_wrench = (
            required_generalized_force[
                0:BASE_DOF
            ].copy()
        )

        (
            left_collision_detected,
            right_collision_detected,
            contact_mode,
        ) = self._detect_contact_mode(
            mj_data=scratch_data
        )

        collision_detected = bool(
            left_collision_detected
            or
            right_collision_detected
        )

        contacts = self._active_fixed_contacts(
            mj_data=scratch_data,
            left_contact=(
                left_collision_detected
            ),
            right_contact=(
                right_collision_detected
            ),
        )

        number_contacts = len(
            contacts
        )

        if number_contacts not in (
            0,
            NUMBER_CONTACT_POINTS_PER_FOOT,
            2 * NUMBER_CONTACT_POINTS_PER_FOOT,
        ):
            raise RuntimeError(
                "Unexpected number of fixed contact points: "
                f"{number_contacts}"
            )

        G_blocks = []
        Jv_blocks = []

        if number_contacts == 0:
            dynamics_rank = 0
            dynamics_condition_number = float(
                "inf"
            )

            (
                distance,
                closest_wrench,
                residual_wrench,
                converged,
                iterations,
                final_support_value,
                active_generators,
            ) = self._distance_without_support(
                required_wrench=(
                    required_wrench
                )
            )

        else:
            (
                G_blocks,
                Jv_blocks,
                friction_coefficients,
            ) = self._build_contact_wrench_maps(
                mj_data=scratch_data,
                contacts=contacts,
            )

            G_total = np.hstack(
                G_blocks
            )

            dynamics_rank = int(
                np.linalg.matrix_rank(
                    G_total
                )
            )
            dynamics_condition_number = (
                self._matrix_condition_number(
                    G_total
                )
            )

            (
                distance,
                closest_wrench,
                residual_wrench,
                converged,
                iterations,
                final_support_value,
                active_generators,
            ) = self._distance_to_feasible_wrench_cone(
                required_wrench=(
                    required_wrench
                ),
                G_blocks=(
                    G_blocks
                ),
                friction_coefficients=(
                    friction_coefficients
                ),
                contacts=(
                    contacts
                ),
            )

        active_generator_count = len(
            active_generators
        )

        feasible = bool(
            converged
            and
            distance
            <=
            DISTANCE_TOLERANCE
        )

        if number_contacts == 0:
            left_contact_forces = np.zeros(
                (
                    NUMBER_CONTACT_POINTS_PER_FOOT,
                    3,
                ),
                dtype=float,
            )
            right_contact_forces = np.zeros(
                (
                    NUMBER_CONTACT_POINTS_PER_FOOT,
                    3,
                ),
                dtype=float,
            )
            left_total_force = np.zeros(
                3,
                dtype=float,
            )
            right_total_force = np.zeros(
                3,
                dtype=float,
            )
            reconstructed_contact_wrench = np.zeros(
                BASE_DOF,
                dtype=float,
            )
            force_reconstruction_error = float(
                np.linalg.norm(
                    closest_wrench
                )
            )
            force_balance_error = float(
                np.linalg.norm(
                    required_wrench
                )
            )

        else:
            (
                left_contact_forces,
                right_contact_forces,
                left_total_force,
                right_total_force,
                reconstructed_contact_wrench,
                force_reconstruction_error,
                force_balance_error,
            ) = self._distribute_contact_forces(
                required_wrench=(
                    required_wrench
                ),
                closest_wrench=(
                    closest_wrench
                ),
                active_generators=(
                    active_generators
                ),
                contacts=(
                    contacts
                ),
                G_blocks=(
                    G_blocks
                ),
            )

        # ----------------------------------------------------
        # FULL GENERALIZED CONTACT FORCE
        # ----------------------------------------------------
        #
        # Complete floating-base dynamics:
        #
        #   M qdd + qfrc_bias - qfrc_passive
        #       = S^T tau + sum_i Jv_i^T f_i
        #
        # Therefore:
        #
        #   S^T tau
        #       = required_generalized_force
        #         - generalized_contact_force
        #
        # The first six components should be approximately zero
        # for a contact-wrench-feasible sample. The selected
        # leg-joint components are the required joint torques.
        #
        if number_contacts == 0:
            generalized_contact_force = np.zeros(
                self.mj_model.nv,
                dtype=float,
            )
        else:
            generalized_contact_force = (
                self._generalized_contact_force(
                    contacts=contacts,
                    Jv_blocks=Jv_blocks,
                    left_contact_forces=(
                        left_contact_forces
                    ),
                    right_contact_forces=(
                        right_contact_forces
                    ),
                )
            )

        generalized_dynamics_residual = (
            required_generalized_force
            -
            generalized_contact_force
        )

        base_dynamics_residual = (
            generalized_dynamics_residual[
                0:BASE_DOF
            ].copy()
        )

        joint_torques = (
            generalized_dynamics_residual[
                self.leg_dof_indices
            ].copy()
        )

        if joint_torques.shape != (
            len(self.leg_joint_names),
        ):
            raise RuntimeError(
                "Unexpected reconstructed joint-torque shape."
            )

        if (
            not np.all(
                np.isfinite(
                    base_dynamics_residual
                )
            )
            or
            not np.all(
                np.isfinite(
                    joint_torques
                )
            )
        ):
            raise RuntimeError(
                "Reconstructed dynamics contains NaN/Inf."
            )

        return WrenchDistanceResult(
            time=float(
                sample.time
            ),
            stance_side=(
                sample.stance_side
            ),
            contact_mode=(
                contact_mode
            ),
            left_collision_detected=bool(
                left_collision_detected
            ),
            right_collision_detected=bool(
                right_collision_detected
            ),
            collision_detected=bool(
                collision_detected
            ),
            number_contacts=int(
                number_contacts
            ),
            dynamics_rank=int(
                dynamics_rank
            ),
            dynamics_condition_number=float(
                dynamics_condition_number
            ),
            qacc=(
                qacc.copy()
            ),
            required_wrench=(
                required_wrench.copy()
            ),
            closest_wrench=np.asarray(
                closest_wrench,
                dtype=float,
            ).copy(),
            residual_wrench=np.asarray(
                residual_wrench,
                dtype=float,
            ).copy(),
            distance=float(
                distance
            ),
            feasible=bool(
                feasible
            ),
            converged=bool(
                converged
            ),
            iterations=int(
                iterations
            ),
            final_support_value=float(
                final_support_value
            ),
            active_generator_count=int(
                active_generator_count
            ),
            left_contact_forces=np.asarray(
                left_contact_forces,
                dtype=float,
            ).copy(),
            right_contact_forces=np.asarray(
                right_contact_forces,
                dtype=float,
            ).copy(),
            left_total_force=np.asarray(
                left_total_force,
                dtype=float,
            ).copy(),
            right_total_force=np.asarray(
                right_total_force,
                dtype=float,
            ).copy(),
            reconstructed_contact_wrench=np.asarray(
                reconstructed_contact_wrench,
                dtype=float,
            ).copy(),
            force_reconstruction_error=float(
                force_reconstruction_error
            ),
            force_balance_error=float(
                force_balance_error
            ),
            required_generalized_force=np.asarray(
                required_generalized_force,
                dtype=float,
            ).copy(),
            generalized_contact_force=np.asarray(
                generalized_contact_force,
                dtype=float,
            ).copy(),
            base_dynamics_residual=np.asarray(
                base_dynamics_residual,
                dtype=float,
            ).copy(),
            joint_torque_names=tuple(
                self.leg_joint_names
            ),
            joint_torques=np.asarray(
                joint_torques,
                dtype=float,
            ).copy(),
        )

    # ========================================================
    # SOLVE COMPLETE TRAJECTORY
    # ========================================================

    def solve_all(
        self,
    ) -> list[WrenchDistanceResult]:
        number_samples = len(
            self.samples
        )

        if number_samples < 3:
            raise RuntimeError(
                "At least three trajectory samples are required."
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
            np.diff(time)
            >
            0.0
        ):
            raise RuntimeError(
                "Sample time must be strictly increasing."
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
            results.append(
                self._solve_one(
                    sample=sample,
                    qacc=(
                        qacc[
                            sample_index
                        ]
                    ),
                    scratch_data=scratch_data,
                )
            )

        return results

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def print_summary(
        results,
    ) -> None:
        results = list(
            results
        )

        if not results:
            print(
                "No wrench-distance results."
            )
            return

        distances = np.asarray(
            [
                result.distance
                for result
                in results
            ],
            dtype=float,
        )
        feasible = np.asarray(
            [
                result.feasible
                for result
                in results
            ],
            dtype=bool,
        )
        converged = np.asarray(
            [
                result.converged
                for result
                in results
            ],
            dtype=bool,
        )
        collision = np.asarray(
            [
                result.collision_detected
                for result
                in results
            ],
            dtype=bool,
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
        ranks = np.asarray(
            [
                result.dynamics_rank
                for result
                in results
            ],
            dtype=int,
        )
        force_reconstruction_error = np.asarray(
            [
                result.force_reconstruction_error
                for result
                in results
            ],
            dtype=float,
        )
        force_balance_error = np.asarray(
            [
                result.force_balance_error
                for result
                in results
            ],
            dtype=float,
        )

        base_dynamics_residual_norm = np.asarray(
            [
                np.linalg.norm(
                    result.base_dynamics_residual
                )
                for result
                in results
            ],
            dtype=float,
        )

        joint_torque_matrix = np.vstack(
            [
                result.joint_torques
                for result
                in results
            ]
        )

        contact_modes = [
            result.contact_mode
            for result
            in results
        ]

        left_single_support_count = contact_modes.count(
            "left_single_support"
        )
        right_single_support_count = contact_modes.count(
            "right_single_support"
        )
        double_support_count = contact_modes.count(
            "double_support"
        )
        no_support_count = contact_modes.count(
            "no_support"
        )

        gait_stance_contact_count = 0

        for result in results:
            if (
                result.stance_side == "left"
                and
                result.left_collision_detected
            ):
                gait_stance_contact_count += 1
            elif (
                result.stance_side == "right"
                and
                result.right_collision_detected
            ):
                gait_stance_contact_count += 1

        print()
        print("================================================")
        print("CONTACT WRENCH DISTANCE SUMMARY")
        print("================================================")
        print(f"Samples                : {len(results)}")
        print(
            "Converged              : "
            f"{np.count_nonzero(converged)}/{len(results)}"
        )
        print(
            "Distance-zero feasible : "
            f"{np.count_nonzero(feasible)}/{len(results)}"
        )
        print(
            "Any foot collision     : "
            f"{np.count_nonzero(collision)}/{len(results)}"
        )
        print(
            "Left single support    : "
            f"{left_single_support_count}"
        )
        print(
            "Right single support   : "
            f"{right_single_support_count}"
        )
        print(
            "Double support         : "
            f"{double_support_count}"
        )
        print(
            "No support             : "
            f"{no_support_count}"
        )
        print(
            "Gait stance in contact : "
            f"{gait_stance_contact_count}/{len(results)}"
        )
        print(
            "Min / max rank(G)      : "
            f"{np.min(ranks)} / {np.max(ranks)}"
        )
        print(
            "Mean distance          : "
            f"{np.mean(distances):.6e}"
        )
        print(
            "Max distance           : "
            f"{np.max(distances):.6e}"
        )
        print(
            "Max iterations         : "
            f"{np.max(iterations)}"
        )
        print(
            "Max active generators  : "
            f"{np.max(generator_count)}"
        )
        print(
            "Max force recon error  : "
            f"{np.max(force_reconstruction_error):.6e}"
        )
        print(
            "Max force balance error: "
            f"{np.max(force_balance_error):.6e}"
        )
        print(
            "Max base dyn residual   : "
            f"{np.max(base_dynamics_residual_norm):.6e}"
        )
        print(
            "Max abs leg torque [Nm] : "
            f"{np.max(np.abs(joint_torque_matrix)):.6e}"
        )
        print("================================================")
        print()


# ============================================================
# PLOT
# ============================================================

def plot_wrench_distance_results(
    results,
    *,
    show=False,
):
    import matplotlib.pyplot as plt

    results = list(
        results
    )

    if not results:
        return None

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
    converged = np.asarray(
        [
            result.converged
            for result
            in results
        ],
        dtype=bool,
    )
    contact_mode = np.asarray(
        [
            result.contact_mode
            for result
            in results
        ],
        dtype=object,
    )

    figure, axis = plt.subplots()

    axis.plot(
        time,
        distance,
        linewidth=1.5,
        label="Wrench distance",
    )

    axis.axhline(
        DISTANCE_TOLERANCE,
        linestyle="--",
        linewidth=1.0,
        label="Distance tolerance",
    )

    not_converged = np.logical_not(
        converged
    )

    if np.any(
        not_converged
    ):
        axis.plot(
            time[
                not_converged
            ],
            distance[
                not_converged
            ],
            "x",
            markersize=5,
            label="Not converged",
        )

    double_support_mask = (
        contact_mode
        ==
        "double_support"
    )

    if np.any(
        double_support_mask
    ):
        axis.plot(
            time[
                double_support_mask
            ],
            distance[
                double_support_mask
            ],
            ".",
            markersize=3,
            label="Double support",
        )

    axis.set_xlabel(
        "Time [s]"
    )
    axis.set_ylabel(
        "Euclidean wrench distance"
    )
    axis.set_title(
        "Distance to feasible contact-wrench cone"
    )
    axis.grid(
        True
    )
    axis.legend()

    figure.tight_layout()

    if show:
        plt.show()

    return (
        figure,
        axis,
    )


# ============================================================
# CONTACT FORCE PLOTS
# ============================================================

def plot_contact_force_results(
    results,
    *,
    show=False,
):
    """
    Plot total contact force of the left and right feet.

    Each foot is plotted in a separate figure with world-frame
    components Fx, Fy and Fz. These forces are reconstructed
    from the final primitive-generator set of the contact-wrench
    distance algorithm; they are not raw MuJoCo contact forces.
    """

    import matplotlib.pyplot as plt

    results = list(results)

    if not results:
        return None

    time = np.asarray(
        [result.time for result in results],
        dtype=float,
    )

    left_force = np.vstack(
        [result.left_total_force for result in results]
    )

    right_force = np.vstack(
        [result.right_total_force for result in results]
    )

    if left_force.ndim != 2 or left_force.shape[1] != 3:
        raise RuntimeError(
            "left_total_force must have shape (N, 3)."
        )

    if right_force.ndim != 2 or right_force.shape[1] != 3:
        raise RuntimeError(
            "right_total_force must have shape (N, 3)."
        )

    component_labels = (
        "Fx",
        "Fy",
        "Fz",
    )

    # --------------------------------------------------------
    # LEFT FOOT
    # --------------------------------------------------------

    figure_left, axis_left = plt.subplots()

    for component_index, component_label in enumerate(
        component_labels
    ):
        axis_left.plot(
            time,
            left_force[:, component_index],
            linewidth=1.5,
            label=component_label,
        )

    axis_left.set_xlabel("Time [s]")
    axis_left.set_ylabel("Force [N]")
    axis_left.set_title("Left foot contact force")
    axis_left.grid(True)
    axis_left.legend()
    figure_left.tight_layout()

    # --------------------------------------------------------
    # RIGHT FOOT
    # --------------------------------------------------------

    figure_right, axis_right = plt.subplots()

    for component_index, component_label in enumerate(
        component_labels
    ):
        axis_right.plot(
            time,
            right_force[:, component_index],
            linewidth=1.5,
            label=component_label,
        )

    axis_right.set_xlabel("Time [s]")
    axis_right.set_ylabel("Force [N]")
    axis_right.set_title("Right foot contact force")
    axis_right.grid(True)
    axis_right.legend()
    figure_right.tight_layout()

    if show:
        plt.show()

    return (
        (figure_left, axis_left),
        (figure_right, axis_right),
    )
