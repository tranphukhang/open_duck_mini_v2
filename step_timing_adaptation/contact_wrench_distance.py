from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

BASE_DOF = 6

MATRIX_RCOND = 1.0e-10

SUPPORT_TOLERANCE = 1.0e-8
DISTANCE_TOLERANCE = 1.0e-6

COEFFICIENT_TOLERANCE = 1.0e-10

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
        [
            FOOT_CONTACT_X_MAX,
            FOOT_CONTACT_Y_MAX,
            0.0,
        ],
        [
            FOOT_CONTACT_X_MAX,
            FOOT_CONTACT_Y_MIN,
            0.0,
        ],
        [
            FOOT_CONTACT_X_MIN,
            FOOT_CONTACT_Y_MAX,
            0.0,
        ],
        [
            FOOT_CONTACT_X_MIN,
            FOOT_CONTACT_Y_MIN,
            0.0,
        ],
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
class WrenchDistanceResult:

    time: float

    stance_side: str

    number_contacts: int

    collision_detected: bool

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


# ============================================================
# CONTACT WRENCH DISTANCE EVALUATOR
# ============================================================

class ContactWrenchDistanceEvaluator:

    """
    Evaluate the distance from the required floating-base wrench
    to the feasible multicontact wrench cone.

    Contact force at point i:

        f_i = [Fx_i, Fy_i, Fz_i]^T

    Friction constraint:

        Fz_i >= 0

        sqrt(Fx_i^2 + Fy_i^2)
            <= mu_i Fz_i

    Primitive force set:

        U_i =
        {
            [mu_i cos(theta),
             mu_i sin(theta),
             1]^T
        }

    Primitive wrench set:

        W_i = G_i(U_i)

    Total feasible wrench cone:

        V = cone(W_1 union ... union W_m)

    The distance algorithm follows Zheng & Chew (2009).

    In this first implementation, the closest point on the
    current simplicial cone is obtained by checking all of its
    faces directly.

    Because the wrench space dimension is only 6, this is small
    and easy to verify before implementing the paper's recursive
    acceleration formulas.
    """

    def __init__(
        self,
        *,
        mj_model,
    ) -> None:

        self.mj_model = mj_model

        self.samples: list[
            TrajectorySample
        ] = []

        # ====================================================
        # OBJECT IDS
        # ====================================================

        self.floor_geom_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                FLOOR_GEOM_NAME,
            )
        )

        self.left_foot_geom_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                LEFT_FOOT_GEOM_NAME,
            )
        )

        self.right_foot_geom_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                RIGHT_FOOT_GEOM_NAME,
            )
        )

        self.left_foot_site_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_SITE,
                LEFT_FOOT_SITE_NAME,
            )
        )

        self.right_foot_site_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_SITE,
                RIGHT_FOOT_SITE_NAME,
            )
        )

        self.left_foot_body_id = int(
            self.mj_model.site_bodyid[
                self.left_foot_site_id
            ]
        )

        self.right_foot_body_id = int(
            self.mj_model.site_bodyid[
                self.right_foot_site_id
            ]
        )

        floating_base_joint_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                FLOATING_BASE_JOINT_NAME,
            )
        )

        if int(
            self.mj_model.jnt_type[
                floating_base_joint_id
            ]
        ) != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):

            raise RuntimeError(
                f"Joint '{FLOATING_BASE_JOINT_NAME}' "
                "is not a free joint."
            )

        if int(
            self.mj_model.jnt_dofadr[
                floating_base_joint_id
            ]
        ) != 0:

            raise RuntimeError(
                "The first six qvel coordinates must "
                "belong to the floating base."
            )

        # ====================================================
        # FRICTION COEFFICIENTS
        # ====================================================

        floor_mu = float(
            self.mj_model.geom_friction[
                self.floor_geom_id,
                0,
            ]
        )

        left_mu = float(
            self.mj_model.geom_friction[
                self.left_foot_geom_id,
                0,
            ]
        )

        right_mu = float(
            self.mj_model.geom_friction[
                self.right_foot_geom_id,
                0,
            ]
        )

        self.left_mu = min(
            floor_mu,
            left_mu,
        )

        self.right_mu = min(
            floor_mu,
            right_mu,
        )

        print()
        print(
            "================================================"
        )
        print(
            "CONTACT WRENCH DISTANCE EVALUATOR"
        )
        print(
            "================================================"
        )
        print(
            "Contact rectangle: "
            f"{(FOOT_CONTACT_X_MAX - FOOT_CONTACT_X_MIN) * 1000.0:.2f}"
            " x "
            f"{(FOOT_CONTACT_Y_MAX - FOOT_CONTACT_Y_MIN) * 1000.0:.2f}"
            " mm"
        )
        print(
            "Contact points / stance foot: "
            f"{NUMBER_CONTACT_POINTS_PER_FOOT}"
        )
        print(
            f"Left friction coefficient : "
            f"{self.left_mu:.4f}"
        )
        print(
            f"Right friction coefficient: "
            f"{self.right_mu:.4f}"
        )
        print(
            "Contact mode source: gait state machine"
        )
        print(
            "Collision detection: diagnostic only"
        )
        print(
            "================================================"
        )
        print()

    # ========================================================
    # OBJECT LOOKUP
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
                f"MuJoCo object '{name}' "
                "was not found."
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
        stance_side,
    ) -> None:

        sample_time = float(
            time
        )

        stance_side = str(
            stance_side
        ).lower()

        if stance_side not in (
            "left",
            "right",
        ):

            raise ValueError(
                f"Invalid stance_side: "
                f"{stance_side}"
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
            self.samples[-1].time
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
                "Unexpected qpos shape."
            )

        if qvel.shape != (
            self.mj_model.nv,
        ):

            raise RuntimeError(
                "Unexpected qvel shape."
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
                "Recorded qpos/qvel contains "
                "NaN or Inf."
            )

        self.samples.append(
            TrajectorySample(
                time=sample_time,
                qpos=qpos,
                qvel=qvel,
                stance_side=stance_side,
            )
        )

    # ========================================================
    # MASS MATRIX
    # ========================================================

    def _full_mass_matrix(
        self,
        *,
        mj_data,
    ) -> np.ndarray:

        M = np.zeros(
            (
                self.mj_model.nv,
                self.mj_model.nv,
            ),
            dtype=float,
        )

        mujoco.mj_fullM(
            self.mj_model,
            M,
            mj_data.qM,
        )

        if not np.all(
            np.isfinite(
                M
            )
        ):

            raise RuntimeError(
                "Mass matrix contains NaN/Inf."
            )

        return M

    # ========================================================
    # COLLISION DIAGNOSTIC
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

    def _stance_collision_detected(
        self,
        *,
        mj_data,
        stance_side,
    ) -> bool:

        if stance_side == "left":

            stance_geom_id = (
                self.left_foot_geom_id
            )

        elif stance_side == "right":

            stance_geom_id = (
                self.right_foot_geom_id
            )

        else:

            raise ValueError(
                f"Invalid stance_side: "
                f"{stance_side}"
            )

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
                in pair
                and
                stance_geom_id
                in pair
            ):

                if (
                    float(
                        contact.dist
                    )
                    <=
                    float(
                        contact.includemargin
                    )
                    +
                    1.0e-9
                ):

                    return True

        return False

    # ========================================================
    # FIXED CONTACT POINTS
    # ========================================================

    def _fixed_contacts(
        self,
        *,
        mj_data,
        stance_side,
    ):

        if stance_side == "left":

            site_id = (
                self.left_foot_site_id
            )

            body_id = (
                self.left_foot_body_id
            )

            mu = (
                self.left_mu
            )

        elif stance_side == "right":

            site_id = (
                self.right_foot_site_id
            )

            body_id = (
                self.right_foot_body_id
            )

            mu = (
                self.right_mu
            )

        else:

            raise ValueError(
                f"Invalid stance_side: "
                f"{stance_side}"
            )

        site_position_world = (
            np.asarray(
                mj_data.site_xpos[
                    site_id
                ],
                dtype=float,
            )
            .reshape(
                3
            )
        )

        R_world_site = (
            np.asarray(
                mj_data.site_xmat[
                    site_id
                ],
                dtype=float,
            )
            .reshape(
                3,
                3,
            )
        )

        contacts = []

        for point_local in (
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
                (
                    point_world.copy(),
                    body_id,
                    float(
                        mu
                    ),
                )
            )

        return contacts

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

        J = np.zeros(
            (
                3,
                self.mj_model.nv,
            ),
            dtype=float,
        )

        mujoco.mj_jac(
            self.mj_model,
            mj_data,
            J,
            None,
            np.asarray(
                position_world,
                dtype=float,
            ),
            int(
                body_id
            ),
        )

        return J

    # ========================================================
    # CONTACT WRENCH MAP Gi
    # ========================================================

    def _build_contact_wrench_maps(
        self,
        *,
        mj_data,
        contacts,
    ):

        G_blocks = []

        friction_coefficients = []

        for (
            position_world,
            body_id,
            mu,
        ) in contacts:

            J_i = (
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

            # -----------------------------------------------
            # World point force:
            #
            #     f_i = [Fx, Fy, Fz]^T
            #
            # Floating-base generalized wrench:
            #
            #     w_i = G_i f_i
            #
            # G_i:
            #
            #     6 x 3
            # -----------------------------------------------

            G_i = (
                J_i.T[
                    0:
                    BASE_DOF,
                    :
                ].copy()
            )

            G_blocks.append(
                G_i
            )

            friction_coefficients.append(
                float(
                    mu
                )
            )

        return (
            G_blocks,
            np.asarray(
                friction_coefficients,
                dtype=float,
            ),
        )

    # ========================================================
    # MATRIX DIAGNOSTIC
    # ========================================================

    @staticmethod
    def _matrix_condition_number(
        matrix,
    ) -> float:

        matrix = np.asarray(
            matrix,
            dtype=float,
        )

        singular_values = (
            np.linalg.svd(
                matrix,
                compute_uv=False,
            )
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
            smallest
            <=
            0.0
            or
            not np.isfinite(
                smallest
            )
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
    ):

        """
        Continuous PCwF support mapping.

        No friction-cone discretization is required.

        Code force ordering:

            [Fx, Fy, Fz]

        Flat ground:

            Fz = 1

            sqrt(Fx^2 + Fy^2) = mu
        """

        r = np.asarray(
            residual,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        best_h = -float(
            "inf"
        )

        best_wrench = None

        for (
            G_i,
            mu_i,
        ) in zip(
            G_blocks,
            friction_coefficients,
        ):

            # -----------------------------------------------
            # u_i = G_i^T r
            #
            # Since G_i maps [Fx, Fy, Fz],
            # u_i has the same component ordering.
            # -----------------------------------------------

            u_i = (
                G_i.T
                @
                r
            )

            tangential_norm = float(
                np.hypot(
                    u_i[0],
                    u_i[1],
                )
            )

            # -----------------------------------------------
            # s_Ui(G_i^T r)
            # -----------------------------------------------

            if (
                tangential_norm
                >
                1.0e-14
            ):

                primitive_force = np.array(
                    [
                        mu_i
                        *
                        u_i[0]
                        /
                        tangential_norm,

                        mu_i
                        *
                        u_i[1]
                        /
                        tangential_norm,

                        1.0,
                    ],
                    dtype=float,
                )

            else:

                # Any point of Ui is valid if the tangential
                # projection is exactly zero.

                primitive_force = np.array(
                    [
                        mu_i,
                        0.0,
                        1.0,
                    ],
                    dtype=float,
                )

            # -----------------------------------------------
            # Primitive wrench:
            #
            #     w_i = G_i s_Ui
            # -----------------------------------------------

            primitive_wrench = (
                G_i
                @
                primitive_force
            )

            support_value = float(
                r
                @
                primitive_wrench
            )

            if (
                support_value
                >
                best_h
            ):

                best_h = (
                    support_value
                )

                best_wrench = (
                    primitive_wrench.copy()
                )

        if best_wrench is None:

            raise RuntimeError(
                "Support mapping failed."
            )

        return (
            best_h,
            best_wrench,
        )

    # ========================================================
    # CLOSEST POINT ON CURRENT SIMPLICIAL CONE
    # ========================================================

    @staticmethod
    def _closest_point_on_current_cone(
        *,
        required_wrench,
        generators,
    ):

        """
        Compute the closest point on cone(generators).

        The 2009 paper gives a dedicated subalgorithm.

        In this first implementation, all faces of the current
        simplicial cone are checked explicitly.

        Wrench dimension is only 6, therefore the number of
        active generators remains small.
        """

        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        if len(
            generators
        ) == 0:

            return (
                np.zeros(
                    BASE_DOF,
                    dtype=float,
                ),
                [],
            )

        A = np.column_stack(
            generators
        )

        number_generators = (
            A.shape[1]
        )

        # Origin is always in the cone.

        best_v = np.zeros(
            BASE_DOF,
            dtype=float,
        )

        best_distance = float(
            np.linalg.norm(
                b
            )
        )

        best_active_generators = []

        # ----------------------------------------------------
        # Check every face of the current cone.
        # ----------------------------------------------------

        for subset_size in range(
            1,
            number_generators + 1,
        ):

            for subset_indices in (
                combinations(
                    range(
                        number_generators
                    ),
                    subset_size,
                )
            ):

                B = A[
                    :,
                    subset_indices,
                ]

                matrix_norm = float(
                    np.linalg.norm(
                        B,
                        ord=2,
                    )
                )

                rank = int(
                    np.linalg.matrix_rank(
                        B,
                        tol=(
                            MATRIX_RCOND
                            *
                            max(
                                B.shape
                            )
                            *
                            max(
                                matrix_norm,
                                1.0,
                            )
                        ),
                    )
                )

                if (
                    rank
                    <
                    subset_size
                ):

                    continue

                # -------------------------------------------
                # Orthogonal projection:
                #
                #     c = (B^T B)^-1 B^T b
                #
                # np.linalg.lstsq is numerically safer.
                # -------------------------------------------

                coefficients, *_ = (
                    np.linalg.lstsq(
                        B,
                        b,
                        rcond=None,
                    )
                )

                # Cone coefficients must be nonnegative.

                if np.any(
                    coefficients
                    <
                    -COEFFICIENT_TOLERANCE
                ):

                    continue

                coefficients = np.maximum(
                    coefficients,
                    0.0,
                )

                v = (
                    B
                    @
                    coefficients
                )

                distance = float(
                    np.linalg.norm(
                        b
                        -
                        v
                    )
                )

                if (
                    distance
                    <
                    best_distance
                    -
                    1.0e-12
                ):

                    best_distance = (
                        distance
                    )

                    best_v = (
                        v.copy()
                    )

                    best_active_generators = [
                        generators[
                            index
                        ]
                        for (
                            index,
                            coefficient,
                        )
                        in zip(
                            subset_indices,
                            coefficients,
                        )
                        if (
                            coefficient
                            >
                            COEFFICIENT_TOLERANCE
                        )
                    ]

        return (
            best_v,
            best_active_generators,
        )

    # ========================================================
    # ZHENG-CHEW DISTANCE ALGORITHM
    # ========================================================

    def _distance_to_feasible_wrench_cone(
        self,
        *,
        required_wrench,
        G_blocks,
        friction_coefficients,
    ):

        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        # ----------------------------------------------------
        # Paper initialization:
        #
        #     v0 = 0
        #     r0 = b
        #     A0 = empty
        # ----------------------------------------------------

        v = np.zeros(
            BASE_DOF,
            dtype=float,
        )

        residual = (
            b.copy()
        )

        active_generators = []

        final_support_value = float(
            "nan"
        )

        for iteration in range(
            MAX_DISTANCE_ITERATIONS
            +
            1
        ):

            # -----------------------------------------------
            # h_W(r_k), s_W(r_k)
            # -----------------------------------------------

            (
                support_value,
                new_generator,
            ) = (
                self._support_mapping(
                    residual=(
                        residual
                    ),
                    G_blocks=(
                        G_blocks
                    ),
                    friction_coefficients=(
                        friction_coefficients
                    ),
                )
            )

            final_support_value = float(
                support_value
            )

            # -----------------------------------------------
            # Termination condition from the paper.
            # -----------------------------------------------

            if (
                support_value
                <
                SUPPORT_TOLERANCE
            ):

                distance = float(
                    np.linalg.norm(
                        residual
                    )
                )

                return (
                    distance,
                    v,
                    residual,
                    True,
                    iteration,
                    final_support_value,
                    len(
                        active_generators
                    ),
                )

            # -----------------------------------------------
            # A_{k+1}
            #
            # Add the newly selected primitive wrench.
            # -----------------------------------------------

            candidate_generators = (
                active_generators
                +
                [
                    new_generator
                ]
            )

            # -----------------------------------------------
            # Find v_{k+1}.
            # -----------------------------------------------

            (
                v,
                active_generators,
            ) = (
                self._closest_point_on_current_cone(
                    required_wrench=(
                        b
                    ),
                    generators=(
                        candidate_generators
                    ),
                )
            )

            # -----------------------------------------------
            # r_{k+1} = b - v_{k+1}
            # -----------------------------------------------

            residual = (
                b
                -
                v
            )

            # Required wrench lies in the cone.

            if (
                float(
                    np.linalg.norm(
                        residual
                    )
                )
                <=
                DISTANCE_TOLERANCE
            ):

                return (
                    0.0,
                    v,
                    residual,
                    True,
                    iteration + 1,
                    0.0,
                    len(
                        active_generators
                    ),
                )

        # ----------------------------------------------------
        # Maximum iteration reached.
        # ----------------------------------------------------

        distance = float(
            np.linalg.norm(
                residual
            )
        )

        return (
            distance,
            v,
            residual,
            False,
            MAX_DISTANCE_ITERATIONS,
            final_support_value,
            len(
                active_generators
            ),
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

        # ====================================================
        # RESTORE STATE
        # ====================================================

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

        # ====================================================
        # REQUIRED FLOATING-BASE WRENCH
        # ====================================================

        M = (
            self._full_mass_matrix(
                mj_data=(
                    scratch_data
                )
            )
        )

        required_generalized_force = (
            M
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
                0:
                BASE_DOF
            ].copy()
        )

        # ====================================================
        # CONTACT MODE / CONTACT GEOMETRY
        # ====================================================

        collision_detected = (
            self._stance_collision_detected(
                mj_data=(
                    scratch_data
                ),
                stance_side=(
                    sample.stance_side
                ),
            )
        )

        contacts = (
            self._fixed_contacts(
                mj_data=(
                    scratch_data
                ),
                stance_side=(
                    sample.stance_side
                ),
            )
        )

        if (
            len(
                contacts
            )
            !=
            NUMBER_CONTACT_POINTS_PER_FOOT
        ):

            raise RuntimeError(
                "Unexpected number of fixed contact points."
            )

        # ====================================================
        # Gi MATRICES
        # ====================================================

        (
            G_blocks,
            friction_coefficients,
        ) = (
            self._build_contact_wrench_maps(
                mj_data=(
                    scratch_data
                ),
                contacts=(
                    contacts
                ),
            )
        )

        G = np.hstack(
            G_blocks
        )

        dynamics_rank = int(
            np.linalg.matrix_rank(
                G
            )
        )

        dynamics_condition_number = (
            self._matrix_condition_number(
                G
            )
        )

        # ====================================================
        # DISTANCE TO FEASIBLE WRENCH CONE
        # ====================================================

        (
            distance,
            closest_wrench,
            residual_wrench,
            converged,
            iterations,
            final_support_value,
            active_generator_count,
        ) = (
            self._distance_to_feasible_wrench_cone(
                required_wrench=(
                    required_wrench
                ),
                G_blocks=(
                    G_blocks
                ),
                friction_coefficients=(
                    friction_coefficients
                ),
            )
        )

        feasible = bool(
            converged
            and
            distance
            <=
            DISTANCE_TOLERANCE
        )

        return WrenchDistanceResult(
            time=float(
                sample.time
            ),
            stance_side=(
                sample.stance_side
            ),
            number_contacts=len(
                contacts
            ),
            collision_detected=bool(
                collision_detected
            ),
            dynamics_rank=(
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
            closest_wrench=(
                np.asarray(
                    closest_wrench,
                    dtype=float,
                ).copy()
            ),
            residual_wrench=(
                np.asarray(
                    residual_wrench,
                    dtype=float,
                ).copy()
            ),
            distance=float(
                distance
            ),
            feasible=(
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
        )

    # ========================================================
    # SOLVE COMPLETE TRAJECTORY
    # ========================================================

    def solve_all(
        self,
    ) -> list[
        WrenchDistanceResult
    ]:

        number_samples = len(
            self.samples
        )

        if number_samples < 3:

            raise RuntimeError(
                "At least three trajectory samples "
                "are required."
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
                "Sample time must be strictly increasing."
            )

        qvel = np.vstack(
            [
                sample.qvel
                for sample
                in self.samples
            ]
        )

        # ----------------------------------------------------
        # Offline qdd reconstruction.
        # ----------------------------------------------------

        qacc = np.gradient(
            qvel,
            time,
            axis=0,
            edge_order=2,
        )

        scratch_data = (
            mujoco.MjData(
                self.mj_model
            )
        )

        results = []

        for (
            sample_index,
            sample,
        ) in enumerate(
            self.samples
        ):

            results.append(
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

        if len(
            results
        ) == 0:

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

        iterations = np.asarray(
            [
                result.iterations
                for result
                in results
            ],
            dtype=int,
        )

        collision = np.asarray(
            [
                result.collision_detected
                for result
                in results
            ],
            dtype=bool,
        )

        ranks = np.asarray(
            [
                result.dynamics_rank
                for result
                in results
            ],
            dtype=int,
        )

        print()
        print(
            "================================================"
        )
        print(
            "CONTACT WRENCH DISTANCE SUMMARY"
        )
        print(
            "================================================"
        )

        print(
            f"Samples                : "
            f"{len(results)}"
        )

        print(
            f"Converged              : "
            f"{np.count_nonzero(converged)}"
            f"/{len(results)}"
        )

        print(
            f"Distance-zero feasible : "
            f"{np.count_nonzero(feasible)}"
            f"/{len(results)}"
        )

        print(
            f"Stance collision       : "
            f"{np.count_nonzero(collision)}"
            f"/{len(results)}"
        )

        print(
            f"Min / max rank(G)      : "
            f"{np.min(ranks)} / "
            f"{np.max(ranks)}"
        )

        print(
            f"Mean distance          : "
            f"{np.mean(distances):.6e}"
        )

        print(
            f"Max distance           : "
            f"{np.max(distances):.6e}"
        )

        print(
            f"Max iterations         : "
            f"{np.max(iterations)}"
        )

        print(
            "================================================"
        )
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

    if len(
        results
    ) == 0:

        return

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

    fig, axis = (
        plt.subplots()
    )

    axis.plot(
        time,
        distance,
        linewidth=1.5,
    )

    axis.axhline(
        DISTANCE_TOLERANCE,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Time [s]"
    )

    axis.set_ylabel(
        "Distance to feasible wrench cone"
    )

    axis.set_title(
        "Contact-wrench feasibility distance"
    )

    axis.grid(
        True
    )

    fig.tight_layout()

    if show:

        plt.show()

    return (
        fig,
        axis,
    )