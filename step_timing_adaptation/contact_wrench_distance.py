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

MATRIX_RCOND = 1.0e-10
SUPPORT_TOLERANCE = 1.0e-8
DISTANCE_TOLERANCE = 1.0e-6
COEFFICIENT_TOLERANCE = 1.0e-10
GENERATOR_TOLERANCE = 1.0e-10
PROGRESS_TOLERANCE = 1.0e-12

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
    Distance from the required floating-base generalized wrench b
    to the feasible contact-wrench cone V.

    At contact i:

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

    Contact wrench:

        w_i = G_i f_i

    Primitive wrench set:

        W_i = G_i(U_i)

    Whole stance-foot primitive wrench set:

        W = union_i W_i

    Feasible wrench cone:

        V = cone(W)

    Distance:

        d(b,V) = min ||b-v||

                   v in V

    The outer iteration uses the Zheng-Chew support-mapping
    algorithm.

    The closest point on the current simplicial cone is computed
    here by explicitly checking all of its faces. Since wrench
    space has dimension 6, this is suitable for the current
    verification implementation.
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

        # ====================================================
        # FLOATING-BASE CHECK
        # ====================================================

        joint_type = int(
            self.mj_model.jnt_type[
                floating_base_joint_id
            ]
        )

        if (
            joint_type
            !=
            int(
                mujoco.mjtJoint.mjJNT_FREE
            )
        ):

            raise RuntimeError(
                f"Joint '{FLOATING_BASE_JOINT_NAME}' "
                "is not a free joint."
            )

        floating_base_dof_address = int(
            self.mj_model.jnt_dofadr[
                floating_base_joint_id
            ]
        )

        if floating_base_dof_address != 0:

            raise RuntimeError(
                "This implementation assumes that the "
                "floating-base DoFs occupy qvel indices 0:6."
            )

        if self.mj_model.nv < BASE_DOF:

            raise RuntimeError(
                "MuJoCo model has fewer than six "
                "generalized DoFs."
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

        left_foot_mu = float(
            self.mj_model.geom_friction[
                self.left_foot_geom_id,
                0,
            ]
        )

        right_foot_mu = float(
            self.mj_model.geom_friction[
                self.right_foot_geom_id,
                0,
            ]
        )

        self.left_mu = min(
            floor_mu,
            left_foot_mu,
        )

        self.right_mu = min(
            floor_mu,
            right_foot_mu,
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
                time=(
                    sample_time
                ),
                qpos=(
                    qpos
                ),
                qvel=(
                    qvel
                ),
                stance_side=(
                    stance_side
                ),
            )
        )

    # ========================================================
    # FULL MASS MATRIX
    # ========================================================

    def _full_mass_matrix(
        self,
        *,
        mj_data,
    ) -> np.ndarray:

        """
        Return dense M(q).

        MuJoCo >= 3.13:

            mj_fullM(model, data, dst)

        Older MuJoCo:

            mj_fullM(model, dst, data.qM)

        The fallback below supports both APIs.
        """

        mass_matrix = np.zeros(
            (
                self.mj_model.nv,
                self.mj_model.nv,
            ),
            dtype=float,
        )

        try:

            # -----------------------------------------------
            # MuJoCo 3.13+ API
            # -----------------------------------------------

            mujoco.mj_fullM(
                self.mj_model,
                mj_data,
                mass_matrix,
            )

        except TypeError:

            # -----------------------------------------------
            # Legacy MuJoCo API
            # -----------------------------------------------

            if not hasattr(
                mj_data,
                "qM",
            ):

                raise RuntimeError(
                    "Unsupported MuJoCo mj_fullM API: "
                    "new API call failed and mjData.qM "
                    "is not available."
                )

            mujoco.mj_fullM(
                self.mj_model,
                mass_matrix,
                mj_data.qM,
            )

        if mass_matrix.shape != (
            self.mj_model.nv,
            self.mj_model.nv,
        ):

            raise RuntimeError(
                "Unexpected mass-matrix shape."
            )

        if not np.all(
            np.isfinite(
                mass_matrix
            )
        ):

            raise RuntimeError(
                "Mass matrix contains NaN/Inf."
            )

        return mass_matrix

    # ========================================================
    # COLLISION DIAGNOSTIC
    # ========================================================

    @staticmethod
    def _contact_geom_ids(
        contact,
    ) -> tuple[
        int,
        int,
    ]:

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
                        geom[
                            0
                        ]
                    ),
                    int(
                        geom[
                            1
                        ]
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

        """
        MuJoCo collision is used only as a diagnostic.

        It does not create, remove, or move the candidate
        contact points used by the feasibility model.
        """

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

                contact_distance = float(
                    contact.dist
                )

                include_margin = float(
                    contact.includemargin
                )

                if (
                    contact_distance
                    <=
                    include_margin
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

        """
        Return the four candidate points on the planned
        stance foot.

        p_world =
            p_site_world
            + R_world_site @ p_local
        """

        if stance_side == "left":

            site_id = (
                self.left_foot_site_id
            )

            body_id = (
                self.left_foot_body_id
            )

            friction_coefficient = (
                self.left_mu
            )

        elif stance_side == "right":

            site_id = (
                self.right_foot_site_id
            )

            body_id = (
                self.right_foot_body_id
            )

            friction_coefficient = (
                self.right_mu
            )

        else:

            raise ValueError(
                f"Invalid stance_side: "
                f"{stance_side}"
            )

        site_position_world = np.asarray(
            mj_data.site_xpos[
                site_id
            ],
            dtype=float,
        ).reshape(
            3
        )

        R_world_site = np.asarray(
            mj_data.site_xmat[
                site_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
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
                        friction_coefficient
                    ),
                )
            )

        return contacts

    # ========================================================
    # POINT TRANSLATION JACOBIAN
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

        if not np.all(
            np.isfinite(
                jacobian
            )
        ):

            raise RuntimeError(
                "Point Jacobian contains NaN/Inf."
            )

        return jacobian

    # ========================================================
    # CONTACT WRENCH MAP
    # ========================================================

    def _build_contact_wrench_maps(
        self,
        *,
        mj_data,
        contacts,
    ):

        """
        For contact point i:

            f_i = [Fx, Fy, Fz]^T

        and

            w_i = G_i f_i

        with

            G_i = J_i^T[0:6, :]
        """

        G_blocks = []

        friction_coefficients = []

        for (
            position_world,
            body_id,
            friction_coefficient,
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

            G_i = (
                J_i.T[
                    0:
                    BASE_DOF,
                    :
                ].copy()
            )

            if G_i.shape != (
                BASE_DOF,
                3,
            ):

                raise RuntimeError(
                    f"Unexpected G_i shape: "
                    f"{G_i.shape}"
                )

            G_blocks.append(
                G_i
            )

            friction_coefficients.append(
                float(
                    friction_coefficient
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
            singular_values[
                0
            ]
        )

        smallest = float(
            singular_values[
                -1
            ]
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

        For contact point i:

            z_i = G_i^T r

        where

            z_i = [z_x, z_y, z_z]^T.

        Define

            rho = sqrt(z_x^2 + z_y^2).

        Then

            h_Ui(z_i)
                =
            z_z + mu_i rho.

        For rho > 0:

            s_Ui(z_i)
                =
            [
                mu_i z_x/rho,
                mu_i z_y/rho,
                1
            ]^T.

        The corresponding primitive wrench is

            s_Wi(r)
                =
            G_i s_Ui(z_i).

        Across all contacts, choose the primitive wrench
        with the largest support value.
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

        best_primitive_wrench = None

        for (
            G_i,
            mu_i,
        ) in zip(
            G_blocks,
            friction_coefficients,
        ):

            z_i = (
                G_i.T
                @
                r
            )

            z_x = float(
                z_i[
                    0
                ]
            )

            z_y = float(
                z_i[
                    1
                ]
            )

            z_z = float(
                z_i[
                    2
                ]
            )

            rho = float(
                np.hypot(
                    z_x,
                    z_y,
                )
            )

            mu_i = float(
                mu_i
            )

            support_value = (
                z_z
                +
                mu_i
                *
                rho
            )

            if rho > 1.0e-14:

                primitive_force = np.array(
                    [
                        mu_i
                        *
                        z_x
                        /
                        rho,

                        mu_i
                        *
                        z_y
                        /
                        rho,

                        1.0,
                    ],
                    dtype=float,
                )

            else:

                # Any tangential direction gives the same
                # support value when rho = 0.

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

                best_primitive_wrench = (
                    primitive_wrench.copy()
                )

        if best_primitive_wrench is None:

            raise RuntimeError(
                "Support mapping failed."
            )

        if not np.all(
            np.isfinite(
                best_primitive_wrench
            )
        ):

            raise RuntimeError(
                "Support mapping produced NaN/Inf."
            )

        return (
            float(
                best_support_value
            ),
            best_primitive_wrench,
        )

    # ========================================================
    # DUPLICATE GENERATOR CHECK
    # ========================================================

    @staticmethod
    def _generator_is_duplicate(
        *,
        generator,
        generators,
    ) -> bool:

        generator = np.asarray(
            generator,
            dtype=float,
        )

        generator_scale = max(
            1.0,
            float(
                np.linalg.norm(
                    generator
                )
            ),
        )

        for old_generator in (
            generators
        ):

            old_generator = np.asarray(
                old_generator,
                dtype=float,
            )

            difference = float(
                np.linalg.norm(
                    generator
                    -
                    old_generator
                )
            )

            if (
                difference
                <=
                GENERATOR_TOLERANCE
                *
                generator_scale
            ):

                return True

        return False

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

        For each linearly independent face B:

            min ||B c - b||

        is solved.

        The projection is accepted only if

            c >= 0.
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

        number_generators = int(
            A.shape[
                1
            ]
        )

        # Origin always belongs to the cone.

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

        # In R^6, a linearly independent active set
        # contains at most six generators.

        maximum_subset_size = min(
            BASE_DOF,
            number_generators,
        )

        for subset_size in range(
            1,
            maximum_subset_size
            +
            1,
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

                rank_tolerance = (
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
                )

                rank = int(
                    np.linalg.matrix_rank(
                        B,
                        tol=(
                            rank_tolerance
                        ),
                    )
                )

                if rank < subset_size:

                    continue

                coefficients, *_ = (
                    np.linalg.lstsq(
                        B,
                        b,
                        rcond=None,
                    )
                )

                coefficients = np.asarray(
                    coefficients,
                    dtype=float,
                )

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

                v_candidate = (
                    B
                    @
                    coefficients
                )

                distance_candidate = float(
                    np.linalg.norm(
                        b
                        -
                        v_candidate
                    )
                )

                if (
                    distance_candidate
                    <
                    best_distance
                    -
                    PROGRESS_TOLERANCE
                ):

                    best_distance = (
                        distance_candidate
                    )

                    best_v = (
                        v_candidate.copy()
                    )

                    best_active_generators = [
                        np.asarray(
                            generators[
                                generator_index
                            ],
                            dtype=float,
                        ).copy()

                        for (
                            generator_index,
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
    # DISTANCE TO FEASIBLE WRENCH CONE
    # ========================================================

    def _distance_to_feasible_wrench_cone(
        self,
        *,
        required_wrench,
        G_blocks,
        friction_coefficients,
    ):

        """
        Zheng-Chew outer loop.

        Initialization:

            v_0 = 0

            r_0 = b

            A_0 = empty.

        Iteration:

            h_W(r_k)

            s_W(r_k)

        If

            h_W(r_k) <= epsilon

        stop.

        Otherwise add

            s_W(r_k)

        to the current generator set, compute the new
        closest point v_{k+1}, and update

            r_{k+1} = b - v_{k+1}.
        """

        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        if not np.all(
            np.isfinite(
                b
            )
        ):

            raise RuntimeError(
                "Required wrench contains NaN/Inf."
            )

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        v = np.zeros(
            BASE_DOF,
            dtype=float,
        )

        residual = (
            b.copy()
        )

        active_generators = []

        previous_distance = float(
            np.linalg.norm(
                residual
            )
        )

        final_support_value = float(
            "nan"
        )

        # Special case: b approximately zero.

        if (
            previous_distance
            <=
            DISTANCE_TOLERANCE
        ):

            return (
                previous_distance,
                v,
                residual,
                True,
                0,
                0.0,
                0,
            )

        # ----------------------------------------------------
        # ITERATION
        # ----------------------------------------------------

        for iteration in range(
            MAX_DISTANCE_ITERATIONS
        ):

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

            # =================================================
            # STOPPING CONDITION
            # =================================================

            if (
                support_value
                <=
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

            # =================================================
            # DUPLICATE GENERATOR
            # =================================================

            if (
                self._generator_is_duplicate(
                    generator=(
                        new_generator
                    ),
                    generators=(
                        active_generators
                    ),
                )
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
                    False,
                    iteration,
                    final_support_value,
                    len(
                        active_generators
                    ),
                )

            # =================================================
            # BUILD CURRENT SIMPLICIAL CONE
            # =================================================

            candidate_generators = (
                active_generators
                +
                [
                    new_generator
                ]
            )

            (
                v_new,
                active_generators_new,
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

            residual_new = (
                b
                -
                v_new
            )

            distance_new = float(
                np.linalg.norm(
                    residual_new
                )
            )

            # =================================================
            # NUMERICALLY INSIDE THE CONE
            # =================================================

            if (
                distance_new
                <=
                DISTANCE_TOLERANCE
            ):

                # Keep the actual residual norm.
                # Do not force distance to exactly zero.

                return (
                    distance_new,
                    v_new,
                    residual_new,
                    True,
                    iteration
                    +
                    1,
                    0.0,
                    len(
                        active_generators_new
                    ),
                )

            # =================================================
            # STAGNATION CHECK
            # =================================================

            improvement = (
                previous_distance
                -
                distance_new
            )

            scale = max(
                1.0,
                previous_distance,
            )

            if (
                improvement
                <=
                PROGRESS_TOLERANCE
                *
                scale
            ):

                return (
                    distance_new,
                    v_new,
                    residual_new,
                    False,
                    iteration
                    +
                    1,
                    final_support_value,
                    len(
                        active_generators_new
                    ),
                )

            # =================================================
            # NEXT ITERATION
            # =================================================

            v = (
                v_new
            )

            residual = (
                residual_new
            )

            active_generators = (
                active_generators_new
            )

            previous_distance = (
                distance_new
            )

        # ----------------------------------------------------
        # MAX ITERATION REACHED
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

        if not np.all(
            np.isfinite(
                qacc
            )
        ):

            raise RuntimeError(
                "qacc contains NaN/Inf."
            )

        # ====================================================
        # RESTORE RECORDED STATE
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
        #
        # Dynamics:
        #
        #     M qdd + qfrc_bias
        #
        #       =
        #
        #     qfrc_passive
        #       + S^T tau
        #       + J_c^T f_c
        #
        # Therefore:
        #
        #     required_generalized_force
        #
        #       =
        #
        #     M qdd
        #       + qfrc_bias
        #       - qfrc_passive
        #
        # For the unactuated floating base:
        #
        #     b
        #
        #       =
        #
        #     required_generalized_force[0:6]
        #
        # ====================================================

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

        required_wrench = (
            required_generalized_force[
                0:
                BASE_DOF
            ].copy()
        )

        # ====================================================
        # CONTACT MODE
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

        number_contacts = len(
            contacts
        )

        if (
            number_contacts
            !=
            NUMBER_CONTACT_POINTS_PER_FOOT
        ):

            raise RuntimeError(
                "Unexpected number of fixed contact points."
            )

        # ====================================================
        # CONTACT WRENCH MAPS
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

        # ====================================================
        # DISTANCE
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

            number_contacts=int(
                number_contacts
            ),

            collision_detected=bool(
                collision_detected
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
        # qdd(t_k) from qdot(t)
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
            "Converged              : "
            f"{np.count_nonzero(converged)}"
            f"/{len(results)}"
        )

        print(
            "Distance-zero feasible : "
            f"{np.count_nonzero(feasible)}"
            f"/{len(results)}"
        )

        print(
            "Stance collision       : "
            f"{np.count_nonzero(collision)}"
            f"/{len(results)}"
        )

        print(
            "Min / max rank(G)      : "
            f"{np.min(ranks)} / "
            f"{np.max(ranks)}"
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

    figure, axis = (
        plt.subplots()
    )

    axis.plot(
        time,
        distance,
        linewidth=1.5,
        label=(
            "Wrench distance"
        ),
    )

    axis.axhline(
        DISTANCE_TOLERANCE,
        linestyle="--",
        linewidth=1.0,
        label=(
            "Distance tolerance"
        ),
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
            label=(
                "Not converged"
            ),
        )

    axis.set_xlabel(
        "Time [s]"
    )

    axis.set_ylabel(
        "Euclidean wrench distance"
    )

    axis.set_title(
        "Distance to feasible "
        "contact-wrench cone"
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