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

# Software tolerance used only to classify d ~= 0.
DISTANCE_TOLERANCE = 1.0e-6

# Numerical tolerances for the paper subalgorithm.
COEFFICIENT_TOLERANCE = 1.0e-10
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

LEFT_FOOT_GEOM_NAME = (
    "left_foot_bottom_tpu"
)

RIGHT_FOOT_GEOM_NAME = (
    "right_foot_bottom_tpu"
)

LEFT_FOOT_SITE_NAME = "left_foot"
RIGHT_FOOT_SITE_NAME = "right_foot"

FLOATING_BASE_JOINT_NAME = (
    "floating_base"
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class TrajectorySample:

    time: float

    qpos: np.ndarray
    qvel: np.ndarray

    # Logical stance from gait planner.
    stance_side: str


@dataclass(frozen=True)
class WrenchDistanceResult:

    time: float

    # Logical gait state.
    stance_side: str

    # Actual support mode from MuJoCo.
    contact_mode: str

    left_collision_detected: bool
    right_collision_detected: bool

    collision_detected: bool

    # 4 for SS, 8 for DS, 0 for no support.
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


# ============================================================
# CONTACT WRENCH DISTANCE EVALUATOR
# ============================================================

class ContactWrenchDistanceEvaluator:

    """
    Distance from the required floating-base generalized wrench
    to the feasible contact-wrench cone.

    Contact mode is determined using MuJoCo collision:

        left only
            -> left single support
            -> 4 fixed contact points

        right only
            -> right single support
            -> 4 fixed contact points

        both feet
            -> double support
            -> 8 fixed contact points

        neither foot
            -> no support
            -> V = {0}

    MuJoCo collision determines only which foot is active.

    The actual MuJoCo contact positions are NOT used to build
    the feasible wrench cone.

    Each active foot still uses the four predefined fixed
    contact points.

    The contact-wrench cone and distance algorithm follow
    Zheng & Chew (2009).
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        *,
        mj_model,
    ) -> None:

        self.mj_model = (
            mj_model
        )

        self.samples: list[
            TrajectorySample
        ] = []

        # ----------------------------------------------------
        # MuJoCo object IDs
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Floating base
        # ----------------------------------------------------

        floating_base_joint_id = (
            self._require_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                FLOATING_BASE_JOINT_NAME,
            )
        )

        if (
            int(
                self.mj_model.jnt_type[
                    floating_base_joint_id
                ]
            )
            !=
            int(
                mujoco.mjtJoint.mjJNT_FREE
            )
        ):

            raise RuntimeError(
                f"Joint "
                f"'{FLOATING_BASE_JOINT_NAME}' "
                f"is not a free joint."
            )

        if (
            int(
                self.mj_model.jnt_dofadr[
                    floating_base_joint_id
                ]
            )
            !=
            0
        ):

            raise RuntimeError(
                "This implementation assumes that "
                "the first six qvel coordinates "
                "belong to the floating base."
            )

        if (
            self.mj_model.nv
            <
            BASE_DOF
        ):

            raise RuntimeError(
                "MuJoCo model has fewer than "
                "six generalized DoFs."
            )

        # ----------------------------------------------------
        # Friction coefficients
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        contact_length_mm = (
            FOOT_CONTACT_X_MAX
            -
            FOOT_CONTACT_X_MIN
        ) * 1000.0

        contact_width_mm = (
            FOOT_CONTACT_Y_MAX
            -
            FOOT_CONTACT_Y_MIN
        ) * 1000.0

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
            f"{contact_length_mm:.2f} x "
            f"{contact_width_mm:.2f} mm"
        )

        print(
            "Contact points / active foot: "
            f"{NUMBER_CONTACT_POINTS_PER_FOOT}"
        )

        print(
            "Left friction coefficient : "
            f"{self.left_mu:.4f}"
        )

        print(
            "Right friction coefficient: "
            f"{self.right_mu:.4f}"
        )

        print(
            "Contact mode source: "
            "MuJoCo foot-floor collision detection"
        )

        print(
            "SS contact model: "
            "4 fixed points"
        )

        print(
            "DS contact model: "
            "8 fixed points "
            "(4 left + 4 right)"
        )

        print(
            "MuJoCo contact locations are not "
            "used for the wrench cone."
        )

        print(
            "================================================"
        )

        print()

    # ========================================================
    # BASIC MODEL HELPERS
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

        if (
            object_id
            <
            0
        ):

            raise RuntimeError(
                f"MuJoCo object "
                f"'{name}' "
                f"was not found."
            )

        return object_id

    # ========================================================
    # RECORD SAMPLE
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

        if (
            stance_side
            not in
            (
                "left",
                "right",
            )
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
            self.samples
            and
            sample_time
            <=
            self.samples[-1].time
        ):

            raise ValueError(
                "Sample time must be "
                "strictly increasing."
            )

        qpos = np.asarray(
            mj_data.qpos,
            dtype=float,
        ).copy()

        qvel = np.asarray(
            mj_data.qvel,
            dtype=float,
        ).copy()

        if (
            qpos.shape
            !=
            (
                self.mj_model.nq,
            )
        ):

            raise RuntimeError(
                "Unexpected qpos shape."
            )

        if (
            qvel.shape
            !=
            (
                self.mj_model.nv,
            )
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
    # FULL MASS MATRIX
    # ========================================================

    def _full_mass_matrix(
        self,
        *,
        mj_data,
    ) -> np.ndarray:

        """
        Return dense M(q), supporting both the
        new and old MuJoCo Python APIs.
        """

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
                    "new API failed and mjData.qM "
                    "is unavailable."
                )

            mujoco.mj_fullM(
                self.mj_model,
                mass_matrix,
                mj_data.qM,
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
    # MUJOCO COLLISION DETECTION
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

            if (
                geom.size
                >=
                2
            ):

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

    def _foot_collision_detected(
        self,
        *,
        mj_data,
        foot_geom_id,
    ) -> bool:

        """
        Return True if the specified foot geom has
        an active MuJoCo contact pair with the floor.
        """

        foot_geom_id = int(
            foot_geom_id
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

            (
                geom_0,
                geom_1,
            ) = (
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
                in
                pair
                and
                foot_geom_id
                in
                pair
            ):

                # Same criterion used previously
                # for collision diagnostics.
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

    def _detect_contact_mode(
        self,
        *,
        mj_data,
    ) -> tuple[
        bool,
        bool,
        str,
    ]:

        left_contact = (
            self._foot_collision_detected(
                mj_data=mj_data,
                foot_geom_id=(
                    self.left_foot_geom_id
                ),
            )
        )

        right_contact = (
            self._foot_collision_detected(
                mj_data=mj_data,
                foot_geom_id=(
                    self.right_foot_geom_id
                ),
            )
        )

        if (
            left_contact
            and
            right_contact
        ):

            contact_mode = (
                "double_support"
            )

        elif left_contact:

            contact_mode = (
                "left_single_support"
            )

        elif right_contact:

            contact_mode = (
                "right_single_support"
            )

        else:

            contact_mode = (
                "no_support"
            )

        return (
            bool(
                left_contact
            ),
            bool(
                right_contact
            ),
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
    ):

        """
        Return the four predefined contact points
        for one foot.

        MuJoCo collision determines whether this
        foot is active.

        The actual collision locations are not used.
        """

        side = str(
            side
        ).lower()

        if (
            side
            ==
            "left"
        ):

            site_id = (
                self.left_foot_site_id
            )

            body_id = (
                self.left_foot_body_id
            )

            mu = (
                self.left_mu
            )

        elif (
            side
            ==
            "right"
        ):

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
                f"Invalid foot side: "
                f"{side}"
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
                        mu
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
    ):

        contacts = []

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
    # CONTACT WRENCH MAPS
    # ========================================================

    def _build_contact_wrench_maps(
        self,
        *,
        mj_data,
        contacts,
    ):

        """
        For every fixed contact point i:

            w_i = G_i f_i

            G_i = J_i^T[0:6, :]
        """

        G_blocks = []

        friction_coefficients = []

        for (
            position_world,
            body_id,
            mu,
        ) in contacts:

            J_i = (
                self._point_translation_jacobian(
                    mj_data=mj_data,
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
                    0:BASE_DOF,
                    :
                ].copy()
            )

            if (
                G_i.shape
                !=
                (
                    BASE_DOF,
                    3,
                )
            ):

                raise RuntimeError(
                    "Unexpected G_i shape: "
                    f"{G_i.shape}"
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
    # MATRIX CONDITION NUMBER
    # ========================================================

    @staticmethod
    def _matrix_condition_number(
        matrix,
    ) -> float:

        singular_values = (
            np.linalg.svd(
                np.asarray(
                    matrix,
                    dtype=float,
                ),
                compute_uv=False,
            )
        )

        if (
            singular_values.size
            ==
            0
        ):

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
        PCwF support mapping from Eq. (40)-(44).

        Paper ordering:

            [normal, tangent1, tangent2]

        Current world-force ordering:

            [Fx, Fy, Fz]

        Therefore:

            normal = Fz.
        """

        r = np.asarray(
            residual,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        best_support_value = (
            -float(
                "inf"
            )
        )

        best_primitive_wrench = (
            None
        )

        for (
            G_i,
            mu_i,
        ) in zip(
            G_blocks,
            friction_coefficients,
        ):

            # -----------------------------------------------
            # u_i = G_i^T r
            # -----------------------------------------------

            u_i = (
                G_i.T
                @
                r
            )

            u_t1 = float(
                u_i[
                    0
                ]
            )

            u_t2 = float(
                u_i[
                    1
                ]
            )

            u_n = float(
                u_i[
                    2
                ]
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

            if (
                h_t
                !=
                0.0
            ):

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

                # Paper:
                #
                # if h_Ti = 0,
                # any point of U_i may be used.

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

        if (
            best_primitive_wrench
            is
            None
        ):

            raise RuntimeError(
                "Support mapping failed."
            )

        return (
            best_support_value,
            best_primitive_wrench,
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

        """
        c_A(b) = (A^T A)^(-1) A^T b

        p_A(b) = A c_A(b)

        r_A(b) = b - p_A(b)
        """

        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        if (
            len(
                generators
            )
            ==
            0
        ):

            raise ValueError(
                "Projection set must not be empty."
            )

        A = np.column_stack(
            [
                np.asarray(
                    a,
                    dtype=float,
                ).reshape(
                    BASE_DOF
                )
                for a
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

            coefficients = (
                np.linalg.solve(
                    gram,
                    rhs,
                )
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

    # ========================================================
    # COEFFICIENT SETS
    # ========================================================

    @staticmethod
    def _coefficient_sets(
        coefficients,
    ):

        c = np.asarray(
            coefficients,
            dtype=float,
        ).reshape(
            -1
        )

        negative = [
            i
            for i, value
            in enumerate(
                c
            )
            if (
                value
                <
                -COEFFICIENT_TOLERANCE
            )
        ]

        zero = [
            i
            for i, value
            in enumerate(
                c
            )
            if (
                abs(
                    value
                )
                <=
                COEFFICIENT_TOLERANCE
            )
        ]

        positive = [
            i
            for i, value
            in enumerate(
                c
            )
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
        Paper subalgorithm Steps 1-5.

        combinations() is used only for the
        candidate-subset search prescribed by the paper.
        """

        b = np.asarray(
            required_wrench,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        new_generator = np.asarray(
            new_generator,
            dtype=float,
        ).reshape(
            BASE_DOF
        )

        working = []

        for (
            index,
            generator,
        ) in enumerate(
            A_k_plus_1
        ):

            working.append(
                {
                    "generator":
                        np.asarray(
                            generator,
                            dtype=float,
                        ).reshape(
                            BASE_DOF
                        ).copy(),

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
            working[
                -1
            ][
                "generator"
            ],
            new_generator,
            rtol=0.0,
            atol=(
                COEFFICIENT_TOLERANCE
            ),
        ):

            raise RuntimeError(
                "The last generator of "
                "A_{k+1} must be s_A(r_k)."
            )

        # ----------------------------------------------------
        # Step 1 / Step 2
        # ----------------------------------------------------

        while True:

            generators = [
                item[
                    "generator"
                ]
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
                        generators
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

            # ------------------------------------------------
            # Step 1:
            #
            # A- is empty.
            # ------------------------------------------------

            if (
                len(
                    negative
                )
                ==
                0
            ):

                A_hat = [
                    working[
                        i
                    ][
                        "generator"
                    ].copy()
                    for i
                    in positive
                ]

                return (
                    projection,
                    A_hat,
                )

            # ------------------------------------------------
            # Step 2:
            #
            # A- contains one point.
            # ------------------------------------------------

            if (
                len(
                    negative
                )
                ==
                1
            ):

                remove_index = (
                    negative[
                        0
                    ]
                )

                if (
                    working[
                        remove_index
                    ][
                        "is_new"
                    ]
                ):

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

        # ----------------------------------------------------
        # Current A- after singleton removals
        # ----------------------------------------------------

        generators = [
            item[
                "generator"
            ]
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
                    generators
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
            if (
                item[
                    "is_new"
                ]
            )
        ]

        if (
            len(
                new_indices
            )
            !=
            1
        ):

            raise RuntimeError(
                "Could not uniquely identify "
                "s_A(r_k)."
            )

        new_index = (
            new_indices[
                0
            ]
        )

        number_points = len(
            working
        )

        # ----------------------------------------------------
        # Steps 3-5
        # ----------------------------------------------------

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

                # --------------------------------------------
                # Theorem 5(1)
                # --------------------------------------------

                if (
                    new_index
                    not in
                    subset
                ):

                    continue

                # --------------------------------------------
                # Theorem 5(2)
                # --------------------------------------------

                if (
                    negative_set.issubset(
                        subset
                    )
                ):

                    continue

                candidate_indices = sorted(
                    subset
                )

                candidate_generators = [
                    working[
                        i
                    ][
                        "generator"
                    ]
                    for i
                    in candidate_indices
                ]

                (
                    candidate_coefficients,
                    candidate_projection,
                    candidate_residual,
                ) = (
                    self._projection_quantities(
                        required_wrench=b,
                        generators=(
                            candidate_generators
                        ),
                    )
                )

                # --------------------------------------------
                # Theorem 4 condition 1
                # --------------------------------------------

                if not np.all(
                    candidate_coefficients
                    >
                    COEFFICIENT_TOLERANCE
                ):

                    continue

                # --------------------------------------------
                # Theorem 4 condition 2
                # --------------------------------------------

                condition_2_satisfied = (
                    True
                )

                for (
                    index,
                    item,
                ) in enumerate(
                    working
                ):

                    if (
                        index
                        in
                        subset
                    ):

                        continue

                    value = float(
                        item[
                            "generator"
                        ]
                        @
                        candidate_residual
                    )

                    if (
                        value
                        >
                        SUPPORT_PLANE_TOLERANCE
                    ):

                        condition_2_satisfied = (
                            False
                        )

                        break

                if not condition_2_satisfied:

                    continue

                A_hat = [
                    a.copy()
                    for a
                    in candidate_generators
                ]

                return (
                    candidate_projection,
                    A_hat,
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
    ):

        """
        Paper Steps 1-5:

        Step 1:
            v0 = 0
            r0 = b
            A0 = empty
            Ahat0 = empty
            k = 0

        Step 2:
            if h_W(r_k) < epsilon:
                stop

        Step 3:
            A_{k+1}
            =
            Ahat_k union {s_W(r_k)}

        Step 4:
            obtain v_{k+1}
            and Ahat_{k+1}

        Step 5:
            r_{k+1}
            =
            b - v_{k+1}
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
                "Required wrench contains "
                "NaN/Inf."
            )

        if (
            len(
                G_blocks
            )
            ==
            0
        ):

            raise ValueError(
                "Distance algorithm requires "
                "at least one active contact point."
            )

        # ----------------------------------------------------
        # Step 1
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Main loop
        # ----------------------------------------------------

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
                )
            )

            final_support_value = float(
                support_value
            )

            # ------------------------------------------------
            # Step 2
            # ------------------------------------------------

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
                    len(
                        A_hat_k
                    ),
                )

            # ------------------------------------------------
            # Step 3
            # ------------------------------------------------

            A_k_plus_1 = [
                a.copy()
                for a
                in A_hat_k
            ]

            A_k_plus_1.append(
                support_generator.copy()
            )

            # ------------------------------------------------
            # Step 4
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Step 5
            # ------------------------------------------------

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

        # ----------------------------------------------------
        # Software guard only.
        # ----------------------------------------------------

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
            len(
                A_hat_k
            ),
        )

    # ========================================================
    # NO SUPPORT SPECIAL CASE
    # ========================================================

    @staticmethod
    def _distance_without_support(
        *,
        required_wrench,
    ):

        """
        If neither foot is active:

            V = {0}

        therefore:

            v* = 0

            d = ||b||
        """

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
            float(
                "nan"
            ),
            0,
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

        # ----------------------------------------------------
        # Restore trajectory state
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

        scratch_data.qfrc_applied[:] = (
            0.0
        )

        scratch_data.xfrc_applied[:] = (
            0.0
        )

        # ----------------------------------------------------
        # Recompute MuJoCo state, dynamics and contacts
        # ----------------------------------------------------

        mujoco.mj_forward(
            self.mj_model,
            scratch_data,
        )

        # ----------------------------------------------------
        # Required generalized force
        #
        # M qdd + qfrc_bias
        # =
        # qfrc_passive
        # + S^T tau
        # + J_c^T f_c
        #
        # required
        # =
        # M qdd
        # + qfrc_bias
        # - qfrc_passive
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

        required_wrench = (
            required_generalized_force[
                0:BASE_DOF
            ].copy()
        )

        # ----------------------------------------------------
        # ACTUAL SUPPORT MODE FROM MUJOCO
        # ----------------------------------------------------

        (
            left_collision_detected,
            right_collision_detected,
            contact_mode,
        ) = (
            self._detect_contact_mode(
                mj_data=(
                    scratch_data
                )
            )
        )

        collision_detected = bool(
            left_collision_detected
            or
            right_collision_detected
        )

        # ----------------------------------------------------
        # BUILD ACTIVE FIXED CONTACT SET
        #
        # Left SS  -> 4 points
        # Right SS -> 4 points
        # DS       -> 8 points
        # ----------------------------------------------------

        contacts = (
            self._active_fixed_contacts(
                mj_data=(
                    scratch_data
                ),
                left_contact=(
                    left_collision_detected
                ),
                right_contact=(
                    right_collision_detected
                ),
            )
        )

        number_contacts = len(
            contacts
        )

        if (
            number_contacts
            not in
            (
                0,
                NUMBER_CONTACT_POINTS_PER_FOOT,
                2
                *
                NUMBER_CONTACT_POINTS_PER_FOOT,
            )
        ):

            raise RuntimeError(
                "Unexpected number of "
                "fixed contact points: "
                f"{number_contacts}"
            )

        # ----------------------------------------------------
        # NO SUPPORT
        # ----------------------------------------------------

        if (
            number_contacts
            ==
            0
        ):

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
                active_generator_count,
            ) = (
                self._distance_without_support(
                    required_wrench=(
                        required_wrench
                    )
                )
            )

        # ----------------------------------------------------
        # SS OR DS
        # ----------------------------------------------------

        else:

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

        # ----------------------------------------------------
        # FEASIBILITY DECISION
        # ----------------------------------------------------

        feasible = bool(
            converged
            and
            distance
            <=
            DISTANCE_TOLERANCE
        )

        return (
            WrenchDistanceResult(
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

        if (
            number_samples
            <
            3
        ):

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
                "Sample time must be "
                "strictly increasing."
            )

        # ----------------------------------------------------
        # Reconstruct qdd from qdot
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Offline evaluation
        # ----------------------------------------------------

        scratch_data = mujoco.MjData(
            self.mj_model
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

        contact_modes = [
            result.contact_mode
            for result
            in results
        ]

        left_single_support_count = (
            contact_modes.count(
                "left_single_support"
            )
        )

        right_single_support_count = (
            contact_modes.count(
                "right_single_support"
            )
        )

        double_support_count = (
            contact_modes.count(
                "double_support"
            )
        )

        no_support_count = (
            contact_modes.count(
                "no_support"
            )
        )

        # ----------------------------------------------------
        # Compare planner stance with actual contact
        # ----------------------------------------------------

        gait_stance_contact_count = 0

        for result in results:

            if (
                result.stance_side
                ==
                "left"
                and
                result.left_collision_detected
            ):

                gait_stance_contact_count += 1

            elif (
                result.stance_side
                ==
                "right"
                and
                result.right_collision_detected
            ):

                gait_stance_contact_count += 1

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

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
            "Samples                : "
            f"{len(results)}"
        )

        print(
            "Converged              : "
            f"{np.count_nonzero(converged)}"
            f"/"
            f"{len(results)}"
        )

        print(
            "Distance-zero feasible : "
            f"{np.count_nonzero(feasible)}"
            f"/"
            f"{len(results)}"
        )

        print(
            "Any foot collision     : "
            f"{np.count_nonzero(collision)}"
            f"/"
            f"{len(results)}"
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
            f"{gait_stance_contact_count}"
            f"/"
            f"{len(results)}"
        )

        print(
            "Min / max rank(G)      : "
            f"{np.min(ranks)}"
            f" / "
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

    # --------------------------------------------------------
    # Main distance curve
    # --------------------------------------------------------

    figure, axis = (
        plt.subplots()
    )

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

    # --------------------------------------------------------
    # Non-converged samples
    # --------------------------------------------------------

    not_converged = (
        np.logical_not(
            converged
        )
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

    # --------------------------------------------------------
    # Mark DS samples
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Figure settings
    # --------------------------------------------------------

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