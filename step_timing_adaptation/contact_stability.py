# step_timing_adaptation/contact_stability.py

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import casadi as ca
import mujoco
import numpy as np


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

CONTACT_DISTANCE_TOLERANCE = 1.0e-10

MATRIX_RCOND = 1.0e-10

# Equality-only diagnostic:
EQUALITY_LS_TOLERANCE = 1.0e-6

# Returned hard-QP solution validation:
EQUALITY_TOLERANCE = 1.0e-5
INEQUALITY_TOLERANCE = 1.0e-7

# Soft diagnostic thresholds:
UNILATERAL_SLACK_TOLERANCE = 1.0e-6
FRICTION_SLACK_TOLERANCE = 1.0e-6

NORMAL_FORCE_TOLERANCE = 1.0e-9


# ============================================================
# MUJOCO GEOM NAMES
# ============================================================

FLOOR_GEOM_NAME = "floor"

LEFT_FOOT_GEOM_NAME = "left_foot_bottom_tpu"
RIGHT_FOOT_GEOM_NAME = "right_foot_bottom_tpu"


# ============================================================
# CLASSIFICATION LABELS
# ============================================================

CLASS_FEASIBLE = "FEASIBLE"

CLASS_EQUALITY_INFEASIBLE = (
    "EQUALITY_INFEASIBLE"
)

CLASS_UNILATERAL_INFEASIBLE = (
    "UNILATERAL_INFEASIBLE"
)

CLASS_FRICTION_INFEASIBLE = (
    "FRICTION_INFEASIBLE"
)

CLASS_SOLVER_FAILURE = (
    "SOLVER_FAILURE"
)

CLASS_NO_CONTACT = (
    "NO_FOOT_GROUND_CONTACT"
)

CLASS_BOUNDARY_SKIPPED = (
    "DERIVATIVE_BOUNDARY_SKIPPED"
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class FootGroundContact:
    """
    One foot-ground point contact extracted from MuJoCo.

    contact_frame_world stores three ROW vectors:

        row 0 : contact normal, oriented ground -> foot
        row 1 : tangent direction 1
        row 2 : tangent direction 2

    Therefore:

        f_contact = contact_frame_world @ f_world

    and:

        f_contact = [f_n, f_t1, f_t2]^T
    """

    foot_side: str

    position_world: np.ndarray
    contact_frame_world: np.ndarray

    friction_tangent_1: float
    friction_tangent_2: float

    distance: float


@dataclass(frozen=True)
class ContactKinematicsSample:
    """
    Raw data recorded from one trajectory sample.
    """

    time: float

    com_position_world: np.ndarray
    com_velocity_world: np.ndarray

    angular_momentum_world: np.ndarray

    left_foot_position_world: np.ndarray
    right_foot_position_world: np.ndarray

    contacts: tuple[FootGroundContact, ...]


@dataclass(frozen=True)
class ContactForceSolution:
    """
    Optimal point-contact force returned by the full feasible QP.
    """

    foot_side: str

    position_world: np.ndarray

    force_world: np.ndarray

    # [f_n, f_t1, f_t2]
    force_contact: np.ndarray

    friction_tangent_1: float
    friction_tangent_2: float

    friction_utilization: float


@dataclass(frozen=True)
class ContactStabilityResult:
    """
    Contact-stability diagnostic result at one trajectory sample.
    """

    time: float

    classification: str

    feasible: bool
    solver_status: str

    number_contacts: int

    # Equality-only diagnostic:
    equality_rank: int
    equality_ls_residual: float

    # Conditioning of the centroidal equality map A_eq.
    equality_sigma_min: float
    equality_sigma_max: float
    equality_condition_number: float

    # Contact geometry diagnostics.
    left_contact_count: int
    right_contact_count: int

    # Maximum pairwise distance between contact points belonging
    # to the SAME foot. Units: meters.
    left_contact_span: float
    right_contact_span: float
    max_foot_contact_span: float

    # Full-body derivatives:
    com_acceleration_world: np.ndarray
    angular_momentum_rate_world: np.ndarray

    # Required resultant ground reaction force:
    required_contact_force_world: np.ndarray

    # Full feasible-QP numerical residuals:
    equality_residual: float
    inequality_violation: float

    # Diagnostic slack values:
    max_unilateral_slack: float
    max_friction_slack: float

    # Meaningful for FEASIBLE samples:
    max_friction_utilization: float

    # Resultant foot wrenches:
    left_resultant_force_world: np.ndarray
    left_resultant_moment_world: np.ndarray

    right_resultant_force_world: np.ndarray
    right_resultant_moment_world: np.ndarray

    contact_forces: tuple[ContactForceSolution, ...]


# ============================================================
# BASIC LINEAR-ALGEBRA HELPERS
# ============================================================

def skew(
    vector,
) -> np.ndarray:
    """
    Return [r]_x such that:

        [r]_x f = r x f
    """

    r = np.asarray(
        vector,
        dtype=float,
    ).reshape(
        3
    )

    return np.array(
        [
            [
                0.0,
                -r[2],
                +r[1],
            ],
            [
                +r[2],
                0.0,
                -r[0],
            ],
            [
                -r[1],
                +r[0],
                0.0,
            ],
        ],
        dtype=float,
    )


def finite_vector3(
    value,
    *,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        value,
        dtype=float,
    ).reshape(
        3
    )

    if not np.all(
        np.isfinite(
            array
        )
    ):

        raise ValueError(
            f"{name} contains NaN/Inf."
        )

    return array.copy()


def numerical_rank(
    matrix,
) -> int:

    A = np.asarray(
        matrix,
        dtype=float,
    )

    if A.size == 0:
        return 0

    singular_values = np.linalg.svd(
        A,
        compute_uv=False,
    )

    if singular_values.size == 0:
        return 0

    tolerance = (
        MATRIX_RCOND
        *
        max(
            A.shape
        )
        *
        singular_values[0]
    )

    return int(
        np.sum(
            singular_values
            >
            tolerance
        )
    )


def matrix_condition_diagnostics(
    matrix,
):
    """
    Return:

        sigma_min,
        sigma_max,
        condition_number

    for the supplied matrix.

    Full rank alone is not enough to guarantee that the centroidal
    wrench map is numerically/geometrically well conditioned.
    """

    A = np.asarray(
        matrix,
        dtype=float,
    )

    if A.size == 0:

        return (
            0.0,
            0.0,
            float(
                "inf"
            ),
        )

    singular_values = np.linalg.svd(
        A,
        compute_uv=False,
    )

    if singular_values.size == 0:

        return (
            0.0,
            0.0,
            float(
                "inf"
            ),
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
        0.0
    ):

        condition_number = float(
            "inf"
        )

    else:

        condition_number = float(
            sigma_max
            /
            sigma_min
        )

    return (
        sigma_min,
        sigma_max,
        condition_number,
    )


def contact_patch_span(
    contacts,
    *,
    foot_side: str,
) -> tuple[int, float]:
    """
    Return:

        contact count on selected foot,
        maximum pairwise distance between its contact points [m].

    Contacts from the opposite foot are excluded. This is important
    during any temporary double-support phase because the distance
    between the two feet is not the contact-patch size of one foot.
    """

    positions = [
        np.asarray(
            contact.position_world,
            dtype=float,
        ).reshape(
            3
        )
        for contact
        in contacts
        if contact.foot_side
        ==
        foot_side
    ]

    count = len(
        positions
    )

    if (
        count
        <=
        1
    ):

        return (
            count,
            0.0,
        )

    max_distance = 0.0

    for i in range(
        count
    ):

        for j in range(
            i + 1,
            count,
        ):

            distance = float(
                np.linalg.norm(
                    positions[i]
                    -
                    positions[j]
                )
            )

            max_distance = max(
                max_distance,
                distance,
            )

    return (
        count,
        float(
            max_distance
        ),
    )


def sample_contact_geometry_diagnostics(
    sample,
):
    """
    Rebuild the 6 x (3*Nc) centroidal wrench mapping from a recorded
    sample and report geometry/conditioning metrics.

    This function does NOT solve a QP and therefore does not alter
    feasibility classification.
    """

    contacts = (
        sample.contacts
    )

    number_contacts = len(
        contacts
    )

    A_equal = np.zeros(
        (
            6,
            3
            *
            number_contacts,
        ),
        dtype=float,
    )

    for contact_index, contact in enumerate(
        contacts
    ):

        column = (
            3
            *
            contact_index
        )

        A_equal[
            0:3,
            column:
            column + 3,
        ] = np.eye(
            3,
            dtype=float,
        )

        lever_arm = (
            contact.position_world
            -
            sample.com_position_world
        )

        A_equal[
            3:6,
            column:
            column + 3,
        ] = skew(
            lever_arm
        )

    rank = numerical_rank(
        A_equal
    )

    (
        sigma_min,
        sigma_max,
        condition_number,
    ) = (
        matrix_condition_diagnostics(
            A_equal
        )
    )

    (
        left_contact_count,
        left_contact_span,
    ) = (
        contact_patch_span(
            contacts,
            foot_side="left",
        )
    )

    (
        right_contact_count,
        right_contact_span,
    ) = (
        contact_patch_span(
            contacts,
            foot_side="right",
        )
    )

    max_foot_contact_span = float(
        max(
            left_contact_span,
            right_contact_span,
        )
    )

    return (
        rank,
        sigma_min,
        sigma_max,
        condition_number,
        left_contact_count,
        right_contact_count,
        left_contact_span,
        right_contact_span,
        max_foot_contact_span,
    )


# ============================================================
# CONTACT STABILITY CHECKER
# ============================================================

class ContactStabilityChecker:
    """
    Weak contact-stability checker for the current Open Duck Mini
    full-body trajectory.

    ------------------------------------------------------------
    CURRENT ASSUMPTIONS
    ------------------------------------------------------------

    1. Joint-torque limits are ignored.

    2. Only foot-ground collision geoms are enabled.

    3. MuJoCo supplies:
           - whole-body CoM position/velocity,
           - centroidal angular momentum,
           - contact positions,
           - contact frames,
           - contact friction coefficients.

    4. MuJoCo contact forces are NOT used as the feasibility proof.

    5. Every MuJoCo foot-ground contact is modeled as a 3-D
       point force.

    6. Coulomb friction is approximated by a friction pyramid.

    ------------------------------------------------------------
    CENTROIDAL DYNAMICS
    ------------------------------------------------------------

        sum_i f_i
            = m (a_G - g)

        sum_i (p_Ci - p_G) x f_i
            = dL_G / dt

    ------------------------------------------------------------
    CONTACT CONSTRAINTS
    ------------------------------------------------------------

    Unilateral:

        f_n >= 0

    Tangential friction pyramid:

        |f_t1| <= mu_1 f_n
        |f_t2| <= mu_2 f_n

    ------------------------------------------------------------
    DIAGNOSTIC SEQUENCE
    ------------------------------------------------------------

    Layer 1:
        Equality only.

        Can the current contact geometry generate the required
        centroidal wrench if contact forces are otherwise arbitrary?

    Layer 2:
        Equality + unilateral contact.

        Can the required wrench be generated without any contact
        point pulling the ground?

    Layer 3:
        Equality + unilateral + tangential friction pyramid.

        Can the required wrench also be generated without exceeding
        the available friction?

    Therefore the classification has a physical hierarchy:

        EQUALITY_INFEASIBLE
            -> current contact geometry cannot generate the wrench.

        UNILATERAL_INFEASIBLE
            -> geometry can generate the wrench, but it would require
               at least one negative normal force.

        FRICTION_INFEASIBLE
            -> equality and nonnegative normal forces are possible,
               but tangential friction limits are exceeded.

        FEASIBLE
            -> all constraints are satisfied.

        SOLVER_FAILURE
            -> numerical solver did not allow a reliable physical
               classification.
    """

    def __init__(
        self,
        *,
        mj_model,
    ) -> None:

        self.mj_model = (
            mj_model
        )

        # ====================================================
        # MASS / GRAVITY FROM MUJOCO
        # ====================================================

        self.robot_mass = float(
            np.sum(
                mj_model.body_mass
            )
        )

        if (
            not np.isfinite(
                self.robot_mass
            )
            or
            self.robot_mass
            <=
            0.0
        ):

            raise RuntimeError(
                "Invalid robot mass in MuJoCo model."
            )

        self.gravity_world = np.asarray(
            mj_model.opt.gravity,
            dtype=float,
        ).reshape(
            3
        ).copy()

        if not np.all(
            np.isfinite(
                self.gravity_world
            )
        ):

            raise RuntimeError(
                "MuJoCo gravity contains NaN/Inf."
            )

        # ====================================================
        # COLLISION GEOM IDS
        # ====================================================

        self.floor_geom_id = (
            self._require_geom_id(
                FLOOR_GEOM_NAME
            )
        )

        self.left_foot_geom_id = (
            self._require_geom_id(
                LEFT_FOOT_GEOM_NAME
            )
        )

        self.right_foot_geom_id = (
            self._require_geom_id(
                RIGHT_FOOT_GEOM_NAME
            )
        )

        # ====================================================
        # RAW TRAJECTORY HISTORY
        # ====================================================

        self.samples: list[
            ContactKinematicsSample
        ] = []

        # ====================================================
        # CASADI QP SOLVER CACHE
        # ====================================================

        self._solver_cache = {}


    # ========================================================
    # MUJOCO GEOM LOOKUP
    # ========================================================

    def _require_geom_id(
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
                f"MuJoCo geom '{geom_name}' "
                "was not found."
            )

        return geom_id


    # ========================================================
    # CONTACT GEOM IDS
    # ========================================================

    @staticmethod
    def _get_contact_geom_ids(
        contact,
    ) -> tuple[int, int]:
        """
        Support both:
            contact.geom[0:2]
        and:
            contact.geom1/contact.geom2
        depending on MuJoCo Python binding version.
        """

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


    # ========================================================
    # FOOT-GROUND CONTACT EXTRACTION
    # ========================================================

    def _extract_foot_ground_contacts(
        self,
        *,
        mj_data,
    ) -> tuple[FootGroundContact, ...]:

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
                self._get_contact_geom_ids(
                    contact
                )
            )

            geom_pair = {
                geom_0,
                geom_1,
            }

            # Ground must participate.
            if (
                self.floor_geom_id
                not in
                geom_pair
            ):

                continue

            # Identify which foot participates.
            if (
                self.left_foot_geom_id
                in
                geom_pair
            ):

                foot_side = "left"

                foot_geom_id = (
                    self.left_foot_geom_id
                )

            elif (
                self.right_foot_geom_id
                in
                geom_pair
            ):

                foot_side = "right"

                foot_geom_id = (
                    self.right_foot_geom_id
                )

            else:

                continue

            # Only force-generating contacts.
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

            if (
                int(
                    contact.dim
                )
                <
                3
            ):

                raise RuntimeError(
                    "Foot-ground contact has condim < 3. "
                    "The checker expects a 3-D frictional "
                    "point contact."
                )

            # ------------------------------------------------
            # MuJoCo contact frame
            #
            # row 0 : normal, geom[0] -> geom[1]
            # row 1 : tangent 1
            # row 2 : tangent 2
            # ------------------------------------------------

            frame = np.asarray(
                contact.frame,
                dtype=float,
            ).reshape(
                3,
                3,
            ).copy()

            normal = (
                frame[
                    0
                ].copy()
            )

            # Orient normal consistently:
            #
            #     ground -> foot
            #
            if (
                geom_0
                ==
                self.floor_geom_id
                and
                geom_1
                ==
                foot_geom_id
            ):

                normal_ground_to_foot = (
                    normal
                )

            elif (
                geom_0
                ==
                foot_geom_id
                and
                geom_1
                ==
                self.floor_geom_id
            ):

                normal_ground_to_foot = (
                    -normal
                )

            else:

                continue

            # Tangential inequalities are symmetric in sign, so
            # frame[1] and frame[2] can be used unchanged.
            contact_frame_world = np.vstack(
                (
                    normal_ground_to_foot,
                    frame[1],
                    frame[2],
                )
            )

            friction = np.asarray(
                contact.friction,
                dtype=float,
            ).reshape(
                -1
            )

            if friction.size < 2:

                raise RuntimeError(
                    "MuJoCo contact friction vector does not "
                    "contain two tangential coefficients."
                )

            mu_1 = float(
                friction[
                    0
                ]
            )

            mu_2 = float(
                friction[
                    1
                ]
            )

            if (
                not np.isfinite(
                    mu_1
                )
                or
                not np.isfinite(
                    mu_2
                )
                or
                mu_1
                <
                0.0
                or
                mu_2
                <
                0.0
            ):

                raise RuntimeError(
                    "Invalid MuJoCo friction coefficient."
                )

            contact_position = np.asarray(
                contact.pos,
                dtype=float,
            ).reshape(
                3
            ).copy()

            contacts.append(
                FootGroundContact(
                    foot_side=(
                        foot_side
                    ),
                    position_world=(
                        contact_position
                    ),
                    contact_frame_world=(
                        contact_frame_world
                    ),
                    friction_tangent_1=(
                        mu_1
                    ),
                    friction_tangent_2=(
                        mu_2
                    ),
                    distance=(
                        contact_distance
                    ),
                )
            )

        return tuple(
            contacts
        )


    # ========================================================
    # RECORD ONE SIMULATION SAMPLE
    # ========================================================

    def record_sample(
        self,
        *,
        time,
        mj_data,
        whole_body_com_position,
        whole_body_com_velocity,
        angular_momentum,
        left_foot_position,
        right_foot_position,
    ) -> None:

        sample_time = float(
            time
        )

        if not np.isfinite(
            sample_time
        ):

            raise ValueError(
                "Sample time is not finite."
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
                "Contact-stability sample time must "
                "be strictly increasing."
            )

        contacts = (
            self._extract_foot_ground_contacts(
                mj_data=(
                    mj_data
                )
            )
        )

        self.samples.append(
            ContactKinematicsSample(
                time=(
                    sample_time
                ),
                com_position_world=(
                    finite_vector3(
                        whole_body_com_position,
                        name=(
                            "whole_body_com_position"
                        ),
                    )
                ),
                com_velocity_world=(
                    finite_vector3(
                        whole_body_com_velocity,
                        name=(
                            "whole_body_com_velocity"
                        ),
                    )
                ),
                angular_momentum_world=(
                    finite_vector3(
                        angular_momentum,
                        name=(
                            "angular_momentum"
                        ),
                    )
                ),
                left_foot_position_world=(
                    finite_vector3(
                        left_foot_position,
                        name=(
                            "left_foot_position"
                        ),
                    )
                ),
                right_foot_position_world=(
                    finite_vector3(
                        right_foot_position,
                        name=(
                            "right_foot_position"
                        ),
                    )
                ),
                contacts=(
                    contacts
                ),
            )
        )


    # ========================================================
    # BUILD CONTACT MATRICES
    # ========================================================

    def _build_contact_matrices(
        self,
        *,
        sample: ContactKinematicsSample,
        com_acceleration,
        angular_momentum_rate,
    ):
        """
        Build:

            A_eq f = b_eq

        unilateral:

            A_n f <= 0

        tangential friction pyramid:

            A_t f <= 0

        Full contact constraints:

            [A_n]
            [A_t] f <= 0
        """

        acceleration = (
            finite_vector3(
                com_acceleration,
                name=(
                    "com_acceleration"
                ),
            )
        )

        momentum_rate = (
            finite_vector3(
                angular_momentum_rate,
                name=(
                    "angular_momentum_rate"
                ),
            )
        )

        required_force = (
            self.robot_mass
            *
            (
                acceleration
                -
                self.gravity_world
            )
        )

        contacts = (
            sample.contacts
        )

        number_contacts = (
            len(
                contacts
            )
        )

        number_force_variables = (
            3
            *
            number_contacts
        )

        # ====================================================
        # NEWTON + EULER
        # ====================================================

        A_equal = np.zeros(
            (
                6,
                number_force_variables,
            ),
            dtype=float,
        )

        for contact_index, contact in enumerate(
            contacts
        ):

            column = (
                3
                *
                contact_index
            )

            # Newton:
            A_equal[
                0:3,
                column:
                column + 3,
            ] = np.eye(
                3,
                dtype=float,
            )

            # Euler about whole-body CoM:
            lever_arm = (
                contact.position_world
                -
                sample.com_position_world
            )

            A_equal[
                3:6,
                column:
                column + 3,
            ] = skew(
                lever_arm
            )

        b_equal = np.concatenate(
            (
                required_force,
                momentum_rate,
            )
        )

        # ====================================================
        # UNILATERAL CONTACT
        #
        #     f_n >= 0
        #
        # ->  -f_n <= 0
        # ====================================================

        A_unilateral = np.zeros(
            (
                number_contacts,
                number_force_variables,
            ),
            dtype=float,
        )

        # ====================================================
        # TANGENTIAL FRICTION PYRAMID
        #
        #     |f_t1| <= mu_1 f_n
        #     |f_t2| <= mu_2 f_n
        #
        # Four inequalities per contact.
        # ====================================================

        A_tangential = np.zeros(
            (
                4
                *
                number_contacts,
                number_force_variables,
            ),
            dtype=float,
        )

        for contact_index, contact in enumerate(
            contacts
        ):

            C = (
                contact.contact_frame_world
            )

            mu_1 = (
                contact.friction_tangent_1
            )

            mu_2 = (
                contact.friction_tangent_2
            )

            column = (
                3
                *
                contact_index
            )

            # Local ordering:
            #
            #     [f_n, f_t1, f_t2]
            #
            normal_local = np.array(
                [
                    -1.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

            tangential_local = np.array(
                [
                    [
                        -mu_1,
                        +1.0,
                        0.0,
                    ],
                    [
                        -mu_1,
                        -1.0,
                        0.0,
                    ],
                    [
                        -mu_2,
                        0.0,
                        +1.0,
                    ],
                    [
                        -mu_2,
                        0.0,
                        -1.0,
                    ],
                ],
                dtype=float,
            )

            A_unilateral[
                contact_index,
                column:
                column + 3,
            ] = (
                normal_local
                @
                C
            )

            row = (
                4
                *
                contact_index
            )

            A_tangential[
                row:
                row + 4,
                column:
                column + 3,
            ] = (
                tangential_local
                @
                C
            )

        return (
            acceleration,
            momentum_rate,
            required_force,
            A_equal,
            b_equal,
            A_unilateral,
            A_tangential,
        )


    # ========================================================
    # EQUALITY DIAGNOSTIC
    # ========================================================

    @staticmethod
    def _equality_diagnostic(
        *,
        A_equal,
        b_equal,
    ):
        """
        Check if the required centroidal wrench lies in the linear
        span generated by the current contact-point force mapping.
        """

        rank = (
            numerical_rank(
                A_equal
            )
        )

        least_squares_solution, _, _, _ = (
            np.linalg.lstsq(
                A_equal,
                b_equal,
                rcond=MATRIX_RCOND,
            )
        )

        residual = float(
            np.max(
                np.abs(
                    A_equal
                    @
                    least_squares_solution
                    -
                    b_equal
                )
            )
        )

        return (
            rank,
            residual,
            least_squares_solution,
        )


    # ========================================================
    # GENERIC CASADI QRQP CREATION
    # ========================================================

    def _get_solver(
        self,
        *,
        cache_key,
        number_variables,
        number_constraints,
        max_iter,
    ):

        if (
            cache_key
            in
            self._solver_cache
        ):

            return (
                self._solver_cache[
                    cache_key
                ]
            )

        qp_structure = {
            "h": ca.Sparsity.dense(
                number_variables,
                number_variables,
            ),
            "a": ca.Sparsity.dense(
                number_constraints,
                number_variables,
            ),
        }

        options = {
            "print_header": False,
            "print_iter": False,
            "print_info": False,
            "error_on_fail": False,
            "max_iter": int(
                max_iter
            ),
        }

        solver = ca.conic(
            str(
                cache_key
            ),
            "qrqp",
            qp_structure,
            options,
        )

        self._solver_cache[
            cache_key
        ] = solver

        return solver


    # ========================================================
    # GENERIC HARD FORCE QP
    # ========================================================

    def _solve_force_qp(
        self,
        *,
        solver_name,
        A_equal,
        b_equal,
        A_inequality,
        max_iter=1500,
    ):
        """
        Solve:

            min ||f||^2

        subject to:

            A_eq f = b_eq
            A_ineq f <= 0
        """

        number_force_variables = (
            A_equal.shape[
                1
            ]
        )

        number_inequalities = (
            A_inequality.shape[
                0
            ]
        )

        A = np.vstack(
            (
                A_equal,
                A_inequality,
            )
        )

        lower_constraints = np.concatenate(
            (
                b_equal,
                np.full(
                    number_inequalities,
                    -np.inf,
                    dtype=float,
                ),
            )
        )

        upper_constraints = np.concatenate(
            (
                b_equal,
                np.zeros(
                    number_inequalities,
                    dtype=float,
                ),
            )
        )

        H = (
            2.0
            *
            np.eye(
                number_force_variables,
                dtype=float,
            )
        )

        g = np.zeros(
            number_force_variables,
            dtype=float,
        )

        number_contacts = (
            number_force_variables
            //
            3
        )

        cache_key = (
            f"{solver_name}_"
            f"{number_contacts}"
        )

        solver = (
            self._get_solver(
                cache_key=(
                    cache_key
                ),
                number_variables=(
                    number_force_variables
                ),
                number_constraints=(
                    A.shape[
                        0
                    ]
                ),
                max_iter=(
                    max_iter
                ),
            )
        )

        result = solver(
            h=ca.DM(
                H
            ),
            g=ca.DM(
                g
            ),
            a=ca.DM(
                A
            ),
            lba=ca.DM(
                lower_constraints
            ),
            uba=ca.DM(
                upper_constraints
            ),
            lbx=ca.DM(
                np.full(
                    number_force_variables,
                    -np.inf,
                    dtype=float,
                )
            ),
            ubx=ca.DM(
                np.full(
                    number_force_variables,
                    +np.inf,
                    dtype=float,
                )
            ),
        )

        stats = (
            solver.stats()
        )

        success = bool(
            stats.get(
                "success",
                False,
            )
        )

        status = str(
            stats.get(
                "return_status",
                "unknown",
            )
        )

        if not success:

            return (
                False,
                status,
                None,
                float(
                    "inf"
                ),
                float(
                    "inf"
                ),
            )

        solution = np.asarray(
            result[
                "x"
            ],
            dtype=float,
        ).reshape(
            number_force_variables
        )

        if not np.all(
            np.isfinite(
                solution
            )
        ):

            return (
                False,
                "QP_RETURNED_NAN_INF",
                None,
                float(
                    "inf"
                ),
                float(
                    "inf"
                ),
            )

        equality_residual = float(
            np.max(
                np.abs(
                    A_equal
                    @
                    solution
                    -
                    b_equal
                )
            )
        )

        if (
            number_inequalities
            >
            0
        ):

            inequality_violation = float(
                max(
                    0.0,
                    np.max(
                        A_inequality
                        @
                        solution
                    ),
                )
            )

        else:

            inequality_violation = 0.0

        numerical_success = (
            equality_residual
            <=
            EQUALITY_TOLERANCE
            and
            inequality_violation
            <=
            INEQUALITY_TOLERANCE
        )

        if not numerical_success:

            status = (
                f"{status}"
                "|NUMERICAL_RESIDUAL"
            )

        return (
            bool(
                numerical_success
            ),
            status,
            solution,
            equality_residual,
            inequality_violation,
        )


    # ========================================================
    # SOFT UNILATERAL DIAGNOSTIC
    # ========================================================

    def _solve_soft_unilateral_qp(
        self,
        *,
        A_equal,
        b_equal,
        A_unilateral,
    ):
        """
        Diagnostic problem:

            min eps ||f||^2 + ||s_n||^2

        subject to:

            A_eq f = b_eq

            A_n f - s_n <= 0

            s_n >= 0

        A positive optimal s_n means the required centroidal wrench
        cannot be produced without negative normal force.
        """

        number_force_variables = (
            A_equal.shape[
                1
            ]
        )

        number_slacks = (
            A_unilateral.shape[
                0
            ]
        )

        number_variables = (
            number_force_variables
            +
            number_slacks
        )

        A_equal_soft = np.hstack(
            (
                A_equal,
                np.zeros(
                    (
                        6,
                        number_slacks,
                    ),
                    dtype=float,
                ),
            )
        )

        A_unilateral_soft = np.hstack(
            (
                A_unilateral,
                -np.eye(
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        A = np.vstack(
            (
                A_equal_soft,
                A_unilateral_soft,
            )
        )

        lower_constraints = np.concatenate(
            (
                b_equal,
                np.full(
                    number_slacks,
                    -np.inf,
                    dtype=float,
                ),
            )
        )

        upper_constraints = np.concatenate(
            (
                b_equal,
                np.zeros(
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        force_weight = 1.0e-8
        slack_weight = 1.0

        weights = np.concatenate(
            (
                np.full(
                    number_force_variables,
                    force_weight,
                    dtype=float,
                ),
                np.full(
                    number_slacks,
                    slack_weight,
                    dtype=float,
                ),
            )
        )

        H = (
            2.0
            *
            np.diag(
                weights
            )
        )

        g = np.zeros(
            number_variables,
            dtype=float,
        )

        lower_bounds = np.concatenate(
            (
                np.full(
                    number_force_variables,
                    -np.inf,
                    dtype=float,
                ),
                np.zeros(
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        upper_bounds = np.full(
            number_variables,
            +np.inf,
            dtype=float,
        )

        number_contacts = (
            number_force_variables
            //
            3
        )

        solver = (
            self._get_solver(
                cache_key=(
                    f"soft_unilateral_"
                    f"{number_contacts}"
                ),
                number_variables=(
                    number_variables
                ),
                number_constraints=(
                    A.shape[
                        0
                    ]
                ),
                max_iter=2500,
            )
        )

        result = solver(
            h=ca.DM(
                H
            ),
            g=ca.DM(
                g
            ),
            a=ca.DM(
                A
            ),
            lba=ca.DM(
                lower_constraints
            ),
            uba=ca.DM(
                upper_constraints
            ),
            lbx=ca.DM(
                lower_bounds
            ),
            ubx=ca.DM(
                upper_bounds
            ),
        )

        stats = (
            solver.stats()
        )

        success = bool(
            stats.get(
                "success",
                False,
            )
        )

        status = str(
            stats.get(
                "return_status",
                "unknown",
            )
        )

        if not success:

            return (
                False,
                status,
                float(
                    "inf"
                ),
            )

        solution = np.asarray(
            result[
                "x"
            ],
            dtype=float,
        ).reshape(
            number_variables
        )

        if not np.all(
            np.isfinite(
                solution
            )
        ):

            return (
                False,
                "SOFT_UNILATERAL_RETURNED_NAN_INF",
                float(
                    "inf"
                ),
            )

        slacks = (
            solution[
                number_force_variables:
            ]
        )

        max_slack = float(
            np.max(
                slacks
            )
        )

        return (
            True,
            status,
            max_slack,
        )


    # ========================================================
    # SOFT TANGENTIAL-FRICTION DIAGNOSTIC
    # ========================================================

    def _solve_soft_friction_qp(
        self,
        *,
        A_equal,
        b_equal,
        A_unilateral,
        A_tangential,
    ):
        """
        Diagnostic problem:

            min eps ||f||^2 + ||s_mu||^2

        subject to:

            A_eq f = b_eq

            A_n f <= 0
                (unilateral remains HARD)

            A_t f - s_mu <= 0

            s_mu >= 0

        This test is run only after equality + unilateral contact has
        already been shown feasible.

        Therefore a positive optimal s_mu indicates that the remaining
        failure is due to tangential friction limits.
        """

        number_force_variables = (
            A_equal.shape[
                1
            ]
        )

        number_slacks = (
            A_tangential.shape[
                0
            ]
        )

        number_variables = (
            number_force_variables
            +
            number_slacks
        )

        A_equal_soft = np.hstack(
            (
                A_equal,
                np.zeros(
                    (
                        6,
                        number_slacks,
                    ),
                    dtype=float,
                ),
            )
        )

        A_unilateral_soft = np.hstack(
            (
                A_unilateral,
                np.zeros(
                    (
                        A_unilateral.shape[
                            0
                        ],
                        number_slacks,
                    ),
                    dtype=float,
                ),
            )
        )

        A_tangential_soft = np.hstack(
            (
                A_tangential,
                -np.eye(
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        A = np.vstack(
            (
                A_equal_soft,
                A_unilateral_soft,
                A_tangential_soft,
            )
        )

        lower_constraints = np.concatenate(
            (
                b_equal,
                np.full(
                    A_unilateral.shape[
                        0
                    ]
                    +
                    number_slacks,
                    -np.inf,
                    dtype=float,
                ),
            )
        )

        upper_constraints = np.concatenate(
            (
                b_equal,
                np.zeros(
                    A_unilateral.shape[
                        0
                    ]
                    +
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        force_weight = 1.0e-8
        slack_weight = 1.0

        weights = np.concatenate(
            (
                np.full(
                    number_force_variables,
                    force_weight,
                    dtype=float,
                ),
                np.full(
                    number_slacks,
                    slack_weight,
                    dtype=float,
                ),
            )
        )

        H = (
            2.0
            *
            np.diag(
                weights
            )
        )

        g = np.zeros(
            number_variables,
            dtype=float,
        )

        lower_bounds = np.concatenate(
            (
                np.full(
                    number_force_variables,
                    -np.inf,
                    dtype=float,
                ),
                np.zeros(
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        upper_bounds = np.full(
            number_variables,
            +np.inf,
            dtype=float,
        )

        number_contacts = (
            number_force_variables
            //
            3
        )

        solver = (
            self._get_solver(
                cache_key=(
                    f"soft_friction_"
                    f"{number_contacts}"
                ),
                number_variables=(
                    number_variables
                ),
                number_constraints=(
                    A.shape[
                        0
                    ]
                ),
                max_iter=3000,
            )
        )

        result = solver(
            h=ca.DM(
                H
            ),
            g=ca.DM(
                g
            ),
            a=ca.DM(
                A
            ),
            lba=ca.DM(
                lower_constraints
            ),
            uba=ca.DM(
                upper_constraints
            ),
            lbx=ca.DM(
                lower_bounds
            ),
            ubx=ca.DM(
                upper_bounds
            ),
        )

        stats = (
            solver.stats()
        )

        success = bool(
            stats.get(
                "success",
                False,
            )
        )

        status = str(
            stats.get(
                "return_status",
                "unknown",
            )
        )

        if not success:

            return (
                False,
                status,
                float(
                    "inf"
                ),
            )

        solution = np.asarray(
            result[
                "x"
            ],
            dtype=float,
        ).reshape(
            number_variables
        )

        if not np.all(
            np.isfinite(
                solution
            )
        ):

            return (
                False,
                "SOFT_FRICTION_RETURNED_NAN_INF",
                float(
                    "inf"
                ),
            )

        slacks = (
            solution[
                number_force_variables:
            ]
        )

        max_slack = float(
            np.max(
                slacks
            )
        )

        return (
            True,
            status,
            max_slack,
        )


    # ========================================================
    # FRICTION UTILIZATION
    # ========================================================

    @staticmethod
    def _friction_utilization(
        *,
        normal_force,
        tangent_force_1,
        tangent_force_2,
        mu_1,
        mu_2,
    ) -> float:

        fn = float(
            normal_force
        )

        ft1 = float(
            tangent_force_1
        )

        ft2 = float(
            tangent_force_2
        )

        mu_1 = float(
            mu_1
        )

        mu_2 = float(
            mu_2
        )

        if (
            fn
            <=
            NORMAL_FORCE_TOLERANCE
        ):

            if (
                abs(
                    ft1
                )
                <=
                NORMAL_FORCE_TOLERANCE
                and
                abs(
                    ft2
                )
                <=
                NORMAL_FORCE_TOLERANCE
            ):

                return 0.0

            return float(
                "inf"
            )

        ratios = []

        if (
            mu_1
            >
            0.0
        ):

            ratios.append(
                abs(
                    ft1
                )
                /
                (
                    mu_1
                    *
                    fn
                )
            )

        elif (
            abs(
                ft1
            )
            <=
            NORMAL_FORCE_TOLERANCE
        ):

            ratios.append(
                0.0
            )

        else:

            return float(
                "inf"
            )

        if (
            mu_2
            >
            0.0
        ):

            ratios.append(
                abs(
                    ft2
                )
                /
                (
                    mu_2
                    *
                    fn
                )
            )

        elif (
            abs(
                ft2
            )
            <=
            NORMAL_FORCE_TOLERANCE
        ):

            ratios.append(
                0.0
            )

        else:

            return float(
                "inf"
            )

        return float(
            max(
                ratios
            )
        )


    # ========================================================
    # GENERIC NON-FEASIBLE RESULT
    # ========================================================

    @staticmethod
    def _diagnostic_result(
        *,
        sample,
        classification,
        solver_status,
        equality_rank,
        equality_ls_residual,
        com_acceleration,
        angular_momentum_rate,
        required_force,
        equality_residual=float("inf"),
        inequality_violation=float("inf"),
        max_unilateral_slack=float("nan"),
        max_friction_slack=float("nan"),
    ) -> ContactStabilityResult:

        zero = np.zeros(
            3,
            dtype=float,
        )

        (
            _,
            equality_sigma_min,
            equality_sigma_max,
            equality_condition_number,
            left_contact_count,
            right_contact_count,
            left_contact_span,
            right_contact_span,
            max_foot_contact_span,
        ) = (
            sample_contact_geometry_diagnostics(
                sample
            )
        )

        return ContactStabilityResult(
            time=(
                sample.time
            ),
            classification=(
                classification
            ),
            feasible=False,
            solver_status=(
                str(
                    solver_status
                )
            ),
            number_contacts=(
                len(
                    sample.contacts
                )
            ),
            equality_rank=int(
                equality_rank
            ),
            equality_ls_residual=float(
                equality_ls_residual
            ),
            equality_sigma_min=float(
                equality_sigma_min
            ),
            equality_sigma_max=float(
                equality_sigma_max
            ),
            equality_condition_number=float(
                equality_condition_number
            ),
            left_contact_count=int(
                left_contact_count
            ),
            right_contact_count=int(
                right_contact_count
            ),
            left_contact_span=float(
                left_contact_span
            ),
            right_contact_span=float(
                right_contact_span
            ),
            max_foot_contact_span=float(
                max_foot_contact_span
            ),
            com_acceleration_world=np.asarray(
                com_acceleration,
                dtype=float,
            ).copy(),
            angular_momentum_rate_world=np.asarray(
                angular_momentum_rate,
                dtype=float,
            ).copy(),
            required_contact_force_world=np.asarray(
                required_force,
                dtype=float,
            ).copy(),
            equality_residual=float(
                equality_residual
            ),
            inequality_violation=float(
                inequality_violation
            ),
            max_unilateral_slack=float(
                max_unilateral_slack
            ),
            max_friction_slack=float(
                max_friction_slack
            ),
            max_friction_utilization=float(
                "nan"
            ),
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
            contact_forces=tuple(),
        )


    # ========================================================
    # BUILD FEASIBLE RESULT
    # ========================================================

    def _build_feasible_result(
        self,
        *,
        sample,
        solution,
        solver_status,
        equality_rank,
        equality_ls_residual,
        com_acceleration,
        angular_momentum_rate,
        required_force,
        equality_residual,
        inequality_violation,
    ) -> ContactStabilityResult:

        contacts = (
            sample.contacts
        )

        (
            _,
            equality_sigma_min,
            equality_sigma_max,
            equality_condition_number,
            left_contact_count,
            right_contact_count,
            left_contact_span,
            right_contact_span,
            max_foot_contact_span,
        ) = (
            sample_contact_geometry_diagnostics(
                sample
            )
        )

        contact_force_solutions = []

        left_resultant_force = np.zeros(
            3,
            dtype=float,
        )

        left_resultant_moment = np.zeros(
            3,
            dtype=float,
        )

        right_resultant_force = np.zeros(
            3,
            dtype=float,
        )

        right_resultant_moment = np.zeros(
            3,
            dtype=float,
        )

        max_friction_utilization = 0.0

        for contact_index, contact in enumerate(
            contacts
        ):

            column = (
                3
                *
                contact_index
            )

            force_world = (
                solution[
                    column:
                    column + 3
                ].copy()
            )

            force_contact = (
                contact.contact_frame_world
                @
                force_world
            )

            friction_utilization = (
                self._friction_utilization(
                    normal_force=(
                        force_contact[0]
                    ),
                    tangent_force_1=(
                        force_contact[1]
                    ),
                    tangent_force_2=(
                        force_contact[2]
                    ),
                    mu_1=(
                        contact.friction_tangent_1
                    ),
                    mu_2=(
                        contact.friction_tangent_2
                    ),
                )
            )

            max_friction_utilization = max(
                max_friction_utilization,
                friction_utilization,
            )

            contact_force_solutions.append(
                ContactForceSolution(
                    foot_side=(
                        contact.foot_side
                    ),
                    position_world=(
                        contact.position_world.copy()
                    ),
                    force_world=(
                        force_world.copy()
                    ),
                    force_contact=(
                        force_contact.copy()
                    ),
                    friction_tangent_1=(
                        contact.friction_tangent_1
                    ),
                    friction_tangent_2=(
                        contact.friction_tangent_2
                    ),
                    friction_utilization=(
                        friction_utilization
                    ),
                )
            )

            if (
                contact.foot_side
                ==
                "left"
            ):

                left_resultant_force += (
                    force_world
                )

                left_resultant_moment += np.cross(
                    (
                        contact.position_world
                        -
                        sample.left_foot_position_world
                    ),
                    force_world,
                )

            elif (
                contact.foot_side
                ==
                "right"
            ):

                right_resultant_force += (
                    force_world
                )

                right_resultant_moment += np.cross(
                    (
                        contact.position_world
                        -
                        sample.right_foot_position_world
                    ),
                    force_world,
                )

        return ContactStabilityResult(
            time=(
                sample.time
            ),
            classification=(
                CLASS_FEASIBLE
            ),
            feasible=True,
            solver_status=(
                str(
                    solver_status
                )
            ),
            number_contacts=(
                len(
                    contacts
                )
            ),
            equality_rank=int(
                equality_rank
            ),
            equality_ls_residual=float(
                equality_ls_residual
            ),
            equality_sigma_min=float(
                equality_sigma_min
            ),
            equality_sigma_max=float(
                equality_sigma_max
            ),
            equality_condition_number=float(
                equality_condition_number
            ),
            left_contact_count=int(
                left_contact_count
            ),
            right_contact_count=int(
                right_contact_count
            ),
            left_contact_span=float(
                left_contact_span
            ),
            right_contact_span=float(
                right_contact_span
            ),
            max_foot_contact_span=float(
                max_foot_contact_span
            ),
            com_acceleration_world=np.asarray(
                com_acceleration,
                dtype=float,
            ).copy(),
            angular_momentum_rate_world=np.asarray(
                angular_momentum_rate,
                dtype=float,
            ).copy(),
            required_contact_force_world=np.asarray(
                required_force,
                dtype=float,
            ).copy(),
            equality_residual=float(
                equality_residual
            ),
            inequality_violation=float(
                inequality_violation
            ),
            max_unilateral_slack=0.0,
            max_friction_slack=0.0,
            max_friction_utilization=float(
                max_friction_utilization
            ),
            left_resultant_force_world=(
                left_resultant_force
            ),
            left_resultant_moment_world=(
                left_resultant_moment
            ),
            right_resultant_force_world=(
                right_resultant_force
            ),
            right_resultant_moment_world=(
                right_resultant_moment
            ),
            contact_forces=tuple(
                contact_force_solutions
            ),
        )


    # ========================================================
    # SOLVE ONE SAMPLE
    # ========================================================

    def _solve_sample(
        self,
        *,
        sample: ContactKinematicsSample,
        com_acceleration,
        angular_momentum_rate,
    ) -> ContactStabilityResult:

        (
            acceleration,
            momentum_rate,
            required_force,
            A_equal,
            b_equal,
            A_unilateral,
            A_tangential,
        ) = (
            self._build_contact_matrices(
                sample=(
                    sample
                ),
                com_acceleration=(
                    com_acceleration
                ),
                angular_momentum_rate=(
                    angular_momentum_rate
                ),
            )
        )

        number_contacts = (
            len(
                sample.contacts
            )
        )

        # ====================================================
        # NO CONTACT
        # ====================================================

        if (
            number_contacts
            ==
            0
        ):

            return (
                self._diagnostic_result(
                    sample=(
                        sample
                    ),
                    classification=(
                        CLASS_NO_CONTACT
                    ),
                    solver_status=(
                        CLASS_NO_CONTACT
                    ),
                    equality_rank=0,
                    equality_ls_residual=float(
                        "inf"
                    ),
                    com_acceleration=(
                        acceleration
                    ),
                    angular_momentum_rate=(
                        momentum_rate
                    ),
                    required_force=(
                        required_force
                    ),
                )
            )

        # ====================================================
        # LAYER 1
        #
        # NEWTON + EULER ONLY
        # ====================================================

        (
            equality_rank,
            equality_ls_residual,
            _,
        ) = (
            self._equality_diagnostic(
                A_equal=(
                    A_equal
                ),
                b_equal=(
                    b_equal
                ),
            )
        )

        if (
            equality_ls_residual
            >
            EQUALITY_LS_TOLERANCE
        ):

            return (
                self._diagnostic_result(
                    sample=(
                        sample
                    ),
                    classification=(
                        CLASS_EQUALITY_INFEASIBLE
                    ),
                    solver_status=(
                        CLASS_EQUALITY_INFEASIBLE
                    ),
                    equality_rank=(
                        equality_rank
                    ),
                    equality_ls_residual=(
                        equality_ls_residual
                    ),
                    com_acceleration=(
                        acceleration
                    ),
                    angular_momentum_rate=(
                        momentum_rate
                    ),
                    required_force=(
                        required_force
                    ),
                )
            )

        # ====================================================
        # LAYER 2
        #
        # NEWTON + EULER + UNILATERAL
        #
        # Determine whether all required normal contact forces
        # can be nonnegative.
        # ====================================================

        (
            unilateral_success,
            unilateral_status,
            _,
            unilateral_eq_residual,
            unilateral_ineq_violation,
        ) = (
            self._solve_force_qp(
                solver_name=(
                    "unilateral_hard"
                ),
                A_equal=(
                    A_equal
                ),
                b_equal=(
                    b_equal
                ),
                A_inequality=(
                    A_unilateral
                ),
                max_iter=1500,
            )
        )

        if not unilateral_success:

            (
                soft_unilateral_success,
                soft_unilateral_status,
                max_unilateral_slack,
            ) = (
                self._solve_soft_unilateral_qp(
                    A_equal=(
                        A_equal
                    ),
                    b_equal=(
                        b_equal
                    ),
                    A_unilateral=(
                        A_unilateral
                    ),
                )
            )

            if soft_unilateral_success:

                if (
                    max_unilateral_slack
                    >
                    UNILATERAL_SLACK_TOLERANCE
                ):

                    return (
                        self._diagnostic_result(
                            sample=(
                                sample
                            ),
                            classification=(
                                CLASS_UNILATERAL_INFEASIBLE
                            ),
                            solver_status=(
                                f"hard={unilateral_status}"
                                f" | soft={soft_unilateral_status}"
                            ),
                            equality_rank=(
                                equality_rank
                            ),
                            equality_ls_residual=(
                                equality_ls_residual
                            ),
                            com_acceleration=(
                                acceleration
                            ),
                            angular_momentum_rate=(
                                momentum_rate
                            ),
                            required_force=(
                                required_force
                            ),
                            equality_residual=(
                                unilateral_eq_residual
                            ),
                            inequality_violation=(
                                unilateral_ineq_violation
                            ),
                            max_unilateral_slack=(
                                max_unilateral_slack
                            ),
                        )
                    )

                # Soft problem says unilateral constraints can be
                # satisfied without meaningful slack, but hard QP
                # failed -> numerical solver issue.
                return (
                    self._diagnostic_result(
                        sample=(
                            sample
                        ),
                        classification=(
                            CLASS_SOLVER_FAILURE
                        ),
                        solver_status=(
                            f"unilateral hard={unilateral_status}"
                            f" | unilateral soft="
                            f"{soft_unilateral_status}"
                        ),
                        equality_rank=(
                            equality_rank
                        ),
                        equality_ls_residual=(
                            equality_ls_residual
                        ),
                        com_acceleration=(
                            acceleration
                        ),
                        angular_momentum_rate=(
                            momentum_rate
                        ),
                        required_force=(
                            required_force
                        ),
                        equality_residual=(
                            unilateral_eq_residual
                        ),
                        inequality_violation=(
                            unilateral_ineq_violation
                        ),
                        max_unilateral_slack=(
                            max_unilateral_slack
                        ),
                    )
                )

            # Neither hard nor soft unilateral problem could be
            # solved reliably.
            return (
                self._diagnostic_result(
                    sample=(
                        sample
                    ),
                    classification=(
                        CLASS_SOLVER_FAILURE
                    ),
                    solver_status=(
                        f"unilateral hard={unilateral_status}"
                        f" | unilateral soft="
                        f"{soft_unilateral_status}"
                    ),
                    equality_rank=(
                        equality_rank
                    ),
                    equality_ls_residual=(
                        equality_ls_residual
                    ),
                    com_acceleration=(
                        acceleration
                    ),
                    angular_momentum_rate=(
                        momentum_rate
                    ),
                    required_force=(
                        required_force
                    ),
                    equality_residual=(
                        unilateral_eq_residual
                    ),
                    inequality_violation=(
                        unilateral_ineq_violation
                    ),
                )
            )

        # ====================================================
        # LAYER 3
        #
        # ADD TANGENTIAL FRICTION PYRAMID
        # ====================================================

        A_full_contact = np.vstack(
            (
                A_unilateral,
                A_tangential,
            )
        )

        (
            full_success,
            full_status,
            full_solution,
            full_eq_residual,
            full_ineq_violation,
        ) = (
            self._solve_force_qp(
                solver_name=(
                    "contact_full"
                ),
                A_equal=(
                    A_equal
                ),
                b_equal=(
                    b_equal
                ),
                A_inequality=(
                    A_full_contact
                ),
                max_iter=2000,
            )
        )

        if full_success:

            return (
                self._build_feasible_result(
                    sample=(
                        sample
                    ),
                    solution=(
                        full_solution
                    ),
                    solver_status=(
                        full_status
                    ),
                    equality_rank=(
                        equality_rank
                    ),
                    equality_ls_residual=(
                        equality_ls_residual
                    ),
                    com_acceleration=(
                        acceleration
                    ),
                    angular_momentum_rate=(
                        momentum_rate
                    ),
                    required_force=(
                        required_force
                    ),
                    equality_residual=(
                        full_eq_residual
                    ),
                    inequality_violation=(
                        full_ineq_violation
                    ),
                )
            )

        # ====================================================
        # SOFT TANGENTIAL-FRICTION FALLBACK
        #
        # Unilateral remains hard.
        # Only tangential pyramid faces receive slack.
        # ====================================================

        (
            soft_friction_success,
            soft_friction_status,
            max_friction_slack,
        ) = (
            self._solve_soft_friction_qp(
                A_equal=(
                    A_equal
                ),
                b_equal=(
                    b_equal
                ),
                A_unilateral=(
                    A_unilateral
                ),
                A_tangential=(
                    A_tangential
                ),
            )
        )

        if soft_friction_success:

            if (
                max_friction_slack
                >
                FRICTION_SLACK_TOLERANCE
            ):

                return (
                    self._diagnostic_result(
                        sample=(
                            sample
                        ),
                        classification=(
                            CLASS_FRICTION_INFEASIBLE
                        ),
                        solver_status=(
                            f"hard={full_status}"
                            f" | soft={soft_friction_status}"
                        ),
                        equality_rank=(
                            equality_rank
                        ),
                        equality_ls_residual=(
                            equality_ls_residual
                        ),
                        com_acceleration=(
                            acceleration
                        ),
                        angular_momentum_rate=(
                            momentum_rate
                        ),
                        required_force=(
                            required_force
                        ),
                        equality_residual=(
                            full_eq_residual
                        ),
                        inequality_violation=(
                            full_ineq_violation
                        ),
                        max_unilateral_slack=0.0,
                        max_friction_slack=(
                            max_friction_slack
                        ),
                    )
                )

            # Soft problem needs essentially no friction slack, but
            # the hard QP failed -> numerical issue.
            return (
                self._diagnostic_result(
                    sample=(
                        sample
                    ),
                    classification=(
                        CLASS_SOLVER_FAILURE
                    ),
                    solver_status=(
                        f"full hard={full_status}"
                        f" | friction soft="
                        f"{soft_friction_status}"
                    ),
                    equality_rank=(
                        equality_rank
                    ),
                    equality_ls_residual=(
                        equality_ls_residual
                    ),
                    com_acceleration=(
                        acceleration
                    ),
                    angular_momentum_rate=(
                        momentum_rate
                    ),
                    required_force=(
                        required_force
                    ),
                    equality_residual=(
                        full_eq_residual
                    ),
                    inequality_violation=(
                        full_ineq_violation
                    ),
                    max_unilateral_slack=0.0,
                    max_friction_slack=(
                        max_friction_slack
                    ),
                )
            )

        return (
            self._diagnostic_result(
                sample=(
                    sample
                ),
                classification=(
                    CLASS_SOLVER_FAILURE
                ),
                solver_status=(
                    f"full hard={full_status}"
                    f" | friction soft="
                    f"{soft_friction_status}"
                ),
                equality_rank=(
                    equality_rank
                ),
                equality_ls_residual=(
                    equality_ls_residual
                ),
                com_acceleration=(
                    acceleration
                ),
                angular_momentum_rate=(
                    momentum_rate
                ),
                required_force=(
                    required_force
                ),
                equality_residual=(
                    full_eq_residual
                ),
                inequality_violation=(
                    full_ineq_violation
                ),
                max_unilateral_slack=0.0,
            )
        )


    # ========================================================
    # BOUNDARY RESULT
    # ========================================================

    @staticmethod
    def _boundary_result(
        *,
        sample,
        com_acceleration,
        angular_momentum_rate,
        required_force,
    ) -> ContactStabilityResult:

        zero = np.zeros(
            3,
            dtype=float,
        )

        (
            _,
            equality_sigma_min,
            equality_sigma_max,
            equality_condition_number,
            left_contact_count,
            right_contact_count,
            left_contact_span,
            right_contact_span,
            max_foot_contact_span,
        ) = (
            sample_contact_geometry_diagnostics(
                sample
            )
        )

        return ContactStabilityResult(
            time=(
                sample.time
            ),
            classification=(
                CLASS_BOUNDARY_SKIPPED
            ),
            feasible=False,
            solver_status=(
                CLASS_BOUNDARY_SKIPPED
            ),
            number_contacts=(
                len(
                    sample.contacts
                )
            ),
            equality_rank=-1,
            equality_ls_residual=float(
                "nan"
            ),
            equality_sigma_min=float(
                equality_sigma_min
            ),
            equality_sigma_max=float(
                equality_sigma_max
            ),
            equality_condition_number=float(
                equality_condition_number
            ),
            left_contact_count=int(
                left_contact_count
            ),
            right_contact_count=int(
                right_contact_count
            ),
            left_contact_span=float(
                left_contact_span
            ),
            right_contact_span=float(
                right_contact_span
            ),
            max_foot_contact_span=float(
                max_foot_contact_span
            ),
            com_acceleration_world=np.asarray(
                com_acceleration,
                dtype=float,
            ).copy(),
            angular_momentum_rate_world=np.asarray(
                angular_momentum_rate,
                dtype=float,
            ).copy(),
            required_contact_force_world=np.asarray(
                required_force,
                dtype=float,
            ).copy(),
            equality_residual=float(
                "nan"
            ),
            inequality_violation=float(
                "nan"
            ),
            max_unilateral_slack=float(
                "nan"
            ),
            max_friction_slack=float(
                "nan"
            ),
            max_friction_utilization=float(
                "nan"
            ),
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
            contact_forces=tuple(),
        )


    # ========================================================
    # SOLVE COMPLETE TRAJECTORY
    # ========================================================

    def solve_all(
        self,
    ) -> list[ContactStabilityResult]:
        """
        Evaluate the complete recorded trajectory.

        Derivatives:

            a_G = d(v_G)/dt
            dL_G/dt

        are reconstructed offline.

        The first and last sample are skipped for physical
        classification because their derivatives are one-sided.
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
                "At least 3 samples are required "
                "for contact-stability evaluation."
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
                "Recorded sample time is not "
                "strictly increasing."
            )

        com_velocity = np.vstack(
            [
                sample.com_velocity_world
                for sample
                in self.samples
            ]
        )

        angular_momentum = np.vstack(
            [
                sample.angular_momentum_world
                for sample
                in self.samples
            ]
        )

        com_acceleration = np.gradient(
            com_velocity,
            time,
            axis=0,
            edge_order=2,
        )

        angular_momentum_rate = np.gradient(
            angular_momentum,
            time,
            axis=0,
            edge_order=2,
        )

        results = []

        for sample_index, sample in enumerate(
            self.samples
        ):

            required_force = (
                self.robot_mass
                *
                (
                    com_acceleration[
                        sample_index
                    ]
                    -
                    self.gravity_world
                )
            )

            if (
                sample_index
                ==
                0
                or
                sample_index
                ==
                number_samples
                -
                1
            ):

                result = (
                    self._boundary_result(
                        sample=(
                            sample
                        ),
                        com_acceleration=(
                            com_acceleration[
                                sample_index
                            ]
                        ),
                        angular_momentum_rate=(
                            angular_momentum_rate[
                                sample_index
                            ]
                        ),
                        required_force=(
                            required_force
                        ),
                    )
                )

            else:

                result = (
                    self._solve_sample(
                        sample=(
                            sample
                        ),
                        com_acceleration=(
                            com_acceleration[
                                sample_index
                            ]
                        ),
                        angular_momentum_rate=(
                            angular_momentum_rate[
                                sample_index
                            ]
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
    def _print_counter(
        *,
        title,
        counter,
    ):

        print(
            title
        )

        if len(
            counter
        ) == 0:

            print(
                "  (none)"
            )

            return

        for key in sorted(
            counter
        ):

            print(
                f"  {key}: "
                f"{counter[key]}"
            )


    @staticmethod
    def _finite_percentiles(
        values,
    ):

        array = np.asarray(
            values,
            dtype=float,
        )

        array = array[
            np.isfinite(
                array
            )
        ]

        if array.size == 0:

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
                    array,
                    50.0,
                )
            ),
            float(
                np.percentile(
                    array,
                    95.0,
                )
            ),
            float(
                np.percentile(
                    array,
                    99.0,
                )
            ),
            float(
                np.max(
                    array
                )
            ),
        )


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
                "No contact-stability results."
            )

            return

        evaluated = [
            result
            for result
            in results
            if result.classification
            !=
            CLASS_BOUNDARY_SKIPPED
        ]

        skipped = [
            result
            for result
            in results
            if result.classification
            ==
            CLASS_BOUNDARY_SKIPPED
        ]

        feasible = [
            result
            for result
            in evaluated
            if result.classification
            ==
            CLASS_FEASIBLE
        ]

        equality_infeasible = [
            result
            for result
            in evaluated
            if result.classification
            ==
            CLASS_EQUALITY_INFEASIBLE
        ]

        unilateral_infeasible = [
            result
            for result
            in evaluated
            if result.classification
            ==
            CLASS_UNILATERAL_INFEASIBLE
        ]

        friction_infeasible = [
            result
            for result
            in evaluated
            if result.classification
            ==
            CLASS_FRICTION_INFEASIBLE
        ]

        solver_failures = [
            result
            for result
            in evaluated
            if result.classification
            ==
            CLASS_SOLVER_FAILURE
        ]

        no_contact = [
            result
            for result
            in evaluated
            if result.classification
            ==
            CLASS_NO_CONTACT
        ]

        if len(
            evaluated
        ) > 0:

            feasible_percentage = (
                100.0
                *
                len(
                    feasible
                )
                /
                len(
                    evaluated
                )
            )

        else:

            feasible_percentage = float(
                "nan"
            )

        # ====================================================
        # CONTACT COUNT / EQUALITY RANK
        # ====================================================

        contact_counter = Counter(
            result.number_contacts
            for result
            in evaluated
        )

        equality_rank_counter = Counter(
            result.equality_rank
            for result
            in evaluated
            if result.equality_rank
            >=
            0
        )

        # ====================================================
        # DERIVATIVE DIAGNOSTICS
        # ====================================================

        acceleration_norm = [
            np.linalg.norm(
                result.com_acceleration_world
            )
            for result
            in evaluated
        ]

        momentum_rate_norm = [
            np.linalg.norm(
                result.angular_momentum_rate_world
            )
            for result
            in evaluated
        ]

        (
            accel_p50,
            accel_p95,
            accel_p99,
            accel_max,
        ) = (
            ContactStabilityChecker
            ._finite_percentiles(
                acceleration_norm
            )
        )

        (
            mom_p50,
            mom_p95,
            mom_p99,
            mom_max,
        ) = (
            ContactStabilityChecker
            ._finite_percentiles(
                momentum_rate_norm
            )
        )

        # ====================================================
        # SLACK DIAGNOSTICS
        # ====================================================

        unilateral_slacks = [
            result.max_unilateral_slack
            for result
            in unilateral_infeasible
        ]

        friction_slacks = [
            result.max_friction_slack
            for result
            in friction_infeasible
        ]

        (
            unilat_p50,
            unilat_p95,
            unilat_p99,
            unilat_max,
        ) = (
            ContactStabilityChecker
            ._finite_percentiles(
                unilateral_slacks
            )
        )

        (
            friction_p50,
            friction_p95,
            friction_p99,
            friction_max,
        ) = (
            ContactStabilityChecker
            ._finite_percentiles(
                friction_slacks
            )
        )

        # ====================================================
        # FEASIBLE-SAMPLE NUMERICS
        # ====================================================

        if len(
            feasible
        ) > 0:

            friction_usage = [
                result.max_friction_utilization
                for result
                in feasible
            ]

            (
                rho_p50,
                rho_p95,
                rho_p99,
                rho_max,
            ) = (
                ContactStabilityChecker
                ._finite_percentiles(
                    friction_usage
                )
            )

            max_equality_residual = max(
                result.equality_residual
                for result
                in feasible
            )

            max_inequality_violation = max(
                result.inequality_violation
                for result
                in feasible
            )

        else:

            rho_p50 = float(
                "nan"
            )

            rho_p95 = float(
                "nan"
            )

            rho_p99 = float(
                "nan"
            )

            rho_max = float(
                "nan"
            )

            max_equality_residual = float(
                "nan"
            )

            max_inequality_violation = float(
                "nan"
            )

        # ====================================================
        # PRINT SUMMARY
        # ====================================================

        print()
        print(
            "================================================"
        )
        print(
            "CONTACT STABILITY DIAGNOSTIC SUMMARY"
        )
        print(
            "================================================"
        )

        print(
            f"Recorded samples       : "
            f"{len(results)}"
        )

        print(
            f"Evaluated samples      : "
            f"{len(evaluated)}"
        )

        print(
            f"Boundary skipped       : "
            f"{len(skipped)}"
        )

        print()

        print(
            "FEASIBILITY CLASSIFICATION"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"Feasible               : "
            f"{len(feasible)}"
        )

        print(
            f"Equality infeasible    : "
            f"{len(equality_infeasible)}"
        )

        print(
            f"Unilateral infeasible  : "
            f"{len(unilateral_infeasible)}"
        )

        print(
            f"Friction infeasible    : "
            f"{len(friction_infeasible)}"
        )

        print(
            f"Solver failure         : "
            f"{len(solver_failures)}"
        )

        print(
            f"No contact             : "
            f"{len(no_contact)}"
        )

        print(
            f"Feasible percentage    : "
            f"{feasible_percentage:.3f} %"
        )

        print()

        ContactStabilityChecker._print_counter(
            title=(
                "CONTACT COUNT HISTOGRAM"
            ),
            counter=(
                contact_counter
            ),
        )

        print()

        ContactStabilityChecker._print_counter(
            title=(
                "EQUALITY RANK HISTOGRAM"
            ),
            counter=(
                equality_rank_counter
            ),
        )

        print()

        # ====================================================
        # CONTACT GEOMETRY / CONDITIONING DIAGNOSTICS
        # ====================================================

        sigma_min_values = [
            result.equality_sigma_min
            for result
            in evaluated
        ]

        condition_values = [
            result.equality_condition_number
            for result
            in evaluated
        ]

        contact_span_values = [
            result.max_foot_contact_span
            for result
            in evaluated
            if result.max_foot_contact_span
            >
            0.0
        ]

        finite_sigma_min = np.asarray(
            sigma_min_values,
            dtype=float,
        )

        finite_sigma_min = finite_sigma_min[
            np.isfinite(
                finite_sigma_min
            )
        ]

        if finite_sigma_min.size > 0:

            sigma_min_min = float(
                np.min(
                    finite_sigma_min
                )
            )

            sigma_min_p50 = float(
                np.percentile(
                    finite_sigma_min,
                    50.0,
                )
            )

            sigma_min_p05 = float(
                np.percentile(
                    finite_sigma_min,
                    5.0,
                )
            )

            sigma_min_p01 = float(
                np.percentile(
                    finite_sigma_min,
                    1.0,
                )
            )

        else:

            sigma_min_min = float(
                "nan"
            )

            sigma_min_p50 = float(
                "nan"
            )

            sigma_min_p05 = float(
                "nan"
            )

            sigma_min_p01 = float(
                "nan"
            )

        (
            cond_p50,
            cond_p95,
            cond_p99,
            cond_max,
        ) = (
            ContactStabilityChecker
            ._finite_percentiles(
                condition_values
            )
        )

        finite_spans = np.asarray(
            contact_span_values,
            dtype=float,
        )

        finite_spans = finite_spans[
            np.isfinite(
                finite_spans
            )
        ]

        if finite_spans.size > 0:

            span_min = float(
                np.min(
                    finite_spans
                )
            )

            span_p50 = float(
                np.percentile(
                    finite_spans,
                    50.0,
                )
            )

            span_p95 = float(
                np.percentile(
                    finite_spans,
                    95.0,
                )
            )

            span_p99 = float(
                np.percentile(
                    finite_spans,
                    99.0,
                )
            )

            span_max = float(
                np.max(
                    finite_spans
                )
            )

        else:

            span_min = float(
                "nan"
            )

            span_p50 = float(
                "nan"
            )

            span_p95 = float(
                "nan"
            )

            span_p99 = float(
                "nan"
            )

            span_max = float(
                "nan"
            )

        print(
            "EQUALITY CONDITIONING DIAGNOSTICS"
        )
        print(
            "------------------------------------------------"
        )

        print(
            "sigma_min(A_eq)"
        )

        print(
            f"  min : {sigma_min_min:.6e}"
        )

        print(
            f"  p01 : {sigma_min_p01:.6e}"
        )

        print(
            f"  p05 : {sigma_min_p05:.6e}"
        )

        print(
            f"  p50 : {sigma_min_p50:.6e}"
        )

        print()

        print(
            "cond(A_eq)"
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
            "CONTACT PATCH SPAN [m]"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  min : {span_min:.6e}"
        )

        print(
            f"  p50 : {span_p50:.6e}"
        )

        print(
            f"  p95 : {span_p95:.6e}"
        )

        print(
            f"  p99 : {span_p99:.6e}"
        )

        print(
            f"  max : {span_max:.6e}"
        )

        print()

        print(
            "DERIVATIVE DIAGNOSTICS"
        )
        print(
            "------------------------------------------------"
        )

        print(
            "||a_G|| [m/s^2]"
        )

        print(
            f"  p50 : {accel_p50:.6f}"
        )

        print(
            f"  p95 : {accel_p95:.6f}"
        )

        print(
            f"  p99 : {accel_p99:.6f}"
        )

        print(
            f"  max : {accel_max:.6f}"
        )

        print()

        print(
            "||dL_G/dt||"
        )

        print(
            f"  p50 : {mom_p50:.6f}"
        )

        print(
            f"  p95 : {mom_p95:.6f}"
        )

        print(
            f"  p99 : {mom_p99:.6f}"
        )

        print(
            f"  max : {mom_max:.6f}"
        )

        print()

        print(
            "UNILATERAL SLACK DIAGNOSTICS"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {unilat_p50:.6e}"
        )

        print(
            f"  p95 : {unilat_p95:.6e}"
        )

        print(
            f"  p99 : {unilat_p99:.6e}"
        )

        print(
            f"  max : {unilat_max:.6e}"
        )

        print()

        print(
            "TANGENTIAL FRICTION SLACK DIAGNOSTICS"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {friction_p50:.6e}"
        )

        print(
            f"  p95 : {friction_p95:.6e}"
        )

        print(
            f"  p99 : {friction_p99:.6e}"
        )

        print(
            f"  max : {friction_max:.6e}"
        )

        print()

        print(
            "FEASIBLE-SAMPLE FRICTION UTILIZATION"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"  p50 : {rho_p50:.6f}"
        )

        print(
            f"  p95 : {rho_p95:.6f}"
        )

        print(
            f"  p99 : {rho_p99:.6f}"
        )

        print(
            f"  max : {rho_max:.6f}"
        )

        print()

        print(
            "FEASIBLE-SAMPLE NUMERICS"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"Max equality residual  : "
            f"{max_equality_residual:.3e}"
        )

        print(
            f"Max inequality viol.   : "
            f"{max_inequality_violation:.3e}"
        )

        # ====================================================
        # FIRST OCCURRENCES
        # ====================================================

        groups = (
            (
                "First equality infeasible",
                equality_infeasible,
            ),
            (
                "First unilateral infeasible",
                unilateral_infeasible,
            ),
            (
                "First friction infeasible",
                friction_infeasible,
            ),
            (
                "First solver failure",
                solver_failures,
            ),
            (
                "First no-contact sample",
                no_contact,
            ),
        )

        print()

        print(
            "FIRST OCCURRENCES"
        )
        print(
            "------------------------------------------------"
        )

        for label, group in groups:

            if len(
                group
            ) == 0:

                continue

            first = (
                group[
                    0
                ]
            )

            print(
                f"{label:28s}: "
                f"t={first.time:.6f} s"
            )

            print(
                f"{'':28s}  "
                f"Nc={first.number_contacts}, "
                f"rank={first.equality_rank}, "
                f"LS resid="
                f"{first.equality_ls_residual:.3e}"
            )

            print(
                f"{'':28s}  "
                f"sigma_min="
                f"{first.equality_sigma_min:.3e}, "
                f"cond="
                f"{first.equality_condition_number:.3e}"
            )

            print(
                f"{'':28s}  "
                f"contacts L/R="
                f"{first.left_contact_count}/"
                f"{first.right_contact_count}, "
                f"span L/R="
                f"{first.left_contact_span:.4f}/"
                f"{first.right_contact_span:.4f} m"
            )

            if np.isfinite(
                first.max_unilateral_slack
            ):

                print(
                    f"{'':28s}  "
                    f"max unilateral slack="
                    f"{first.max_unilateral_slack:.3e}"
                )

            if np.isfinite(
                first.max_friction_slack
            ):

                print(
                    f"{'':28s}  "
                    f"max friction slack="
                    f"{first.max_friction_slack:.3e}"
                )

            print(
                f"{'':28s}  "
                f"status="
                f"{first.solver_status}"
            )

        print(
            "================================================"
        )
        print()
