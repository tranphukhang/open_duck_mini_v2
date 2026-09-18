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

# Centroidal equality is considered representable by the current
# set of contact points when the least-squares residual is below
# this threshold.
EQUALITY_LS_TOLERANCE = 1.0e-6

# Residual tolerances used to validate a returned hard-QP solution.
EQUALITY_TOLERANCE = 1.0e-5
INEQUALITY_TOLERANCE = 1.0e-7

# Friction slack above this value is considered genuine violation
# of the friction pyramid.
FRICTION_SLACK_TOLERANCE = 1.0e-6

NORMAL_FORCE_TOLERANCE = 1.0e-9

# Numerical rank threshold.
MATRIX_RCOND = 1.0e-10


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
    One foot-ground point contact extracted from mjData.contact.

    contact_frame_world stores three ROW vectors:

        row 0 : normal direction, oriented ground -> foot
        row 1 : first tangent direction
        row 2 : second tangent direction

    Therefore:

        f_contact = contact_frame_world @ f_world

    gives:

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
    Raw data recorded from the kinematic trajectory and MuJoCo.
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
    Optimal point-contact force returned by the hard QP.
    """

    foot_side: str

    position_world: np.ndarray

    force_world: np.ndarray

    # [normal, tangent_1, tangent_2]
    force_contact: np.ndarray

    friction_tangent_1: float
    friction_tangent_2: float

    friction_utilization: float


@dataclass(frozen=True)
class ContactStabilityResult:
    """
    Diagnostic result at one trajectory sample.
    """

    time: float

    classification: str

    feasible: bool
    solver_status: str

    number_contacts: int

    # Rank / representability of:
    #
    #     A_eq f = b_eq
    #
    equality_rank: int
    equality_ls_residual: float

    # Derivative quantities used by centroidal dynamics.
    com_acceleration_world: np.ndarray
    angular_momentum_rate_world: np.ndarray

    required_contact_force_world: np.ndarray

    # Residuals of the hard-QP solution.
    equality_residual: float
    inequality_violation: float

    # Result of the soft-friction fallback QP.
    max_friction_slack: float

    max_friction_utilization: float

    # Resultant wrench of each foot.
    #
    # Force is expressed in world coordinates.
    # Moment is computed about the corresponding foot-site
    # position recorded by run.py.
    left_resultant_force_world: np.ndarray
    left_resultant_moment_world: np.ndarray

    right_resultant_force_world: np.ndarray
    right_resultant_moment_world: np.ndarray

    contact_forces: tuple[ContactForceSolution, ...]


# ============================================================
# BASIC HELPERS
# ============================================================

def skew(
    vector,
) -> np.ndarray:
    """
    Return [r]_x such that

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


# ============================================================
# CONTACT STABILITY CHECKER
# ============================================================

class ContactStabilityChecker:
    """
    Weak contact-stability checker.

    Current assumptions:

        1. Joint-torque limits are ignored.

        2. Only foot-ground collision geoms are enabled.

        3. MuJoCo provides:
               - whole-body CoM position/velocity,
               - centroidal angular momentum,
               - point-contact geometry,
               - contact frame,
               - friction coefficients.

        4. MuJoCo contact forces are NOT used to decide
           feasibility.

        5. Every foot-ground contact is modeled as a 3-D point
           force.

        6. Coulomb friction is approximated by a friction pyramid.

    Centroidal dynamics:

        sum_i f_i
            = m (a_G - g)

        sum_i (p_Ci - p_G) x f_i
            = dL_G/dt

    Friction pyramid:

        f_n >= 0

        |f_t1| <= mu_1 f_n
        |f_t2| <= mu_2 f_n

    Diagnostic logic:

        A) First test the equality alone using least squares.

           If A_eq f = b_eq cannot be satisfied, the sample is
           classified as EQUALITY_INFEASIBLE.

        B) If the equality is representable, solve the hard QP.

           If successful -> FEASIBLE.

        C) If the hard QP fails, solve a second QP with nonnegative
           friction slacks.

           Positive required friction slack
               -> FRICTION_INFEASIBLE

           Near-zero friction slack but hard QP failed
               -> SOLVER_FAILURE
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
        # ROBOT MASS / GRAVITY FROM MUJOCO
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
        # RECORDED TRAJECTORY
        # ====================================================

        self.samples: list[
            ContactKinematicsSample
        ] = []

        # ====================================================
        # QP SOLVER CACHE
        # ====================================================

        self._hard_solver_cache = {}
        self._soft_solver_cache = {}


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
        Support both newer contact.geom[0:2] and older
        contact.geom1/contact.geom2 bindings.
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

            # ------------------------------------------------
            # Ground must participate.
            # ------------------------------------------------

            if (
                self.floor_geom_id
                not in
                geom_pair
            ):

                continue

            # ------------------------------------------------
            # Identify foot.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Keep only force-generating contacts.
            # ------------------------------------------------

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

            contact_dimension = int(
                contact.dim
            )

            if (
                contact_dimension
                <
                3
            ):

                raise RuntimeError(
                    "Foot-ground contact has condim < 3. "
                    "The checker expects a regular 3-D "
                    "frictional point contact."
                )

            # ------------------------------------------------
            # MuJoCo contact frame:
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

            # ------------------------------------------------
            # Orient normal consistently ground -> foot.
            #
            # Tangential inequalities are symmetric, so their
            # sign does not affect the friction pyramid.
            # ------------------------------------------------

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

            contact_frame_world = np.vstack(
                (
                    normal_ground_to_foot,
                    frame[1],
                    frame[2],
                )
            )

            # ------------------------------------------------
            # Friction from MuJoCo's actual contact object.
            # ------------------------------------------------

            friction = np.asarray(
                contact.friction,
                dtype=float,
            ).reshape(
                -1
            )

            if friction.size < 2:

                raise RuntimeError(
                    "MuJoCo contact friction vector "
                    "does not contain two tangential "
                    "coefficients."
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
                    "Invalid MuJoCo tangential "
                    "friction coefficient."
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
                "Contact-stability sample time "
                "must be strictly increasing."
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
    # CONTACT MATRICES
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

        and

            A_ineq f <= 0
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

        number_variables = (
            3
            *
            number_contacts
        )

        A_equal = np.zeros(
            (
                6,
                number_variables,
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

            # Linear momentum.
            A_equal[
                0:3,
                column:
                column + 3,
            ] = np.eye(
                3,
                dtype=float,
            )

            # Angular momentum.
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

        A_inequality = np.zeros(
            (
                5
                *
                number_contacts,
                number_variables,
            ),
            dtype=float,
        )

        for contact_index, contact in enumerate(
            contacts
        ):

            mu_1 = (
                contact.friction_tangent_1
            )

            mu_2 = (
                contact.friction_tangent_2
            )

            # Local ordering:
            #
            #     [f_n, f_t1, f_t2]
            #
            # Pyramid:
            #
            #    -f_n                  <= 0
            #    -mu1*f_n + f_t1      <= 0
            #    -mu1*f_n - f_t1      <= 0
            #    -mu2*f_n + f_t2      <= 0
            #    -mu2*f_n - f_t2      <= 0

            friction_pyramid_local = np.array(
                [
                    [
                        -1.0,
                        0.0,
                        0.0,
                    ],
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

            # f_contact = C f_world
            block = (
                friction_pyramid_local
                @
                contact.contact_frame_world
            )

            row = (
                5
                *
                contact_index
            )

            column = (
                3
                *
                contact_index
            )

            A_inequality[
                row:
                row + 5,
                column:
                column + 3,
            ] = block

        return (
            acceleration,
            momentum_rate,
            required_force,
            A_equal,
            b_equal,
            A_inequality,
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
        Determine whether the current contact geometry can reproduce
        the required centroidal wrench even before friction is applied.
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

        residual_vector = (
            A_equal
            @
            least_squares_solution
            -
            b_equal
        )

        residual = float(
            np.max(
                np.abs(
                    residual_vector
                )
            )
        )

        return (
            rank,
            residual,
            least_squares_solution,
        )


    # ========================================================
    # HARD QP SOLVER
    # ========================================================

    def _get_hard_solver(
        self,
        number_contacts: int,
    ):

        number_contacts = int(
            number_contacts
        )

        if (
            number_contacts
            in
            self._hard_solver_cache
        ):

            return (
                self._hard_solver_cache[
                    number_contacts
                ]
            )

        number_variables = (
            3
            *
            number_contacts
        )

        number_constraints = (
            6
            +
            5
            *
            number_contacts
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
            "max_iter": 1000,
        }

        solver = ca.conic(
            (
                "contact_hard_"
                f"{number_contacts}"
            ),
            "qrqp",
            qp_structure,
            options,
        )

        self._hard_solver_cache[
            number_contacts
        ] = solver

        return solver


    # ========================================================
    # SOFT-FRICTION DIAGNOSTIC QP
    # ========================================================

    def _get_soft_solver(
        self,
        number_contacts: int,
    ):
        """
        Soft-friction fallback QP.

        Variables:

            y = [f, s]

        where s >= 0 softens every friction-pyramid inequality:

            A_ineq f - s <= 0

        Centroidal equalities remain HARD.

        If the equality is feasible, this problem should normally
        remain feasible. A positive optimal slack indicates that the
        original friction pyramid excludes the required wrench.
        """

        number_contacts = int(
            number_contacts
        )

        if (
            number_contacts
            in
            self._soft_solver_cache
        ):

            return (
                self._soft_solver_cache[
                    number_contacts
                ]
            )

        number_force_variables = (
            3
            *
            number_contacts
        )

        number_slacks = (
            5
            *
            number_contacts
        )

        number_variables = (
            number_force_variables
            +
            number_slacks
        )

        number_constraints = (
            6
            +
            number_slacks
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
            "max_iter": 2000,
        }

        solver = ca.conic(
            (
                "contact_soft_"
                f"{number_contacts}"
            ),
            "qrqp",
            qp_structure,
            options,
        )

        self._soft_solver_cache[
            number_contacts
        ] = solver

        return solver


    # ========================================================
    # RUN HARD QP
    # ========================================================

    def _solve_hard_qp(
        self,
        *,
        A_equal,
        b_equal,
        A_inequality,
    ):
        number_variables = (
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

        lower = np.concatenate(
            (
                b_equal,
                np.full(
                    number_inequalities,
                    -np.inf,
                    dtype=float,
                ),
            )
        )

        upper = np.concatenate(
            (
                b_equal,
                np.zeros(
                    number_inequalities,
                    dtype=float,
                ),
            )
        )

        # CasADi:
        #
        #     0.5 x^T H x + g^T x
        #
        # H = 2I -> sum ||f_i||^2.
        H = (
            2.0
            *
            np.eye(
                number_variables,
                dtype=float,
            )
        )

        g = np.zeros(
            number_variables,
            dtype=float,
        )

        solver = (
            self._get_hard_solver(
                number_variables
                //
                3
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
                lower
            ),
            uba=ca.DM(
                upper
            ),
            lbx=ca.DM(
                np.full(
                    number_variables,
                    -np.inf,
                )
            ),
            ubx=ca.DM(
                np.full(
                    number_variables,
                    +np.inf,
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
            number_variables
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

        inequality_value = (
            A_inequality
            @
            solution
        )

        inequality_violation = float(
            max(
                0.0,
                np.max(
                    inequality_value
                ),
            )
        )

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

            return (
                False,
                (
                    f"{status}"
                    "|NUMERICAL_RESIDUAL"
                ),
                solution,
                equality_residual,
                inequality_violation,
            )

        return (
            True,
            status,
            solution,
            equality_residual,
            inequality_violation,
        )


    # ========================================================
    # RUN SOFT-FRICTION QP
    # ========================================================

    def _solve_soft_friction_qp(
        self,
        *,
        A_equal,
        b_equal,
        A_inequality,
    ):
        number_force_variables = (
            A_equal.shape[
                1
            ]
        )

        number_slacks = (
            A_inequality.shape[
                0
            ]
        )

        number_variables = (
            number_force_variables
            +
            number_slacks
        )

        # ----------------------------------------------------
        # Equality rows:
        #
        #     A_eq f = b_eq
        # ----------------------------------------------------

        A_equal_soft = np.hstack(
            (
                A_equal,
                np.zeros(
                    (
                        A_equal.shape[
                            0
                        ],
                        number_slacks,
                    ),
                    dtype=float,
                ),
            )
        )

        # ----------------------------------------------------
        # Soft friction:
        #
        #     A_ineq f - s <= 0
        # ----------------------------------------------------

        A_friction_soft = np.hstack(
            (
                A_inequality,
                -np.eye(
                    number_slacks,
                    dtype=float,
                ),
            )
        )

        A = np.vstack(
            (
                A_equal_soft,
                A_friction_soft,
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

        # ----------------------------------------------------
        # Objective:
        #
        #     epsilon * ||f||^2 + ||s||^2
        #
        # Slack dominates the diagnostic.
        # ----------------------------------------------------

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

        solver = (
            self._get_soft_solver(
                number_force_variables
                //
                3
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
                None,
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
                "SOFT_QP_RETURNED_NAN_INF",
                None,
                float(
                    "inf"
                ),
            )

        force_solution = (
            solution[
                :
                number_force_variables
            ].copy()
        )

        slack_solution = (
            solution[
                number_force_variables:
            ]
        )

        max_slack = float(
            np.max(
                slack_solution
            )
        )

        return (
            True,
            status,
            force_solution,
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
    # GENERIC RESULT CREATOR
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
        max_friction_slack=float("inf"),
    ) -> ContactStabilityResult:

        zero = np.zeros(
            3,
            dtype=float,
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
            max_friction_slack=float(
                max_friction_slack
            ),
            max_friction_utilization=float(
                "inf"
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
            A_inequality,
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
        # 1) EQUALITY-ONLY DIAGNOSTIC
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
        # 2) HARD CONTACT-FEASIBILITY QP
        # ====================================================

        (
            hard_success,
            hard_status,
            hard_solution,
            equality_residual,
            inequality_violation,
        ) = (
            self._solve_hard_qp(
                A_equal=(
                    A_equal
                ),
                b_equal=(
                    b_equal
                ),
                A_inequality=(
                    A_inequality
                ),
            )
        )

        if hard_success:

            return (
                self._build_feasible_result(
                    sample=(
                        sample
                    ),
                    solution=(
                        hard_solution
                    ),
                    solver_status=(
                        hard_status
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
                        equality_residual
                    ),
                    inequality_violation=(
                        inequality_violation
                    ),
                )
            )

        # ====================================================
        # 3) SOFT-FRICTION FALLBACK
        # ====================================================

        (
            soft_success,
            soft_status,
            _,
            max_friction_slack,
        ) = (
            self._solve_soft_friction_qp(
                A_equal=(
                    A_equal
                ),
                b_equal=(
                    b_equal
                ),
                A_inequality=(
                    A_inequality
                ),
            )
        )

        if soft_success:

            if (
                max_friction_slack
                >
                FRICTION_SLACK_TOLERANCE
            ):

                classification = (
                    CLASS_FRICTION_INFEASIBLE
                )

            else:

                classification = (
                    CLASS_SOLVER_FAILURE
                )

            combined_status = (
                f"hard={hard_status}"
                f" | soft={soft_status}"
            )

            return (
                self._diagnostic_result(
                    sample=(
                        sample
                    ),
                    classification=(
                        classification
                    ),
                    solver_status=(
                        combined_status
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
                        equality_residual
                    ),
                    inequality_violation=(
                        inequality_violation
                    ),
                    max_friction_slack=(
                        max_friction_slack
                    ),
                )
            )

        # ====================================================
        # BOTH QP SOLVERS FAILED
        # ====================================================

        combined_status = (
            f"hard={hard_status}"
            f" | soft={soft_status}"
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
                    combined_status
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
                    equality_residual
                ),
                inequality_violation=(
                    inequality_violation
                ),
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

        Interior derivatives use centered differences through
        np.gradient.

        The very first and last sample are explicitly classified as
        DERIVATIVE_BOUNDARY_SKIPPED because their derivative estimates
        are one-sided and should not be used to judge contact
        feasibility.
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
    # SUMMARY UTILITIES
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
        # CONTACT COUNT / EQUALITY RANK HISTOGRAMS
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
        # FEASIBLE-SAMPLE METRICS
        # ====================================================

        if len(
            feasible
        ) > 0:

            max_friction_utilization = max(
                result.max_friction_utilization
                for result
                in feasible
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

            max_friction_utilization = float(
                "nan"
            )

            max_equality_residual = float(
                "nan"
            )

            max_inequality_violation = float(
                "nan"
            )

        # ====================================================
        # PRINT
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
            "FEASIBLE-SAMPLE NUMERICS"
        )
        print(
            "------------------------------------------------"
        )

        print(
            f"Max friction usage     : "
            f"{max_friction_utilization:.6f}"
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
        # FIRST OCCURRENCE OF EACH FAILURE TYPE
        # ====================================================

        groups = (
            (
                "First equality infeasible",
                equality_infeasible,
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
            ) > 0:

                first = (
                    group[
                        0
                    ]
                )

                print(
                    f"{label:27s}: "
                    f"t={first.time:.6f} s"
                )

                print(
                    f"{'':27s}  "
                    f"Nc={first.number_contacts}, "
                    f"rank={first.equality_rank}, "
                    f"LS resid="
                    f"{first.equality_ls_residual:.3e}"
                )

                if np.isfinite(
                    first.max_friction_slack
                ):

                    print(
                        f"{'':27s}  "
                        f"max friction slack="
                        f"{first.max_friction_slack:.3e}"
                    )

                print(
                    f"{'':27s}  "
                    f"status="
                    f"{first.solver_status}"
                )

        print(
            "================================================"
        )
        print()
