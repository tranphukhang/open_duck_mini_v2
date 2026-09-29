# step_timing_adaptation/contact_force_reconstruction.py

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MATRIX_RCOND = 1.0e-10

DYNAMICS_EQUALITY_TOLERANCE = 1.0e-8
UNILATERAL_FORCE_TOLERANCE = 1.0e-10

BASE_DOF = 6


# ============================================================
# FIXED FOOT CONTACT MODEL
# ============================================================
#
# Each stance foot is represented by four fixed candidate
# point contacts.
#
# The rectangle is centered at the existing MuJoCo foot site:
#
#     left_foot
#     right_foot
#
# Rectangle dimensions:
#
#     x direction: 30 mm
#     y direction: 15 mm
#
# Therefore the local coordinates relative to the foot site are:
#
#     (+15 mm, +7.5 mm, 0)
#     (+15 mm, -7.5 mm, 0)
#     (-15 mm, +7.5 mm, 0)
#     (-15 mm, -7.5 mm, 0)
#
# IMPORTANT:
#
# These are candidate contact points.
#
# They do NOT mean that all four points must carry positive force.
# The optimization only imposes:
#
#     Fz_i >= 0
#
# so one or more points may naturally obtain zero normal force.
# ============================================================

FOOT_CONTACT_LENGTH_X = 0.030
FOOT_CONTACT_WIDTH_Y = 0.015

FOOT_CONTACT_HALF_LENGTH_X = (
    0.5 * FOOT_CONTACT_LENGTH_X
)

FOOT_CONTACT_HALF_WIDTH_Y = (
    0.5 * FOOT_CONTACT_WIDTH_Y
)

FOOT_CONTACT_POINTS_LOCAL = np.array(
    [
        [
            +FOOT_CONTACT_HALF_LENGTH_X,
            +FOOT_CONTACT_HALF_WIDTH_Y,
            0.0,
        ],
        [
            +FOOT_CONTACT_HALF_LENGTH_X,
            -FOOT_CONTACT_HALF_WIDTH_Y,
            0.0,
        ],
        [
            -FOOT_CONTACT_HALF_LENGTH_X,
            +FOOT_CONTACT_HALF_WIDTH_Y,
            0.0,
        ],
        [
            -FOOT_CONTACT_HALF_LENGTH_X,
            -FOOT_CONTACT_HALF_WIDTH_Y,
            0.0,
        ],
    ],
    dtype=float,
)

NUMBER_CONTACT_POINTS_PER_FOOT = 4


# ============================================================
# OPTIONAL FRICTION CONE
# ============================================================
#
# False:
#
#     min 1/2 ||f_c||^2
#
#     subject to
#
#         J_c,b^T f_c = M_b qdd + h_b
#         Fz_i >= 0
#
#
# True:
#
# additionally impose
#
#     sqrt(Fx_i^2 + Fy_i^2) <= mu_i Fz_i
#
# The default remains False so the first implementation only
# checks the unilateral contact condition.
# ============================================================

ENABLE_FRICTION_CONE = False

FRICTION_CONE_TOLERANCE = 1.0e-8
FRICTION_SOLVER_MAX_ITERATIONS = 300
FRICTION_SOLVER_FTOL = 1.0e-12


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
    """
    One state of the imposed kinematic trajectory.

    stance_side is supplied by the walking state machine.

    It is NOT inferred from MuJoCo collision detection.
    """

    time: float

    qpos: np.ndarray
    qvel: np.ndarray

    stance_side: str


@dataclass(frozen=True)
class PointContactForce:
    """
    Force reconstructed at one fixed candidate contact point.

    position_world:
        fixed foot contact point transformed into world frame.

    force_world:
        force applied by the ground to the robot.
    """

    foot_side: str

    position_world: np.ndarray
    force_world: np.ndarray

    friction_coefficient: float
    friction_ratio: float


@dataclass(frozen=True)
class ContactForceResult:
    """
    Contact-force reconstruction result at one sample.
    """

    time: float

    stance_side: str

    # Always 4 for the current single-support model.
    number_contacts: int

    # Diagnostic only.
    #
    # True:
    #     MuJoCo also detects collision between the planned
    #     stance foot and floor.
    #
    # False:
    #     planner still defines the stance foot and the four
    #     fixed points are still used, but the geometry should
    #     be inspected.
    collision_detected: bool

    dynamics_rank: int
    dynamics_condition_number: float

    qacc: np.ndarray

    base_required_generalized_force: np.ndarray

    dynamics_residual_inf: float
    dynamics_residual_l2: float

    unilateral_feasible: bool

    friction_cone_enabled: bool
    friction_cone_feasible: bool | None

    max_friction_ratio: float

    point_contact_forces: tuple[PointContactForce, ...]

    left_resultant_force_world: np.ndarray
    left_resultant_moment_world: np.ndarray

    right_resultant_force_world: np.ndarray
    right_resultant_moment_world: np.ndarray


# ============================================================
# CONTACT FORCE RECONSTRUCTOR
# ============================================================

class ContactForceReconstructor:

    def __init__(
        self,
        *,
        mj_model,
    ) -> None:

        self.mj_model = mj_model

        self.samples: list[TrajectorySample] = []

        # ====================================================
        # OBJECT IDS
        # ====================================================

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
            self.mj_model.site_bodyid[
                self.left_foot_site_id
            ]
        )

        self.right_foot_body_id = int(
            self.mj_model.site_bodyid[
                self.right_foot_site_id
            ]
        )

        self.floating_base_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            FLOATING_BASE_JOINT_NAME,
        )

        # ====================================================
        # CHECK FLOATING BASE
        # ====================================================

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
                "is not a free joint."
            )

        dof_address = int(
            self.mj_model.jnt_dofadr[
                self.floating_base_joint_id
            ]
        )

        if dof_address != 0:

            raise RuntimeError(
                "This implementation assumes that the first "
                "six qvel coordinates belong to the floating base."
            )

        if self.mj_model.nv < BASE_DOF:

            raise RuntimeError(
                "MuJoCo model has fewer than six generalized DoF."
            )

        # ====================================================
        # FIXED FRICTION COEFFICIENTS
        # ====================================================
        #
        # Contact geometry no longer comes from collision.
        #
        # Therefore friction is also defined independently from
        # individual MuJoCo contact records.
        #
        # We conservatively take the smaller sliding-friction
        # coefficient of the floor and foot geoms.
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

        self.left_friction_coefficient = min(
            floor_mu,
            left_mu,
        )

        self.right_friction_coefficient = min(
            floor_mu,
            right_mu,
        )

        print()
        print(
            "================================================"
        )
        print(
            "FIXED CONTACT MODEL"
        )
        print(
            "================================================"
        )
        print(
            "Contact rectangle:"
            f" {FOOT_CONTACT_LENGTH_X * 1000.0:.1f}"
            " x "
            f"{FOOT_CONTACT_WIDTH_Y * 1000.0:.1f}"
            " mm"
        )
        print(
            "Contact points / stance foot:"
            f" {NUMBER_CONTACT_POINTS_PER_FOOT}"
        )
        print(
            f"Left friction coefficient : "
            f"{self.left_friction_coefficient:.4f}"
        )
        print(
            f"Right friction coefficient: "
            f"{self.right_friction_coefficient:.4f}"
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
                f"Invalid stance_side: {stance_side}"
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

        # Compatibility with different MuJoCo Python APIs.
        if hasattr(
            mj_data,
            "qM",
        ):

            mujoco.mj_fullM(
                self.mj_model,
                M,
                mj_data.qM,
            )

        else:

            mujoco.mj_fullM(
                self.mj_model,
                mj_data,
                M,
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
    # MUJOCO COLLISION DIAGNOSTIC
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
        """
        Collision is used ONLY as a diagnostic.

        It does not create, remove, or move candidate contact points.
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
                f"Invalid stance_side: {stance_side}"
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
        Return the four predefined contact points for the stance foot.

        Local geometry:

             +x

             p1 -------- p2
              |          |
              |   site   |
              |          |
             p3 -------- p4

        The local points are transformed using the actual site pose:

            p_world = p_site + R_world_site @ p_local
        """

        if stance_side == "left":

            site_id = (
                self.left_foot_site_id
            )

            body_id = (
                self.left_foot_body_id
            )

            friction_coefficient = (
                self.left_friction_coefficient
            )

        elif stance_side == "right":

            site_id = (
                self.right_foot_site_id
            )

            body_id = (
                self.right_foot_body_id
            )

            friction_coefficient = (
                self.right_friction_coefficient
            )

        else:

            raise ValueError(
                f"Invalid stance_side: {stance_side}"
            )

        site_position_world = np.asarray(
            mj_data.site_xpos[
                site_id
            ],
            dtype=float,
        ).reshape(
            3
        ).copy()

        R_world_site = np.asarray(
            mj_data.site_xmat[
                site_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        ).copy()

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
                    stance_side,
                    point_world.copy(),
                    body_id,
                    friction_coefficient,
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

        return jacobian


    def _build_contact_jacobian(
        self,
        *,
        mj_data,
        contacts,
    ) -> np.ndarray:

        J_c = np.zeros(
            (
                3 * len(
                    contacts
                ),
                self.mj_model.nv,
            ),
            dtype=float,
        )

        for contact_index, contact in enumerate(
            contacts
        ):

            (
                _,
                position_world,
                body_id,
                _,
            ) = contact

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

            row = (
                3
                *
                contact_index
            )

            J_c[
                row:
                row + 3,
                :
            ] = J_i

        return J_c


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
            not np.isfinite(
                largest
            )
            or
            not np.isfinite(
                smallest
            )
            or
            smallest
            <=
            MATRIX_RCOND
            *
            max(
                largest,
                1.0,
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
    # EQUALITY MINIMUM-NORM SOLVE
    # ========================================================

    @staticmethod
    def _minimum_norm_equality_solution(
        A,
        b,
    ):

        A = np.asarray(
            A,
            dtype=float,
        )

        b = np.asarray(
            b,
            dtype=float,
        )

        x = np.linalg.lstsq(
            A,
            b,
            rcond=MATRIX_RCOND,
        )[0]

        residual = (
            A
            @
            x
            -
            b
        )

        residual_inf = float(
            np.linalg.norm(
                residual,
                ord=np.inf,
            )
        )

        residual_l2 = float(
            np.linalg.norm(
                residual,
                ord=2,
            )
        )

        if (
            residual_inf
            >
            DYNAMICS_EQUALITY_TOLERANCE
        ):

            return (
                None,
                residual_inf,
                residual_l2,
            )

        return (
            x,
            residual_inf,
            residual_l2,
        )


    # ========================================================
    # UNILATERAL MINIMUM-NORM FORCE
    # ========================================================

    def _solve_unilateral_minimum_norm(
        self,
        *,
        A,
        b,
        number_contacts,
    ):
        """
        Solve:

            min 1/2 ||f||^2

            subject to

                A f = b
                Fz_i >= 0

        With four contacts there are only 2^4 = 16 possible
        active sets, so exact enumeration is simple and robust.
        """

        number_variables = (
            3
            *
            number_contacts
        )

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

        contact_indices = tuple(
            range(
                number_contacts
            )
        )

        # ----------------------------------------------------
        # Enumerate active normal-force constraints.
        #
        # Active contact i means:
        #
        #     Fz_i = 0
        # ----------------------------------------------------

        for number_active in range(
            number_contacts
            +
            1
        ):

            for active_contacts in combinations(
                contact_indices,
                number_active,
            ):

                if number_active > 0:

                    E = np.zeros(
                        (
                            number_active,
                            number_variables,
                        ),
                        dtype=float,
                    )

                    for row, contact_index in enumerate(
                        active_contacts
                    ):

                        E[
                            row,
                            3 * contact_index + 2,
                        ] = 1.0

                    C = np.vstack(
                        (
                            A,
                            E,
                        )
                    )

                    d = np.concatenate(
                        (
                            b,
                            np.zeros(
                                number_active,
                                dtype=float,
                            ),
                        )
                    )

                else:

                    C = A
                    d = b

                (
                    force,
                    _,
                    _,
                ) = (
                    self._minimum_norm_equality_solution(
                        C,
                        d,
                    )
                )

                if force is None:

                    continue

                normal_forces = (
                    force[
                        2::3
                    ]
                )

                if np.any(
                    normal_forces
                    <
                    -UNILATERAL_FORCE_TOLERANCE
                ):

                    continue

                # Remove tiny numerical negative values.
                force = force.copy()

                for contact_index in range(
                    number_contacts
                ):

                    z_index = (
                        3
                        *
                        contact_index
                        +
                        2
                    )

                    if (
                        force[
                            z_index
                        ]
                        <
                        0.0
                        and
                        force[
                            z_index
                        ]
                        >=
                        -UNILATERAL_FORCE_TOLERANCE
                    ):

                        force[
                            z_index
                        ] = 0.0

                residual = (
                    A
                    @
                    force
                    -
                    b
                )

                residual_inf = float(
                    np.linalg.norm(
                        residual,
                        ord=np.inf,
                    )
                )

                residual_l2 = float(
                    np.linalg.norm(
                        residual,
                        ord=2,
                    )
                )

                if (
                    residual_inf
                    >
                    DYNAMICS_EQUALITY_TOLERANCE
                ):

                    continue

                force_norm_squared = float(
                    force
                    @
                    force
                )

                if (
                    force_norm_squared
                    <
                    best_norm_squared
                ):

                    best_force = (
                        force.copy()
                    )

                    best_norm_squared = (
                        force_norm_squared
                    )

                    best_residual_inf = (
                        residual_inf
                    )

                    best_residual_l2 = (
                        residual_l2
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

        force = np.asarray(
            contact_force_vector,
            dtype=float,
        )

        friction_coefficients = np.asarray(
            friction_coefficients,
            dtype=float,
        )

        number_contacts = (
            friction_coefficients.size
        )

        ratios = np.full(
            number_contacts,
            np.nan,
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

            Fx = float(
                force[
                    column
                ]
            )

            Fy = float(
                force[
                    column
                    +
                    1
                ]
            )

            Fz = float(
                force[
                    column
                    +
                    2
                ]
            )

            mu = float(
                friction_coefficients[
                    contact_index
                ]
            )

            tangential_force = float(
                np.hypot(
                    Fx,
                    Fy,
                )
            )

            denominator = (
                mu
                *
                max(
                    Fz,
                    0.0,
                )
            )

            if denominator > 1.0e-12:

                ratios[
                    contact_index
                ] = (
                    tangential_force
                    /
                    denominator
                )

            elif tangential_force <= 1.0e-12:

                ratios[
                    contact_index
                ] = 0.0

            else:

                ratios[
                    contact_index
                ] = float(
                    "inf"
                )

        return ratios


    # ========================================================
    # OPTIONAL FRICTION-CONE SOLVE
    # ========================================================

    def _solve_friction_cone_minimum_norm(
        self,
        *,
        A,
        b,
        number_contacts,
        friction_coefficients,
        initial_force,
    ):
        """
        Optional nonlinear constrained solve:

            min 1/2 ||f||^2

            A f = b
            Fz >= 0
            ||Ft|| <= mu Fz
        """

        try:

            from scipy.optimize import minimize

        except ImportError as exc:

            raise RuntimeError(
                "ENABLE_FRICTION_CONE=True requires scipy."
            ) from exc

        A = np.asarray(
            A,
            dtype=float,
        )

        b = np.asarray(
            b,
            dtype=float,
        )

        mu = np.asarray(
            friction_coefficients,
            dtype=float,
        )

        x0 = np.asarray(
            initial_force,
            dtype=float,
        ).copy()

        def objective(
            force,
        ):

            return (
                0.5
                *
                float(
                    force
                    @
                    force
                )
            )

        def objective_jacobian(
            force,
        ):

            return np.asarray(
                force,
                dtype=float,
            )

        def equality_constraint(
            force,
        ):

            return (
                A
                @
                force
                -
                b
            )

        def inequality_constraint(
            force,
        ):

            values = []

            for contact_index in range(
                number_contacts
            ):

                column = (
                    3
                    *
                    contact_index
                )

                Fx = float(
                    force[
                        column
                    ]
                )

                Fy = float(
                    force[
                        column
                        +
                        1
                    ]
                )

                Fz = float(
                    force[
                        column
                        +
                        2
                    ]
                )

                tangential = float(
                    np.hypot(
                        Fx,
                        Fy,
                    )
                )

                # Fz >= 0
                values.append(
                    Fz
                )

                # mu*Fz - ||Ft|| >= 0
                values.append(
                    mu[
                        contact_index
                    ]
                    *
                    Fz
                    -
                    tangential
                )

            return np.asarray(
                values,
                dtype=float,
            )

        result = minimize(
            objective,
            x0,
            jac=objective_jacobian,
            method="SLSQP",
            constraints=[
                {
                    "type": "eq",
                    "fun": equality_constraint,
                },
                {
                    "type": "ineq",
                    "fun": inequality_constraint,
                },
            ],
            options={
                "maxiter":
                    FRICTION_SOLVER_MAX_ITERATIONS,
                "ftol":
                    FRICTION_SOLVER_FTOL,
                "disp":
                    False,
            },
        )

        if not result.success:

            return (
                None,
                float(
                    "nan"
                ),
                float(
                    "nan"
                ),
            )

        force = np.asarray(
            result.x,
            dtype=float,
        )

        residual = (
            A
            @
            force
            -
            b
        )

        residual_inf = float(
            np.linalg.norm(
                residual,
                ord=np.inf,
            )
        )

        residual_l2 = float(
            np.linalg.norm(
                residual,
                ord=2,
            )
        )

        if (
            residual_inf
            >
            DYNAMICS_EQUALITY_TOLERANCE
        ):

            return (
                None,
                residual_inf,
                residual_l2,
            )

        inequality = (
            inequality_constraint(
                force
            )
        )

        if np.min(
            inequality
        ) < -FRICTION_CONE_TOLERANCE:

            return (
                None,
                residual_inf,
                residual_l2,
            )

        return (
            force,
            residual_inf,
            residual_l2,
        )


    # ========================================================
    # INVALID RESULT
    # ========================================================

    def _invalid_result(
        self,
        *,
        sample,
        qacc,
        collision_detected,
        dynamics_rank,
        condition_number,
        base_required,
        unilateral_feasible,
        friction_cone_feasible,
    ):

        nan3 = np.full(
            3,
            np.nan,
            dtype=float,
        )

        return ContactForceResult(
            time=float(
                sample.time
            ),
            stance_side=(
                sample.stance_side
            ),
            number_contacts=(
                NUMBER_CONTACT_POINTS_PER_FOOT
            ),
            collision_detected=bool(
                collision_detected
            ),
            dynamics_rank=int(
                dynamics_rank
            ),
            dynamics_condition_number=float(
                condition_number
            ),
            qacc=np.asarray(
                qacc,
                dtype=float,
            ).copy(),
            base_required_generalized_force=np.asarray(
                base_required,
                dtype=float,
            ).copy(),
            dynamics_residual_inf=float(
                "nan"
            ),
            dynamics_residual_l2=float(
                "nan"
            ),
            unilateral_feasible=bool(
                unilateral_feasible
            ),
            friction_cone_enabled=(
                ENABLE_FRICTION_CONE
            ),
            friction_cone_feasible=(
                friction_cone_feasible
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


    # ========================================================
    # SOLVE ONE SAMPLE
    # ========================================================

    def _solve_one(
        self,
        *,
        sample,
        qacc,
        scratch_data,
    ) -> ContactForceResult:

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
        # DYNAMICS
        # ====================================================
        #
        # M qdd + qfrc_bias
        #
        #     = qfrc_passive
        #       + S^T tau
        #       + J_c^T f_c
        #
        # Floating base:
        #
        # M_b qdd + h_b
        #
        #     = J_c,b^T f_c
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

        base_required = (
            required_generalized_force[
                0:
                BASE_DOF
            ].copy()
        )

        # ====================================================
        # PLANNED CONTACT MODE
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

        # ====================================================
        # CONTACT JACOBIAN
        # ====================================================

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

        # J_c:
        #
        #     12 x nv
        #
        # J_c,b^T:
        #
        #     6 x 12

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

        # ====================================================
        # UNILATERAL FORCE SOLVE
        # ====================================================

        (
            unilateral_force,
            residual_inf,
            residual_l2,
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

        if unilateral_force is None:

            return self._invalid_result(
                sample=(
                    sample
                ),
                qacc=(
                    qacc
                ),
                collision_detected=(
                    collision_detected
                ),
                dynamics_rank=(
                    dynamics_rank
                ),
                condition_number=(
                    condition_number
                ),
                base_required=(
                    base_required
                ),
                unilateral_feasible=False,
                friction_cone_feasible=(
                    False
                    if
                    ENABLE_FRICTION_CONE
                    else
                    None
                ),
            )

        # ====================================================
        # OPTIONAL FRICTION CONE
        # ====================================================

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
                        unilateral_force
                    ),
                )
            )

            friction_cone_feasible = (
                contact_force_vector
                is not None
            )

            if not friction_cone_feasible:

                return self._invalid_result(
                    sample=(
                        sample
                    ),
                    qacc=(
                        qacc
                    ),
                    collision_detected=(
                        collision_detected
                    ),
                    dynamics_rank=(
                        dynamics_rank
                    ),
                    condition_number=(
                        condition_number
                    ),
                    base_required=(
                        base_required
                    ),
                    unilateral_feasible=True,
                    friction_cone_feasible=False,
                )

        else:

            contact_force_vector = (
                unilateral_force
            )

            friction_cone_feasible = None

        # ====================================================
        # FRICTION DIAGNOSTIC
        # ====================================================

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

        finite_ratios = (
            friction_ratios[
                np.isfinite(
                    friction_ratios
                )
            ]
        )

        if finite_ratios.size > 0:

            max_friction_ratio = float(
                np.max(
                    finite_ratios
                )
            )

        elif np.any(
            np.isinf(
                friction_ratios
            )
        ):

            max_friction_ratio = float(
                "inf"
            )

        else:

            max_friction_ratio = float(
                "nan"
            )

        # ====================================================
        # RESULTANTS
        # ====================================================

        left_reference = np.asarray(
            scratch_data.site_xpos[
                self.left_foot_site_id
            ],
            dtype=float,
        ).copy()

        right_reference = np.asarray(
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

        for contact_index, contact in enumerate(
            contacts
        ):

            (
                foot_side,
                position_world,
                _,
                friction_coefficient,
            ) = contact

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

            point_force = (
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

            point_forces.append(
                point_force
            )

            if foot_side == "left":

                left_force += (
                    force_world
                )

                left_moment += np.cross(
                    (
                        position_world
                        -
                        left_reference
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
                        right_reference
                    ),
                    force_world,
                )

        return ContactForceResult(
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

        # Offline acceleration reconstruction.
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
                "No contact force results."
            )

            return

        total = len(
            results
        )

        unilateral_feasible_count = sum(
            result.unilateral_feasible
            for result
            in results
        )

        collision_detected_count = sum(
            result.collision_detected
            for result
            in results
        )

        ranks = np.asarray(
            [
                result.dynamics_rank
                for result
                in results
            ],
            dtype=int,
        )

        residuals = np.asarray(
            [
                result.dynamics_residual_inf
                for result
                in results
            ],
            dtype=float,
        )

        finite_residuals = (
            residuals[
                np.isfinite(
                    residuals
                )
            ]
        )

        print()
        print(
            "================================================"
        )
        print(
            "FIXED-POINT CONTACT FORCE RECONSTRUCTION"
        )
        print(
            "================================================"
        )

        print(
            f"Samples                  : {total}"
        )

        print(
            "Fixed contacts / stance : "
            f"{NUMBER_CONTACT_POINTS_PER_FOOT}"
        )

        print(
            "Unilateral feasible      : "
            f"{unilateral_feasible_count}/{total}"
        )

        print(
            "Planned stance collision : "
            f"{collision_detected_count}/{total}"
        )

        print(
            "Collision mismatch        : "
            f"{total - collision_detected_count}/{total}"
        )

        print(
            "Dynamics rank:"
        )

        for rank in sorted(
            set(
                ranks.tolist()
            )
        ):

            count = int(
                np.count_nonzero(
                    ranks
                    ==
                    rank
                )
            )

            print(
                f"  rank {rank}: {count}"
            )

        if finite_residuals.size > 0:

            print(
                "Dynamics residual "
                "||A f - b||_inf:"
            )

            print(
                f"  p50 : "
                f"{np.percentile(finite_residuals, 50):.6e}"
            )

            print(
                f"  p95 : "
                f"{np.percentile(finite_residuals, 95):.6e}"
            )

            print(
                f"  max : "
                f"{np.max(finite_residuals):.6e}"
            )

        friction_ratios = np.asarray(
            [
                result.max_friction_ratio
                for result
                in results
            ],
            dtype=float,
        )

        finite_friction = (
            friction_ratios[
                np.isfinite(
                    friction_ratios
                )
            ]
        )

        if finite_friction.size > 0:

            print(
                "Max friction ratio:"
            )

            print(
                f"  p50 : "
                f"{np.percentile(finite_friction, 50):.4f}"
            )

            print(
                f"  p95 : "
                f"{np.percentile(finite_friction, 95):.4f}"
            )

            print(
                f"  max : "
                f"{np.max(finite_friction):.4f}"
            )

        print(
            "================================================"
        )
        print()


# ============================================================
# PLOTTING
# ============================================================

def plot_contact_force_results(
    results,
    *,
    mj_model=None,
    show=False,
):
    """
    Plot resultant reconstructed force for each foot.

    mj_model is retained only for compatibility with run.py.
    """

    import matplotlib.pyplot as plt

    _ = mj_model

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

    component_names = (
        "Fx",
        "Fy",
        "Fz",
    )

    # ========================================================
    # FORCE COMPONENTS
    # ========================================================

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
            "Reconstructed Contact Force - "
            f"{component_name}"
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
    # FORCE NORM
    # ========================================================

    left_force_norm = np.linalg.norm(
        left_force,
        axis=1,
    )

    right_force_norm = np.linalg.norm(
        right_force,
        axis=1,
    )

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
    # FRICTION RATIO
    # ========================================================

    max_friction_ratio = np.asarray(
        [
            result.max_friction_ratio
            for result
            in results
        ],
        dtype=float,
    )

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
        label="max point-contact ratio",
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="friction cone boundary",
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
    # COLLISION DIAGNOSTIC
    # ========================================================

    collision = np.asarray(
        [
            1.0
            if result.collision_detected
            else
            0.0
            for result
            in results
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(
            11,
            3.5,
        )
    )

    axis.step(
        time,
        collision,
        where="post",
        linewidth=1.2,
    )

    axis.set_title(
        "Planned Stance Foot - MuJoCo Collision Diagnostic"
    )

    axis.set_xlabel(
        "Time (s)"
    )

    axis.set_ylabel(
        "Collision"
    )

    axis.set_yticks(
        [
            0.0,
            1.0,
        ]
    )

    axis.set_yticklabels(
        [
            "No",
            "Yes",
        ]
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    if show:

        plt.show()