import numpy as np
import pinocchio as pin


# ============================================================
# LOCAL IMPORTS
# ============================================================

if __package__:

    from .pinocchio_model import (
        LEFT_FOOT_FRAME,
        RIGHT_FOOT_FRAME,
    )

else:

    from pinocchio_model import (
        LEFT_FOOT_FRAME,
        RIGHT_FOOT_FRAME,
    )


TRUNK_FRAME = "trunk"


# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def _matrix_rank(
    A,
    rcond=1e-10,
):

    A = np.asarray(
        A,
        dtype=float,
    )

    if A.size == 0:

        return 0

    s = np.linalg.svd(
        A,
        compute_uv=False,
    )

    if s.size == 0:

        return 0

    tol = (
        rcond
        *
        max(
            A.shape
        )
        *
        s[0]
    )

    return int(
        np.sum(
            s > tol
        )
    )


def damped_pseudoinverse(
    A,
    damping=1e-8,
    rcond=1e-10,
):

    A = np.asarray(
        A,
        dtype=float,
    )

    if A.ndim != 2:

        raise ValueError(
            "A must be a 2D matrix."
        )

    if damping < 0.0:

        raise ValueError(
            "damping cannot be negative."
        )

    U, s, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    if s.size == 0:

        return np.zeros(
            (
                A.shape[1],
                A.shape[0],
            ),
            dtype=float,
        )

    tol = (
        rcond
        *
        max(
            A.shape
        )
        *
        s[0]
    )

    valid = (
        s > tol
    )

    gains = np.zeros_like(
        s
    )

    if damping == 0.0:

        gains[valid] = (
            1.0
            /
            s[valid]
        )

    else:

        sigma = (
            s[valid]
        )

        gains[valid] = (
            sigma
            /
            (
                sigma**2
                +
                damping**2
            )
        )

    return (
        Vt.T
        @
        np.diag(
            gains
        )
        @
        U.T
    )


def nullspace_basis(
    A,
    rcond=1e-10,
):

    A = np.asarray(
        A,
        dtype=float,
    )

    if A.ndim != 2:

        raise ValueError(
            "A must be a 2D matrix."
        )

    n = (
        A.shape[1]
    )

    if A.shape[0] == 0:

        return np.eye(
            n,
            dtype=float,
        )

    _, s, Vt = np.linalg.svd(
        A,
        full_matrices=True,
    )

    if s.size == 0:

        rank = 0

    else:

        tol = (
            rcond
            *
            max(
                A.shape
            )
            *
            s[0]
        )

        rank = int(
            np.sum(
                s > tol
            )
        )

    return (
        Vt[
            rank:,
            :
        ]
        .T
        .copy()
    )


# ============================================================
# STRICT HIERARCHICAL SOLVER
# ============================================================

def solve_task_hierarchy(
    tasks,
    n_dof,
    damping=1e-8,
    rcond=1e-10,
):

    qdot = np.zeros(
        n_dof,
        dtype=float,
    )

    Z = np.eye(
        n_dof,
        dtype=float,
    )

    diagnostics = []

    for (
        task_name,
        J,
        v,
    ) in tasks:

        J = np.asarray(
            J,
            dtype=float,
        )

        v = np.asarray(
            v,
            dtype=float,
        ).reshape(
            -1
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        if J.ndim != 2:

            raise ValueError(
                f"{task_name}: "
                "J must be 2D."
            )

        if (
            J.shape[1]
            !=
            n_dof
        ):

            raise ValueError(
                f"{task_name}: "
                f"J has {J.shape[1]} columns, "
                f"expected {n_dof}."
            )

        if (
            J.shape[0]
            !=
            v.shape[0]
        ):

            raise ValueError(
                f"{task_name}: "
                "J rows and velocity size differ."
            )

        # ====================================================
        # CURRENT RESIDUAL
        # ====================================================

        residual_before = (
            v
            -
            J
            @
            qdot
        )

        # ====================================================
        # REDUCED TASK
        # ====================================================

        J_reduced = (
            J
            @
            Z
        )

        reduced_rank = (
            _matrix_rank(
                J_reduced,
                rcond=rcond,
            )
        )

        # ====================================================
        # SOLVE ONLY INSIDE CURRENT NULLSPACE
        # ====================================================

        if Z.shape[1] > 0:

            J_reduced_pinv = (
                damped_pseudoinverse(
                    J_reduced,
                    damping=damping,
                    rcond=rcond,
                )
            )

            y = (
                J_reduced_pinv
                @
                residual_before
            )

            qdot = (
                qdot
                +
                Z
                @
                y
            )

        # ====================================================
        # DIAGNOSTIC
        # ====================================================

        residual_after = (
            v
            -
            J
            @
            qdot
        )

        # ====================================================
        # UPDATE NULLSPACE
        # ====================================================

        Z_local = (
            nullspace_basis(
                J_reduced,
                rcond=rcond,
            )
        )

        Z = (
            Z
            @
            Z_local
        )

        diagnostics.append(
            {
                "name":
                    task_name,

                "rows":
                    int(
                        J.shape[0]
                    ),

                "reduced_rank":
                    int(
                        reduced_rank
                    ),

                "residual_before_norm":
                    float(
                        np.linalg.norm(
                            residual_before
                        )
                    ),

                "residual_after_norm":
                    float(
                        np.linalg.norm(
                            residual_after
                        )
                    ),

                "remaining_nullity":
                    int(
                        Z.shape[1]
                    ),
            }
        )

    return (
        qdot,
        diagnostics,
        Z,
    )


# ============================================================
# INPUT CHECKS
# ============================================================

def _vector3(
    value,
    name,
):

    value = np.asarray(
        value,
        dtype=float,
    )

    if value.shape != (
        3,
    ):

        raise ValueError(
            f"{name} must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} must contain finite values."
        )

    return (
        value
    )


def _rotation_matrix(
    value,
    name,
):

    value = np.asarray(
        value,
        dtype=float,
    )

    if value.shape != (
        3,
        3,
    ):

        raise ValueError(
            f"{name} must have shape (3, 3)."
        )

    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} must contain finite values."
        )

    return (
        value
    )


# ============================================================
# TRUNK PITCH TASK
# ============================================================

def build_trunk_pitch_task(
    robot,
    trunk_rotation_ref,
    trunk_orientation_gain,
):

    if trunk_orientation_gain < 0.0:

        raise ValueError(
            "trunk_orientation_gain "
            "cannot be negative."
        )

    R_ref = (
        _rotation_matrix(
            trunk_rotation_ref,
            "trunk_rotation_ref",
        )
    )

    _, R_actual = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    # ========================================================
    # LOCAL ORIENTATION ERROR
    # ========================================================

    rotation_error_local = (
        pin.log3(
            R_actual.T
            @
            R_ref
        )
    )

    omega_y_cmd = (
        trunk_orientation_gain
        *
        float(
            rotation_error_local[1]
        )
    )

    # ========================================================
    # TRUNK LOCAL JACOBIAN
    # ========================================================

    J_local = (
        robot.get_frame_jacobian_local(
            TRUNK_FRAME
        )
    )

    # Pinocchio spatial Jacobian:
    #
    # rows 0:3 -> linear
    # rows 3:6 -> angular
    #
    # local angular-Y -> row 4
    # ========================================================

    J_pitch = (
        J_local[
            4:5,
            robot.walking_velocity_indices,
        ]
    )

    v_pitch = np.array(
        [
            omega_y_cmd
        ],
        dtype=float,
    )

    return (
        J_pitch,
        v_pitch,
        rotation_error_local,
    )


# ============================================================
# SINGLE SUPPORT IK
# ============================================================

def solve_single_support_ik(
    robot,
    q_pin,
    support_side,

    support_position_ref,
    swing_position_ref,
    swing_linear_velocity_ref,

    com_position_ref,
    com_velocity_ref,

    trunk_rotation_ref,

    support_position_gain=25.0,
    swing_position_gain=20.0,
    com_position_gain=10.0,
    trunk_orientation_gain=10.0,

    damping=1e-8,
    rcond=1e-10,
):
    """
    Strict hierarchy during SINGLE SUPPORT:

        P1: support foot 5D
        P2: CoM 3D
        P3: swing foot 5D
        P4: trunk local-Y 1D

    The CoM task is intentionally placed above the swing-foot
    task because the CoM trajectory will be generated by
    LIPM-MPC.

    Expected active walking DoF:

        16

    Expected task dimensions:

        support = 5
        CoM     = 3
        swing   = 5
        trunk   = 1

    Expected final nullity when all reduced tasks are full rank:

        16 - 5 - 3 - 5 - 1 = 2
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    support_position_ref = (
        _vector3(
            support_position_ref,
            "support_position_ref",
        )
    )

    swing_position_ref = (
        _vector3(
            swing_position_ref,
            "swing_position_ref",
        )
    )

    swing_linear_velocity_ref = (
        _vector3(
            swing_linear_velocity_ref,
            "swing_linear_velocity_ref",
        )
    )

    com_position_ref = (
        _vector3(
            com_position_ref,
            "com_position_ref",
        )
    )

    com_velocity_ref = (
        _vector3(
            com_velocity_ref,
            "com_velocity_ref",
        )
    )

    trunk_rotation_ref = (
        _rotation_matrix(
            trunk_rotation_ref,
            "trunk_rotation_ref",
        )
    )

    if support_position_gain < 0.0:

        raise ValueError(
            "support_position_gain "
            "cannot be negative."
        )

    if swing_position_gain < 0.0:

        raise ValueError(
            "swing_position_gain "
            "cannot be negative."
        )

    if com_position_gain < 0.0:

        raise ValueError(
            "com_position_gain "
            "cannot be negative."
        )

    support_side = (
        support_side
        .strip()
        .lower()
    )

    if support_side not in (
        "left",
        "right",
    ):

        raise ValueError(
            "support_side must be "
            "'left' or 'right'."
        )

    # ========================================================
    # CURRENT KINEMATICS
    # ========================================================

    robot.update(
        q_pin
    )

    p_left_actual, _ = (
        robot.get_left_foot_pose()
    )

    p_right_actual, _ = (
        robot.get_right_foot_pose()
    )

    p_com_actual = (
        robot.get_com()
    )

    # ========================================================
    # JACOBIANS
    # ========================================================

    J_left = (
        robot.get_foot_task_jacobian(
            LEFT_FOOT_FRAME,
            active_only=True,
        )
    )

    J_right = (
        robot.get_foot_task_jacobian(
            RIGHT_FOOT_FRAME,
            active_only=True,
        )
    )

    J_com = (
        robot.get_com_jacobian(
            active_only=True,
        )
    )

    # ========================================================
    # SUPPORT / SWING SELECTION
    # ========================================================

    if support_side == "left":

        J_support = (
            J_left
        )

        J_swing = (
            J_right
        )

        p_support_actual = (
            p_left_actual
        )

        p_swing_actual = (
            p_right_actual
        )

        support_name = (
            "left_support"
        )

        swing_name = (
            "right_swing"
        )

    else:

        J_support = (
            J_right
        )

        J_swing = (
            J_left
        )

        p_support_actual = (
            p_right_actual
        )

        p_swing_actual = (
            p_left_actual
        )

        support_name = (
            "right_support"
        )

        swing_name = (
            "left_swing"
        )

    # ========================================================
    # TRACKING ERRORS
    # ========================================================

    support_error = (
        support_position_ref
        -
        p_support_actual
    )

    swing_error = (
        swing_position_ref
        -
        p_swing_actual
    )

    com_error = (
        com_position_ref
        -
        p_com_actual
    )

    # ========================================================
    # TASK VELOCITY COMMANDS
    # ========================================================

    support_linear_cmd = (
        support_position_gain
        *
        support_error
    )

    swing_linear_cmd = (
        swing_linear_velocity_ref
        +
        swing_position_gain
        *
        swing_error
    )

    com_velocity_cmd = (
        com_velocity_ref
        +
        com_position_gain
        *
        com_error
    )

    # ========================================================
    # FOOT 5D COMMANDS
    #
    # 3D position + two orientation components
    # ========================================================

    v_support = np.concatenate(
        [
            support_linear_cmd,

            np.zeros(
                2,
                dtype=float,
            ),
        ]
    )

    v_swing = np.concatenate(
        [
            swing_linear_cmd,

            np.zeros(
                2,
                dtype=float,
            ),
        ]
    )

    # ========================================================
    # TRUNK TASK
    # ========================================================

    (
        J_trunk_pitch,
        v_trunk_pitch,
        trunk_error_local,
    ) = build_trunk_pitch_task(
        robot=(
            robot
        ),

        trunk_rotation_ref=(
            trunk_rotation_ref
        ),

        trunk_orientation_gain=(
            trunk_orientation_gain
        ),
    )

    # ========================================================
    # STRICT PRIORITY
    #
    # IMPORTANT:
    #
    # support > CoM > swing > trunk
    # ========================================================

    tasks = [
        (
            support_name,
            J_support,
            v_support,
        ),

        (
            "com",
            J_com,
            com_velocity_cmd,
        ),

        (
            swing_name,
            J_swing,
            v_swing,
        ),

        (
            "trunk_pitch",
            J_trunk_pitch,
            v_trunk_pitch,
        ),
    ]

    # ========================================================
    # SOLVE
    # ========================================================

    n_active = len(
        robot.walking_velocity_indices
    )

    (
        qdot_active,
        diagnostics,
        final_nullspace,
    ) = solve_task_hierarchy(
        tasks=(
            tasks
        ),

        n_dof=(
            n_active
        ),

        damping=(
            damping
        ),

        rcond=(
            rcond
        ),
    )

    # ========================================================
    # TRACKING DIAGNOSTICS
    # ========================================================

    diagnostics.append(
        {
            "name":
                "tracking_errors",

            "support_position_error_norm":
                float(
                    np.linalg.norm(
                        support_error
                    )
                ),

            "swing_position_error_norm":
                float(
                    np.linalg.norm(
                        swing_error
                    )
                ),

            "com_position_error_norm":
                float(
                    np.linalg.norm(
                        com_error
                    )
                ),

            "trunk_local_y_error_rad":
                float(
                    trunk_error_local[1]
                ),
        }
    )

    # ========================================================
    # ACTIVE -> FULL PINOCCHIO VELOCITY
    # ========================================================

    qdot_full = np.zeros(
        robot.model.nv,
        dtype=float,
    )

    qdot_full[
        robot.walking_velocity_indices
    ] = (
        qdot_active
    )

    return (
        qdot_full,
        diagnostics,
        final_nullspace,
    )


# ============================================================
# DOUBLE SUPPORT IK
# ============================================================

def solve_double_support_ik(
    robot,
    q_pin,

    left_position_ref,
    right_position_ref,

    com_position_ref,
    com_velocity_ref,

    trunk_rotation_ref,

    foot_position_gain=25.0,
    com_position_gain=10.0,
    trunk_orientation_gain=10.0,

    damping=1e-8,
    rcond=1e-10,
):
    """
    Strict hierarchy during DOUBLE SUPPORT:

        P1: both feet 10D
        P2: CoM 3D
        P3: trunk local-Y 1D

    Expected final nullity:

        16 - 10 - 3 - 1 = 2
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    left_position_ref = (
        _vector3(
            left_position_ref,
            "left_position_ref",
        )
    )

    right_position_ref = (
        _vector3(
            right_position_ref,
            "right_position_ref",
        )
    )

    com_position_ref = (
        _vector3(
            com_position_ref,
            "com_position_ref",
        )
    )

    com_velocity_ref = (
        _vector3(
            com_velocity_ref,
            "com_velocity_ref",
        )
    )

    trunk_rotation_ref = (
        _rotation_matrix(
            trunk_rotation_ref,
            "trunk_rotation_ref",
        )
    )

    if foot_position_gain < 0.0:

        raise ValueError(
            "foot_position_gain "
            "cannot be negative."
        )

    if com_position_gain < 0.0:

        raise ValueError(
            "com_position_gain "
            "cannot be negative."
        )

    # ========================================================
    # CURRENT KINEMATICS
    # ========================================================

    robot.update(
        q_pin
    )

    p_left_actual, _ = (
        robot.get_left_foot_pose()
    )

    p_right_actual, _ = (
        robot.get_right_foot_pose()
    )

    p_com_actual = (
        robot.get_com()
    )

    # ========================================================
    # TRACKING ERRORS
    # ========================================================

    left_error = (
        left_position_ref
        -
        p_left_actual
    )

    right_error = (
        right_position_ref
        -
        p_right_actual
    )

    com_error = (
        com_position_ref
        -
        p_com_actual
    )

    # ========================================================
    # JACOBIANS
    # ========================================================

    J_left = (
        robot.get_foot_task_jacobian(
            LEFT_FOOT_FRAME,
            active_only=True,
        )
    )

    J_right = (
        robot.get_foot_task_jacobian(
            RIGHT_FOOT_FRAME,
            active_only=True,
        )
    )

    J_com = (
        robot.get_com_jacobian(
            active_only=True,
        )
    )

    # ========================================================
    # TASK COMMANDS
    # ========================================================

    left_linear_cmd = (
        foot_position_gain
        *
        left_error
    )

    right_linear_cmd = (
        foot_position_gain
        *
        right_error
    )

    com_velocity_cmd = (
        com_velocity_ref
        +
        com_position_gain
        *
        com_error
    )

    v_left = np.concatenate(
        [
            left_linear_cmd,

            np.zeros(
                2,
                dtype=float,
            ),
        ]
    )

    v_right = np.concatenate(
        [
            right_linear_cmd,

            np.zeros(
                2,
                dtype=float,
            ),
        ]
    )

    # ========================================================
    # BOTH FEET AS ONE HIGHEST-PRIORITY TASK
    # ========================================================

    J_both_feet = np.vstack(
        [
            J_left,
            J_right,
        ]
    )

    v_both_feet = np.concatenate(
        [
            v_left,
            v_right,
        ]
    )

    # ========================================================
    # TRUNK
    # ========================================================

    (
        J_trunk_pitch,
        v_trunk_pitch,
        trunk_error_local,
    ) = build_trunk_pitch_task(
        robot=(
            robot
        ),

        trunk_rotation_ref=(
            trunk_rotation_ref
        ),

        trunk_orientation_gain=(
            trunk_orientation_gain
        ),
    )

    # ========================================================
    # HIERARCHY
    # ========================================================

    tasks = [
        (
            "double_support_feet",
            J_both_feet,
            v_both_feet,
        ),

        (
            "com",
            J_com,
            com_velocity_cmd,
        ),

        (
            "trunk_pitch",
            J_trunk_pitch,
            v_trunk_pitch,
        ),
    ]

    # ========================================================
    # SOLVE
    # ========================================================

    n_active = len(
        robot.walking_velocity_indices
    )

    (
        qdot_active,
        diagnostics,
        final_nullspace,
    ) = solve_task_hierarchy(
        tasks=(
            tasks
        ),

        n_dof=(
            n_active
        ),

        damping=(
            damping
        ),

        rcond=(
            rcond
        ),
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics.append(
        {
            "name":
                "tracking_errors",

            "left_position_error_norm":
                float(
                    np.linalg.norm(
                        left_error
                    )
                ),

            "right_position_error_norm":
                float(
                    np.linalg.norm(
                        right_error
                    )
                ),

            "com_position_error_norm":
                float(
                    np.linalg.norm(
                        com_error
                    )
                ),

            "trunk_local_y_error_rad":
                float(
                    trunk_error_local[1]
                ),
        }
    )

    # ========================================================
    # ACTIVE -> FULL
    # ========================================================

    qdot_full = np.zeros(
        robot.model.nv,
        dtype=float,
    )

    qdot_full[
        robot.walking_velocity_indices
    ] = (
        qdot_active
    )

    return (
        qdot_full,
        diagnostics,
        final_nullspace,
    )