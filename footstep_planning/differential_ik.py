import numpy as np

try:
    from .pinocchio_model import (
        LEFT_FOOT_FRAME,
        RIGHT_FOOT_FRAME,
    )
except ImportError:
    from pinocchio_model import (
        LEFT_FOOT_FRAME,
        RIGHT_FOOT_FRAME,
    )


# ============================================================
# NUMERICAL UTILITIES
# ============================================================

def _matrix_rank(
    A,
    rcond=1e-10,
):
    """
    Numerical matrix rank using the same tolerance convention
    throughout the hierarchical solver.
    """

    A = np.asarray(
        A,
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
        rcond
        * max(A.shape)
        * singular_values[0]
    )

    return int(
        np.sum(
            singular_values > tolerance
        )
    )


def damped_pseudoinverse(
    A,
    damping=1e-8,
    rcond=1e-10,
):
    """
    Damped SVD pseudoinverse.

    If:

        A = U Sigma V^T

    then:

        A# = V diag(
                  sigma_i /
                  (sigma_i^2 + lambda^2)
               ) U^T

    Singular values below the numerical-rank threshold
    are ignored.
    """

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

    U, singular_values, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    if singular_values.size == 0:
        return np.zeros(
            (
                A.shape[1],
                A.shape[0],
            ),
            dtype=float,
        )

    tolerance = (
        rcond
        * max(A.shape)
        * singular_values[0]
    )

    gains = np.zeros_like(
        singular_values
    )

    valid = (
        singular_values
        > tolerance
    )

    sigma = singular_values[
        valid
    ]

    if damping == 0.0:

        gains[valid] = (
            1.0 / sigma
        )

    else:

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
        @ np.diag(gains)
        @ U.T
    )


def nullspace_basis(
    A,
    rcond=1e-10,
):
    """
    Return an orthonormal basis Z of null(A).

    A @ Z ~= 0

    If:

        A in R^(m x n)

    then:

        Z in R^(n x (n-rank(A)))
    """

    A = np.asarray(
        A,
        dtype=float,
    )

    if A.ndim != 2:
        raise ValueError(
            "A must be a 2D matrix."
        )

    n = A.shape[1]

    if A.shape[0] == 0:
        return np.eye(
            n,
            dtype=float,
        )

    _, singular_values, Vt = np.linalg.svd(
        A,
        full_matrices=True,
    )

    if singular_values.size == 0:

        rank = 0

    else:

        tolerance = (
            rcond
            * max(A.shape)
            * singular_values[0]
        )

        rank = int(
            np.sum(
                singular_values
                > tolerance
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
# GENERIC STRICT-HIERARCHY SOLVER
# ============================================================

def solve_task_hierarchy(
    tasks,
    n_dof,
    damping=1e-8,
    rcond=1e-10,
):
    """
    Solve a strict velocity-level task hierarchy.

    Parameters
    ----------
    tasks : list of tuples

        [
            (task_name, J1, v1),
            (task_name, J2, v2),
            ...
        ]

    n_dof : int
        Number of optimization variables.

    Returns
    -------
    qdot : ndarray, shape (n_dof,)

    diagnostics : list[dict]

    final_nullspace_basis : ndarray

    ----------------------------------------------------------------

    At each priority level:

        qdot_new
        =
        qdot
        +
        Z y

    where Z spans the null space of all higher-priority tasks.

    The reduced problem is:

        J Z y
        =
        v - J qdot

    therefore:

        y
        =
        (J Z)# (v - J qdot)

    After solving the current task:

        Z_new
        =
        Z null(J Z)

    so all subsequent motions remain inside the null space
    of every higher-priority task.
    """

    qdot = np.zeros(
        n_dof,
        dtype=float,
    )

    # Initially every direction is available.
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
        ).reshape(-1)

        if J.ndim != 2:
            raise ValueError(
                f"{task_name}: J must be 2D."
            )

        if J.shape[1] != n_dof:
            raise ValueError(
                f"{task_name}: "
                f"J has {J.shape[1]} columns, "
                f"expected {n_dof}."
            )

        if J.shape[0] != v.shape[0]:
            raise ValueError(
                f"{task_name}: "
                f"J rows and velocity size differ."
            )

        # ----------------------------------------------------
        # Current task residual before solving it
        # ----------------------------------------------------

        residual_before = (
            v
            -
            J @ qdot
        )

        # ----------------------------------------------------
        # Project current task into remaining null space
        # ----------------------------------------------------

        J_reduced = (
            J @ Z
        )

        reduced_rank = _matrix_rank(
            J_reduced,
            rcond=rcond,
        )

        # ----------------------------------------------------
        # Solve only in the remaining null space
        # ----------------------------------------------------

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
                @ residual_before
            )

            qdot = (
                qdot
                +
                Z @ y
            )

        # ----------------------------------------------------
        # Residual after current priority
        # ----------------------------------------------------

        residual_after = (
            v
            -
            J @ qdot
        )

        # ----------------------------------------------------
        # Restrict all lower priorities
        # ----------------------------------------------------

        Z_local = nullspace_basis(
            J_reduced,
            rcond=rcond,
        )

        Z = (
            Z
            @ Z_local
        )

        diagnostics.append(
            {
                "name": task_name,
                "rows": J.shape[0],
                "reduced_rank": reduced_rank,
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
# SINGLE SUPPORT IK
# ============================================================

def solve_single_support_ik(
    robot,
    q_pin,
    support_side,
    swing_linear_velocity_ref,
    com_velocity_ref,
    damping=1e-8,
    rcond=1e-10,
):
    """
    Solve:

        Priority 1:
            support foot 5D

        Priority 2:
            swing foot 5D

        Priority 3:
            CoM 3D

    Foot task:

        [vx, vy, vz, omega_local_y, omega_local_z]

    Current baseline:

        support velocity = 0

        swing angular velocity = 0

    No position/orientation feedback is used.
    """

    swing_linear_velocity_ref = np.asarray(
        swing_linear_velocity_ref,
        dtype=float,
    )

    com_velocity_ref = np.asarray(
        com_velocity_ref,
        dtype=float,
    )

    if swing_linear_velocity_ref.shape != (3,):
        raise ValueError(
            "swing_linear_velocity_ref must have shape (3,)."
        )

    if com_velocity_ref.shape != (3,):
        raise ValueError(
            "com_velocity_ref must have shape (3,)."
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
            "support_side must be 'left' or 'right'."
        )

    # --------------------------------------------------------
    # Update Pinocchio kinematics
    # --------------------------------------------------------

    robot.update(
        q_pin
    )

    # --------------------------------------------------------
    # Jacobians in the 16-DoF walking space
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Assign support / swing
    # --------------------------------------------------------

    if support_side == "left":

        J_support = J_left
        J_swing = J_right

        support_name = (
            "left_support"
        )

        swing_name = (
            "right_swing"
        )

    else:

        J_support = J_right
        J_swing = J_left

        support_name = (
            "right_support"
        )

        swing_name = (
            "left_swing"
        )

    # --------------------------------------------------------
    # Task velocities
    # --------------------------------------------------------

    v_support = np.zeros(
        5,
        dtype=float,
    )

    v_swing = np.concatenate(
        [
            swing_linear_velocity_ref,
            np.zeros(
                2,
                dtype=float,
            ),
        ]
    )

    # --------------------------------------------------------
    # Hierarchy
    # --------------------------------------------------------

    tasks = [
        (
            support_name,
            J_support,
            v_support,
        ),
        (
            swing_name,
            J_swing,
            v_swing,
        ),
        (
            "com",
            J_com,
            com_velocity_ref,
        ),
    ]

    n_active = len(
        robot.walking_velocity_indices
    )

    (
        qdot_active,
        diagnostics,
        final_nullspace,
    ) = solve_task_hierarchy(
        tasks=tasks,
        n_dof=n_active,
        damping=damping,
        rcond=rcond,
    )

    # --------------------------------------------------------
    # Map 16-DoF solution back to Pinocchio nv=20
    # --------------------------------------------------------

    qdot_full = np.zeros(
        robot.model.nv,
        dtype=float,
    )

    qdot_full[
        robot.walking_velocity_indices
    ] = qdot_active

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
    com_velocity_ref,
    damping=1e-8,
    rcond=1e-10,
):
    """
    Solve:

        Priority 1:
            both feet fixed, 10D

        Priority 2:
            CoM, 3D

    No feedback is used.
    """

    com_velocity_ref = np.asarray(
        com_velocity_ref,
        dtype=float,
    )

    if com_velocity_ref.shape != (3,):
        raise ValueError(
            "com_velocity_ref must have shape (3,)."
        )

    robot.update(
        q_pin
    )

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

    J_both_feet = np.vstack(
        [
            J_left,
            J_right,
        ]
    )

    v_both_feet = np.zeros(
        10,
        dtype=float,
    )

    tasks = [
        (
            "double_support_feet",
            J_both_feet,
            v_both_feet,
        ),
        (
            "com",
            J_com,
            com_velocity_ref,
        ),
    ]

    n_active = len(
        robot.walking_velocity_indices
    )

    (
        qdot_active,
        diagnostics,
        final_nullspace,
    ) = solve_task_hierarchy(
        tasks=tasks,
        n_dof=n_active,
        damping=damping,
        rcond=rcond,
    )

    qdot_full = np.zeros(
        robot.model.nv,
        dtype=float,
    )

    qdot_full[
        robot.walking_velocity_indices
    ] = qdot_active

    return (
        qdot_full,
        diagnostics,
        final_nullspace,
    )