import numpy as np

from cost_function import build_cost_function
from force_constraint import build_force_constraints
from moment_constraint import build_moment_constraints
from contact_constraint import build_contact_constraints

from QP_parameters import (
    Q,
    R,
    mu,
    Fmax,
    l_t,
    l_h,
    Mx_tolerance,
)


def build_QP_problem(
    Phi,
    Gamma,
    x_current,
    X_ref,
    N,
    L6_R_W,
    R6_R_W,
    contact_schedule,
):
    """
    Build the complete SRBD-MPC quadratic programming problem.

    QP:

        minimize

            1/2 * U.T @ H @ U
            +
            f.T @ U

        subject to

            l <= A @ U <= u


    Decision variable:

        U =
        [
            u_0
            u_1
            ...
            u_{N-1}
        ]

    with:

        u_k =
        [
            F_Lx, F_Ly, F_Lz,
            M_Lx, M_Ly, M_Lz,

            F_Rx, F_Ry, F_Rz,
            M_Rx, M_Ry, M_Rz
        ]


    Parameters
    ----------
    Phi : np.ndarray
        Prediction state matrix.

        Shape:
            (13*N, 13)

    Gamma : np.ndarray
        Prediction input matrix.

        Shape:
            (13*N, 12*N)

    x_current : np.ndarray
        Current augmented SRBD state.

        Shape:
            (13,)

    X_ref : np.ndarray
        Reference trajectory over the prediction horizon.

        Shape:
            (13*N,)

    N : int
        Prediction horizon.

    L6_R_W : np.ndarray
        World -> left contact-frame rotation matrix.

        Shape:
            (3, 3)

    R6_R_W : np.ndarray
        World -> right contact-frame rotation matrix.

        Shape:
            (3, 3)

    contact_schedule : np.ndarray
        Planned contact schedule.

        Shape:
            (N, 2)

        [:, 0] = left foot
        [:, 1] = right foot

        1 = stance
        0 = swing


    Returns
    -------
    H : np.ndarray
        QP Hessian.

    f : np.ndarray
        QP linear cost vector.

    A : np.ndarray
        Full constraint matrix.

    l : np.ndarray
        Full lower-bound vector.

    u : np.ndarray
        Full upper-bound vector.
    """

    # ========================================================
    # 1. DIMENSIONS
    # ========================================================

    nx = 13
    nu = 12

    nU = (
        N
        *
        nu
    )


    # ========================================================
    # 2. INPUT CONVERSION
    # ========================================================

    Phi = np.asarray(
        Phi,
        dtype=float,
    )

    Gamma = np.asarray(
        Gamma,
        dtype=float,
    )

    x_current = np.asarray(
        x_current,
        dtype=float,
    ).reshape(-1)

    X_ref = np.asarray(
        X_ref,
        dtype=float,
    ).reshape(-1)

    L6_R_W = np.asarray(
        L6_R_W,
        dtype=float,
    )

    R6_R_W = np.asarray(
        R6_R_W,
        dtype=float,
    )

    contact_schedule = np.asarray(
        contact_schedule,
        dtype=int,
    )


    # ========================================================
    # 3. BASIC VALIDATION
    # ========================================================

    if not isinstance(
        N,
        (int, np.integer),
    ):
        raise TypeError(
            "N phải là số nguyên."
        )

    if N <= 0:
        raise ValueError(
            "N phải lớn hơn 0."
        )


    # --------------------------------------------------------
    # Phi
    # --------------------------------------------------------

    if Phi.shape != (
        N * nx,
        nx,
    ):
        raise ValueError(
            f"Phi phải có shape "
            f"({N * nx}, {nx}), "
            f"nhưng nhận được {Phi.shape}."
        )


    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    if Gamma.shape != (
        N * nx,
        N * nu,
    ):
        raise ValueError(
            f"Gamma phải có shape "
            f"({N * nx}, {N * nu}), "
            f"nhưng nhận được {Gamma.shape}."
        )


    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    if x_current.shape != (
        nx,
    ):
        raise ValueError(
            "x_current phải có shape (13,)."
        )


    # --------------------------------------------------------
    # Reference trajectory
    # --------------------------------------------------------

    if X_ref.shape != (
        N * nx,
    ):
        raise ValueError(
            f"X_ref phải có shape "
            f"({N * nx},), "
            f"nhưng nhận được {X_ref.shape}."
        )


    # --------------------------------------------------------
    # Foot rotation matrices
    # --------------------------------------------------------

    if L6_R_W.shape != (
        3,
        3,
    ):
        raise ValueError(
            "L6_R_W phải có shape (3, 3)."
        )

    if R6_R_W.shape != (
        3,
        3,
    ):
        raise ValueError(
            "R6_R_W phải có shape (3, 3)."
        )


    # --------------------------------------------------------
    # Contact schedule
    # --------------------------------------------------------

    if contact_schedule.shape != (
        N,
        2,
    ):
        raise ValueError(
            "contact_schedule phải có shape "
            f"({N}, 2), "
            f"nhưng nhận được "
            f"{contact_schedule.shape}."
        )

    if not np.all(
        np.isin(
            contact_schedule,
            [
                0,
                1,
            ],
        )
    ):
        raise ValueError(
            "contact_schedule chỉ được chứa 0 hoặc 1."
        )


    # --------------------------------------------------------
    # Finite validation
    # --------------------------------------------------------

    for name, value in (
        (
            "Phi",
            Phi,
        ),
        (
            "Gamma",
            Gamma,
        ),
        (
            "x_current",
            x_current,
        ),
        (
            "X_ref",
            X_ref,
        ),
        (
            "L6_R_W",
            L6_R_W,
        ),
        (
            "R6_R_W",
            R6_R_W,
        ),
    ):

        if not np.all(
            np.isfinite(
                value
            )
        ):
            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


    # ========================================================
    # 4. COST FUNCTION
    # ========================================================

    (
        H,
        f,
        _,
        _,
        _,
    ) = build_cost_function(
        Phi=Phi,
        Gamma=Gamma,
        x_current=x_current,
        X_ref=X_ref,
        Q=Q,
        R=R,
    )


    # ========================================================
    # 5. CONSTRAINT BLOCKS
    #
    # Each block:
    #
    #     (
    #         name,
    #         A_constraint,
    #         l_constraint,
    #         u_constraint
    #     )
    #
    # ========================================================

    constraint_blocks = []


    # ========================================================
    # 5.1 FORCE CONSTRAINT
    #
    # Friction pyramid:
    #
    #     |Fx| <= mu*Fz
    #
    #     |Fy| <= mu*Fz
    #
    # Normal force:
    #
    #     0 <= Fz <= Fmax
    # ========================================================

    (
        A_force,
        l_force,
        u_force,
    ) = build_force_constraints(
        N=N,
        mu=mu,
        Fz_max=Fmax,
        Fz_min=0.0,
    )

    constraint_blocks.append(
        (
            "force",
            A_force,
            l_force,
            u_force,
        )
    )


    # ========================================================
    # 5.2 MOMENT / LINE-CONTACT CONSTRAINT
    #
    # Relaxed roll moment:
    #
    #     -Mx_tolerance
    #     <=
    #     Mx_foot
    #     <=
    #     Mx_tolerance
    #
    # Toe / heel:
    #
    #     -l_t*Fz
    #     <=
    #     My
    #     <=
    #     l_h*Fz
    #
    # Additional toe / heel lateral-friction constraints
    # remain unchanged from the current implementation.
    # ========================================================

    (
        A_moment,
        l_moment,
        u_moment,
    ) = build_moment_constraints(
        N=N,
        L6_R_W=L6_R_W,
        R6_R_W=R6_R_W,
        l_t=l_t,
        l_h=l_h,
        mu=mu,
        Mx_tolerance=Mx_tolerance,
    )

    constraint_blocks.append(
        (
            "moment",
            A_moment,
            l_moment,
            u_moment,
        )
    )


    # ========================================================
    # 5.3 CONTACT ACTIVATION CONSTRAINT
    #
    #     (1 - c_{k,i}) * u_{k,i} = 0
    #
    # c = 1:
    #
    #     stance
    #     -> wrench is allowed
    #
    # c = 0:
    #
    #     swing
    #     -> complete foot wrench is forced to zero
    #
    # ========================================================

    (
        A_contact,
        l_contact,
        u_contact,
    ) = build_contact_constraints(
        N=N,
        contact_schedule=contact_schedule,
    )

    constraint_blocks.append(
        (
            "contact",
            A_contact,
            l_contact,
            u_contact,
        )
    )


    # ========================================================
    # 6. VALIDATE CONSTRAINT BLOCKS
    # ========================================================

    for (
        name,
        A_block,
        l_block,
        u_block,
    ) in constraint_blocks:

        A_block = np.asarray(
            A_block,
            dtype=float,
        )

        l_block = np.asarray(
            l_block,
            dtype=float,
        ).reshape(-1)

        u_block = np.asarray(
            u_block,
            dtype=float,
        ).reshape(-1)


        # ----------------------------------------------------
        # Matrix dimensions
        # ----------------------------------------------------

        if A_block.ndim != 2:
            raise ValueError(
                f"Constraint '{name}': "
                "A phải là ma trận 2 chiều."
            )

        if A_block.shape[1] != nU:
            raise ValueError(
                f"Constraint '{name}': "
                f"A phải có {nU} cột, "
                f"nhưng nhận được "
                f"{A_block.shape[1]}."
            )


        # ----------------------------------------------------
        # Bound dimensions
        # ----------------------------------------------------

        if l_block.shape != (
            A_block.shape[0],
        ):
            raise ValueError(
                f"Constraint '{name}': "
                "dimension của l không khớp A."
            )

        if u_block.shape != (
            A_block.shape[0],
        ):
            raise ValueError(
                f"Constraint '{name}': "
                "dimension của u không khớp A."
            )


        # ----------------------------------------------------
        # Bounds consistency
        # ----------------------------------------------------

        if np.any(
            l_block
            >
            u_block
        ):
            raise ValueError(
                f"Constraint '{name}': "
                "tồn tại lower bound > upper bound."
            )


        # ----------------------------------------------------
        # Matrix finite validation
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(
                A_block
            )
        ):
            raise ValueError(
                f"Constraint '{name}': "
                "A chứa NaN hoặc Inf."
            )


    # ========================================================
    # 7. ASSEMBLE FULL CONSTRAINT MATRIX
    # ========================================================

    A = np.vstack(
        [
            block[1]
            for block in constraint_blocks
        ]
    )

    l = np.concatenate(
        [
            block[2]
            for block in constraint_blocks
        ]
    )

    u = np.concatenate(
        [
            block[3]
            for block in constraint_blocks
        ]
    )


    # ========================================================
    # 8. FINAL QP VALIDATION
    # ========================================================

    if H.shape != (
        nU,
        nU,
    ):
        raise ValueError(
            f"H phải có shape "
            f"({nU}, {nU}), "
            f"nhưng nhận được "
            f"{H.shape}."
        )

    if f.shape != (
        nU,
    ):
        raise ValueError(
            f"f phải có shape "
            f"({nU},), "
            f"nhưng nhận được "
            f"{f.shape}."
        )

    if A.ndim != 2:
        raise ValueError(
            "A phải là ma trận 2 chiều."
        )

    if A.shape[1] != nU:
        raise ValueError(
            f"A phải có {nU} cột."
        )

    if l.shape != (
        A.shape[0],
    ):
        raise ValueError(
            "Dimension của l không khớp A."
        )

    if u.shape != (
        A.shape[0],
    ):
        raise ValueError(
            "Dimension của u không khớp A."
        )

    if np.any(
        l
        >
        u
    ):
        raise ValueError(
            "Full QP constraints tồn tại "
            "lower bound > upper bound."
        )

    if not np.all(
        np.isfinite(
            H
        )
    ):
        raise ValueError(
            "H chứa NaN hoặc Inf."
        )

    if not np.all(
        np.isfinite(
            f
        )
    ):
        raise ValueError(
            "f chứa NaN hoặc Inf."
        )


    # ========================================================
    # 9. RETURN COMPLETE QP PROBLEM
    # ========================================================

    return (
        H,
        f,
        A,
        l,
        u,
    )