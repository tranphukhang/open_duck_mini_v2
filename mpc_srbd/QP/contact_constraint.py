import numpy as np


# ============================================================
# BUILD CONTACT CONSTRAINTS
# ============================================================

def build_contact_constraints(
    N,
    contact_schedule
):
    """
    Xây dựng contact activation constraints cho SRBD-MPC.

    Constraint tương ứng với Eq. (2b):

        (1 - c_{k,i}) * u_{k,i} = 0

    với:

        c_{k,i} = 1
            foot i đang stance / contact

        c_{k,i} = 0
            foot i đang swing / non-contact


    Decision variable
    -----------------

        U =
        [
            u_0
            u_1
            ...
            u_{N-1}
        ]

    với mỗi:

        u_k =
        [
            F_Lx, F_Ly, F_Lz,
            M_Lx, M_Ly, M_Lz,

            F_Rx, F_Ry, F_Rz,
            M_Rx, M_Ry, M_Rz
        ]


    Contact schedule
    ----------------

        contact_schedule[k] =
        [
            c_L,
            c_R
        ]

    Convention:

        [1, 1]
            double support

        [1, 0]
            left stance
            right swing

        [0, 1]
            left swing
            right stance


    Physical meaning
    ----------------

    Nếu:

        c_{k,i} = 1

    thì:

        (1 - 1) * u_{k,i} = 0

        0 * u_{k,i} = 0

    nên contact constraint không giới hạn wrench của chân stance.


    Nếu:

        c_{k,i} = 0

    thì:

        (1 - 0) * u_{k,i} = 0

        u_{k,i} = 0

    do đó:

        Fx = 0
        Fy = 0
        Fz = 0

        Mx = 0
        My = 0
        Mz = 0

    cho chân swing.


    OSQP form
    ---------

        l_contact
        <=
        A_contact @ U
        <=
        u_contact

    Vì đây là equality constraint:

        l_contact = 0

        u_contact = 0


    Parameters
    ----------
    N : int

        MPC prediction horizon.


    contact_schedule : np.ndarray

        Open-loop contact schedule over the prediction horizon.

        Shape:

            (N, 2)

        Columns:

            [:, 0] = left foot contact state

            [:, 1] = right foot contact state


    Returns
    -------
    A_contact : np.ndarray

        Contact constraint matrix.

        Shape:

            (12*N, 12*N)


    l_contact : np.ndarray

        Lower bound vector.

        Shape:

            (12*N,)


    u_contact : np.ndarray

        Upper bound vector.

        Shape:

            (12*N,)
    """


    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    if not isinstance(
        N,
        (int, np.integer)
    ):

        raise TypeError(
            "N phải là số nguyên."
        )


    if N <= 0:

        raise ValueError(
            "N phải lớn hơn 0."
        )


    contact_schedule = np.asarray(
        contact_schedule,
        dtype=int
    )


    if contact_schedule.shape != (
        N,
        2
    ):

        raise ValueError(
            "contact_schedule phải có shape "
            f"({N}, 2), nhưng nhận được "
            f"{contact_schedule.shape}."
        )


    if not np.all(
        np.isin(
            contact_schedule,
            [
                0,
                1
            ]
        )
    ):

        raise ValueError(
            "contact_schedule chỉ được chứa 0 hoặc 1."
        )


    # ========================================================
    # 2. DIMENSIONS
    # ========================================================

    nu = 12


    wrench_dimension_per_foot = 6


    feet_per_step = 2


    constraints_per_step = (
        wrench_dimension_per_foot
        *
        feet_per_step
    )


    total_constraints = (
        N
        *
        constraints_per_step
    )


    nU = (
        N
        *
        nu
    )


    # ========================================================
    # 3. ALLOCATE MATRICES
    # ========================================================

    A_contact = np.zeros(
        (
            total_constraints,
            nU
        ),
        dtype=float
    )


    l_contact = np.zeros(
        total_constraints,
        dtype=float
    )


    u_contact = np.zeros(
        total_constraints,
        dtype=float
    )


    # ========================================================
    # 4. BUILD CONTACT CONSTRAINTS
    # ========================================================

    row = 0


    for k in range(
        N
    ):

        # ----------------------------------------------------
        # Column offset of u_k inside U
        # ----------------------------------------------------

        step_col = (
            k
            *
            nu
        )


        # ----------------------------------------------------
        # Current contact states
        # ----------------------------------------------------

        c_L = contact_schedule[
            k,
            0
        ]


        c_R = contact_schedule[
            k,
            1
        ]


        # ----------------------------------------------------
        # Contact activation coefficients
        #
        # stance:
        #
        #     c = 1
        #     alpha = 0
        #
        # swing:
        #
        #     c = 0
        #     alpha = 1
        # ----------------------------------------------------

        alpha_L = (
            1
            -
            c_L
        )


        alpha_R = (
            1
            -
            c_R
        )


        # ====================================================
        # LEFT FOOT
        #
        # alpha_L * u_L = 0
        #
        # u_L =
        #
        # [
        #     F_Lx
        #     F_Ly
        #     F_Lz
        #     M_Lx
        #     M_Ly
        #     M_Lz
        # ]
        # ====================================================

        for j in range(
            wrench_dimension_per_foot
        ):

            col = (
                step_col
                +
                j
            )


            A_contact[
                row,
                col
            ] = alpha_L


            l_contact[
                row
            ] = 0.0


            u_contact[
                row
            ] = 0.0


            row += 1


        # ====================================================
        # RIGHT FOOT
        #
        # alpha_R * u_R = 0
        #
        # u_R =
        #
        # [
        #     F_Rx
        #     F_Ry
        #     F_Rz
        #     M_Rx
        #     M_Ry
        #     M_Rz
        # ]
        # ====================================================

        for j in range(
            wrench_dimension_per_foot
        ):

            col = (
                step_col
                +
                6
                +
                j
            )


            A_contact[
                row,
                col
            ] = alpha_R


            l_contact[
                row
            ] = 0.0


            u_contact[
                row
            ] = 0.0


            row += 1


    # ========================================================
    # 5. FINAL VALIDATION
    # ========================================================

    if row != total_constraints:

        raise RuntimeError(
            "Số contact constraints được xây dựng "
            "không khớp với kích thước dự kiến."
        )


    if A_contact.shape != (
        12 * N,
        12 * N
    ):

        raise RuntimeError(
            "A_contact phải có shape "
            f"({12 * N}, {12 * N}), "
            f"nhưng nhận được {A_contact.shape}."
        )


    if l_contact.shape != (
        12 * N,
    ):

        raise RuntimeError(
            "l_contact có dimension không đúng."
        )


    if u_contact.shape != (
        12 * N,
    ):

        raise RuntimeError(
            "u_contact có dimension không đúng."
        )


    if not np.all(
        np.isfinite(
            A_contact
        )
    ):

        raise RuntimeError(
            "A_contact chứa NaN hoặc Inf."
        )


    # ========================================================
    # 6. RETURN
    # ========================================================

    return (
        A_contact,
        l_contact,
        u_contact
    )