import numpy as np


def build_force_constraints(
    N,
    mu,
    Fz_max,
    Fz_min=0.0
):
    """
    Xây dựng force constraints cho SRBD-MPC.

    Decision variable:

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

    =========================================================
    Friction pyramid - Eq. (2c)
    =========================================================

        |Fx| <= mu * Fz
        |Fy| <= mu * Fz

    tương đương:

        Fx  - mu*Fz <= 0
       -Fx  - mu*Fz <= 0
        Fy  - mu*Fz <= 0
       -Fy  - mu*Fz <= 0

    =========================================================
    Normal force - Eq. (2d)
    =========================================================

        Fz_min <= Fz <= Fz_max

    =========================================================
    OSQP form
    =========================================================

        l_force <= A_force @ U <= u_force

    Parameters
    ----------
    N : int
        Prediction horizon.

    mu : float
        Hệ số ma sát.

    Fz_max : float
        Giới hạn trên của normal force.

    Fz_min : float, optional
        Giới hạn dưới của normal force.
        Mặc định = 0.

    Returns
    -------
    A_force : np.ndarray
        Force constraint matrix.
        Shape = (10*N, 12*N)

    l_force : np.ndarray
        Lower bound vector.
        Shape = (10*N,)

    u_force : np.ndarray
        Upper bound vector.
        Shape = (10*N,)
    """

    # ========================================================
    # Validation
    # ========================================================

    if N <= 0:
        raise ValueError(
            "N phải lớn hơn 0."
        )

    if mu < 0.0:
        raise ValueError(
            "mu phải >= 0."
        )

    if Fz_min < 0.0:
        raise ValueError(
            "Fz_min phải >= 0."
        )

    if Fz_max <= Fz_min:
        raise ValueError(
            "Fz_max phải lớn hơn Fz_min."
        )


    # ========================================================
    # Dimensions
    # ========================================================

    nu = 12

    # Mỗi chân:
    #
    # 4 friction constraints
    # 1 Fz bound
    #
    # => 5 constraints / foot
    #
    # 2 feet => 10 constraints / step

    constraints_per_foot = 5
    constraints_per_step = 10

    total_constraints = (
        N * constraints_per_step
    )


    # ========================================================
    # Allocate matrices
    # ========================================================

    A_force = np.zeros(
        (
            total_constraints,
            N * nu
        ),
        dtype=float
    )

    l_force = np.full(
        total_constraints,
        -np.inf,
        dtype=float
    )

    u_force = np.full(
        total_constraints,
        np.inf,
        dtype=float
    )


    # ========================================================
    # Build constraints
    # ========================================================

    row = 0

    for k in range(N):

        # Column offset of u_k in U
        step_col = k * nu


        # ====================================================
        # Two feet
        #
        # Left foot:
        #   offset = 0
        #
        # Right foot:
        #   offset = 6
        # ====================================================

        for foot_offset in (0, 6):

            Fx_col = (
                step_col
                + foot_offset
                + 0
            )

            Fy_col = (
                step_col
                + foot_offset
                + 1
            )

            Fz_col = (
                step_col
                + foot_offset
                + 2
            )


            # ================================================
            # 1. Fx - mu*Fz <= 0
            # ================================================

            A_force[
                row,
                Fx_col
            ] = 1.0

            A_force[
                row,
                Fz_col
            ] = -mu

            l_force[row] = -np.inf
            u_force[row] = 0.0

            row += 1


            # ================================================
            # 2. -Fx - mu*Fz <= 0
            # ================================================

            A_force[
                row,
                Fx_col
            ] = -1.0

            A_force[
                row,
                Fz_col
            ] = -mu

            l_force[row] = -np.inf
            u_force[row] = 0.0

            row += 1


            # ================================================
            # 3. Fy - mu*Fz <= 0
            # ================================================

            A_force[
                row,
                Fy_col
            ] = 1.0

            A_force[
                row,
                Fz_col
            ] = -mu

            l_force[row] = -np.inf
            u_force[row] = 0.0

            row += 1


            # ================================================
            # 4. -Fy - mu*Fz <= 0
            # ================================================

            A_force[
                row,
                Fy_col
            ] = -1.0

            A_force[
                row,
                Fz_col
            ] = -mu

            l_force[row] = -np.inf
            u_force[row] = 0.0

            row += 1


            # ================================================
            # 5. Fz_min <= Fz <= Fz_max
            # ================================================

            A_force[
                row,
                Fz_col
            ] = 1.0

            l_force[row] = Fz_min
            u_force[row] = Fz_max

            row += 1


    return (
        A_force,
        l_force,
        u_force
    )