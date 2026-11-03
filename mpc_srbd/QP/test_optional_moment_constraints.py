import numpy as np


from QP_parameters import (
    Mx_tolerance,
    mu,
    l_t,
    l_h,
    enable_toe_heel_lateral_friction,
)

from moment_constraint import (
    build_moment_constraints,
)

from QP_problem import (
    build_QP_problem,
)


# ============================================================
# TEST SETTINGS
# ============================================================

TEST_N = 2

EXPECTED_MX_TOLERANCE = 0.01

ATOL = 1e-12


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_close(
    actual,
    expected,
    name,
):

    if not np.isclose(
        actual,
        expected,
        atol=ATOL,
        rtol=0.0,
    ):

        raise AssertionError(
            f"{name}: "
            f"expected {expected}, "
            f"got {actual}"
        )


def check_array_close(
    actual,
    expected,
    name,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    expected = np.asarray(
        expected,
        dtype=float,
    )


    if not np.allclose(
        actual,
        expected,
        atol=ATOL,
        rtol=0.0,
    ):

        raise AssertionError(
            f"{name}:\n"
            f"expected = {expected}\n"
            f"actual   = {actual}"
        )


# ============================================================
# TEST 1
# PARAMETER CHECK
# ============================================================

def test_parameters():

    print("\n" + "=" * 78)

    print(
        "TEST 1: QP parameters"
    )

    print("=" * 78)


    check_close(
        Mx_tolerance,
        EXPECTED_MX_TOLERANCE,
        "Mx_tolerance",
    )


    print(
        "Mx_tolerance =",
        Mx_tolerance,
        "N.m"
    )


    print(
        "enable_toe_heel_lateral_friction =",
        enable_toe_heel_lateral_friction
    )


    if enable_toe_heel_lateral_friction:

        print(
            "Current mode: EXTENDED"
        )

    else:

        print(
            "Current mode: BASELINE"
        )


    print(
        "TEST 1: PASS"
    )


# ============================================================
# TEST 2
# BASELINE MOMENT CONSTRAINT
#
# Expected:
#
#     3 constraints / foot
#
#     6 constraints / step
#
# ============================================================

def test_baseline_moment_constraints():

    print("\n" + "=" * 78)

    print(
        "TEST 2: Baseline moment constraints"
    )

    print("=" * 78)


    N = TEST_N

    I3 = np.eye(
        3,
        dtype=float,
    )


    (
        A_moment,
        l_moment,
        u_moment,
    ) = build_moment_constraints(

        N=N,

        L6_R_W=I3,

        R6_R_W=I3,

        l_t=l_t,

        l_h=l_h,

        mu=mu,

        Mx_tolerance=Mx_tolerance,

        enable_toe_heel_lateral_friction=False,
    )


    expected_rows = (
        6
        *
        N
    )


    expected_shape = (
        expected_rows,
        12 * N,
    )


    if A_moment.shape != expected_shape:

        raise AssertionError(
            f"Baseline A_moment shape sai. "
            f"Expected {expected_shape}, "
            f"received {A_moment.shape}."
        )


    if l_moment.shape != (
        expected_rows,
    ):

        raise AssertionError(
            "Baseline l_moment shape sai."
        )


    if u_moment.shape != (
        expected_rows,
    ):

        raise AssertionError(
            "Baseline u_moment shape sai."
        )


    # --------------------------------------------------------
    # Per prediction step:
    #
    # row + 0 = Left Mx
    #
    # row + 1 = Left toe My
    #
    # row + 2 = Left heel My
    #
    # row + 3 = Right Mx
    #
    # row + 4 = Right toe My
    #
    # row + 5 = Right heel My
    # --------------------------------------------------------

    for k in range(N):

        row_base = (
            6
            *
            k
        )

        col_base = (
            12
            *
            k
        )


        left_mx_row = (
            row_base
            +
            0
        )


        right_mx_row = (
            row_base
            +
            3
        )


        # ----------------------------------------------------
        # Mx bounds
        # ----------------------------------------------------

        check_close(
            l_moment[left_mx_row],
            -Mx_tolerance,
            f"k={k}, baseline left Mx lower",
        )


        check_close(
            u_moment[left_mx_row],
            Mx_tolerance,
            f"k={k}, baseline left Mx upper",
        )


        check_close(
            l_moment[right_mx_row],
            -Mx_tolerance,
            f"k={k}, baseline right Mx lower",
        )


        check_close(
            u_moment[right_mx_row],
            Mx_tolerance,
            f"k={k}, baseline right Mx upper",
        )


        # ----------------------------------------------------
        # Identity rotation:
        #
        # Left Mx = U[col_base + 3]
        #
        # Right Mx = U[col_base + 9]
        # ----------------------------------------------------

        expected_left = np.zeros(
            12 * N,
            dtype=float,
        )

        expected_left[
            col_base + 3
        ] = 1.0


        expected_right = np.zeros(
            12 * N,
            dtype=float,
        )

        expected_right[
            col_base + 9
        ] = 1.0


        check_array_close(
            A_moment[left_mx_row],
            expected_left,
            f"k={k}, baseline left Mx selector",
        )


        check_array_close(
            A_moment[right_mx_row],
            expected_right,
            f"k={k}, baseline right Mx selector",
        )


    print(
        f"Baseline rows = {expected_rows}"
    )


    print(
        "Expected = 6*N =",
        6 * N
    )


    print(
        "TEST 2: PASS"
    )


# ============================================================
# TEST 3
# EXTENDED MOMENT CONSTRAINT
#
# Expected:
#
#     7 constraints / foot
#
#     14 constraints / step
# ============================================================

def test_extended_moment_constraints():

    print("\n" + "=" * 78)

    print(
        "TEST 3: Extended toe/heel lateral-friction constraints"
    )

    print("=" * 78)


    N = TEST_N

    I3 = np.eye(
        3,
        dtype=float,
    )


    (
        A_moment,
        l_moment,
        u_moment,
    ) = build_moment_constraints(

        N=N,

        L6_R_W=I3,

        R6_R_W=I3,

        l_t=l_t,

        l_h=l_h,

        mu=mu,

        Mx_tolerance=Mx_tolerance,

        enable_toe_heel_lateral_friction=True,
    )


    expected_rows = (
        14
        *
        N
    )


    expected_shape = (
        expected_rows,
        12 * N,
    )


    if A_moment.shape != expected_shape:

        raise AssertionError(
            f"Extended A_moment shape sai. "
            f"Expected {expected_shape}, "
            f"received {A_moment.shape}."
        )


    if l_moment.shape != (
        expected_rows,
    ):

        raise AssertionError(
            "Extended l_moment shape sai."
        )


    if u_moment.shape != (
        expected_rows,
    ):

        raise AssertionError(
            "Extended u_moment shape sai."
        )


    # --------------------------------------------------------
    # Extended layout:
    #
    # rows 0...6:
    #     left foot
    #
    # rows 7...13:
    #     right foot
    # --------------------------------------------------------

    for k in range(N):

        row_base = (
            14
            *
            k
        )


        left_mx_row = (
            row_base
            +
            0
        )


        right_mx_row = (
            row_base
            +
            7
        )


        check_close(
            l_moment[left_mx_row],
            -Mx_tolerance,
            f"k={k}, extended left Mx lower",
        )


        check_close(
            u_moment[left_mx_row],
            Mx_tolerance,
            f"k={k}, extended left Mx upper",
        )


        check_close(
            l_moment[right_mx_row],
            -Mx_tolerance,
            f"k={k}, extended right Mx lower",
        )


        check_close(
            u_moment[right_mx_row],
            Mx_tolerance,
            f"k={k}, extended right Mx upper",
        )


    print(
        f"Extended rows = {expected_rows}"
    )


    print(
        "Expected = 14*N =",
        14 * N
    )


    print(
        "TEST 3: PASS"
    )


# ============================================================
# TEST 4
# NUMERICAL Mx BOUND
# ============================================================

def test_mx_bounds():

    print("\n" + "=" * 78)

    print(
        "TEST 4: Numerical Mx tolerance"
    )

    print("=" * 78)


    N = 1


    I3 = np.eye(
        3,
        dtype=float,
    )


    (
        A_moment,
        l_moment,
        u_moment,
    ) = build_moment_constraints(

        N=N,

        L6_R_W=I3,

        R6_R_W=I3,

        l_t=l_t,

        l_h=l_h,

        mu=mu,

        Mx_tolerance=Mx_tolerance,

        enable_toe_heel_lateral_friction=False,
    )


    # Baseline layout:
    #
    # row 0 = left Mx
    #
    # row 3 = right Mx

    left_row = 0

    right_row = 3


    # --------------------------------------------------------
    # INSIDE
    # --------------------------------------------------------

    U_inside = np.zeros(
        12,
        dtype=float,
    )


    U_inside[3] = 0.009

    U_inside[9] = -0.009


    value_inside = (
        A_moment
        @
        U_inside
    )


    if not (
        l_moment[left_row]
        <=
        value_inside[left_row]
        <=
        u_moment[left_row]
    ):

        raise AssertionError(
            "Left Mx = 0.009 phải được chấp nhận."
        )


    if not (
        l_moment[right_row]
        <=
        value_inside[right_row]
        <=
        u_moment[right_row]
    ):

        raise AssertionError(
            "Right Mx = -0.009 phải được chấp nhận."
        )


    # --------------------------------------------------------
    # OUTSIDE
    # --------------------------------------------------------

    U_outside = np.zeros(
        12,
        dtype=float,
    )


    U_outside[3] = 0.011

    U_outside[9] = -0.011


    value_outside = (
        A_moment
        @
        U_outside
    )


    if not (
        value_outside[left_row]
        >
        u_moment[left_row]
    ):

        raise AssertionError(
            "Left Mx = 0.011 phải vi phạm upper bound."
        )


    if not (
        value_outside[right_row]
        <
        l_moment[right_row]
    ):

        raise AssertionError(
            "Right Mx = -0.011 phải vi phạm lower bound."
        )


    print(
        "Mx = +/-0.009 N.m : accepted"
    )


    print(
        "Mx = +/-0.011 N.m : rejected"
    )


    print(
        "TEST 4: PASS"
    )


# ============================================================
# TEST 5
# FULL QP INTEGRATION
#
# QP_problem.py uses the CURRENT flag from QP_parameters.py.
# ============================================================

def test_full_qp_integration():

    print("\n" + "=" * 78)

    print(
        "TEST 5: Full QP integration"
    )

    print("=" * 78)


    N = TEST_N

    nx = 13

    nu = 12


    # --------------------------------------------------------
    # Minimal prediction matrices
    # --------------------------------------------------------

    Phi = np.zeros(
        (
            nx * N,
            nx,
        ),
        dtype=float,
    )


    Gamma = np.zeros(
        (
            nx * N,
            nu * N,
        ),
        dtype=float,
    )


    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    x_current = np.zeros(
        nx,
        dtype=float,
    )


    x_current[
        12
    ] = 1.0


    # --------------------------------------------------------
    # Reference trajectory
    # --------------------------------------------------------

    X_ref = np.zeros(
        nx * N,
        dtype=float,
    )


    for k in range(N):

        X_ref[
            k * nx
            +
            12
        ] = 1.0


    # --------------------------------------------------------
    # Identity foot frames
    # --------------------------------------------------------

    I3 = np.eye(
        3,
        dtype=float,
    )


    # --------------------------------------------------------
    # Double support
    # --------------------------------------------------------

    contact_schedule = np.ones(
        (
            N,
            2,
        ),
        dtype=int,
    )


    (
        H,
        f,
        A,
        l,
        u,
    ) = build_QP_problem(

        Phi=Phi,

        Gamma=Gamma,

        x_current=x_current,

        X_ref=X_ref,

        N=N,

        L6_R_W=I3,

        R6_R_W=I3,

        contact_schedule=contact_schedule,
    )


    # ========================================================
    # EXPECTED ROW COUNTS
    # ========================================================

    force_rows = (
        10
        *
        N
    )


    if enable_toe_heel_lateral_friction:

        moment_rows = (
            14
            *
            N
        )

        moment_rows_per_step = 14

        right_mx_offset = 7

    else:

        moment_rows = (
            6
            *
            N
        )

        moment_rows_per_step = 6

        right_mx_offset = 3


    contact_rows = (
        12
        *
        N
    )


    expected_total_rows = (
        force_rows
        +
        moment_rows
        +
        contact_rows
    )


    # ========================================================
    # DIMENSION CHECKS
    # ========================================================

    if H.shape != (
        nu * N,
        nu * N,
    ):

        raise AssertionError(
            f"H shape sai: {H.shape}"
        )


    if f.shape != (
        nu * N,
    ):

        raise AssertionError(
            f"f shape sai: {f.shape}"
        )


    if A.shape != (
        expected_total_rows,
        nu * N,
    ):

        raise AssertionError(
            f"Full A shape sai. "
            f"Expected "
            f"({expected_total_rows}, {nu * N}), "
            f"received {A.shape}."
        )


    if l.shape != (
        expected_total_rows,
    ):

        raise AssertionError(
            "Full l shape sai."
        )


    if u.shape != (
        expected_total_rows,
    ):

        raise AssertionError(
            "Full u shape sai."
        )


    # ========================================================
    # CHECK Mx ROWS INSIDE FULL QP
    # ========================================================

    moment_start = (
        force_rows
    )


    for k in range(N):

        moment_step_start = (
            moment_start
            +
            moment_rows_per_step
            *
            k
        )


        left_mx_row = (
            moment_step_start
            +
            0
        )


        right_mx_row = (
            moment_step_start
            +
            right_mx_offset
        )


        check_close(
            l[left_mx_row],
            -Mx_tolerance,
            f"Full QP k={k}, left Mx lower",
        )


        check_close(
            u[left_mx_row],
            Mx_tolerance,
            f"Full QP k={k}, left Mx upper",
        )


        check_close(
            l[right_mx_row],
            -Mx_tolerance,
            f"Full QP k={k}, right Mx lower",
        )


        check_close(
            u[right_mx_row],
            Mx_tolerance,
            f"Full QP k={k}, right Mx upper",
        )


    print(
        "Current full-QP mode:"
    )


    if enable_toe_heel_lateral_friction:

        print(
            "    EXTENDED"
        )

    else:

        print(
            "    BASELINE"
        )


    print(
        "Force rows   =",
        force_rows
    )


    print(
        "Moment rows  =",
        moment_rows
    )


    print(
        "Contact rows =",
        contact_rows
    )


    print(
        "Total rows   =",
        expected_total_rows
    )


    print(
        "TEST 5: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 78)

    print(
        "TEST OPTIONAL MOMENT CONSTRAINTS"
    )

    print("=" * 78)


    test_parameters()


    test_baseline_moment_constraints()


    test_extended_moment_constraints()


    test_mx_bounds()


    test_full_qp_integration()


    print("\n")

    print("=" * 78)

    print(
        "ALL TESTS: PASS"
    )

    print("=" * 78)


    print(
        "\nCurrent configured mode:"
    )


    if enable_toe_heel_lateral_friction:

        print(
            "    EXTENDED"
        )

        print(
            "    14 moment constraints / MPC step"
        )

    else:

        print(
            "    BASELINE"
        )

        print(
            "    6 moment constraints / MPC step"
        )


    print(
        "\nMx constraint:"
    )


    print(
        f"    {-Mx_tolerance:.6f} "
        f"<= Mx_L <= "
        f"{Mx_tolerance:.6f} N.m"
    )


    print(
        f"    {-Mx_tolerance:.6f} "
        f"<= Mx_R <= "
        f"{Mx_tolerance:.6f} N.m"
    )


if __name__ == "__main__":

    main()