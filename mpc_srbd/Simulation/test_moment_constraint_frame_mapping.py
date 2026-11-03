from pathlib import Path
import sys

import numpy as np


# ============================================================
# PATH
# ============================================================

THIS_FILE = Path(__file__).resolve()

SIMULATION_DIR = THIS_FILE.parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent

STATE_DIR = MPC_SRBD_DIR / "State"
QP_DIR = MPC_SRBD_DIR / "QP"


for directory in (
    MPC_SRBD_DIR,
    SIMULATION_DIR,
    STATE_DIR,
    QP_DIR,
):
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(
            0,
            directory_str,
        )


# ============================================================
# PROJECT MODULES
# ============================================================

from simulation_setup import (
    initialize_simulation,
)

from get_current_state_mpc import (
    get_current_state_mpc,
)

from moment_constraint import (
    build_moment_constraints,
)

from QP_parameters import (
    mu,
    l_t,
    l_h,
    Mx_tolerance,
    enable_toe_heel_lateral_friction,
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=9,
    suppress=True,
    linewidth=220,
)


# ============================================================
# XML
# ============================================================

XML_PATH = (
    REPO_ROOT
    / "xmls"
    / "scene_flat_terrian_v3.xml"
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

N = 1
NU = 12

DT_MPC = 0.04
DT_MPC_UPDATE = 0.005

ABS_TOL = 1e-10


# ============================================================
# INPUT ORDER
#
# u =
#
# [
#   0  F_Lx
#   1  F_Ly
#   2  F_Lz
#
#   3  M_Lx
#   4  M_Ly
#   5  M_Lz
#
#   6  F_Rx
#   7  F_Ry
#   8  F_Rz
#
#   9  M_Rx
#   10 M_Ry
#   11 M_Rz
# ]
#
# All above are WORLD-frame quantities.
# ============================================================

INPUT_NAMES = (
    "F_Lx_W",
    "F_Ly_W",
    "F_Lz_W",
    "M_Lx_W",
    "M_Ly_W",
    "M_Lz_W",
    "F_Rx_W",
    "F_Ry_W",
    "F_Rz_W",
    "M_Rx_W",
    "M_Ry_W",
    "M_Rz_W",
)


# ============================================================
# HELPERS
# ============================================================

def print_header(
    title,
):

    print()

    print(
        "=" * 120
    )

    print(
        title
    )

    print(
        "=" * 120
    )


def print_subheader(
    title,
):

    print()

    print(
        "-" * 120
    )

    print(
        title
    )

    print(
        "-" * 120
    )


# ============================================================
# MANUAL FOOT-FRAME CONVERSION
# ============================================================

def world_to_foot_wrench(
    R_foot_W,
    F_W,
    M_W,
):

    R_foot_W = np.asarray(
        R_foot_W,
        dtype=float,
    ).reshape(
        3,
        3,
    )


    F_W = np.asarray(
        F_W,
        dtype=float,
    ).reshape(
        3
    )


    M_W = np.asarray(
        M_W,
        dtype=float,
    ).reshape(
        3
    )


    F_foot = (
        R_foot_W
        @
        F_W
    )


    M_foot = (
        R_foot_W
        @
        M_W
    )


    return (
        F_foot,
        M_foot,
    )


# ============================================================
# EXPECTED BASELINE MOMENT ROWS
#
# toe/heel lateral friction = False:
#
# row 0 : Left  Mx
# row 1 : Left  toe  My - l_h Fz <= 0
# row 2 : Left  heel -My - l_t Fz <= 0
#
# row 3 : Right Mx
# row 4 : Right toe
# row 5 : Right heel
# ============================================================

def manual_baseline_constraint_values(
    U,
    L6_R_W,
    R6_R_W,
):

    U = np.asarray(
        U,
        dtype=float,
    ).reshape(
        NU
    )


    F_L_W = U[
        0:3
    ]

    M_L_W = U[
        3:6
    ]


    F_R_W = U[
        6:9
    ]

    M_R_W = U[
        9:12
    ]


    (
        F_L_foot,
        M_L_foot,

    ) = world_to_foot_wrench(
        R_foot_W=L6_R_W,
        F_W=F_L_W,
        M_W=M_L_W,
    )


    (
        F_R_foot,
        M_R_foot,

    ) = world_to_foot_wrench(
        R_foot_W=R6_R_W,
        F_W=F_R_W,
        M_W=M_R_W,
    )


    expected = np.array(
        [
            # Left Mx
            M_L_foot[0],

            # Left toe
            M_L_foot[1]
            -
            l_h
            *
            F_L_foot[2],

            # Left heel
            -M_L_foot[1]
            -
            l_t
            *
            F_L_foot[2],

            # Right Mx
            M_R_foot[0],

            # Right toe
            M_R_foot[1]
            -
            l_h
            *
            F_R_foot[2],

            # Right heel
            -M_R_foot[1]
            -
            l_t
            *
            F_R_foot[2],
        ],
        dtype=float,
    )


    return {
        "expected":
            expected,

        "F_L_foot":
            F_L_foot,

        "M_L_foot":
            M_L_foot,

        "F_R_foot":
            F_R_foot,

        "M_R_foot":
            M_R_foot,
    }


# ============================================================
# TEST 1
#
# BASIS / INDEX MAPPING
# ============================================================

def test_basis_mapping(
    A,
):

    print_header(
        "TEST 1: DECISION-VARIABLE INDEX -> MOMENT-CONSTRAINT ROW"
    )


    print(
        "For each input element u[j] = 1,"
    )

    print(
        "the table below shows which moment-constraint rows respond."
    )


    print()

    print(
        "Expected baseline rows:"
    )

    print(
        "  row 0 = LEFT  local Mx"
    )

    print(
        "  row 1 = LEFT  toe"
    )

    print(
        "  row 2 = LEFT  heel"
    )

    print(
        "  row 3 = RIGHT local Mx"
    )

    print(
        "  row 4 = RIGHT toe"
    )

    print(
        "  row 5 = RIGHT heel"
    )


    for column in range(
        NU
    ):

        U = np.zeros(
            NU,
            dtype=float,
        )

        U[
            column
        ] = 1.0


        value = (
            A
            @
            U
        )


        active_rows = np.where(
            np.abs(
                value
            )
            >
            1e-12
        )[0]


        print()

        print(
            f"u[{column:2d}] = {INPUT_NAMES[column]:8s}"
        )

        print(
            f"active rows = {active_rows.tolist()}"
        )


        for row in active_rows:

            print(
                f"    row {row:2d}: "
                f"A[row] @ U = {value[row]:+.9f}"
            )


    print()

    print(
        "TEST 1 COMPLETE"
    )


# ============================================================
# TEST 2
#
# MANUAL MATRIX CONSISTENCY
# ============================================================

def test_manual_consistency(
    A,
    L6_R_W,
    R6_R_W,
):

    print_header(
        "TEST 2: A @ U VS MANUAL WORLD -> FOOT TRANSFORMATION"
    )


    # --------------------------------------------------------
    # Deliberately use non-zero values for EVERY component.
    #
    # This makes frame mixing visible.
    # --------------------------------------------------------

    U = np.array(
        [
            # LEFT
            +1.20,
            -0.80,
            +14.00,

            -0.015749362,
            +0.074786737,
            +0.002500000,

            # RIGHT
            -0.60,
            +0.45,
            +10.50,

            +0.015749362,
            +0.055000000,
            -0.003000000,
        ],
        dtype=float,
    )


    matrix_value = (
        A
        @
        U
    )


    manual = (
        manual_baseline_constraint_values(
            U=U,
            L6_R_W=L6_R_W,
            R6_R_W=R6_R_W,
        )
    )


    expected = manual[
        "expected"
    ]


    error = (
        matrix_value
        -
        expected
    )


    print(
        "Test U [WORLD frame] ="
    )

    print(
        U
    )


    print()

    print(
        "LEFT force in foot frame:"
    )

    print(
        manual[
            "F_L_foot"
        ]
    )

    print(
        "LEFT moment in foot frame:"
    )

    print(
        manual[
            "M_L_foot"
        ]
    )


    print()

    print(
        "RIGHT force in foot frame:"
    )

    print(
        manual[
            "F_R_foot"
        ]
    )

    print(
        "RIGHT moment in foot frame:"
    )

    print(
        manual[
            "M_R_foot"
        ]
    )


    print()

    print(
        "row | A@U              manual             error"
    )

    print(
        "-" * 70
    )


    for row in range(
        A.shape[0]
    ):

        print(
            f"{row:3d} | "
            f"{matrix_value[row]:+16.10f} "
            f"{expected[row]:+16.10f} "
            f"{error[row]:+12.3e}"
        )


    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )


    print()

    print(
        f"max consistency error = "
        f"{max_error:.12e}"
    )


    if max_error <= ABS_TOL:

        print(
            "TEST 2: PASS"
        )

    else:

        print(
            "TEST 2: FAIL"
        )


# ============================================================
# TEST 3
#
# EXACT LOCAL Mx LIMIT
#
# Construct a moment IN FOOT FRAME:
#
#     M_foot = [Mx_tolerance, 0, 0]
#
# Convert it back:
#
#     M_W = R^T M_foot
#
# Then verify:
#
#     A_Mx @ U = Mx_tolerance
# ============================================================

def test_exact_local_mx_limit(
    A,
    L6_R_W,
    R6_R_W,
):

    print_header(
        "TEST 3: EXACT FOOT-FRAME Mx LIMIT"
    )


    target_local_moment = np.array(
        [
            Mx_tolerance,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    # ========================================================
    # LEFT
    # ========================================================

    M_L_W = (
        L6_R_W.T
        @
        target_local_moment
    )


    U_L = np.zeros(
        NU,
        dtype=float,
    )


    U_L[
        3:6
    ] = (
        M_L_W
    )


    value_L = (
        A
        @
        U_L
    )


    recovered_L_local = (
        L6_R_W
        @
        M_L_W
    )


    print_subheader(
        "LEFT FOOT"
    )


    print(
        "Target M_L_foot ="
    )

    print(
        target_local_moment
    )

    print(
        "Equivalent M_L_W ="
    )

    print(
        M_L_W
    )

    print(
        "Recovered M_L_foot ="
    )

    print(
        recovered_L_local
    )


    print()

    print(
        f"World-frame Mx          = "
        f"{M_L_W[0]:+.12f} N.m"
    )

    print(
        f"Foot-frame Mx           = "
        f"{recovered_L_local[0]:+.12f} N.m"
    )

    print(
        f"A moment row 0 @ U      = "
        f"{value_L[0]:+.12f} N.m"
    )

    print(
        f"Mx tolerance            = "
        f"{Mx_tolerance:+.12f} N.m"
    )


    left_pass = (
        abs(
            value_L[0]
            -
            Mx_tolerance
        )
        <=
        ABS_TOL
    )


    print(
        f"LEFT TEST              : "
        f"{'PASS' if left_pass else 'FAIL'}"
    )


    # ========================================================
    # RIGHT
    # ========================================================

    M_R_W = (
        R6_R_W.T
        @
        target_local_moment
    )


    U_R = np.zeros(
        NU,
        dtype=float,
    )


    U_R[
        9:12
    ] = (
        M_R_W
    )


    value_R = (
        A
        @
        U_R
    )


    recovered_R_local = (
        R6_R_W
        @
        M_R_W
    )


    print_subheader(
        "RIGHT FOOT"
    )


    print(
        "Target M_R_foot ="
    )

    print(
        target_local_moment
    )

    print(
        "Equivalent M_R_W ="
    )

    print(
        M_R_W
    )

    print(
        "Recovered M_R_foot ="
    )

    print(
        recovered_R_local
    )


    print()

    print(
        f"World-frame Mx          = "
        f"{M_R_W[0]:+.12f} N.m"
    )

    print(
        f"Foot-frame Mx           = "
        f"{recovered_R_local[0]:+.12f} N.m"
    )

    print(
        f"A moment row 3 @ U      = "
        f"{value_R[3]:+.12f} N.m"
    )

    print(
        f"Mx tolerance            = "
        f"{Mx_tolerance:+.12f} N.m"
    )


    right_pass = (
        abs(
            value_R[3]
            -
            Mx_tolerance
        )
        <=
        ABS_TOL
    )


    print(
        f"RIGHT TEST             : "
        f"{'PASS' if right_pass else 'FAIL'}"
    )


    return (
        left_pass
        and
        right_pass
    )


# ============================================================
# TEST 4
#
# WORLD Mx IS NOT NECESSARILY LOCAL Mx
# ============================================================

def test_world_vs_local_mx(
    A,
    L6_R_W,
    R6_R_W,
):

    print_header(
        "TEST 4: WORLD-FRAME Mx VS FOOT-FRAME Mx"
    )


    # --------------------------------------------------------
    # Deliberately include My and Mz.
    #
    # Then:
    #
    # local Mx =
    #
    # first row of R_foot_W
    # dot
    # [Mx_W, My_W, Mz_W]
    #
    # Therefore local Mx is NOT generally equal to Mx_W.
    # --------------------------------------------------------

    M_L_W = np.array(
        [
            -0.015749362,
            +0.074786737,
            +0.003000000,
        ],
        dtype=float,
    )


    M_R_W = np.array(
        [
            -0.015749362,
            +0.074786737,
            -0.003000000,
        ],
        dtype=float,
    )


    U = np.zeros(
        NU,
        dtype=float,
    )


    U[
        3:6
    ] = M_L_W


    U[
        9:12
    ] = M_R_W


    constraint_value = (
        A
        @
        U
    )


    M_L_foot = (
        L6_R_W
        @
        M_L_W
    )


    M_R_foot = (
        R6_R_W
        @
        M_R_W
    )


    print(
        "LEFT:"
    )

    print(
        f"Mx WORLD               = "
        f"{M_L_W[0]:+.12f}"
    )

    print(
        f"Mx FOOT                = "
        f"{M_L_foot[0]:+.12f}"
    )

    print(
        f"A row 0 @ U            = "
        f"{constraint_value[0]:+.12f}"
    )


    print()

    print(
        "RIGHT:"
    )

    print(
        f"Mx WORLD               = "
        f"{M_R_W[0]:+.12f}"
    )

    print(
        f"Mx FOOT                = "
        f"{M_R_foot[0]:+.12f}"
    )

    print(
        f"A row 3 @ U            = "
        f"{constraint_value[3]:+.12f}"
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The constraint acts on FOOT-frame Mx,"
    )

    print(
        "NOT directly on u[3] or u[9] interpreted as "
        "World-frame Mx."
    )


# ============================================================
# TEST 5
#
# PRINT EXACT Mx ROW COEFFICIENTS
# ============================================================

def test_print_mx_rows(
    A,
    L6_R_W,
    R6_R_W,
):

    print_header(
        "TEST 5: EXACT Mx CONSTRAINT ROW COEFFICIENTS"
    )


    print(
        "L6_R_W ="
    )

    print(
        L6_R_W
    )


    print()

    print(
        "R6_R_W ="
    )

    print(
        R6_R_W
    )


    print()

    print(
        "LEFT Mx constraint row A[0] ="
    )

    print(
        A[
            0
        ]
    )


    print()

    print(
        "Relevant LEFT moment coefficients:"
    )

    print(
        "A[0, 3:6] ="
    )

    print(
        A[
            0,
            3:6
        ]
    )


    print()

    print(
        "Expected = first row of L6_R_W:"
    )

    print(
        L6_R_W[
            0,
            :
        ]
    )


    print()

    print(
        "RIGHT Mx constraint row A[3] ="
    )

    print(
        A[
            3
        ]
    )


    print()

    print(
        "Relevant RIGHT moment coefficients:"
    )

    print(
        "A[3, 9:12] ="
    )

    print(
        A[
            3,
            9:12
        ]
    )


    print()

    print(
        "Expected = first row of R6_R_W:"
    )

    print(
        R6_R_W[
            0,
            :
        ]
    )


    left_error = np.max(
        np.abs(
            A[
                0,
                3:6
            ]
            -
            L6_R_W[
                0,
                :
            ]
        )
    )


    right_error = np.max(
        np.abs(
            A[
                3,
                9:12
            ]
            -
            R6_R_W[
                0,
                :
            ]
        )
    )


    passed = (
        left_error <= ABS_TOL
        and
        right_error <= ABS_TOL
    )


    print()

    print(
        f"LEFT row error  = "
        f"{left_error:.12e}"
    )

    print(
        f"RIGHT row error = "
        f"{right_error:.12e}"
    )


    print(
        f"TEST 5: "
        f"{'PASS' if passed else 'FAIL'}"
    )


    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "TEST: MOMENT-CONSTRAINT INPUT ORDER AND FRAME MAPPING"
    )


    print(
        f"XML = {XML_PATH}"
    )


    print()

    print(
        f"Mx_tolerance = "
        f"{Mx_tolerance} N.m"
    )

    print(
        f"l_t          = "
        f"{l_t} m"
    )

    print(
        f"l_h          = "
        f"{l_h} m"
    )

    print(
        f"mu           = "
        f"{mu}"
    )

    print(
        f"extended toe/heel friction = "
        f"{enable_toe_heel_lateral_friction}"
    )


    # ========================================================
    # THIS TEST EXPECTS BASELINE MODE
    # ========================================================

    if enable_toe_heel_lateral_friction:

        raise RuntimeError(
            "This diagnostic is written for baseline mode "
            "enable_toe_heel_lateral_friction = False."
        )


    # ========================================================
    # INITIALIZE GEOMETRY
    # ========================================================

    setup = (
        initialize_simulation(
            xml_path=XML_PATH,
            horizon=N,
            dt_mpc=DT_MPC,
            dt_mpc_update=(
                DT_MPC_UPDATE
            ),
            home_key_name="home",
            qp_verbose=False,
            print_initialization=False,
        )
    )


    initial = (
        get_current_state_mpc(
            model=setup.model,
            data=setup.data,
            B_inertia_CoM=(
                setup.B_inertia_CoM
            ),
        )
    )


    L6_R_W = np.asarray(
        initial.L6_IK_R_W,
        dtype=float,
    ).reshape(
        3,
        3,
    )


    R6_R_W = np.asarray(
        initial.R6_IK_R_W,
        dtype=float,
    ).reshape(
        3,
        3,
    )


    # ========================================================
    # BUILD MOMENT CONSTRAINT
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
        Mx_tolerance=(
            Mx_tolerance
        ),
        enable_toe_heel_lateral_friction=(
            enable_toe_heel_lateral_friction
        ),
    )


    print()

    print(
        f"A_moment shape = "
        f"{A_moment.shape}"
    )


    print(
        "lower bounds ="
    )

    print(
        l_moment
    )


    print(
        "upper bounds ="
    )

    print(
        u_moment
    )


    if A_moment.shape != (
        6,
        12,
    ):

        raise RuntimeError(
            f"Expected baseline moment matrix shape "
            f"(6, 12), received {A_moment.shape}."
        )


    # ========================================================
    # RUN TESTS
    # ========================================================

    test_basis_mapping(
        A=A_moment,
    )


    test_manual_consistency(
        A=A_moment,
        L6_R_W=L6_R_W,
        R6_R_W=R6_R_W,
    )


    pass_test_3 = (
        test_exact_local_mx_limit(
            A=A_moment,
            L6_R_W=L6_R_W,
            R6_R_W=R6_R_W,
        )
    )


    test_world_vs_local_mx(
        A=A_moment,
        L6_R_W=L6_R_W,
        R6_R_W=R6_R_W,
    )


    pass_test_5 = (
        test_print_mx_rows(
            A=A_moment,
            L6_R_W=L6_R_W,
            R6_R_W=R6_R_W,
        )
    )


    # ========================================================
    # FINAL
    # ========================================================

    print_header(
        "FINAL INTERPRETATION"
    )


    if (
        pass_test_3
        and
        pass_test_5
    ):

        print(
            "MOMENT CONSTRAINT MAPPING: PASS"
        )


        print()

        print(
            "Expected decision-variable ordering is:"
        )

        print(
            "[FL_W, ML_W, FR_W, MR_W]"
        )


        print()

        print(
            "IMPORTANT:"
        )

        print(
            "u[3] and u[9] are WORLD-frame Mx components."
        )

        print(
            "The constrained quantities are FOOT-frame Mx:"
        )


        print()

        print(
            "Mx_L_foot = "
            "L6_R_W[0,:] @ U[3:6]"
        )

        print(
            "Mx_R_foot = "
            "R6_R_W[0,:] @ U[9:12]"
        )


        print()

        print(
            "Therefore:"
        )

        print(
            "|u[3]| > Mx_tolerance"
        )

        print(
            "does NOT automatically mean the Mx constraint "
            "is violated."
        )


        print()

        print(
            "For MPC diagnostics, print BOTH:"
        )

        print(
            "Mx_world and Mx_foot."
        )


    else:

        print(
            "MOMENT CONSTRAINT MAPPING: FAIL"
        )

        print(
            "Inspect moment_constraint.py before continuing."
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()