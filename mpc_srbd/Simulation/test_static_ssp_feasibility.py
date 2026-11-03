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

from force_constraint import (
    build_force_constraints,
)

from moment_constraint import (
    build_moment_constraints,
)

from contact_constraint import (
    build_contact_constraints,
)

from QP_parameters import (
    mu,
    Fmax,
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
# TEST PARAMETERS
# ============================================================

N = 1

DT_MPC = 0.04

DT_MPC_UPDATE = 0.005

GRAVITY = 9.81


# Numerical feasibility tolerance only for this diagnostic.
#
# This is NOT the OSQP tolerance.
STATIC_CONSTRAINT_TOL = 1e-9


# ============================================================
# LATERAL COM SWEEP
#
# Sweep sufficiently beyond both feet.
# ============================================================

Y_SWEEP_MIN = -0.12
Y_SWEEP_MAX = +0.12

Y_SWEEP_STEP = 0.00025


# ============================================================
# HELPERS
# ============================================================

def build_static_single_support_wrench(
    mass,
    p_com,
    p_stance,
    stance_leg,
):
    """
    Build the UNIQUE wrench required for exact static equilibrium.

    Static translational equilibrium:

        F + m g = 0

    therefore:

        F = [0, 0, mg]

    Static rotational equilibrium about CoM:

        r x F + M = 0

    therefore:

        M = -(r x F)

    where:

        r = p_stance - p_com

    Right/swing foot wrench is exactly zero.
    """

    mass = float(mass)

    p_com = np.asarray(
        p_com,
        dtype=float,
    ).reshape(3)

    p_stance = np.asarray(
        p_stance,
        dtype=float,
    ).reshape(3)


    # --------------------------------------------------------
    # Moment arm: CoM -> stance contact point
    # --------------------------------------------------------

    r = (
        p_stance
        -
        p_com
    )


    # --------------------------------------------------------
    # Static support force
    # --------------------------------------------------------

    F = np.array(
        [
            0.0,
            0.0,
            mass * GRAVITY,
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # Required contact moment
    #
    # r x F + M = 0
    # --------------------------------------------------------

    M = (
        -np.cross(
            r,
            F,
        )
    )


    # --------------------------------------------------------
    # Full MPC input:
    #
    # [FL, ML, FR, MR]
    # --------------------------------------------------------

    U = np.zeros(
        12,
        dtype=float,
    )


    if stance_leg == "L":

        U[0:3] = F
        U[3:6] = M

    elif stance_leg == "R":

        U[6:9] = F
        U[9:12] = M

    else:

        raise ValueError(
            "stance_leg must be 'L' or 'R'."
        )


    return {
        "U":
            U,

        "r":
            r,

        "F":
            F,

        "M":
            M,
    }


# ============================================================
# CONSTRAINT VIOLATION
# ============================================================

def compute_constraint_violation(
    A,
    lower,
    upper,
    U,
):
    """
    Compute maximum violation of:

        lower <= A @ U <= upper
    """

    A = np.asarray(
        A,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    ).reshape(-1)

    upper = np.asarray(
        upper,
        dtype=float,
    ).reshape(-1)

    U = np.asarray(
        U,
        dtype=float,
    ).reshape(-1)


    value = (
        A
        @
        U
    )


    lower_violation = np.zeros_like(
        value,
        dtype=float,
    )

    upper_violation = np.zeros_like(
        value,
        dtype=float,
    )


    finite_lower = np.isfinite(
        lower
    )

    finite_upper = np.isfinite(
        upper
    )


    lower_violation[
        finite_lower
    ] = np.maximum(
        lower[
            finite_lower
        ]
        -
        value[
            finite_lower
        ],
        0.0,
    )


    upper_violation[
        finite_upper
    ] = np.maximum(
        value[
            finite_upper
        ]
        -
        upper[
            finite_upper
        ],
        0.0,
    )


    row_violation = np.maximum(
        lower_violation,
        upper_violation,
    )


    if row_violation.size == 0:

        max_violation = 0.0
        max_row = None

    else:

        max_row = int(
            np.argmax(
                row_violation
            )
        )

        max_violation = float(
            row_violation[
                max_row
            ]
        )


    return {
        "value":
            value,

        "lower_violation":
            lower_violation,

        "upper_violation":
            upper_violation,

        "row_violation":
            row_violation,

        "max_violation":
            max_violation,

        "max_row":
            max_row,
    }


# ============================================================
# BUILD CURRENT PROJECT CONSTRAINTS
# ============================================================

def build_static_constraints(
    initial,
    stance_leg,
):
    """
    Build the exact same force/moment/contact constraints
    currently used by the project, for N = 1.
    """

    # --------------------------------------------------------
    # Force constraints
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Moment constraints
    #
    # Use corrected contact-frame rotations,
    # consistent with current MPC call.
    # --------------------------------------------------------

    (
        A_moment,
        l_moment,
        u_moment,

    ) = build_moment_constraints(
        N=N,
        L6_R_W=(
            initial.L6_IK_R_W
        ),
        R6_R_W=(
            initial.R6_IK_R_W
        ),
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


    # --------------------------------------------------------
    # Contact activation
    # --------------------------------------------------------

    if stance_leg == "L":

        contact_schedule = np.array(
            [
                [1, 0],
            ],
            dtype=int,
        )

    elif stance_leg == "R":

        contact_schedule = np.array(
            [
                [0, 1],
            ],
            dtype=int,
        )

    else:

        raise ValueError(
            "stance_leg must be 'L' or 'R'."
        )


    (
        A_contact,
        l_contact,
        u_contact,

    ) = build_contact_constraints(
        N=N,
        contact_schedule=(
            contact_schedule
        ),
    )


    return {
        "force":
            (
                A_force,
                l_force,
                u_force,
            ),

        "moment":
            (
                A_moment,
                l_moment,
                u_moment,
            ),

        "contact":
            (
                A_contact,
                l_contact,
                u_contact,
            ),

        "contact_schedule":
            contact_schedule,
    }


# ============================================================
# EVALUATE ONE STATIC CONFIGURATION
# ============================================================

def evaluate_static_configuration(
    mass,
    p_com,
    p_stance,
    stance_leg,
    constraint_set,
):

    static = (
        build_static_single_support_wrench(
            mass=mass,
            p_com=p_com,
            p_stance=p_stance,
            stance_leg=stance_leg,
        )
    )


    U = static[
        "U"
    ]


    results = {}


    for block_name in (
        "force",
        "moment",
        "contact",
    ):

        (
            A,
            lower,
            upper,

        ) = constraint_set[
            block_name
        ]


        results[
            block_name
        ] = (
            compute_constraint_violation(
                A=A,
                lower=lower,
                upper=upper,
                U=U,
            )
        )


    max_force = (
        results[
            "force"
        ][
            "max_violation"
        ]
    )

    max_moment = (
        results[
            "moment"
        ][
            "max_violation"
        ]
    )

    max_contact = (
        results[
            "contact"
        ][
            "max_violation"
        ]
    )


    max_total = max(
        max_force,
        max_moment,
        max_contact,
    )


    feasible = (
        max_total
        <=
        STATIC_CONSTRAINT_TOL
    )


    return {
        "feasible":
            feasible,

        "max_total":
            max_total,

        "force":
            results[
                "force"
            ],

        "moment":
            results[
                "moment"
            ],

        "contact":
            results[
                "contact"
            ],

        "U":
            U,

        "r":
            static[
                "r"
            ],

        "F":
            static[
                "F"
            ],

        "M":
            static[
                "M"
            ],
    }


# ============================================================
# PRINT INITIAL SSP RESULT
# ============================================================

def print_initial_result(
    stance_leg,
    p_com,
    p_stance,
    evaluation,
):

    print()

    print(
        "=" * 100
    )

    print(
        f"STATIC SSP TEST - {stance_leg} STANCE"
    )

    print(
        "=" * 100
    )


    print(
        f"CoM position             : "
        f"{p_com}"
    )

    print(
        f"stance contact position  : "
        f"{p_stance}"
    )

    print(
        f"r = p_foot - p_CoM       : "
        f"{evaluation['r']}"
    )


    print()

    print(
        "Required static wrench:"
    )


    print(
        f"F stance [World]         : "
        f"{evaluation['F']}"
    )

    print(
        f"M stance [World]         : "
        f"{evaluation['M']}"
    )


    print()

    print(
        f"required |Mx World|      : "
        f"{abs(evaluation['M'][0]):.9f} N.m"
    )

    print(
        f"configured Mx tolerance  : "
        f"{Mx_tolerance:.9f} N.m"
    )


    print()

    print(
        f"force max violation      : "
        f"{evaluation['force']['max_violation']:.12e}"
    )

    print(
        f"moment max violation     : "
        f"{evaluation['moment']['max_violation']:.12e}"
    )

    print(
        f"contact max violation    : "
        f"{evaluation['contact']['max_violation']:.12e}"
    )

    print(
        f"overall max violation    : "
        f"{evaluation['max_total']:.12e}"
    )


    print()

    print(
        f"STATIC FEASIBLE          : "
        f"{evaluation['feasible']}"
    )


# ============================================================
# SWEEP COM Y
# ============================================================

def sweep_com_y(
    mass,
    p_com_initial,
    p_stance,
    stance_leg,
    constraint_set,
):

    y_values = np.arange(
        Y_SWEEP_MIN,
        Y_SWEEP_MAX
        +
        0.5
        *
        Y_SWEEP_STEP,
        Y_SWEEP_STEP,
        dtype=float,
    )


    records = []


    for y_candidate in y_values:

        p_com = (
            np.asarray(
                p_com_initial,
                dtype=float,
            )
            .copy()
        )


        p_com[1] = float(
            y_candidate
        )


        evaluation = (
            evaluate_static_configuration(
                mass=mass,
                p_com=p_com,
                p_stance=p_stance,
                stance_leg=stance_leg,
                constraint_set=(
                    constraint_set
                ),
            )
        )


        records.append(
            {
                "y":
                    float(
                        y_candidate
                    ),

                "feasible":
                    bool(
                        evaluation[
                            "feasible"
                        ]
                    ),

                "max_total":
                    float(
                        evaluation[
                            "max_total"
                        ]
                    ),

                "max_force":
                    float(
                        evaluation[
                            "force"
                        ][
                            "max_violation"
                        ]
                    ),

                "max_moment":
                    float(
                        evaluation[
                            "moment"
                        ][
                            "max_violation"
                        ]
                    ),

                "max_contact":
                    float(
                        evaluation[
                            "contact"
                        ][
                            "max_violation"
                        ]
                    ),

                "Mx_required":
                    float(
                        evaluation[
                            "M"
                        ][0]
                    ),

                "My_required":
                    float(
                        evaluation[
                            "M"
                        ][1]
                    ),
            }
        )


    return records


# ============================================================
# PRINT SWEEP SUMMARY
# ============================================================

def print_sweep_summary(
    stance_leg,
    records,
    y_initial,
    y_stance,
):

    feasible_records = [
        record
        for record in records
        if record[
            "feasible"
        ]
    ]


    print()

    print(
        "=" * 100
    )

    print(
        f"LATERAL CoM SWEEP - {stance_leg} STANCE"
    )

    print(
        "=" * 100
    )


    print(
        f"initial CoM y            : "
        f"{y_initial:+.9f} m"
    )

    print(
        f"stance foot y            : "
        f"{y_stance:+.9f} m"
    )

    print(
        f"sweep range              : "
        f"[{Y_SWEEP_MIN:+.6f}, "
        f"{Y_SWEEP_MAX:+.6f}] m"
    )

    print(
        f"sweep step               : "
        f"{Y_SWEEP_STEP:.6f} m"
    )


    if len(
        feasible_records
    ) == 0:

        print()

        print(
            "NO STATICALLY FEASIBLE y_CoM "
            "FOUND IN SWEEP RANGE."
        )

        return


    feasible_y = np.array(
        [
            record[
                "y"
            ]
            for record
            in feasible_records
        ],
        dtype=float,
    )


    y_min = float(
        np.min(
            feasible_y
        )
    )

    y_max = float(
        np.max(
            feasible_y
        )
    )


    # --------------------------------------------------------
    # Find feasible point requiring minimum movement
    # from initial CoM y
    # --------------------------------------------------------

    best_record = min(
        feasible_records,
        key=lambda record:
            abs(
                record[
                    "y"
                ]
                -
                y_initial
            ),
    )


    y_nearest = float(
        best_record[
            "y"
        ]
    )


    required_shift = (
        y_nearest
        -
        y_initial
    )


    distance_to_stance = (
        y_stance
        -
        y_initial
    )


    print()

    print(
        f"STATIC FEASIBLE y range : "
        f"[{y_min:+.9f}, "
        f"{y_max:+.9f}] m"
    )


    print()

    print(
        f"nearest feasible y_CoM  : "
        f"{y_nearest:+.9f} m"
    )

    print(
        f"required lateral shift  : "
        f"{required_shift:+.9f} m"
    )

    print(
        f"distance CoM -> foot y  : "
        f"{distance_to_stance:+.9f} m"
    )


    if abs(
        distance_to_stance
    ) > 1e-12:

        fraction = (
            required_shift
            /
            distance_to_stance
        )

        print(
            f"fraction toward stance  : "
            f"{fraction:+.6f}"
        )


    print()

    print(
        "Nearest feasible point:"
    )

    print(
        f"Mx required             : "
        f"{best_record['Mx_required']:+.9f} N.m"
    )

    print(
        f"My required             : "
        f"{best_record['My_required']:+.9f} N.m"
    )

    print(
        f"max force violation     : "
        f"{best_record['max_force']:.3e}"
    )

    print(
        f"max moment violation    : "
        f"{best_record['max_moment']:.3e}"
    )


    # --------------------------------------------------------
    # Print several points around the feasibility boundary
    # --------------------------------------------------------

    print()

    print(
        "-" * 100
    )

    print(
        "SAMPLES NEAR NEAREST FEASIBLE POINT"
    )

    print(
        "-" * 100
    )


    index_nearest = min(
        range(
            len(
                records
            )
        ),
        key=lambda index:
            abs(
                records[
                    index
                ][
                    "y"
                ]
                -
                y_nearest
            ),
    )


    start = max(
        0,
        index_nearest
        -
        4,
    )

    stop = min(
        len(
            records
        ),
        index_nearest
        +
        5,
    )


    print(
        "       y_CoM | feasible |"
        "   Mx_required |"
        " max_moment_violation"
    )


    for record in records[
        start:
        stop
    ]:

        print(
            f"{record['y']:+.9f} | "
            f"{str(record['feasible']):>8} | "
            f"{record['Mx_required']:+.9f} | "
            f"{record['max_moment']:.9e}"
        )


# ============================================================
# ANALYTIC ROLL-MOMENT ESTIMATE
# ============================================================

def print_analytic_estimate(
    mass,
    y_initial,
    y_stance,
):

    mg = (
        mass
        *
        GRAVITY
    )


    # For level foot and purely vertical support force:
    #
    # Mx_required = -(y_foot - y_CoM) * mg
    #
    # Therefore:
    #
    # |y_foot - y_CoM|
    # <=
    # Mx_tolerance / mg
    #
    # approximately.
    #

    max_static_lateral_offset = (
        Mx_tolerance
        /
        mg
    )


    current_offset = abs(
        y_stance
        -
        y_initial
    )


    print()

    print(
        "-" * 100
    )

    print(
        "ANALYTIC LEVEL-FOOT ESTIMATE"
    )

    print(
        "-" * 100
    )


    print(
        f"mg                       : "
        f"{mg:.9f} N"
    )

    print(
        f"|y_foot - y_CoM| initial : "
        f"{current_offset:.9f} m"
    )

    print(
        f"Mx tolerance             : "
        f"{Mx_tolerance:.9f} N.m"
    )

    print(
        f"approx max static offset : "
        f"{max_static_lateral_offset:.9f} m"
    )

    print(
        f"approx max static offset : "
        f"{1000.0 * max_static_lateral_offset:.3f} mm"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "TEST: STATIC SINGLE-SUPPORT FEASIBILITY"
    )

    print(
        "=" * 100
    )


    print(
        f"XML                         : "
        f"{XML_PATH}"
    )

    print(
        "MuJoCo mj_step              : NO"
    )

    print(
        "MPC solve                   : NO"
    )

    print(
        "Gait scheduler              : NO"
    )

    print(
        "Static equilibrium          : YES"
    )

    print(
        "Current project constraints : YES"
    )


    print()

    print(
        f"mu                          : "
        f"{mu}"
    )

    print(
        f"Fmax                        : "
        f"{Fmax} N"
    )

    print(
        f"Mx_tolerance                : "
        f"{Mx_tolerance} N.m"
    )

    print(
        f"l_t                         : "
        f"{l_t} m"
    )

    print(
        f"l_h                         : "
        f"{l_h} m"
    )

    print(
        f"toe/heel lateral friction   : "
        f"{enable_toe_heel_lateral_friction}"
    )


    # ========================================================
    # INITIALIZE MODEL
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


    p_com_initial = np.asarray(
        initial.x[
            0:3
        ],
        dtype=float,
    )


    # ========================================================
    # Recover actual L6 / R6 positions
    #
    # r = p_foot - p_CoM
    # ========================================================

    p_L6 = (
        p_com_initial
        +
        initial.W_r_L
    )

    p_R6 = (
        p_com_initial
        +
        initial.W_r_R
    )


    mass = float(
        setup.mass_CoM
    )


    print()

    print(
        "=" * 100
    )

    print(
        "INITIAL GEOMETRY"
    )

    print(
        "=" * 100
    )


    print(
        f"mass_CoM              : "
        f"{mass:.9f} kg"
    )

    print(
        f"robot weight          : "
        f"{mass * GRAVITY:.9f} N"
    )

    print(
        f"CoM initial           : "
        f"{p_com_initial}"
    )

    print(
        f"L6 position           : "
        f"{p_L6}"
    )

    print(
        f"R6 position           : "
        f"{p_R6}"
    )

    print(
        f"L6 y - CoM y         : "
        f"{p_L6[1] - p_com_initial[1]:+.9f} m"
    )

    print(
        f"R6 y - CoM y         : "
        f"{p_R6[1] - p_com_initial[1]:+.9f} m"
    )


    # ========================================================
    # LEFT STANCE
    # ========================================================

    left_constraints = (
        build_static_constraints(
            initial=initial,
            stance_leg="L",
        )
    )


    left_initial = (
        evaluate_static_configuration(
            mass=mass,
            p_com=p_com_initial,
            p_stance=p_L6,
            stance_leg="L",
            constraint_set=(
                left_constraints
            ),
        )
    )


    print_initial_result(
        stance_leg="L",
        p_com=p_com_initial,
        p_stance=p_L6,
        evaluation=left_initial,
    )


    print_analytic_estimate(
        mass=mass,
        y_initial=float(
            p_com_initial[1]
        ),
        y_stance=float(
            p_L6[1]
        ),
    )


    left_records = (
        sweep_com_y(
            mass=mass,
            p_com_initial=(
                p_com_initial
            ),
            p_stance=p_L6,
            stance_leg="L",
            constraint_set=(
                left_constraints
            ),
        )
    )


    print_sweep_summary(
        stance_leg="L",
        records=left_records,
        y_initial=float(
            p_com_initial[1]
        ),
        y_stance=float(
            p_L6[1]
        ),
    )


    # ========================================================
    # RIGHT STANCE
    # ========================================================

    right_constraints = (
        build_static_constraints(
            initial=initial,
            stance_leg="R",
        )
    )


    right_initial = (
        evaluate_static_configuration(
            mass=mass,
            p_com=p_com_initial,
            p_stance=p_R6,
            stance_leg="R",
            constraint_set=(
                right_constraints
            ),
        )
    )


    print_initial_result(
        stance_leg="R",
        p_com=p_com_initial,
        p_stance=p_R6,
        evaluation=right_initial,
    )


    print_analytic_estimate(
        mass=mass,
        y_initial=float(
            p_com_initial[1]
        ),
        y_stance=float(
            p_R6[1]
        ),
    )


    right_records = (
        sweep_com_y(
            mass=mass,
            p_com_initial=(
                p_com_initial
            ),
            p_stance=p_R6,
            stance_leg="R",
            constraint_set=(
                right_constraints
            ),
        )
    )


    print_sweep_summary(
        stance_leg="R",
        records=right_records,
        y_initial=float(
            p_com_initial[1]
        ),
        y_stance=float(
            p_R6[1]
        ),
    )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "FINAL INTERPRETATION"
    )

    print(
        "=" * 100
    )


    print(
        "1. STATIC FEASIBLE = False at initial CoM:"
    )

    print(
        "   The robot cannot maintain exact zero-acceleration "
        "single support at that CoM under the current "
        "contact-wrench limits."
    )


    print()

    print(
        "2. The lateral sweep gives the y_CoM region where "
        "exact static single support becomes feasible."
    )


    print()

    print(
        "3. This is a STATIC / quasi-static diagnostic."
    )

    print(
        "   Dynamic walking may remain feasible outside that "
        "region because nonzero lateral acceleration allows "
        "Fy and angular dynamics to participate."
    )


    print()

    print(
        "4. Therefore this test must NOT be interpreted as:"
    )

    print(
        "   'the robot cannot walk'."
    )

    print(
        "   It answers only:"
    )

    print(
        "   'Can this configuration support the full robot "
        "weight on one foot with zero linear and angular "
        "acceleration?'"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()