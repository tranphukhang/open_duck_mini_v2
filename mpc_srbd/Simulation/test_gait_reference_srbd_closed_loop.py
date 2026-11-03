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
SRBD_DIR = MPC_SRBD_DIR / "SRBD"
QP_DIR = MPC_SRBD_DIR / "QP"
REF_DIR = MPC_SRBD_DIR / "Ref_generator"


for directory in (
    MPC_SRBD_DIR,
    SIMULATION_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    REF_DIR,
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

from W_euler_base import (
    get_Ew_inverse,
)

from continuous_model import (
    build_continuous_model,
)

from discretization import (
    discretize_model,
)

from build_SRBD_model import (
    build_SRBD_model,
)

from build_and_solve_QP import (
    build_and_solve_QP,
)

from QP_parameters import (
    mu,
    Fmax,
    Mx_tolerance,
    enable_toe_heel_lateral_friction,
)

from gait_scheduler import (
    GaitScheduler,
)

from reference_generator import (
    ReferenceGenerator,
    euler_to_rotation_matrix,
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=9,
    suppress=True,
    linewidth=240,
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
# DIMENSIONS
# ============================================================

NX = 13
NU = 12

LEFT = 0
RIGHT = 1


# ============================================================
# TIMING
# ============================================================

CONTROL_DT = 0.001

MPC_UPDATE_DT = 0.005

MPC_DT = 0.040

MPC_HORIZON = 10

REFERENCE_HORIZON = (
    MPC_HORIZON
    +
    1
)


MPC_UPDATE_DECIMATION = int(
    round(
        MPC_UPDATE_DT
        /
        CONTROL_DT
    )
)


if not np.isclose(
    MPC_UPDATE_DECIMATION
    *
    CONTROL_DT,
    MPC_UPDATE_DT,
):
    raise ValueError(
        "MPC_UPDATE_DT must be an integer multiple "
        "of CONTROL_DT."
    )


# ============================================================
# GAIT
# ============================================================

SSP_DURATIONS = (
    6,
    6,
)

DSP_DURATIONS = (
    3,
    3,
)

STARTUP_HOLD_TIME = 0.250

FIRST_SWING_LEG = "R"


# ============================================================
# TEST DURATION
# ============================================================

TEST_DURATION = 0.605


# ============================================================
# USER COMMAND
#
# U0 diagnostic:
#
# vx = 0
# vy = 0
# yaw_rate = 0
# ============================================================

V_DES_BODY_TARGET = np.array(
    [
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

YAW_RATE_DES_TARGET = 0.0


# ============================================================
# QP SETTINGS
# ============================================================

QP_MAX_ITER = 10000

QP_CONSTRAINT_TOLERANCE = 1e-4


# ============================================================
# GENERAL DIAGNOSTICS
# ============================================================

INACTIVE_WRENCH_TOL = 5e-4

FORCE_TOL = 5e-4

PRINT_EVERY_N_MPC = 10

TOP_VIOLATIONS_TO_PRINT = 15

VIOLATION_PRINT_THRESHOLD = 1e-10


# ============================================================
# NEW:
# PREDICTION-HORIZON DIAGNOSTIC TIMES
#
# At these actual closed-loop times we print:
#
#   predicted:
#       y
#       vy
#       z
#       vz
#       roll
#       wx
#
#   and optimized wrench:
#       FyL, FzL, MxL
#       FyR, FzR, MxR
#
# for every prediction knot.
#
# No control law is changed.
# ============================================================

PREDICTION_LOG_TIMES = (
    0.250,
    0.300,
    0.350,
    0.370,
)


# ============================================================
# USER COMMAND
# ============================================================

def get_reference_command(t):

    if t < STARTUP_HOLD_TIME:

        return (
            np.zeros(
                3,
                dtype=float,
            ),
            0.0,
        )

    return (
        V_DES_BODY_TARGET.copy(),
        float(
            YAW_RATE_DES_TARGET
        ),
    )


# ============================================================
# CHECK IF PREDICTION LOG IS REQUIRED
# ============================================================

def should_print_prediction_log(t):

    for target_time in PREDICTION_LOG_TIMES:

        if np.isclose(
            t,
            target_time,
            atol=1e-12,
            rtol=0.0,
        ):
            return True

    return False


# ============================================================
# SPLIT WRENCH
# ============================================================

def split_wrench(u):

    u = np.asarray(
        u,
        dtype=float,
    ).reshape(
        NU
    )

    return {
        "F_L":
            u[0:3].copy(),

        "M_L":
            u[3:6].copy(),

        "F_R":
            u[6:9].copy(),

        "M_R":
            u[9:12].copy(),
    }


# ============================================================
# RUNTIME SRBD QUANTITIES
# ============================================================

def build_runtime_srbd_quantities(
    x_current,
    B_inertia_CoM,
    W_p_L6,
    W_p_R6,
):

    x_current = np.asarray(
        x_current,
        dtype=float,
    ).reshape(
        NX
    )

    B_inertia_CoM = np.asarray(
        B_inertia_CoM,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    W_p_L6 = np.asarray(
        W_p_L6,
        dtype=float,
    ).reshape(
        3
    )

    W_p_R6 = np.asarray(
        W_p_R6,
        dtype=float,
    ).reshape(
        3
    )


    W_p_CoM = x_current[0:3]

    W_euler = x_current[3:6]


    # --------------------------------------------------------
    # Euler kinematics
    # --------------------------------------------------------

    Ew_inverse = (
        get_Ew_inverse(
            W_euler
        )
    )


    # --------------------------------------------------------
    # Body -> World rotation
    # --------------------------------------------------------

    W_R_B = (
        euler_to_rotation_matrix(
            W_euler
        )
    )


    # --------------------------------------------------------
    # Whole-body inertia in World
    # --------------------------------------------------------

    W_inertia_CoM = (
        W_R_B
        @
        B_inertia_CoM
        @
        W_R_B.T
    )


    # --------------------------------------------------------
    # Moment arms
    # --------------------------------------------------------

    W_r_L = (
        W_p_L6
        -
        W_p_CoM
    )

    W_r_R = (
        W_p_R6
        -
        W_p_CoM
    )


    return {
        "Ew_inverse":
            np.asarray(
                Ew_inverse,
                dtype=float,
            ).copy(),

        "W_inertia_CoM":
            np.asarray(
                W_inertia_CoM,
                dtype=float,
            ).copy(),

        "W_r_L":
            np.asarray(
                W_r_L,
                dtype=float,
            ).copy(),

        "W_r_R":
            np.asarray(
                W_r_R,
                dtype=float,
            ).copy(),

        "W_R_B":
            np.asarray(
                W_R_B,
                dtype=float,
            ).copy(),
    }


# ============================================================
# CLOSED-LOOP SRBD PLANT
#
# x_k
#   ->
# apply u_k
#   ->
# x_(k+1)
# ============================================================

def integrate_srbd_plant(
    x_current,
    held_u,
    mass_CoM,
    B_inertia_CoM,
    W_p_L6,
    W_p_R6,
):

    runtime = (
        build_runtime_srbd_quantities(
            x_current=x_current,
            B_inertia_CoM=B_inertia_CoM,
            W_p_L6=W_p_L6,
            W_p_R6=W_p_R6,
        )
    )


    Ac, Bc = (
        build_continuous_model(
            Ew_inverse=runtime[
                "Ew_inverse"
            ],
            mass_CoM=mass_CoM,
            W_inertia_CoM=runtime[
                "W_inertia_CoM"
            ],
            W_r_L=runtime[
                "W_r_L"
            ],
            W_r_R=runtime[
                "W_r_R"
            ],
        )
    )


    Ad, Bd = (
        discretize_model(
            Ac=Ac,
            Bc=Bc,
            Ts=CONTROL_DT,
        )
    )


    x_next = (
        Ad
        @
        np.asarray(
            x_current,
            dtype=float,
        ).reshape(
            NX
        )
        +
        Bd
        @
        np.asarray(
            held_u,
            dtype=float,
        ).reshape(
            NU
        )
    )


    x_next[12] = 1.0


    if not np.all(
        np.isfinite(
            x_next
        )
    ):
        raise RuntimeError(
            "SRBD plant produced NaN or Inf."
        )


    return x_next


# ============================================================
# NEW:
# BUILD PREDICTED STATE TRAJECTORY
#
# X =
# Phi*x_current + Gamma*U
#
# X contains:
#
# x[k+1|k]
# ...
# x[k+N|k]
# ============================================================

def build_predicted_state_trajectory(
    Phi,
    Gamma,
    x_current,
    U_opt,
):

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
    ).reshape(
        NX
    )

    U_opt = np.asarray(
        U_opt,
        dtype=float,
    ).reshape(
        MPC_HORIZON
        *
        NU
    )


    X_pred_flat = (
        Phi
        @
        x_current
        +
        Gamma
        @
        U_opt
    )


    X_pred = (
        X_pred_flat.reshape(
            MPC_HORIZON,
            NX,
        )
    )


    return X_pred


# ============================================================
# NEW:
# PRINT COMPLETE PREDICTION HORIZON
#
# Purpose:
#
# Answer:
#
# "MPC đang dự đoán cái gì khiến Fz hiện tại giảm mạnh?"
#
#
# Table 1:
#
# predicted states
#
# Table 2:
#
# optimized wrench
# ============================================================

def print_prediction_horizon_diagnostics(
    t,
    x_current,
    X_ref_mpc,
    Phi,
    Gamma,
    U_opt,
    contact_schedule,
    runtime,
    mass_CoM,
):

    if U_opt is None:
        return


    X_pred = (
        build_predicted_state_trajectory(
            Phi=Phi,
            Gamma=Gamma,
            x_current=x_current,
            U_opt=U_opt,
        )
    )


    U = np.asarray(
        U_opt,
        dtype=float,
    ).reshape(
        MPC_HORIZON,
        NU,
    )


    X_ref_mpc = np.asarray(
        X_ref_mpc,
        dtype=float,
    ).reshape(
        MPC_HORIZON,
        NX,
    )


    contact_schedule = np.asarray(
        contact_schedule,
        dtype=int,
    ).reshape(
        MPC_HORIZON,
        2,
    )


    r_L = np.asarray(
        runtime[
            "W_r_L"
        ],
        dtype=float,
    ).reshape(
        3
    )

    r_R = np.asarray(
        runtime[
            "W_r_R"
        ],
        dtype=float,
    ).reshape(
        3
    )


    robot_weight = (
        float(
            mass_CoM
        )
        *
        9.81
    )


    print()

    print(
        "#" * 150
    )

    print(
        "PREDICTION-HORIZON DIAGNOSTICS"
    )

    print(
        "#" * 150
    )


    print(
        f"actual time t                 : "
        f"{t:.3f} s"
    )

    print(
        f"prediction spacing            : "
        f"{MPC_DT:.3f} s"
    )

    print(
        f"horizon                       : "
        f"{MPC_HORIZON}"
    )

    print(
        f"robot weight                  : "
        f"{robot_weight:.9f} N"
    )

    print(
        f"current y                     : "
        f"{x_current[1]:+.9f} m"
    )

    print(
        f"current z                     : "
        f"{x_current[2]:+.9f} m"
    )

    print(
        f"current vy                    : "
        f"{x_current[7]:+.9f} m/s"
    )

    print(
        f"current vz                    : "
        f"{x_current[8]:+.9f} m/s"
    )

    print(
        f"current roll                  : "
        f"{x_current[3]:+.9f} rad"
    )

    print(
        f"current wx                    : "
        f"{x_current[9]:+.9f} rad/s"
    )

    print(
        f"frozen W_r_L                  : "
        f"{r_L}"
    )

    print(
        f"frozen W_r_R                  : "
        f"{r_R}"
    )


    # ========================================================
    # TABLE 1: PREDICTED STATES
    # ========================================================

    print()

    print(
        "-" * 150
    )

    print(
        "TABLE 1 - PREDICTED STATE TRAJECTORY"
    )

    print(
        "-" * 150
    )


    print(
        " j | t_pred | C |"
        "      y_pred      y_ref         e_y |"
        "      vy_pred     vy_ref |"
        "      z_pred      z_ref         e_z |"
        "      vz_pred     vz_ref |"
        "      roll_pred      wx_pred"
    )

    print(
        "-" * 150
    )


    for j in range(
        MPC_HORIZON
    ):

        xj = X_pred[
            j
        ]

        rj = X_ref_mpc[
            j
        ]


        cL = int(
            contact_schedule[
                j,
                0
            ]
        )

        cR = int(
            contact_schedule[
                j,
                1
            ]
        )


        predicted_time = (
            t
            +
            (
                j
                +
                1
            )
            *
            MPC_DT
        )


        y_pred = float(
            xj[1]
        )

        y_ref = float(
            rj[1]
        )

        vy_pred = float(
            xj[7]
        )

        vy_ref = float(
            rj[7]
        )


        z_pred = float(
            xj[2]
        )

        z_ref = float(
            rj[2]
        )

        vz_pred = float(
            xj[8]
        )

        vz_ref = float(
            rj[8]
        )


        roll_pred = float(
            xj[3]
        )

        wx_pred = float(
            xj[9]
        )


        print(
            f"{j:2d} | "
            f"{predicted_time:6.3f} | "
            f"{cL}{cR} | "
            f"{y_pred:+.9f} "
            f"{y_ref:+.9f} "
            f"{(y_pred-y_ref):+.9f} | "
            f"{vy_pred:+.9f} "
            f"{vy_ref:+.9f} | "
            f"{z_pred:+.9f} "
            f"{z_ref:+.9f} "
            f"{(z_pred-z_ref):+.9f} | "
            f"{vz_pred:+.9f} "
            f"{vz_ref:+.9f} | "
            f"{roll_pred:+.9f} "
            f"{wx_pred:+.9f}"
        )


    # ========================================================
    # TABLE 2: OPTIMIZED WRENCH
    # ========================================================

    print()

    print(
        "-" * 165
    )

    print(
        "TABLE 2 - OPTIMIZED WRENCH OVER HORIZON"
    )

    print(
        "-" * 165
    )


    print(
        " j | C |"
        "      Fy_L       Fz_L       Mx_L |"
        "      Fy_R       Fz_R       Mx_R |"
        "    Fz_sum     Fz_sum-mg        az |"
        "    |FyL|/FzL    |FyR|/FzR |"
        "       tau_x_net"
    )

    print(
        "-" * 165
    )


    min_total_fz = np.inf
    min_total_fz_step = None

    max_total_fz = -np.inf
    max_total_fz_step = None

    max_abs_roll_torque = -np.inf
    max_abs_roll_torque_step = None


    for j in range(
        MPC_HORIZON
    ):

        uj = U[
            j
        ]


        F_L = uj[
            0:3
        ]

        M_L = uj[
            3:6
        ]

        F_R = uj[
            6:9
        ]

        M_R = uj[
            9:12
        ]


        cL = int(
            contact_schedule[
                j,
                0
            ]
        )

        cR = int(
            contact_schedule[
                j,
                1
            ]
        )


        Fz_L = float(
            F_L[2]
        )

        Fz_R = float(
            F_R[2]
        )

        Fy_L = float(
            F_L[1]
        )

        Fy_R = float(
            F_R[1]
        )

        Mx_L = float(
            M_L[0]
        )

        Mx_R = float(
            M_R[0]
        )


        Fz_sum = (
            Fz_L
            +
            Fz_R
        )


        az = (
            Fz_sum
            /
            float(
                mass_CoM
            )
            -
            9.81
        )


        if abs(
            Fz_L
        ) > 1e-8:

            friction_ratio_L = (
                abs(
                    Fy_L
                )
                /
                abs(
                    Fz_L
                )
            )

        else:

            friction_ratio_L = np.nan


        if abs(
            Fz_R
        ) > 1e-8:

            friction_ratio_R = (
                abs(
                    Fy_R
                )
                /
                abs(
                    Fz_R
                )
            )

        else:

            friction_ratio_R = np.nan


        # ----------------------------------------------------
        # Net roll torque predicted by the frozen SRBD model
        #
        # tau =
        # rL x FL + ML + rR x FR + MR
        #
        # tau_x is roll-axis torque.
        # ----------------------------------------------------

        tau_net = (
            np.cross(
                r_L,
                F_L
            )
            +
            M_L
            +
            np.cross(
                r_R,
                F_R
            )
            +
            M_R
        )


        tau_x_net = float(
            tau_net[0]
        )


        if Fz_sum < min_total_fz:

            min_total_fz = Fz_sum
            min_total_fz_step = j


        if Fz_sum > max_total_fz:

            max_total_fz = Fz_sum
            max_total_fz_step = j


        if abs(
            tau_x_net
        ) > max_abs_roll_torque:

            max_abs_roll_torque = abs(
                tau_x_net
            )

            max_abs_roll_torque_step = j


        print(
            f"{j:2d} | "
            f"{cL}{cR} | "
            f"{Fy_L:+10.6f} "
            f"{Fz_L:+10.6f} "
            f"{Mx_L:+10.6f} | "
            f"{Fy_R:+10.6f} "
            f"{Fz_R:+10.6f} "
            f"{Mx_R:+10.6f} | "
            f"{Fz_sum:+10.6f} "
            f"{(Fz_sum-robot_weight):+12.6f} "
            f"{az:+10.6f} | "
            f"{friction_ratio_L:12.6f} "
            f"{friction_ratio_R:12.6f} | "
            f"{tau_x_net:+14.8f}"
        )


    # ========================================================
    # TABLE 3: FIRST-INPUT DIAGNOSTIC
    # ========================================================

    u0 = U[
        0
    ]


    Fz0_sum = (
        float(
            u0[2]
        )
        +
        float(
            u0[8]
        )
    )


    az0 = (
        Fz0_sum
        /
        float(
            mass_CoM
        )
        -
        9.81
    )


    print()

    print(
        "-" * 100
    )

    print(
        "FIRST INPUT u0 DIAGNOSTIC"
    )

    print(
        "-" * 100
    )


    print(
        f"Fz_L(0)                       : "
        f"{u0[2]:+.9f} N"
    )

    print(
        f"Fz_R(0)                       : "
        f"{u0[8]:+.9f} N"
    )

    print(
        f"Fz_total(0)                   : "
        f"{Fz0_sum:+.9f} N"
    )

    print(
        f"mg                            : "
        f"{robot_weight:+.9f} N"
    )

    print(
        f"Fz_total(0) - mg             : "
        f"{(Fz0_sum-robot_weight):+.9f} N"
    )

    print(
        f"predicted instantaneous az   : "
        f"{az0:+.9f} m/s^2"
    )


    print()

    print(
        f"minimum horizon Fz_total     : "
        f"{min_total_fz:+.9f} N "
        f"at step {min_total_fz_step}"
    )

    print(
        f"maximum horizon Fz_total     : "
        f"{max_total_fz:+.9f} N "
        f"at step {max_total_fz_step}"
    )

    print(
        f"maximum |tau_x_net|          : "
        f"{max_abs_roll_torque:+.9f} N.m "
        f"at step {max_abs_roll_torque_step}"
    )


    # ========================================================
    # ZERO-INPUT PREDICTION
    #
    # Useful to verify gravity / augmented-state behavior.
    # ========================================================

    X_zero = (
        Phi
        @
        np.asarray(
            x_current,
            dtype=float,
        ).reshape(
            NX
        )
    ).reshape(
        MPC_HORIZON,
        NX,
    )


    print()

    print(
        "-" * 100
    )

    print(
        "ZERO-INPUT PREDICTION CHECK"
    )

    print(
        "-" * 100
    )


    print(
        f"zero-input first z            : "
        f"{X_zero[0, 2]:+.9f} m"
    )

    print(
        f"zero-input first vz           : "
        f"{X_zero[0, 8]:+.9f} m/s"
    )

    print(
        f"zero-input last z             : "
        f"{X_zero[-1, 2]:+.9f} m"
    )

    print(
        f"zero-input last vz            : "
        f"{X_zero[-1, 8]:+.9f} m/s"
    )


    print(
        "#" * 150
    )


# ============================================================
# DECODE GLOBAL CONSTRAINT ROW
# ============================================================

def decode_constraint_row(
    global_row,
    N,
):

    global_row = int(
        global_row
    )


    # --------------------------------------------------------
    # FORCE
    #
    # 5 / foot
    # 10 / step
    # --------------------------------------------------------

    force_rows_per_step = 10

    force_total_rows = (
        force_rows_per_step
        *
        N
    )


    # --------------------------------------------------------
    # MOMENT
    # --------------------------------------------------------

    if enable_toe_heel_lateral_friction:

        moment_rows_per_foot = 7

    else:

        moment_rows_per_foot = 3


    moment_rows_per_step = (
        2
        *
        moment_rows_per_foot
    )


    moment_total_rows = (
        moment_rows_per_step
        *
        N
    )


    # --------------------------------------------------------
    # CONTACT
    #
    # 6 / foot
    # 12 / step
    # --------------------------------------------------------

    contact_rows_per_step = 12

    contact_total_rows = (
        contact_rows_per_step
        *
        N
    )


    total_expected_rows = (
        force_total_rows
        +
        moment_total_rows
        +
        contact_total_rows
    )


    if (
        global_row < 0
        or
        global_row >= total_expected_rows
    ):

        return {
            "block": "unknown",
            "step": None,
            "foot": None,
            "constraint": "unknown",
            "local_row": None,
        }


    # ========================================================
    # FORCE BLOCK
    # ========================================================

    if global_row < force_total_rows:

        local_row = global_row

        step = (
            local_row
            //
            force_rows_per_step
        )

        row_in_step = (
            local_row
            %
            force_rows_per_step
        )


        if row_in_step < 5:

            foot = "L"

            row_in_foot = row_in_step

        else:

            foot = "R"

            row_in_foot = (
                row_in_step
                -
                5
            )


        labels = [
            "Fx - mu*Fz <= 0",
            "-Fx - mu*Fz <= 0",
            "Fy - mu*Fz <= 0",
            "-Fy - mu*Fz <= 0",
            "Fz_min <= Fz <= Fz_max",
        ]


        return {
            "block":
                "force",

            "step":
                int(
                    step
                ),

            "foot":
                foot,

            "constraint":
                labels[
                    row_in_foot
                ],

            "local_row":
                int(
                    local_row
                ),
        }


    # ========================================================
    # MOMENT BLOCK
    # ========================================================

    moment_start = force_total_rows

    moment_end = (
        moment_start
        +
        moment_total_rows
    )


    if global_row < moment_end:

        local_row = (
            global_row
            -
            moment_start
        )


        step = (
            local_row
            //
            moment_rows_per_step
        )


        row_in_step = (
            local_row
            %
            moment_rows_per_step
        )


        if row_in_step < moment_rows_per_foot:

            foot = "L"

            row_in_foot = row_in_step

        else:

            foot = "R"

            row_in_foot = (
                row_in_step
                -
                moment_rows_per_foot
            )


        if enable_toe_heel_lateral_friction:

            labels = [
                "-Mx_tol <= Mx <= Mx_tol",
                "Toe CoP: My - l_h*Fz <= 0",
                "Heel CoP: -My - l_t*Fz <= 0",
                "Toe lateral #1",
                "Toe lateral #2",
                "Heel lateral #1",
                "Heel lateral #2",
            ]

        else:

            labels = [
                "-Mx_tol <= Mx <= Mx_tol",
                "Toe CoP: My - l_h*Fz <= 0",
                "Heel CoP: -My - l_t*Fz <= 0",
            ]


        return {
            "block":
                "moment",

            "step":
                int(
                    step
                ),

            "foot":
                foot,

            "constraint":
                labels[
                    row_in_foot
                ],

            "local_row":
                int(
                    local_row
                ),
        }


    # ========================================================
    # CONTACT BLOCK
    # ========================================================

    contact_start = moment_end

    local_row = (
        global_row
        -
        contact_start
    )


    step = (
        local_row
        //
        contact_rows_per_step
    )


    row_in_step = (
        local_row
        %
        contact_rows_per_step
    )


    if row_in_step < 6:

        foot = "L"

        component_index = row_in_step

    else:

        foot = "R"

        component_index = (
            row_in_step
            -
            6
        )


    component_labels = [
        "Fx",
        "Fy",
        "Fz",
        "Mx",
        "My",
        "Mz",
    ]


    return {
        "block":
            "contact",

        "step":
            int(
                step
            ),

        "foot":
            foot,

        "constraint":
            (
                "inactive-wrench equality: "
                +
                component_labels[
                    component_index
                ]
                +
                " = 0"
            ),

        "local_row":
            int(
                local_row
            ),
    }


# ============================================================
# FORMAT BOUND
# ============================================================

def format_bound(
    value,
):

    value = float(
        value
    )


    if np.isneginf(
        value
    ):
        return "-inf"


    if np.isposinf(
        value
    ):
        return "+inf"


    return (
        f"{value:+.12e}"
    )


# ============================================================
# DETAILED QP CONSTRAINT DIAGNOSTICS
# ============================================================

def analyze_qp_constraint_violations(
    U_opt,
    qp_solver,
    contact_schedule,
    top_n=TOP_VIOLATIONS_TO_PRINT,
):

    print()

    print(
        "=" * 110
    )

    print(
        "DETAILED QP CONSTRAINT DIAGNOSTICS"
    )

    print(
        "=" * 110
    )


    if U_opt is None:

        print(
            "U_opt is None -> "
            "cannot evaluate A @ U."
        )

        return None


    if (
        qp_solver.A is None
        or
        qp_solver.l is None
        or
        qp_solver.u is None
    ):

        print(
            "QPSolver does not contain "
            "A/l/u from the last QP."
        )

        return None


    U_opt = np.asarray(
        U_opt,
        dtype=float,
    ).reshape(
        -1
    )


    A = np.asarray(
        qp_solver.A,
        dtype=float,
    )

    l = np.asarray(
        qp_solver.l,
        dtype=float,
    ).reshape(
        -1
    )

    u = np.asarray(
        qp_solver.u,
        dtype=float,
    ).reshape(
        -1
    )


    value = (
        A
        @
        U_opt
    )


    lower_violation = np.zeros(
        value.shape,
        dtype=float,
    )

    upper_violation = np.zeros(
        value.shape,
        dtype=float,
    )


    finite_lower = np.isfinite(
        l
    )

    finite_upper = np.isfinite(
        u
    )


    lower_violation[
        finite_lower
    ] = np.maximum(
        l[
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
        u[
            finite_upper
        ],
        0.0,
    )


    row_violation = np.maximum(
        lower_violation,
        upper_violation,
    )


    order = np.argsort(
        row_violation
    )[::-1]


    max_row = int(
        order[
            0
        ]
    )


    max_violation = float(
        row_violation[
            max_row
        ]
    )


    max_meta = (
        decode_constraint_row(
            global_row=max_row,
            N=MPC_HORIZON,
        )
    )


    print(
        f"Total constraint rows : "
        f"{A.shape[0]}"
    )

    print(
        f"Maximum violation     : "
        f"{max_violation:.12e}"
    )

    print(
        f"Acceptance tolerance  : "
        f"{QP_CONSTRAINT_TOLERANCE:.12e}"
    )


    print()

    print(
        "MOST VIOLATED CONSTRAINT"
    )

    print(
        "-" * 110
    )


    print(
        f"global row            : "
        f"{max_row}"
    )

    print(
        f"block                 : "
        f"{max_meta['block']}"
    )

    print(
        f"prediction step       : "
        f"{max_meta['step']}"
    )

    print(
        f"foot                  : "
        f"{max_meta['foot']}"
    )

    print(
        f"constraint            : "
        f"{max_meta['constraint']}"
    )

    print(
        f"A_i @ U               : "
        f"{value[max_row]:+.12e}"
    )

    print(
        f"lower bound           : "
        f"{format_bound(l[max_row])}"
    )

    print(
        f"upper bound           : "
        f"{format_bound(u[max_row])}"
    )

    print(
        f"lower violation       : "
        f"{lower_violation[max_row]:.12e}"
    )

    print(
        f"upper violation       : "
        f"{upper_violation[max_row]:.12e}"
    )


    step = max_meta[
        "step"
    ]


    if step is not None:

        U_matrix = np.asarray(
            U_opt,
            dtype=float,
        ).reshape(
            MPC_HORIZON,
            NU,
        )

        step_wrench = (
            U_matrix[
                step,
                :
            ]
        )


        print()

        print(
            f"contact_schedule[{step}] = "
            f"{contact_schedule[step]}"
        )

        print(
            f"u[{step}] ="
        )

        print(
            step_wrench
        )

        print(
            f"F_L[{step}] = "
            f"{step_wrench[0:3]}"
        )

        print(
            f"M_L[{step}] = "
            f"{step_wrench[3:6]}"
        )

        print(
            f"F_R[{step}] = "
            f"{step_wrench[6:9]}"
        )

        print(
            f"M_R[{step}] = "
            f"{step_wrench[9:12]}"
        )


    print()

    print(
        f"TOP {top_n} CONSTRAINT VIOLATIONS"
    )

    print(
        "-" * 110
    )


    number_printed = 0


    for row in order:

        row = int(
            row
        )


        violation = float(
            row_violation[
                row
            ]
        )


        if (
            violation
            <=
            VIOLATION_PRINT_THRESHOLD
        ):
            break


        meta = (
            decode_constraint_row(
                global_row=row,
                N=MPC_HORIZON,
            )
        )


        if lower_violation[
            row
        ] >= upper_violation[
            row
        ]:

            side = "LOWER"

        else:

            side = "UPPER"


        print()

        print(
            f"[{number_printed + 1}]"
        )

        print(
            f"  global row      : "
            f"{row}"
        )

        print(
            f"  block           : "
            f"{meta['block']}"
        )

        print(
            f"  step            : "
            f"{meta['step']}"
        )

        print(
            f"  foot            : "
            f"{meta['foot']}"
        )

        print(
            f"  constraint      : "
            f"{meta['constraint']}"
        )

        print(
            f"  violated side   : "
            f"{side}"
        )

        print(
            f"  A_i @ U         : "
            f"{value[row]:+.12e}"
        )

        print(
            f"  lower           : "
            f"{format_bound(l[row])}"
        )

        print(
            f"  upper           : "
            f"{format_bound(u[row])}"
        )

        print(
            f"  lower violation : "
            f"{lower_violation[row]:.12e}"
        )

        print(
            f"  upper violation : "
            f"{upper_violation[row]:.12e}"
        )

        print(
            f"  violation       : "
            f"{violation:.12e}"
        )


        number_printed += 1


        if number_printed >= top_n:
            break


    # ========================================================
    # BLOCK MAX
    # ========================================================

    print()

    print(
        "MAXIMUM VIOLATION BY CONSTRAINT BLOCK"
    )

    print(
        "-" * 110
    )


    block_max = {
        "force": 0.0,
        "moment": 0.0,
        "contact": 0.0,
    }


    block_row = {
        "force": None,
        "moment": None,
        "contact": None,
    }


    for row in range(
        row_violation.size
    ):

        meta = (
            decode_constraint_row(
                global_row=row,
                N=MPC_HORIZON,
            )
        )


        block = meta[
            "block"
        ]

        violation = float(
            row_violation[
                row
            ]
        )


        if (
            block in block_max
            and
            violation
            >
            block_max[
                block
            ]
        ):

            block_max[
                block
            ] = violation

            block_row[
                block
            ] = row


    for block in (
        "force",
        "moment",
        "contact",
    ):

        print(
            f"{block:<10}: "
            f"{block_max[block]:.12e}"
            f"  row={block_row[block]}"
        )


    return {
        "max_violation":
            max_violation,

        "max_row":
            max_row,

        "max_meta":
            max_meta,
    }


# ============================================================
# BASIC MPC SOLUTION VALIDATION
# ============================================================

def validate_mpc_solution(
    U_opt,
    contact_schedule,
):

    U = np.asarray(
        U_opt,
        dtype=float,
    ).reshape(
        MPC_HORIZON,
        NU,
    )


    contact_schedule = np.asarray(
        contact_schedule,
        dtype=int,
    ).reshape(
        MPC_HORIZON,
        2,
    )


    max_inactive_wrench = 0.0

    min_fz = np.inf

    max_fz = -np.inf

    min_friction_margin = np.inf


    for j in range(
        MPC_HORIZON
    ):

        for leg_index, offset in (
            (
                LEFT,
                0,
            ),
            (
                RIGHT,
                6,
            ),
        ):

            wrench = (
                U[
                    j,
                    offset:
                    offset + 6
                ]
            )


            force = wrench[
                0:3
            ]


            contact = int(
                contact_schedule[
                    j,
                    leg_index
                ]
            )


            if contact == 0:

                max_inactive_wrench = max(
                    max_inactive_wrench,
                    float(
                        np.max(
                            np.abs(
                                wrench
                            )
                        )
                    ),
                )

            else:

                Fx = float(
                    force[0]
                )

                Fy = float(
                    force[1]
                )

                Fz = float(
                    force[2]
                )


                min_fz = min(
                    min_fz,
                    Fz,
                )

                max_fz = max(
                    max_fz,
                    Fz,
                )


                friction_margin_x = (
                    mu
                    *
                    Fz
                    -
                    abs(
                        Fx
                    )
                )


                friction_margin_y = (
                    mu
                    *
                    Fz
                    -
                    abs(
                        Fy
                    )
                )


                min_friction_margin = min(
                    min_friction_margin,
                    friction_margin_x,
                    friction_margin_y,
                )


    return {
        "max_inactive_wrench":
            float(
                max_inactive_wrench
            ),

        "min_fz":
            float(
                min_fz
            ),

        "max_fz":
            float(
                max_fz
            ),

        "min_friction_margin":
            float(
                min_friction_margin
            ),
    }


# ============================================================
# NORMAL STATUS PRINT
# ============================================================

def print_status(
    t,
    mpc_iteration,
    phase_name,
    contact_state,
    contact_schedule,
    x_current,
    X_ref_mpc,
    u0,
    runtime,
    solver_info,
    post_diagnostics,
):

    wrench = (
        split_wrench(
            u0
        )
    )


    total_fz = float(
        wrench[
            "F_L"
        ][2]
        +
        wrench[
            "F_R"
        ][2]
    )


    print()

    print(
        "=" * 110
    )

    print(
        f"MPC ITERATION {mpc_iteration} "
        f"@ t = {t:.3f} s"
    )

    print(
        "=" * 110
    )


    print(
        f"phase_name             : "
        f"{phase_name}"
    )

    print(
        f"current contact        : "
        f"{contact_state}"
    )

    print(
        "future contact schedule:"
    )

    print(
        contact_schedule
    )


    print()

    print(
        f"CoM p                  : "
        f"{x_current[0:3]}"
    )

    print(
        f"CoM euler              : "
        f"{x_current[3:6]}"
    )

    print(
        f"CoM v                  : "
        f"{x_current[6:9]}"
    )

    print(
        f"CoM omega              : "
        f"{x_current[9:12]}"
    )


    print()

    print(
        f"Reference first p      : "
        f"{X_ref_mpc[0, 0:3]}"
    )

    print(
        f"Reference last p       : "
        f"{X_ref_mpc[-1, 0:3]}"
    )

    print(
        f"Reference first v      : "
        f"{X_ref_mpc[0, 6:9]}"
    )


    print()

    print(
        f"W_r_L                  : "
        f"{runtime['W_r_L']}"
    )

    print(
        f"W_r_R                  : "
        f"{runtime['W_r_R']}"
    )


    print()

    print(
        f"F_L [World]            : "
        f"{wrench['F_L']}"
    )

    print(
        f"M_L [World]            : "
        f"{wrench['M_L']}"
    )

    print(
        f"F_R [World]            : "
        f"{wrench['F_R']}"
    )

    print(
        f"M_R [World]            : "
        f"{wrench['M_R']}"
    )

    print(
        f"Fz total               : "
        f"{total_fz:.9f} N"
    )


    print()

    print(
        f"OSQP status            : "
        f"{solver_info.get('status')}"
    )

    print(
        f"OSQP status_val        : "
        f"{solver_info.get('status_val')}"
    )

    print(
        f"OSQP iterations        : "
        f"{solver_info.get('iterations')}"
    )

    print(
        f"OSQP primal residual   : "
        f"{solver_info.get('primal_residual')}"
    )

    print(
        f"OSQP dual residual     : "
        f"{solver_info.get('dual_residual')}"
    )


    if post_diagnostics is not None:

        print(
            f"QP solution_ok         : "
            f"{post_diagnostics.get('solution_ok')}"
        )

        print(
            f"QP max violation       : "
            f"{post_diagnostics.get('max_violation')}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "TEST: CLOSED-LOOP GAIT + REFERENCE + SRBD-MPC "
        "+ PREDICTION-HORIZON DIAGNOSTICS"
    )

    print(
        "=" * 110
    )


    print(
        f"XML                          : "
        f"{XML_PATH}"
    )

    print(
        "MuJoCo mj_step               : NO"
    )

    print(
        "Joint torque mapping         : NO"
    )

    print(
        "Swing controller             : NO"
    )

    print(
        "Reference lateral shift      : NO"
    )

    print(
        "SRBD plant                   : YES"
    )

    print(
        "MPC feedback                 : SRBD x_(k+1)"
    )


    print()

    print(
        f"CONTROL_DT                   : "
        f"{CONTROL_DT:.6f} s"
    )

    print(
        f"MPC_UPDATE_DT                : "
        f"{MPC_UPDATE_DT:.6f} s"
    )

    print(
        f"MPC_DT                       : "
        f"{MPC_DT:.6f} s"
    )

    print(
        f"MPC_HORIZON                  : "
        f"{MPC_HORIZON}"
    )

    print(
        f"Velocity target              : "
        f"{V_DES_BODY_TARGET}"
    )


    print()

    print(
        f"mu                           : "
        f"{mu}"
    )

    print(
        f"Fmax                         : "
        f"{Fmax} N"
    )

    print(
        f"Mx_tolerance                 : "
        f"{Mx_tolerance} N.m"
    )

    print(
        f"toe/heel lateral friction    : "
        f"{enable_toe_heel_lateral_friction}"
    )

    print(
        f"QP constraint tolerance      : "
        f"{QP_CONSTRAINT_TOLERANCE:.3e}"
    )

    print(
        f"Prediction log times         : "
        f"{PREDICTION_LOG_TIMES}"
    )


    # ========================================================
    # INITIALIZATION
    # ========================================================

    setup = (
        initialize_simulation(
            xml_path=XML_PATH,
            horizon=MPC_HORIZON,
            dt_mpc=MPC_DT,
            dt_mpc_update=MPC_UPDATE_DT,
            home_key_name="home",
            qp_verbose=False,
            qp_eps_abs=1e-5,
            qp_eps_rel=1e-5,
            qp_max_iter=QP_MAX_ITER,
            qp_polishing=True,
            qp_enable_warm_start=False,
            qp_constraint_tolerance=(
                QP_CONSTRAINT_TOLERANCE
            ),
            print_initialization=True,
        )
    )


    model = setup.model
    data = setup.data


    # ========================================================
    # INITIAL GEOMETRY
    # ========================================================

    initial_geometry = (
        get_current_state_mpc(
            model=model,
            data=data,
            B_inertia_CoM=(
                setup.B_inertia_CoM
            ),
        )
    )


    x_current = np.asarray(
        setup.x_initial,
        dtype=float,
    ).reshape(
        NX
    ).copy()


    # ========================================================
    # FIXED WORLD CONTACT POINTS
    # ========================================================

    W_p_L6 = (
        x_current[0:3]
        +
        initial_geometry.W_r_L
    )


    W_p_R6 = (
        x_current[0:3]
        +
        initial_geometry.W_r_R
    )


    z_des = float(
        x_current[2]
    )

    roll_des = float(
        x_current[3]
    )

    pitch_des = float(
        x_current[4]
    )

    y_ref_initial = float(
        x_current[1]
    )


    robot_weight = float(
        setup.mass_CoM
        *
        9.81
    )


    print()

    print(
        "Initial CoM:"
    )

    print(
        x_current[0:3]
    )

    print(
        "Initial L6:"
    )

    print(
        W_p_L6
    )

    print(
        "Initial R6:"
    )

    print(
        W_p_R6
    )


    print()

    print(
        f"y_ref fixed = "
        f"{y_ref_initial:.9f} m"
    )

    print(
        f"robot weight = "
        f"{robot_weight:.9f} N"
    )


    # ========================================================
    # GAIT
    # ========================================================

    gait = (
        GaitScheduler(
            mpc_horizon=MPC_HORIZON,
            dsp_durations=DSP_DURATIONS,
            ssp_durations=SSP_DURATIONS,
            control_dt=CONTROL_DT,
            dt_mpc=MPC_DT,
            startup_hold_time=STARTUP_HOLD_TIME,
            first_swing_leg=FIRST_SWING_LEG,
        )
    )


    # ========================================================
    # REFERENCE
    # ========================================================

    reference_generator = (
        ReferenceGenerator(
            horizon_length=REFERENCE_HORIZON,
            dt_mpc=MPC_DT,
            reference_update_dt=MPC_UPDATE_DT,
        )
    )


    # ========================================================
    # RUNTIME
    # ========================================================

    held_u = np.zeros(
        NU,
        dtype=float,
    )


    mpc_iteration = 0

    total_mpc_solves = 0

    hard_solver_failures = 0

    rejected_solutions = 0

    previous_phase_name = None


    number_of_steps = int(
        round(
            TEST_DURATION
            /
            CONTROL_DT
        )
    )


    # ========================================================
    # CLOSED LOOP
    # ========================================================

    for k in range(
        number_of_steps
        +
        1
    ):

        t = (
            k
            *
            CONTROL_DT
        )


        # ====================================================
        # A. GAIT
        # ====================================================

        gait_output = (
            gait.get_output()
        )


        phase_name = str(
            gait_output[
                "phase_name"
            ]
        )


        contact_state = np.asarray(
            gait_output[
                "contact_state"
            ],
            dtype=int,
        ).reshape(
            2
        )


        # ====================================================
        # B. COMMAND
        # ====================================================

        (
            v_des_body,
            yaw_rate_des,

        ) = get_reference_command(
            t
        )


        # ====================================================
        # C. MPC UPDATE
        # ====================================================

        is_mpc_tick = (
            k
            %
            MPC_UPDATE_DECIMATION
            ==
            0
        )


        if is_mpc_tick:

            # ================================================
            # C1. REFERENCE
            # ================================================

            ref_output = (
                reference_generator.update(
                    x_actual=x_current,
                    v_des_body=v_des_body,
                    yaw_rate_des=yaw_rate_des,
                    z_des=z_des,
                    roll_des=roll_des,
                    pitch_des=pitch_des,
                )
            )


            X_ref_full = np.asarray(
                ref_output[
                    "X_ref"
                ],
                dtype=float,
            )


            if X_ref_full.shape != (
                REFERENCE_HORIZON,
                NX,
            ):

                raise RuntimeError(
                    f"Wrong X_ref shape: "
                    f"{X_ref_full.shape}"
                )


            X_ref_mpc = (
                X_ref_full[
                    1:
                    MPC_HORIZON
                    +
                    1,
                    :
                ].copy()
            )


            X_ref_flat = (
                X_ref_mpc.reshape(
                    MPC_HORIZON
                    *
                    NX
                )
            )


            # ================================================
            # C2. CONTACT SCHEDULE
            # ================================================

            contact_schedule = np.asarray(
                gait_output[
                    "mpc_contact_schedule"
                ],
                dtype=int,
            )


            if contact_schedule.shape != (
                MPC_HORIZON,
                2,
            ):

                raise RuntimeError(
                    "Wrong contact schedule shape."
                )


            # ================================================
            # C3. CURRENT MODEL QUANTITIES
            # ================================================

            runtime = (
                build_runtime_srbd_quantities(
                    x_current=x_current,
                    B_inertia_CoM=(
                        setup.B_inertia_CoM
                    ),
                    W_p_L6=W_p_L6,
                    W_p_R6=W_p_R6,
                )
            )


            # ================================================
            # C4. PREDICTION MODEL
            # ================================================

            Phi, Gamma = (
                build_SRBD_model(
                    Ew_inverse=runtime[
                        "Ew_inverse"
                    ],
                    mass_CoM=(
                        setup.mass_CoM
                    ),
                    W_inertia_CoM=runtime[
                        "W_inertia_CoM"
                    ],
                    W_r_L=runtime[
                        "W_r_L"
                    ],
                    W_r_R=runtime[
                        "W_r_R"
                    ],
                    dt_mpc=MPC_DT,
                    horizon=MPC_HORIZON,
                )
            )


            # ================================================
            # C5. SOLVE
            # ================================================

            (
                U_opt,
                u0_opt,
                solver_info,
                post_diagnostics,

            ) = build_and_solve_QP(
                qp_solver=(
                    setup.qp_solver
                ),
                Phi=Phi,
                Gamma=Gamma,
                x_current=x_current,
                X_ref=X_ref_flat,
                N=MPC_HORIZON,
                L6_R_W=(
                    initial_geometry.L6_IK_R_W
                ),
                R6_R_W=(
                    initial_geometry.R6_IK_R_W
                ),
                contact_schedule=(
                    contact_schedule
                ),
                print_diagnostics=False,
            )


            total_mpc_solves += 1


            # ================================================
            # C6. NEW PREDICTION LOG
            # ================================================

            prediction_log_printed = False


            if (
                U_opt is not None
                and
                should_print_prediction_log(
                    t
                )
            ):

                print_prediction_horizon_diagnostics(
                    t=t,
                    x_current=x_current,
                    X_ref_mpc=X_ref_mpc,
                    Phi=Phi,
                    Gamma=Gamma,
                    U_opt=U_opt,
                    contact_schedule=(
                        contact_schedule
                    ),
                    runtime=runtime,
                    mass_CoM=(
                        setup.mass_CoM
                    ),
                )


                prediction_log_printed = True


            # ================================================
            # C7. HARD FAILURE
            # ================================================

            if U_opt is None:

                hard_solver_failures += 1


                print()

                print(
                    "!" * 110
                )

                print(
                    "OSQP HARD FAILURE"
                )

                print(
                    "!" * 110
                )

                print(
                    f"t                    = "
                    f"{t:.3f} s"
                )

                print(
                    f"phase                = "
                    f"{phase_name}"
                )

                print(
                    f"contact              = "
                    f"{contact_state}"
                )

                print(
                    f"solver status        = "
                    f"{solver_info.get('status')}"
                )

                print(
                    f"solver status_val    = "
                    f"{solver_info.get('status_val')}"
                )

                print(
                    f"primal residual      = "
                    f"{solver_info.get('primal_residual')}"
                )

                print(
                    f"dual residual        = "
                    f"{solver_info.get('dual_residual')}"
                )


                break


            # ================================================
            # C8. POST-SOLVE REJECTION
            # ================================================

            if u0_opt is None:

                rejected_solutions += 1


                print()

                print(
                    "!" * 110
                )

                print(
                    "MPC SOLUTION REJECTED BY POST-SOLVE CHECK"
                )

                print(
                    "!" * 110
                )


                print(
                    f"t                    = "
                    f"{t:.3f} s"
                )

                print(
                    f"phase                = "
                    f"{phase_name}"
                )

                print(
                    f"current contact      = "
                    f"{contact_state}"
                )

                print(
                    f"solver status        = "
                    f"{solver_info.get('status')}"
                )

                print(
                    f"solver status_val    = "
                    f"{solver_info.get('status_val')}"
                )

                print(
                    f"OSQP iterations      = "
                    f"{solver_info.get('iterations')}"
                )

                print(
                    f"primal residual      = "
                    f"{solver_info.get('primal_residual')}"
                )

                print(
                    f"dual residual        = "
                    f"{solver_info.get('dual_residual')}"
                )

                print(
                    f"post diagnostics     = "
                    f"{post_diagnostics}"
                )


                # --------------------------------------------
                # If failure did not happen exactly at one
                # of our selected log times, print the full
                # prediction now anyway.
                # --------------------------------------------

                if not prediction_log_printed:

                    print_prediction_horizon_diagnostics(
                        t=t,
                        x_current=x_current,
                        X_ref_mpc=X_ref_mpc,
                        Phi=Phi,
                        Gamma=Gamma,
                        U_opt=U_opt,
                        contact_schedule=(
                            contact_schedule
                        ),
                        runtime=runtime,
                        mass_CoM=(
                            setup.mass_CoM
                        ),
                    )


                # --------------------------------------------
                # Exact violated constraint
                # --------------------------------------------

                analyze_qp_constraint_violations(
                    U_opt=U_opt,
                    qp_solver=(
                        setup.qp_solver
                    ),
                    contact_schedule=(
                        contact_schedule
                    ),
                    top_n=(
                        TOP_VIOLATIONS_TO_PRINT
                    ),
                )


                break


            # ================================================
            # C9. ACCEPT u0
            # ================================================

            held_u = np.asarray(
                u0_opt,
                dtype=float,
            ).reshape(
                NU
            ).copy()


            # ================================================
            # C10. NORMAL SOLUTION CHECK
            # ================================================

            solution_check = (
                validate_mpc_solution(
                    U_opt=U_opt,
                    contact_schedule=(
                        contact_schedule
                    ),
                )
            )


            # ================================================
            # C11. NORMAL STATUS
            # ================================================

            phase_changed = (
                phase_name
                !=
                previous_phase_name
            )


            periodic_print = (
                mpc_iteration
                %
                PRINT_EVERY_N_MPC
                ==
                0
            )


            if (
                phase_changed
                or
                periodic_print
            ):

                print_status(
                    t=t,
                    mpc_iteration=(
                        mpc_iteration
                    ),
                    phase_name=(
                        phase_name
                    ),
                    contact_state=(
                        contact_state
                    ),
                    contact_schedule=(
                        contact_schedule
                    ),
                    x_current=(
                        x_current
                    ),
                    X_ref_mpc=(
                        X_ref_mpc
                    ),
                    u0=held_u,
                    runtime=runtime,
                    solver_info=(
                        solver_info
                    ),
                    post_diagnostics=(
                        post_diagnostics
                    ),
                )


                print()

                print(
                    f"max inactive wrench    : "
                    f"{solution_check['max_inactive_wrench']:.6e}"
                )

                print(
                    f"min active Fz          : "
                    f"{solution_check['min_fz']:.6f} N"
                )

                print(
                    f"max active Fz          : "
                    f"{solution_check['max_fz']:.6f} N"
                )

                print(
                    f"min friction margin    : "
                    f"{solution_check['min_friction_margin']:.6e}"
                )


            previous_phase_name = phase_name

            mpc_iteration += 1


        # ====================================================
        # D. LAST SAMPLE
        # ====================================================

        if k == number_of_steps:
            break


        # ====================================================
        # E. SRBD PLANT
        #
        # x_k
        #  ->
        # apply u_k
        #  ->
        # x_(k+1)
        # ====================================================

        x_current = (
            integrate_srbd_plant(
                x_current=x_current,
                held_u=held_u,
                mass_CoM=(
                    setup.mass_CoM
                ),
                B_inertia_CoM=(
                    setup.B_inertia_CoM
                ),
                W_p_L6=W_p_L6,
                W_p_R6=W_p_R6,
            )
        )


        # ====================================================
        # F. ADVANCE GAIT
        # ====================================================

        gait.advance()


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 110
    )


    print(
        f"Total MPC solves       : "
        f"{total_mpc_solves}"
    )

    print(
        f"OSQP hard failures     : "
        f"{hard_solver_failures}"
    )

    print(
        f"Post-check rejections  : "
        f"{rejected_solutions}"
    )


    print()

    print(
        "Final state:"
    )

    print(
        x_current
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This test does NOT change Q, R, mu, Fmax, "
        "Mx_tolerance, contact schedule or reference."
    )

    print(
        "It only adds prediction-horizon diagnostics."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()