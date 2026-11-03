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
    Q,
    R,
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
#
# Return to the lighter baseline:
#
# dt_mpc = 0.04
# N      = 10
#
# prediction horizon = 0.4 s
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
#
# SSP:
#     6 * 0.04 = 0.24 s
#
# DSP:
#     5 * 0.04 = 0.20 s
#
# Startup:
#     0.25 s
#
# Timeline for first swing leg = R:
#
# 0.000 - 0.250 : STARTUP_DS
# 0.250 - 0.450 : DSP2, transfer to LEFT
# 0.450 - 0.690 : SSP_L, RIGHT swing
# 0.690 - 0.890 : DSP1, transfer to RIGHT
# 0.890 - 1.130 : SSP_R, LEFT swing
# ============================================================

SSP_DURATIONS = (
    6,
    6,
)

DSP_DURATIONS = (
    10,
    10,
)

STARTUP_HOLD_TIME = 0.250

FIRST_SWING_LEG = "R"


# ============================================================
# LATERAL LOAD TRANSFER
#
# +y -> LEFT foot
# -y -> RIGHT foot
#
# Initial diagnostic amplitude:
#
#     +/- 0.04 m
#
# Do NOT interpret this as final gait tuning.
# ============================================================

LATERAL_SHIFT_AMPLITUDE = 0.040


# ============================================================
# TEST DURATION
#
# Long enough to attempt:
#
# DSP2
# SSP_L
# DSP1
# SSP_R
# ============================================================

TEST_DURATION = 1.7


# ============================================================
# USER COMMAND
#
# No forward walking command yet.
#
# This test isolates lateral load transfer.
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
# PRINT SETTINGS
# ============================================================

PRINT_EVERY_N_MPC = 20


# ============================================================
# IMPORTANT TIMES
# ============================================================

PREDICTION_LOG_TIMES = (
    0.250,   # begin DSP2 load transfer
    0.300,
    0.350,
    0.400,
    0.445,   # immediately before SSP_L
    0.450,   # SSP_L begins
    0.500,
    0.600,
    0.685,   # end SSP_L
    0.690,   # DSP1 begins
    0.790,
    0.885,
    0.890,   # SSP_R begins
)


# ============================================================
# SMOOTHSTEP
#
# h(a) = 3a^2 - 2a^3
#
# h(0)=0, h(1)=1
#
# dh/dt = 0 at both ends.
# ============================================================

def smoothstep_transition(
    tau,
    duration,
    y0,
    y1,
):

    tau = float(tau)
    duration = float(duration)
    y0 = float(y0)
    y1 = float(y1)


    if duration <= 0.0:
        return (
            y1,
            0.0,
        )


    alpha = np.clip(
        tau
        /
        duration,
        0.0,
        1.0,
    )


    h = (
        3.0
        *
        alpha**2
        -
        2.0
        *
        alpha**3
    )


    dh_dalpha = (
        6.0
        *
        alpha
        -
        6.0
        *
        alpha**2
    )


    y = (
        y0
        +
        (
            y1
            -
            y0
        )
        *
        h
    )


    vy = (
        (
            y1
            -
            y0
        )
        *
        dh_dalpha
        /
        duration
    )


    return (
        float(y),
        float(vy),
    )


# ============================================================
# LATERAL LOAD-TRANSFER REFERENCE
#
# first_swing_leg = R:
#
# STARTUP:
#
#     y = 0
#
# first DSP2:
#
#     0 -> +A
#
# SSP_L:
#
#     +A
#
# DSP1:
#
#     +A -> -A
#
# SSP_R:
#
#     -A
#
# DSP2:
#
#     -A -> +A
#
# repeat.
# ============================================================

def lateral_reference_at_time(
    t,
):

    t = float(t)

    A = float(
        LATERAL_SHIFT_AMPLITUDE
    )


    T_ssp_L = (
        SSP_DURATIONS[0]
        *
        MPC_DT
    )

    T_ssp_R = (
        SSP_DURATIONS[1]
        *
        MPC_DT
    )

    T_dsp1 = (
        DSP_DURATIONS[0]
        *
        MPC_DT
    )

    T_dsp2 = (
        DSP_DURATIONS[1]
        *
        MPC_DT
    )


    # ========================================================
    # STARTUP HOLD
    # ========================================================

    if t < STARTUP_HOLD_TIME:

        return (
            0.0,
            0.0,
            "STARTUP_CENTER",
        )


    tau = (
        t
        -
        STARTUP_HOLD_TIME
    )


    # ========================================================
    # RIGHT FOOT SWINGS FIRST
    # ========================================================

    if FIRST_SWING_LEG == "R":

        # ----------------------------------------------------
        # Initial DSP2
        #
        # center -> LEFT
        # ----------------------------------------------------

        if tau < T_dsp2:

            y, vy = smoothstep_transition(
                tau=tau,
                duration=T_dsp2,
                y0=0.0,
                y1=+A,
            )

            return (
                y,
                vy,
                "DSP2_INITIAL_0_TO_LEFT",
            )


        tau -= T_dsp2


        cycle_duration = (
            T_ssp_L
            +
            T_dsp1
            +
            T_ssp_R
            +
            T_dsp2
        )


        tau = (
            tau
            %
            cycle_duration
        )


        # ----------------------------------------------------
        # SSP_L
        # ----------------------------------------------------

        if tau < T_ssp_L:

            return (
                +A,
                0.0,
                "SSP_L_HOLD_LEFT",
            )


        tau -= T_ssp_L


        # ----------------------------------------------------
        # DSP1
        #
        # LEFT -> RIGHT
        # ----------------------------------------------------

        if tau < T_dsp1:

            y, vy = smoothstep_transition(
                tau=tau,
                duration=T_dsp1,
                y0=+A,
                y1=-A,
            )

            return (
                y,
                vy,
                "DSP1_LEFT_TO_RIGHT",
            )


        tau -= T_dsp1


        # ----------------------------------------------------
        # SSP_R
        # ----------------------------------------------------

        if tau < T_ssp_R:

            return (
                -A,
                0.0,
                "SSP_R_HOLD_RIGHT",
            )


        tau -= T_ssp_R


        # ----------------------------------------------------
        # DSP2
        #
        # RIGHT -> LEFT
        # ----------------------------------------------------

        y, vy = smoothstep_transition(
            tau=tau,
            duration=T_dsp2,
            y0=-A,
            y1=+A,
        )

        return (
            y,
            vy,
            "DSP2_RIGHT_TO_LEFT",
        )


    # ========================================================
    # LEFT FOOT SWINGS FIRST
    #
    # Included for completeness.
    # ========================================================

    if tau < T_dsp1:

        y, vy = smoothstep_transition(
            tau=tau,
            duration=T_dsp1,
            y0=0.0,
            y1=-A,
        )

        return (
            y,
            vy,
            "DSP1_INITIAL_0_TO_RIGHT",
        )


    tau -= T_dsp1


    cycle_duration = (
        T_ssp_R
        +
        T_dsp2
        +
        T_ssp_L
        +
        T_dsp1
    )


    tau = (
        tau
        %
        cycle_duration
    )


    if tau < T_ssp_R:

        return (
            -A,
            0.0,
            "SSP_R_HOLD_RIGHT",
        )


    tau -= T_ssp_R


    if tau < T_dsp2:

        y, vy = smoothstep_transition(
            tau=tau,
            duration=T_dsp2,
            y0=-A,
            y1=+A,
        )

        return (
            y,
            vy,
            "DSP2_RIGHT_TO_LEFT",
        )


    tau -= T_dsp2


    if tau < T_ssp_L:

        return (
            +A,
            0.0,
            "SSP_L_HOLD_LEFT",
        )


    tau -= T_ssp_L


    y, vy = smoothstep_transition(
        tau=tau,
        duration=T_dsp1,
        y0=+A,
        y1=-A,
    )


    return (
        y,
        vy,
        "DSP1_LEFT_TO_RIGHT",
    )


# ============================================================
# BUILD LOAD-TRANSFER PROFILE
#
# Full reference horizon includes:
#
# k
# k+1
# ...
# k+N
#
# Therefore length = N+1.
# ============================================================

def build_lateral_reference_profile(
    t_now,
):

    times = np.zeros(
        REFERENCE_HORIZON,
        dtype=float,
    )

    y_offset = np.zeros(
        REFERENCE_HORIZON,
        dtype=float,
    )

    vy_offset = np.zeros(
        REFERENCE_HORIZON,
        dtype=float,
    )

    labels = []


    for i in range(
        REFERENCE_HORIZON
    ):

        t_i = (
            float(t_now)
            +
            float(i)
            *
            MPC_DT
        )


        y_i, vy_i, label_i = (
            lateral_reference_at_time(
                t_i
            )
        )


        times[i] = t_i

        y_offset[i] = y_i

        vy_offset[i] = vy_i

        labels.append(
            label_i
        )


    return {
        "time":
            times,

        "y_offset":
            y_offset,

        "vy_offset":
            vy_offset,

        "labels":
            labels,
    }


# ============================================================
# USER COMMAND
# ============================================================

def get_reference_command(
    t,
):

    return (
        V_DES_BODY_TARGET.copy(),
        float(
            YAW_RATE_DES_TARGET
        ),
    )


# ============================================================
# TIME CHECK
# ============================================================

def should_print_prediction_log(
    t,
):

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


    W_p_CoM = (
        x_current[
            0:3
        ]
    )

    W_euler = (
        x_current[
            3:6
        ]
    )


    Ew_inverse = (
        get_Ew_inverse(
            W_euler
        )
    )


    W_R_B = (
        euler_to_rotation_matrix(
            W_euler
        )
    )


    W_inertia_CoM = (
        W_R_B
        @
        B_inertia_CoM
        @
        W_R_B.T
    )


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
    }


# ============================================================
# CLOSED-LOOP SRBD PLANT
#
# Correct simulation order:
#
# x_k
#   ->
# compute u_k
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
            B_inertia_CoM=(
                B_inertia_CoM
            ),
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
# PREDICTED STATE TRAJECTORY
# ============================================================

def build_predicted_state_trajectory(
    Phi,
    Gamma,
    x_current,
    U_opt,
):

    X_flat = (
        np.asarray(
            Phi,
            dtype=float,
        )
        @
        np.asarray(
            x_current,
            dtype=float,
        ).reshape(
            NX
        )
        +
        np.asarray(
            Gamma,
            dtype=float,
        )
        @
        np.asarray(
            U_opt,
            dtype=float,
        ).reshape(
            MPC_HORIZON
            *
            NU
        )
    )


    return (
        X_flat.reshape(
            MPC_HORIZON,
            NX,
        )
    )


# ============================================================
# FIRST INPUT METRICS
# ============================================================

def compute_u0_metrics(
    u0,
    mass_CoM,
):

    u0 = np.asarray(
        u0,
        dtype=float,
    ).reshape(
        NU
    )


    Fy_L = float(
        u0[1]
    )

    Fz_L = float(
        u0[2]
    )

    Mx_L = float(
        u0[3]
    )


    Fy_R = float(
        u0[7]
    )

    Fz_R = float(
        u0[8]
    )

    Mx_R = float(
        u0[9]
    )


    total_fz = (
        Fz_L
        +
        Fz_R
    )


    robot_weight = (
        float(
            mass_CoM
        )
        *
        9.81
    )


    friction_ratio_L = np.nan

    if Fz_L > 1e-6:

        friction_ratio_L = (
            abs(
                Fy_L
            )
            /
            Fz_L
        )


    friction_ratio_R = np.nan

    if Fz_R > 1e-6:

        friction_ratio_R = (
            abs(
                Fy_R
            )
            /
            Fz_R
        )


    return {
        "Fy_L":
            Fy_L,

        "Fz_L":
            Fz_L,

        "Mx_L":
            Mx_L,

        "Fy_R":
            Fy_R,

        "Fz_R":
            Fz_R,

        "Mx_R":
            Mx_R,

        "total_fz":
            total_fz,

        "robot_weight":
            robot_weight,

        "friction_ratio_L":
            friction_ratio_L,

        "friction_ratio_R":
            friction_ratio_R,
    }


# ============================================================
# PREDICTION LOG
# ============================================================

def print_prediction_diagnostics(
    t,
    phase_name,
    x_current,
    X_ref_mpc,
    Phi,
    Gamma,
    U_opt,
    contact_schedule,
    lateral_profile,
    mass_CoM,
):

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


    robot_weight = (
        float(
            mass_CoM
        )
        *
        9.81
    )


    print()

    print(
        "#" * 155
    )

    print(
        "LOAD-TRANSFER PREDICTION DIAGNOSTICS"
    )

    print(
        "#" * 155
    )


    print(
        f"actual time                  : "
        f"{t:.3f} s"
    )

    print(
        f"current gait phase           : "
        f"{phase_name}"
    )

    print(
        f"actual y                     : "
        f"{x_current[1]:+.9f} m"
    )

    print(
        f"actual vy                    : "
        f"{x_current[7]:+.9f} m/s"
    )

    print(
        f"actual z                     : "
        f"{x_current[2]:+.9f} m"
    )

    print(
        f"actual vz                    : "
        f"{x_current[8]:+.9f} m/s"
    )

    print(
        f"shift amplitude              : "
        f"{LATERAL_SHIFT_AMPLITUDE:+.6f} m"
    )

    print(
        f"robot weight                 : "
        f"{robot_weight:.9f} N"
    )


    print()

    print(
        "FULL LATERAL REFERENCE INCLUDING CURRENT KNOT:"
    )

    print(
        "time:"
    )

    print(
        lateral_profile[
            "time"
        ]
    )

    print(
        "y_offset:"
    )

    print(
        lateral_profile[
            "y_offset"
        ]
    )

    print(
        "vy_offset:"
    )

    print(
        lateral_profile[
            "vy_offset"
        ]
    )


    print()

    print(
        "-" * 155
    )

    print(
        "PREDICTION HORIZON"
    )

    print(
        "-" * 155
    )


    print(
        " j | C |"
        "      y_ref      y_pred       e_y |"
        "     vy_ref     vy_pred |"
        "      z_ref      z_pred       e_z |"
        "      Fz_L      Fz_R     Fz_sum |"
        "    FyL/FzL    FyR/FzR |"
        "      Mx_L      Mx_R"
    )


    for j in range(
        MPC_HORIZON
    ):

        x_pred = X_pred[
            j
        ]

        x_ref = X_ref_mpc[
            j
        ]

        u = U[
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


        Fy_L = float(
            u[1]
        )

        Fz_L = float(
            u[2]
        )

        Mx_L = float(
            u[3]
        )


        Fy_R = float(
            u[7]
        )

        Fz_R = float(
            u[8]
        )

        Mx_R = float(
            u[9]
        )


        ratio_L = np.nan

        if Fz_L > 1e-6:

            ratio_L = (
                abs(
                    Fy_L
                )
                /
                Fz_L
            )


        ratio_R = np.nan

        if Fz_R > 1e-6:

            ratio_R = (
                abs(
                    Fy_R
                )
                /
                Fz_R
            )


        print(
            f"{j:2d} | "
            f"{cL}{cR} | "
            f"{x_ref[1]:+10.6f} "
            f"{x_pred[1]:+10.6f} "
            f"{(x_pred[1]-x_ref[1]):+10.6f} | "
            f"{x_ref[7]:+10.6f} "
            f"{x_pred[7]:+10.6f} | "
            f"{x_ref[2]:+10.6f} "
            f"{x_pred[2]:+10.6f} "
            f"{(x_pred[2]-x_ref[2]):+10.6f} | "
            f"{Fz_L:+9.4f} "
            f"{Fz_R:+9.4f} "
            f"{(Fz_L+Fz_R):+9.4f} | "
            f"{ratio_L:11.6f} "
            f"{ratio_R:11.6f} | "
            f"{Mx_L:+10.6f} "
            f"{Mx_R:+10.6f}"
        )


    print(
        "#" * 155
    )


# ============================================================
# QP FAILURE SUMMARY
# ============================================================

def print_qp_failure(
    t,
    phase_name,
    solver_info,
    post_diagnostics,
    qp_solver,
    U_opt,
):

    print()

    print(
        "!" * 120
    )

    print(
        "QP SOLUTION FAILURE / REJECTION"
    )

    print(
        "!" * 120
    )


    print(
        f"t                    : "
        f"{t:.3f} s"
    )

    print(
        f"phase                : "
        f"{phase_name}"
    )

    print(
        f"OSQP status          : "
        f"{solver_info.get('status')}"
    )

    print(
        f"status_val           : "
        f"{solver_info.get('status_val')}"
    )

    print(
        f"iterations           : "
        f"{solver_info.get('iterations')}"
    )

    print(
        f"primal residual      : "
        f"{solver_info.get('primal_residual')}"
    )

    print(
        f"dual residual        : "
        f"{solver_info.get('dual_residual')}"
    )

    print(
        f"post diagnostics     : "
        f"{post_diagnostics}"
    )


    if U_opt is None:

        return


    if (
        qp_solver.A is None
        or
        qp_solver.l is None
        or
        qp_solver.u is None
    ):

        return


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

    lower = np.asarray(
        qp_solver.l,
        dtype=float,
    ).reshape(
        -1
    )

    upper = np.asarray(
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


    lower_violation = np.zeros_like(
        value
    )

    upper_violation = np.zeros_like(
        value
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


    violation = np.maximum(
        lower_violation,
        upper_violation,
    )


    worst_rows = np.argsort(
        violation
    )[::-1][
        :10
    ]


    print()

    print(
        "TOP RAW CONSTRAINT VIOLATIONS"
    )

    print(
        "-" * 120
    )


    for rank, row in enumerate(
        worst_rows,
        start=1,
    ):

        if violation[
            row
        ] <= 0.0:

            break


        print(
            f"{rank:2d} | "
            f"row={int(row):4d} | "
            f"A@U={value[row]:+.9e} | "
            f"lower={lower[row]:+.9e} | "
            f"upper={upper[row]:+.9e} | "
            f"violation={violation[row]:.9e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "TEST: SRBD-MPC WITH GAIT-AWARE LATERAL LOAD TRANSFER"
    )

    print(
        "=" * 120
    )


    print(
        f"XML                         : "
        f"{XML_PATH}"
    )

    print(
        "MuJoCo mj_step              : NO"
    )

    print(
        "Joint torque mapping        : NO"
    )

    print(
        "Swing controller            : NO"
    )

    print(
        "SRBD closed-loop plant      : YES"
    )

    print(
        "Lateral load transfer       : YES"
    )

    print(
        "MPC core modified           : NO"
    )

    print(
        "ReferenceGenerator modified : NO"
    )


    print()

    print(
        f"CONTROL_DT                  : "
        f"{CONTROL_DT:.6f} s"
    )

    print(
        f"MPC_UPDATE_DT               : "
        f"{MPC_UPDATE_DT:.6f} s"
    )

    print(
        f"MPC_DT                      : "
        f"{MPC_DT:.6f} s"
    )

    print(
        f"MPC_HORIZON                 : "
        f"{MPC_HORIZON}"
    )

    print(
        f"prediction horizon          : "
        f"{MPC_DT*MPC_HORIZON:.3f} s"
    )


    print()

    print(
        f"SSP duration                : "
        f"{SSP_DURATIONS[0]*MPC_DT:.3f} s"
    )

    print(
        f"DSP duration                : "
        f"{DSP_DURATIONS[0]*MPC_DT:.3f} s"
    )

    print(
        f"startup hold                : "
        f"{STARTUP_HOLD_TIME:.3f} s"
    )

    print(
        f"lateral amplitude           : "
        f"{LATERAL_SHIFT_AMPLITUDE:.3f} m"
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
        f"toe/heel lateral friction   : "
        f"{enable_toe_heel_lateral_friction}"
    )


    Q_diag = np.diag(
        np.asarray(
            Q,
            dtype=float,
        )
    )


    R_diag = np.diag(
        np.asarray(
            R,
            dtype=float,
        )
    )


    print()

    print(
        "Q diagonal:"
    )

    print(
        Q_diag
    )

    print(
        f"Q_y                         : "
        f"{Q_diag[1]}"
    )

    print(
        f"Q_vy                        : "
        f"{Q_diag[7]}"
    )

    print(
        f"Q_z                         : "
        f"{Q_diag[2]}"
    )

    print(
        f"Q_vz                        : "
        f"{Q_diag[8]}"
    )

    print(
        "R diagonal:"
    )

    print(
        R_diag
    )


    # ========================================================
    # INITIALIZATION
    # ========================================================

    setup = (
        initialize_simulation(
            xml_path=XML_PATH,
            horizon=MPC_HORIZON,
            dt_mpc=MPC_DT,
            dt_mpc_update=(
                MPC_UPDATE_DT
            ),
            home_key_name="home",
            qp_verbose=False,
            qp_eps_abs=1e-6,
            qp_eps_rel=1e-6,
            qp_max_iter=20000,
            qp_polishing=True,
            qp_enable_warm_start=False,
            qp_constraint_tolerance=(
                QP_CONSTRAINT_TOLERANCE
            ),
            print_initialization=True,
        )
    )


    initial_geometry = (
        get_current_state_mpc(
            model=setup.model,
            data=setup.data,
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
    # FIX CONTACT POINTS IN WORLD
    #
    # Standalone SRBD diagnostic only.
    # ========================================================

    W_p_L6 = (
        x_current[
            0:3
        ]
        +
        initial_geometry.W_r_L
    )


    W_p_R6 = (
        x_current[
            0:3
        ]
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


    y_center = float(
        x_current[1]
    )


    print()

    print(
        "Initial CoM:"
    )

    print(
        x_current[
            0:3
        ]
    )

    print(
        "L6:"
    )

    print(
        W_p_L6
    )

    print(
        "R6:"
    )

    print(
        W_p_R6
    )

    print(
        f"center y                    : "
        f"{y_center:+.9f} m"
    )

    print(
        f"target LEFT load y          : "
        f"{y_center + LATERAL_SHIFT_AMPLITUDE:+.9f} m"
    )

    print(
        f"target RIGHT load y         : "
        f"{y_center - LATERAL_SHIFT_AMPLITUDE:+.9f} m"
    )


    # ========================================================
    # GAIT
    # ========================================================

    gait = (
        GaitScheduler(
            mpc_horizon=MPC_HORIZON,
            dsp_durations=(
                DSP_DURATIONS
            ),
            ssp_durations=(
                SSP_DURATIONS
            ),
            control_dt=CONTROL_DT,
            dt_mpc=MPC_DT,
            startup_hold_time=(
                STARTUP_HOLD_TIME
            ),
            first_swing_leg=(
                FIRST_SWING_LEG
            ),
        )
    )


    # ========================================================
    # BASE REFERENCE GENERATOR
    # ========================================================

    reference_generator = (
        ReferenceGenerator(
            horizon_length=(
                REFERENCE_HORIZON
            ),
            dt_mpc=MPC_DT,
            reference_update_dt=(
                MPC_UPDATE_DT
            ),
        )
    )


    # ========================================================
    # RUNTIME
    # ========================================================

    held_u = np.zeros(
        NU,
        dtype=float,
    )


    previous_phase_name = None

    total_mpc_solves = 0

    hard_solver_failures = 0

    rejected_solutions = 0


    # --------------------------------------------------------
    # Performance summary
    # --------------------------------------------------------

    min_total_fz = np.inf
    max_total_fz = -np.inf

    max_abs_mx = 0.0

    max_friction_ratio = 0.0

    max_abs_y_error = 0.0


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
        # A. GAIT OUTPUT AT x_k
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
            # C1. BASE REFERENCE
            # ================================================

            ref_output = (
                reference_generator.update(
                    x_actual=x_current,
                    v_des_body=(
                        v_des_body
                    ),
                    yaw_rate_des=(
                        yaw_rate_des
                    ),
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
            ).copy()


            if X_ref_full.shape != (
                REFERENCE_HORIZON,
                NX,
            ):

                raise RuntimeError(
                    f"Wrong X_ref shape: "
                    f"{X_ref_full.shape}"
                )


            # ================================================
            # C2. LATERAL LOAD-TRANSFER PROFILE
            # ================================================

            lateral_profile = (
                build_lateral_reference_profile(
                    t_now=t
                )
            )


            # -----------------------------------------------
            # Base persistent center + gait offset.
            #
            # y index  = 1
            # vy index = 7
            # -----------------------------------------------

            X_ref_full[
                :,
                1
            ] += (
                lateral_profile[
                    "y_offset"
                ]
            )


            X_ref_full[
                :,
                7
            ] += (
                lateral_profile[
                    "vy_offset"
                ]
            )


            # ================================================
            # C3. MPC REFERENCE
            #
            # Prediction:
            #
            # x[k+1|k] ... x[k+N|k]
            #
            # Therefore skip current knot.
            # ================================================

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
            # C4. CONTACT SCHEDULE
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
            # C5. CURRENT SRBD MODEL
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
                    horizon=(
                        MPC_HORIZON
                    ),
                )
            )


            # ================================================
            # C6. SOLVE QP
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
                x_current=(
                    x_current
                ),
                X_ref=(
                    X_ref_flat
                ),
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
            # C7. HARD SOLVER FAILURE
            # ================================================

            if U_opt is None:

                hard_solver_failures += 1


                print_qp_failure(
                    t=t,
                    phase_name=phase_name,
                    solver_info=(
                        solver_info
                    ),
                    post_diagnostics=(
                        post_diagnostics
                    ),
                    qp_solver=(
                        setup.qp_solver
                    ),
                    U_opt=None,
                )


                break


            # ================================================
            # C8. SELECTED PREDICTION LOG
            # ================================================

            if should_print_prediction_log(
                t
            ):

                print_prediction_diagnostics(
                    t=t,
                    phase_name=(
                        phase_name
                    ),
                    x_current=(
                        x_current
                    ),
                    X_ref_mpc=(
                        X_ref_mpc
                    ),
                    Phi=Phi,
                    Gamma=Gamma,
                    U_opt=U_opt,
                    contact_schedule=(
                        contact_schedule
                    ),
                    lateral_profile=(
                        lateral_profile
                    ),
                    mass_CoM=(
                        setup.mass_CoM
                    ),
                )


            # ================================================
            # C9. POST-SOLVE REJECTION
            # ================================================

            if u0_opt is None:

                rejected_solutions += 1


                # Print full prediction at failure.
                print_prediction_diagnostics(
                    t=t,
                    phase_name=(
                        phase_name
                    ),
                    x_current=(
                        x_current
                    ),
                    X_ref_mpc=(
                        X_ref_mpc
                    ),
                    Phi=Phi,
                    Gamma=Gamma,
                    U_opt=U_opt,
                    contact_schedule=(
                        contact_schedule
                    ),
                    lateral_profile=(
                        lateral_profile
                    ),
                    mass_CoM=(
                        setup.mass_CoM
                    ),
                )


                print_qp_failure(
                    t=t,
                    phase_name=(
                        phase_name
                    ),
                    solver_info=(
                        solver_info
                    ),
                    post_diagnostics=(
                        post_diagnostics
                    ),
                    qp_solver=(
                        setup.qp_solver
                    ),
                    U_opt=U_opt,
                )


                break


            # ================================================
            # C10. ACCEPT u0
            # ================================================

            held_u = np.asarray(
                u0_opt,
                dtype=float,
            ).reshape(
                NU
            ).copy()


            # ================================================
            # C11. METRICS
            # ================================================

            metrics = (
                compute_u0_metrics(
                    u0=held_u,
                    mass_CoM=(
                        setup.mass_CoM
                    ),
                )
            )


            min_total_fz = min(
                min_total_fz,
                metrics[
                    "total_fz"
                ],
            )


            max_total_fz = max(
                max_total_fz,
                metrics[
                    "total_fz"
                ],
            )


            max_abs_mx = max(
                max_abs_mx,
                abs(
                    metrics[
                        "Mx_L"
                    ]
                ),
                abs(
                    metrics[
                        "Mx_R"
                    ]
                ),
            )


            for ratio in (
                metrics[
                    "friction_ratio_L"
                ],
                metrics[
                    "friction_ratio_R"
                ],
            ):

                if np.isfinite(
                    ratio
                ):

                    max_friction_ratio = max(
                        max_friction_ratio,
                        float(
                            ratio
                        ),
                    )


            # Current reference knot approximately at k.
            y_ref_current = float(
                X_ref_full[
                    0,
                    1
                ]
            )


            y_error = (
                float(
                    x_current[
                        1
                    ]
                )
                -
                y_ref_current
            )


            max_abs_y_error = max(
                max_abs_y_error,
                abs(
                    y_error
                ),
            )


            # ================================================
            # C12. PHASE / PERIODIC STATUS
            # ================================================

            phase_changed = (
                phase_name
                !=
                previous_phase_name
            )


            periodic_print = (
                total_mpc_solves
                %
                PRINT_EVERY_N_MPC
                ==
                1
            )


            if (
                phase_changed
                or
                periodic_print
            ):

                y_offset_now, vy_offset_now, load_label = (
                    lateral_reference_at_time(
                        t
                    )
                )


                print()

                print(
                    "=" * 120
                )

                print(
                    f"STATUS @ t={t:.3f} s"
                )

                print(
                    "=" * 120
                )


                print(
                    f"gait phase            : "
                    f"{phase_name}"
                )

                print(
                    f"contact               : "
                    f"{contact_state}"
                )

                print(
                    f"load-transfer phase   : "
                    f"{load_label}"
                )


                print()

                print(
                    f"actual y              : "
                    f"{x_current[1]:+.9f} m"
                )

                print(
                    f"current y offset ref  : "
                    f"{y_offset_now:+.9f} m"
                )

                print(
                    f"final y ref           : "
                    f"{X_ref_full[0,1]:+.9f} m"
                )

                print(
                    f"actual vy             : "
                    f"{x_current[7]:+.9f} m/s"
                )

                print(
                    f"vy offset ref         : "
                    f"{vy_offset_now:+.9f} m/s"
                )


                print()

                print(
                    f"Fz L                  : "
                    f"{metrics['Fz_L']:+.6f} N"
                )

                print(
                    f"Fz R                  : "
                    f"{metrics['Fz_R']:+.6f} N"
                )

                print(
                    f"Fz total              : "
                    f"{metrics['total_fz']:+.6f} N"
                )

                print(
                    f"mg                    : "
                    f"{metrics['robot_weight']:+.6f} N"
                )


                print()

                print(
                    f"|FyL|/FzL             : "
                    f"{metrics['friction_ratio_L']}"
                )

                print(
                    f"|FyR|/FzR             : "
                    f"{metrics['friction_ratio_R']}"
                )

                print(
                    f"Mx L                  : "
                    f"{metrics['Mx_L']:+.9f} N.m"
                )

                print(
                    f"Mx R                  : "
                    f"{metrics['Mx_R']:+.9f} N.m"
                )


                print()

                print(
                    f"OSQP status           : "
                    f"{solver_info.get('status')}"
                )

                print(
                    f"iterations            : "
                    f"{solver_info.get('iterations')}"
                )

                print(
                    f"max violation         : "
                    f"{post_diagnostics.get('max_violation')}"
                )


            previous_phase_name = (
                phase_name
            )


        # ====================================================
        # D. END SAMPLE
        # ====================================================

        if k == number_of_steps:
            break


        # ====================================================
        # E. APPLY u_k TO SRBD PLANT
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
        # F. ADVANCE GAIT AFTER STATE UPDATE
        # ====================================================

        gait.advance()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 120
    )


    print(
        f"Total MPC solves        : "
        f"{total_mpc_solves}"
    )

    print(
        f"OSQP hard failures      : "
        f"{hard_solver_failures}"
    )

    print(
        f"Post-check rejections   : "
        f"{rejected_solutions}"
    )


    print()

    print(
        f"minimum applied Fz sum  : "
        f"{min_total_fz:.9f} N"
    )

    print(
        f"maximum applied Fz sum  : "
        f"{max_total_fz:.9f} N"
    )

    print(
        f"maximum applied |Mx|    : "
        f"{max_abs_mx:.9f} N.m"
    )

    print(
        f"maximum |Fy|/Fz         : "
        f"{max_friction_ratio:.9f}"
    )

    print(
        f"maximum |y-y_ref|       : "
        f"{max_abs_y_error:.9f} m"
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
        "TEST PURPOSE:"
    )

    print(
        "Check whether an explicit smooth lateral load-transfer "
        "reference allows the existing SRBD-MPC to pass "
        "DS -> SSP transitions without changing MPC core "
        "or contact constraints."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()