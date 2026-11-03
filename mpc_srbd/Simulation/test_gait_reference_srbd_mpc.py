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

    directory_str = str(
        directory
    )

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

from build_SRBD_model import (
    build_SRBD_model,
)

from build_and_solve_QP import (
    build_and_solve_QP,
)

from QP_parameters import (
    mu,
    Fmax,
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
    precision=8,
    suppress=True,
    linewidth=220,
)


# ============================================================
# XML
#
# Use same model / home configuration as the existing
# SRBD standing test.
# ============================================================

XML_PATH = (
    REPO_ROOT
    /
    "xmls"
    /
    "scene_flat_terrian_v3.xml"
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
# IMPORTANT:
#
# Existing old QP_parameters.py:
#
#     dt_mpc = 0.05
#
# Current gait/reference architecture:
#
#     dt_mpc = 0.04
#
# This integration test intentionally uses 0.04 everywhere
# so that:
#
#     Reference horizon
#     Contact horizon
#     SRBD prediction horizon
#
# share the same timestamps.
# ============================================================

CONTROL_DT = 0.001

MPC_UPDATE_DT = 0.005

MPC_DT = 0.040

MPC_HORIZON = 10


# Reference Generator needs one additional point because
# SRBD prediction is:
#
#     x[k+1]
#     ...
#     x[k+N]
#
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


if MPC_UPDATE_DECIMATION <= 0:

    raise ValueError(
        "MPC_UPDATE_DECIMATION must be > 0."
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
#
# Enough to include:
#
# STARTUP_DS
# DSP2
# SSP_L   -> right swing
# DSP1
# SSP_R   -> left swing
# DSP2
# ============================================================

TEST_DURATION = 1.10


# ============================================================
# VELOCITY COMMAND
#
# Same command as previous integration test.
# ============================================================

V_DES_BODY_TARGET = np.array(
    [
        0.08,
        0.00,
        0.00,
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
# VALIDATION TOLERANCES
# ============================================================

# Inactive contact wrench should be approximately zero.
INACTIVE_WRENCH_TOL = 5e-4

# Numerical tolerance for force constraints.
FORCE_CONSTRAINT_TOL = 5e-4


# ============================================================
# PRINT
# ============================================================

PRINT_EVERY_N_MPC = 20


# ============================================================
# REFERENCE COMMAND
#
# Robot remains stationary during STARTUP_DS.
# ============================================================

def get_reference_command(
    t,
):

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
# PERFECT-TRACKING VIRTUAL STATE
#
# IMPORTANT:
#
# This test does NOT integrate SRBD dynamics.
#
# It does NOT call mj_step().
#
# x_virtual only emulates an ideal robot that follows the
# commanded body velocity exactly.
#
# Purpose:
#
# isolate:
#
#     gait
#     reference
#     contact schedule
#     SRBD prediction
#     QP constraints
#
# from whole-body dynamics.
# ============================================================

def propagate_virtual_state(
    x_current,
    v_des_body,
    yaw_rate_des,
    z_des,
    roll_des,
    pitch_des,
):

    x_current = np.asarray(
        x_current,
        dtype=float,
    ).reshape(
        NX
    )


    x_next = (
        x_current.copy()
    )


    # ========================================================
    # BODY -> WORLD VELOCITY
    # ========================================================

    R_WB = (
        euler_to_rotation_matrix(
            x_current[
                3:6
            ]
        )
    )


    v_des_world = (

        R_WB

        @

        np.asarray(
            v_des_body,
            dtype=float,
        ).reshape(
            3
        )
    )


    # ========================================================
    # POSITION
    # ========================================================

    x_next[
        0:3
    ] += (

        v_des_world

        *

        CONTROL_DT
    )


    x_next[
        2
    ] = float(
        z_des
    )


    # ========================================================
    # ORIENTATION
    # ========================================================

    x_next[
        3
    ] = float(
        roll_des
    )

    x_next[
        4
    ] = float(
        pitch_des
    )

    x_next[
        5
    ] += (

        float(
            yaw_rate_des
        )

        *

        CONTROL_DT
    )


    # ========================================================
    # LINEAR VELOCITY
    # ========================================================

    x_next[
        6:9
    ] = v_des_world


    # ========================================================
    # ANGULAR VELOCITY
    # ========================================================

    x_next[
        9:12
    ] = np.array(
        [
            0.0,
            0.0,
            float(
                yaw_rate_des
            ),
        ],
        dtype=float,
    )


    # ========================================================
    # AUGMENTED STATE
    # ========================================================

    x_next[
        12
    ] = 1.0


    return x_next


# ============================================================
# SPLIT WRENCH
#
# u =
#
# [
#   F_L,
#   M_L,
#   F_R,
#   M_R
# ]
# ============================================================

def split_wrench(
    u,
):

    u = np.asarray(
        u,
        dtype=float,
    ).reshape(
        NU
    )


    return {

        "F_L":
            u[
                0:3
            ].copy(),

        "M_L":
            u[
                3:6
            ].copy(),

        "F_R":
            u[
                6:9
            ].copy(),

        "M_R":
            u[
                9:12
            ].copy(),
    }


# ============================================================
# CONTACT-FRAME WRENCH
# ============================================================

def get_contact_frame_wrench(
    u,
    L6_R_W,
    R6_R_W,
):

    wrench = (
        split_wrench(
            u
        )
    )


    return {

        "L_F":
            L6_R_W
            @
            wrench[
                "F_L"
            ],

        "L_M":
            L6_R_W
            @
            wrench[
                "M_L"
            ],

        "R_F":
            R6_R_W
            @
            wrench[
                "F_R"
            ],

        "R_M":
            R6_R_W
            @
            wrench[
                "M_R"
            ],
    }


# ============================================================
# VALIDATE COMPLETE OPTIMAL WRENCH SEQUENCE
#
# U_opt shape:
#
#     (N * 12,)
#
# Contact schedule:
#
#     (N, 2)
#
# Checks:
#
# 1. Swing foot wrench approximately zero.
#
# 2. Active-contact force satisfies:
#
#        Fz >= 0
#        Fz <= Fmax
#
#        |Fx| <= mu Fz
#        |Fy| <= mu Fz
#
# Moment constraints are already checked by the QP's own
# post-solve diagnostics.
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

    inactive_samples = 0

    active_samples = 0


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
                    offset
                    +
                    6
                ]
            )


            force = (
                wrench[
                    0:3
                ]
            )


            contact = int(
                contact_schedule[
                    j,
                    leg_index
                ]
            )


            # =================================================
            # INACTIVE CONTACT
            # =================================================

            if contact == 0:

                inactive_samples += 1


                max_abs = float(
                    np.max(
                        np.abs(
                            wrench
                        )
                    )
                )


                max_inactive_wrench = max(
                    max_inactive_wrench,
                    max_abs,
                )


            # =================================================
            # ACTIVE CONTACT
            # =================================================

            else:

                active_samples += 1


                Fx = float(
                    force[
                        0
                    ]
                )

                Fy = float(
                    force[
                        1
                    ]
                )

                Fz = float(
                    force[
                        2
                    ]
                )


                min_fz = min(
                    min_fz,
                    Fz,
                )


                max_fz = max(
                    max_fz,
                    Fz,
                )


                margin_x = (

                    mu
                    *
                    Fz

                    -

                    abs(
                        Fx
                    )
                )


                margin_y = (

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

                    margin_x,

                    margin_y,
                )


    if active_samples == 0:

        min_fz = np.nan

        max_fz = np.nan

        min_friction_margin = np.nan


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

        "inactive_samples":
            int(
                inactive_samples
            ),

        "active_samples":
            int(
                active_samples
            ),
    }


# ============================================================
# STATUS PRINT
# ============================================================

def print_mpc_status(
    t,
    mpc_iteration,
    phase_name,
    contact_state,
    contact_schedule,
    x_current,
    X_ref_mpc,
    u0_opt,
    solver_info,
    post_diagnostics,
    solution_check,
    L6_R_W,
    R6_R_W,
    mass_CoM,
):

    local_wrench = (
        get_contact_frame_wrench(

            u=
                u0_opt,

            L6_R_W=
                L6_R_W,

            R6_R_W=
                R6_R_W,
        )
    )


    total_fz = float(

        u0_opt[
            2
        ]

        +

        u0_opt[
            8
        ]
    )


    expected_weight = float(
        mass_CoM
        *
        9.81
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
        f"phase_name               : "
        f"{phase_name}"
    )

    print(
        f"current contact_state     : "
        f"{contact_state}"
    )

    print(
        "MPC contact schedule:"
    )

    print(
        contact_schedule
    )


    print()

    print(
        f"x_current p               : "
        f"{x_current[0:3]}"
    )

    print(
        f"x_current euler           : "
        f"{x_current[3:6]}"
    )

    print(
        f"x_current v               : "
        f"{x_current[6:9]}"
    )

    print(
        f"x_ref MPC first p         : "
        f"{X_ref_mpc[0, 0:3]}"
    )

    print(
        f"x_ref MPC last p          : "
        f"{X_ref_mpc[-1, 0:3]}"
    )

    print(
        f"x_ref MPC velocity        : "
        f"{X_ref_mpc[0, 6:9]}"
    )


    print()

    print(
        "u0 MPC [World]:"
    )

    print(
        u0_opt
    )


    print()

    print(
        f"Left F [contact frame]    : "
        f"{local_wrench['L_F']}"
    )

    print(
        f"Left M [contact frame]    : "
        f"{local_wrench['L_M']}"
    )

    print(
        f"Right F [contact frame]   : "
        f"{local_wrench['R_F']}"
    )

    print(
        f"Right M [contact frame]   : "
        f"{local_wrench['R_M']}"
    )


    print()

    print(
        f"Fz_L + Fz_R              : "
        f"{total_fz:.6f} N"
    )

    print(
        f"m*g                      : "
        f"{expected_weight:.6f} N"
    )

    print(
        f"Vertical force difference : "
        f"{total_fz - expected_weight:.6f} N"
    )


    print()

    print(
        f"OSQP status               : "
        f"{solver_info.get('status')}"
    )

    print(
        f"OSQP iterations           : "
        f"{solver_info.get('iterations')}"
    )

    print(
        f"OSQP solve time           : "
        f"{solver_info.get('solve_time')} s"
    )


    if post_diagnostics is not None:

        print(
            f"QP solution_ok            : "
            f"{post_diagnostics.get('solution_ok')}"
        )

        print(
            f"QP max violation          : "
            f"{post_diagnostics.get('max_violation')}"
        )


    print()

    print(
        f"Max inactive wrench       : "
        f"{solution_check['max_inactive_wrench']:.6e}"
    )

    print(
        f"Active Fz min             : "
        f"{solution_check['min_fz']:.6f} N"
    )

    print(
        f"Active Fz max             : "
        f"{solution_check['max_fz']:.6f} N"
    )

    print(
        f"Minimum friction margin   : "
        f"{solution_check['min_friction_margin']:.6e}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "TEST: GAIT + REFERENCE GENERATOR + SRBD-MPC"
    )

    print(
        "=" * 110
    )


    print(
        f"XML                     : "
        f"{XML_PATH}"
    )

    print(
        "MuJoCo dynamics         : NO mj_step()"
    )

    print(
        "Joint torque mapping    : NO"
    )

    print(
        "Swing controller        : NO"
    )

    print(
        "SRBD-MPC QP             : YES"
    )

    print(
        "Current-state model     : VIRTUAL PERFECT TRACKING"
    )


    print()

    print(
        f"CONTROL_DT              : "
        f"{CONTROL_DT:.6f} s"
    )

    print(
        f"MPC_UPDATE_DT           : "
        f"{MPC_UPDATE_DT:.6f} s"
    )

    print(
        f"MPC update frequency    : "
        f"{1.0 / MPC_UPDATE_DT:.1f} Hz"
    )

    print(
        f"MPC_DT prediction       : "
        f"{MPC_DT:.6f} s"
    )

    print(
        f"MPC_HORIZON             : "
        f"{MPC_HORIZON}"
    )

    print(
        f"Prediction horizon      : "
        f"{MPC_HORIZON * MPC_DT:.3f} s"
    )

    print(
        f"Reference horizon       : "
        f"{REFERENCE_HORIZON} points"
    )

    print(
        f"Test duration           : "
        f"{TEST_DURATION:.3f} s"
    )

    print(
        f"Velocity target         : "
        f"{V_DES_BODY_TARGET}"
    )

    print(
        f"Startup hold            : "
        f"{STARTUP_HOLD_TIME:.3f} s"
    )

    print()


    # ========================================================
    # 1. INITIALIZE STATIC MUJOCO / SRBD RESOURCES
    #
    # initialize_simulation() performs:
    #
    #     XML load
    #     home key reset
    #     mj_forward()
    #     mass computation
    #     inertia computation
    #     QP solver construction
    #
    # No mj_step().
    # ========================================================

    setup = initialize_simulation(

        xml_path=
            XML_PATH,

        horizon=
            MPC_HORIZON,

        dt_mpc=
            MPC_DT,

        dt_mpc_update=
            MPC_UPDATE_DT,

        home_key_name=
            "home",

        qp_verbose=
            False,

        qp_eps_abs=
            1e-5,

        qp_eps_rel=
            1e-5,

        qp_max_iter=
            QP_MAX_ITER,

        qp_polishing=
            True,

        qp_enable_warm_start=
            False,

        qp_constraint_tolerance=
            QP_CONSTRAINT_TOLERANCE,

        print_initialization=
            True,
    )


    model = (
        setup.model
    )

    data = (
        setup.data
    )


    # ========================================================
    # 2. INITIAL GEOMETRY SNAPSHOT
    #
    # This standalone test intentionally freezes MuJoCo
    # geometry because no dynamics are integrated.
    #
    # Later full simulation will recompute these from feedback
    # at every MPC update.
    # ========================================================

    current_geometry = (
        get_current_state_mpc(

            model=
                model,

            data=
                data,

            B_inertia_CoM=
                setup.B_inertia_CoM,
        )
    )


    # ========================================================
    # 3. VIRTUAL CURRENT STATE
    #
    # Start from the real initial SRBD state extracted from
    # MuJoCo.
    # ========================================================

    x_virtual = np.asarray(
        setup.x_initial,
        dtype=float,
    ).reshape(
        NX
    ).copy()


    z_des = float(
        x_virtual[
            2
        ]
    )

    roll_des = float(
        x_virtual[
            3
        ]
    )

    pitch_des = float(
        x_virtual[
            4
        ]
    )


    print()

    print(
        f"Initial SRBD CoM state:"
    )

    print(
        x_virtual
    )

    print()

    print(
        f"CoM z reference         : "
        f"{z_des:.6f} m"
    )

    print(
        f"Robot mass              : "
        f"{setup.mass_CoM:.6f} kg"
    )

    print(
        f"Robot weight            : "
        f"{setup.mass_CoM * 9.81:.6f} N"
    )


    # ========================================================
    # 4. GAIT SCHEDULER
    # ========================================================

    gait = GaitScheduler(

        mpc_horizon=
            MPC_HORIZON,

        dsp_durations=
            DSP_DURATIONS,

        ssp_durations=
            SSP_DURATIONS,

        control_dt=
            CONTROL_DT,

        dt_mpc=
            MPC_DT,

        startup_hold_time=
            STARTUP_HOLD_TIME,

        first_swing_leg=
            FIRST_SWING_LEG,
    )


    # ========================================================
    # 5. REFERENCE GENERATOR
    #
    # N+1 points:
    #
    # row 0:
    #     current-time reference
    #
    # rows 1:N+1:
    #     reference for MPC predicted states
    # ========================================================

    reference_generator = (
        ReferenceGenerator(

            horizon_length=
                REFERENCE_HORIZON,

            dt_mpc=
                MPC_DT,

            reference_update_dt=
                MPC_UPDATE_DT,
        )
    )


    # ========================================================
    # 6. STATISTICS
    # ========================================================

    total_mpc_solves = 0

    solver_failures = 0

    solution_check_failures = 0


    maximum_inactive_wrench = 0.0

    minimum_active_fz = np.inf

    maximum_active_fz = -np.inf

    minimum_friction_margin = np.inf

    maximum_qp_violation = 0.0


    seen_phases = set()

    seen_current_contacts = set()


    # ========================================================
    # 7. LOOP STATE
    # ========================================================

    ref_output = None

    previous_phase_name = None

    mpc_iteration = 0


    number_of_control_steps = int(
        round(
            TEST_DURATION
            /
            CONTROL_DT
        )
    )


    # ========================================================
    # 8. MAIN LOOP
    #
    # No mj_step().
    #
    # Sequence:
    #
    # x_virtual(k)
    #       |
    #       v
    # Gait Scheduler
    #       |
    #       +--> contact_schedule
    #       |
    #       v
    # Reference Generator
    #       |
    #       +--> X_ref
    #       |
    #       v
    # frozen SRBD
    #       |
    #       v
    # QP
    #       |
    #       v
    # U_opt, u0
    # ========================================================

    for k in range(
        number_of_control_steps
        +
        1
    ):

        t = (
            k
            *
            CONTROL_DT
        )


        # ====================================================
        # A. GAIT STATE AT k
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


        seen_phases.add(
            phase_name
        )


        seen_current_contacts.add(
            tuple(
                contact_state.tolist()
            )
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
        # C. MPC / REFERENCE UPDATE
        #
        # 200 Hz
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
            # C1. REFERENCE GENERATOR
            # ================================================

            ref_output = (
                reference_generator.update(

                    x_actual=
                        x_virtual,

                    v_des_body=
                        v_des_body,

                    yaw_rate_des=
                        yaw_rate_des,

                    z_des=
                        z_des,

                    roll_des=
                        roll_des,

                    pitch_des=
                        pitch_des,
                )
            )


            X_ref_full = np.asarray(
                ref_output[
                    "X_ref"
                ],
                dtype=float,
            )


            # ================================================
            # REFERENCE DIMENSION CHECK
            # ================================================

            expected_shape = (
                REFERENCE_HORIZON,
                NX,
            )


            if X_ref_full.shape != expected_shape:

                raise RuntimeError(

                    f"Reference Generator returned "
                    f"{X_ref_full.shape}, "
                    f"expected {expected_shape}."
                )


            # ================================================
            # C2. ALIGN REFERENCE WITH SRBD PREDICTION
            #
            # SRBD X:
            #
            #     x[k+1|k]
            #     ...
            #     x[k+N|k]
            #
            # Therefore:
            #
            # discard row 0.
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
            # C3. PLANNED MPC CONTACT SCHEDULE
            #
            # Shape:
            #
            #     N x 2
            #
            # columns:
            #
            #     L, R
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

                    "mpc_contact_schedule has wrong shape: "
                    f"{contact_schedule.shape}"
                )


            # ================================================
            # C4. BUILD FROZEN SRBD MODEL
            #
            # NOTE:
            #
            # Geometry is intentionally frozen at initial
            # configuration in THIS standalone test.
            #
            # Full dynamics test will reconstruct all these
            # quantities at every MPC update.
            # ================================================

            (
                Phi,
                Gamma,

            ) = build_SRBD_model(

                Ew_inverse=
                    current_geometry.Ew_inverse,

                mass_CoM=
                    setup.mass_CoM,

                W_inertia_CoM=
                    current_geometry.W_inertia_CoM,

                W_r_L=
                    current_geometry.W_r_L,

                W_r_R=
                    current_geometry.W_r_R,

                dt_mpc=
                    MPC_DT,

                horizon=
                    MPC_HORIZON,
            )


            # ================================================
            # C5. SOLVE QP
            # ================================================

            (
                U_opt,
                u0_opt,
                solver_info,
                post_diagnostics,

            ) = build_and_solve_QP(

                qp_solver=
                    setup.qp_solver,

                Phi=
                    Phi,

                Gamma=
                    Gamma,

                x_current=
                    x_virtual,

                X_ref=
                    X_ref_flat,

                N=
                    MPC_HORIZON,

                L6_R_W=
                    current_geometry.L6_IK_R_W,

                R6_R_W=
                    current_geometry.R6_IK_R_W,

                contact_schedule=
                    contact_schedule,

                print_diagnostics=
                    False,
            )


            total_mpc_solves += 1


            # ================================================
            # C6. SOLVER FAILURE
            # ================================================

            if (
                U_opt is None
                or
                u0_opt is None
            ):

                solver_failures += 1


                print()

                print(
                    "!" * 110
                )

                print(
                    f"MPC FAILURE @ t = {t:.3f} s"
                )

                print(
                    f"phase = {phase_name}"
                )

                print(
                    f"contact state = "
                    f"{contact_state}"
                )

                print(
                    f"solver status = "
                    f"{solver_info.get('status')}"
                )

                print(
                    f"post diagnostics = "
                    f"{post_diagnostics}"
                )

                print(
                    "!" * 110
                )


                break


            # ================================================
            # C7. VALIDATE SOLUTION
            # ================================================

            solution_check = (
                validate_mpc_solution(

                    U_opt=
                        U_opt,

                    contact_schedule=
                        contact_schedule,
                )
            )


            maximum_inactive_wrench = max(

                maximum_inactive_wrench,

                solution_check[
                    "max_inactive_wrench"
                ],
            )


            if np.isfinite(
                solution_check[
                    "min_fz"
                ]
            ):

                minimum_active_fz = min(

                    minimum_active_fz,

                    solution_check[
                        "min_fz"
                    ],
                )


            if np.isfinite(
                solution_check[
                    "max_fz"
                ]
            ):

                maximum_active_fz = max(

                    maximum_active_fz,

                    solution_check[
                        "max_fz"
                    ],
                )


            if np.isfinite(
                solution_check[
                    "min_friction_margin"
                ]
            ):

                minimum_friction_margin = min(

                    minimum_friction_margin,

                    solution_check[
                        "min_friction_margin"
                    ],
                )


            if post_diagnostics is not None:

                qp_violation = float(

                    post_diagnostics.get(
                        "max_violation",
                        0.0,
                    )
                )


                maximum_qp_violation = max(
                    maximum_qp_violation,
                    qp_violation,
                )


                if not bool(
                    post_diagnostics.get(
                        "solution_ok",
                        False,
                    )
                ):

                    solution_check_failures += 1


            # ================================================
            # C8. PRINT STATUS
            #
            # Print:
            #
            # - gait transitions
            # - every N MPC iterations
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

                print_mpc_status(

                    t=
                        t,

                    mpc_iteration=
                        mpc_iteration,

                    phase_name=
                        phase_name,

                    contact_state=
                        contact_state,

                    contact_schedule=
                        contact_schedule,

                    x_current=
                        x_virtual,

                    X_ref_mpc=
                        X_ref_mpc,

                    u0_opt=
                        u0_opt,

                    solver_info=
                        solver_info,

                    post_diagnostics=
                        post_diagnostics,

                    solution_check=
                        solution_check,

                    L6_R_W=
                        current_geometry.L6_IK_R_W,

                    R6_R_W=
                        current_geometry.R6_IK_R_W,

                    mass_CoM=
                        setup.mass_CoM,
                )


            previous_phase_name = (
                phase_name
            )


            mpc_iteration += 1


        # ====================================================
        # D. END
        # ====================================================

        if k == number_of_control_steps:

            break


        # ====================================================
        # E. PROPAGATE VIRTUAL PERFECT-TRACKING STATE
        #
        # NO SRBD dynamics here.
        # NO wrench application here.
        # ====================================================

        x_virtual = (
            propagate_virtual_state(

                x_current=
                    x_virtual,

                v_des_body=
                    v_des_body,

                yaw_rate_des=
                    yaw_rate_des,

                z_des=
                    z_des,

                roll_des=
                    roll_des,

                pitch_des=
                    pitch_des,
            )
        )


        # ====================================================
        # F. ADVANCE GAIT
        # ====================================================

        gait.advance()


    # ========================================================
    # 9. FINALIZE EXTREMA
    # ========================================================

    if not np.isfinite(
        minimum_active_fz
    ):

        minimum_active_fz = np.nan


    if not np.isfinite(
        maximum_active_fz
    ):

        maximum_active_fz = np.nan


    if not np.isfinite(
        minimum_friction_margin
    ):

        minimum_friction_margin = np.nan


    # ========================================================
    # 10. PASS CONDITIONS
    # ========================================================

    required_contacts = {
        (
            1,
            1,
        ),
        (
            1,
            0,
        ),
        (
            0,
            1,
        ),
    }


    contact_coverage_ok = (
        required_contacts
        .issubset(
            seen_current_contacts
        )
    )


    solver_ok = (
        solver_failures
        ==
        0
    )


    qp_solution_ok = (
        solution_check_failures
        ==
        0
    )


    inactive_wrench_ok = (

        maximum_inactive_wrench

        <=

        INACTIVE_WRENCH_TOL
    )


    fz_lower_ok = (

        np.isfinite(
            minimum_active_fz
        )

        and

        minimum_active_fz
        >=
        -FORCE_CONSTRAINT_TOL
    )


    fz_upper_ok = (

        np.isfinite(
            maximum_active_fz
        )

        and

        maximum_active_fz
        <=
        Fmax
        +
        FORCE_CONSTRAINT_TOL
    )


    friction_ok = (

        np.isfinite(
            minimum_friction_margin
        )

        and

        minimum_friction_margin
        >=
        -FORCE_CONSTRAINT_TOL
    )


    final_pass = all(
        [
            solver_ok,
            qp_solution_ok,
            contact_coverage_ok,
            inactive_wrench_ok,
            fz_lower_ok,
            fz_upper_ok,
            friction_ok,
        ]
    )


    # ========================================================
    # 11. SUMMARY
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 110
    )


    print(
        f"Total MPC solves             : "
        f"{total_mpc_solves}"
    )

    print(
        f"Solver failures              : "
        f"{solver_failures}"
    )

    print(
        f"QP solution check failures   : "
        f"{solution_check_failures}"
    )


    print()

    print(
        f"Seen gait phases:"
    )

    for phase in sorted(
        seen_phases
    ):

        print(
            f"  {phase}"
        )


    print()

    print(
        f"Seen current contact states  : "
        f"{sorted(seen_current_contacts)}"
    )

    print(
        f"Required contact states      : "
        f"{sorted(required_contacts)}"
    )

    print(
        f"Contact coverage             : "
        f"{'PASS' if contact_coverage_ok else 'FAIL'}"
    )


    print()

    print(
        f"Maximum inactive wrench      : "
        f"{maximum_inactive_wrench:.6e}"
    )

    print(
        f"Inactive-wrench tolerance    : "
        f"{INACTIVE_WRENCH_TOL:.6e}"
    )

    print(
        f"Inactive contact constraint  : "
        f"{'PASS' if inactive_wrench_ok else 'FAIL'}"
    )


    print()

    print(
        f"Minimum active Fz            : "
        f"{minimum_active_fz:.6f} N"
    )

    print(
        f"Maximum active Fz            : "
        f"{maximum_active_fz:.6f} N"
    )

    print(
        f"Configured Fmax              : "
        f"{Fmax:.6f} N"
    )

    print(
        f"Fz bounds                    : "
        f"{'PASS' if (fz_lower_ok and fz_upper_ok) else 'FAIL'}"
    )


    print()

    print(
        f"Minimum friction margin      : "
        f"{minimum_friction_margin:.6e}"
    )

    print(
        f"Friction coefficient mu      : "
        f"{mu:.3f}"
    )

    print(
        f"Friction constraints         : "
        f"{'PASS' if friction_ok else 'FAIL'}"
    )


    print()

    print(
        f"Maximum QP violation         : "
        f"{maximum_qp_violation:.6e}"
    )


    print()

    print(
        f"Final virtual SRBD state:"
    )

    print(
        x_virtual
    )


    print()

    print(
        "=" * 110
    )


    if final_pass:

        print(
            "RESULT: PASS"
        )

    else:

        print(
            "RESULT: FAIL"
        )


    print(
        "=" * 110
    )


    print()

    print(
        "Interpretation:"
    )

    print(
        "  PASS here means:"
    )

    print(
        "    Gait Scheduler"
    )

    print(
        "      +"
    )

    print(
        "    Reference Generator"
    )

    print(
        "      +"
    )

    print(
        "    frozen SRBD prediction"
    )

    print(
        "      +"
    )

    print(
        "    contact-constrained QP"
    )

    print(
        "  are numerically compatible."
    )


    print()

    print(
        "  PASS here does NOT mean:"
    )

    print(
        "    the physical robot can already walk."
    )

    print(
        "    No wrench was applied to MuJoCo."
    )

    print(
        "    No joint torque was generated."
    )

    print(
        "    No floating-base dynamics were integrated."
    )


# ============================================================
# ENTRYtest_gait_reference_srbd_closed_loop.py