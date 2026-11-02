from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

SIMULATION_DIR = (
    Path(__file__).resolve().parent
)

MPC_SRBD_DIR = (
    SIMULATION_DIR.parent
)

REPO_ROOT = (
    MPC_SRBD_DIR.parent
)


STATE_DIR = (
    MPC_SRBD_DIR
    /
    "State"
)

SRBD_DIR = (
    MPC_SRBD_DIR
    /
    "SRBD"
)

QP_DIR = (
    MPC_SRBD_DIR
    /
    "QP"
)

KINEMATICS_DIR = (
    MPC_SRBD_DIR
    /
    "Kinematics"
)

ACTUATOR_DIR = (
    MPC_SRBD_DIR
    /
    "Actuator"
)

COMMAND_DIR = (
    MPC_SRBD_DIR
    /
    "Command"
)


for directory in (
    MPC_SRBD_DIR,
    SIMULATION_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
    COMMAND_DIR,
):

    directory_str = str(
        directory
    )

    if directory_str not in sys.path:

        sys.path.insert(
            0,
            directory_str
        )


# ============================================================
# SIMULATION
# ============================================================

from simulation_setup import (
    initialize_simulation,
)

from simulation_logger import (
    SimulationLogger,
)


# ============================================================
# STATE
# ============================================================

from get_current_state_mpc import (
    get_current_state_mpc,
)

from get_contact_state import (
    get_contact_state,
)


# ============================================================
# COMMAND / REFERENCE
# ============================================================

from Command.srbd_reference import (
    build_srbd_reference,
)


# ============================================================
# SRBD
# ============================================================

from build_SRBD_model import (
    build_SRBD_model,
)


# ============================================================
# QP
# ============================================================

from build_and_solve_QP import (
    build_and_solve_QP,
)

from QP_diagnostics import (
    print_QP_diagnostics,
)


# ============================================================
# KINEMATICS
# ============================================================

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench,
)


# ============================================================
# ACTUATOR
# ============================================================

from actuator_mapping import (
    apply_joint_torque,
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
# ============================================================

XML_PATH = (
    REPO_ROOT
    /
    "xmls"
    /
    "scene_flat_terrian_v3.xml"
)


# ============================================================
# STANDING MPC PARAMETERS
#
# Keep these consistent with the current walking architecture:
#
#     N = 10
#
#     prediction spacing:
#         dt_mpc = 0.05 s
#
#     actual MPC update:
#         dt_mpc_update = 0.01 s
#
# IMPORTANT:
#
#     dt_mpc != dt_mpc_update
# ============================================================

N = 10

DT_MPC = 0.05

DT_MPC_UPDATE = 0.01


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

SIM_DURATION = 3.0


# Print one block every N MPC updates.
PRINT_EVERY_N_MPC = 10


# ============================================================
# QP PARAMETERS
# ============================================================

QP_MAX_ITER = 10000

QP_CONSTRAINT_TOLERANCE = 1e-4


# ============================================================
# DIAGNOSTIC CONSTANTS
# ============================================================

GRAVITY = 9.81


# ============================================================
# BUILD STANDING CONTACT SCHEDULE
# ============================================================

def build_standing_contact_schedule(
    horizon,
):
    """
    Standing:

        both feet are stance feet

    for the complete prediction horizon.

    contact_schedule[j] = [1, 1]
    """

    horizon = int(
        horizon
    )


    if horizon <= 0:

        raise ValueError(
            "horizon phải > 0."
        )


    contact_schedule = np.ones(
        (
            horizon,
            2,
        ),
        dtype=int,
    )


    if contact_schedule.shape != (
        horizon,
        2,
    ):

        raise RuntimeError(
            "Standing contact schedule sai kích thước."
        )


    return contact_schedule


# ============================================================
# BUILD STANDING SRBD REFERENCE
# ============================================================

def build_standing_reference(
    x_initial,
    horizon,
):
    """
    Standing target:

        p_CoM_ref
            =
        initial CoM position

        v_CoM_ref
            =
        0

        euler_CoM_ref
            =
        initial orientation

        w_CoM_ref
            =
        0


    The resulting MPC reference follows:

        X_ref =
        [
            x_ref(k+1)
            ...
            x_ref(k+N)
        ]
    """

    x_initial = np.asarray(
        x_initial,
        dtype=float,
    ).reshape(-1)


    if x_initial.shape != (
        13,
    ):

        raise ValueError(
            "x_initial phải có shape (13,)."
        )


    horizon = int(
        horizon
    )


    if horizon <= 0:

        raise ValueError(
            "horizon phải > 0."
        )


    # ========================================================
    # CoM POSITION
    # ========================================================

    W_p_CoM_ref = np.tile(

        x_initial[
            0:3
        ].reshape(
            1,
            3,
        ),

        (
            horizon,
            1,
        ),
    )


    # ========================================================
    # CoM LINEAR VELOCITY
    # ========================================================

    W_v_CoM_ref = np.zeros(
        (
            horizon,
            3,
        ),
        dtype=float,
    )


    # ========================================================
    # BODY / CoM ORIENTATION
    # ========================================================

    W_euler_CoM_ref = (
        x_initial[
            3:6
        ].copy()
    )


    # ========================================================
    # ANGULAR VELOCITY
    # ========================================================

    W_w_CoM_ref = np.zeros(
        3,
        dtype=float,
    )


    # ========================================================
    # BUILD SRBD REFERENCE
    # ========================================================

    reference = build_srbd_reference(

        W_p_CoM_ref=(
            W_p_CoM_ref
        ),

        W_v_CoM_ref=(
            W_v_CoM_ref
        ),

        W_euler_CoM_ref=(
            W_euler_CoM_ref
        ),

        W_w_CoM_ref=(
            W_w_CoM_ref
        ),
    )


    return reference


# ============================================================
# CONTACT-FRAME WRENCH
# ============================================================

def get_contact_frame_wrench(
    u0_MPC,
    L6_IK_R_W,
    R6_IK_R_W,
):
    """
    MPC wrench convention:

        u =
        [
            W_F_L
            W_M_L
            W_F_R
            W_M_R
        ]

    Transform World-frame forces/moments into the corrected
    L6_IK and R6_IK contact frames for diagnostics only.
    """

    u0_MPC = np.asarray(
        u0_MPC,
        dtype=float,
    ).reshape(-1)


    if u0_MPC.shape != (
        12,
    ):

        raise ValueError(
            "u0_MPC phải có shape (12,)."
        )


    W_F_L = (
        u0_MPC[
            0:3
        ]
    )


    W_M_L = (
        u0_MPC[
            3:6
        ]
    )


    W_F_R = (
        u0_MPC[
            6:9
        ]
    )


    W_M_R = (
        u0_MPC[
            9:12
        ]
    )


    L_F = (
        L6_IK_R_W
        @
        W_F_L
    )


    L_M = (
        L6_IK_R_W
        @
        W_M_L
    )


    R_F = (
        R6_IK_R_W
        @
        W_F_R
    )


    R_M = (
        R6_IK_R_W
        @
        W_M_R
    )


    return (
        L_F,
        L_M,
        R_F,
        R_M,
    )


# ============================================================
# PRINT STATUS
# ============================================================

def print_status(
    mpc_iteration,
    control_time,
    current,
    standing_reference,
    held_u0_MPC,
    tau_command,
    solver_info,
    post_diagnostics,
    mpc_compute_time,
    mass_CoM,
    contact_state,
):
    """
    Compact runtime diagnostic.
    """

    (
        L_F,
        L_M,
        R_F,
        R_M,

    ) = get_contact_frame_wrench(

        u0_MPC=(
            held_u0_MPC
        ),

        L6_IK_R_W=(
            current.L6_IK_R_W
        ),

        R6_IK_R_W=(
            current.R6_IK_R_W
        ),
    )


    W_p_CoM_ref = (
        standing_reference[
            "W_p_CoM_ref"
        ][
            0
        ]
    )


    W_v_CoM_ref = (
        standing_reference[
            "W_v_CoM_ref"
        ][
            0
        ]
    )


    total_Fz = float(

        held_u0_MPC[
            2
        ]

        +

        held_u0_MPC[
            8
        ]
    )


    expected_mg = float(

        mass_CoM

        *

        GRAVITY
    )


    print("\n")
    print("-" * 70)


    print(
        "MPC iteration =",
        mpc_iteration
    )


    print(
        "control_time =",
        f"{control_time:.4f}",
        "s"
    )


    print(
        "mode = STANDING"
    )


    # ========================================================
    # CONTACT
    # ========================================================

    print(
        "\nphysical contact =",
        [
            contact_state[
                "c_L"
            ],
            contact_state[
                "c_R"
            ],
        ]
    )


    print(
        "left contact count =",
        contact_state[
            "left_contact_count"
        ]
    )


    print(
        "right contact count =",
        contact_state[
            "right_contact_count"
        ]
    )


    # ========================================================
    # STATE
    # ========================================================

    print(
        "\nCoM position =",
        current.x[
            0:3
        ]
    )


    print(
        "CoM position reference =",
        W_p_CoM_ref
    )


    print(
        "CoM position error =",
        current.x[
            0:3
        ]
        -
        W_p_CoM_ref
    )


    print(
        "\nEuler =",
        current.x[
            3:6
        ]
    )


    print(
        "\nCoM velocity =",
        current.x[
            6:9
        ]
    )


    print(
        "CoM velocity reference =",
        W_v_CoM_ref
    )


    print(
        "\nAngular velocity =",
        current.x[
            9:12
        ]
    )


    # ========================================================
    # MPC WRENCH
    # ========================================================

    print(
        "\nu0_MPC [World] ="
    )


    print(
        held_u0_MPC
    )


    print(
        "\nLeft wrench [L6_IK frame]"
    )


    print(
        "F_L =",
        L_F
    )


    print(
        "M_L =",
        L_M
    )


    print(
        "\nRight wrench [R6_IK frame]"
    )


    print(
        "F_R =",
        R_F
    )


    print(
        "M_R =",
        R_M
    )


    # ========================================================
    # VERTICAL FORCE
    # ========================================================

    print(
        "\nFz_L + Fz_R =",
        total_Fz,
        "N"
    )


    print(
        "m*g =",
        expected_mg,
        "N"
    )


    print(
        "vertical force error =",
        total_Fz
        -
        expected_mg,
        "N"
    )


    # ========================================================
    # TORQUE
    # ========================================================

    print(
        "\ntau_command ="
    )


    print(
        tau_command
    )


    # ========================================================
    # SOLVER
    # ========================================================

    print(
        "\nOSQP status =",
        solver_info.get(
            "status"
        )
    )


    print(
        "OSQP iterations =",
        solver_info.get(
            "iterations"
        )
    )


    print(
        "OSQP solve time =",
        solver_info.get(
            "solve_time"
        ),
        "s"
    )


    if post_diagnostics is not None:

        print(
            "Constraint max violation =",
            post_diagnostics.get(
                "max_violation"
            )
        )


    print(
        "MPC compute time =",
        mpc_compute_time,
        "s"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. INITIALIZE COMMON SIMULATION
    # ========================================================

    setup = initialize_simulation(

        xml_path=(
            XML_PATH
        ),

        horizon=(
            N
        ),

        dt_mpc=(
            DT_MPC
        ),

        dt_mpc_update=(
            DT_MPC_UPDATE
        ),

        home_key_name="home",

        qp_verbose=False,

        qp_eps_abs=1e-5,

        qp_eps_rel=1e-5,

        qp_max_iter=(
            QP_MAX_ITER
        ),

        qp_polishing=True,

        qp_enable_warm_start=False,

        qp_constraint_tolerance=(
            QP_CONSTRAINT_TOLERANCE
        ),

        print_initialization=True,
    )


    model = (
        setup.model
    )


    data = (
        setup.data
    )


    # ========================================================
    # 2. TIMING
    # ========================================================

    DT_SIM = float(
        setup.dt_sim
    )


    MPC_EVERY = int(
        setup.steps_per_mpc_update
    )


    # --------------------------------------------------------
    # Low-level wrench-to-torque mapping is evaluated at every
    # MuJoCo physics step:
    #
    #     DT_LOW = DT_SIM = 0.0005 s
    #
    # Therefore:
    #
    #     2000 Hz
    # --------------------------------------------------------

    DT_LOW = (
        DT_SIM
    )


    # ========================================================
    # 3. STANDING REFERENCE
    #
    # Reference is scenario-specific and therefore belongs
    # here, NOT in simulation_setup.py.
    # ========================================================

    standing_reference = (
        build_standing_reference(

            x_initial=(
                setup.x_initial
            ),

            horizon=(
                N
            ),
        )
    )


    X_ref = np.asarray(

        standing_reference[
            "X_ref"
        ],

        dtype=float,

    ).reshape(-1)


    if X_ref.shape != (
        13 * N,
    ):

        raise RuntimeError(
            "Standing X_ref sai dimension."
        )


    # ========================================================
    # 4. FIXED DOUBLE-SUPPORT SCHEDULE
    # ========================================================

    contact_schedule = (
        build_standing_contact_schedule(
            N
        )
    )


    # ========================================================
    # 5. HELD CONTROL SIGNALS
    #
    # MPC wrench is updated every DT_MPC_UPDATE and held with
    # ZOH between MPC solves.
    # ========================================================

    held_u0_MPC = np.zeros(
        12,
        dtype=float,
    )


    tau_command = np.zeros(
        10,
        dtype=float,
    )


    # ========================================================
    # 6. SOLVER DIAGNOSTICS
    # ========================================================

    last_solver_info = None

    last_post_diagnostics = None

    last_mpc_compute_time = 0.0


    # ========================================================
    # 7. LOGGER
    # ========================================================

    logger = (
        SimulationLogger()
    )


    # ========================================================
    # 8. COUNTERS
    # ========================================================

    physics_step = 0

    mpc_iteration = 0


    # ========================================================
    # 9. CONFIGURATION PRINT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STANDING SIMULATION - SRBD MPC")
    print("=" * 70)


    print(
        "\nDT_SIM =",
        DT_SIM,
        "s"
    )


    print(
        "Physics rate =",
        1.0 / DT_SIM,
        "Hz"
    )


    print(
        "\nDT_LOW =",
        DT_LOW,
        "s"
    )


    print(
        "Low-level rate =",
        1.0 / DT_LOW,
        "Hz"
    )


    print(
        "\nDT_MPC_UPDATE =",
        DT_MPC_UPDATE,
        "s"
    )


    print(
        "MPC update rate =",
        1.0 / DT_MPC_UPDATE,
        "Hz"
    )


    print(
        "\nDT_MPC prediction spacing =",
        DT_MPC,
        "s"
    )


    print(
        "N =",
        N
    )


    print(
        "Prediction horizon =",
        N * DT_MPC,
        "s"
    )


    print(
        "\nStanding CoM position target =",
        standing_reference[
            "W_p_CoM_ref"
        ][
            0
        ]
    )


    print(
        "Standing velocity target =",
        standing_reference[
            "W_v_CoM_ref"
        ][
            0
        ]
    )


    print(
        "\nStanding contact schedule ="
    )


    print(
        contact_schedule
    )


    print(
        "\nControl order:"
    )


    print(
        "x_k"
    )


    print(
        " -> MPC update if scheduled"
    )


    print(
        " -> ZOH u_k"
    )


    print(
        " -> tau = -J(q_k)^T W_k"
    )


    print(
        " -> data.ctrl"
    )


    print(
        " -> mj_step"
    )


    print(
        " -> x_{k+1}"
    )


    # ========================================================
    # 10. PASSIVE VIEWER
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # ====================================================
        # MAIN PHYSICS LOOP
        # ====================================================

        while (

            viewer.is_running()

            and

            data.time
            <
            SIM_DURATION
        ):

            wall_step_start = (
                time.perf_counter()
            )


            # =================================================
            # CURRENT DISCRETE CONTROL INDEX
            # =================================================

            control_time = (

                physics_step

                *

                DT_SIM
            )


            is_mpc_tick = (

                physics_step

                %

                MPC_EVERY

                ==

                0
            )


            # Shared x_k snapshot for this physics step.
            current = None


            # =================================================
            # A. MPC UPDATE
            #
            # Only every:
            #
            #     DT_MPC_UPDATE = 0.01 s
            #
            # Sequence:
            #
            #     x_k
            #       ->
            #     SRBD prediction
            #       ->
            #     QP
            #       ->
            #     u_k
            #
            # No mj_step occurs here.
            # =================================================

            if is_mpc_tick:

                mpc_compute_start = (
                    time.perf_counter()
                )


                # =============================================
                # A1. CURRENT STATE x_k
                # =============================================

                current = (
                    get_current_state_mpc(
                        model,
                        data,
                    )
                )


                x_k = (
                    current.x.copy()
                )


                # =============================================
                # A2. BUILD SRBD PREDICTION
                #
                # Standing uses current moment arms frozen
                # across this MPC prediction horizon.
                # =============================================

                (
                    Phi,
                    Gamma,

                ) = build_SRBD_model(

                    Ew_inverse=(
                        current.Ew_inverse
                    ),

                    mass_CoM=(
                        setup.mass_CoM
                    ),

                    W_inertia_CoM=(
                        setup.W_inertia_CoM_q0
                    ),

                    W_r_L=(
                        current.W_r_L
                    ),

                    W_r_R=(
                        current.W_r_R
                    ),

                    dt_mpc=(
                        DT_MPC
                    ),

                    horizon=(
                        N
                    ),
                )


                # =============================================
                # A3. SOLVE QP
                #
                # IMPORTANT:
                #
                # QP moment / CoP constraints use corrected
                # L6_IK / R6_IK orientation frames.
                # =============================================

                (
                    U_opt,
                    u0_opt,
                    solver_info,
                    post_diagnostics,

                ) = build_and_solve_QP(

                    qp_solver=(
                        setup.qp_solver
                    ),

                    Phi=(
                        Phi
                    ),

                    Gamma=(
                        Gamma
                    ),

                    x_current=(
                        x_k
                    ),

                    X_ref=(
                        X_ref
                    ),

                    N=(
                        N
                    ),

                    L6_R_W=(
                        current.L6_IK_R_W
                    ),

                    R6_R_W=(
                        current.R6_IK_R_W
                    ),

                    contact_schedule=(
                        contact_schedule
                    ),

                    print_diagnostics=False,
                )


                # =============================================
                # A4. QP FAILURE
                # =============================================

                if u0_opt is None:

                    data.ctrl[
                        :
                    ] = 0.0


                    print("\n")
                    print("=" * 70)
                    print("MPC STOPPED")
                    print("=" * 70)


                    print(
                        "control_time =",
                        control_time,
                        "s"
                    )


                    print(
                        "MPC iteration =",
                        mpc_iteration
                    )


                    print(
                        "\nsolver_info =",
                        solver_info
                    )


                    if post_diagnostics is not None:

                        print_QP_diagnostics(
                            post_diagnostics
                        )


                    raise RuntimeError(
                        "Standing QP không trả về nghiệm hợp lệ."
                    )


                # =============================================
                # A5. HOLD FIRST MPC INPUT
                # =============================================

                held_u0_MPC = np.asarray(

                    u0_opt,

                    dtype=float,

                ).reshape(-1)


                if held_u0_MPC.shape != (
                    12,
                ):

                    raise RuntimeError(
                        "u0_MPC phải có shape (12,)."
                    )


                if not np.all(
                    np.isfinite(
                        held_u0_MPC
                    )
                ):

                    raise RuntimeError(
                        "u0_MPC chứa NaN hoặc Inf."
                    )


                # =============================================
                # A6. STORE DIAGNOSTICS
                # =============================================

                last_solver_info = (
                    solver_info
                )


                last_post_diagnostics = (
                    post_diagnostics
                )


                last_mpc_compute_time = (

                    time.perf_counter()

                    -

                    mpc_compute_start
                )


            # =================================================
            # B. LOW-LEVEL WRENCH -> JOINT TORQUE
            #
            # This runs EVERY physics step.
            #
            # MPC wrench is ZOH-held, but Jacobian is evaluated
            # at the CURRENT q_k:
            #
            #     tau_k = -J(q_k)^T W_k
            #
            # No mj_step has occurred yet.
            # =================================================

            if current is None:

                current = (
                    get_current_state_mpc(
                        model,
                        data,
                    )
                )


            (
                tau_command,
                tau_L,
                tau_R,

            ) = get_joint_torque_from_MPC_wrench(

                model,

                data,

                held_u0_MPC,
            )


            tau_command = np.asarray(
                tau_command,
                dtype=float,
            ).reshape(-1)


            if tau_command.shape != (
                10,
            ):

                raise RuntimeError(
                    "tau_command phải có shape (10,)."
                )


            if not np.all(
                np.isfinite(
                    tau_command
                )
            ):

                raise RuntimeError(
                    "tau_command chứa NaN hoặc Inf."
                )


            # =================================================
            # C. APPLY CONTROL
            #
            # This writes data.ctrl only.
            #
            # No clipping here.
            #
            # MuJoCo ctrlrange remains the actuator limit.
            # =================================================

            apply_joint_torque(

                model,

                data,

                tau_command,
            )


            # =================================================
            # D. MPC-CYCLE LOGGING / PRINT
            #
            # Still before mj_step.
            # =================================================

            if is_mpc_tick:

                contact_state = (
                    get_contact_state(

                        model=(
                            model
                        ),

                        data=(
                            data
                        ),

                        contact_map=(
                            setup.contact_map
                        ),
                    )
                )


                qp_solve_time = float(

                    last_solver_info.get(
                        "solve_time",
                        0.0,
                    )
                )


                logger.log_step(

                    time=(
                        control_time
                    ),

                    x=(
                        current.x
                    ),

                    u0=(
                        held_u0_MPC
                    ),

                    tau=(
                        tau_command
                    ),

                    mpc_compute_time=(
                        last_mpc_compute_time
                    ),

                    qp_solve_time=(
                        qp_solve_time
                    ),
                )


                if (
                    mpc_iteration
                    %
                    PRINT_EVERY_N_MPC
                    ==
                    0
                ):

                    print_status(

                        mpc_iteration=(
                            mpc_iteration
                        ),

                        control_time=(
                            control_time
                        ),

                        current=(
                            current
                        ),

                        standing_reference=(
                            standing_reference
                        ),

                        held_u0_MPC=(
                            held_u0_MPC
                        ),

                        tau_command=(
                            tau_command
                        ),

                        solver_info=(
                            last_solver_info
                        ),

                        post_diagnostics=(
                            last_post_diagnostics
                        ),

                        mpc_compute_time=(
                            last_mpc_compute_time
                        ),

                        mass_CoM=(
                            setup.mass_CoM
                        ),

                        contact_state=(
                            contact_state
                        ),
                    )


                mpc_iteration += 1


            # =================================================
            # E. PHYSICS STEP
            #
            # This is the ONLY place where MuJoCo state is
            # advanced.
            #
            # Strict sequence:
            #
            #     x_k
            #       ->
            #     compute u_k
            #       ->
            #     apply u_k
            #       ->
            #     mj_step
            #       ->
            #     x_{k+1}
            # =================================================

            mujoco.mj_step(
                model,
                data,
            )


            physics_step += 1


            # =================================================
            # F. VIEWER
            # =================================================

            viewer.sync()


            # =================================================
            # G. OPTIONAL REAL-TIME PACING
            #
            # If computation is slower than DT_SIM, no sleep
            # is applied.
            # =================================================

            wall_elapsed = (

                time.perf_counter()

                -

                wall_step_start
            )


            sleep_time = (

                DT_SIM

                -

                wall_elapsed
            )


            if sleep_time > 0.0:

                time.sleep(
                    sleep_time
                )


    # ========================================================
    # 11. FINAL STATE
    # ========================================================

    final_state = (
        get_current_state_mpc(
            model,
            data,
        )
    )


    W_p_CoM_ref = (
        standing_reference[
            "W_p_CoM_ref"
        ][
            0
        ]
    )


    W_v_CoM_ref = (
        standing_reference[
            "W_v_CoM_ref"
        ][
            0
        ]
    )


    # ========================================================
    # 12. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STANDING SIMULATION FINISHED")
    print("=" * 70)


    print(
        "\nSimulation time =",
        data.time,
        "s"
    )


    print(
        "Physics steps =",
        physics_step
    )


    print(
        "MPC updates =",
        mpc_iteration
    )


    print(
        "\nFinal CoM position =",
        final_state.x[
            0:3
        ]
    )


    print(
        "Reference CoM position =",
        W_p_CoM_ref
    )


    print(
        "Position error =",
        final_state.x[
            0:3
        ]
        -
        W_p_CoM_ref
    )


    print(
        "\nFinal CoM velocity =",
        final_state.x[
            6:9
        ]
    )


    print(
        "Reference CoM velocity =",
        W_v_CoM_ref
    )


    print(
        "Velocity error =",
        final_state.x[
            6:9
        ]
        -
        W_v_CoM_ref
    )


    print(
        "\nFinal Euler =",
        final_state.x[
            3:6
        ]
    )


    print(
        "Final angular velocity =",
        final_state.x[
            9:12
        ]
    )


    # ========================================================
    # 13. LOGGER SUMMARY
    # ========================================================

    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()