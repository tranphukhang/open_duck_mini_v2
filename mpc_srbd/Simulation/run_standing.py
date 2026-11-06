from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

SIMULATION_DIR = Path(__file__).resolve().parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent

STATE_DIR = MPC_SRBD_DIR / "State"
SRBD_DIR = MPC_SRBD_DIR / "SRBD"
QP_DIR = MPC_SRBD_DIR / "QP"
KINEMATICS_DIR = MPC_SRBD_DIR / "Kinematics"
ACTUATOR_DIR = MPC_SRBD_DIR / "Actuator"

for directory in (
    MPC_SRBD_DIR,
    SIMULATION_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
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
# MPC PARAMETERS
# ============================================================

N = 10

# Prediction discretization.
DT_MPC = 0.05

# Actual MPC solve/update period.
DT_MPC_UPDATE = 0.005


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

# Thời gian kiểm tra giữ thăng bằng.
SIM_DURATION = 10.0

PRINT_EVERY_N_MPC = 10


# ============================================================
# QP PARAMETERS
# ============================================================

QP_MAX_ITER = 10000

QP_CONSTRAINT_TOLERANCE = 1e-4


# ============================================================
# CONSTANTS
# ============================================================

GRAVITY = 9.81


STANCE_LEG = "L"   # "L" hoặc "R"
# ============================================================
# STANDING CONTACT SCHEDULE
# ============================================================

def build_standing_contact_schedule(
    horizon,
):
    """
    Build the PLANNED contact schedule for fixed standing.

    HECTOR / paper convention
    -------------------------

    The MPC contact sequence is supplied by the planned gait mode,
    not directly overwritten by raw measured contact.

    For fixed standing, both feet are planned to remain in contact
    over the complete prediction horizon:

        c_0     = [1, 1]
        c_1     = [1, 1]
        ...
        c_N-1   = [1, 1]

    Therefore:

        contact_schedule.shape = (N, 2)

    with:

        column 0 = left foot
        column 1 = right foot

        1 = planned stance/contact
        0 = planned swing/non-contact

    Important
    ---------
    Physical contact measurement is still read separately in the
    simulation for diagnostics and future gait-phase correction.

    However, raw contact measurement does NOT overwrite c_0 in the
    MPC prediction schedule.
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

    return contact_schedule

# def build_standing_contact_schedule(
#     horizon,
# ):
#     horizon = int(horizon)

#     if horizon <= 0:
#         raise ValueError(
#             "horizon must be > 0."
#         )

#     contact_schedule = np.zeros(
#         (horizon, 2),
#         dtype=int,
#     )

#     if STANCE_LEG == "L":

#         # Left stance, right swing
#         contact_schedule[:, 0] = 1

#     elif STANCE_LEG == "R":

#         # Right stance, left swing
#         contact_schedule[:, 1] = 1

#     else:

#         raise ValueError(
#             "STANCE_LEG must be 'L' or 'R'."
#         )

#     return contact_schedule


# ============================================================
# CONTACT-FRAME WRENCH FOR DIAGNOSTICS
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
            W_F_L,
            W_M_L,
            W_F_R,
            W_M_R
        ]

    Transform World-frame wrench into corrected contact frames
    for diagnostics only.
    """

    u0_MPC = np.asarray(
        u0_MPC,
        dtype=float,
    ).reshape(-1)

    if u0_MPC.shape != (12,):
        raise ValueError(
            "u0_MPC phải có shape (12,)."
        )

    W_F_L = (
        u0_MPC[0:3]
    )

    W_M_L = (
        u0_MPC[3:6]
    )

    W_F_R = (
        u0_MPC[6:9]
    )

    W_M_R = (
        u0_MPC[9:12]
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
# STATUS PRINT
# ============================================================

def print_status(
    mpc_iteration,
    control_time,
    current,
    x_ref_current,
    held_u0_MPC,
    tau_command,
    solver_info,
    post_diagnostics,
    mpc_compute_time,
    mass_CoM,
    contact_state,
    contact_schedule,
):
    """
    Runtime diagnostics for fixed standing test.
    """

    (
        L_F,
        L_M,
        R_F,
        R_M,
    ) = get_contact_frame_wrench(
        u0_MPC=held_u0_MPC,
        L6_IK_R_W=current.L6_IK_R_W,
        R6_IK_R_W=current.R6_IK_R_W,
    )

    total_Fz = float(
        held_u0_MPC[2]
        +
        held_u0_MPC[8]
    )

    expected_mg = float(
        mass_CoM
        *
        GRAVITY
    )

    print("\n")
    print("-" * 78)

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
        "mode = FIXED STANDING"
    )

    # ========================================================
    # CONTACT
    # ========================================================

    print(
        "\ncontact_meas [diagnostics only] =",
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
        "planned contact_schedule[0] =",
        contact_schedule[0]
    )

    # ========================================================
    # POSITION
    # ========================================================

    print(
        "\nCoM position actual =",
        current.x[0:3]
    )

    print(
        "CoM position ref    =",
        x_ref_current[0:3]
    )

    print(
        "CoM position error  =",
        current.x[0:3]
        -
        x_ref_current[0:3]
    )

    # ========================================================
    # Z POSITION
    # ========================================================

    print(
        "\nz actual =",
        current.x[2]
    )

    print(
        "z ref    =",
        x_ref_current[2]
    )

    print(
        "z error  =",
        current.x[2]
        -
        x_ref_current[2]
    )

    print(
        "vz actual =",
        current.x[8]
    )

    print(
        "vz ref    =",
        x_ref_current[8]
    )

    # ========================================================
    # ATTITUDE
    # ========================================================

    print(
        "\nEuler actual =",
        current.x[3:6]
    )

    print(
        "Euler ref    =",
        x_ref_current[3:6]
    )

    print(
        "Angular velocity =",
        current.x[9:12]
    )

    # ========================================================
    # FEEDBACK-LINEARIZED SRBD PARAMETERS
    # ========================================================

    print(
        "\nEw_inverse ="
    )

    print(
        current.Ew_inverse
    )

    print(
        "\nW_r_L =",
        current.W_r_L
    )

    print(
        "W_r_R =",
        current.W_r_R
    )

    print(
        "\nW_inertia_CoM(k) ="
    )

    print(
        current.W_inertia_CoM
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
    # 1. INITIALIZATION
    # ========================================================

    setup = initialize_simulation(
        xml_path=XML_PATH,
        horizon=N,
        dt_mpc=DT_MPC,
        dt_mpc_update=DT_MPC_UPDATE,
        home_key_name="home",
        qp_verbose=False,
        qp_eps_abs=1e-5,
        qp_eps_rel=1e-5,
        qp_max_iter=QP_MAX_ITER,
        qp_polishing=True,
        qp_enable_warm_start=False,
        qp_constraint_tolerance=QP_CONSTRAINT_TOLERANCE,
        print_initialization=True,
    )

    model = (
        setup.model
    )

    data = (
        setup.data
    )

    DT_SIM = float(
        setup.dt_sim
    )

    MPC_EVERY = int(
        setup.steps_per_mpc_update
    )

    # ========================================================
    # FIXED STANDING TARGET
    #
    # No external reference generator is used.
    #
    # The QP formulation still requires X_ref in the tracking
    # cost, so the initial standing state is used as one fixed
    # equilibrium target over the full prediction horizon.
    # ========================================================

    x_target = np.asarray(
        setup.x_initial,
        dtype=float,
    ).reshape(13).copy()

    X_ref = np.tile(
        x_target.reshape(
            1,
            13,
        ),
        (
            N,
            1,
        ),
    ).reshape(
        N
        *
        13
    )

    if not np.all(
        np.isfinite(
            X_ref
        )
    ):
        raise RuntimeError(
            "Fixed standing X_ref chứa NaN hoặc Inf."
        )

    # ========================================================
    # 2. HELD CONTROL
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
    # 3. LOGGER / COUNTERS
    # ========================================================

    logger = (
        SimulationLogger()
    )

    physics_step = 0

    mpc_iteration = 0

    last_solver_info = None

    last_post_diagnostics = None

    last_mpc_compute_time = 0.0

    # ========================================================
    # 4. CONFIGURATION
    # ========================================================

    print("\n")
    print("=" * 78)
    print("STANDING SIMULATION - FIXED EQUILIBRIUM TARGET")
    print("=" * 78)

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
        "\nControl order:"
    )

    print(
        "x_k"
        " + planned contact_schedule"
        " -> fixed X_ref"
        " -> build frozen SRBD"
        " -> solve QP"
        " -> u_k"
        " -> tau_k"
        " -> data.ctrl"
        " -> mj_step"
        " -> x_{k+1}"
    )

    # ========================================================
    # 5. PASSIVE VIEWER
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

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

            # =================================================
            # A. MPC UPDATE
            # =================================================

            if is_mpc_tick:

                mpc_compute_start = (
                    time.perf_counter()
                )

                # =============================================
                # A1. CURRENT FEEDBACK SNAPSHOT
                #
                # Continuous-state feedback used by MPC:
                #
                #   x_k
                #   Ew_inverse(k)
                #   W_inertia_CoM(k)
                #   W_r_L(k)
                #   W_r_R(k)
                #
                # No mj_step() occurs here.
                # =============================================

                current = (
                    get_current_state_mpc(
                        model=model,
                        data=data,
                        B_inertia_CoM=(
                            setup.B_inertia_CoM
                        ),
                    )
                )

                x_k = (
                    current.x.copy()
                )

                # =============================================
                # A2. CURRENT PHYSICAL CONTACT FEEDBACK
                #
                # IMPORTANT:
                #
                # This measurement is retained for:
                #
                #   - diagnostics
                #   - logging
                #   - future touchdown/liftoff detection
                #   - future gait-phase correction
                #
                # It does NOT directly overwrite c_0 of the MPC
                # planned contact schedule.
                # =============================================

                contact_state = (
                    get_contact_state(
                        model=model,
                        data=data,
                        contact_map=(
                            setup.contact_map
                        ),
                    )
                )

                contact_meas = np.array(
                    [
                        int(
                            contact_state[
                                "c_L"
                            ]
                        ),
                        int(
                            contact_state[
                                "c_R"
                            ]
                        ),
                    ],
                    dtype=int,
                )

                # =============================================
                # A3. PLANNED CONTACT SCHEDULE
                #
                # HECTOR / paper convention:
                #
                # The complete MPC contact sequence, including
                # c_0, comes from the planned gait mode.
                #
                # For fixed standing:
                #
                #   c_j = [1, 1]
                #
                # for:
                #
                #   j = 0, ..., N-1
                #
                # contact_meas is NOT used here.
                # =============================================

                contact_schedule = (
                    build_standing_contact_schedule(
                        horizon=N,
                    )
                )

                # =============================================
                # A4. FIXED STANDING TARGET
                #
                # No trajectory/reference generator is used.
                # X_ref was built once from setup.x_initial.
                # =============================================

                x_ref_current = (
                    x_target
                )

                # =============================================
                # A5. BUILD FROZEN SRBD MODEL
                #
                # Current feedback values:
                #
                #   Ew_inverse(k)
                #   W_inertia_CoM(k)
                #   W_r_L(k)
                #   W_r_R(k)
                #
                # are frozen over the current prediction horizon.
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
                        current.W_inertia_CoM
                    ),
                    W_r_L=(
                        current.W_r_L
                    ),
                    W_r_R=(
                        current.W_r_R
                    ),
                    dt_mpc=DT_MPC,
                    horizon=N,
                )

                # =============================================
                # A6. SOLVE QP
                #
                # MPC receives:
                #
                #   x_k                 <- feedback
                #
                #   contact_schedule    <- planned gait mode
                #
                # Therefore:
                #
                #   (x_k, c_0:N-1)
                #       ->
                #   U_opt
                #
                # and only:
                #
                #   u_0
                #
                # is applied.
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
                    Phi=Phi,
                    Gamma=Gamma,
                    x_current=x_k,
                    X_ref=X_ref,
                    N=N,
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
                # A7. QP FAILURE
                # =============================================

                if u0_opt is None:

                    data.ctrl[:] = 0.0

                    print("\n")
                    print("=" * 78)
                    print("MPC STOPPED")
                    print("=" * 78)

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
                        "Fixed-standing QP không trả về nghiệm hợp lệ."
                    )

                # =============================================
                # A8. HOLD FIRST MPC INPUT u_k
                # =============================================

                held_u0_MPC = np.asarray(
                    u0_opt,
                    dtype=float,
                ).reshape(-1)

                if held_u0_MPC.shape != (12,):
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
            # Runs every MuJoCo physics step.
            #
            # MPC wrench:
            #     held by ZOH between MPC solves.
            #
            # Jacobian:
            #     evaluated at current q_k.
            # =================================================

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

            if tau_command.shape != (10,):
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
            # C. APPLY tau_k
            # =================================================

            apply_joint_torque(
                model,
                data,
                tau_command,
            )

            # =================================================
            # D. LOG CURRENT MPC CYCLE
            #
            # Still before mj_step().
            # =================================================

            if is_mpc_tick:

                qp_solve_time = float(
                    last_solver_info.get(
                        "solve_time",
                        0.0,
                    )
                )

                logger.log_step(
                    time=control_time,
                    x=current.x,
                    u0=held_u0_MPC,
                    tau=tau_command,
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
                        current=current,
                        x_ref_current=(
                            x_ref_current
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
                        contact_schedule=(
                            contact_schedule
                        ),
                    )

                mpc_iteration += 1

            # =================================================
            # E. PHYSICS INTEGRATION
            #
            # IMPORTANT:
            #
            # Exactly ONE mj_step() in the main loop:
            #
            #     x_k
            #       ->
            #     u_k
            #       ->
            #     tau_k
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

            viewer.sync()

            # =================================================
            # F. OPTIONAL REAL-TIME PACING
            # =================================================

            elapsed = (
                time.perf_counter()
                -
                wall_step_start
            )

            remaining = (
                DT_SIM
                -
                elapsed
            )

            if remaining > 0.0:

                time.sleep(
                    remaining
                )

    # ========================================================
    # 6. FINAL SUMMARY
    # ========================================================

    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()