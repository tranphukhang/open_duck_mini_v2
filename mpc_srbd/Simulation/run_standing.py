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
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(
            0,
            directory_str,
        )


# ============================================================
# SIMULATION
# ============================================================

from simulation_setup import initialize_simulation
from simulation_logger import SimulationLogger


# ============================================================
# STATE
# ============================================================

from get_current_state_mpc import get_current_state_mpc
from get_contact_state import get_contact_state


# ============================================================
# SRBD
# ============================================================

from build_SRBD_model import build_SRBD_model


# ============================================================
# QP
# ============================================================

from build_and_solve_QP import build_and_solve_QP
from QP_diagnostics import print_QP_diagnostics


# ============================================================
# KINEMATICS
# ============================================================

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench,
)


# ============================================================
# ACTUATOR
# ============================================================

from actuator_mapping import apply_joint_torque


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

# Actual MPC re-solve period.
DT_MPC_UPDATE = 0.01


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

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


# ============================================================
# STANDING REFERENCE
# ============================================================

def build_standing_X_ref(
    x_initial,
    horizon,
):
    """
    Build the stacked standing SRBD reference directly.

    State:

        x =
        [
            p_CoM,      # 3
            euler,      # 3
            v_CoM,      # 3
            omega,      # 3
            1           # 1
        ]

    Standing target:

        p_ref     = p_initial
        euler_ref = euler_initial
        v_ref     = 0
        omega_ref = 0
        x[12]     = 1

    The same reference is repeated over the complete prediction
    horizon:

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

    horizon = int(
        horizon
    )

    if x_initial.shape != (13,):
        raise ValueError(
            "x_initial phải có shape (13,)."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon phải > 0."
        )

    if not np.all(
        np.isfinite(
            x_initial
        )
    ):
        raise ValueError(
            "x_initial chứa NaN hoặc Inf."
        )

    x_ref = np.zeros(
        13,
        dtype=float,
    )

    # CoM position target.
    x_ref[0:3] = x_initial[0:3]

    # Orientation target.
    x_ref[3:6] = x_initial[3:6]

    # Linear velocity target.
    x_ref[6:9] = 0.0

    # Angular velocity target.
    x_ref[9:12] = 0.0

    # Homogeneous constant state.
    x_ref[12] = 1.0

    X_ref = np.tile(
        x_ref,
        horizon,
    )

    expected_shape = (
        13 * horizon,
    )

    if X_ref.shape != expected_shape:
        raise RuntimeError(
            f"X_ref phải có shape {expected_shape}, "
            f"nhưng nhận được {X_ref.shape}."
        )

    return (
        x_ref,
        X_ref,
    )


# ============================================================
# STANDING CONTACT SCHEDULE
# ============================================================

def build_standing_contact_schedule(
    horizon,
    contact_meas,
):
    """
    Build standing contact schedule.

    Important convention:

        row 0:
            CURRENT measured physical contact feedback.

        rows 1 ... N-1:
            standing command = double support [1, 1].

    Therefore:

        c_0 = c_k^meas

    while the future standing plan remains:

        [1, 1].
    """

    horizon = int(
        horizon
    )

    if horizon <= 0:
        raise ValueError(
            "horizon phải > 0."
        )

    contact_meas = np.asarray(
        contact_meas,
        dtype=int,
    ).reshape(-1)

    if contact_meas.shape != (2,):
        raise ValueError(
            "contact_meas phải có shape (2,)."
        )

    if not np.all(
        np.isin(
            contact_meas,
            [0, 1],
        )
    ):
        raise ValueError(
            "contact_meas chỉ được chứa 0 hoặc 1."
        )

    contact_schedule = np.ones(
        (
            horizon,
            2,
        ),
        dtype=int,
    )

    # Current MPC input u0 must use current physical contact.
    contact_schedule[0] = contact_meas

    return contact_schedule


# ============================================================
# CONTACT-FRAME WRENCH FOR DIAGNOSTICS
# ============================================================

def get_contact_frame_wrench(
    u0_MPC,
    L6_IK_R_W,
    R6_IK_R_W,
):
    """
    MPC wrench:

        u =
        [
            W_F_L,
            W_M_L,
            W_F_R,
            W_M_R
        ]

    Convert World-frame wrench to corrected foot contact frames
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

    W_F_L = u0_MPC[0:3]
    W_M_L = u0_MPC[3:6]
    W_F_R = u0_MPC[6:9]
    W_M_R = u0_MPC[9:12]

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
    x_ref,
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
    Compact standing-MPC diagnostics.
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
        "mode = STANDING"
    )

    print(
        "\ncontact_meas =",
        [
            contact_state["c_L"],
            contact_state["c_R"],
        ]
    )

    print(
        "contact_schedule first rows ="
    )

    print(
        contact_schedule[
            :min(
                4,
                contact_schedule.shape[0],
            )
        ]
    )

    print(
        "\nCoM actual =",
        current.x[0:3]
    )

    print(
        "CoM ref    =",
        x_ref[0:3]
    )

    print(
        "CoM error  =",
        current.x[0:3]
        -
        x_ref[0:3]
    )

    print(
        "\nEuler actual =",
        current.x[3:6]
    )

    print(
        "Euler ref    =",
        x_ref[3:6]
    )

    print(
        "\nCoM velocity =",
        current.x[6:9]
    )

    print(
        "Angular velocity =",
        current.x[9:12]
    )

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

    print(
        "\ntau_command ="
    )

    print(
        tau_command
    )

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

    model = setup.model
    data = setup.data

    DT_SIM = float(
        setup.dt_sim
    )

    MPC_EVERY = int(
        setup.steps_per_mpc_update
    )

    # ========================================================
    # 2. STANDING REFERENCE
    # ========================================================

    (
        x_ref,
        X_ref,
    ) = build_standing_X_ref(
        x_initial=setup.x_initial,
        horizon=N,
    )

    # ========================================================
    # 3. HELD CONTROL
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
    # 4. LOGGER / COUNTERS
    # ========================================================

    logger = SimulationLogger()

    physics_step = 0
    mpc_iteration = 0

    last_solver_info = None
    last_post_diagnostics = None
    last_mpc_compute_time = 0.0

    # ========================================================
    # 5. CONFIGURATION
    # ========================================================

    print("\n")
    print("=" * 78)
    print("STANDING SIMULATION - FROZEN-LINEARIZATION SRBD MPC")
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
        "\nStanding x_ref ="
    )

    print(
        x_ref
    )

    print(
        "\nSRBD model convention:"
    )

    print(
        "B_inertia_CoM = constant"
    )

    print(
        "At each MPC update:"
    )

    print(
        "x_k, Ew_inverse, r_L, r_R, W_inertia_CoM <- feedback"
    )

    print(
        "W_inertia_CoM(k) = W_R_B(k) @ B_inertia_CoM @ W_R_B(k).T"
    )

    print(
        "Ew_inverse, r_L, r_R, W_inertia_CoM are frozen "
        "over the current prediction horizon."
    )

    print(
        "\nControl order:"
    )

    print(
        "x_k/contact_k"
        " -> build frozen SRBD"
        " -> solve QP"
        " -> u_k"
        " -> tau_k"
        " -> data.ctrl"
        " -> mj_step"
        " -> x_{k+1}"
    )

    # ========================================================
    # 6. PASSIVE VIEWER
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while (
            viewer.is_running()
            and
            data.time < SIM_DURATION
        ):

            wall_step_start = time.perf_counter()

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

                # ---------------------------------------------
                # A1. CURRENT FEEDBACK SNAPSHOT
                # ---------------------------------------------

                current = get_current_state_mpc(
                    model=model,
                    data=data,
                    B_inertia_CoM=setup.B_inertia_CoM,
                )

                x_k = current.x.copy()

                # ---------------------------------------------
                # A2. CURRENT PHYSICAL CONTACT FEEDBACK
                # ---------------------------------------------

                contact_state = get_contact_state(
                    model=model,
                    data=data,
                    contact_map=setup.contact_map,
                )

                contact_meas = np.array(
                    [
                        int(
                            contact_state["c_L"]
                        ),
                        int(
                            contact_state["c_R"]
                        ),
                    ],
                    dtype=int,
                )

                # ---------------------------------------------
                # A3. CONTACT SCHEDULE
                #
                # row 0:
                #     current contact feedback
                #
                # future:
                #     standing command = double support
                # ---------------------------------------------

                contact_schedule = (
                    build_standing_contact_schedule(
                        horizon=N,
                        contact_meas=contact_meas,
                    )
                )

                # ---------------------------------------------
                # A4. BUILD FROZEN SRBD MODEL
                #
                # All model quantities below are the current
                # feedback values and are frozen across this
                # prediction horizon.
                # ---------------------------------------------

                (
                    Phi,
                    Gamma,
                ) = build_SRBD_model(
                    Ew_inverse=current.Ew_inverse,
                    mass_CoM=setup.mass_CoM,
                    W_inertia_CoM=current.W_inertia_CoM,
                    W_r_L=current.W_r_L,
                    W_r_R=current.W_r_R,
                    dt_mpc=DT_MPC,
                    horizon=N,
                )

                # ---------------------------------------------
                # A5. SOLVE QP
                # ---------------------------------------------

                (
                    U_opt,
                    u0_opt,
                    solver_info,
                    post_diagnostics,
                ) = build_and_solve_QP(
                    qp_solver=setup.qp_solver,
                    Phi=Phi,
                    Gamma=Gamma,
                    x_current=x_k,
                    X_ref=X_ref,
                    N=N,
                    L6_R_W=current.L6_IK_R_W,
                    R6_R_W=current.R6_IK_R_W,
                    contact_schedule=contact_schedule,
                    print_diagnostics=False,
                )

                # ---------------------------------------------
                # A6. FAILURE HANDLING
                # ---------------------------------------------

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
                        "Standing QP không trả về nghiệm hợp lệ."
                    )

                # ---------------------------------------------
                # A7. HOLD CURRENT MPC INPUT u_k
                # ---------------------------------------------

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

                last_solver_info = solver_info
                last_post_diagnostics = post_diagnostics

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
            # The MPC wrench is ZOH-held between MPC updates,
            # but J(q_k) is evaluated at the current q_k.
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
                    mpc_compute_time=last_mpc_compute_time,
                    qp_solve_time=qp_solve_time,
                )

                if (
                    mpc_iteration
                    %
                    PRINT_EVERY_N_MPC
                    ==
                    0
                ):
                    print_status(
                        mpc_iteration=mpc_iteration,
                        control_time=control_time,
                        current=current,
                        x_ref=x_ref,
                        held_u0_MPC=held_u0_MPC,
                        tau_command=tau_command,
                        solver_info=last_solver_info,
                        post_diagnostics=last_post_diagnostics,
                        mpc_compute_time=last_mpc_compute_time,
                        mass_CoM=setup.mass_CoM,
                        contact_state=contact_state,
                        contact_schedule=contact_schedule,
                    )

                mpc_iteration += 1

            # =================================================
            # E. PHYSICS INTEGRATION
            #
            # Exactly one mj_step() here:
            #
            #     x_k
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

            # Optional real-time pacing.
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
    # 7. FINAL SUMMARY
    # ========================================================

    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
