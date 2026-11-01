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
COMMAND_DIR = MPC_SRBD_DIR / "Command"

for directory in (
    MPC_SRBD_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
    COMMAND_DIR,
):
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(
            0,
            directory_str
        )


# ============================================================
# SIMULATION
# ============================================================

from simulation_setup import (
    initialize_simulation
)

from simulation_logger import (
    SimulationLogger
)


# ============================================================
# STATE
# ============================================================

from get_current_state_mpc import (
    get_current_state_mpc
)


# ============================================================
# SRBD
# ============================================================

from build_SRBD_model import (
    build_SRBD_model
)


# ============================================================
# QP
# ============================================================

from QP_parameters import (
    N,
    dt_mpc,
    dt_mpc_update
)

from build_and_solve_QP import (
    build_and_solve_QP
)

from QP_diagnostics import (
    print_QP_diagnostics
)


# ============================================================
# CONTACT
# ============================================================

from Command.contact_schedule import (
    DOUBLE_SUPPORT
)


# ============================================================
# KINEMATICS
# ============================================================

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench
)


# ============================================================
# ACTUATOR
# ============================================================

from actuator_mapping import (
    apply_joint_torque
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=200
)


# ============================================================
# STANDING PARAMETERS
# ============================================================

# Low-level control / wrench-to-torque mapping rate.
#
# XML:
#
#     dt_sim = 0.0005 s
#
# Therefore:
#
#     DT_LOW = 0.0005 s
#
#     2000 Hz
#
DT_LOW = 0.0005


# Standing command:
#
#     desired CoM linear velocity = 0
#
# Therefore the CoM position reference remains fixed at
# its initial value.
V_COM_CMD = np.array(
    [
        0.0,
        0.0,
        0.0
    ],
    dtype=float
)


# Total simulation duration [s]
SIM_DURATION = 3.0


# Print one status block every this many MPC updates
PRINT_EVERY_N_MPC = 10


# Same generous solver iteration limit used during current
# walking development.
QP_MAX_ITER = 10000


# Standing has only one fixed contact mode, so use the normal
# stricter post-solve constraint tolerance.
QP_CONSTRAINT_TOLERANCE = 1e-4


# Gravity used only for diagnostic printing.
GRAVITY = 9.81


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
# TIMING HELPER
# ============================================================

def validate_timing_multiple(
    larger_dt,
    smaller_dt,
    name
):
    """
    Verify that larger_dt is an integer multiple
    of smaller_dt.
    """

    steps = int(
        round(
            larger_dt
            /
            smaller_dt
        )
    )


    if steps <= 0:

        raise ValueError(
            f"{name}: số step phải > 0."
        )


    if not np.isclose(
        steps
        *
        smaller_dt,
        larger_dt,
        atol=1e-12
    ):

        raise ValueError(
            f"{name}: {larger_dt} phải là bội số nguyên "
            f"của {smaller_dt}."
        )


    return steps


# ============================================================
# STANDING CONTACT SCHEDULE
# ============================================================

def build_standing_contact_schedule(
    horizon
):
    """
    Standing uses Double Support for the entire MPC horizon:

        [1, 1]
        [1, 1]
        ...
        [1, 1]
    """

    horizon = int(
        horizon
    )


    if horizon <= 0:

        raise ValueError(
            "horizon phải > 0."
        )


    contact_schedule = np.tile(

        np.asarray(
            DOUBLE_SUPPORT,
            dtype=int
        ).reshape(
            1,
            2
        ),

        (
            horizon,
            1
        )
    )


    if contact_schedule.shape != (
        horizon,
        2
    ):

        raise RuntimeError(
            "Standing contact_schedule sai kích thước."
        )


    if not np.all(
        contact_schedule
        ==
        1
    ):

        raise RuntimeError(
            "Standing contact_schedule phải luôn là [1, 1]."
        )


    return contact_schedule


# ============================================================
# CONTACT-FRAME WRENCH DIAGNOSTIC
# ============================================================

def get_contact_frame_wrench(
    u0_MPC,
    L6_IK_R_W,
    R6_IK_R_W
):
    """
    Transform the current World-frame MPC wrenches into the
    corrected L6_IK / R6_IK contact frames.

    This function is used only for diagnostics.

    MPC input order:

        u =
        [
            W_F_L,
            W_M_L,
            W_F_R,
            W_M_R
        ]
    """

    u0_MPC = np.asarray(
        u0_MPC,
        dtype=float
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
        R_M
    )


# ============================================================
# PRINT STATUS
# ============================================================

def print_status(
    mpc_iteration,
    control_time,
    current,
    held_u0_MPC,
    tau_command,
    last_solver_info,
    last_post_diagnostics,
    last_mpc_compute_time,
    reference_generator,
    mass_CoM
):
    """
    Print one compact standing status block.
    """


    (
        L_F,
        L_M,
        R_F,
        R_M

    ) = get_contact_frame_wrench(

        u0_MPC=(
            held_u0_MPC
        ),

        L6_IK_R_W=(
            current.L6_IK_R_W
        ),

        R6_IK_R_W=(
            current.R6_IK_R_W
        )
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


    print(
        "\ncurrent_contact =",
        DOUBLE_SUPPORT
    )


    # --------------------------------------------------------
    # CURRENT STATE
    # --------------------------------------------------------

    print(
        "\nCoM position =",
        current.x[
            0:3
        ]
    )


    print(
        "CoM position reference =",
        reference_generator.get_position_reference()
    )


    print(
        "\nBase Euler [rad] =",
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
        reference_generator.get_velocity_reference()
    )


    print(
        "\nBase angular velocity =",
        current.x[
            9:12
        ]
    )


    # --------------------------------------------------------
    # MPC WRENCH
    # --------------------------------------------------------

    print(
        "\nu0_MPC [World frame] ="
    )


    print(
        held_u0_MPC
    )


    print(
        "\nLeft wrench [L6_IK frame]:"
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
        "\nRight wrench [R6_IK frame]:"
    )


    print(
        "F_R =",
        R_F
    )


    print(
        "M_R =",
        R_M
    )


    # --------------------------------------------------------
    # VERTICAL FORCE BALANCE
    # --------------------------------------------------------

    print(
        "\nFz_L + Fz_R [World] =",
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


    # --------------------------------------------------------
    # JOINT TORQUE
    # --------------------------------------------------------

    print(
        "\ntau_command ="
    )


    print(
        tau_command
    )


    # --------------------------------------------------------
    # QP
    # --------------------------------------------------------

    print(
        "\nOSQP status =",
        last_solver_info[
            "status"
        ]
    )


    print(
        "OSQP iterations =",
        last_solver_info[
            "iterations"
        ]
    )


    print(
        "OSQP solve time =",
        last_solver_info[
            "solve_time"
        ],
        "s"
    )


    if last_post_diagnostics is not None:

        print(
            "Constraint max violation =",
            last_post_diagnostics[
                "max_violation"
            ]
        )


    print(
        "MPC compute time =",
        last_mpc_compute_time,
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

        xml_path=XML_PATH,

        horizon=N,

        dt_mpc=dt_mpc,

        dt_mpc_update=dt_mpc_update,

        qp_max_iter=QP_MAX_ITER,

        qp_constraint_tolerance=(
            QP_CONSTRAINT_TOLERANCE
        )
    )


    model = (
        setup.model
    )


    data = (
        setup.data
    )


    # ========================================================
    # 2. SIMULATION TIMING
    # ========================================================

    DT_SIM = float(
        model.opt.timestep
    )


    # --------------------------------------------------------
    # Number of MuJoCo physics steps per low-level update
    # --------------------------------------------------------

    LOW_LEVEL_EVERY = (
        validate_timing_multiple(

            larger_dt=DT_LOW,

            smaller_dt=DT_SIM,

            name="DT_LOW / DT_SIM"
        )
    )


    # --------------------------------------------------------
    # Number of MuJoCo physics steps per MPC update
    # --------------------------------------------------------

    MPC_EVERY = (
        validate_timing_multiple(

            larger_dt=dt_mpc_update,

            smaller_dt=DT_SIM,

            name="dt_mpc_update / DT_SIM"
        )
    )


    # --------------------------------------------------------
    # Every MPC tick must also be a low-level tick
    # --------------------------------------------------------

    if (
        MPC_EVERY
        %
        LOW_LEVEL_EVERY
        !=
        0
    ):

        raise ValueError(
            "MPC update phải trùng với một "
            "low-level update."
        )


    # ========================================================
    # 3. STANDING CONTACT SCHEDULE
    # ========================================================

    contact_schedule = (
        build_standing_contact_schedule(
            horizon=N
        )
    )


    # ========================================================
    # 4. HELD / CURRENT SIGNALS
    # ========================================================

    # --------------------------------------------------------
    # MPC wrench:
    #
    # [
    #     W_L
    #     W_R
    # ]
    #
    # Held by ZOH between MPC updates.
    # --------------------------------------------------------

    held_u0_MPC = np.zeros(
        12,
        dtype=float
    )


    # --------------------------------------------------------
    # Final actuator torque:
    #
    # [
    #     tau_L
    #     tau_R
    # ]
    # --------------------------------------------------------

    tau_command = np.zeros(
        10,
        dtype=float
    )


    last_solver_info = None

    last_post_diagnostics = None

    last_mpc_compute_time = 0.0


    # ========================================================
    # 5. LOGGER / COUNTERS
    # ========================================================

    logger = (
        SimulationLogger()
    )


    physics_step = 0

    mpc_iteration = 0

    low_level_iteration = 0


    # ========================================================
    # 6. PRINT CONFIGURATION
    # ========================================================

    initial_current = (
        get_current_state_mpc(
            model,
            data
        )
    )


    print("\n")

    print("=" * 70)

    print(
        "STANDING SIMULATION - SRBD MPC"
    )

    print("=" * 70)


    print(
        "DT_SIM =",
        DT_SIM,
        "s"
    )


    print(
        "DT_LOW =",
        DT_LOW,
        "s"
    )


    print(
        "Low-level rate =",
        1.0 / DT_LOW,
        "Hz"
    )


    print(
        "dt_mpc_update =",
        dt_mpc_update,
        "s"
    )


    print(
        "MPC update rate =",
        1.0 / dt_mpc_update,
        "Hz"
    )


    print(
        "dt_mpc =",
        dt_mpc,
        "s"
    )


    print(
        "Prediction horizon =",
        N
        *
        dt_mpc,
        "s"
    )


    print(
        "QP max iterations =",
        QP_MAX_ITER
    )


    print(
        "QP post-check tolerance =",
        QP_CONSTRAINT_TOLERANCE
    )


    print(
        "\nV_COM_CMD =",
        V_COM_CMD
    )


    print(
        "\nStanding contact schedule ="
    )


    print(
        contact_schedule
    )


    # --------------------------------------------------------
    # FRAME DIAGNOSTIC
    # --------------------------------------------------------

    print(
        "\nOriginal L6_R_W at home ="
    )


    print(
        initial_current.L6_R_W
    )


    print(
        "\nOriginal R6_R_W at home ="
    )


    print(
        initial_current.R6_R_W
    )


    print(
        "\nL6_IK_R_W used by QP moment/CoP constraint ="
    )


    print(
        initial_current.L6_IK_R_W
    )


    print(
        "\nR6_IK_R_W used by QP moment/CoP constraint ="
    )


    print(
        initial_current.R6_IK_R_W
    )


    print(
        "\nIMPORTANT:"
    )


    print(
        "QP moment/CoP constraint uses "
        "L6_IK/R6_IK frames."
    )


    print(
        "Wrench -> joint torque mapping keeps the "
        "physical L6/R6 Jacobians."
    )


    # ========================================================
    # 7. MAIN SIMULATION LOOP
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:


        while (

            viewer.is_running()

            and

            data.time
            <
            SIM_DURATION

        ):

            # =================================================
            # CURRENT DETERMINISTIC CONTROL TIME
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


            is_low_level_tick = (

                physics_step
                %
                LOW_LEVEL_EVERY
                ==
                0
            )


            # Shared snapshot if both loops occur at the
            # same physics step.
            current = None


            # =================================================
            # A. MPC UPDATE
            # =================================================

            if is_mpc_tick:

                mpc_compute_start = (
                    time.perf_counter()
                )


                # =============================================
                # A1. CURRENT STATE
                # =============================================

                current = (
                    get_current_state_mpc(
                        model,
                        data
                    )
                )


                x_k = (
                    current.x.copy()
                )


                # =============================================
                # A2. STANDING REFERENCE
                # =============================================

                X_ref = (
                    setup.reference_generator.generate_reference(

                        V_COM_CMD.copy()
                    )
                )


                # =============================================
                # A3. BUILD SRBD PREDICTION MODEL
                # =============================================

                (
                    Phi,
                    Gamma

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
                        dt_mpc
                    ),

                    horizon=N
                )


                # =============================================
                # A4. BUILD AND SOLVE QP
                #
                # IMPORTANT
                # ---------
                #
                # Current QP API argument names remain:
                #
                #     L6_R_W
                #     R6_R_W
                #
                # but the actual matrices passed are:
                #
                #     current.L6_IK_R_W
                #     current.R6_IK_R_W
                #
                # Therefore moment / CoP constraints now use
                # the corrected contact orientation frames.
                # =============================================

                (
                    U_opt,
                    u0_opt,
                    solver_info,
                    post_diagnostics

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

                    print_diagnostics=False
                )


                # =============================================
                # A5. INVALID MPC SOLUTION
                # =============================================

                if u0_opt is None:

                    data.ctrl[
                        :
                    ] = 0.0


                    print("\n")

                    print("=" * 70)

                    print(
                        "MPC STOPPED"
                    )

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
                        "contact_schedule ="
                    )


                    print(
                        contact_schedule
                    )


                    if post_diagnostics is not None:

                        print_QP_diagnostics(
                            post_diagnostics
                        )


                    else:

                        print(
                            "solver status =",
                            solver_info.get(
                                "status"
                            )
                        )


                        print(
                            "solver iterations =",
                            solver_info.get(
                                "iterations"
                            )
                        )


                        print(
                            "primal residual =",
                            solver_info.get(
                                "primal_residual"
                            )
                        )


                        print(
                            "dual residual =",
                            solver_info.get(
                                "dual_residual"
                            )
                        )


                    raise RuntimeError(
                        "Không có MPC input hợp lệ."
                    )


                # =============================================
                # A6. HOLD FIRST MPC WRENCH
                # =============================================

                held_u0_MPC = np.asarray(
                    u0_opt,
                    dtype=float
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
                # A7. STORE MPC INFORMATION
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
            # B. LOW-LEVEL UPDATE
            #
            # Standing:
            #
            # both legs are stance.
            #
            # The MPC wrench is ZOH-held between solves,
            # while:
            #
            #     tau = -J(q)^T W
            #
            # is recomputed at every low-level step.
            # =================================================

            if is_low_level_tick:


                if current is None:

                    current = (
                        get_current_state_mpc(
                            model,
                            data
                        )
                    )


                (
                    tau_command,
                    _,
                    _

                ) = get_joint_torque_from_MPC_wrench(

                    model,

                    data,

                    held_u0_MPC
                )


                tau_command = np.asarray(
                    tau_command,
                    dtype=float
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


                # No manual clipping.
                #
                # MuJoCo actuator ctrlrange remains the
                # actuator-limit source of truth.
                apply_joint_torque(

                    model,

                    data,

                    tau_command
                )


                low_level_iteration += 1


            # =================================================
            # C. LOG / PRINT CURRENT MPC CYCLE
            # =================================================

            if is_mpc_tick:


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
                        last_solver_info[
                            "solve_time"
                        ]
                    )
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

                        held_u0_MPC=(
                            held_u0_MPC
                        ),

                        tau_command=(
                            tau_command
                        ),

                        last_solver_info=(
                            last_solver_info
                        ),

                        last_post_diagnostics=(
                            last_post_diagnostics
                        ),

                        last_mpc_compute_time=(
                            last_mpc_compute_time
                        ),

                        reference_generator=(
                            setup.reference_generator
                        ),

                        mass_CoM=(
                            setup.mass_CoM
                        )
                    )


                mpc_iteration += 1


            # =================================================
            # D. MUJOCO PHYSICS STEP
            # =================================================

            mujoco.mj_step(
                model,
                data
            )


            physics_step += 1


            # =================================================
            # E. VIEWER
            #
            # Refresh display at MPC rate.
            # =================================================

            if (
                physics_step
                %
                MPC_EVERY
                ==
                0
            ):

                viewer.sync()


    # ========================================================
    # 8. FINALIZE LOGGER
    # ========================================================

    logger.finalize()


    # ========================================================
    # 9. FINAL SUMMARY
    # ========================================================

    final_state = (
        get_current_state_mpc(
            model,
            data
        )
    )


    final_position_reference = (
        setup.reference_generator.get_position_reference()
    )


    final_velocity_reference = (
        setup.reference_generator.get_velocity_reference()
    )


    final_position_error = (

        final_state.x[
            0:3
        ]

        -

        final_position_reference
    )


    final_velocity_error = (

        final_state.x[
            6:9
        ]

        -

        final_velocity_reference
    )


    print("\n")

    print("=" * 70)

    print(
        "STANDING SIMULATION FINISHED"
    )

    print("=" * 70)


    print(
        "physics steps =",
        physics_step
    )


    print(
        "MPC iterations =",
        mpc_iteration
    )


    print(
        "low-level iterations =",
        low_level_iteration
    )


    print(
        "final simulation time =",
        data.time,
        "s"
    )


    print(
        "\nFinal CoM position =",
        final_state.x[
            0:3
        ]
    )


    print(
        "Final CoM position reference =",
        final_position_reference
    )


    print(
        "Final CoM position error =",
        final_position_error
    )


    print(
        "Final CoM position error norm =",
        np.linalg.norm(
            final_position_error
        ),
        "m"
    )


    print(
        "\nFinal CoM velocity =",
        final_state.x[
            6:9
        ]
    )


    print(
        "Final CoM velocity reference =",
        final_velocity_reference
    )


    print(
        "Final CoM velocity error norm =",
        np.linalg.norm(
            final_velocity_error
        ),
        "m/s"
    )


    print(
        "\nFinal base Euler [rad] =",
        final_state.x[
            3:6
        ]
    )


    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()