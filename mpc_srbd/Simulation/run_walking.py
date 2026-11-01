from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

SIMULATION_DIR = Path(
    __file__
).resolve().parent


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


FOOTHOLD_PLANNER_DIR = (
    MPC_SRBD_DIR
    /
    "Foothold_Planner"
)


TRAJECTORY_PLANNER_DIR = (
    MPC_SRBD_DIR
    /
    "Trajectory_Planner"
)


for directory in (

    MPC_SRBD_DIR,

    STATE_DIR,

    SRBD_DIR,

    QP_DIR,

    KINEMATICS_DIR,

    ACTUATOR_DIR,

    COMMAND_DIR,

    FOOTHOLD_PLANNER_DIR,

    TRAJECTORY_PLANNER_DIR

):

    if str(directory) not in sys.path:

        sys.path.insert(
            0,
            str(directory)
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

from initial_standing import (
    get_initial_standing_info
)

from swing_low_level_controller import (
    compute_swing_low_level_control
)


# ============================================================
# STATE
# ============================================================

from get_current_state_mpc import (
    get_current_state_mpc
)

from W_position_L6 import (
    get_W_position_L6
)

from W_position_R6 import (
    get_W_position_R6
)

from W_position_L6_ref import (
    get_W_position_L6_ref
)

from W_position_R6_ref import (
    get_W_position_R6_ref
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
# CONTACT SCHEDULE
# ============================================================

from Command.contact_schedule import (
    N_SS,
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
    get_contact_schedule,
    get_current_contact_info
)


# ============================================================
# SWING TRAJECTORY
# ============================================================

from Trajectory_Planner.cubic_Bezier import (
    CubicBezierSwingTrajectory
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
# WALKING SCENARIO PARAMETERS
# ============================================================

# Low-level controller update period:
#
#     400 Hz
#
DT_LOW = 0.0025


# Initial standing duration [s]
T_STAND = 1.0


# Swing-foot clearance [m]
H_SWING = 0.015


# Raibert velocity-feedback gain
K_D_RAIBERT = 0.05


# Flat-terrain ground height [m]
GROUND_HEIGHT = 0.0


# Initial bring-up test:
#
#     stepping in place
#
V_COM_CMD = np.array(
    [
        0.0,
        0.0,
        0.0
    ],
    dtype=float
)


# Total simulation duration [s]
#
# 1 s standing
#
# +
#
# 2 s gait
#
# With current gait:
#
#     T_cycle = 0.5 s
#
# this gives approximately 4 gait cycles.
#
SIM_DURATION = 3.0


# Print one status every this many MPC updates
PRINT_EVERY_N_MPC = 10


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

        dt_mpc_update=dt_mpc_update
    )


    model = setup.model

    data = setup.data


    # ========================================================
    # 2. SIMULATION TIMING
    # ========================================================

    DT_SIM = float(
        model.opt.timestep
    )


    # --------------------------------------------------------
    # Swing duration
    #
    #     T_SWING = N_SS * dt_mpc
    # --------------------------------------------------------

    T_SWING = (
        N_SS
        *
        dt_mpc
    )


    # --------------------------------------------------------
    # Number of MuJoCo physics steps per low-level update
    # --------------------------------------------------------

    LOW_LEVEL_EVERY = int(
        round(
            DT_LOW
            /
            DT_SIM
        )
    )


    # --------------------------------------------------------
    # Number of MuJoCo physics steps per MPC update
    # --------------------------------------------------------

    MPC_EVERY = int(
        round(
            dt_mpc_update
            /
            DT_SIM
        )
    )


    # --------------------------------------------------------
    # Number of physics steps during initial standing
    # --------------------------------------------------------

    STAND_STEPS = int(
        round(
            T_STAND
            /
            DT_SIM
        )
    )


    # ========================================================
    # 3. TIMING VALIDATION
    # ========================================================

    if not np.isclose(

        LOW_LEVEL_EVERY
        *
        DT_SIM,

        DT_LOW,

        atol=1e-12

    ):

        raise ValueError(
            "DT_LOW phải là bội số nguyên của DT_SIM."
        )


    if not np.isclose(

        MPC_EVERY
        *
        DT_SIM,

        dt_mpc_update,

        atol=1e-12

    ):

        raise ValueError(
            "dt_mpc_update phải là bội số nguyên "
            "của DT_SIM."
        )


    if not np.isclose(

        STAND_STEPS
        *
        DT_SIM,

        T_STAND,

        atol=1e-12

    ):

        raise ValueError(
            "T_STAND phải là bội số nguyên của DT_SIM."
        )


    # MPC tick should also be a low-level tick.
    #
    # Current system:
    #
    #     LOW_LEVEL_EVERY = 5
    #
    #     MPC_EVERY = 20
    #
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


    # Initial walking transition should occur exactly on
    # both MPC and low-level update boundaries.
    if (
        STAND_STEPS
        %
        LOW_LEVEL_EVERY
        !=
        0
    ):

        raise ValueError(
            "T_STAND phải kết thúc tại low-level tick."
        )


    if (
        STAND_STEPS
        %
        MPC_EVERY
        !=
        0
    ):

        raise ValueError(
            "T_STAND phải kết thúc tại MPC tick."
        )


    # --------------------------------------------------------
    # Swing duration should correspond to an integer number
    # of low-level updates.
    # --------------------------------------------------------

    SWING_LOW_LEVEL_UPDATES = int(
        round(
            T_SWING
            /
            DT_LOW
        )
    )


    if not np.isclose(

        SWING_LOW_LEVEL_UPDATES
        *
        DT_LOW,

        T_SWING,

        atol=1e-12

    ):

        raise ValueError(
            "T_SWING phải là bội số nguyên của DT_LOW."
        )


    # ========================================================
    # 4. INITIALIZE SWING TRAJECTORY
    # ========================================================

    swing_trajectory = (
        CubicBezierSwingTrajectory(

            dt_low=DT_LOW,

            T_s=T_SWING,

            h_swing=H_SWING
        )
    )


    # ========================================================
    # 5. INITIAL HELD SIGNALS
    # ========================================================

    # First MPC input held by ZOH:
    #
    # u =
    #
    # [
    #     W_L
    #     W_R
    # ]
    #
    held_u0_MPC = np.zeros(
        12,
        dtype=float
    )


    # Final joint torque command:
    #
    # [
    #     tau_L
    #     tau_R
    # ]
    #
    tau_command = np.zeros(
        10,
        dtype=float
    )


    # --------------------------------------------------------
    # Current low-level information
    # --------------------------------------------------------

    current_contact = (
        DOUBLE_SUPPORT.copy()
    )

    swing_foot = None

    delta_T = None

    last_swing_result = None


    # --------------------------------------------------------
    # Last MPC information
    # --------------------------------------------------------

    last_solver_info = None

    last_post_diagnostics = None

    last_mpc_compute_time = 0.0

    last_contact_schedule = None


    # ========================================================
    # 6. LOGGER
    # ========================================================

    logger = SimulationLogger()


    # ========================================================
    # 7. COUNTERS
    # ========================================================

    physics_step = 0

    mpc_iteration = 0

    low_level_iteration = 0


    # ========================================================
    # 8. PRINT CONFIGURATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WALKING SIMULATION")
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
        "dt_mpc_update =",
        dt_mpc_update,
        "s"
    )


    print(
        "dt_mpc =",
        dt_mpc,
        "s"
    )


    print(
        "T_STAND =",
        T_STAND,
        "s"
    )


    print(
        "T_SWING =",
        T_SWING,
        "s"
    )


    print(
        "Low-level updates / swing =",
        SWING_LOW_LEVEL_UPDATES
    )


    print(
        "\nV_COM_CMD =",
        V_COM_CMD
    )


    print(
        "\nInitial test:"
        "\n    standing"
        "\n        ->"
        "\n    stepping in place"
    )


    # ========================================================
    # 9. MAIN SIMULATION LOOP
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:


        while (

            viewer.is_running()

            and

            data.time < SIM_DURATION

        ):

            # =================================================
            # CURRENT DETERMINISTIC CONTROL TIME
            #
            # Use integer physics-step count instead of
            # accumulated floating-point time for controller
            # scheduling.
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


            # -------------------------------------------------
            # Current state snapshot.
            #
            # If MPC and low-level updates occur at the same
            # physics step, the same snapshot is reused.
            # -------------------------------------------------

            current = None


            # =================================================
            # A. MPC UPDATE
            #
            #     100 Hz
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
                # A2. INITIAL STANDING / WALKING MODE
                # =============================================

                (
                    is_standing,
                    _,
                    _,
                    _,
                    standing_contact_schedule

                ) = get_initial_standing_info(

                    current_time=control_time,

                    T_stand=T_STAND,

                    N=N
                )


                # ---------------------------------------------
                # Initial standing
                # ---------------------------------------------

                if is_standing:

                    contact_schedule = (
                        standing_contact_schedule
                    )


                    # During initial standing:
                    #
                    #     CoM velocity reference = 0
                    #
                    v_CoM_cmd_current = np.zeros(
                        3,
                        dtype=float
                    )


                # ---------------------------------------------
                # Walking
                # ---------------------------------------------

                else:

                    t_gait = (
                        control_time
                        -
                        T_STAND
                    )


                    contact_schedule = (
                        get_contact_schedule(

                            current_time=t_gait,

                            N=N,

                            dt_mpc=dt_mpc
                        )
                    )


                    v_CoM_cmd_current = (
                        V_COM_CMD.copy()
                    )


                # =============================================
                # A3. MPC REFERENCE
                # =============================================

                X_ref = (
                    setup.reference_generator.generate_reference(
                        v_CoM_cmd_current
                    )
                )


                # =============================================
                # A4. BUILD CURRENT SRBD PREDICTION MODEL
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

                    dt_mpc=dt_mpc,

                    horizon=N
                )


                # =============================================
                # A5. BUILD AND SOLVE QP
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

                    Phi=Phi,

                    Gamma=Gamma,

                    x_current=x_k,

                    X_ref=X_ref,

                    N=N,

                    L6_R_W=(
                        current.L6_R_W
                    ),

                    R6_R_W=(
                        current.R6_R_W
                    ),

                    contact_schedule=(
                        contact_schedule
                    ),

                    print_diagnostics=False
                )


                # =============================================
                # A6. INVALID MPC SOLUTION
                # =============================================

                if u0_opt is None:

                    data.ctrl[:] = 0.0


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


                    raise RuntimeError(
                        "Không có MPC input hợp lệ."
                    )


                # =============================================
                # A7. HOLD FIRST MPC INPUT
                #
                # ZOH between MPC updates:
                #
                #     100 Hz MPC
                #
                #         ↓
                #
                #     400 Hz low-level
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
                # A8. STORE MPC INFORMATION
                # =============================================

                last_solver_info = (
                    solver_info
                )


                last_post_diagnostics = (
                    post_diagnostics
                )


                last_contact_schedule = (
                    contact_schedule.copy()
                )


                last_mpc_compute_time = (

                    time.perf_counter()

                    -

                    mpc_compute_start
                )


            # =================================================
            # B. LOW-LEVEL UPDATE
            #
            #     400 Hz
            # =================================================

            if is_low_level_tick:

                # =============================================
                # B1. CURRENT STATE
                # =============================================

                if current is None:

                    current = (
                        get_current_state_mpc(
                            model,
                            data
                        )
                    )


                # Current whole-body CoM velocity:
                #
                #     x[6:9]
                #
                # =
                #
                #     {}^W p_dot_CoM
                #
                W_p_dot_CoM = (
                    current.x[
                        6:9
                    ].copy()
                )


                # Current reference CoM velocity:
                #
                #     {}^W p_dot_CoM^ref
                #
                W_p_dot_CoM_ref = (
                    setup.reference_generator.get_velocity_reference()
                )


                # =============================================
                # B2. CURRENT CONTACT MODE
                # =============================================

                (
                    is_standing,
                    standing_current_contact,
                    standing_swing_foot,
                    standing_delta_T,
                    _

                ) = get_initial_standing_info(

                    current_time=control_time,

                    T_stand=T_STAND,

                    N=N
                )


                # ---------------------------------------------
                # Initial standing
                # ---------------------------------------------

                if is_standing:

                    current_contact = (
                        standing_current_contact
                    )


                    swing_foot = (
                        standing_swing_foot
                    )


                    delta_T = (
                        standing_delta_T
                    )


                # ---------------------------------------------
                # Walking
                # ---------------------------------------------

                else:

                    t_gait = (
                        control_time
                        -
                        T_STAND
                    )


                    (
                        current_contact,
                        swing_foot,
                        delta_T

                    ) = get_current_contact_info(

                        current_time=t_gait,

                        dt_mpc=dt_mpc
                    )


                # =============================================
                # B3. MPC WRENCH -> CURRENT JOINT TORQUE
                #
                # MPC wrench is held from the latest 100-Hz
                # solve, but Jacobian mapping is evaluated at
                # the current 400-Hz robot configuration.
                # =============================================

                (
                    _,
                    tau_L_MPC,
                    tau_R_MPC

                ) = get_joint_torque_from_MPC_wrench(

                    model,

                    data,

                    held_u0_MPC
                )


                # =============================================
                # B4. CONTACT-BASED TORQUE ARBITRATION
                # =============================================

                # ---------------------------------------------
                # DOUBLE SUPPORT
                #
                #     [1, 1]
                #
                # Both legs use MPC wrench control.
                # ---------------------------------------------

                if np.array_equal(

                    current_contact,

                    DOUBLE_SUPPORT

                ):

                    if swing_foot is not None:

                        raise RuntimeError(
                            "Double Support nhưng "
                            "swing_foot không phải None."
                        )


                    tau_L = (
                        tau_L_MPC.copy()
                    )


                    tau_R = (
                        tau_R_MPC.copy()
                    )


                    # Reset swing trajectory so that the next
                    # stance -> swing transition captures a new
                    # physical liftoff position p0.
                    swing_trajectory.reset()


                    last_swing_result = None


                # ---------------------------------------------
                # LEFT SINGLE SUPPORT
                #
                #     [1, 0]
                #
                # Left  = stance -> MPC
                # Right = swing  -> low-level controller
                # ---------------------------------------------

                elif np.array_equal(

                    current_contact,

                    LEFT_SINGLE_SUPPORT

                ):

                    if swing_foot != "R":

                        raise RuntimeError(
                            "LSS phải có swing_foot='R'."
                        )


                    # -----------------------------------------
                    # Current physical foot positions
                    # -----------------------------------------

                    W_p_L6 = (
                        get_W_position_L6(
                            model,
                            data
                        )
                    )


                    W_p_R6 = (
                        get_W_position_R6(
                            model,
                            data
                        )
                    )


                    # -----------------------------------------
                    # Current nominal foot reference positions
                    # -----------------------------------------

                    W_p_L6_ref = (
                        get_W_position_L6_ref(
                            model,
                            data
                        )
                    )


                    W_p_R6_ref = (
                        get_W_position_R6_ref(
                            model,
                            data
                        )
                    )


                    # -----------------------------------------
                    # Raibert
                    # -> Bezier
                    # -> IK
                    # -> PD
                    # -----------------------------------------

                    swing_result = (
                        compute_swing_low_level_control(

                            model=model,

                            data=data,

                            swing_trajectory=(
                                swing_trajectory
                            ),

                            swing_foot="R",

                            delta_T=delta_T,

                            W_p_L6_ref=(
                                W_p_L6_ref
                            ),

                            W_p_R6_ref=(
                                W_p_R6_ref
                            ),

                            W_p_L6=(
                                W_p_L6
                            ),

                            W_p_R6=(
                                W_p_R6
                            ),

                            W_p_dot_CoM=(
                                W_p_dot_CoM
                            ),

                            W_p_dot_CoM_ref=(
                                W_p_dot_CoM_ref
                            ),

                            k_d=(
                                K_D_RAIBERT
                            ),

                            ground_height=(
                                GROUND_HEIGHT
                            )
                        )
                    )


                    tau_L = (
                        tau_L_MPC.copy()
                    )


                    tau_R = (
                        swing_result.tau_swing.copy()
                    )


                    last_swing_result = (
                        swing_result
                    )


                # ---------------------------------------------
                # RIGHT SINGLE SUPPORT
                #
                #     [0, 1]
                #
                # Left  = swing  -> low-level controller
                # Right = stance -> MPC
                # ---------------------------------------------

                elif np.array_equal(

                    current_contact,

                    RIGHT_SINGLE_SUPPORT

                ):

                    if swing_foot != "L":

                        raise RuntimeError(
                            "RSS phải có swing_foot='L'."
                        )


                    # -----------------------------------------
                    # Current physical foot positions
                    # -----------------------------------------

                    W_p_L6 = (
                        get_W_position_L6(
                            model,
                            data
                        )
                    )


                    W_p_R6 = (
                        get_W_position_R6(
                            model,
                            data
                        )
                    )


                    # -----------------------------------------
                    # Current nominal foot reference positions
                    # -----------------------------------------

                    W_p_L6_ref = (
                        get_W_position_L6_ref(
                            model,
                            data
                        )
                    )


                    W_p_R6_ref = (
                        get_W_position_R6_ref(
                            model,
                            data
                        )
                    )


                    # -----------------------------------------
                    # Raibert
                    # -> Bezier
                    # -> IK
                    # -> PD
                    # -----------------------------------------

                    swing_result = (
                        compute_swing_low_level_control(

                            model=model,

                            data=data,

                            swing_trajectory=(
                                swing_trajectory
                            ),

                            swing_foot="L",

                            delta_T=delta_T,

                            W_p_L6_ref=(
                                W_p_L6_ref
                            ),

                            W_p_R6_ref=(
                                W_p_R6_ref
                            ),

                            W_p_L6=(
                                W_p_L6
                            ),

                            W_p_R6=(
                                W_p_R6
                            ),

                            W_p_dot_CoM=(
                                W_p_dot_CoM
                            ),

                            W_p_dot_CoM_ref=(
                                W_p_dot_CoM_ref
                            ),

                            k_d=(
                                K_D_RAIBERT
                            ),

                            ground_height=(
                                GROUND_HEIGHT
                            )
                        )
                    )


                    tau_L = (
                        swing_result.tau_swing.copy()
                    )


                    tau_R = (
                        tau_R_MPC.copy()
                    )


                    last_swing_result = (
                        swing_result
                    )


                # ---------------------------------------------
                # INVALID CONTACT MODE
                # ---------------------------------------------

                else:

                    raise RuntimeError(
                        "current_contact không hợp lệ: "
                        f"{current_contact}"
                    )


                # =============================================
                # B5. BUILD FULL TORQUE COMMAND
                # =============================================

                tau_command = np.concatenate(
                    [
                        tau_L,
                        tau_R
                    ]
                )


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


                # =============================================
                # B6. APPLY TORQUE
                #
                # No manual clipping here.
                #
                # MuJoCo actuator ctrlrange defined in XML
                # handles actuator control limits.
                # =============================================

                apply_joint_torque(
                    model,
                    data,
                    tau_command
                )


                low_level_iteration += 1


            # =================================================
            # C. LOG CURRENT MPC CYCLE
            #
            # Every MPC tick is also a low-level tick, so
            # tau_command here is the torque command obtained
            # after current contact arbitration.
            # =================================================

            if is_mpc_tick:

                logger.log_step(

                    time=control_time,

                    x=current.x,

                    u0=held_u0_MPC,

                    tau=tau_command,

                    mpc_compute_time=(
                        last_mpc_compute_time
                    ),

                    qp_solve_time=(
                        last_solver_info[
                            "solve_time"
                        ]
                    )
                )


                # =============================================
                # STATUS PRINT
                # =============================================

                if (
                    mpc_iteration
                    %
                    PRINT_EVERY_N_MPC
                    ==
                    0
                ):

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


                    if control_time < T_STAND:

                        print(
                            "mode = INITIAL STANDING"
                        )


                    else:

                        print(
                            "mode = WALKING"
                        )


                        print(
                            "t_gait =",
                            f"{control_time - T_STAND:.4f}",
                            "s"
                        )


                    print(
                        "\ncurrent_contact =",
                        current_contact
                    )


                    print(
                        "swing_foot =",
                        swing_foot
                    )


                    print(
                        "delta_T =",
                        delta_T
                    )


                    print(
                        "\nCoM position =",
                        current.x[
                            0:3
                        ]
                    )


                    print(
                        "CoM velocity =",
                        current.x[
                            6:9
                        ]
                    )


                    print(
                        "CoM velocity reference =",
                        setup.reference_generator.get_velocity_reference()
                    )


                    print(
                        "\nu0_MPC ="
                    )


                    print(
                        held_u0_MPC
                    )


                    print(
                        "\ntau_command ="
                    )


                    print(
                        tau_command
                    )


                    if last_swing_result is not None:

                        print(
                            "\nfoothold =",
                            last_swing_result.W_p_foothold
                        )


                        print(
                            "swing foot target =",
                            last_swing_result.W_p_swing_des
                        )


                        print(
                            "Bezier phase next =",
                            last_swing_result.phase_next
                        )


                        print(
                            "IK converged =",
                            last_swing_result.ik_diagnostics[
                                "converged"
                            ]
                        )


                        print(
                            "IK final error norm =",
                            last_swing_result.ik_diagnostics[
                                "final_error_norm"
                            ]
                        )


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


                    print(
                        "\nMPC horizon first contact =",
                        last_contact_schedule[
                            0
                        ]
                    )


                mpc_iteration += 1


            # =================================================
            # D. MUJOCO PHYSICS STEP
            #
            #     2000 Hz
            #
            # data.ctrl automatically holds the latest torque
            # between low-level updates.
            # =================================================

            mujoco.mj_step(
                model,
                data
            )


            physics_step += 1


            # =================================================
            # E. VIEWER
            #
            # Update display at MPC rate.
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
    # 10. FINALIZE LOGGER
    # ========================================================

    logger.finalize()


    # ========================================================
    # 11. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WALKING SIMULATION FINISHED")
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
        get_current_state_mpc(
            model,
            data
        ).x[
            0:3
        ]
    )


    print(
        "Final CoM position reference =",
        setup.reference_generator.get_position_reference()
    )


    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()