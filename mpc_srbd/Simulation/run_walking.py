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
FOOTHOLD_PLANNER_DIR = MPC_SRBD_DIR / "Foothold_Planner"
TRAJECTORY_PLANNER_DIR = MPC_SRBD_DIR / "Trajectory_Planner"

for directory in (
    MPC_SRBD_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
    COMMAND_DIR,
    FOOTHOLD_PLANNER_DIR,
    TRAJECTORY_PLANNER_DIR,
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

# Debug mode:
#
# Low-level controller is updated at every MuJoCo
# physics step.
#
# XML:
#
#     DT_SIM = 0.0005 s
#
# Therefore:
#
#     DT_LOW = DT_SIM
#
#     = 0.0005 s
#
#     = 2000 Hz
#
DT_LOW = 0.0005


# Swing-foot clearance [m]
H_SWING = 0.015


# Raibert velocity-feedback gain
K_D_RAIBERT = 0.05


# Flat-terrain ground height [m]
GROUND_HEIGHT = 0.0


# CoM velocity command in World frame [m/s].
#
# Debug first with stepping in place:
#
#     [0.0, 0.0, 0.0]
#
# Later, for slow forward walking:
#
#     [0.02, 0.0, 0.0]
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
SIM_DURATION = 3.0


# Print one status every this many MPC updates
PRINT_EVERY_N_MPC = 10


# Current QP needs more than the default 1000
# iterations when the prediction horizon contains
# mixed contact modes.
QP_MAX_ITER = 10000


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
# HELPERS
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
# PRINT STATUS
# ============================================================

def print_status(
    mpc_iteration,
    control_time,
    current,
    current_contact,
    swing_foot,
    delta_T,
    held_u0_MPC,
    tau_command,
    last_swing_result,
    last_solver_info,
    last_post_diagnostics,
    last_mpc_compute_time,
    last_contact_schedule,
    reference_generator
):
    """
    Print one compact simulation status block.
    """

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
        "mode = WALKING"
    )

    print(
        "t_gait =",
        f"{control_time:.4f}",
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
        reference_generator.get_velocity_reference()
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

        qp_max_iter=QP_MAX_ITER
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
    # Single-support duration is also swing duration:
    #
    #     T_SWING
    #
    # =
    #
    #     N_SS * dt_mpc
    # --------------------------------------------------------

    T_SWING = (
        N_SS
        *
        dt_mpc
    )


    # --------------------------------------------------------
    # Number of physics steps per low-level update
    # --------------------------------------------------------

    LOW_LEVEL_EVERY = validate_timing_multiple(

        larger_dt=DT_LOW,

        smaller_dt=DT_SIM,

        name="DT_LOW / DT_SIM"
    )


    # --------------------------------------------------------
    # Number of physics steps per MPC update
    # --------------------------------------------------------

    MPC_EVERY = validate_timing_multiple(

        larger_dt=dt_mpc_update,

        smaller_dt=DT_SIM,

        name="dt_mpc_update / DT_SIM"
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


    # --------------------------------------------------------
    # Number of low-level updates during one swing
    # --------------------------------------------------------

    SWING_LOW_LEVEL_UPDATES = (
        validate_timing_multiple(

            larger_dt=T_SWING,

            smaller_dt=DT_LOW,

            name="T_SWING / DT_LOW"
        )
    )


    # ========================================================
    # 3. INITIALIZE SWING TRAJECTORY
    # ========================================================

    swing_trajectory = (
        CubicBezierSwingTrajectory(

            dt_low=DT_LOW,

            T_s=T_SWING,

            h_swing=H_SWING
        )
    )


    # ========================================================
    # 4. HELD / CURRENT SIGNALS
    # ========================================================

    # --------------------------------------------------------
    # First MPC input:
    #
    # [
    #     W_L
    #     W_R
    # ]
    #
    # held by ZOH between MPC updates.
    # --------------------------------------------------------

    held_u0_MPC = np.zeros(
        12,
        dtype=float
    )


    # --------------------------------------------------------
    # Final joint torque:
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


    # --------------------------------------------------------
    # Current contact information
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
    # 5. LOGGER / COUNTERS
    # ========================================================

    logger = SimulationLogger()


    physics_step = 0

    mpc_iteration = 0

    low_level_iteration = 0


    # ========================================================
    # 6. PRINT CONFIGURATION
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "WALKING SIMULATION"
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
        "T_SWING =",
        T_SWING,
        "s"
    )


    print(
        "Low-level updates / swing =",
        SWING_LOW_LEVEL_UPDATES
    )


    print(
        "QP max iterations =",
        QP_MAX_ITER
    )


    print(
        "\nV_COM_CMD =",
        V_COM_CMD
    )


    print(
        "\nGait starts immediately at t = 0."
    )


    print(
        "No separate initial-standing mode is used."
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


            # -------------------------------------------------
            # No separate standing mode:
            #
            #     t_gait = control_time
            # -------------------------------------------------

            t_gait = (
                control_time
            )


            # -------------------------------------------------
            # MPC tick
            # -------------------------------------------------

            is_mpc_tick = (

                physics_step
                %
                MPC_EVERY
                ==
                0
            )


            # -------------------------------------------------
            # Low-level tick
            # -------------------------------------------------

            is_low_level_tick = (

                physics_step
                %
                LOW_LEVEL_EVERY
                ==
                0
            )


            # -------------------------------------------------
            # Shared current-state snapshot.
            #
            # If MPC and low-level occur at the same physics
            # step, the same state snapshot is reused.
            # -------------------------------------------------

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
                # A2. WALKING CONTACT SCHEDULE
                #
                # Scheduler is active from t = 0.
                #
                # With:
                #
                #     N_DS = 2
                #     N_SS = 8
                #     dt_mpc = 0.025
                #
                # at t = 0:
                #
                #     k0 : DS
                #     k1 : DS
                #     k2...k9 : LSS
                #
                # Therefore the first MPC horizon already
                # anticipates the DS -> LSS transition.
                # =============================================

                contact_schedule = (
                    get_contact_schedule(

                        current_time=t_gait,

                        N=N,

                        dt_mpc=dt_mpc
                    )
                )


                # =============================================
                # A3. MPC REFERENCE
                # =============================================

                v_CoM_cmd_current = (
                    V_COM_CMD.copy()
                )


                X_ref = (
                    setup.reference_generator.generate_reference(

                        v_CoM_cmd_current
                    )
                )


                # =============================================
                # A4. BUILD SRBD PREDICTION MODEL
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
                        "t_gait =",
                        t_gait,
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
                # A7. HOLD FIRST MPC INPUT
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
                #     {}^W p_dot_CoM
                #
                W_p_dot_CoM = (
                    current.x[
                        6:9
                    ].copy()
                )


                # Current desired CoM velocity:
                #
                #     {}^W p_dot_CoM_ref
                #
                W_p_dot_CoM_ref = (
                    setup.reference_generator.get_velocity_reference()
                )


                # =============================================
                # B2. CURRENT CONTACT MODE
                #
                # Same scheduler and same t_gait used by MPC.
                # =============================================

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
                # MPC wrench is ZOH-held between MPC solves.
                #
                # Jacobian mapping is recomputed at every
                # low-level / MuJoCo physics step.
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


                    # Reset swing trajectory so the next
                    # stance -> swing transition captures a
                    # fresh physical liftoff point.
                    swing_trajectory.reset()


                    last_swing_result = None


                # ---------------------------------------------
                # LEFT SINGLE SUPPORT
                #
                #     [1, 0]
                #
                # Left:
                #     stance -> MPC
                #
                # Right:
                #     swing -> Raibert -> Bezier -> IK -> PD
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


                    # Left leg = stance
                    tau_L = (
                        tau_L_MPC.copy()
                    )


                    # Right leg = swing
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
                # Left:
                #     swing -> Raibert -> Bezier -> IK -> PD
                #
                # Right:
                #     stance -> MPC
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


                    # Left leg = swing
                    tau_L = (
                        swing_result.tau_swing.copy()
                    )


                    # Right leg = stance
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
                # No manual clipping.
                #
                # MuJoCo actuator ctrlrange defined in XML
                # remains the actuator-limit source of truth.
                # =============================================

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


                # ---------------------------------------------
                # LOGGER
                # ---------------------------------------------

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


                # ---------------------------------------------
                # STATUS PRINT
                # ---------------------------------------------

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

                        current_contact=(
                            current_contact
                        ),

                        swing_foot=(
                            swing_foot
                        ),

                        delta_T=(
                            delta_T
                        ),

                        held_u0_MPC=(
                            held_u0_MPC
                        ),

                        tau_command=(
                            tau_command
                        ),

                        last_swing_result=(
                            last_swing_result
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

                        last_contact_schedule=(
                            last_contact_schedule
                        ),

                        reference_generator=(
                            setup.reference_generator
                        )
                    )


                mpc_iteration += 1


            # =================================================
            # D. MUJOCO PHYSICS STEP
            #
            # 2000 Hz
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


    print("\n")

    print("=" * 70)

    print(
        "WALKING SIMULATION FINISHED"
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
        setup.reference_generator.get_position_reference()
    )


    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()