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
# WALKING PARAMETERS
# ============================================================

# MuJoCo physics / low-level controller:
#
#     0.0005 s
#
#     2000 Hz
#
DT_LOW = 0.0005


# Swing-foot clearance [m]
H_SWING = 0.015


# Raibert velocity feedback gain
K_D_RAIBERT = 0.05


# Flat-ground height [m]
GROUND_HEIGHT = 0.0


# ------------------------------------------------------------
# DEBUG FIRST:
#
# stepping in place
# ------------------------------------------------------------

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


# Print every N MPC updates
PRINT_EVERY_N_MPC = 10


# Walking QP currently needs generous maximum iterations.
QP_MAX_ITER = 10000


# Temporary debug tolerance for mixed-contact horizons.
QP_CONSTRAINT_TOLERANCE = 5e-4


# Gravity used only for diagnostics.
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
    Verify:

        larger_dt = integer * smaller_dt
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
        steps * smaller_dt,
        larger_dt,
        atol=1e-12
    ):

        raise ValueError(
            f"{name}: {larger_dt} phải là bội số nguyên "
            f"của {smaller_dt}."
        )

    return steps


# ============================================================
# BASE ROTATION MATRIX
# ============================================================

def get_W_R_B(
    model,
    data,
    site_name="base"
):
    """
    Return the current Body-to-World rotation matrix:

        W_R_B

    using the same MuJoCo base site convention as
    State/W_euler_base.py.

    MuJoCo:

        data.site_xmat

    stores the site orientation in the World frame.
    """

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    if site_id < 0:

        raise ValueError(
            f"Không tìm thấy site '{site_name}' trong MuJoCo model."
        )


    W_R_B = (
        data.site_xmat[
            site_id
        ]
        .reshape(
            3,
            3
        )
        .copy()
    )


    if W_R_B.shape != (
        3,
        3
    ):

        raise RuntimeError(
            "W_R_B phải có shape (3, 3)."
        )


    if not np.all(
        np.isfinite(
            W_R_B
        )
    ):

        raise RuntimeError(
            "W_R_B chứa NaN hoặc Inf."
        )


    return W_R_B


# ============================================================
# CONTACT-FRAME WRENCH
# ============================================================

def get_contact_frame_wrench(
    u0_MPC,
    L6_IK_R_W,
    R6_IK_R_W
):
    """
    MPC decision:

        u0 =
        [
            W_F_L6,
            W_M_L6,
            W_F_R6,
            W_M_R6
        ]

    All MPC wrench components are expressed in World frame.

    This helper rotates them into the corrected foot contact
    frames:

        L6_IK
        R6_IK

    for diagnostics only.

    No wrench-reference-point translation is required because:

        p_L6 = p_L6_IK

        p_R6 = p_R6_IK
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
        R_M
    )


# ============================================================
# IK DIAGNOSTIC HELPERS
# ============================================================

def get_diagnostic_value(
    diagnostics,
    key,
    default=None
):
    if diagnostics is None:
        return default

    return diagnostics.get(
        key,
        default
    )


def get_error_norm(
    diagnostics,
    norm_key,
    vector_key
):
    """
    Support both current and older diagnostic dictionaries.
    """

    if diagnostics is None:
        return None

    if norm_key in diagnostics:

        value = diagnostics[
            norm_key
        ]

        if value is None:
            return None

        return float(
            value
        )

    if vector_key in diagnostics:

        value = diagnostics[
            vector_key
        ]

        if value is None:
            return None

        value = np.asarray(
            value,
            dtype=float
        ).reshape(-1)

        return float(
            np.linalg.norm(
                value
            )
        )

    return None


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
    reference_generator,
    mass_CoM
):
    """
    Print one walking-control diagnostic block.
    """

    (
        L_F,
        L_M,
        R_F,
        R_M

    ) = get_contact_frame_wrench(

        u0_MPC=held_u0_MPC,

        L6_IK_R_W=(
            current.L6_IK_R_W
        ),

        R6_IK_R_W=(
            current.R6_IK_R_W
        )
    )


    mg = float(
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
        "mode = WALKING"
    )

    print(
        "t_gait =",
        f"{control_time:.4f}",
        "s"
    )


    # ========================================================
    # CONTACT
    # ========================================================

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


    # ========================================================
    # ROBOT STATE
    # ========================================================

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


    # ========================================================
    # WORLD-FRAME MPC WRENCH
    # ========================================================

    print(
        "\nu0_MPC [World frame] ="
    )

    print(
        held_u0_MPC
    )


    # ========================================================
    # CONTACT-FRAME WRENCH
    # ========================================================

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


    # ========================================================
    # STANCE LOAD DIAGNOSTIC
    # ========================================================

    if np.array_equal(
        current_contact,
        DOUBLE_SUPPORT
    ):

        print(
            "\ncontact mode = DOUBLE SUPPORT"
        )

        print(
            "L local Fz =",
            L_F[
                2
            ],
            "N"
        )

        print(
            "R local Fz =",
            R_F[
                2
            ],
            "N"
        )


    elif np.array_equal(
        current_contact,
        LEFT_SINGLE_SUPPORT
    ):

        print(
            "\ncontact mode = LEFT SINGLE SUPPORT"
        )

        print(
            "STANCE foot = L"
        )

        print(
            "STANCE local Fz =",
            L_F[
                2
            ],
            "N"
        )

        print(
            "SWING local Fz =",
            R_F[
                2
            ],
            "N"
        )

        print(
            "m*g =",
            mg,
            "N"
        )


    elif np.array_equal(
        current_contact,
        RIGHT_SINGLE_SUPPORT
    ):

        print(
            "\ncontact mode = RIGHT SINGLE SUPPORT"
        )

        print(
            "STANCE foot = R"
        )

        print(
            "STANCE local Fz =",
            R_F[
                2
            ],
            "N"
        )

        print(
            "SWING local Fz =",
            L_F[
                2
            ],
            "N"
        )

        print(
            "m*g =",
            mg,
            "N"
        )


    # --------------------------------------------------------
    # World vertical-force diagnostic
    # --------------------------------------------------------

    total_Fz_W = float(
        held_u0_MPC[
            2
        ]
        +
        held_u0_MPC[
            8
        ]
    )

    print(
        "\nFz_L + Fz_R [World] =",
        total_Fz_W,
        "N"
    )

    print(
        "m*g =",
        mg,
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
    # SWING
    # ========================================================

    if last_swing_result is not None:

        diagnostics = (
            last_swing_result.ik_diagnostics
        )

        position_error_norm = (
            get_error_norm(
                diagnostics,
                "final_position_error_norm",
                "final_position_error"
            )
        )

        orientation_error_norm = (
            get_error_norm(
                diagnostics,
                "final_orientation_error_norm",
                "final_orientation_error"
            )
        )

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
            "\nq_swing =",
            last_swing_result.q_swing
        )

        print(
            "q_swing_des =",
            last_swing_result.q_swing_des
        )

        print(
            "\nIK site =",
            get_diagnostic_value(
                diagnostics,
                "site_name"
            )
        )

        print(
            "IK converged =",
            get_diagnostic_value(
                diagnostics,
                "converged"
            )
        )

        print(
            "IK iterations =",
            get_diagnostic_value(
                diagnostics,
                "iterations"
            )
        )

        print(
            "IK method =",
            get_diagnostic_value(
                diagnostics,
                "method_used"
            )
        )

        print(
            "IK sigma_min =",
            get_diagnostic_value(
                diagnostics,
                "sigma_min"
            )
        )

        print(
            "IK position error norm =",
            position_error_norm
        )

        print(
            "IK orientation error norm =",
            orientation_error_norm
        )

        print(
            "IK desired roll =",
            get_diagnostic_value(
                diagnostics,
                "roll_swing_des"
            )
        )

        print(
            "IK desired pitch =",
            get_diagnostic_value(
                diagnostics,
                "pitch_swing_des"
            )
        )


    # ========================================================
    # QP
    # ========================================================

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


    # ========================================================
    # PREDICTED CONTACT HORIZON
    # ========================================================

    print(
        "\nMPC horizon first contact =",
        last_contact_schedule[
            0
        ]
    )

    print(
        "MPC contact horizon ="
    )

    print(
        last_contact_schedule
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
    # Single-support duration = swing duration
    # --------------------------------------------------------

    T_SWING = (
        N_SS
        *
        dt_mpc
    )


    # --------------------------------------------------------
    # Stance duration used by the HECTOR-style Raibert term:
    #
    #     1/2 * v_CoM * T_stance
    #
    # For the current symmetric gait:
    #
    #     T_STANCE = T_SWING = N_SS * dt_mpc
    # --------------------------------------------------------

    T_STANCE = (
        N_SS
        *
        dt_mpc
    )


    # --------------------------------------------------------
    # Physics steps per low-level update
    # --------------------------------------------------------

    LOW_LEVEL_EVERY = (
        validate_timing_multiple(

            larger_dt=DT_LOW,

            smaller_dt=DT_SIM,

            name="DT_LOW / DT_SIM"
        )
    )


    # --------------------------------------------------------
    # Physics steps per MPC update
    # --------------------------------------------------------

    MPC_EVERY = (
        validate_timing_multiple(

            larger_dt=dt_mpc_update,

            smaller_dt=DT_SIM,

            name="dt_mpc_update / DT_SIM"
        )
    )


    # --------------------------------------------------------
    # MPC tick must coincide with low-level tick
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
    # Low-level updates per swing
    # --------------------------------------------------------

    SWING_LOW_LEVEL_UPDATES = (
        validate_timing_multiple(

            larger_dt=T_SWING,

            smaller_dt=DT_LOW,

            name="T_SWING / DT_LOW"
        )
    )


    # ========================================================
    # 3. SWING TRAJECTORY
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
    # First MPC wrench:
    #
    # [
    #     W_F_L
    #     W_M_L
    #     W_F_R
    #     W_M_R
    # ]
    #
    # held by ZOH between MPC updates.
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


    # --------------------------------------------------------
    # Contact state
    # --------------------------------------------------------

    current_contact = (
        DOUBLE_SUPPORT.copy()
    )

    swing_foot = None

    delta_T = None


    # --------------------------------------------------------
    # Last swing result
    # --------------------------------------------------------

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

    logger = (
        SimulationLogger()
    )


    physics_step = 0

    mpc_iteration = 0

    low_level_iteration = 0


    # --------------------------------------------------------
    # IK statistics
    # --------------------------------------------------------

    ik_swing_updates = 0

    ik_nonconverged_updates = 0

    max_ik_position_error = 0.0

    max_ik_orientation_error = 0.0


    # ========================================================
    # 6. INITIAL FRAME DIAGNOSTIC
    # ========================================================

    initial_current = (
        get_current_state_mpc(
            model,
            data
        )
    )


    # ========================================================
    # 6A. HOME GEOMETRY FOR HECTOR-STYLE RAIBERT
    #
    # The new foothold planner does not use a fixed World-frame:
    #
    #     W_p_L6_ref
    #     W_p_R6_ref
    #
    # as the nominal foothold base point.
    #
    # Instead it uses:
    #
    #     W_p_CoM + W_R_B @ B_p_foot_offset
    #
    # where the left/right nominal offsets are identified once
    # from the home configuration:
    #
    #     B_p_foot_offset
    #
    #       =
    #
    #     W_R_B_home.T
    #
    #       @
    #
    #     (W_p_foot_home - W_p_CoM_home)
    #
    # ========================================================

    W_p_CoM_home = (
        initial_current.x[
            0:3
        ].copy()
    )


    W_R_B_home = (
        get_W_R_B(
            model,
            data
        )
    )


    W_p_L6_home = (
        get_W_position_L6(
            model,
            data
        )
    )


    W_p_R6_home = (
        get_W_position_R6(
            model,
            data
        )
    )


    B_p_L6_offset = (

        W_R_B_home.T

        @

        (
            W_p_L6_home
            -
            W_p_CoM_home
        )

    )


    B_p_R6_offset = (

        W_R_B_home.T

        @

        (
            W_p_R6_home
            -
            W_p_CoM_home
        )

    )


    # --------------------------------------------------------
    # Physical touchdown heights of the L6/R6 sites.
    #
    # The terrain plane may be z = 0 while the physical L6/R6
    # sites have a small geometric offset from that plane.
    #
    # Therefore preserve the actual home site heights.
    # --------------------------------------------------------

    L6_TOUCHDOWN_HEIGHT = float(
        W_p_L6_home[
            2
        ]
    )


    R6_TOUCHDOWN_HEIGHT = float(
        W_p_R6_home[
            2
        ]
    )


    # ========================================================
    # 7. PRINT CONFIGURATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WALKING SIMULATION - SRBD MPC")
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
        "T_STANCE =",
        T_STANCE,
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
        "QP post-check tolerance =",
        QP_CONSTRAINT_TOLERANCE
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


    # --------------------------------------------------------
    # Frame diagnostic
    # --------------------------------------------------------

    print(
        "\nW_p_CoM_home =",
        W_p_CoM_home
    )

    print(
        "W_p_L6_home =",
        W_p_L6_home
    )

    print(
        "W_p_R6_home =",
        W_p_R6_home
    )

    print(
        "\nW_R_B_home ="
    )

    print(
        W_R_B_home
    )

    print(
        "\nB_p_L6_offset =",
        B_p_L6_offset
    )

    print(
        "B_p_R6_offset =",
        B_p_R6_offset
    )

    print(
        "\nL6 touchdown height =",
        L6_TOUCHDOWN_HEIGHT,
        "m"
    )

    print(
        "R6 touchdown height =",
        R6_TOUCHDOWN_HEIGHT,
        "m"
    )


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
        "\nL6_IK_R_W used by MPC moment/CoP constraint ="
    )

    print(
        initial_current.L6_IK_R_W
    )


    print(
        "\nR6_IK_R_W used by MPC moment/CoP constraint ="
    )

    print(
        initial_current.R6_IK_R_W
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "MPC decision wrench remains expressed in World frame."
    )

    print(
        "Moment/CoP constraint uses L6_IK/R6_IK frames."
    )

    print(
        "Moment arms still use physical L6/R6 positions."
    )

    print(
        "Wrench -> joint torque still uses physical "
        "L6/R6 Jacobians."
    )


    # ========================================================
    # 8. MAIN SIMULATION LOOP
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


            # No separate standing phase.
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
            # Shared state snapshot
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
                # A2. CONTACT SCHEDULE
                #
                # Example at t = 0:
                #
                #     DS
                #     DS
                #     LSS
                #     ...
                #
                # so MPC anticipates DS -> SS transition.
                # =============================================

                contact_schedule = (
                    get_contact_schedule(

                        current_time=t_gait,

                        N=N,

                        dt_mpc=dt_mpc
                    )
                )


                # =============================================
                # A3. REFERENCE
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
                # A4. SRBD PREDICTION MODEL
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
                #
                # IMPORTANT
                # ---------
                #
                # QP API still calls its arguments:
                #
                #     L6_R_W
                #     R6_R_W
                #
                # but here we deliberately pass:
                #
                #     current.L6_IK_R_W
                #     current.R6_IK_R_W
                #
                # Therefore:
                #
                #     Fz_foot
                #     My_foot
                #
                # for moment / CoP constraints are extracted
                # in the corrected contact frame.
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
                # A6. INVALID MPC SOLUTION
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
                        "t_gait =",
                        t_gait,
                        "s"
                    )

                    print(
                        "MPC iteration =",
                        mpc_iteration
                    )

                    print(
                        "\ncurrent state ="
                    )

                    print(
                        x_k
                    )

                    print(
                        "\ncontact_schedule ="
                    )

                    print(
                        contact_schedule
                    )

                    print(
                        "\nL6_IK_R_W ="
                    )

                    print(
                        current.L6_IK_R_W
                    )

                    print(
                        "\nR6_IK_R_W ="
                    )

                    print(
                        current.R6_IK_R_W
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
                # A7. HOLD FIRST MPC WRENCH
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


                W_p_dot_CoM = (
                    current.x[
                        6:9
                    ].copy()
                )


                W_p_dot_CoM_ref = (
                    setup.reference_generator.get_velocity_reference()
                )


                # =============================================
                # B1A. CURRENT BODY POSE FOR RAIBERT
                # =============================================

                W_p_CoM = (
                    current.x[
                        0:3
                    ].copy()
                )


                W_R_B = (
                    get_W_R_B(
                        model,
                        data
                    )
                )


                # =============================================
                # B2. CURRENT CONTACT MODE
                #
                # Same gait scheduler as MPC.
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
                # B3. MPC WRENCH -> JOINT TORQUE
                #
                # IMPORTANT:
                #
                # The wrench remains:
                #
                #     World-frame wrench at physical L6/R6.
                #
                # Mapping remains:
                #
                #     tau = -J^T W
                #
                # using the physical L6/R6 Jacobians.
                #
                # This part is NOT changed by the new
                # L6_IK/R6_IK contact orientation frames.
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
                # Both legs:
                #
                #     MPC wrench control
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


                    # Prepare next liftoff point.
                    swing_trajectory.reset()


                    last_swing_result = None


                # ---------------------------------------------
                # LEFT SINGLE SUPPORT
                #
                #     [1, 0]
                #
                # L = stance = MPC
                #
                # R = swing =
                #
                #     Raibert
                #       ->
                #     Bezier
                #       ->
                #     5D IK
                #       ->
                #     joint PD
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
                    # Physical current foot positions
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
                    # Swing controller
                    #
                    # Right foot is swing.
                    #
                    # HECTOR-style Raibert uses:
                    #
                    #     current CoM position
                    #     current body orientation
                    #     nominal body-frame foot offsets
                    #     remaining swing time
                    #     stance duration
                    #     CoM velocity feedback
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

                            W_p_CoM=(
                                W_p_CoM
                            ),

                            W_R_B=(
                                W_R_B
                            ),

                            B_p_L6_offset=(
                                B_p_L6_offset
                            ),

                            B_p_R6_offset=(
                                B_p_R6_offset
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

                            T_stance=(
                                T_STANCE
                            ),

                            k_d=(
                                K_D_RAIBERT
                            ),

                            ground_height=(
                                R6_TOUCHDOWN_HEIGHT
                            )
                        )
                    )


                    # Left = stance
                    tau_L = (
                        tau_L_MPC.copy()
                    )


                    # Right = swing
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
                # L = swing
                #
                # R = stance
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
                    # Physical current foot positions
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
                    # Swing controller
                    #
                    # Left foot is swing.
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

                            W_p_CoM=(
                                W_p_CoM
                            ),

                            W_R_B=(
                                W_R_B
                            ),

                            B_p_L6_offset=(
                                B_p_L6_offset
                            ),

                            B_p_R6_offset=(
                                B_p_R6_offset
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

                            T_stance=(
                                T_STANCE
                            ),

                            k_d=(
                                K_D_RAIBERT
                            ),

                            ground_height=(
                                L6_TOUCHDOWN_HEIGHT
                            )
                        )
                    )


                    # Left = swing
                    tau_L = (
                        swing_result.tau_swing.copy()
                    )


                    # Right = stance
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
                # B5. IK STATISTICS
                # =============================================

                if last_swing_result is not None:

                    ik_swing_updates += 1

                    diagnostics = (
                        last_swing_result.ik_diagnostics
                    )


                    converged = bool(
                        get_diagnostic_value(
                            diagnostics,
                            "converged",
                            False
                        )
                    )


                    if not converged:

                        ik_nonconverged_updates += 1


                    position_error_norm = (
                        get_error_norm(
                            diagnostics,
                            "final_position_error_norm",
                            "final_position_error"
                        )
                    )


                    orientation_error_norm = (
                        get_error_norm(
                            diagnostics,
                            "final_orientation_error_norm",
                            "final_orientation_error"
                        )
                    )


                    if position_error_norm is not None:

                        max_ik_position_error = max(
                            max_ik_position_error,
                            position_error_norm
                        )


                    if orientation_error_norm is not None:

                        max_ik_orientation_error = max(
                            max_ik_orientation_error,
                            orientation_error_norm
                        )


                # =============================================
                # B6. BUILD FULL TORQUE COMMAND
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
                # B7. APPLY TORQUE
                #
                # No manual torque clipping.
                #
                # MuJoCo actuator ctrlrange is the actuator
                # saturation source of truth.
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


                # =============================================
                # C1. LOGGER
                # =============================================

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
                # C2. STATUS
                # =============================================

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
    # 9. FINALIZE LOGGER
    # ========================================================

    logger.finalize()


    # ========================================================
    # 10. FINAL STATE
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


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # IK statistics
    # --------------------------------------------------------

    print(
        "\nIK swing updates =",
        ik_swing_updates
    )

    print(
        "IK non-converged updates =",
        ik_nonconverged_updates
    )


    if ik_swing_updates > 0:

        print(
            "IK non-converged ratio =",
            ik_nonconverged_updates
            /
            ik_swing_updates
        )


    print(
        "Maximum IK position error norm =",
        max_ik_position_error,
        "m"
    )

    print(
        "Maximum IK orientation error norm =",
        max_ik_orientation_error,
        "rad"
    )


    # --------------------------------------------------------
    # Logger
    # --------------------------------------------------------

    logger.print_summary()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()