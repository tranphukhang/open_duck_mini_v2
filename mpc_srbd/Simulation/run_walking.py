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
DCM_DIR = MPC_SRBD_DIR / "DCM"
TRAJECTORY_DIR = MPC_SRBD_DIR / "Trajectory_Planner"

for directory in (
    MPC_SRBD_DIR,
    SIMULATION_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
    COMMAND_DIR,
    DCM_DIR,
    TRAJECTORY_DIR,
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
from gait_manager import (
    initialize_gait_manager,
    update_gait_manager,
)
from swing_low_level_controller import (
    compute_swing_low_level_control,
    reset_swing_low_level_state,
)


# ============================================================
# STATE
# ============================================================

from get_current_state_mpc import get_current_state_mpc
from get_contact_state import get_contact_state


# ============================================================
# COMMAND / GAIT / REFERENCE
# ============================================================

from Command.gait_timing import build_gait_timing
from Command.contact_schedule import (
    PHASE_LSS,
    PHASE_DS_L_TO_R,
    PHASE_RSS,
    PHASE_DS_R_TO_L,
    build_contact_preview,
)
from Command.foothold_generator import (
    HOME_NOMINAL_STEP_WIDTH,
    generate_foothold,
    generate_nominal_foothold,
    get_opposite_foot,
)
from Command.foot_plan_horizon import (
    build_foot_plan_horizon,
    build_moment_arm_horizon,
    get_initial_support_foot,
    get_support_foot_from_index,
)
from Command.srbd_reference import build_srbd_reference


# ============================================================
# DCM
# ============================================================

from DCM.mpc_reference import build_mpc_translational_reference


# ============================================================
# SRBD
# ============================================================

from build_SRBD_model import build_SRBD_model_sequence


# ============================================================
# QP
# ============================================================

from build_and_solve_QP import build_and_solve_QP


# ============================================================
# LOW-LEVEL CONTROL
# ============================================================

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench,
)
from actuator_mapping import apply_joint_torque
from Actuator.swing_joint_PD import compute_swing_joint_PD
from Actuator.swing_joint_PD_params import (
    KP_SWING,
    KD_SWING,
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=7,
    suppress=True,
    linewidth=220,
)


# ============================================================
# XML
# ============================================================

XML_PATH = REPO_ROOT / "xmls" / "scene_flat_terrian_v3.xml"


# ============================================================
# WALKING PARAMETERS
# ============================================================

N = 10

T_STEP = 0.5
N_SS = 8
N_DS = 2

TIMING = build_gait_timing(
    T_step=T_STEP,
    horizon=N,
    N_SS=N_SS,
    N_DS=N_DS,
)

DT_MPC = float(
    TIMING["dt_mpc"]
)

# ALL high-level logic:
#   contact sample
#   FSM
#   phase/contact preview
#   foot-plan update
#   DCM
#   IK reference
#   SRBD/QP
#
# runs at 100 Hz.
DT_HIGH_LEVEL = 0.01

T_SS = float(
    TIMING["T_SS"]
)

T_DS = float(
    TIMING["T_DS"]
)

# Full first step.
#
# Robot is physically initialized in DS.
#
# FIRST_SWING_FOOT = "R":
#   initial phase = DS_R_TO_L
#   then LO_R -> LSS
#
# FIRST_SWING_FOOT = "L":
#   initial phase = DS_L_TO_R
#   then LO_L -> RSS
FIRST_SWING_FOOT = "R"

STEP_LENGTH = 0.05
STEP_WIDTH = HOME_NOMINAL_STEP_WIDTH
SWING_HEIGHT = 0.04
GROUND_HEIGHT = 0.0

SUPPORT_PLAN_LENGTH = 10

SIM_DURATION = 5.0

GRAVITY = 9.81

QP_MAX_ITER = 10000
QP_CONSTRAINT_TOLERANCE = 3e-4

PRINT_EVERY_N_HIGH_LEVEL = 20


# ============================================================
# SWING LEG JOINT ORDER
# ============================================================

SWING_LEG_JOINT_NAMES = {
    "L": (
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee",
        "L_ankle",
    ),
    "R": (
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee",
        "R_ankle",
    ),
}


# ============================================================
# BASIC HELPERS
# ============================================================

def get_site_id(
    model,
    site_name,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise RuntimeError(
            f"Không tìm thấy site '{site_name}' trong MuJoCo model."
        )

    return int(
        site_id
    )


def get_foot_positions(
    data,
    L6_site_id,
    R6_site_id,
):
    W_p_L6 = np.asarray(
        data.site_xpos[
            L6_site_id
        ],
        dtype=float,
    ).reshape(3).copy()

    W_p_R6 = np.asarray(
        data.site_xpos[
            R6_site_id
        ],
        dtype=float,
    ).reshape(3).copy()

    if not np.all(
        np.isfinite(
            W_p_L6
        )
    ):
        raise RuntimeError(
            "W_p_L6 chứa NaN hoặc Inf."
        )

    if not np.all(
        np.isfinite(
            W_p_R6
        )
    ):
        raise RuntimeError(
            "W_p_R6 chứa NaN hoặc Inf."
        )

    return (
        W_p_L6,
        W_p_R6,
    )


def get_leg_joint_state(
    model,
    data,
    foot,
):
    """
    Read current joint state only.

    This helper contains no control logic and no mj_step().
    """
    foot = str(
        foot
    ).upper()

    if foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "foot phải là 'L' hoặc 'R'."
        )

    q = np.zeros(
        5,
        dtype=float,
    )

    q_dot = np.zeros(
        5,
        dtype=float,
    )

    for index, joint_name in enumerate(
        SWING_LEG_JOINT_NAMES[
            foot
        ]
    ):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"Không tìm thấy joint '{joint_name}'."
            )

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        dof_address = int(
            model.jnt_dofadr[
                joint_id
            ]
        )

        q[index] = data.qpos[
            qpos_address
        ]

        q_dot[index] = data.qvel[
            dof_address
        ]

    if not (
        np.all(
            np.isfinite(
                q
            )
        )
        and
        np.all(
            np.isfinite(
                q_dot
            )
        )
    ):
        raise RuntimeError(
            "Swing joint state chứa NaN hoặc Inf."
        )

    return (
        q,
        q_dot,
    )


def rotation_matrix_from_rpy(
    euler,
):
    """
    World-from-body ZYX rotation using [roll, pitch, yaw].
    """
    roll, pitch, yaw = np.asarray(
        euler,
        dtype=float,
    ).reshape(3)

    cr = np.cos(
        roll
    )
    sr = np.sin(
        roll
    )
    cp = np.cos(
        pitch
    )
    sp = np.sin(
        pitch
    )
    cy = np.cos(
        yaw
    )
    sy = np.sin(
        yaw
    )

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )

    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return (
        Rz
        @
        Ry
        @
        Rx
    )


def get_kinematic_reference(
    current_state,
    swing_foot,
    home_CoM,
    home_L6,
    home_R6,
):
    """
    Home-equivalent swing-foot position for the current body pose.

    This helper is unchanged from the previous planner architecture.
    """
    current_state = np.asarray(
        current_state,
        dtype=float,
    ).reshape(13)

    W_p_CoM = (
        current_state[
            0:3
        ]
    )

    euler = (
        current_state[
            3:6
        ]
    )

    if swing_foot == "L":
        home_offset = (
            home_L6
            -
            home_CoM
        )

    elif swing_foot == "R":
        home_offset = (
            home_R6
            -
            home_CoM
        )

    else:
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    R_W_B = (
        rotation_matrix_from_rpy(
            euler
        )
    )

    return (
        W_p_CoM
        +
        R_W_B
        @
        home_offset
    )


def mix_walking_joint_torque(
    tau_mpc,
    swing_foot=None,
    tau_swing=None,
):
    """
    Torque architecture:

        LSS:
            [tau_L_MPC, tau_R_PD]

        RSS:
            [tau_L_PD, tau_R_MPC]

        DS:
            [tau_L_MPC, tau_R_MPC]
    """
    tau_mpc = np.asarray(
        tau_mpc,
        dtype=float,
    ).reshape(-1)

    if tau_mpc.shape != (
        10,
    ):
        raise ValueError(
            "tau_mpc phải có shape (10,)."
        )

    if swing_foot is None:
        return (
            tau_mpc.copy()
        )

    tau_swing = np.asarray(
        tau_swing,
        dtype=float,
    ).reshape(-1)

    if tau_swing.shape != (
        5,
    ):
        raise ValueError(
            "tau_swing phải có shape (5,)."
        )

    tau = (
        tau_mpc.copy()
    )

    if swing_foot == "L":
        tau[0:5] = (
            tau_swing
        )

    elif swing_foot == "R":
        tau[5:10] = (
            tau_swing
        )

    else:
        raise ValueError(
            "swing_foot phải là 'L', 'R' hoặc None."
        )

    return tau


# ============================================================
# GAIT INITIALIZATION
# ============================================================

def get_initial_ds_phase(
    first_swing_foot,
):
    """
    Robot starts physically in double support.

    No separate walking-startup FSM is used.

    Right foot is the first swing foot:
        DS_R_TO_L -> LSS

    Left foot is the first swing foot:
        DS_L_TO_R -> RSS
    """
    first_swing_foot = str(
        first_swing_foot
    ).upper()

    if first_swing_foot == "R":
        return (
            PHASE_DS_R_TO_L
        )

    if first_swing_foot == "L":
        return (
            PHASE_DS_L_TO_R
        )

    raise ValueError(
        "FIRST_SWING_FOOT phải là 'L' hoặc 'R'."
    )


# ============================================================
# SUPPORT PLAN HELPERS
# ============================================================

def ensure_support_plan_length(
    support_points,
    current_phase,
    target_length=SUPPORT_PLAN_LENGTH,
):
    """
    Extend ordered flat-ground support points with the existing nominal
    foothold generator.

    No correction law is introduced here.
    """
    points = [
        np.asarray(
            point,
            dtype=float,
        ).reshape(3).copy()

        for point in support_points
    ]

    if len(
        points
    ) < 2:
        raise ValueError(
            "support_points cần ít nhất 2 điểm."
        )

    initial_support_foot = (
        get_initial_support_foot(
            current_phase
        )
    )

    while len(
        points
    ) < int(
        target_length
    ):
        last_index = (
            len(points) - 1
        )

        last_foot = (
            get_support_foot_from_index(
                initial_support_foot,
                last_index,
            )
        )

        next_foot = (
            get_opposite_foot(
                last_foot
            )
        )

        next_point = (
            generate_nominal_foothold(
                W_p_support=(
                    points[-1]
                ),
                swing_foot=next_foot,
                step_length=STEP_LENGTH,
                step_width=STEP_WIDTH,
                ground_height=GROUND_HEIGHT,
            )
        )

        points.append(
            next_point
        )

    return points


def build_initial_support_plan(
    initial_phase,
    first_swing_foot,
    W_p_L6,
    W_p_R6,
    current_state,
    home_CoM,
    home_L6,
    home_R6,
):
    """
    Build the support queue directly from the physical initial DS geometry.

    There is NO half-step and NO startup trajectory.

    Right-first:
        DS_R_TO_L
        [R_current, L_current, R_full_step_target, ...]

    Left-first:
        DS_L_TO_R
        [L_current, R_current, L_full_step_target, ...]
    """
    if first_swing_foot == "R":
        if initial_phase != PHASE_DS_R_TO_L:
            raise RuntimeError(
                "Right-first walking phải bắt đầu bằng DS_R_TO_L."
            )

        points = [
            W_p_R6.copy(),
            W_p_L6.copy(),
        ]

        W_p_support = (
            W_p_L6
        )

    elif first_swing_foot == "L":
        if initial_phase != PHASE_DS_L_TO_R:
            raise RuntimeError(
                "Left-first walking phải bắt đầu bằng DS_L_TO_R."
            )

        points = [
            W_p_L6.copy(),
            W_p_R6.copy(),
        ]

        W_p_support = (
            W_p_R6
        )

    else:
        raise ValueError(
            "first_swing_foot phải là 'L' hoặc 'R'."
        )

    W_p_kinematic_reference = (
        get_kinematic_reference(
            current_state=current_state,
            swing_foot=first_swing_foot,
            home_CoM=home_CoM,
            home_L6=home_L6,
            home_R6=home_R6,
        )
    )

    first_target, diagnostics = (
        generate_foothold(
            W_p_support=W_p_support,
            W_p_kinematic_reference=(
                W_p_kinematic_reference
            ),
            swing_foot=first_swing_foot,
            step_length=STEP_LENGTH,
            step_width=STEP_WIDTH,
            correction_xy=None,
            ground_height=GROUND_HEIGHT,
        )
    )

    points.append(
        np.asarray(
            first_target,
            dtype=float,
        ).reshape(3).copy()
    )

    points = (
        ensure_support_plan_length(
            points,
            current_phase=initial_phase,
            target_length=SUPPORT_PLAN_LENGTH,
        )
    )

    return (
        points,
        diagnostics,
    )


def get_gait_foothold_inputs(
    current_phase,
    support_points,
):
    """
    Swing targets supplied to gait_manager.

    In DS:
        index 2 is the target for the foot that will swing after LO.

    In SS:
        index 1 is the current swing-foot touchdown target.
    """
    points = [
        np.asarray(
            point,
            dtype=float,
        ).reshape(3).copy()

        for point in support_points
    ]

    if len(
        points
    ) < 3:
        raise ValueError(
            "support_points cần ít nhất 3 điểm."
        )

    if current_phase == PHASE_LSS:
        return (
            None,
            points[1].copy(),
        )

    if current_phase == PHASE_RSS:
        return (
            points[1].copy(),
            None,
        )

    if current_phase == PHASE_DS_L_TO_R:
        return (
            points[2].copy(),
            None,
        )

    if current_phase == PHASE_DS_R_TO_L:
        return (
            None,
            points[2].copy(),
        )

    raise ValueError(
        f"Unsupported current_phase: {current_phase}"
    )


def update_support_queue_after_transition(
    support_points,
    gait_output,
):
    """
    Queue topology update only.

    TD:
        no index shift.
        The 100-Hz foot feedback re-anchor updates realized DS geometry.

    LO:
        outgoing support is removed.
        Incoming support becomes support_points[0].

    No extra foothold correction is added here.
    """
    points = [
        np.asarray(
            point,
            dtype=float,
        ).reshape(3).copy()

        for point in support_points
    ]

    event = (
        gait_output.transition_event
    )

    if event in (
        "LO_L",
        "LO_R",
    ):
        if len(
            points
        ) < 3:
            raise RuntimeError(
                "Support queue quá ngắn tại liftoff."
            )

        points.pop(
            0
        )

    points = (
        ensure_support_plan_length(
            points,
            current_phase=(
                gait_output.current_phase
            ),
            target_length=(
                SUPPORT_PLAN_LENGTH
            ),
        )
    )

    return points


# ============================================================
# MPC
# ============================================================

def solve_ltv_mpc(
    setup,
    current,
    W_p_CoM_ref,
    W_v_CoM_ref,
    W_p_L6_plan,
    W_p_R6_plan,
    contact_schedule,
):
    srbd_reference = (
        build_srbd_reference(
            W_p_CoM_ref=W_p_CoM_ref,
            W_v_CoM_ref=W_v_CoM_ref,
            W_euler_CoM_ref=np.zeros(
                3,
                dtype=float,
            ),
            W_w_CoM_ref=np.zeros(
                3,
                dtype=float,
            ),
        )
    )

    (
        W_r_L_sequence,
        W_r_R_sequence,
    ) = build_moment_arm_horizon(
        W_p_CoM_ref=(
            srbd_reference[
                "W_p_CoM_ref"
            ]
        ),
        W_p_L6_plan=W_p_L6_plan,
        W_p_R6_plan=W_p_R6_plan,
    )

    Phi, Gamma = (
        build_SRBD_model_sequence(
            Ew_inverse=(
                current.Ew_inverse
            ),
            mass_CoM=(
                setup.mass_CoM
            ),
            W_inertia_CoM=(
                setup.W_inertia_CoM_q0
            ),
            W_r_L_sequence=(
                W_r_L_sequence
            ),
            W_r_R_sequence=(
                W_r_R_sequence
            ),
            dt_mpc=DT_MPC,
            horizon=N,
        )
    )

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
            current.x
        ),
        X_ref=(
            srbd_reference[
                "X_ref"
            ]
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
        print_diagnostics=False,
    )

    return (
        U_opt,
        u0_opt,
        solver_info,
        post_diagnostics,
        srbd_reference,
    )


# ============================================================
# STATUS
# ============================================================

def print_status(
    high_level_iteration,
    data,
    gait_output,
    current,
    W_p_CoM_ref,
    held_u0_MPC,
    held_tau_mpc,
    tau_command,
    solver_info,
    post_diagnostics,
    support_points,
    contact_schedule,
):
    max_violation = None

    if post_diagnostics is not None:
        max_violation = (
            post_diagnostics.get(
                "max_violation"
            )
        )

    print(
        "\n"
        +
        "-" * 78
    )

    print(
        f"t={data.time:.4f} s | "
        f"HL={high_level_iteration} | "
        f"phase={gait_output.current_phase} | "
        f"t_phase={gait_output.t_phase:.4f}"
    )

    print(
        f"contact=[{gait_output.c_L},{gait_output.c_R}] | "
        f"waiting={gait_output.waiting_for_contact} | "
        f"swing={gait_output.swing_foot} | "
        f"swing_phase={gait_output.swing_phase}"
    )

    print(
        "contact preview first rows =\n",
        np.asarray(
            contact_schedule[
                :min(
                    4,
                    len(
                        contact_schedule
                    ),
                )
            ],
            dtype=int,
        ),
    )

    print(
        "CoM actual =",
        current.x[
            0:3
        ],
    )

    print(
        "CoM ref[0] =",
        W_p_CoM_ref[
            0
        ],
    )

    print(
        "Fz current [L,R] =",
        np.array(
            [
                held_u0_MPC[2],
                held_u0_MPC[8],
            ]
        ),
        "N",
    )

    print(
        "held tau_mpc =",
        held_tau_mpc,
    )

    print(
        "tau command  =",
        tau_command,
    )

    print(
        "QP status =",
        solver_info.get(
            "status"
        ),
        "| solve_time =",
        solver_info.get(
            "solve_time"
        ),
        "| max_violation =",
        max_violation,
    )

    print(
        "support queue first points =\n",
        np.asarray(
            support_points[
                :min(
                    4,
                    len(
                        support_points
                    ),
                )
            ],
            dtype=float,
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Simulation setup
    # --------------------------------------------------------
    setup = initialize_simulation(
        xml_path=XML_PATH,
        horizon=N,
        dt_mpc=DT_MPC,
        dt_mpc_update=DT_HIGH_LEVEL,
        home_key_name="home",
        qp_verbose=False,
        qp_eps_abs=1e-5,
        qp_eps_rel=1e-5,
        qp_max_iter=QP_MAX_ITER,
        qp_polishing=True,
        qp_enable_warm_start=True,
        qp_constraint_tolerance=(
            QP_CONSTRAINT_TOLERANCE
        ),
        print_initialization=True,
    )

    model = setup.model
    data = setup.data

    DT_SIM = float(
        setup.dt_sim
    )

    # --------------------------------------------------------
    # 2. Physical foot sites / home geometry
    # --------------------------------------------------------
    L6_site_id = get_site_id(
        model,
        "L6",
    )

    R6_site_id = get_site_id(
        model,
        "R6",
    )

    (
        W_p_L6_home,
        W_p_R6_home,
    ) = get_foot_positions(
        data,
        L6_site_id,
        R6_site_id,
    )

    home_CoM = (
        setup.x_initial[
            0:3
        ].copy()
    )

    home_L6 = (
        W_p_L6_home.copy()
    )

    home_R6 = (
        W_p_R6_home.copy()
    )

    COM_HEIGHT = float(
        home_CoM[
            2
        ]
    )

    # --------------------------------------------------------
    # 3. Initial physical state/contact
    # --------------------------------------------------------
    current = (
        get_current_state_mpc(
            model,
            data,
        )
    )

    contact_state = (
        get_contact_state(
            model=model,
            data=data,
            contact_map=(
                setup.contact_map
            ),
        )
    )

    c_L = int(
        contact_state[
            "c_L"
        ]
    )

    c_R = int(
        contact_state[
            "c_R"
        ]
    )

    (
        W_p_L6,
        W_p_R6,
    ) = get_foot_positions(
        data,
        L6_site_id,
        R6_site_id,
    )

    if (
        c_L,
        c_R,
    ) != (
        1,
        1,
    ):
        raise RuntimeError(
            "run_walking.py mới yêu cầu robot khởi tạo vật lý ở DS: "
            f"contact=[{c_L},{c_R}], expected=[1,1]."
        )

    # --------------------------------------------------------
    # 4. Direct DS initialization
    # --------------------------------------------------------
    initial_phase = (
        get_initial_ds_phase(
            FIRST_SWING_FOOT
        )
    )

    gait_state = (
        initialize_gait_manager(
            initial_phase=(
                initial_phase
            ),
            c_L=c_L,
            c_R=c_R,
            W_p_L6=W_p_L6,
            W_p_R6=W_p_R6,
            t_phase=0.0,
            strict_contact=True,
        )
    )

    (
        support_points,
        last_foothold_diagnostics,
    ) = build_initial_support_plan(
        initial_phase=(
            initial_phase
        ),
        first_swing_foot=(
            FIRST_SWING_FOOT
        ),
        W_p_L6=W_p_L6,
        W_p_R6=W_p_R6,
        current_state=(
            current.x
        ),
        home_CoM=home_CoM,
        home_L6=home_L6,
        home_R6=home_R6,
    )

    # Initial no-time-advance FSM snapshot.
    (
        W_p_foothold_L,
        W_p_foothold_R,
    ) = get_gait_foothold_inputs(
        gait_state.current_phase,
        support_points,
    )

    gait_state, gait_output = (
        update_gait_manager(
            state=gait_state,
            c_L=c_L,
            c_R=c_R,
            dt_elapsed=0.0,
            timing=TIMING,
            W_p_L6=W_p_L6,
            W_p_R6=W_p_R6,
            W_p_foothold_L=(
                W_p_foothold_L
            ),
            W_p_foothold_R=(
                W_p_foothold_R
            ),
        )
    )

    # --------------------------------------------------------
    # 5. Nominal CoM planner state
    #
    # This is NOT measured feedback.
    # --------------------------------------------------------
    W_p_CoM_nominal_current = (
        home_CoM.copy()
    )

    held_nominal_velocity = np.zeros(
        3,
        dtype=float,
    )

    propagate_nominal_com = False

    # --------------------------------------------------------
    # 6. Held 100-Hz control/reference outputs
    # --------------------------------------------------------
    held_u0_MPC = np.zeros(
        12,
        dtype=float,
    )

    held_tau_mpc = np.zeros(
        10,
        dtype=float,
    )

    held_swing_foot = None

    held_q_swing_des = None
    held_q_dot_swing_des = None

    held_swing_phase = None
    held_swing_result = None

    last_W_p_CoM_ref = np.tile(
        home_CoM.reshape(
            1,
            3,
        ),
        (
            N,
            1,
        ),
    )

    last_contact_schedule = np.ones(
        (
            N,
            2,
        ),
        dtype=int,
    )

    last_solver_info = {
        "status": "not_solved",
        "solve_time": 0.0,
    }

    last_post_diagnostics = None
    last_mpc_compute_time = 0.0

    tau_command = np.zeros(
        10,
        dtype=float,
    )

    logger = SimulationLogger()

    reset_swing_low_level_state(
        None
    )

    # --------------------------------------------------------
    # 7. High-level timing state
    # --------------------------------------------------------
    physics_step = 0
    high_level_iteration = 0

    next_high_level_time = 0.0
    last_high_level_time = float(
        data.time
    )

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        "WALKING SIMULATION - DIRECT DS START + 100-Hz HIGH LEVEL"
    )

    print(
        "=" * 78
    )

    print(
        "dt_sim            =",
        DT_SIM,
        "s",
    )

    print(
        "dt_high_level     =",
        DT_HIGH_LEVEL,
        "s",
    )

    print(
        "dt_mpc prediction =",
        DT_MPC,
        "s",
    )

    print(
        "T_SS              =",
        T_SS,
        "s",
    )

    print(
        "T_DS              =",
        T_DS,
        "s",
    )

    print(
        "initial contact    =",
        [
            c_L,
            c_R,
        ],
    )

    print(
        "initial phase      =",
        initial_phase,
    )

    print(
        "first swing foot   =",
        FIRST_SWING_FOOT,
    )

    print(
        "step length        =",
        STEP_LENGTH,
        "m",
    )

    print(
        "step width         =",
        STEP_WIDTH,
        "m",
    )

    print(
        "swing height       =",
        SWING_HEIGHT,
        "m",
    )

    print(
        "\nHigh-level order:"
    )

    print(
        "x_k/contact -> FSM -> phase preview -> foot feedback/re-anchor "
        "-> DCM -> IK reference -> SRBD MPC -> held torques"
    )

    print(
        "Physics order:"
    )

    print(
        "x_k -> 2000-Hz swing joint PD (if swing) -> data.ctrl "
        "-> mj_step -> x_{k+1}"
    )

    # --------------------------------------------------------
    # 8. Viewer + simulation loop
    # --------------------------------------------------------
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

            current_time = float(
                data.time
            )

            # ====================================================
            # A. HIGH-LEVEL 100 Hz
            # ====================================================
            high_level_due = bool(
                current_time
                >=
                next_high_level_time
                -
                1e-12
            )

            high_level_updated = False

            if high_level_due:
                high_level_start = (
                    time.perf_counter()
                )

                # --------------------------------------------
                # A1. Read x_k / contact / feet BEFORE mj_step
                # --------------------------------------------
                current = (
                    get_current_state_mpc(
                        model,
                        data,
                    )
                )

                contact_state = (
                    get_contact_state(
                        model=model,
                        data=data,
                        contact_map=(
                            setup.contact_map
                        ),
                    )
                )

                c_L = int(
                    contact_state[
                        "c_L"
                    ]
                )

                c_R = int(
                    contact_state[
                        "c_R"
                    ]
                )

                (
                    W_p_L6,
                    W_p_R6,
                ) = get_foot_positions(
                    data,
                    L6_site_id,
                    R6_site_id,
                )

                # --------------------------------------------
                # A2. High-level elapsed time
                # --------------------------------------------
                if high_level_iteration == 0:
                    dt_high = 0.0

                else:
                    dt_high = max(
                        current_time
                        -
                        last_high_level_time,
                        0.0,
                    )

                last_high_level_time = (
                    current_time
                )

                # --------------------------------------------
                # A3. Advance NOMINAL CoM planner state only
                #     at the high-level rate.
                # --------------------------------------------
                if (
                    high_level_iteration > 0
                    and
                    propagate_nominal_com
                ):
                    W_p_CoM_nominal_current[
                        0:2
                    ] += (
                        held_nominal_velocity[
                            0:2
                        ]
                        *
                        dt_high
                    )

                W_p_CoM_nominal_current[
                    2
                ] = COM_HEIGHT

                # --------------------------------------------
                # A4. Ensure support queue exists for current
                #     accepted physical phase.
                # --------------------------------------------
                support_points = (
                    ensure_support_plan_length(
                        support_points,
                        current_phase=(
                            gait_state.current_phase
                        ),
                        target_length=(
                            SUPPORT_PLAN_LENGTH
                        ),
                    )
                )

                (
                    W_p_foothold_L,
                    W_p_foothold_R,
                ) = get_gait_foothold_inputs(
                    gait_state.current_phase,
                    support_points,
                )

                previous_phase = (
                    gait_state.current_phase
                )

                # --------------------------------------------
                # A5. Contact-synchronized FSM
                # --------------------------------------------
                gait_state, gait_output = (
                    update_gait_manager(
                        state=gait_state,
                        c_L=c_L,
                        c_R=c_R,
                        dt_elapsed=dt_high,
                        timing=TIMING,
                        W_p_L6=W_p_L6,
                        W_p_R6=W_p_R6,
                        W_p_foothold_L=(
                            W_p_foothold_L
                        ),
                        W_p_foothold_R=(
                            W_p_foothold_R
                        ),
                    )
                )

                if not (
                    gait_output.phase_contact_consistent
                ):
                    data.ctrl[:] = 0.0

                    raise RuntimeError(
                        "Gait/contact inconsistency: "
                        f"phase={gait_output.current_phase}, "
                        f"contact=[{c_L},{c_R}], "
                        f"events={gait_output.events}"
                    )

                # --------------------------------------------
                # A6. Queue topology update after accepted
                #     physical transition.
                # --------------------------------------------
                if gait_output.transitioned:
                    support_points = (
                        update_support_queue_after_transition(
                            support_points=(
                                support_points
                            ),
                            gait_output=(
                                gait_output
                            ),
                        )
                    )

                    print(
                        f"\n[t={current_time:.4f}] gait transition: "
                        f"{previous_phase} "
                        f"--{gait_output.transition_event}--> "
                        f"{gait_output.current_phase}"
                    )

                    if gait_output.transition_event in (
                        "LO_L",
                        "LO_R",
                    ):
                        if gait_output.swing_foot is None:
                            raise RuntimeError(
                                "DS -> SS transition nhưng swing_foot=None."
                            )

                        # Fresh physical swing begins at the real LO.
                        reset_swing_low_level_state(
                            gait_output.swing_foot
                        )

                support_points = (
                    ensure_support_plan_length(
                        support_points,
                        current_phase=(
                            gait_output.current_phase
                        ),
                        target_length=(
                            SUPPORT_PLAN_LENGTH
                        ),
                    )
                )

                # --------------------------------------------
                # A7. One shared nominal phase/contact preview.
                #
                # row 0 = current u_k interval.
                # --------------------------------------------
                phase_preview = (
                    build_contact_preview(
                        current_phase=(
                            gait_output.current_phase
                        ),
                        t_phase=(
                            gait_output.t_phase
                        ),
                        timing=TIMING,
                        horizon=N,
                        dt_mpc=DT_MPC,
                    )
                )

                contact_schedule = (
                    phase_preview[
                        "contact_schedule"
                    ].copy()
                )

                # --------------------------------------------
                # A8. 100-Hz realized-foot feedback ->
                #     re-anchor current/future foot geometry.
                # --------------------------------------------
                foot_plan = (
                    build_foot_plan_horizon(
                        current_phase=(
                            gait_output.current_phase
                        ),
                        phase_sequence=(
                            phase_preview[
                                "phase_sequence"
                            ]
                        ),
                        support_points=(
                            np.asarray(
                                support_points,
                                dtype=float,
                            )
                        ),
                        W_p_L6_feedback=(
                            W_p_L6
                        ),
                        W_p_R6_feedback=(
                            W_p_R6
                        ),
                    )
                )

                W_p_L6_plan = (
                    foot_plan[
                        "W_p_L6_plan"
                    ]
                )

                W_p_R6_plan = (
                    foot_plan[
                        "W_p_R6_plan"
                    ]
                )

                # Persist the 100-Hz re-anchored support queue.
                support_points = [
                    point.copy()

                    for point in
                    foot_plan[
                        "support_points_reanchored"
                    ]
                ]

                # --------------------------------------------
                # A9. Nominal DCM -> CoM reference.
                #
                # No measured-CoM correction.
                # --------------------------------------------
                translational_reference = (
                    build_mpc_translational_reference(
                        phase_preview=(
                            phase_preview
                        ),
                        W_p_L6_plan=(
                            W_p_L6_plan
                        ),
                        W_p_R6_plan=(
                            W_p_R6_plan
                        ),
                        W_p_CoM_nominal_current=(
                            W_p_CoM_nominal_current
                        ),
                        com_height=(
                            COM_HEIGHT
                        ),
                        gravity=(
                            GRAVITY
                        ),
                        horizon=N,
                        dt_mpc=DT_MPC,
                    )
                )

                W_p_CoM_ref = (
                    translational_reference[
                        "W_p_CoM_ref"
                    ].copy()
                )

                W_v_CoM_ref = (
                    translational_reference[
                        "W_v_CoM_ref"
                    ].copy()
                )

                held_nominal_velocity = (
                    np.asarray(
                        translational_reference[
                            "W_v_CoM_nominal_current"
                        ],
                        dtype=float,
                    ).reshape(3).copy()
                )

                # Contact-gated sliding:
                # nominal current state does not drift beyond a
                # physical phase boundary while waiting for contact.
                propagate_nominal_com = bool(
                    not
                    gait_output.waiting_for_contact
                )

                if not (
                    propagate_nominal_com
                ):
                    held_nominal_velocity[:] = 0.0

                # --------------------------------------------
                # A10. Swing Cartesian trajectory + IK at 100 Hz.
                #
                # Only q_des / q_dot_des are held.
                # Joint PD itself runs later at 2000 Hz.
                # --------------------------------------------
                held_swing_foot = None
                held_q_swing_des = None
                held_q_dot_swing_des = None
                held_swing_phase = None
                held_swing_result = None

                if (
                    gait_output.swing_foot
                    is not None
                ):
                    if (
                        gait_output.W_p_liftoff
                        is None
                    ):
                        raise RuntimeError(
                            "SS swing thiếu W_p_liftoff."
                        )

                    if (
                        gait_output.W_p_foothold
                        is None
                    ):
                        raise RuntimeError(
                            "SS swing thiếu W_p_foothold."
                        )

                    held_swing_result = (
                        compute_swing_low_level_control(
                            model=model,
                            data=data,
                            swing_foot=(
                                gait_output.swing_foot
                            ),
                            W_p_liftoff=(
                                gait_output.W_p_liftoff
                            ),
                            W_p_foothold=(
                                gait_output.W_p_foothold
                            ),
                            swing_phase=(
                                gait_output.swing_phase
                            ),
                            T_swing=T_SS,
                            swing_height=(
                                SWING_HEIGHT
                            ),
                        )
                    )

                    held_swing_foot = (
                        gait_output.swing_foot
                    )

                    held_q_swing_des = (
                        held_swing_result[
                            "q_swing_des"
                        ]
                        if isinstance(
                            held_swing_result,
                            dict,
                        )
                        else
                        held_swing_result.q_swing_des
                    )

                    held_q_dot_swing_des = (
                        held_swing_result[
                            "q_dot_swing_des"
                        ]
                        if isinstance(
                            held_swing_result,
                            dict,
                        )
                        else
                        held_swing_result.q_dot_swing_des
                    )

                    held_q_swing_des = (
                        np.asarray(
                            held_q_swing_des,
                            dtype=float,
                        ).reshape(5).copy()
                    )

                    held_q_dot_swing_des = (
                        np.asarray(
                            held_q_dot_swing_des,
                            dtype=float,
                        ).reshape(5).copy()
                    )

                    held_swing_phase = float(
                        gait_output.swing_phase
                    )

                # --------------------------------------------
                # A11. SRBD MPC/QP at 100 Hz.
                # --------------------------------------------
                mpc_compute_start = (
                    time.perf_counter()
                )

                (
                    U_opt,
                    u0_opt,
                    solver_info,
                    post_diagnostics,
                    srbd_reference,
                ) = solve_ltv_mpc(
                    setup=setup,
                    current=current,
                    W_p_CoM_ref=(
                        W_p_CoM_ref
                    ),
                    W_v_CoM_ref=(
                        W_v_CoM_ref
                    ),
                    W_p_L6_plan=(
                        W_p_L6_plan
                    ),
                    W_p_R6_plan=(
                        W_p_R6_plan
                    ),
                    contact_schedule=(
                        contact_schedule
                    ),
                )

                del U_opt
                del srbd_reference

                if u0_opt is None:
                    data.ctrl[:] = 0.0

                    raise RuntimeError(
                        "Walking QP không trả về nghiệm hợp lệ.\n"
                        f"phase={gait_output.current_phase}\n"
                        f"solver_info={solver_info}\n"
                        f"post_diagnostics={post_diagnostics}"
                    )

                held_u0_MPC = (
                    np.asarray(
                        u0_opt,
                        dtype=float,
                    ).reshape(12).copy()
                )

                if not np.all(
                    np.isfinite(
                        held_u0_MPC
                    )
                ):
                    raise RuntimeError(
                        "u0_MPC chứa NaN hoặc Inf."
                    )

                # --------------------------------------------
                # A12. Virtual-work wrench -> stance joint torque
                #      also held at 100 Hz.
                # --------------------------------------------
                (
                    held_tau_mpc,
                    _,
                    _,
                ) = get_joint_torque_from_MPC_wrench(
                    model,
                    data,
                    held_u0_MPC,
                )

                held_tau_mpc = (
                    np.asarray(
                        held_tau_mpc,
                        dtype=float,
                    ).reshape(10).copy()
                )

                if not np.all(
                    np.isfinite(
                        held_tau_mpc
                    )
                ):
                    raise RuntimeError(
                        "held_tau_mpc chứa NaN hoặc Inf."
                    )

                last_solver_info = (
                    solver_info
                )

                last_post_diagnostics = (
                    post_diagnostics
                )

                last_W_p_CoM_ref = (
                    W_p_CoM_ref.copy()
                )

                last_contact_schedule = (
                    contact_schedule.copy()
                )

                last_mpc_compute_time = (
                    time.perf_counter()
                    -
                    mpc_compute_start
                )

                high_level_updated = True

                next_high_level_time = (
                    current_time
                    +
                    DT_HIGH_LEVEL
                )

            # ====================================================
            # B. 2000-Hz LOW-LEVEL
            #
            # ONLY joint PD is recomputed at physics rate.
            # MPC / Jacobian wrench mapping / FSM / DCM / IK are held.
            # ====================================================
            if held_swing_foot is not None:
                if (
                    held_q_swing_des
                    is None
                    or
                    held_q_dot_swing_des
                    is None
                ):
                    raise RuntimeError(
                        "Swing PD thiếu held q_des/q_dot_des."
                    )

                (
                    q_swing,
                    q_dot_swing,
                ) = get_leg_joint_state(
                    model,
                    data,
                    held_swing_foot,
                )

                tau_swing = (
                    compute_swing_joint_PD(
                        q_des=(
                            held_q_swing_des
                        ),
                        q=q_swing,
                        q_dot=q_dot_swing,
                        Kp=KP_SWING,
                        Kd=KD_SWING,
                        q_dot_des=(
                            held_q_dot_swing_des
                        ),
                    )
                )

                tau_swing = (
                    np.asarray(
                        tau_swing,
                        dtype=float,
                    ).reshape(5)
                )

                tau_command = (
                    mix_walking_joint_torque(
                        tau_mpc=(
                            held_tau_mpc
                        ),
                        swing_foot=(
                            held_swing_foot
                        ),
                        tau_swing=(
                            tau_swing
                        ),
                    )
                )

            else:
                tau_command = (
                    held_tau_mpc.copy()
                )

            if not np.all(
                np.isfinite(
                    tau_command
                )
            ):
                raise RuntimeError(
                    "tau_command chứa NaN hoặc Inf."
                )

            # ====================================================
            # C. APPLY u_k / tau_k
            # ====================================================
            apply_joint_torque(
                model,
                data,
                tau_command,
            )

            # ====================================================
            # D. LOG HIGH-LEVEL RESULT AT x_k
            # ====================================================
            if high_level_updated:
                logger.log_step(
                    time=float(
                        data.time
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
                    qp_solve_time=float(
                        last_solver_info.get(
                            "solve_time",
                            0.0,
                        )
                    ),
                )

                if (
                    high_level_iteration
                    %
                    PRINT_EVERY_N_HIGH_LEVEL
                    ==
                    0
                ):
                    print_status(
                        high_level_iteration=(
                            high_level_iteration
                        ),
                        data=data,
                        gait_output=(
                            gait_output
                        ),
                        current=current,
                        W_p_CoM_ref=(
                            last_W_p_CoM_ref
                        ),
                        held_u0_MPC=(
                            held_u0_MPC
                        ),
                        held_tau_mpc=(
                            held_tau_mpc
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
                        support_points=(
                            support_points
                        ),
                        contact_schedule=(
                            last_contact_schedule
                        ),
                    )

                high_level_iteration += 1

            # ====================================================
            # E. THE ONLY PHYSICS ADVANCE
            #
            # x_k -> apply u_k -> mj_step -> x_{k+1}
            # ====================================================
            mujoco.mj_step(
                model,
                data,
            )

            physics_step += 1

            viewer.sync()

            # Optional wall-clock pacing.
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
    # 9. FINAL SUMMARY
    # ========================================================
    final_state = (
        get_current_state_mpc(
            model,
            data,
        )
    )

    final_contact = (
        get_contact_state(
            model=model,
            data=data,
            contact_map=(
                setup.contact_map
            ),
        )
    )

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        "WALKING SIMULATION FINISHED"
    )

    print(
        "=" * 78
    )

    print(
        "simulation time =",
        data.time,
        "s",
    )

    print(
        "physics steps   =",
        physics_step,
    )

    print(
        "high-level runs =",
        high_level_iteration,
    )

    print(
        "final contact   =",
        [
            final_contact[
                "c_L"
            ],
            final_contact[
                "c_R"
            ],
        ],
    )

    print(
        "final CoM       =",
        final_state.x[
            0:3
        ],
    )

    print(
        "final Euler     =",
        final_state.x[
            3:6
        ],
    )

    print(
        "final velocity  =",
        final_state.x[
            6:9
        ],
    )

    print(
        "nominal CoM     =",
        W_p_CoM_nominal_current,
    )

    print(
        "final gait phase=",
        gait_output.current_phase,
    )

    print(
        "final t_phase   =",
        gait_output.t_phase,
    )

    print(
        "expected event  =",
        gait_output.expected_event,
    )

    logger.print_summary()


if __name__ == "__main__":
    main()
