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
        sys.path.insert(0, directory_str)


# ============================================================
# SIMULATION
# ============================================================

from simulation_setup import initialize_simulation
from simulation_logger import SimulationLogger
from walking_startup import (
    STAGE_SETTLE,
    STAGE_INITIAL_DS,
    STAGE_COMPLETE,
    STAGE_FAULT,
    initialize_walking_startup,
    update_walking_startup,
)
from gait_manager import initialize_gait_manager, update_gait_manager
from swing_low_level_controller import (
    compute_swing_low_level_control,
    reset_swing_joint_velocity_reference,
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

from Command.gait_timing import build_gait_timing, get_nominal_phase_duration
from Command.contact_schedule import build_phase_preview, build_contact_schedule
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

from W_wrench_to_joint_torque import get_joint_torque_from_MPC_wrench
from actuator_mapping import apply_joint_torque


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

DT_MPC = float(TIMING["dt_mpc"])
DT_MPC_UPDATE = 0.01
T_SS = float(TIMING["T_SS"])
T_DS = float(TIMING["T_DS"])

FIRST_SWING_FOOT = "R"

STEP_LENGTH = 0.05
STEP_WIDTH = HOME_NOMINAL_STEP_WIDTH
SWING_HEIGHT = 0.04
GROUND_HEIGHT = 0.0

SIM_DURATION = 5.0

# Startup:
#   HOME DS -> SETTLE -> directional INITIAL_DS -> actual LO -> cyclic gait
SETTLE_TIME = 0.50
STARTUP_LIFTOFF_TIMEOUT = 2.00
INITIAL_DS_SHIFT_TIME = 0.30

# The first physical liftoff is allowed only after the measured CoM
# has actually shifted toward the stance/support foot.
#
# Right foot swings first:
#     left foot supports -> positive World-y shift required.
#
# Left foot swings first:
#     right foot supports -> negative World-y shift required.
STARTUP_COM_SHIFT_MIN = 0.015

# Startup physical liftoff assistance.
#
# The first foot is unloaded and lifted vertically before the first
# contact-switch LO event. Horizontal motion is held at the latched
# liftoff position.
STARTUP_PRELIFTOFF_PHASE_MAX = 0.225
STARTUP_PRELIFTOFF_RAMP_TIME = 0.05

# Cyclic DS -> SS transition policy.
#
# The physical FSM remains contact-synchronized:
#
#     DS_L_TO_R --LO_L--> RSS
#     DS_R_TO_L --LO_R--> LSS
#
# During DS, the controller waits for:
#   1) nominal DS time completed, and
#   2) measured CoM transferred toward the incoming support.
#
# Then the outgoing foot is unloaded in the CURRENT MPC input and
# lifted vertically until contact_switch detects the real LO edge.
#
# The phase reached at LO is retained as an offset for the following
# single-support swing, so the swing reference does not reset to zero.
CYCLIC_LIFTOFF_PHASE_MAX = 0.225
CYCLIC_LIFTOFF_RAMP_TIME = 0.05
CYCLIC_COM_SHIFT_MIN = 0.015

# Future support points retained in the walking queue.
SUPPORT_PLAN_LENGTH = 10

# If an expected cyclic event does not occur sufficiently long after the
# nominal phase duration, stop instead of inventing a contact transition.
PHASE_EVENT_TIMEOUT_MARGIN = 1.0

# QP
QP_MAX_ITER = 10000
QP_CONSTRAINT_TOLERANCE = 3e-4

PRINT_EVERY_N_MPC = 20

# Physical liftoff diagnostics.
#
# During the first swing command, print one compact physical/IK/PD snapshot
# at this interval. These diagnostics are read-only and do not change the
# controller or advance MuJoCo.
LIFTOFF_DIAGNOSTIC_PERIOD = 0.05

GRAVITY = 9.81


# ============================================================
# BASIC HELPERS
# ============================================================

def get_site_id(model, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise RuntimeError(
            f"Không tìm thấy site '{site_name}' trong MuJoCo model."
        )

    return int(site_id)


def get_foot_positions(data, L6_site_id, R6_site_id):
    W_p_L6 = np.asarray(
        data.site_xpos[L6_site_id],
        dtype=float,
    ).reshape(3).copy()

    W_p_R6 = np.asarray(
        data.site_xpos[R6_site_id],
        dtype=float,
    ).reshape(3).copy()

    if not np.all(np.isfinite(W_p_L6)):
        raise RuntimeError("W_p_L6 chứa NaN hoặc Inf.")

    if not np.all(np.isfinite(W_p_R6)):
        raise RuntimeError("W_p_R6 chứa NaN hoặc Inf.")

    return W_p_L6, W_p_R6


def rotation_matrix_from_rpy(euler):
    """
    World-from-body ZYX rotation using [roll, pitch, yaw].

    Home orientation is zero, so the home CoM-to-foot vectors may be used
    as body-fixed nominal offsets for the flat-ground baseline.
    """
    roll, pitch, yaw = np.asarray(euler, dtype=float).reshape(3)

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

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

    return Rz @ Ry @ Rx


def get_kinematic_reference(
    current_state,
    swing_foot,
    home_CoM,
    home_L6,
    home_R6,
):
    """
    Home-equivalent nominal swing-foot location for the current body pose.
    """
    current_state = np.asarray(current_state, dtype=float).reshape(13)

    W_p_CoM = current_state[0:3]
    euler = current_state[3:6]

    if swing_foot == "L":
        home_offset = home_L6 - home_CoM
    elif swing_foot == "R":
        home_offset = home_R6 - home_CoM
    else:
        raise ValueError("swing_foot phải là 'L' hoặc 'R'.")

    R_W_B = rotation_matrix_from_rpy(euler)

    return W_p_CoM + R_W_B @ home_offset


def mix_walking_joint_torque(
    tau_mpc,
    swing_foot=None,
    tau_swing=None,
):
    """
    Torque architecture:

        LSS: [tau_L_MPC, tau_R_PD]
        RSS: [tau_L_PD,  tau_R_MPC]
        DS:  [tau_L_MPC, tau_R_MPC]
    """
    tau_mpc = np.asarray(tau_mpc, dtype=float).reshape(-1)

    if tau_mpc.shape != (10,):
        raise ValueError("tau_mpc phải có shape (10,).")

    if swing_foot is None:
        return tau_mpc.copy()

    tau_swing = np.asarray(tau_swing, dtype=float).reshape(-1)

    if tau_swing.shape != (5,):
        raise ValueError("tau_swing phải có shape (5,).")

    tau = tau_mpc.copy()

    if swing_foot == "L":
        tau[0:5] = tau_swing
    elif swing_foot == "R":
        tau[5:10] = tau_swing
    else:
        raise ValueError("swing_foot phải là 'L', 'R' hoặc None.")

    return tau


def get_ds_transition_spec(phase):
    """
    Return the physical DS -> SS transition definition.

    World-y convention:
        left foot  -> positive y
        right foot -> negative y
    """
    if phase == "DS_L_TO_R":
        return {
            "outgoing_foot": "L",
            "incoming_support": "R",
            "support_direction_y": -1.0,
            "expected_lo": "LO_L",
        }

    if phase == "DS_R_TO_L":
        return {
            "outgoing_foot": "R",
            "incoming_support": "L",
            "support_direction_y": +1.0,
            "expected_lo": "LO_R",
        }

    raise ValueError(
        f"Phase {phase!r} không phải directional double-support phase."
    )


def evaluate_weight_transfer_guard(
    phase,
    t_phase,
    current_state,
    W_p_L6,
    W_p_R6,
):
    """
    Algorithmic DS -> liftoff guard.

    Liftoff assistance is allowed only when:
        - nominal DS time has completed, and
        - measured CoM has crossed sufficiently toward the incoming support.

    This function does not modify the FSM, contact state, controller state,
    or simulation.
    """
    spec = get_ds_transition_spec(phase)

    feet_mid_y = 0.5 * (
        float(W_p_L6[1]) + float(W_p_R6[1])
    )

    com_y = float(
        np.asarray(current_state, dtype=float).reshape(13)[1]
    )

    signed_shift = (
        float(spec["support_direction_y"])
        * (com_y - feet_mid_y)
    )

    time_ready = bool(
        float(t_phase) >= T_DS - 1e-12
    )

    com_ready = bool(
        signed_shift >= CYCLIC_COM_SHIFT_MIN
    )

    return {
        **spec,
        "ready": bool(time_ready and com_ready),
        "time_ready": time_ready,
        "com_ready": com_ready,
        "feet_mid_y": feet_mid_y,
        "com_y": com_y,
        "signed_shift": signed_shift,
    }


def compute_cyclic_liftoff_phase(current_time, start_time):
    """
    Smooth vertical liftoff phase used while the outgoing foot is still
    physically in contact.
    """
    elapsed = max(
        float(current_time) - float(start_time),
        0.0,
    )

    ratio = min(
        elapsed / CYCLIC_LIFTOFF_RAMP_TIME,
        1.0,
    )

    return float(
        CYCLIC_LIFTOFF_PHASE_MAX * ratio
    )


def continue_swing_phase(raw_swing_phase, phase_at_lo):
    """
    Continue the single-support swing from the phase already reached at LO.

        raw gait phase : 0 -> 1
        effective phase:
            s = s_LO + (1 - s_LO) * raw

    Therefore:
        raw = 0 -> s = s_LO
        raw = 1 -> s = 1
    """
    raw = float(
        np.clip(raw_swing_phase, 0.0, 1.0)
    )

    offset = float(
        np.clip(phase_at_lo, 0.0, 1.0)
    )

    return float(
        offset + (1.0 - offset) * raw
    )


# ============================================================
# SUPPORT-PLAN HELPERS
# ============================================================

def ensure_support_plan_length(
    support_points,
    current_phase,
    target_length=SUPPORT_PLAN_LENGTH,
):
    """
    Extend the ordered support queue with simple nominal flat-ground steps.

    Only the immediate physical foothold is kinematically clipped. More
    distant points are nominal DCM/MPC planning points and are replanned
    before they become the immediate swing target.
    """
    points = [
        np.asarray(point, dtype=float).reshape(3).copy()
        for point in support_points
    ]

    if len(points) < 2:
        raise ValueError("support_points cần ít nhất 2 điểm.")

    initial_support_foot = get_initial_support_foot(current_phase)

    while len(points) < int(target_length):
        last_index = len(points) - 1

        last_foot = get_support_foot_from_index(
            initial_support_foot,
            last_index,
        )

        next_foot = get_opposite_foot(last_foot)

        next_point = generate_nominal_foothold(
            W_p_support=points[-1],
            swing_foot=next_foot,
            step_length=STEP_LENGTH,
            step_width=STEP_WIDTH,
            ground_height=GROUND_HEIGHT,
        )

        points.append(next_point)

    return points


def plan_bounded_support_point(
    support_points,
    current_phase,
    target_index,
    swing_foot,
    current_state,
    home_CoM,
    home_L6,
    home_R6,
):
    """
    Replace/append one immediate foothold using the measured safe workspace.
    """
    points = [
        np.asarray(point, dtype=float).reshape(3).copy()
        for point in support_points
    ]

    target_index = int(target_index)

    if target_index <= 0:
        raise ValueError("target_index phải > 0.")

    if target_index - 1 >= len(points):
        raise ValueError("Thiếu support point đứng trước target_index.")

    initial_support_foot = get_initial_support_foot(current_phase)

    expected_foot = get_support_foot_from_index(
        initial_support_foot,
        target_index,
    )

    if expected_foot != swing_foot:
        raise RuntimeError(
            "Support-plan foot ordering không khớp swing_foot. "
            f"Expected={expected_foot}, received={swing_foot}."
        )

    W_p_kinematic_reference = get_kinematic_reference(
        current_state=current_state,
        swing_foot=swing_foot,
        home_CoM=home_CoM,
        home_L6=home_L6,
        home_R6=home_R6,
    )

    W_p_plan, diagnostics = generate_foothold(
        W_p_support=points[target_index - 1],
        W_p_kinematic_reference=W_p_kinematic_reference,
        swing_foot=swing_foot,
        step_length=STEP_LENGTH,
        step_width=STEP_WIDTH,
        correction_xy=None,
        ground_height=GROUND_HEIGHT,
    )

    if target_index < len(points):
        points[target_index] = W_p_plan.copy()
    elif target_index == len(points):
        points.append(W_p_plan.copy())
    else:
        raise ValueError("target_index tạo khoảng trống trong support plan.")

    points = ensure_support_plan_length(
        points,
        current_phase=current_phase,
        target_length=SUPPORT_PLAN_LENGTH,
    )

    return points, diagnostics


def build_initial_support_plan(
    initial_ds_phase,
    first_swing_foot,
    W_p_L6,
    W_p_R6,
    current_state,
    home_CoM,
    home_L6,
    home_R6,
):
    """
    INITIAL_DS ordered support queue.

    Right first:
        DS_R_TO_L -> [R_outgoing, L_incoming, R_first_target, ...]

    Left first:
        DS_L_TO_R -> [L_outgoing, R_incoming, L_first_target, ...]
    """
    if first_swing_foot == "R":
        if initial_ds_phase != "DS_R_TO_L":
            raise RuntimeError("Right-first startup phải dùng DS_R_TO_L.")
        points = [W_p_R6.copy(), W_p_L6.copy()]

    elif first_swing_foot == "L":
        if initial_ds_phase != "DS_L_TO_R":
            raise RuntimeError("Left-first startup phải dùng DS_L_TO_R.")
        points = [W_p_L6.copy(), W_p_R6.copy()]

    else:
        raise ValueError("first_swing_foot phải là 'L' hoặc 'R'.")

    # index 2 is the first actual swing target.
    points, diagnostics = plan_bounded_support_point(
        support_points=points,
        current_phase=initial_ds_phase,
        target_index=2,
        swing_foot=first_swing_foot,
        current_state=current_state,
        home_CoM=home_CoM,
        home_L6=home_L6,
        home_R6=home_R6,
    )

    return points, diagnostics


def get_gait_foothold_inputs(current_phase, support_points):
    """
    Foothold arguments supplied to update_gait_manager().

    In DS, queue index 2 is the next swing target that becomes active if
    the expected LO event occurs in the current sample.
    """
    points = support_points

    if current_phase == "LSS":
        return None, points[1].copy()

    if current_phase == "RSS":
        return points[1].copy(), None

    if current_phase == "DS_L_TO_R":
        return points[2].copy(), None

    if current_phase == "DS_R_TO_L":
        return None, points[2].copy()

    raise ValueError(f"Unsupported current_phase: {current_phase}")


def update_support_plan_after_gait_transition(
    support_points,
    gait_output,
    W_p_L6,
    W_p_R6,
    current_state,
    home_CoM,
    home_L6,
    home_R6,
):
    """
    Higher-layer realized-foothold / queue update.

    TD:
        replace planned incoming support with realized L6/R6 and plan the
        next swing target while still in directional DS.

    LO:
        discard outgoing support so queue[0] becomes the new current support.
    """
    points = [
        np.asarray(point, dtype=float).reshape(3).copy()
        for point in support_points
    ]

    event = gait_output.transition_event
    phase = gait_output.current_phase

    if event == "TD_R":
        if phase != "DS_L_TO_R":
            raise RuntimeError("TD_R phải đưa FSM vào DS_L_TO_R.")

        points[1] = W_p_R6.copy()

        points, diagnostics = plan_bounded_support_point(
            support_points=points,
            current_phase=phase,
            target_index=2,
            swing_foot="L",
            current_state=current_state,
            home_CoM=home_CoM,
            home_L6=home_L6,
            home_R6=home_R6,
        )

        return points, diagnostics

    if event == "TD_L":
        if phase != "DS_R_TO_L":
            raise RuntimeError("TD_L phải đưa FSM vào DS_R_TO_L.")

        points[1] = W_p_L6.copy()

        points, diagnostics = plan_bounded_support_point(
            support_points=points,
            current_phase=phase,
            target_index=2,
            swing_foot="R",
            current_state=current_state,
            home_CoM=home_CoM,
            home_L6=home_L6,
            home_R6=home_R6,
        )

        return points, diagnostics

    if event in ("LO_L", "LO_R"):
        if len(points) < 3:
            raise RuntimeError("Support queue quá ngắn tại liftoff.")

        points.pop(0)

        points = ensure_support_plan_length(
            points,
            current_phase=phase,
            target_length=SUPPORT_PLAN_LENGTH,
        )

        return points, None

    return points, None


# ============================================================
# STARTUP -> GAIT HANDOFF
# ============================================================

def initialize_gait_from_startup(
    startup_output,
    c_L,
    c_R,
    W_p_L6,
    W_p_R6,
):
    if not startup_output.handoff_ready:
        raise ValueError("Startup chưa sẵn sàng handoff.")

    if startup_output.W_p_liftoff is None:
        raise ValueError("Startup chưa có realized liftoff position.")

    gait_state = initialize_gait_manager(
        initial_phase=startup_output.handoff_phase,
        c_L=c_L,
        c_R=c_R,
        W_p_L6=W_p_L6,
        W_p_R6=W_p_R6,
        t_phase=0.0,
        strict_contact=True,
    )

    if startup_output.first_swing_foot == "R":
        gait_state.W_p_liftoff_R = startup_output.W_p_liftoff.copy()
    else:
        gait_state.W_p_liftoff_L = startup_output.W_p_liftoff.copy()

    return gait_state


# ============================================================
# MPC HELPERS
# ============================================================

def build_standing_reference(x_initial, horizon):
    x_initial = np.asarray(x_initial, dtype=float).reshape(13)

    W_p_CoM_ref = np.tile(x_initial[0:3].reshape(1, 3), (horizon, 1))
    W_v_CoM_ref = np.zeros((horizon, 3), dtype=float)

    return build_srbd_reference(
        W_p_CoM_ref=W_p_CoM_ref,
        W_v_CoM_ref=W_v_CoM_ref,
        W_euler_CoM_ref=x_initial[3:6],
        W_w_CoM_ref=np.zeros(3, dtype=float),
    )


def solve_ltv_mpc(
    setup,
    current,
    W_p_CoM_ref,
    W_v_CoM_ref,
    W_p_L6_plan,
    W_p_R6_plan,
    contact_schedule,
):
    srbd_reference = build_srbd_reference(
        W_p_CoM_ref=W_p_CoM_ref,
        W_v_CoM_ref=W_v_CoM_ref,
        W_euler_CoM_ref=np.zeros(3, dtype=float),
        W_w_CoM_ref=np.zeros(3, dtype=float),
    )

    W_r_L_sequence, W_r_R_sequence = build_moment_arm_horizon(
        W_p_CoM_ref=srbd_reference["W_p_CoM_ref"],
        W_p_L6_plan=W_p_L6_plan,
        W_p_R6_plan=W_p_R6_plan,
    )

    Phi, Gamma = build_SRBD_model_sequence(
        Ew_inverse=current.Ew_inverse,
        mass_CoM=setup.mass_CoM,
        W_inertia_CoM=setup.W_inertia_CoM_q0,
        W_r_L_sequence=W_r_L_sequence,
        W_r_R_sequence=W_r_R_sequence,
        dt_mpc=DT_MPC,
        horizon=N,
    )

    U_opt, u0_opt, solver_info, post_diagnostics = build_and_solve_QP(
        qp_solver=setup.qp_solver,
        Phi=Phi,
        Gamma=Gamma,
        x_current=current.x,
        X_ref=srbd_reference["X_ref"],
        N=N,
        L6_R_W=current.L6_IK_R_W,
        R6_R_W=current.R6_IK_R_W,
        contact_schedule=contact_schedule,
        print_diagnostics=False,
    )

    return (
        U_opt,
        u0_opt,
        solver_info,
        post_diagnostics,
        srbd_reference,
    )


def get_reference_phase_time(current_phase, t_phase):
    """
    DCM is nominal. If actual contact is late, freeze the nominal phase at
    its boundary instead of letting nominal time force the physical FSM.
    """
    T_phase = float(
        get_nominal_phase_duration(
            current_phase,
            TIMING,
        )
    )

    t_phase = float(t_phase)
    t_reference = min(t_phase, T_phase)

    at_or_beyond_nominal_end = bool(
        t_phase >= T_phase - 1e-12
    )

    return t_reference, T_phase, at_or_beyond_nominal_end


# ============================================================
# STATUS
# ============================================================

def print_status(
    mpc_iteration,
    data,
    mode,
    phase,
    t_phase,
    c_L,
    c_R,
    swing_foot,
    swing_phase,
    current,
    W_p_CoM_ref,
    held_u0_MPC,
    tau_command,
    solver_info,
    post_diagnostics,
    support_points,
):
    max_violation = None
    if post_diagnostics is not None:
        max_violation = post_diagnostics.get("max_violation")

    print("\n" + "-" * 78)
    print(
        f"t={data.time:.4f} s | MPC={mpc_iteration} | "
        f"mode={mode} | phase={phase} | t_phase={t_phase:.4f}"
    )
    print(
        f"contact=[{c_L},{c_R}] | swing={swing_foot} | "
        f"swing_phase={swing_phase}"
    )
    print("CoM actual =", current.x[0:3])
    print("CoM ref[0] =", W_p_CoM_ref[0])
    print(
        "Fz current [L,R] =",
        np.array([held_u0_MPC[2], held_u0_MPC[8]]),
        "N",
    )
    print("tau =", tau_command)
    print(
        "QP status =",
        solver_info.get("status"),
        "| solve_time =",
        solver_info.get("solve_time"),
        "| max_violation =",
        max_violation,
    )

    if support_points is not None:
        preview = np.asarray(support_points[:4], dtype=float)
        print("support queue first points =\n", preview)


# ============================================================
# PHYSICAL LIFTOFF DIAGNOSTICS
# ============================================================

def _get_foot_contact_diagnostics(
    contact_state,
    swing_foot,
):
    """
    Return current physical floor-contact diagnostics for one foot.

    This reads the output of State/get_contact_state.py only.
    It does not modify contact state or gait logic.
    """
    if swing_foot == "L":
        count = int(contact_state["left_contact_count"])
        contacts = contact_state["left_contacts"]
    elif swing_foot == "R":
        count = int(contact_state["right_contact_count"])
        contacts = contact_state["right_contacts"]
    else:
        raise ValueError("swing_foot phải là 'L' hoặc 'R'.")

    positions = []
    distances = []

    for entry in contacts:
        positions.append(
            np.asarray(
                entry["position"],
                dtype=float,
            ).reshape(3).copy()
        )
        distances.append(
            float(
                entry["distance"]
            )
        )

    return count, positions, distances


def print_liftoff_diagnostics(
    data,
    control_mode,
    phase,
    contact_state,
    W_p_L6,
    W_p_R6,
    swing_result,
    held_u0_MPC,
    tau_command,
    current=None,
    label="LIFTOFF DEBUG",
):
    """
    Read-only diagnostic snapshot for physical liftoff.

    The purpose is to separate four possibilities:

        1. swing trajectory is not commanding vertical lift,
        2. IK does not generate the required joint motion,
        3. PD torque is insufficient to realize the IK command,
        4. the foot has almost lifted but MuJoCo still reports one or more
           physical floor contacts.

    No controller state is modified here.
    No mj_step() is called here.
    """
    if swing_result is None:
        print("\n" + "=" * 78)
        print(f"{label}: no swing_result available")
        print("=" * 78)
        print(
            "contact =",
            [
                int(contact_state["c_L"]),
                int(contact_state["c_R"]),
            ],
            "| contact_count =",
            [
                int(contact_state["left_contact_count"]),
                int(contact_state["right_contact_count"]),
            ],
        )
        return

    swing_foot = str(
        swing_result.swing_foot
    )

    if swing_foot == "L":
        W_p_actual = np.asarray(
            W_p_L6,
            dtype=float,
        ).reshape(3)
        commanded_mpc_fz = float(
            held_u0_MPC[2]
        )
    elif swing_foot == "R":
        W_p_actual = np.asarray(
            W_p_R6,
            dtype=float,
        ).reshape(3)
        commanded_mpc_fz = float(
            held_u0_MPC[8]
        )
    else:
        raise RuntimeError(
            f"Unsupported swing foot in diagnostics: {swing_foot}"
        )

    W_p_des = np.asarray(
        swing_result.W_p_swing_des,
        dtype=float,
    ).reshape(3)

    W_v_des = np.asarray(
        swing_result.W_v_swing_des,
        dtype=float,
    ).reshape(3)

    q_actual = np.asarray(
        swing_result.q_swing,
        dtype=float,
    ).reshape(5)

    q_des = np.asarray(
        swing_result.q_swing_des,
        dtype=float,
    ).reshape(5)

    q_posture_ref = np.asarray(
        swing_result.q_posture_ref,
        dtype=float,
    ).reshape(5)

    q_error = (
        q_des
        -
        q_actual
    )

    tau_swing = np.asarray(
        swing_result.tau_swing,
        dtype=float,
    ).reshape(5)

    contact_count, contact_positions, contact_distances = (
        _get_foot_contact_diagnostics(
            contact_state=contact_state,
            swing_foot=swing_foot,
        )
    )

    z_error = float(
        W_p_des[2]
        -
        W_p_actual[2]
    )

    print("\n" + "=" * 78)
    print(
        f"{label} | t={data.time:.4f} s | "
        f"mode={control_mode} | phase={phase}"
    )
    print("=" * 78)

    print(
        f"swing foot={swing_foot} | "
        f"swing_phase={swing_result.swing_phase:.6f}"
    )

    print(
        "physical contact =",
        [
            int(contact_state["c_L"]),
            int(contact_state["c_R"]),
        ],
        "| contact_count [L,R] =",
        [
            int(contact_state["left_contact_count"]),
            int(contact_state["right_contact_count"]),
        ],
        "| ncon =",
        int(contact_state["ncon"]),
    )

    print(
        f"{swing_foot} contact_count = {contact_count}"
    )

    if contact_count > 0:
        print(
            f"{swing_foot} contact positions ="
        )
        for i, position in enumerate(
            contact_positions
        ):
            print(
                f"  contact[{i}] pos={position} "
                f"dist={contact_distances[i]:+.8f} m"
            )
    else:
        print(
            f"{swing_foot} contact positions = []"
        )

    print(
        f"{swing_foot} foot actual xyz =",
        W_p_actual,
    )

    print(
        f"{swing_foot} foot desired xyz =",
        W_p_des,
    )

    print(
        f"{swing_foot} desired velocity xyz =",
        W_v_des,
    )

    print(
        f"{swing_foot} z actual/des/error = "
        f"{W_p_actual[2]:+.7f} / "
        f"{W_p_des[2]:+.7f} / "
        f"{z_error:+.7f} m "
        f"({1000.0 * z_error:+.3f} mm)"
    )

    print(
        f"MPC commanded Fz_{swing_foot} = "
        f"{commanded_mpc_fz:+.7f} N "
        "(QP command, NOT measured GRF)"
    )

    if current is not None:
        print(
            "CoM actual =",
            np.asarray(
                current.x[0:3],
                dtype=float,
            ),
        )

    print(
        "q actual      =",
        q_actual,
    )

    print(
        "q posture_ref =",
        q_posture_ref,
    )

    print(
        "q desired     =",
        q_des,
    )

    print(
        "q error       =",
        q_error,
    )

    print(
        "tau swing     =",
        tau_swing,
    )

    print(
        "full tau cmd  =",
        np.asarray(
            tau_command,
            dtype=float,
        ).reshape(10),
    )

    ik_diag = swing_result.ik_diagnostics

    print(
        "IK converged  =",
        ik_diag.get("converged"),
        "| IK iterations =",
        ik_diag.get("iterations"),
        "| method =",
        ik_diag.get("method_used"),
        "| final pos error norm =",
        ik_diag.get("final_position_error_norm"),
    )

    print(
        "posture step norm(last) =",
        ik_diag.get("posture_step_norm_last"),
        "| null projection error(last) =",
        ik_diag.get("null_space_projection_error_last"),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Common setup
    # --------------------------------------------------------
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
        qp_enable_warm_start=True,
        qp_constraint_tolerance=QP_CONSTRAINT_TOLERANCE,
        print_initialization=True,
    )

    model = setup.model
    data = setup.data
    DT_SIM = float(setup.dt_sim)

    # --------------------------------------------------------
    # 2. Physical foot sites
    # --------------------------------------------------------
    L6_site_id = get_site_id(model, "L6")
    R6_site_id = get_site_id(model, "R6")

    W_p_L6_home, W_p_R6_home = get_foot_positions(
        data,
        L6_site_id,
        R6_site_id,
    )

    home_CoM = setup.x_initial[0:3].copy()
    home_L6 = W_p_L6_home.copy()
    home_R6 = W_p_R6_home.copy()
    COM_HEIGHT = float(home_CoM[2])

    # --------------------------------------------------------
    # 3. Initial physical contact
    # --------------------------------------------------------
    contact_state = get_contact_state(
        model=model,
        data=data,
        contact_map=setup.contact_map,
    )

    c_L = int(contact_state["c_L"])
    c_R = int(contact_state["c_R"])

    # --------------------------------------------------------
    # 4. Walking startup
    # --------------------------------------------------------
    startup_state, startup_output = initialize_walking_startup(
        first_swing_foot=FIRST_SWING_FOOT,
        settle_time=SETTLE_TIME,
        liftoff_timeout=STARTUP_LIFTOFF_TIMEOUT,
        c_L=c_L,
        c_R=c_R,
        W_p_L6=W_p_L6_home,
        W_p_R6=W_p_R6_home,
    )

    gait_state = None
    gait_output = None

    # Built only when INITIAL_DS becomes active.
    support_points = None
    last_foothold_diagnostics = None

    # First physical lift command is intentionally separated from SETTLE.
    startup_swing_command_active = False
    startup_swing_command_start_time = None
    startup_command_liftoff_position = None

    # Cyclic DS -> SS transition state.
    #
    # This state exists only while the outgoing foot is being physically
    # unloaded/lifted inside DS. The FSM itself is still changed only by
    # the real LO event from contact_switch.
    cyclic_liftoff_active = False
    cyclic_liftoff_ds_phase = None
    cyclic_liftoff_foot = None
    cyclic_liftoff_start_time = None
    cyclic_liftoff_position = None

    # Swing phase continuity across LO.
    #
    # Example:
    #     DS liftoff phase reaches 0.225
    #     contact_switch detects LO
    #     SS swing continues from 0.225 instead of resetting to 0.0
    swing_phase_offset = {
        "L": 0.0,
        "R": 0.0,
    }

    # --------------------------------------------------------
    # 5. Nominal CoM state for DCM reference
    #
    # IMPORTANT:
    # Never reset this state to measured CoM at every solve.
    # --------------------------------------------------------
    W_p_CoM_nominal_current = home_CoM.copy()
    W_v_CoM_nominal_hold = np.zeros(3, dtype=float)
    propagate_nominal_com = False

    # --------------------------------------------------------
    # 6. Standing reference used during SETTLE
    # --------------------------------------------------------
    standing_reference = build_standing_reference(
        setup.x_initial,
        N,
    )

    # --------------------------------------------------------
    # 7. Held control / diagnostics
    # --------------------------------------------------------
    held_u0_MPC = np.zeros(12, dtype=float)
    tau_command = np.zeros(10, dtype=float)

    last_solver_info = {
        "status": "not_solved",
        "solve_time": 0.0,
    }
    last_post_diagnostics = None
    last_mpc_compute_time = 0.0
    last_W_p_CoM_ref = standing_reference["W_p_CoM_ref"].copy()

    # Last available swing result is retained only for diagnostics.
    # It lets us print the final IK/PD state immediately before a startup FAULT.
    last_swing_result = None

    # Next simulation time at which physical-liftoff diagnostics are printed.
    next_liftoff_diagnostic_time = 0.0

    logger = SimulationLogger()

    reset_swing_low_level_state(None)

    # --------------------------------------------------------
    # 8. Timing state
    # --------------------------------------------------------
    physics_step = 0
    mpc_iteration = 0
    next_mpc_update_time = 0.0
    last_logic_time = float(data.time)

    print("\n" + "=" * 78)
    print("WALKING SIMULATION - SRBD MPC + DCM + CONTACT-SYNCHRONIZED FSM")
    print("=" * 78)
    print("dt_sim          =", DT_SIM, "s")
    print("dt_mpc_update   =", DT_MPC_UPDATE, "s")
    print("dt_mpc          =", DT_MPC, "s")
    print("T_SS            =", T_SS, "s")
    print("T_DS            =", T_DS, "s")
    print("first swing foot=", FIRST_SWING_FOOT)
    print("step length     =", STEP_LENGTH, "m")
    print("step width      =", STEP_WIDTH, "m")
    print("swing height    =", SWING_HEIGHT, "m")
    print("\nStrict order:")
    print("x_k -> contact/FSM/reference -> MPC -> torque -> data.ctrl -> mj_step -> x_{k+1}")

    # --------------------------------------------------------
    # 9. Viewer + physics loop
    # --------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            wall_step_start = time.perf_counter()

            # ====================================================
            # A. Read x_k contact / feet BEFORE any control or step
            # ====================================================
            current_time = float(data.time)
            dt_logic = max(current_time - last_logic_time, 0.0)
            last_logic_time = current_time

            contact_state = get_contact_state(
                model=model,
                data=data,
                contact_map=setup.contact_map,
            )

            c_L = int(contact_state["c_L"])
            c_R = int(contact_state["c_R"])

            W_p_L6, W_p_R6 = get_foot_positions(
                data,
                L6_site_id,
                R6_site_id,
            )

            current = None
            force_mpc_update = False

            # Current control-mode descriptors.
            control_mode = None
            current_phase = None
            current_t_phase = 0.0
            current_swing_foot = None
            current_swing_phase = None

            # ====================================================
            # B. Startup logic until the first REAL liftoff
            # ====================================================
            if gait_state is None:
                previous_startup_stage = startup_state.stage

                startup_state, startup_output = update_walking_startup(
                    state=startup_state,
                    dt_elapsed=dt_logic,
                    c_L=c_L,
                    c_R=c_R,
                    W_p_L6=W_p_L6,
                    W_p_R6=W_p_R6,
                )

                if startup_output.faulted or startup_state.stage == STAGE_FAULT:
                    current = (
                        current
                        if current is not None
                        else get_current_state_mpc(model, data)
                    )

                    print_liftoff_diagnostics(
                        data=data,
                        control_mode="STARTUP_FAULT",
                        phase=startup_output.initial_ds_phase,
                        contact_state=contact_state,
                        W_p_L6=W_p_L6,
                        W_p_R6=W_p_R6,
                        swing_result=last_swing_result,
                        held_u0_MPC=held_u0_MPC,
                        tau_command=tau_command,
                        current=current,
                        label="STARTUP FAULT SNAPSHOT",
                    )

                    data.ctrl[:] = 0.0

                    raise RuntimeError(
                        "Walking startup FAULT: "
                        f"{startup_output.fault_reason}"
                    )

                if startup_state.stage != previous_startup_stage:
                    force_mpc_update = True
                    print(
                        f"\n[t={data.time:.4f}] startup: "
                        f"{previous_startup_stage} -> {startup_state.stage}"
                    )

                # ------------------------------------------------
                # Enter INITIAL_DS: create the directional support plan.
                # ------------------------------------------------
                if (
                    startup_state.stage == STAGE_INITIAL_DS
                    and support_points is None
                ):
                    current = get_current_state_mpc(model, data)

                    support_points, last_foothold_diagnostics = (
                        build_initial_support_plan(
                            initial_ds_phase=startup_output.initial_ds_phase,
                            first_swing_foot=startup_output.first_swing_foot,
                            W_p_L6=W_p_L6,
                            W_p_R6=W_p_R6,
                            current_state=current.x,
                            home_CoM=home_CoM,
                            home_L6=home_L6,
                            home_R6=home_R6,
                        )
                    )

                    print(
                        f"[t={data.time:.4f}] INITIAL_DS support plan created. "
                        f"First target={support_points[2]}"
                    )

                # ------------------------------------------------
                # After the minimum INITIAL_DS weight-shift time, command
                # the first foot to lift ONLY when the measured CoM has
                # actually shifted far enough toward the support foot.
                #
                # IMPORTANT:
                #
                # The nominal timer alone must never force physical liftoff.
                # ------------------------------------------------
                if (
                    startup_state.stage == STAGE_INITIAL_DS
                    and not startup_swing_command_active
                ):
                    current = (
                        current
                        if current is not None
                        else get_current_state_mpc(model, data)
                    )

                    feet_mid_y = 0.5 * (
                        float(W_p_L6[1])
                        +
                        float(W_p_R6[1])
                    )

                    if FIRST_SWING_FOOT == "R":
                        # Right foot swings -> left foot is support.
                        # Left foot is on the positive World-y side.
                        support_direction_y = +1.0
                    else:
                        # Left foot swings -> right foot is support.
                        # Right foot is on the negative World-y side.
                        support_direction_y = -1.0

                    signed_com_shift = (
                        support_direction_y
                        *
                        (
                            float(current.x[1])
                            -
                            feet_mid_y
                        )
                    )

                    shift_time_ready = bool(
                        startup_output.t_initial_ds
                        >= INITIAL_DS_SHIFT_TIME - 1e-12
                    )

                    com_ready = bool(
                        signed_com_shift
                        >= STARTUP_COM_SHIFT_MIN
                    )

                    if shift_time_ready and com_ready:
                        startup_swing_command_active = True
                        startup_swing_command_start_time = float(data.time)

                        # Print the first physical-liftoff diagnostic immediately
                        # after the swing controller produces its first result.
                        next_liftoff_diagnostic_time = float(data.time)

                        if FIRST_SWING_FOOT == "R":
                            startup_command_liftoff_position = W_p_R6.copy()
                        else:
                            startup_command_liftoff_position = W_p_L6.copy()

                        reset_swing_joint_velocity_reference(
                            FIRST_SWING_FOOT
                        )

                        force_mpc_update = True

                        print(
                            f"[t={data.time:.4f}] first swing command enabled: "
                            f"foot={FIRST_SWING_FOOT} | "
                            f"CoM_y={current.x[1]:.5f} | "
                            f"signed_shift={signed_com_shift:.5f}"
                        )

                # ------------------------------------------------
                # REAL first LO -> handoff to cyclic gait manager.
                # ------------------------------------------------
                if startup_state.stage == STAGE_COMPLETE:
                    if support_points is None:
                        raise RuntimeError(
                            "Startup complete nhưng support_points chưa tồn tại."
                        )

                    current = (
                        current
                        if current is not None
                        else get_current_state_mpc(model, data)
                    )

                    # Replace the incoming support with its realized position.
                    if FIRST_SWING_FOOT == "R":
                        # [R_outgoing, L_incoming, R_target, ...]
                        support_points[1] = W_p_L6.copy()
                    else:
                        # [L_outgoing, R_incoming, L_target, ...]
                        support_points[1] = W_p_R6.copy()

                    # Remove outgoing support. The queue now begins at the
                    # physical stance foot of LSS/RSS.
                    support_points.pop(0)

                    # Recompute the first cyclic swing target from the realized
                    # stance foot / current body pose.
                    support_points, last_foothold_diagnostics = (
                        plan_bounded_support_point(
                            support_points=support_points,
                            current_phase=startup_output.handoff_phase,
                            target_index=1,
                            swing_foot=startup_output.first_swing_foot,
                            current_state=current.x,
                            home_CoM=home_CoM,
                            home_L6=home_L6,
                            home_R6=home_R6,
                        )
                    )

                    gait_state = initialize_gait_from_startup(
                        startup_output=startup_output,
                        c_L=c_L,
                        c_R=c_R,
                        W_p_L6=W_p_L6,
                        W_p_R6=W_p_R6,
                    )

                    W_p_foothold_L, W_p_foothold_R = (
                        get_gait_foothold_inputs(
                            gait_state.current_phase,
                            support_points,
                        )
                    )

                    gait_state, gait_output = update_gait_manager(
                        state=gait_state,
                        c_L=c_L,
                        c_R=c_R,
                        dt_elapsed=0.0,
                        timing=TIMING,
                        W_p_L6=W_p_L6,
                        W_p_R6=W_p_R6,
                        W_p_foothold_L=W_p_foothold_L,
                        W_p_foothold_R=W_p_foothold_R,
                    )

                    reset_swing_joint_velocity_reference(
                        startup_output.first_swing_foot
                    )

                    startup_swing_command_active = False
                    force_mpc_update = True

                    print(
                        f"[t={data.time:.4f}] startup handoff: "
                        f"{startup_output.event} -> {gait_output.current_phase}"
                    )

            # ====================================================
            # C. Cyclic contact-synchronized gait manager
            # ====================================================
            else:
                support_points = ensure_support_plan_length(
                    support_points,
                    current_phase=gait_state.current_phase,
                    target_length=SUPPORT_PLAN_LENGTH,
                )

                W_p_foothold_L, W_p_foothold_R = (
                    get_gait_foothold_inputs(
                        gait_state.current_phase,
                        support_points,
                    )
                )

                gait_state, gait_output = update_gait_manager(
                    state=gait_state,
                    c_L=c_L,
                    c_R=c_R,
                    dt_elapsed=dt_logic,
                    timing=TIMING,
                    W_p_L6=W_p_L6,
                    W_p_R6=W_p_R6,
                    W_p_foothold_L=W_p_foothold_L,
                    W_p_foothold_R=W_p_foothold_R,
                )

                if not gait_output.phase_contact_consistent:
                    data.ctrl[:] = 0.0
                    raise RuntimeError(
                        "Gait/contact inconsistency: "
                        f"phase={gait_output.current_phase}, "
                        f"contact=[{c_L},{c_R}], "
                        f"events={gait_output.events}"
                    )

                # ------------------------------------------------
                # CYCLIC DS -> SS TRANSITION POLICY
                #
                # The FSM remains contact-driven. This layer only creates the
                # physical conditions required for the expected LO event.
                # ------------------------------------------------
                if (
                    gait_output.current_phase in ("DS_L_TO_R", "DS_R_TO_L")
                    and not cyclic_liftoff_active
                ):
                    current = (
                        current
                        if current is not None
                        else get_current_state_mpc(model, data)
                    )

                    transfer_guard = evaluate_weight_transfer_guard(
                        phase=gait_output.current_phase,
                        t_phase=gait_output.t_phase,
                        current_state=current.x,
                        W_p_L6=W_p_L6,
                        W_p_R6=W_p_R6,
                    )

                    if transfer_guard["ready"]:
                        cyclic_liftoff_active = True
                        cyclic_liftoff_ds_phase = gait_output.current_phase
                        cyclic_liftoff_foot = transfer_guard["outgoing_foot"]
                        cyclic_liftoff_start_time = float(data.time)

                        if cyclic_liftoff_foot == "L":
                            cyclic_liftoff_position = W_p_L6.copy()
                        else:
                            cyclic_liftoff_position = W_p_R6.copy()

                        # New physical swing begins here. Latch a fresh
                        # posture/orientation reference for the outgoing leg.
                        reset_swing_low_level_state(
                            cyclic_liftoff_foot
                        )

                        # Re-solve MPC at the same x_k so the outgoing-foot
                        # current wrench becomes zero before applying torque.
                        force_mpc_update = True
                        next_liftoff_diagnostic_time = float(data.time)

                        print(
                            f"[t={data.time:.4f}] cyclic liftoff start: "
                            f"phase={gait_output.current_phase} | "
                            f"outgoing={cyclic_liftoff_foot} | "
                            f"incoming={transfer_guard['incoming_support']} | "
                            f"signed_shift={transfer_guard['signed_shift']:+.5f}"
                        )

                # Real LO has moved the physical FSM from DS to SS.
                #
                # Preserve the phase already reached during the liftoff
                # transition so the SS swing reference is continuous.
                if (
                    cyclic_liftoff_active
                    and gait_output.current_phase != cyclic_liftoff_ds_phase
                ):
                    expected_event = (
                        "LO_L"
                        if cyclic_liftoff_foot == "L"
                        else "LO_R"
                    )

                    if gait_output.transition_event != expected_event:
                        data.ctrl[:] = 0.0
                        raise RuntimeError(
                            "Unexpected event during cyclic liftoff: "
                            f"foot={cyclic_liftoff_foot}, "
                            f"expected={expected_event}, "
                            f"received={gait_output.transition_event}."
                        )

                    phase_at_lo = compute_cyclic_liftoff_phase(
                        current_time=data.time,
                        start_time=cyclic_liftoff_start_time,
                    )

                    swing_phase_offset[
                        cyclic_liftoff_foot
                    ] = phase_at_lo

                    print(
                        f"[t={data.time:.4f}] cyclic liftoff complete: "
                        f"foot={cyclic_liftoff_foot} | "
                        f"event={gait_output.transition_event} | "
                        f"swing phase continues from {phase_at_lo:.4f}"
                    )

                    cyclic_liftoff_active = False
                    cyclic_liftoff_ds_phase = None
                    cyclic_liftoff_foot = None
                    cyclic_liftoff_start_time = None
                    cyclic_liftoff_position = None

                T_phase_nominal = float(
                    get_nominal_phase_duration(
                        gait_output.current_phase,
                        TIMING,
                    )
                )

                if (
                    gait_output.t_phase
                    > T_phase_nominal + PHASE_EVENT_TIMEOUT_MARGIN
                ):
                    data.ctrl[:] = 0.0
                    raise RuntimeError(
                        "Expected gait event timeout: "
                        f"phase={gait_output.current_phase}, "
                        f"expected={gait_output.expected_event}, "
                        f"t_phase={gait_output.t_phase:.4f} s"
                    )

                if gait_output.transitioned:
                    force_mpc_update = True

                    current = get_current_state_mpc(model, data)

                    support_points, last_foothold_diagnostics = (
                        update_support_plan_after_gait_transition(
                            support_points=support_points,
                            gait_output=gait_output,
                            W_p_L6=W_p_L6,
                            W_p_R6=W_p_R6,
                            current_state=current.x,
                            home_CoM=home_CoM,
                            home_L6=home_L6,
                            home_R6=home_R6,
                        )
                    )

                    if gait_output.transition_event in ("TD_L", "TD_R"):
                        touched_foot = (
                            "L"
                            if gait_output.transition_event == "TD_L"
                            else "R"
                        )

                        # This swing cycle is complete.
                        swing_phase_offset[touched_foot] = 0.0
                        reset_swing_joint_velocity_reference(touched_foot)

                    print(
                        f"\n[t={data.time:.4f}] gait transition: "
                        f"event={gait_output.transition_event} -> "
                        f"phase={gait_output.current_phase}"
                    )

            # ====================================================
            # D. Current mode / phase descriptors after logic update
            # ====================================================
            if gait_state is None:
                if startup_state.stage == STAGE_SETTLE:
                    control_mode = "SETTLE"
                    current_phase = "STANDING_DS"
                    current_t_phase = startup_output.t_settle
                    current_swing_foot = None
                    current_swing_phase = None

                elif startup_state.stage == STAGE_INITIAL_DS:
                    control_mode = "INITIAL_DS"
                    current_phase = startup_output.initial_ds_phase
                    current_t_phase = startup_output.t_initial_ds
                    current_swing_foot = (
                        FIRST_SWING_FOOT
                        if startup_swing_command_active
                        else None
                    )
                    current_swing_phase = None

                else:
                    raise RuntimeError(
                        f"Unexpected startup stage: {startup_state.stage}"
                    )

            else:
                control_mode = "CYCLIC"
                current_phase = gait_output.current_phase
                current_t_phase = gait_output.t_phase

                if cyclic_liftoff_active:
                    current_swing_foot = cyclic_liftoff_foot
                    current_swing_phase = compute_cyclic_liftoff_phase(
                        current_time=data.time,
                        start_time=cyclic_liftoff_start_time,
                    )

                elif gait_output.swing_foot is not None:
                    current_swing_foot = gait_output.swing_foot
                    current_swing_phase = continue_swing_phase(
                        raw_swing_phase=gait_output.swing_phase,
                        phase_at_lo=swing_phase_offset[
                            gait_output.swing_foot
                        ],
                    )

                else:
                    current_swing_foot = None
                    current_swing_phase = None

            # ====================================================
            # E. MPC update schedule
            #
            # Contact transitions force an immediate rebuild. The next regular
            # update is then scheduled DT_MPC_UPDATE later.
            # ====================================================
            periodic_mpc_due = bool(
                data.time >= next_mpc_update_time - 1e-12
            )

            is_mpc_update = bool(
                periodic_mpc_due or force_mpc_update
            )

            mpc_updated_this_step = False

            if is_mpc_update:
                mpc_compute_start = time.perf_counter()

                current = (
                    current
                    if current is not None
                    else get_current_state_mpc(model, data)
                )

                # ------------------------------------------------
                # E1. SETTLE: fixed standing reference, both feet stance.
                # ------------------------------------------------
                if control_mode == "SETTLE":
                    W_p_CoM_ref = standing_reference["W_p_CoM_ref"].copy()
                    W_v_CoM_ref = standing_reference["W_v_CoM_ref"].copy()

                    W_p_L6_plan = np.tile(
                        W_p_L6.reshape(1, 3),
                        (N, 1),
                    )
                    W_p_R6_plan = np.tile(
                        W_p_R6.reshape(1, 3),
                        (N, 1),
                    )

                    contact_schedule = np.ones((N, 2), dtype=int)

                    propagate_nominal_com = False
                    W_v_CoM_nominal_hold[:] = 0.0

                # ------------------------------------------------
                # E2. INITIAL_DS / CYCLIC: DCM -> CoM reference.
                # ------------------------------------------------
                else:
                    if support_points is None:
                        raise RuntimeError(
                            "Walking reference yêu cầu support_points."
                        )

                    support_points = ensure_support_plan_length(
                        support_points,
                        current_phase=current_phase,
                        target_length=SUPPORT_PLAN_LENGTH,
                    )

                    t_reference, T_phase_nominal, hold_nominal_at_boundary = (
                        get_reference_phase_time(
                            current_phase,
                            current_t_phase,
                        )
                    )

                    translational_reference = (
                        build_mpc_translational_reference(
                            current_phase=current_phase,
                            t_phase=t_reference,
                            timing=TIMING,
                            support_points=np.asarray(
                                support_points,
                                dtype=float,
                            ),
                            W_p_CoM_nominal_current=(
                                W_p_CoM_nominal_current
                            ),
                            com_height=COM_HEIGHT,
                            gravity=GRAVITY,
                            horizon=N,
                            dt_mpc=DT_MPC,
                        )
                    )

                    W_p_CoM_ref = translational_reference[
                        "W_p_CoM_ref"
                    ].copy()
                    W_v_CoM_ref = translational_reference[
                        "W_v_CoM_ref"
                    ].copy()

                    # Interval-start phase preview for SRBD contact / foot plan.
                    phase_preview = build_phase_preview(
                        current_phase=current_phase,
                        t_phase=current_t_phase,
                        timing=TIMING,
                        horizon=N,
                        dt_mpc=DT_MPC,
                    )

                    phase_sequence = phase_preview["phase_sequence"]

                    foot_plan = build_foot_plan_horizon(
                        current_phase=current_phase,
                        phase_sequence=phase_sequence,
                        support_points=np.asarray(
                            support_points,
                            dtype=float,
                        ),
                    )

                    W_p_L6_plan = foot_plan["W_p_L6_plan"]
                    W_p_R6_plan = foot_plan["W_p_R6_plan"]

                    contact_schedule = build_contact_schedule(
                        current_phase=current_phase,
                        t_phase=current_t_phase,
                        timing=TIMING,
                        horizon=N,
                        dt_mpc=DT_MPC,
                    )

                    # During the first actual liftoff command, do not let the
                    # current QP input load the foot that PD is trying to lift.
                    # Only j=0 is overridden; future contacts remain nominal.
                    if (
                        control_mode == "INITIAL_DS"
                        and startup_swing_command_active
                    ):
                        if FIRST_SWING_FOOT == "R":
                            contact_schedule[0, :] = np.array([1, 0])
                        else:
                            contact_schedule[0, :] = np.array([0, 1])

                    # Cyclic DS -> SS transition:
                    # unload only the outgoing foot in the CURRENT QP input.
                    # The physical contact schedule / FSM is still changed
                    # only after contact_switch detects the real LO edge.
                    if (
                        control_mode == "CYCLIC"
                        and cyclic_liftoff_active
                    ):
                        if cyclic_liftoff_foot == "L":
                            contact_schedule[0, :] = np.array([0, 1])
                        elif cyclic_liftoff_foot == "R":
                            contact_schedule[0, :] = np.array([1, 0])
                        else:
                            raise RuntimeError(
                                "cyclic_liftoff_active nhưng foot không hợp lệ."
                            )

                    if hold_nominal_at_boundary:
                        propagate_nominal_com = False
                        W_v_CoM_nominal_hold[:] = 0.0
                    else:
                        propagate_nominal_com = True
                        W_v_CoM_nominal_hold = np.asarray(
                            translational_reference[
                                "W_v_CoM_nominal_current"
                            ],
                            dtype=float,
                        ).reshape(3).copy()

                # ------------------------------------------------
                # E3. LTV SRBD + QP
                # ------------------------------------------------
                (
                    U_opt,
                    u0_opt,
                    solver_info,
                    post_diagnostics,
                    srbd_reference,
                ) = solve_ltv_mpc(
                    setup=setup,
                    current=current,
                    W_p_CoM_ref=W_p_CoM_ref,
                    W_v_CoM_ref=W_v_CoM_ref,
                    W_p_L6_plan=W_p_L6_plan,
                    W_p_R6_plan=W_p_R6_plan,
                    contact_schedule=contact_schedule,
                )

                if u0_opt is None:
                    data.ctrl[:] = 0.0

                    raise RuntimeError(
                        "Walking QP không trả về nghiệm hợp lệ.\n"
                        f"mode={control_mode}\n"
                        f"phase={current_phase}\n"
                        f"solver_info={solver_info}\n"
                        f"post_diagnostics={post_diagnostics}"
                    )

                held_u0_MPC = np.asarray(
                    u0_opt,
                    dtype=float,
                ).reshape(12).copy()

                if not np.all(np.isfinite(held_u0_MPC)):
                    raise RuntimeError("u0_MPC chứa NaN hoặc Inf.")

                last_solver_info = solver_info
                last_post_diagnostics = post_diagnostics
                last_W_p_CoM_ref = W_p_CoM_ref.copy()

                last_mpc_compute_time = (
                    time.perf_counter() - mpc_compute_start
                )

                next_mpc_update_time = (
                    float(data.time) + DT_MPC_UPDATE
                )

                mpc_updated_this_step = True

            # ====================================================
            # F. World wrench -> current Jacobian stance torque
            #
            # This is evaluated every physics step using q_k.
            # ====================================================
            tau_mpc, _, _ = get_joint_torque_from_MPC_wrench(
                model,
                data,
                held_u0_MPC,
            )

            # ====================================================
            # G. Swing low-level control / torque mixing
            # ====================================================
            swing_result = None

            if (
                control_mode == "INITIAL_DS"
                and startup_swing_command_active
            ):
                if support_points is None:
                    raise RuntimeError(
                        "First swing command yêu cầu support plan."
                    )

                elapsed_command = max(
                    float(data.time) - float(startup_swing_command_start_time),
                    0.0,
                )

                ramp_ratio = min(
                    elapsed_command / STARTUP_PRELIFTOFF_RAMP_TIME,
                    1.0,
                )

                preliftoff_phase = (
                    STARTUP_PRELIFTOFF_PHASE_MAX * ramp_ratio
                )

                # ------------------------------------------------
                # PRE-LIFTOFF:
                #
                # While the selected swing foot is still physically in
                # contact with the floor, do NOT command horizontal motion.
                #
                # The pre-liftoff trajectory therefore keeps:
                #
                #     x_des = x_liftoff
                #     y_des = y_liftoff
                #
                # and uses only the vertical part of the quintic swing
                # trajectory to unload / lift the foot.
                #
                # The planned forward foothold support_points[2] becomes
                # active only after the real LO event and cyclic handoff.
                # ------------------------------------------------
                preliftoff_target = (
                    startup_command_liftoff_position.copy()
                )

                swing_result = compute_swing_low_level_control(
                    model=model,
                    data=data,
                    swing_foot=FIRST_SWING_FOOT,
                    W_p_liftoff=startup_command_liftoff_position,
                    W_p_foothold=preliftoff_target,
                    swing_phase=float(preliftoff_phase),
                    T_swing=T_SS,
                    swing_height=SWING_HEIGHT,
                )

                tau_command = mix_walking_joint_torque(
                    tau_mpc=tau_mpc,
                    swing_foot=FIRST_SWING_FOOT,
                    tau_swing=swing_result.tau_swing,
                )

                current_swing_phase = float(preliftoff_phase)

            elif (
                control_mode == "CYCLIC"
                and cyclic_liftoff_active
            ):
                if cyclic_liftoff_foot not in ("L", "R"):
                    raise RuntimeError(
                        "Cyclic liftoff transition thiếu outgoing foot."
                    )

                if cyclic_liftoff_position is None:
                    raise RuntimeError(
                        "Cyclic liftoff transition thiếu latched foot position."
                    )

                liftoff_phase = compute_cyclic_liftoff_phase(
                    current_time=data.time,
                    start_time=cyclic_liftoff_start_time,
                )

                # DS liftoff transition is vertical-only:
                #     x_des = x_start
                #     y_des = y_start
                # The regular planned foothold becomes active after real LO.
                vertical_target = cyclic_liftoff_position.copy()

                swing_result = compute_swing_low_level_control(
                    model=model,
                    data=data,
                    swing_foot=cyclic_liftoff_foot,
                    W_p_liftoff=cyclic_liftoff_position,
                    W_p_foothold=vertical_target,
                    swing_phase=liftoff_phase,
                    T_swing=T_SS,
                    swing_height=SWING_HEIGHT,
                )

                tau_command = mix_walking_joint_torque(
                    tau_mpc=tau_mpc,
                    swing_foot=cyclic_liftoff_foot,
                    tau_swing=swing_result.tau_swing,
                )

                current_swing_phase = liftoff_phase

            elif (
                control_mode == "CYCLIC"
                and gait_output.swing_foot is not None
            ):
                if gait_output.W_p_liftoff is None:
                    raise RuntimeError(
                        "Cyclic swing thiếu W_p_liftoff."
                    )

                if gait_output.W_p_foothold is None:
                    raise RuntimeError(
                        "Cyclic swing thiếu W_p_foothold."
                    )

                effective_swing_phase = continue_swing_phase(
                    raw_swing_phase=gait_output.swing_phase,
                    phase_at_lo=swing_phase_offset[
                        gait_output.swing_foot
                    ],
                )

                swing_result = compute_swing_low_level_control(
                    model=model,
                    data=data,
                    swing_foot=gait_output.swing_foot,
                    W_p_liftoff=gait_output.W_p_liftoff,
                    W_p_foothold=gait_output.W_p_foothold,
                    swing_phase=effective_swing_phase,
                    T_swing=T_SS,
                    swing_height=SWING_HEIGHT,
                )

                current_swing_phase = effective_swing_phase

                tau_command = mix_walking_joint_torque(
                    tau_mpc=tau_mpc,
                    swing_foot=gait_output.swing_foot,
                    tau_swing=swing_result.tau_swing,
                )

            else:
                tau_command = np.asarray(
                    tau_mpc,
                    dtype=float,
                ).reshape(10).copy()

            if not np.all(np.isfinite(tau_command)):
                raise RuntimeError("tau_command chứa NaN hoặc Inf.")

            # ====================================================
            # G2. Physical liftoff diagnostics
            #
            # Read-only:
            #   - no controller state modification,
            #   - no contact-state modification,
            #   - no mj_step().
            # ====================================================
            if swing_result is not None:
                last_swing_result = swing_result

                liftoff_debug_active = bool(
                    (
                        control_mode == "INITIAL_DS"
                        and startup_swing_command_active
                    )
                    or
                    (
                        control_mode == "CYCLIC"
                        and cyclic_liftoff_active
                    )
                )

                if (
                    liftoff_debug_active
                    and float(data.time)
                    >= next_liftoff_diagnostic_time - 1e-12
                ):
                    current = (
                        current
                        if current is not None
                        else get_current_state_mpc(model, data)
                    )

                    diagnostic_label = (
                        "CYCLIC LIFTOFF DEBUG"
                        if (
                            control_mode == "CYCLIC"
                            and cyclic_liftoff_active
                        )
                        else
                        "PRE-LIFTOFF DEBUG"
                    )

                    print_liftoff_diagnostics(
                        data=data,
                        control_mode=control_mode,
                        phase=current_phase,
                        contact_state=contact_state,
                        W_p_L6=W_p_L6,
                        W_p_R6=W_p_R6,
                        swing_result=swing_result,
                        held_u0_MPC=held_u0_MPC,
                        tau_command=tau_command,
                        current=current,
                        label=diagnostic_label,
                    )

                    next_liftoff_diagnostic_time = (
                        float(data.time)
                        +
                        LIFTOFF_DIAGNOSTIC_PERIOD
                    )

            # ====================================================
            # H. Apply u_k / tau_k to MuJoCo -- no step yet
            # ====================================================
            apply_joint_torque(
                model,
                data,
                tau_command,
            )

            # ====================================================
            # I. Log / print the x_k control result before mj_step
            # ====================================================
            if mpc_updated_this_step:
                current = (
                    current
                    if current is not None
                    else get_current_state_mpc(model, data)
                )

                logger.log_step(
                    time=float(data.time),
                    x=current.x,
                    u0=held_u0_MPC,
                    tau=tau_command,
                    mpc_compute_time=last_mpc_compute_time,
                    qp_solve_time=float(
                        last_solver_info.get("solve_time", 0.0)
                    ),
                )

                if mpc_iteration % PRINT_EVERY_N_MPC == 0:
                    print_status(
                        mpc_iteration=mpc_iteration,
                        data=data,
                        mode=control_mode,
                        phase=current_phase,
                        t_phase=current_t_phase,
                        c_L=c_L,
                        c_R=c_R,
                        swing_foot=current_swing_foot,
                        swing_phase=current_swing_phase,
                        current=current,
                        W_p_CoM_ref=last_W_p_CoM_ref,
                        held_u0_MPC=held_u0_MPC,
                        tau_command=tau_command,
                        solver_info=last_solver_info,
                        post_diagnostics=last_post_diagnostics,
                        support_points=support_points,
                    )

                mpc_iteration += 1

            # ====================================================
            # J. THE ONLY PHYSICS ADVANCE
            #
            # x_k -> apply u_k -> mj_step -> x_{k+1}
            # ====================================================
            mujoco.mj_step(model, data)
            physics_step += 1

            # ----------------------------------------------------
            # Nominal reference state advances with simulation time, not with
            # the 0.05-s MPC prediction grid. Velocity is refreshed at each
            # MPC solve and held between solves.
            # ----------------------------------------------------
            if propagate_nominal_com:
                W_p_CoM_nominal_current[0:2] += (
                    W_v_CoM_nominal_hold[0:2] * DT_SIM
                )

            W_p_CoM_nominal_current[2] = COM_HEIGHT

            viewer.sync()

            # Optional wall-clock pacing.
            wall_elapsed = time.perf_counter() - wall_step_start
            sleep_time = DT_SIM - wall_elapsed

            if sleep_time > 0.0:
                time.sleep(sleep_time)

    # ========================================================
    # 10. FINAL SUMMARY
    # ========================================================
    final_state = get_current_state_mpc(model, data)
    final_contact = get_contact_state(
        model=model,
        data=data,
        contact_map=setup.contact_map,
    )

    print("\n" + "=" * 78)
    print("WALKING SIMULATION FINISHED")
    print("=" * 78)
    print("simulation time =", data.time, "s")
    print("physics steps   =", physics_step)
    print("MPC updates     =", mpc_iteration)
    print("final contact   =", [final_contact["c_L"], final_contact["c_R"]])
    print("final CoM       =", final_state.x[0:3])
    print("final Euler     =", final_state.x[3:6])
    print("final velocity  =", final_state.x[6:9])
    print("nominal CoM     =", W_p_CoM_nominal_current)

    if gait_output is not None:
        print("final gait phase=", gait_output.current_phase)
        print("final t_phase   =", gait_output.t_phase)
        print("expected event  =", gait_output.expected_event)

    logger.print_summary()


if __name__ == "__main__":
    main()
