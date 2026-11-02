from pathlib import Path
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
SIMULATION_DIR = Path(__file__).resolve().parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent
for directory in (MPC_SRBD_DIR, SIMULATION_DIR, MPC_SRBD_DIR / 'State', MPC_SRBD_DIR / 'SRBD', MPC_SRBD_DIR / 'QP', MPC_SRBD_DIR / 'Kinematics', MPC_SRBD_DIR / 'Actuator', MPC_SRBD_DIR / 'Command', MPC_SRBD_DIR / 'DCM', MPC_SRBD_DIR / 'Trajectory_Planner'):
    directory_str = str(directory)
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)
from simulation_setup import initialize_simulation
from simulation_logger import SimulationLogger
from gait_manager import initialize_gait_manager, update_gait_manager
from swing_low_level_controller import reset_swing_low_level_state, reset_swing_joint_velocity_reference, _get_swing_joint_state, _get_swing_foot_rotation_matrix, _get_or_latch_swing_reference, _estimate_q_dot_swing_des, SWING_K_POSTURE, SWING_K_ROLL, SWING_K_PITCH, SWING_PHASE_ENDPOINT_TOLERANCE
from get_current_state_mpc import get_current_state_mpc
from get_contact_state import get_contact_state
from Command.gait_timing import build_gait_timing
from Command.contact_schedule import PHASE_DS_L_TO_R, PHASE_DS_R_TO_L, build_contact_preview
from Command.foothold_generator import HOME_NOMINAL_STEP_WIDTH, generate_foothold, generate_nominal_foothold, get_opposite_foot
from Command.foot_plan_horizon import build_foot_plan_horizon, build_moment_arm_horizon, get_initial_support_foot, get_support_foot_from_index
from Command.srbd_reference import build_srbd_reference
from DCM.mpc_reference import build_mpc_translational_reference
from build_SRBD_model import build_SRBD_model_sequence
from build_and_solve_QP import build_and_solve_QP
from W_wrench_to_joint_torque import get_joint_torque_from_MPC_wrench
from actuator_mapping import apply_joint_torque
from Actuator.swing_joint_PD import compute_swing_joint_PD
from Actuator.swing_joint_PD_params import KP_SWING, KD_SWING
np.set_printoptions(precision=7, suppress=True, linewidth=220)
XML_PATH = REPO_ROOT / 'xmls' / 'scene_flat_terrian_v3.xml'
N = 10
T_STEP = 0.5
N_SS = 8
N_DS = 2
TIMING = build_gait_timing(T_step=T_STEP, horizon=N, N_SS=N_SS, N_DS=N_DS)
DT_MPC = float(TIMING['dt_mpc'])
DT_HIGH_LEVEL = 0.01
T_SS = float(TIMING['T_SS'])
T_DS = float(TIMING['T_DS'])
FIRST_SWING_FOOT = 'R'
STEP_LENGTH = 0.01
STEP_WIDTH = HOME_NOMINAL_STEP_WIDTH
SWING_HEIGHT = 0.01
GROUND_HEIGHT = 0.0
SUPPORT_PLAN_LENGTH = 10
SIM_DURATION = 5.0
GRAVITY = 9.81
QP_MAX_ITER = 10000
QP_CONSTRAINT_TOLERANCE = 0.0005
PRINT_EVERY_N_HIGH_LEVEL = 20
SWING_LEG_JOINT_NAMES = {'L': ('L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle'), 'R': ('R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle')}

def get_site_id(model, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Không tìm thấy site '{site_name}'.")
    return int(site_id)

def get_foot_positions(data, L6_site_id, R6_site_id):
    W_p_L6 = np.asarray(data.site_xpos[L6_site_id], dtype=float).reshape(3).copy()
    W_p_R6 = np.asarray(data.site_xpos[R6_site_id], dtype=float).reshape(3).copy()
    if not np.all(np.isfinite(W_p_L6)):
        raise RuntimeError('W_p_L6 chứa NaN/Inf.')
    if not np.all(np.isfinite(W_p_R6)):
        raise RuntimeError('W_p_R6 chứa NaN/Inf.')
    return (W_p_L6, W_p_R6)

def get_leg_joint_state(model, data, foot):
    """
    Read current swing-leg q, qdot at physics rate.

    No IK, no trajectory planning, no mj_step().
    """
    foot = str(foot).upper()
    if foot not in ('L', 'R'):
        raise ValueError("foot phải là 'L' hoặc 'R'.")
    q = np.zeros(5, dtype=float)
    q_dot = np.zeros(5, dtype=float)
    for index, joint_name in enumerate(SWING_LEG_JOINT_NAMES[foot]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Không tìm thấy joint '{joint_name}'.")
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        q[index] = data.qpos[qpos_address]
        q_dot[index] = data.qvel[dof_address]
    return (q, q_dot)

def rotation_matrix_from_rpy(euler):
    roll, pitch, yaw = np.asarray(euler, dtype=float).reshape(3)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx

def get_kinematic_reference(current_state, swing_foot, home_CoM, home_L6, home_R6):
    """
    Existing flat-ground kinematic reference used by foothold_generator.
    """
    current_state = np.asarray(current_state, dtype=float).reshape(13)
    W_p_CoM = current_state[0:3]
    euler = current_state[3:6]
    if swing_foot == 'L':
        home_offset = home_L6 - home_CoM
    elif swing_foot == 'R':
        home_offset = home_R6 - home_CoM
    else:
        raise ValueError("swing_foot phải là 'L' hoặc 'R'.")
    return W_p_CoM + rotation_matrix_from_rpy(euler) @ home_offset

def mix_walking_joint_torque(tau_mpc, swing_foot=None, tau_swing=None):
    """
    DS:
        both legs use held MPC/virtual-work torque.

    LSS:
        left  = held MPC torque
        right = 2000-Hz swing joint PD

    RSS:
        left  = 2000-Hz swing joint PD
        right = held MPC torque
    """
    tau_mpc = np.asarray(tau_mpc, dtype=float).reshape(10)
    if swing_foot is None:
        return tau_mpc.copy()
    tau_swing = np.asarray(tau_swing, dtype=float).reshape(5)
    tau = tau_mpc.copy()
    if swing_foot == 'L':
        tau[0:5] = tau_swing
    elif swing_foot == 'R':
        tau[5:10] = tau_swing
    else:
        raise ValueError("swing_foot phải là 'L', 'R' hoặc None.")
    return tau

def compute_swing_reference_100hz(model, data, swing_foot, W_p_liftoff, W_p_foothold, swing_phase, T_swing, swing_height):
    """
    Build swing Cartesian trajectory + IK reference at 100 Hz.

    IMPORTANT:
        This function does NOT run joint PD.
        Joint PD is evaluated only in the 2000-Hz physics loop.
    """
    from Trajectory_Planner.quintic_swing_trajectory import evaluate_quintic_swing_trajectory
    from Kinematics.numerical_IK_hybrid import solve_numerical_IK_hybrid
    from Kinematics.numerical_IK_params import SIGMA_THRESHOLD, ALPHA_NEWTON, ETA_GRADIENT, IK_TOLERANCE, IK_MAX_ITERATIONS
    swing_phase = float(np.clip(swing_phase, 0.0, 1.0))
    trajectory = evaluate_quintic_swing_trajectory(W_p_liftoff=np.asarray(W_p_liftoff, dtype=float).reshape(3), W_p_foothold=np.asarray(W_p_foothold, dtype=float).reshape(3), swing_phase=swing_phase, T_swing=float(T_swing), swing_height=float(swing_height))
    q_swing, _ = _get_swing_joint_state(model=model, data=data, swing_foot=swing_foot)
    R_swing = _get_swing_foot_rotation_matrix(model=model, data=data, swing_foot=swing_foot)
    q_posture_ref, R_swing_ref, _ = _get_or_latch_swing_reference(q_swing=q_swing, R_swing=R_swing, swing_foot=swing_foot, swing_phase=swing_phase, current_time=float(data.time))
    q_swing_des, ik_diagnostics = solve_numerical_IK_hybrid(model=model, data=data, W_p_swing_des=np.asarray(trajectory['W_p_swing_des'], dtype=float).reshape(3), swing_foot=swing_foot, sigma_threshold=SIGMA_THRESHOLD, alpha_newton=ALPHA_NEWTON, eta_gradient=ETA_GRADIENT, tolerance=IK_TOLERANCE, max_iterations=IK_MAX_ITERATIONS, q_posture_ref=q_posture_ref, k_posture=SWING_K_POSTURE, R_swing_ref=R_swing_ref, k_roll=SWING_K_ROLL, k_pitch=SWING_K_PITCH)
    if q_swing_des is None:
        raise RuntimeError('Hybrid IK trả về q_swing_des=None.')
    q_swing_des = np.asarray(q_swing_des, dtype=float).reshape(5)
    if swing_phase <= SWING_PHASE_ENDPOINT_TOLERANCE:
        reset_swing_joint_velocity_reference(swing_foot)
        q_dot_swing_des = _estimate_q_dot_swing_des(q_swing_des=q_swing_des, swing_foot=swing_foot, current_time=float(data.time))
    elif swing_phase >= 1.0 - SWING_PHASE_ENDPOINT_TOLERANCE:
        q_dot_swing_des = np.zeros(5, dtype=float)
        reset_swing_joint_velocity_reference(swing_foot)
    else:
        q_dot_swing_des = _estimate_q_dot_swing_des(q_swing_des=q_swing_des, swing_foot=swing_foot, current_time=float(data.time))
    return {'q_swing_des': q_swing_des.copy(), 'q_dot_swing_des': np.asarray(q_dot_swing_des, dtype=float).reshape(5).copy(), 'W_p_swing_des': np.asarray(trajectory['W_p_swing_des'], dtype=float).reshape(3).copy(), 'W_v_swing_des': np.asarray(trajectory['W_v_swing_des'], dtype=float).reshape(3).copy(), 'ik_diagnostics': ik_diagnostics}

def get_initial_ds_phase(first_swing_foot):
    first_swing_foot = str(first_swing_foot).upper()
    if first_swing_foot == 'R':
        return PHASE_DS_R_TO_L
    if first_swing_foot == 'L':
        return PHASE_DS_L_TO_R
    raise ValueError("FIRST_SWING_FOOT phải là 'L' hoặc 'R'.")

def ensure_support_plan_length(support_points, current_phase, target_length=SUPPORT_PLAN_LENGTH):
    """
    Extend ordered support queue with existing nominal foothold generator.
    """
    points = [np.asarray(point, dtype=float).reshape(3).copy() for point in support_points]
    if len(points) < 2:
        raise ValueError('support_points cần ít nhất 2 điểm.')
    initial_support_foot = get_initial_support_foot(current_phase)
    while len(points) < int(target_length):
        last_index = len(points) - 1
        last_foot = get_support_foot_from_index(initial_support_foot, last_index)
        next_foot = get_opposite_foot(last_foot)
        next_point = generate_nominal_foothold(W_p_support=points[-1], swing_foot=next_foot, step_length=STEP_LENGTH, step_width=STEP_WIDTH, ground_height=GROUND_HEIGHT)
        points.append(np.asarray(next_point, dtype=float).reshape(3).copy())
    return points

def build_initial_support_plan(initial_phase, first_swing_foot, W_p_L6, W_p_R6, current_state, home_CoM, home_L6, home_R6):
    """
    No half-step.

    Right first:
        DS_R_TO_L
        [R_current, L_current, R_full_target, L_next, ...]

    Left first:
        DS_L_TO_R
        [L_current, R_current, L_full_target, R_next, ...]
    """
    if first_swing_foot == 'R':
        if initial_phase != PHASE_DS_R_TO_L:
            raise RuntimeError('Right-first phải bắt đầu DS_R_TO_L.')
        points = [W_p_R6.copy(), W_p_L6.copy()]
        W_p_support = W_p_L6
    elif first_swing_foot == 'L':
        if initial_phase != PHASE_DS_L_TO_R:
            raise RuntimeError('Left-first phải bắt đầu DS_L_TO_R.')
        points = [W_p_L6.copy(), W_p_R6.copy()]
        W_p_support = W_p_R6
    else:
        raise ValueError("first_swing_foot phải là 'L' hoặc 'R'.")
    W_p_kinematic_reference = get_kinematic_reference(current_state=current_state, swing_foot=first_swing_foot, home_CoM=home_CoM, home_L6=home_L6, home_R6=home_R6)
    first_target, diagnostics = generate_foothold(W_p_support=W_p_support, W_p_kinematic_reference=W_p_kinematic_reference, swing_foot=first_swing_foot, step_length=STEP_LENGTH, step_width=STEP_WIDTH, correction_xy=None, ground_height=GROUND_HEIGHT)
    points.append(np.asarray(first_target, dtype=float).reshape(3).copy())
    points = ensure_support_plan_length(support_points=points, current_phase=initial_phase, target_length=SUPPORT_PLAN_LENGTH)
    return (points, diagnostics)

def get_gait_foothold_inputs(phase_cmd, support_points):
    """
    Foothold target supplied to gait_manager BEFORE its update.

    DS:
        index 2 is the foot that will begin swing after timed DS -> SS.

    SS:
        index 1 is the current swing touchdown target.
    """
    points = [np.asarray(point, dtype=float).reshape(3).copy() for point in support_points]
    if len(points) < 3:
        raise ValueError('support_points cần ít nhất 3 điểm.')
    if phase_cmd == 'LSS':
        return (None, points[1].copy())
    if phase_cmd == 'RSS':
        return (points[1].copy(), None)
    if phase_cmd == 'DS_L_TO_R':
        return (points[2].copy(), None)
    if phase_cmd == 'DS_R_TO_L':
        return (None, points[2].copy())
    raise ValueError(f'Unsupported phase_cmd={phase_cmd!r}.')

def update_support_queue_after_command_transition(support_points, gait_output):
    """
    Queue follows COMMAND phase topology.

    DS -> SS:
        remove outgoing support immediately because phase_cmd has already
        changed to the new SS mode.

        Example:
            before:
                DS_R_TO_L
                [R_outgoing, L_incoming, R_target, ...]

            after command transition to LSS:
                [L_support, R_target, L_next, ...]

    SS -> DS:
        no queue shift.
        100-Hz realized-foot re-anchoring will replace incoming support
        geometry with the measured touchdown position.
    """
    points = [np.asarray(point, dtype=float).reshape(3).copy() for point in support_points]
    if not gait_output.transitioned:
        return points
    if gait_output.transition_reason == 'DS_TIME_COMPLETE':
        if len(points) < 3:
            raise RuntimeError('Support queue quá ngắn tại DS -> SS.')
        points.pop(0)
    elif gait_output.transition_reason == 'TOUCHDOWN_CONFIRMED':
        pass
    else:
        raise RuntimeError(f'Unknown gait transition reason: {gait_output.transition_reason!r}')
    return ensure_support_plan_length(support_points=points, current_phase=gait_output.phase_cmd, target_length=SUPPORT_PLAN_LENGTH)

def solve_ltv_mpc(setup, current, W_p_CoM_ref, W_v_CoM_ref, W_p_L6_plan, W_p_R6_plan, contact_schedule_mpc):
    """
    contact_schedule_mpc uses measured physical contact at row 0 and
    commanded/planned contact for future rows.
    """
    srbd_reference = build_srbd_reference(W_p_CoM_ref=W_p_CoM_ref, W_v_CoM_ref=W_v_CoM_ref, W_euler_CoM_ref=np.zeros(3, dtype=float), W_w_CoM_ref=np.zeros(3, dtype=float))
    W_r_L_sequence, W_r_R_sequence = build_moment_arm_horizon(W_p_CoM_ref=srbd_reference['W_p_CoM_ref'], W_p_L6_plan=W_p_L6_plan, W_p_R6_plan=W_p_R6_plan)
    Phi, Gamma = build_SRBD_model_sequence(Ew_inverse=current.Ew_inverse, mass_CoM=setup.mass_CoM, W_inertia_CoM=setup.W_inertia_CoM_q0, W_r_L_sequence=W_r_L_sequence, W_r_R_sequence=W_r_R_sequence, dt_mpc=DT_MPC, horizon=N)
    return build_and_solve_QP(qp_solver=setup.qp_solver, Phi=Phi, Gamma=Gamma, x_current=current.x, X_ref=srbd_reference['X_ref'], N=N, L6_R_W=current.L6_IK_R_W, R6_R_W=current.R6_IK_R_W, contact_schedule=contact_schedule_mpc, print_diagnostics=False) + (srbd_reference,)

def print_status(high_level_iteration, data, gait_output, current, W_p_CoM_ref, held_u0_MPC, tau_command, solver_info, post_diagnostics, support_points, contact_schedule_cmd, contact_schedule_mpc):
    max_violation = None
    if post_diagnostics is not None:
        max_violation = post_diagnostics.get('max_violation')

    print('\n' + '-' * 82)
    print(
        f't={data.time:.4f} s | HL={high_level_iteration} | '
        f'phase_cmd={gait_output.phase_cmd} | '
        f't_cmd={gait_output.t_phase_cmd:.4f}'
    )
    print(
        'contact_meas=', gait_output.contact_meas,
        '| contact_cmd=', gait_output.contact_cmd,
        '| achieved=', gait_output.contact_cmd_achieved,
    )
    print(
        'events=', gait_output.events,
        '| pending_LO=', gait_output.pending_liftoff_confirmation,
        '| waiting_TD=', gait_output.waiting_for_contact,
    )
    print(
        'swing=', gait_output.swing_foot,
        '| swing_phase=', gait_output.swing_phase,
        '| swing_start=', gait_output.W_p_swing_start,
        '| swing_target=', gait_output.W_p_foothold,
    )
    print(
        'contact CMD horizon first rows =\n',
        np.asarray(contact_schedule_cmd[:min(4, len(contact_schedule_cmd))], dtype=int),
    )
    print(
        'contact MPC horizon first rows =\n',
        np.asarray(contact_schedule_mpc[:min(4, len(contact_schedule_mpc))], dtype=int),
    )
    print('CoM actual =', current.x[0:3])
    print('CoM ref[0] =', W_p_CoM_ref[0])
    print(
        'Fz MPC row0 [L,R] =',
        np.array([held_u0_MPC[2], held_u0_MPC[8]], dtype=float),
        'N',
    )
    print('tau =', tau_command)
    print(
        'QP status =', solver_info.get('status'),
        '| solve_time =', solver_info.get('solve_time'),
        '| max_violation =', max_violation,
    )
    print(
        'support queue first points =\n',
        np.asarray(support_points[:min(4, len(support_points))], dtype=float),
    )

def main():
    setup = initialize_simulation(xml_path=XML_PATH, horizon=N, dt_mpc=DT_MPC, dt_mpc_update=DT_HIGH_LEVEL, home_key_name='home', qp_verbose=False, qp_eps_abs=1e-05, qp_eps_rel=1e-05, qp_max_iter=QP_MAX_ITER, qp_polishing=True, qp_enable_warm_start=True, qp_constraint_tolerance=QP_CONSTRAINT_TOLERANCE, print_initialization=True)
    model = setup.model
    data = setup.data
    DT_SIM = float(setup.dt_sim)
    L6_site_id = get_site_id(model, 'L6')
    R6_site_id = get_site_id(model, 'R6')
    W_p_L6_home, W_p_R6_home = get_foot_positions(data, L6_site_id, R6_site_id)
    home_CoM = setup.x_initial[0:3].copy()
    home_L6 = W_p_L6_home.copy()
    home_R6 = W_p_R6_home.copy()
    COM_HEIGHT = float(home_CoM[2])
    current = get_current_state_mpc(model, data)
    contact_state = get_contact_state(model=model, data=data, contact_map=setup.contact_map)
    c_L = int(contact_state['c_L'])
    c_R = int(contact_state['c_R'])
    W_p_L6, W_p_R6 = get_foot_positions(data, L6_site_id, R6_site_id)
    if (c_L, c_R) != (1, 1):
        raise RuntimeError(f'Robot phải khởi tạo vật lý ở DS: contact_meas=[{c_L},{c_R}], expected=[1,1].')
    initial_phase_cmd = get_initial_ds_phase(FIRST_SWING_FOOT)
    gait_state = initialize_gait_manager(initial_phase=initial_phase_cmd, c_L=c_L, c_R=c_R, W_p_L6=W_p_L6, W_p_R6=W_p_R6, t_phase=0.0, strict_contact=True)
    support_points, last_foothold_diagnostics = build_initial_support_plan(initial_phase=initial_phase_cmd, first_swing_foot=FIRST_SWING_FOOT, W_p_L6=W_p_L6, W_p_R6=W_p_R6, current_state=current.x, home_CoM=home_CoM, home_L6=home_L6, home_R6=home_R6)
    W_p_foothold_L, W_p_foothold_R = get_gait_foothold_inputs(gait_state.phase_cmd, support_points)
    gait_state, gait_output = update_gait_manager(state=gait_state, c_L=c_L, c_R=c_R, dt_elapsed=0.0, timing=TIMING, W_p_L6=W_p_L6, W_p_R6=W_p_R6, W_p_foothold_L=W_p_foothold_L, W_p_foothold_R=W_p_foothold_R)
    W_p_CoM_nominal_current = home_CoM.copy()
    held_nominal_velocity = np.zeros(3, dtype=float)
    propagate_nominal_com = False
    held_u0_MPC = np.zeros(12, dtype=float)
    held_tau_mpc = np.zeros(10, dtype=float)
    held_swing_foot = None
    held_q_swing_des = None
    held_q_dot_swing_des = None
    last_W_p_CoM_ref = np.tile(home_CoM.reshape(1, 3), (N, 1))
    last_contact_schedule_cmd = np.ones((N, 2), dtype=int)
    last_contact_schedule_mpc = np.ones((N, 2), dtype=int)
    last_solver_info = {'status': 'not_solved', 'solve_time': 0.0}
    last_post_diagnostics = None
    last_mpc_compute_time = 0.0
    tau_command = np.zeros(10, dtype=float)
    logger = SimulationLogger()
    reset_swing_low_level_state(None)
    physics_step = 0
    high_level_iteration = 0
    next_high_level_time = 0.0
    last_high_level_time = float(data.time)
    print('\n' + '=' * 82)
    print('WALKING - SEPARATED CONTACT COMMAND / CONTACT FEEDBACK')
    print('=' * 82)
    print('dt_sim        =', DT_SIM, 's')
    print('dt_high_level =', DT_HIGH_LEVEL, 's')
    print('dt_mpc grid   =', DT_MPC, 's')
    print('initial contact_meas =', [c_L, c_R])
    print('initial phase_cmd    =', initial_phase_cmd)
    print('first swing foot     =', FIRST_SWING_FOOT)
    print('\nCommand path:')
    print('timing/FSM -> phase_cmd -> command preview -> DCM/reference')
    print('MPC contact row0 = measured feedback; future rows = command preview')
    print('Feedback path:')
    print('MuJoCo -> contact_meas/foot_meas -> gait feedback + foot-plan re-anchor')
    print('Physics:')
    print('x_k -> tau_k -> data.ctrl -> mj_step -> x_{k+1}')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            wall_start = time.perf_counter()
            current_time = float(data.time)
            high_level_due = bool(current_time >= next_high_level_time - 1e-12)
            high_level_updated = False
            if high_level_due:
                current = get_current_state_mpc(model, data)
                contact_state = get_contact_state(model=model, data=data, contact_map=setup.contact_map)
                c_L = int(contact_state['c_L'])
                c_R = int(contact_state['c_R'])
                W_p_L6, W_p_R6 = get_foot_positions(data, L6_site_id, R6_site_id)
                if high_level_iteration == 0:
                    dt_high = 0.0
                else:
                    dt_high = max(current_time - last_high_level_time, 0.0)
                last_high_level_time = current_time
                if high_level_iteration > 0 and propagate_nominal_com:
                    W_p_CoM_nominal_current[0:2] += held_nominal_velocity[0:2] * dt_high
                W_p_CoM_nominal_current[2] = COM_HEIGHT
                support_points = ensure_support_plan_length(support_points=support_points, current_phase=gait_state.phase_cmd, target_length=SUPPORT_PLAN_LENGTH)
                W_p_foothold_L, W_p_foothold_R = get_gait_foothold_inputs(gait_state.phase_cmd, support_points)
                gait_state, gait_output = update_gait_manager(state=gait_state, c_L=c_L, c_R=c_R, dt_elapsed=dt_high, timing=TIMING, W_p_L6=W_p_L6, W_p_R6=W_p_R6, W_p_foothold_L=W_p_foothold_L, W_p_foothold_R=W_p_foothold_R)
                if not gait_output.phase_contact_consistent:
                    data.ctrl[:] = 0.0
                    raise RuntimeError(f'Measured contact is inconsistent with current/adjacent command phase: phase_cmd={gait_output.phase_cmd}, contact_meas={gait_output.contact_meas.tolist()}.')
                if gait_output.transitioned:
                    support_points = update_support_queue_after_command_transition(support_points=support_points, gait_output=gait_output)
                    print(f'\n[t={current_time:.4f}] COMMAND transition: {gait_output.previous_phase_cmd} --{gait_output.transition_reason}--> {gait_output.phase_cmd} | contact_meas={gait_output.contact_meas.tolist()} | contact_cmd={gait_output.contact_cmd.tolist()}')
                    if gait_output.swing_foot is not None:
                        reset_swing_low_level_state(gait_output.swing_foot)
                # Command phase preview is used for DCM/reference planning.
                # MPC row 0 is physically anchored to measured contact.
                phase_preview_cmd = build_contact_preview(
                    current_phase=gait_output.phase_cmd,
                    t_phase=gait_output.t_phase_cmd,
                    timing=TIMING,
                    horizon=N,
                    dt_mpc=DT_MPC,
                    current_contact_meas=gait_output.contact_meas,
                )
                contact_schedule_cmd = phase_preview_cmd['contact_schedule_cmd'].copy()
                contact_schedule_mpc = phase_preview_cmd['contact_schedule_mpc'].copy()

                if not np.array_equal(
                    contact_schedule_cmd[0],
                    gait_output.contact_cmd,
                ):
                    raise RuntimeError(
                        'contact_schedule_cmd[0] không khớp contact_cmd.'
                    )

                if not np.array_equal(
                    contact_schedule_mpc[0],
                    gait_output.contact_meas,
                ):
                    raise RuntimeError(
                        'contact_schedule_mpc[0] phải bằng contact feedback hiện tại.'
                    )

                foot_plan = build_foot_plan_horizon(
                    current_phase=gait_output.phase_cmd,
                    phase_sequence=phase_preview_cmd['phase_sequence'],
                    support_points=np.asarray(support_points, dtype=float),
                    W_p_L6_feedback=W_p_L6,
                    W_p_R6_feedback=W_p_R6,
                    current_contact_meas=gait_output.contact_meas,
                )
                W_p_L6_plan = foot_plan['W_p_L6_plan']
                W_p_R6_plan = foot_plan['W_p_R6_plan']
                support_points = [point.copy() for point in foot_plan['support_points_reanchored']]
                translational_reference = build_mpc_translational_reference(phase_preview=phase_preview_cmd, W_p_L6_plan=W_p_L6_plan, W_p_R6_plan=W_p_R6_plan, W_p_CoM_nominal_current=W_p_CoM_nominal_current, com_height=COM_HEIGHT, gravity=GRAVITY, horizon=N, dt_mpc=DT_MPC)
                W_p_CoM_ref = translational_reference['W_p_CoM_ref'].copy()
                W_v_CoM_ref = translational_reference['W_v_CoM_ref'].copy()
                held_nominal_velocity = np.asarray(translational_reference['W_v_CoM_nominal_current'], dtype=float).reshape(3).copy()
                propagate_nominal_com = bool(not gait_output.waiting_for_contact)
                if not propagate_nominal_com:
                    held_nominal_velocity[:] = 0.0
                held_swing_foot = None
                held_q_swing_des = None
                held_q_dot_swing_des = None
                if gait_output.swing_foot is not None:
                    if gait_output.W_p_swing_start is None:
                        raise RuntimeError('Swing command thiếu W_p_swing_start.')
                    if gait_output.W_p_foothold is None:
                        raise RuntimeError('Swing command thiếu W_p_foothold.')
                    # W_p_swing_start and W_p_foothold are BOTH latched at
                    # DS -> SS.  At 100 Hz only swing_phase changes; the
                    # trajectory is evaluated on the same fixed geometry.
                    swing_reference = compute_swing_reference_100hz(
                        model=model,
                        data=data,
                        swing_foot=gait_output.swing_foot,
                        W_p_liftoff=gait_output.W_p_swing_start,
                        W_p_foothold=gait_output.W_p_foothold,
                        swing_phase=gait_output.swing_phase,
                        T_swing=T_SS,
                        swing_height=SWING_HEIGHT,
                    )
                    held_swing_foot = gait_output.swing_foot
                    held_q_swing_des = np.asarray(swing_reference['q_swing_des'], dtype=float).reshape(5).copy()
                    held_q_dot_swing_des = np.asarray(swing_reference['q_dot_swing_des'], dtype=float).reshape(5).copy()
                mpc_start = time.perf_counter()
                U_opt, u0_opt, solver_info, post_diagnostics, srbd_reference = solve_ltv_mpc(
                    setup=setup,
                    current=current,
                    W_p_CoM_ref=W_p_CoM_ref,
                    W_v_CoM_ref=W_v_CoM_ref,
                    W_p_L6_plan=W_p_L6_plan,
                    W_p_R6_plan=W_p_R6_plan,
                    contact_schedule_mpc=contact_schedule_mpc,
                )
                del U_opt
                del srbd_reference
                if u0_opt is None:
                    data.ctrl[:] = 0.0
                    raise RuntimeError(f'QP không trả nghiệm hợp lệ.\nphase_cmd={gait_output.phase_cmd}\ncontact_meas={gait_output.contact_meas}\ncontact_cmd={gait_output.contact_cmd}\nsolver_info={solver_info}\npost_diagnostics={post_diagnostics}')
                held_u0_MPC = np.asarray(u0_opt, dtype=float).reshape(12).copy()
                # u0 is constrained by CURRENT measured contact, not contact_cmd.
                if contact_schedule_mpc[0, 0] == 0:
                    if np.linalg.norm(held_u0_MPC[0:6]) > 1e-05:
                        raise RuntimeError(
                            'Left foot contact_meas=0 nhưng u0 left wrench != 0.'
                        )
                if contact_schedule_mpc[0, 1] == 0:
                    if np.linalg.norm(held_u0_MPC[6:12]) > 1e-05:
                        raise RuntimeError(
                            'Right foot contact_meas=0 nhưng u0 right wrench != 0.'
                        )
                held_tau_mpc, _, _ = get_joint_torque_from_MPC_wrench(model, data, held_u0_MPC)
                held_tau_mpc = np.asarray(held_tau_mpc, dtype=float).reshape(10).copy()
                last_solver_info = solver_info
                last_post_diagnostics = post_diagnostics
                last_W_p_CoM_ref = W_p_CoM_ref.copy()
                last_contact_schedule_cmd = contact_schedule_cmd.copy()
                last_contact_schedule_mpc = contact_schedule_mpc.copy()
                last_mpc_compute_time = time.perf_counter() - mpc_start
                high_level_updated = True
                next_high_level_time = current_time + DT_HIGH_LEVEL
            if held_swing_foot is not None:
                q_swing, q_dot_swing = get_leg_joint_state(model, data, held_swing_foot)
                tau_swing = compute_swing_joint_PD(q_des=held_q_swing_des, q=q_swing, q_dot=q_dot_swing, Kp=KP_SWING, Kd=KD_SWING, q_dot_des=held_q_dot_swing_des)
                tau_command = mix_walking_joint_torque(tau_mpc=held_tau_mpc, swing_foot=held_swing_foot, tau_swing=tau_swing)
            else:
                tau_command = held_tau_mpc.copy()
            if not np.all(np.isfinite(tau_command)):
                raise RuntimeError('tau_command chứa NaN/Inf.')
            apply_joint_torque(model, data, tau_command)
            if high_level_updated:
                logger.log_step(time=float(data.time), x=current.x, u0=held_u0_MPC, tau=tau_command, mpc_compute_time=last_mpc_compute_time, qp_solve_time=float(last_solver_info.get('solve_time', 0.0)))
                if high_level_iteration % PRINT_EVERY_N_HIGH_LEVEL == 0:
                    print_status(
                        high_level_iteration=high_level_iteration,
                        data=data,
                        gait_output=gait_output,
                        current=current,
                        W_p_CoM_ref=last_W_p_CoM_ref,
                        held_u0_MPC=held_u0_MPC,
                        tau_command=tau_command,
                        solver_info=last_solver_info,
                        post_diagnostics=last_post_diagnostics,
                        support_points=support_points,
                        contact_schedule_cmd=last_contact_schedule_cmd,
                        contact_schedule_mpc=last_contact_schedule_mpc,
                    )
                high_level_iteration += 1
            mujoco.mj_step(model, data)
            physics_step += 1
            viewer.sync()
            wall_elapsed = time.perf_counter() - wall_start
            sleep_time = DT_SIM - wall_elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    final_state = get_current_state_mpc(model, data)
    final_contact = get_contact_state(model=model, data=data, contact_map=setup.contact_map)
    print('\n' + '=' * 82)
    print('WALKING SIMULATION FINISHED')
    print('=' * 82)
    print('simulation time =', data.time, 's')
    print('physics steps   =', physics_step)
    print('high-level runs =', high_level_iteration)
    print('final contact_meas =', [final_contact['c_L'], final_contact['c_R']])
    print('final phase_cmd    =', gait_output.phase_cmd)
    print('final contact_cmd  =', gait_output.contact_cmd)
    print('final t_phase_cmd  =', gait_output.t_phase_cmd)
    print('final CoM          =', final_state.x[0:3])
    print('final Euler        =', final_state.x[3:6])
    print('nominal CoM        =', W_p_CoM_nominal_current)
    logger.print_summary()
if __name__ == '__main__':
    main()
