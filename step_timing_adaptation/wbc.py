import numpy as np
import pinocchio as pin
import casadi as cs
import io
import contextlib

# Import các hàm tiện ích từ utils
from .utils import mj_qpos_to_pin_q, compute_S_matrix

# ============================================================
# Cấu hình mặc định cho WBC (có thể ghi đè từ run.py)
# ============================================================
DEFAULT_WBC_CFG = {
    'W_COM':   200.0,
    'W_FOOT':  500.0,
    'W_TAU':   1e-3,
    'W_FC':    1e-4,
    'KP_COM':  400.0,
    'KD_COM':  40.0,
    'KP_FOOT': 800.0,
    'KD_FOOT': 60.0,
    'TAU_MAX': 3.23,
    'MU':      0.6,
}

# ============================================================
# Định nghĩa các Task
# ============================================================
class CoMTask:
    def __init__(self, x_des, Kp=400.0, Kd=40.0):
        self.x_des = np.asarray(x_des).copy()
        self.Kp, self.Kd = Kp, Kd
    def compute(self, pin_model, pin_data, q, v):
        x = pin.centerOfMass(pin_model, pin_data, q).copy()
        J = pin.jacobianCenterOfMass(pin_model, pin_data, q).copy()
        Jdot = np.zeros_like(J)
        xdot = J @ v
        xddot_des = self.Kp * (self.x_des - x) - self.Kd * xdot
        return J, Jdot, xddot_des

class FootTask:
    def __init__(self, foot_name, x_des, Kp=800.0, Kd=60.0):
        self.foot_name = foot_name
        self.x_des = np.asarray(x_des).copy()
        self.Kp, self.Kd = Kp, Kd
    def compute(self, pin_model, pin_data, q, v):
        fid = pin_model.getFrameId(self.foot_name)
        x = pin_data.oMf[fid].translation.copy()
        J = pin.getFrameJacobian(
            pin_model, pin_data, fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        Jdot = pin.getFrameJacobianTimeVariation(
            pin_model, pin_data, fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        xdot = J @ v
        xddot_des = self.Kp * (self.x_des - x) - self.Kd * xdot
        return J, Jdot, xddot_des

# ============================================================
# WBC Solver
# ============================================================
class WBCSolver:
    N_QDOT = 20
    N_TAU = 14
    N_FC = 12
    N_X = 46
    N_DYN = 20
    N_CONTACT = 6
    N_FRICTION = 10
    N_EQ = N_DYN + N_CONTACT
    N_INEQ = N_FRICTION
    N_CON = N_EQ + N_INEQ

    def __init__(self, cfg=None, solver_backend=None, verbose=False):
        self.cfg = {**DEFAULT_WBC_CFG, **(cfg or {})}

        n_x = self.N_X
        n_con = self.N_CON

        H_dummy = cs.DM(np.eye(n_x))
        A_dummy = cs.DM(np.zeros((n_con, n_x)))
        H_sp = H_dummy.sparsity()
        A_sp = A_dummy.sparsity()

        qp_dict = {'h': H_sp, 'a': A_sp}

        backends_to_try = ([solver_backend] if solver_backend else []) + \
                          ['osqp', 'proxqp', 'qpoases']

        self.S = None
        for be in backends_to_try:
            try:
                opts = {
                    'print_time': False,
                    'error_on_fail': False,
                    'verbose': False,
                    'osqp': {'verbose': False, 'polish': True},
                    'proxqp': {'verbose': False},
                    'qpoases': {'printLevel': 'none'},
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    S = cs.conic('S', be, qp_dict, opts)
                self._test_solver(S, be)
                self.S = S
                print(f"[WBC] QP solver OK: {be}", flush=True)
                break
            except Exception as e:
                try:
                    opts2 = {'print_time': False, 'error_on_fail': False}
                    with contextlib.redirect_stdout(io.StringIO()):
                        S = cs.conic('S', be, qp_dict, opts2)
                    self._test_solver(S, be)
                    self.S = S
                    print(f"[WBC] QP solver OK (min opts): {be}", flush=True)
                    break
                except Exception as e2:
                    print(f"[WBC] Backend '{be}' không work: {str(e2)[:120]}", flush=True)

        if self.S is None:
            raise RuntimeError("Không có QP backend nào hoạt động. Cài: pip install osqp")

        self.x_prev = np.zeros(n_x)

    @staticmethod
    def _test_solver(S, name):
        n_x = WBCSolver.N_X
        n_con = WBCSolver.N_CON
        H = np.eye(n_x); g = np.zeros(n_x); A = np.zeros((n_con, n_x))
        with contextlib.redirect_stdout(io.StringIO()):
            sol = S(h=cs.DM(H), g=cs.DM(g), a=cs.DM(A),
                    lba=cs.DM(-np.inf * np.ones(n_con)),
                    uba=cs.DM( np.inf * np.ones(n_con)),
                    lbx=cs.DM(-np.ones(n_x)),
                    ubx=cs.DM( np.ones(n_x)),
                    x0=cs.DM(np.zeros(n_x)))
        x = np.array(sol['x']).flatten()
        assert x.shape == (n_x,), f"Bad shape {x.shape}"

    def solve(self, M, h, S_mat, Jc, Jc_pos, Jdot_pos_v,
              J_com, a_com_des, Jdot_com_v,
              J_footL, a_footL_des, Jdot_footL_v,
              J_footR, a_footR_des, Jdot_footR_v):
        n_x = self.N_X
        nv = self.N_QDOT
        cfg = self.cfg

        H = np.zeros((n_x, n_x)); g = np.zeros(n_x)

        def add_task(J, a_des, Jdot_v, w):
            b = a_des - Jdot_v
            H[:nv, :nv] += 2 * w * (J.T @ J)
            g[:nv] += -2 * w * (J.T @ b)

        add_task(J_com, a_com_des, Jdot_com_v, cfg['W_COM'])
        add_task(J_footL, a_footL_des, Jdot_footL_v, cfg['W_FOOT'])
        add_task(J_footR, a_footR_des, Jdot_footR_v, cfg['W_FOOT'])
        H[20:34, 20:34] += 2 * cfg['W_TAU'] * np.eye(14)
        H[34:46, 34:46] += 2 * cfg['W_FC']  * np.eye(12)
        H += 1e-8 * np.eye(n_x)

        A_dyn  = np.hstack([M, -S_mat.T, -Jc.T])
        b_dyn  = -h
        A_cont = np.hstack([Jc_pos, np.zeros((6, 14)), np.zeros((6, 12))])
        b_cont = -Jdot_pos_v
        A_eq = np.vstack([A_dyn, A_cont])
        b_eq = np.concatenate([b_dyn, b_cont])

        A_ineq_rows = []; b_ineq_vals = []
        mu = cfg['MU']
        for foot in range(2):
            base = 34 + foot * 6
            r = np.zeros(n_x); r[base+2] = -1.0
            A_ineq_rows.append(r); b_ineq_vals.append(0.0)
            r = np.zeros(n_x); r[base]   =  1.0; r[base+2] = -mu
            A_ineq_rows.append(r); b_ineq_vals.append(0.0)
            r = np.zeros(n_x); r[base]   = -1.0; r[base+2] = -mu
            A_ineq_rows.append(r); b_ineq_vals.append(0.0)
            r = np.zeros(n_x); r[base+1] =  1.0; r[base+2] = -mu
            A_ineq_rows.append(r); b_ineq_vals.append(0.0)
            r = np.zeros(n_x); r[base+1] = -1.0; r[base+2] = -mu
            A_ineq_rows.append(r); b_ineq_vals.append(0.0)
        A_ineq = np.array(A_ineq_rows); b_ineq = np.array(b_ineq_vals)

        A_all = np.vstack([A_eq, A_ineq])
        lba = np.concatenate([b_eq, -np.inf * np.ones(len(b_ineq))])
        uba = np.concatenate([b_eq,           b_ineq])

        lbx = np.full(n_x, -np.inf); ubx = np.full(n_x, np.inf)
        lbx[20:34] = -cfg['TAU_MAX']
        ubx[20:34] =  cfg['TAU_MAX']

        try:
            sol = self.S(h=cs.DM(H), g=cs.DM(g), a=cs.DM(A_all),
                         lba=cs.DM(lba), uba=cs.DM(uba),
                         lbx=cs.DM(lbx), ubx=cs.DM(ubx),
                         x0=cs.DM(self.x_prev))
            x_opt = np.array(sol['x']).flatten()
            self.x_prev = x_opt
            return x_opt[20:34], x_opt
        except Exception as e:
            print(f"[WBC] QP solve failed: {e}", flush=True)
            return None, None

    # Phương thức tiện ích: tính toàn bộ control từ state
    def compute_control(self, qpos, v, t, pin_model, pin_data,
                        x_des_com_fn, com0, fl0, fr0, S_mat):
        """Trả về tau (14,) từ state hiện tại."""
        q_pin = mj_qpos_to_pin_q(qpos, pin_model)
        pin.forwardKinematics(pin_model, pin_data, q_pin)
        pin.updateFramePlacements(pin_model, pin_data)
        pin.computeJointJacobians(pin_model, pin_data, q_pin)
        pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q_pin, v)

        M = pin.crba(pin_model, pin_data, q_pin).copy()
        if hasattr(pin_model, "armature"):
            M += np.diag(pin_model.armature)
        h = pin.nonLinearEffects(pin_model, pin_data, q_pin, v)

        J_com = pin.jacobianCenterOfMass(pin_model, pin_data, q_pin).copy()
        com = pin.centerOfMass(pin_model, pin_data, q_pin).copy()
        com_dot = J_com @ v
        x_des_com = np.asarray(x_des_com_fn(t, com0))
        a_com_des = self.cfg['KP_COM'] * (x_des_com - com) - self.cfg['KD_COM'] * com_dot
        Jdot_com_v = np.zeros(3)

        fid_l = pin_model.getFrameId("left_foot")
        fid_r = pin_model.getFrameId("right_foot")
        J_full_l = pin.getFrameJacobian(pin_model, pin_data, fid_l, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_full_r = pin.getFrameJacobian(pin_model, pin_data, fid_r, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_footL = J_full_l[:3, :]; J_footR = J_full_r[:3, :]

        dJ_full_l = pin.getFrameJacobianTimeVariation(pin_model, pin_data, fid_l, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJ_full_r = pin.getFrameJacobianTimeVariation(pin_model, pin_data, fid_r, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jdot_footL_v = dJ_full_l[:3, :] @ v
        Jdot_footR_v = dJ_full_r[:3, :] @ v

        fl = pin_data.oMf[fid_l].translation
        fr = pin_data.oMf[fid_r].translation
        fl_dot = J_footL @ v; fr_dot = J_footR @ v
        a_footL_des = self.cfg['KP_FOOT'] * (fl0 - fl) - self.cfg['KD_FOOT'] * fl_dot
        a_footR_des = self.cfg['KP_FOOT'] * (fr0 - fr) - self.cfg['KD_FOOT'] * fr_dot

        Jc = np.vstack([J_full_l, J_full_r])
        Jc_pos = np.vstack([J_full_l[:3, :], J_full_r[:3, :]])
        Jdot_pos_v = np.concatenate([Jdot_footL_v, Jdot_footR_v])

        return self.solve(M, h, S_mat, Jc, Jc_pos, Jdot_pos_v,
                          J_com, a_com_des, Jdot_com_v,
                          J_footL, a_footL_des, Jdot_footL_v,
                          J_footR, a_footR_des, Jdot_footR_v)