import numpy as np
import pinocchio as pin

def mj_qpos_to_pin_q(qpos_mj, pin_model):
    """Chuyển qpos MuJoCo sang q Pinocchio."""
    q_pin = np.zeros(pin_model.nq)
    q_pin[0:3] = qpos_mj[0:3]
    q_pin[3] = qpos_mj[4]; q_pin[4] = qpos_mj[5]
    q_pin[5] = qpos_mj[6]; q_pin[6] = qpos_mj[3]
    q_pin[7:pin_model.nq] = qpos_mj[7:pin_model.nq]
    return q_pin

def compute_S_matrix(nv, n_act):
    """Ma trận chọn lọc S (n_act x nv)."""
    S = np.zeros((n_act, nv))
    for i in range(n_act):
        S[i, 6 + i] = 1.0
    return S

def smooth_ref(t, target_fn, com0, ramp_time=1.5):
    """Ramp mượt từ com0 đến target_fn(t)."""
    if t >= ramp_time:
        return target_fn(t, com0)
    u = t / ramp_time
    s = 3 * u**2 - 2 * u**3
    return com0 + s * (np.asarray(target_fn(t, com0)) - com0)