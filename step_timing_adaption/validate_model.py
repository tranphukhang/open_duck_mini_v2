"""
test_stage2.py
==============
Giai đoạn 2: Task-space PD controller (kinematic mode, double support).

Scope:
  - Chưa có planner -> chưa có quỹ đạo chân vung.
  - Chỉ test ở double support, CoM đứng yên hoặc di chuyển nhỏ.
  - 3 tasks: Foot L, Foot R, CoM (theo thứ tự ưu tiên).
  - Hierarchical resolver với null-space projection.
  - Kinematic integration (không dùng physics).

Convention đã chốt ở Giai đoạn 1:
  - Jacobian: LOCAL_WORLD_ALIGNED
  - Jdot: analytic từ Pinocchio
  - qvel MJ == v PIN
"""

import os
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: pip install mujoco"); sys.exit(1)

try:
    import pinocchio as pin
except ImportError:
    print("ERROR: pip install pin"); sys.exit(1)


# ============================================================
# Helpers
# ============================================================
def mj_qpos_to_pin_q(qpos_mj, pin_model):
    """MuJoCo qpos -> Pinocchio q (khác convention quaternion)."""
    q_pin = np.zeros(pin_model.nq)
    q_pin[0:3] = qpos_mj[0:3]
    q_pin[3] = qpos_mj[4]   # qx
    q_pin[4] = qpos_mj[5]   # qy
    q_pin[5] = qpos_mj[6]   # qz
    q_pin[6] = qpos_mj[3]   # qw
    q_pin[7:pin_model.nq] = qpos_mj[7:pin_model.nq]
    return q_pin


def damped_pinv(J, lam=1e-3):
    """Damped pseudoinverse: J^T (JJ^T + lam^2 I)^-1."""
    m, n = J.shape
    if m == 0 or n == 0:
        return np.zeros((n, m))
    JJT = J @ J.T + lam**2 * np.eye(m)
    return J.T @ np.linalg.solve(JJT, np.eye(m))


# ============================================================
# Task definitions
# ============================================================
class CoMTask:
    """CoM position task (3D)."""
    def __init__(self, x_des, Kp=100.0, Kd=20.0):
        self.name = "CoM"
        self.x_des = np.asarray(x_des).copy()
        self.Kp = Kp
        self.Kd = Kd

    def compute(self, pin_model, pin_data, q, v):
        # Kinematics đã được pre-compute trong resolver
        x = pin.centerOfMass(pin_model, pin_data, q).copy()
        J = pin.jacobianCenterOfMass(pin_model, pin_data, q).copy()
        # Pinocchio không có API CoM Jdot -> xấp xỉ = 0 (đủ cho kinematic)
        Jdot = np.zeros_like(J)
        xdot = J @ v
        xddot_des = self.Kp * (self.x_des - x) - self.Kd * xdot
        return x, xdot, J, Jdot, xddot_des


class FootTask:
    """Foot position task (3D)."""
    def __init__(self, foot_name, x_des, Kp=400.0, Kd=40.0):
        self.name = f"Foot_{foot_name}"
        self.foot_name = foot_name
        self.x_des = np.asarray(x_des).copy()
        self.Kp = Kp
        self.Kd = Kd

    def compute(self, pin_model, pin_data, q, v):
        fid = pin_model.getFrameId(self.foot_name)
        x = pin_data.oMf[fid].translation.copy()
        J_full = pin.getFrameJacobian(
            pin_model, pin_data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:3, :]
        dJ_full = pin.getFrameJacobianTimeVariation(
            pin_model, pin_data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jdot = dJ_full[:3, :]
        xdot = J @ v
        xddot_des = self.Kp * (self.x_des - x) - self.Kd * xdot
        return x, xdot, J, Jdot, xddot_des


# ============================================================
# Hierarchical resolver
# ============================================================
def resolve_hierarchical(tasks, q, v, pin_model, pin_data, lam=1e-3):
    """
    Null-space projection hierarchy.
    Task i (ưu tiên giảm dần):
        qddot_i = qddot_{i-1} + (J_i N_{i-1})^+ (xddot_i - Jdot_i v - J_i qddot_{i-1})
        N_i = N_{i-1} (I - (J_i N_{i-1})^+ (J_i N_{i-1}))
    """
    # Pre-compute kinematics once
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    pin.computeJointJacobians(pin_model, pin_data, q)
    pin.computeJointJacobiansTimeVariation(pin_model, pin_data, q, v)

    nv = pin_model.nv
    qddot = np.zeros(nv)
    N = np.eye(nv)

    for task in tasks:
        x, xdot, J, Jdot, xddot_des = task.compute(pin_model, pin_data, q, v)
        J_eff = J @ N
        J_eff_pinv = damped_pinv(J_eff, lam)
        b = xddot_des - Jdot @ v - J @ qddot
        qddot += J_eff_pinv @ b
        N = N @ (np.eye(nv) - J_eff_pinv @ J_eff)

    return qddot


# ============================================================
# Test runner
# ============================================================
def run_test(test_name, x_des_com_fn, n_steps, dt,
             mj_model, mj_data, pin_model, pin_data,
             qpos0, foot_l0, foot_r0, com0):
    print(f"\n{'='*72}")
    print(f"  {test_name}")
    print(f"{'='*72}")

    qpos = qpos0.copy()
    v = np.zeros(pin_model.nv)

    for step in range(n_steps):
        t = step * dt
        q_pin = mj_qpos_to_pin_q(qpos, pin_model)
        x_des_com = np.asarray(x_des_com_fn(t, com0))

        tasks = [
            FootTask("left_foot", foot_l0),
            FootTask("right_foot", foot_r0),
            CoMTask(x_des_com),
        ]

        qddot = resolve_hierarchical(tasks, q_pin, v, pin_model, pin_data)

        # Semi-implicit integration
        v = v + qddot * dt
        mujoco.mj_integratePos(mj_model, qpos, v, dt)

        # Log
        if step % 500 == 0 or step == n_steps - 1:
            q_pin_now = mj_qpos_to_pin_q(qpos, pin_model)
            pin.forwardKinematics(pin_model, pin_data, q_pin_now)
            pin.updateFramePlacements(pin_model, pin_data)
            com = pin.centerOfMass(pin_model, pin_data, q_pin_now).copy()
            fl = pin_data.oMf[pin_model.getFrameId("left_foot")].translation
            fr = pin_data.oMf[pin_model.getFrameId("right_foot")].translation

            com_err = np.linalg.norm(com - x_des_com)
            fl_err = np.linalg.norm(fl - foot_l0)
            fr_err = np.linalg.norm(fr - foot_r0)
            print(f"  t={t:.3f}s  |com-x_des|={com_err*1000:.4f} mm  "
                  f"|footL-L0|={fl_err*1000:.6f} mm  "
                  f"|footR-R0|={fr_err*1000:.6f} mm")


# ============================================================
# Main
# ============================================================
def main():
    xml = sys.argv[1] if len(sys.argv) > 1 else "xmls/open_duck_mini_v2_torque.xml"
    if not os.path.isfile(xml):
        print(f"Không tìm thấy: {xml}"); sys.exit(1)

    print(f"File XML: {os.path.abspath(xml)}")
    print(f"Pinocchio: {pin.__version__}")

    mj_model = mujoco.MjModel.from_xml_path(xml)
    mj_data = mujoco.MjData(mj_model)
    pin_model = pin.buildModelFromMJCF(xml)
    pin_data = pin_model.createData()

    # --- Initial state ---
    qpos0 = mj_model.qpos0.copy()
    q_pin0 = mj_qpos_to_pin_q(qpos0, pin_model)
    pin.forwardKinematics(pin_model, pin_data, q_pin0)
    pin.updateFramePlacements(pin_model, pin_data)
    com0 = pin.centerOfMass(pin_model, pin_data, q_pin0).copy()
    foot_l0 = pin_data.oMf[pin_model.getFrameId("left_foot")].translation.copy()
    foot_r0 = pin_data.oMf[pin_model.getFrameId("right_foot")].translation.copy()

    print(f"\nInitial state (qpos0):")
    print(f"  CoM   = {com0}")
    print(f"  FootL = {foot_l0}")
    print(f"  FootR = {foot_r0}")

    n_steps = 3000
    dt = 0.001  # 1 kHz

    # Test 1: baseline — CoM ở nguyên vị trí
    run_test("TEST 1: BASELINE (x_des_com = com0)",
             lambda t, c: c,
             n_steps, dt, mj_model, mj_data, pin_model, pin_data,
             qpos0, foot_l0, foot_r0, com0)

    # Test 2: CoM dịch chuyển +1cm trục X
    run_test("TEST 2: CoM SHIFT +1cm X",
             lambda t, c: c + np.array([0.01, 0, 0]),
             n_steps, dt, mj_model, mj_data, pin_model, pin_data,
             qpos0, foot_l0, foot_r0, com0)

    # Test 3: CoM sinusoid ±1cm, 0.5 Hz
    run_test("TEST 3: CoM SINUSOID ±1cm, 0.5Hz",
             lambda t, c: c + np.array([0.01 * np.sin(2*np.pi*0.5*t), 0, 0]),
             n_steps, dt, mj_model, mj_data, pin_model, pin_data,
             qpos0, foot_l0, foot_r0, com0)

    print("\n" + "="*72)
    print("  Kỳ vọng:")
    print("  TEST 1: |com-x_des| < 0.01 mm, foot err < 1e-6 mm")
    print("  TEST 2: |com-x_des| giảm dần (có thể không tới 0 do động học)")
    print("  TEST 3: CoM bám sin ±1cm, foot err < 1e-4 mm")
    print("="*72)


if __name__ == "__main__":
    main()