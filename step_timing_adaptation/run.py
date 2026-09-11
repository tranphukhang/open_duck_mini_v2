"""
step_timing_adaptation/run.py
==============================
Entry point: load models, khởi tạo WBC, chạy các test.

Cách chạy:
    python -m step_timing_adaptation.run          # all tests
    python -m step_timing_adaptation.run 1        # test 1
    python -m step_timing_adaptation.run 2        # test 2
    python -m step_timing_adaptation.run 3        # test 3
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin
import casadi as cs

from .wbc import WBCSolver
from .utils import mj_qpos_to_pin_q, compute_S_matrix, smooth_ref


# ============================================================
# CẤU HÌNH
# ============================================================
CFG = {
    'WBC': {
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
    },
    'WBC_RATE_HZ': 1000,
    'XML_SCENE': 'xmls/scene_flat_terrain_torque.xml',
    'XML_ROBOT': 'xmls/open_duck_mini_v2_torque.xml',
    'KEYFRAME_NAME': 'home',
    'TEST_SELECTOR': 'all',
    'USE_VIEWER': True,
}


# ============================================================
# RUN 1 TEST
# ============================================================
def run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, qpos0,
            x_des_com_fn, test_name):
    print(f"\n{'='*72}", flush=True)
    print(f"  {test_name}", flush=True)
    print(f"{'='*72}", flush=True)

    mj_data.qpos[:] = qpos0
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    qpos = qpos0.copy()
    v = np.zeros(pin_model.nv)
    S_mat = compute_S_matrix(pin_model.nv, 14)

    q_pin0 = mj_qpos_to_pin_q(qpos, pin_model)
    pin.forwardKinematics(pin_model, pin_data, q_pin0)
    pin.updateFramePlacements(pin_model, pin_data)
    com0 = pin.centerOfMass(pin_model, pin_data, q_pin0).copy()
    fid_l = pin_model.getFrameId("left_foot")
    fid_r = pin_model.getFrameId("right_foot")
    fl0 = pin_data.oMf[fid_l].translation.copy()
    fr0 = pin_data.oMf[fid_r].translation.copy()
    print(f"  CoM0   = {com0}", flush=True)
    print(f"  FootL0 = {fl0}", flush=True)
    print(f"  FootR0 = {fr0}", flush=True)

    dt = mj_model.opt.timestep
    wbc_dt = 1.0 / CFG['WBC_RATE_HZ']
    decimation = max(1, int(round(wbc_dt / dt)))
    print(f"  Physics {1/dt:.0f} Hz, WBC {CFG['WBC_RATE_HZ']} Hz, "
          f"decimation = {decimation}", flush=True)

    step = 0
    qp_times = []
    tau_held = np.zeros(14)

    # Warm-up
    _ = wbc.compute_control(qpos, v, 0.0, pin_model, pin_data,
                             x_des_com_fn, com0, fl0, fr0, S_mat)

    def loop(viewer=None):
        nonlocal qpos, v, step, tau_held
        while True:
            if viewer is not None and not viewer.is_running():
                break

            t = step * dt

            if step % decimation == 0:
                t0 = time.perf_counter()
                tau_new, _ = wbc.compute_control(
                    qpos, v, t, pin_model, pin_data,
                    x_des_com_fn, com0, fl0, fr0, S_mat)
                t1 = time.perf_counter()
                qp_times.append((t1 - t0) * 1e6)
                if tau_new is None:
                    print(f"  [t={t:.2f}] QP FAIL — giữ τ cũ", flush=True)
                else:
                    tau_held = tau_new

            mj_data.ctrl[:] = tau_held
            mujoco.mj_step(mj_model, mj_data)
            qpos = mj_data.qpos.copy()
            v = mj_data.qvel.copy()

            if viewer is not None:
                viewer.sync()

            if step % 500 == 0:
                q_pin_now = mj_qpos_to_pin_q(qpos, pin_model)
                pin.forwardKinematics(pin_model, pin_data, q_pin_now)
                pin.updateFramePlacements(pin_model, pin_data)
                com = pin.centerOfMass(pin_model, pin_data, q_pin_now).copy()
                fl = pin_data.oMf[fid_l].translation
                fr = pin_data.oMf[fid_r].translation
                x_des_com = np.asarray(x_des_com_fn(t, com0))
                com_err = np.linalg.norm(com - x_des_com) * 1000
                fl_err = np.linalg.norm(fl - fl0) * 1000
                fr_err = np.linalg.norm(fr - fr0) * 1000
                qp_avg = np.mean(qp_times[-100:]) if qp_times else 0
                print(f"  t={t:5.2f}s  ncon={mj_data.ncon:2d}  "
                      f"base_z={qpos[2]:.3f}  "
                      f"|comE|={com_err:6.2f}mm  "
                      f"|ftL|={fl_err:5.2f}  |ftR|={fr_err:5.2f}  "
                      f"|τ|={np.abs(tau_held).max():.3f}  QP={qp_avg:.0f}µs",
                      flush=True)
            step += 1

    if CFG['USE_VIEWER']:
        print(f"  >>> Tắt cửa sổ viewer để dừng.", flush=True)
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            loop(viewer)
    else:
        loop(None)

    if qp_times:
        print(f"  QP avg={np.mean(qp_times):.0f}µs  "
              f"max={np.max(qp_times):.0f}µs  "
              f"(budget @{CFG['WBC_RATE_HZ']}Hz = {1e6/CFG['WBC_RATE_HZ']:.0f}µs)",
              flush=True)


# ============================================================
# MAIN
# ============================================================
def main():
    test_sel = sys.argv[1] if len(sys.argv) > 1 else CFG['TEST_SELECTOR']
    xml_scene = sys.argv[2] if len(sys.argv) > 2 else CFG['XML_SCENE']
    xml_robot = CFG['XML_ROBOT']

    if not os.path.isfile(xml_scene):
        print(f"Không tìm thấy scene: {xml_scene}", flush=True)
        sys.exit(1)

    print(f"CasADi: {cs.__version__}", flush=True)
    print(f"Scene : {os.path.abspath(xml_scene)}", flush=True)
    print(f"Robot : {os.path.abspath(xml_robot)}", flush=True)
    print(f"Test  : {test_sel}", flush=True)

    mj_model = mujoco.MjModel.from_xml_path(xml_scene)
    mj_data = mujoco.MjData(mj_model)
    pin_model = pin.buildModelFromMJCF(xml_robot)
    pin_data = pin_model.createData()

    kf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY,
                               CFG['KEYFRAME_NAME'])
    if kf_id < 0:
        print(f"Không tìm thấy keyframe '{CFG['KEYFRAME_NAME']}'")
        sys.exit(1)
    kf_qpos = mj_model.key_qpos[kf_id].copy()
    print(f"Keyframe '{CFG['KEYFRAME_NAME']}': base_z={kf_qpos[2]}", flush=True)

    wbc = WBCSolver(cfg=CFG['WBC'], solver_backend=None)

    if test_sel in ("all", "1"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: c,
                "TEST 1: WBC hold pose")

    if test_sel in ("all", "2"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: smooth_ref(
                    t, lambda tt, cc: cc + np.array([0.01, 0, 0]),
                    c, ramp_time=1.5),
                "TEST 2: WBC CoM +1cm X (ramped 1.5s)")

    if test_sel in ("all", "3"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: c + np.array([0.01 * np.sin(2*np.pi*0.5*t), 0, 0]),
                "TEST 3: WBC CoM sinusoid ±1cm 0.5Hz")


if __name__ == "__main__":
    main()