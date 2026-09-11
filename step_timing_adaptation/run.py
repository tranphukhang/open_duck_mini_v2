"""
step_timing_adaptation/run.py
==============================
Entry point: load models, khởi tạo WBC + DCM + visualizer, chạy test.

Quy tắc đọc dữ liệu:
    - CoM position/velocity/acceleration: Pinocchio (analytical).
    - Foot site pose: đọc trực tiếp từ MuJoCo.

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
from .dcm import DCMCalculator
from .visualization import WBCVisualizer


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
    'WBC_RATE_HZ':    1000,
    'VIS_DECIMATION': 5,
    'XML_SCENE':      'xmls/scene_flat_terrain_torque.xml',
    'XML_ROBOT':      'xmls/open_duck_mini_v2_torque.xml',
    'KEYFRAME_NAME':  'home',
    'TEST_SELECTOR':  'all',
    'USE_VIEWER':     True,
    'DEBUG_DCM':      True,
}


# ============================================================
# HELPERS
# ============================================================
def pin_get_com_state(pin_model, pin_data, q_pin, v_pin, a_pin):
    """CoM position / velocity / acceleration từ Pinocchio."""
    pin.centerOfMass(pin_model, pin_data, q_pin, v_pin, a_pin)
    return (pin_data.com[0].copy(),
            pin_data.vcom[0].copy(),
            pin_data.acom[0].copy())


def mj_get_site_pose(mj_data, site_id):
    """Đọc pose của site trực tiếp từ MuJoCo."""
    pos = mj_data.site_xpos[site_id].copy()
    rot = mj_data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot


# ============================================================
# RUN 1 TEST
# ============================================================
def run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, qpos0,
            x_des_com_fn, test_name,
            dcm_calc, visualizer,
            site_L_id, site_R_id):
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
    v_zero = np.zeros(pin_model.nv)
    a_zero = np.zeros(pin_model.nv)
    com0, _, _ = pin_get_com_state(pin_model, pin_data,
                                    q_pin0, v_zero, a_zero)
    fl0, _ = mj_get_site_pose(mj_data, site_L_id)
    fr0, _ = mj_get_site_pose(mj_data, site_R_id)
    print(f"  CoM0   = {com0}", flush=True)
    print(f"  FootL0 = {fl0}", flush=True)
    print(f"  FootR0 = {fr0}", flush=True)

    dt = mj_model.opt.timestep
    wbc_dt = 1.0 / CFG['WBC_RATE_HZ']
    decimation = max(1, int(round(wbc_dt / dt)))
    vis_every = CFG['VIS_DECIMATION']
    print(f"  Physics {1/dt:.0f} Hz, WBC {CFG['WBC_RATE_HZ']} Hz, "
          f"decim={decimation}, vis_every={vis_every}", flush=True)

    step = 0
    qp_times = []
    tau_held = np.zeros(14)
    visualizer.reset()

    _ = wbc.compute_control(qpos, v, 0.0, pin_model, pin_data,
                            x_des_com_fn, com0, fl0, fr0, S_mat)

    def loop(viewer=None):
        nonlocal qpos, v, step, tau_held
        while True:
            if viewer is not None and not viewer.is_running():
                break

            t = step * dt

            # --- WBC update (1 kHz) ---
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

            # --- Physics step ---
            mj_data.ctrl[:] = tau_held
            mujoco.mj_step(mj_model, mj_data)
            # Sync qacc với state mới để Pinocchio tính đúng CoM acceleration
            mujoco.mj_forward(mj_model, mj_data)
            qpos = mj_data.qpos.copy()
            v = mj_data.qvel.copy()

            # --- Visualization ---
            if viewer is not None:
                if step % vis_every == 0:
                    q_pin_now = mj_qpos_to_pin_q(qpos, pin_model)
                    a_pin_now = mj_data.qacc.copy()

                    com, com_vel, com_acc = pin_get_com_state(
                        pin_model, pin_data,
                        q_pin_now, v, a_pin_now)

                    com_xy     = com[:2]
                    com_vel_xy = com_vel[:2]
                    com_acc_xy = com_acc[:2]
                    com_des_xy = np.asarray(x_des_com_fn(t, com0))[:2]

                    dcm_xy = dcm_calc.compute_dcm_xy(com_xy, com_vel_xy)
                    zmp_xy = dcm_calc.compute_zmp_from_lipm(com_xy, com_acc_xy)

                    if CFG['DEBUG_DCM'] and step % 1000 == 0:
                        offset_mm = np.linalg.norm(dcm_xy - com_xy) * 1000
                        v_mag = np.linalg.norm(com_vel_xy)
                        print(f"  [DBG t={t:5.2f}] |v_com|={v_mag:.5f} m/s, "
                              f"|DCM-CoM|={offset_mm:.3f} mm",
                              flush=True)

                    p_l, R_l = mj_get_site_pose(mj_data, site_L_id)
                    p_r, R_r = mj_get_site_pose(mj_data, site_R_id)

                    visualizer.update(
                        viewer, t,
                        com_xy=com_xy,
                        com_des_xy=com_des_xy,
                        dcm_xy=dcm_xy,
                        zmp_xy=zmp_xy,
                        foot_L_pose=(p_l, R_l),
                        foot_R_pose=(p_r, R_r),
                    )
                else:
                    viewer.sync()

            # --- Log ---
            if step % 500 == 0:
                q_pin_now = mj_qpos_to_pin_q(qpos, pin_model)
                a_pin_now = mj_data.qacc.copy()
                com, _, _ = pin_get_com_state(pin_model, pin_data,
                                               q_pin_now, v, a_pin_now)
                fl, _ = mj_get_site_pose(mj_data, site_L_id)
                fr, _ = mj_get_site_pose(mj_data, site_R_id)
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
            visualizer.configure_camera(viewer)
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

    site_L_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                   "left_foot")
    site_R_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                   "right_foot")
    if site_L_id < 0 or site_R_id < 0:
        print(f"Không tìm thấy site 'left_foot' hoặc 'right_foot'")
        sys.exit(1)
    print(f"[Site] left_foot id={site_L_id}, right_foot id={site_R_id}",
          flush=True)

    wbc = WBCSolver(cfg=CFG['WBC'], solver_backend=None)

    q_pin0 = mj_qpos_to_pin_q(kf_qpos, pin_model)
    v_zero = np.zeros(pin_model.nv)
    a_zero = np.zeros(pin_model.nv)
    com_init, _, _ = pin_get_com_state(pin_model, pin_data,
                                        q_pin0, v_zero, a_zero)
    com_z = float(com_init[2])
    dcm_calc = DCMCalculator(com_height=com_z)
    print(f"[DCM] com_height = {com_z:.4f} m, "
          f"omega = {dcm_calc.omega:.4f} rad/s", flush=True)

    visualizer = WBCVisualizer(mj_model)

    if test_sel in ("all", "1"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: c,
                "TEST 1: WBC hold pose",
                dcm_calc, visualizer, site_L_id, site_R_id)

    if test_sel in ("all", "2"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: smooth_ref(
                    t, lambda tt, cc: cc + np.array([0.01, 0, 0]),
                    c, ramp_time=1.5),
                "TEST 2: WBC CoM +1cm X (ramped 1.5s)",
                dcm_calc, visualizer, site_L_id, site_R_id)

    if test_sel in ("all", "3"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: c + np.array([0.01 * np.sin(2*np.pi*0.5*t), 0, 0]),
                "TEST 3: WBC CoM sinusoid ±1cm 0.5Hz",
                dcm_calc, visualizer, site_L_id, site_R_id)


if __name__ == "__main__":
    main()