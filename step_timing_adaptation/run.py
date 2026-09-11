"""
step_timing_adaptation/run.py
==============================
Entry point: load models, khởi tạo WBC + DCM + visualizer, chạy test.

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
    'VIS_DECIMATION': 5,      # Cứ 5 physics step thì update marker (400 Hz)
    'XML_SCENE':      'xmls/scene_flat_terrain_torque.xml',
    'XML_ROBOT':      'xmls/open_duck_mini_v2_torque.xml',
    'KEYFRAME_NAME':  'home',
    'TEST_SELECTOR':  'all',
    'USE_VIEWER':     True,

    'FOOT_GEOM_NAMES':  ['left_foot_bottom_tpu', 'right_foot_bottom_tpu'],
    'FLOOR_GEOM_NAME':  'floor',
}


# ============================================================
# RUN 1 TEST
# ============================================================
def run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, qpos0,
            x_des_com_fn, test_name,
            dcm_calc, visualizer,
            foot_geom_ids, floor_geom_id):
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

            # --- WBC update ---
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
            qpos = mj_data.qpos.copy()
            v = mj_data.qvel.copy()

            # --- Visualization ---
            if viewer is not None:
                if step % vis_every == 0:
                    q_pin_now = mj_qpos_to_pin_q(qpos, pin_model)
                    pin.forwardKinematics(pin_model, pin_data, q_pin_now)
                    pin.updateFramePlacements(pin_model, pin_data)
                    pin.computeJointJacobians(pin_model, pin_data, q_pin_now)

                    com = pin.centerOfMass(pin_model, pin_data, q_pin_now).copy()
                    J_com = pin.jacobianCenterOfMass(
                        pin_model, pin_data, q_pin_now)
                    com_vel = J_com @ v

                    com_xy     = com[:2]
                    com_vel_xy = com_vel[:2]
                    com_des_xy = np.asarray(x_des_com_fn(t, com0))[:2]
                    dcm_xy     = dcm_calc.compute_dcm_xy(com_xy, com_vel_xy)
                    cop_xy     = dcm_calc.compute_cop_xy(
                        mj_model, mj_data, foot_geom_ids, floor_geom_id)

                    # Full pose cho support polygon
                    oMf_l = pin_data.oMf[fid_l]
                    oMf_r = pin_data.oMf[fid_r]
                    foot_L_pose = (oMf_l.translation.copy(),
                                   oMf_l.rotation.copy())
                    foot_R_pose = (oMf_r.translation.copy(),
                                   oMf_r.rotation.copy())

                    visualizer.update(
                        viewer,
                        com_xy=com_xy,
                        com_des_xy=com_des_xy,
                        dcm_xy=dcm_xy,
                        cop_xy=cop_xy,
                        foot_L_pose=foot_L_pose,
                        foot_R_pose=foot_R_pose,
                    )
                else:
                    viewer.sync()

            # --- Log ---
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

    # --- WBC ---
    wbc = WBCSolver(cfg=CFG['WBC'], solver_backend=None)

    # --- DCM calculator ---
    q_pin0 = mj_qpos_to_pin_q(kf_qpos, pin_model)
    pin.forwardKinematics(pin_model, pin_data, q_pin0)
    com_z = float(pin.centerOfMass(pin_model, pin_data, q_pin0)[2])
    dcm_calc = DCMCalculator(com_height=com_z)
    print(f"[DCM] com_height = {com_z:.4f} m, "
          f"omega = {dcm_calc.omega:.4f} rad/s", flush=True)

    # --- Visualizer ---
    visualizer = WBCVisualizer(mj_model)

    # --- Geom IDs cho CoP ---
    foot_geom_ids = []
    for name in CFG['FOOT_GEOM_NAMES']:
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            print(f"[WARN] Không tìm thấy geom '{name}'")
        foot_geom_ids.append(gid)
    floor_geom_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, CFG['FLOOR_GEOM_NAME'])
    if floor_geom_id < 0:
        print(f"[WARN] Không tìm thấy geom '{CFG['FLOOR_GEOM_NAME']}'")
    print(f"[CoP] foot geoms = {foot_geom_ids}, floor geom = {floor_geom_id}",
          flush=True)

    # --- Chạy test ---
    if test_sel in ("all", "1"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: c,
                "TEST 1: WBC hold pose",
                dcm_calc, visualizer, foot_geom_ids, floor_geom_id)

    if test_sel in ("all", "2"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: smooth_ref(
                    t, lambda tt, cc: cc + np.array([0.01, 0, 0]),
                    c, ramp_time=1.5),
                "TEST 2: WBC CoM +1cm X (ramped 1.5s)",
                dcm_calc, visualizer, foot_geom_ids, floor_geom_id)

    if test_sel in ("all", "3"):
        run_wbc(mj_model, mj_data, pin_model, pin_data, wbc, kf_qpos,
                lambda t, c: c + np.array([0.01 * np.sin(2*np.pi*0.5*t), 0, 0]),
                "TEST 3: WBC CoM sinusoid ±1cm 0.5Hz",
                dcm_calc, visualizer, foot_geom_ids, floor_geom_id)


if __name__ == "__main__":
    main()