"""Kiem thu tich hop gait references -> WholeBodyIK, Open Duck Mini V2 MJCF v5.

Dat file tai: <repo>/capture_point/test_integrated_wbik.py

Chay tu thu muc capture_point/:
    python test_integrated_wbik.py
    python test_integrated_wbik.py --duration 4 --vx 0.08 --plot
    python test_integrated_wbik.py --duration 4 --vx 0.08 --viewer
    python test_integrated_wbik.py --cp --plot
    python test_integrated_wbik.py --envs 2 --save wbik_trace.npz

Duong du lieu (tat ca trong he toa do ke hoach {P}):
    ContactScheduler -> FootstepPlanner -> ZMPGenerator -> LIPMMPC
    -> SwingTrajectoryGenerator -> WholeBodyIK -> RobotKinematics (FK).

- CoM/feet tham chieu duoc sinh tu cac module that; home pose/FK tu MJCF v5.
- WBIK tinh bo 16 bien (base + 10 khop), warm-start 8/1 vong lap.
- --viewer: MuJoCo chi HIEN THI qpos theo nghiem IK (mj_forward, KHONG mj_step).
  Khong kiem chung can bang/contact/torque/servo hoac co the di thuc te.
- --cp (tuy chon): kiem thu API CapturePoint bang 'measurement' lay tu FK cua
  nghiem IK truoc do. Day KHONG PHAI measurement tu mo phong vat ly.
- Khong dung reanchor_x: chua co do luong FK tu he vat ly tach biet voi plan.
- Khong ghi file neu khong truyen --save. --plot chi hien thi do thi.

Yeu cau: torch, numpy, mujoco. matplotlib neu --plot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner
from capture_point import CapturePoint
from zmp_generator import ZMPGenerator
from lipm_mpc import LIPMMPC
from swing_trajectory import SwingTrajectoryGenerator
from robot_kinematics import RobotKinematics
from whole_body_IK import WholeBodyIK, IkWeights  # Ten file tren GitHub: IK viet hoa

DT = 0.02
PERIOD_PHASE = 0.4
SHIFT = 0.4
DOUBLE_SUPPORT = 0.1
SWING_HEIGHT = 0.02
HORIZON = 80
NUM_FOOTSTEPS = 14

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)


def find_mjcf(xml_arg: str | None) -> Path:
    if xml_arg is not None:
        path = Path(xml_arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MJCF khong ton tai: {path}")
        return path
    for root in (HERE, *HERE.parents):
        path = root / "xmls" / "scene_flat_terrian_v5.xml"
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Khong tim thay xmls/scene_flat_terrian_v5.xml. "
        "Hay dat script trong capture_point/ cua repository, hoac truyen --xml."
    )


def read_home_pose(model: mujoco.MjModel):
    """Doc qpos 'home' trong MuJoCo; KHONG hard-code q tu test cu."""
    home_id = None
    for idx in range(model.nkey):
        if model.key(idx).name == "home":
            home_id = idx
            break
    if home_id is None:
        raise ValueError("MJCF v5 can keyframe 'home' de khoi tao test.")
    home = np.array(model.key_qpos[home_id], dtype=np.float64)
    free_id = model.joint("floating_base").id
    base_adr = int(model.jnt_qposadr[free_id])
    p0 = home[base_adr:base_adr + 3].copy()
    quat = home[base_adr + 3:base_adr + 7]
    R0 = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R0, quat)
    R0 = R0.reshape(3, 3)
    q0 = np.array([
        home[int(model.jnt_qposadr[model.joint(n).id])]
        for n in JOINT_NAMES
    ], dtype=np.float64)
    return p0, R0, q0, base_adr


def set_mujoco_pose(model, data, base_adr, base_p, base_R, q):
    """Ghi qpos va mj_forward de visualize / doi chieu FK; KHONG mj_step."""
    data.qpos[:] = model.qpos0
    data.qpos[base_adr:base_adr + 3] = base_p
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(base_R, dtype=np.float64).reshape(9))
    data.qpos[base_adr + 3:base_adr + 7] = quat
    for j, name in enumerate(JOINT_NAMES):
        adr = int(model.jnt_qposadr[model.joint(name).id])
        data.qpos[adr] = float(q[j])
    mujoco.mj_forward(model, data)


def validate_mujoco_fk(model, data, base_adr, p, R, q, ks, tick):
    """Doi chieu FK torch va MJCF MuJoCo (chi env=0, khong mo phong dynamics)."""
    set_mujoco_pose(
        model, data, base_adr,
        p[0].detach().cpu().numpy(),
        R[0].detach().cpu().numpy(),
        q[0].detach().cpu().numpy(),
    )
    site_ids = [model.site(name).id for name in ("L6_IK", "R6_IK")]
    feet_mj = np.asarray(data.site_xpos[site_ids], dtype=np.float64)
    com_mj = np.asarray(data.subtree_com[model.body("base_link").id], dtype=np.float64)
    feet_torch = ks.site_pos[0].detach().cpu().numpy()
    com_torch = ks.com[0].detach().cpu().numpy()
    pos_diff = float(np.max(np.abs(feet_torch - feet_mj)))
    com_diff = float(np.max(np.abs(com_torch - com_mj)))
    if pos_diff > 3e-4 or com_diff > 3e-4:
        raise AssertionError(
            f"MuJoCo-FK mismatch at tick={tick}: "
            f"feet max={pos_diff:.3e}m, CoM max={com_diff:.3e}m"
        )


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def check_finite(name, value, tick):
    require(bool(torch.isfinite(value).all().item()),
            f"{name} chua NaN/Inf tai tick={tick}")


@torch.no_grad()
def run_pipeline(args, model, device):
    """Tich hop that cac module, KHONG mo phong dynamics/contact physics."""
    B = args.envs
    if B < 1:
        raise ValueError("--envs phai >= 1")
    n_ticks = round(args.duration / DT)
    if n_ticks < round((SHIFT + PERIOD_PHASE) / DT) or abs(n_ticks * DT - args.duration) > 1e-7:
        raise ValueError("--duration phai >= 0.8s va la boi so cua 0.02s")
    if args.iter_first < 1 or args.iter_warm < 1:
        raise ValueError("--iter-first va --iter-warm phai >= 1")

    kin = RobotKinematics(model, device=device)
    ik = WholeBodyIK(kin, IkWeights())
    p_init_np, R_init_np, q_init_np, base_adr = read_home_pose(model)

    p = torch.tensor(p_init_np, dtype=torch.float32, device=device)[None].repeat(B, 1)
    R = torch.tensor(R_init_np, dtype=torch.float32, device=device)[None].repeat(B, 1, 1)
    q = torch.tensor(q_init_np, dtype=torch.float32, device=device)[None].repeat(B, 1)
    q_home = q.clone()
    ks = kin.forward_kinematics(p, R, q)
    feet0 = ks.site_pos.detach().clone()
    feet_rot0 = ks.site_rot.detach().clone()  # giu huong chan tai tu the home
    com_z = ks.com[:, 2].clone()

    # LIPM hoat dong tren mat phang chan: h = z_CoM - z_foot (khong hard-code).
    floor_z = feet0[:, :, 2].mean(dim=1)
    h = (com_z - floor_z).mean().item()
    if h <= 0.03:
        raise AssertionError(f"Chieu cao LIPM khong hop le: h={h:.6f} m")

    scheduler = ContactScheduler(
        dt=DT, period_phase=PERIOD_PHASE, shift=SHIFT,
        double_support=DOUBLE_SUPPORT, num_envs=B, device=device,
    )
    planner = FootstepPlanner(
        num_envs=B, period_phase=PERIOD_PHASE,
        num_footsteps=NUM_FOOTSTEPS, device=device,
    )
    zmp_gen = ZMPGenerator(horizon=HORIZON)
    mpc = LIPMMPC(com_height=h, dt=DT, horizon=HORIZON, device=device)
    mpc.build_model()
    mpc.build_state_prediction()
    mpc.build_output_prediction()
    mpc.precompute_gains()
    swing_gen = SwingTrajectoryGenerator(
        period_phase=PERIOD_PHASE, double_support=DOUBLE_SUPPORT,
        swing_height=args.swing_height,
    )

    # CP chi la che do kiem tra giao dien; FK cua IK khong phai do luong vat ly.
    initial_width = (feet0[:, 0, 1] - feet0[:, 1, 1]).abs().min().item()
    min_width = min(0.12, max(0.04, initial_width - 0.01))
    cp = CapturePoint(com_height=h, min_step_width=min_width, device=device)

    scheduler.reset()
    planner.reset(torch.arange(B, device=device), feet0)
    com_state = torch.zeros(B, 2, 3, dtype=torch.float32, device=device)
    com_state[:, :, 0] = ks.com[:, :2]

    # Neu envs > 1, giam nhe vx theo env de kiem tra xu ly batch.
    speeds = torch.linspace(1.0, 0.65, B, device=device) if B > 1 else torch.ones(1, device=device)
    cmd = torch.stack([speeds * args.vx, torch.full((B,), args.vy, device=device)], dim=1)
    last_step = torch.full((B,), -1, dtype=torch.long, device=device)
    prev_com_for_velocity = ks.com.clone()
    batch = torch.arange(B, device=device)

    store_keys = (
        "t", "j", "contact", "com_ref", "com_fk", "feet_ref", "feet_fk",
        "base_pos", "base_rot", "q", "zmp_ref", "landing", "q_delta",
    )
    trace = {key: [] for key in store_keys}
    n_steps = torch.zeros(B, dtype=torch.long, device=device)
    n_swing = torch.zeros(B, dtype=torch.long, device=device)
    n_adjust = torch.zeros(B, dtype=torch.long, device=device)
    initial_feet = feet0.clone()
    data_check = mujoco.MjData(model)
    validate_mujoco_fk(model, data_check, base_adr, p, R, q, ks, tick=-1)

    for tick in range(n_ticks):
        t = scheduler.t.clone()                # t_k
        j_now = scheduler.step_index(t)
        stance = scheduler.stance_feet(j_now)
        new_step = scheduler.new_step_event(j_now, last_step)
        planner.update_new_step(j_now, new_step, stance, cmd)
        n_steps += new_step.long()

        # CP tuy chon: so sanh IK-FK tai t_k voi du doan LIPM.
        # Day KHONG PHAI phan hoi tu cam bien hoac mo phong vat ly.
        if args.cp:
            vcom_proxy = (ks.com - prev_com_for_velocity) / DT
            _, _, dcm_err = cp.compute_dcm_error(ks.com, vcom_proxy, com_state)
            corr = cp.compute_correction(dcm_err)
            stance_y = planner.feet_start[batch, stance, 1]
            landing_cp = cp.correct_landing(planner.nominal_landing, corr, stance, stance_y)
            allow = scheduler.allow_footstep_adjust(t, j_now) & (j_now >= 0)
            planner.set_landing(landing_cp, allow)
            n_adjust += allow.long()

        footsteps, j_base = planner.footstep_sequence(j_now, cmd)
        zmp_ref = zmp_gen.generate(footsteps, planner.feet0, j_base, scheduler)
        vel_ref = cmd[:, None, :].expand(-1, HORIZON, -1)
        check_finite("zmp_ref", zmp_ref, tick)
        com_state_next = mpc.step(com_state, zmp_ref, vel_ref)
        check_finite("com_state_next", com_state_next, tick)

        # Swing target is t_(k+1), same as com_state_next.
        t_next = t + DT
        j_next = scheduler.step_index(t_next)
        local_next = scheduler.local_time_phase(t_next, j_next)
        _, _, ss_next = scheduler.phase_state(j_next, local_next)
        contact_next = scheduler.contact_state(scheduler.stance_feet(j_next), ss_next)
        feet_des, contact_ret = swing_gen.generate(
            footsteps, planner.feet0, j_base, j_next, local_next, contact_next
        )
        require(bool(torch.equal(contact_ret, contact_next)), "Swing contact mismatch")
        n_swing += (~contact_next).sum(dim=1)

        com_des = torch.cat([com_state_next[:, :, 0], com_z[:, None]], dim=1)
        check_finite("com_des", com_des, tick)
        check_finite("feet_des", feet_des, tick)

        p_before = p.clone()
        q_before = q.clone()
        old_com = ks.com.clone()
        n_iter = args.iter_first if tick == 0 else args.iter_warm
        p, R, q, ks = ik.solve(
            p, R, q, com_des, feet_des, feet_rot0, q_home,
            iterations=n_iter, ks0=ks,
        )

        # Kiem tra tinh nhat quan va an toan so hoc cua nghiem WBIK.
        for name, value in (("base_pos", p), ("base_rot", R),
                            ("joints", q), ("com_fk", ks.com),
                            ("feet_fk", ks.site_pos)):
            check_finite(name, value, tick)
        require(q.shape == (B, 10) and ks.com.shape == (B, 3), "WBIK output shape mismatch")
        require(bool(torch.all(q >= kin.qmin - 2e-5).item()), "q < qmin")
        require(bool(torch.all(q <= kin.qmax + 2e-5).item()), "q > qmax")
        R_err = R @ R.transpose(-1, -2)
        require(bool(torch.allclose(
            R_err, torch.eye(3, device=device)[None].expand_as(R_err),
            atol=1e-3, rtol=0,
        )), "Base rotation is not orthonormal")

        # Validate that returned KinState agrees with freshly recomputed FK.
        ks_check = kin.forward_kinematics(p, R, q)
        require(bool(torch.allclose(ks.com, ks_check.com, atol=2e-5, rtol=0)),
                "Returned KinState does not match FK (CoM)")
        require(bool(torch.allclose(ks.site_pos, ks_check.site_pos, atol=2e-5, rtol=0)),
                "Returned KinState does not match FK (feet)")

        # Doi chieu doc lap voi engine MuJoCo tai mot so tick.
        if tick % 20 == 0 or tick == n_ticks - 1:
            validate_mujoco_fk(model, data_check, base_adr, p, R, q, ks, tick)

        trace["t"].append(np.full((B,), (tick + 1) * DT))
        trace["j"].append(j_next.cpu().numpy().copy())
        trace["contact"].append(contact_next.cpu().numpy().copy())
        trace["com_ref"].append(com_des.cpu().numpy().copy())
        trace["com_fk"].append(ks.com.cpu().numpy().copy())
        trace["feet_ref"].append(feet_des.cpu().numpy().copy())
        trace["feet_fk"].append(ks.site_pos.cpu().numpy().copy())
        trace["base_pos"].append(p.cpu().numpy().copy())
        trace["base_rot"].append(R.cpu().numpy().copy())
        trace["q"].append(q.cpu().numpy().copy())
        trace["zmp_ref"].append(zmp_ref[:, 0, :].cpu().numpy().copy())
        trace["landing"].append(planner.updated_landing.cpu().numpy().copy())
        trace["q_delta"].append((q - q_before).cpu().numpy().copy())

        # Prepare for the next 20-ms tick.
        prev_com_for_velocity = old_com
        com_state = com_state_next
        last_step = j_now
        scheduler.count()

    for key in store_keys:
        trace[key] = np.stack(trace[key], axis=0)
    trace["initial_feet"] = initial_feet.cpu().numpy()
    trace["n_steps"] = n_steps.cpu().numpy()
    trace["n_swing"] = n_swing.cpu().numpy()
    trace["n_adjust"] = n_adjust.cpu().numpy()
    trace["cmd"] = cmd.cpu().numpy()
    trace["h"] = np.array(h)
    trace["dt"] = np.array(DT)
    trace["model"] = np.array("open_duck_mini_v5")
    return trace, base_adr


def print_report(trace):
    com = trace["com_fk"] - trace["com_ref"]
    feet = trace["feet_fk"] - trace["feet_ref"]
    contacts = trace["contact"]
    print("\n========== WBIK INTEGRATION TEST ==========")
    print(f"Duration: {trace['t'].shape[0]*DT:.2f}s | dt={DT:.3f}s "
          f"| h_LIPM={float(trace['h']):.4f}m")
    for b in range(trace["cmd"].shape[0]):
        com_xy = np.linalg.norm(com[:, b, :2], axis=-1)
        foot_norm = np.linalg.norm(feet[:, b, :, :], axis=-1)
        qvel_max = np.max(np.abs(trace["q_delta"][:, b, :])) / DT
        stance_mask = contacts[:, b, :]
        swing_mask = ~stance_mask
        stance_rmse = float(np.sqrt(np.mean(foot_norm[:, :][stance_mask] ** 2))) if np.any(stance_mask) else float("nan")
        swing_rmse = float(np.sqrt(np.mean(foot_norm[:, :][swing_mask] ** 2))) if np.any(swing_mask) else float("nan")
        print(f"\nEnv {b}: cmd=[{trace['cmd'][b,0]:.3f}, {trace['cmd'][b,1]:.3f}] m/s")
        print(f"  steps={trace['n_steps'][b]}, swing_samples={trace['n_swing'][b]}, "
              f"CP_applied={trace['n_adjust'][b]}")
        print(f"  CoM XY RMSE       = {np.sqrt(np.mean(com_xy**2))*1000:.3f} mm")
        print(f"  CoM XYZ RMSE      = {np.sqrt(np.mean(np.sum(com[:,b,:]**2,axis=-1)))*1000:.3f} mm")
        print(f"  Foot left RMSE    = {np.sqrt(np.mean(foot_norm[:,0]**2))*1000:.3f} mm")
        print(f"  Foot right RMSE   = {np.sqrt(np.mean(foot_norm[:,1]**2))*1000:.3f} mm")
        print(f"  Stance foot RMSE  = {stance_rmse*1000:.3f} mm")
        print(f"  Swing foot RMSE   = {swing_rmse*1000:.3f} mm")
        print(f"  Max |delta q|/dt  = {qvel_max:.3f} rad/s (kinematic only)")
        if np.sqrt(np.mean(com_xy**2)) > 0.02 or stance_rmse > 0.02:
            print("  [WARN] Tracking errors exceed 20mm; review gains, targets, and warm IK iterations.")
    assert np.all(np.isfinite(trace["q"]))
    assert np.all(trace["n_steps"] > 0), "No step event recorded"
    assert np.all(trace["n_swing"] > 0), "No swing phase recorded"
    print("\n[PASS] Pipeline output: finite, joint limits, FK consistency, gait events.")
    print("[NOTE] PASS does not imply stable contact or physically feasible joint torques.")


def plot_trace(trace, env=0):
    import matplotlib.pyplot as plt
    t = trace["t"][:, env]
    com_ref = trace["com_ref"][:, env]
    com_fk = trace["com_fk"][:, env]
    feet_ref = trace["feet_ref"][:, env]
    feet_fk = trace["feet_fk"][:, env]
    contact = trace["contact"][:, env]
    q = trace["q"][:, env]
    com_err = np.linalg.norm(com_fk - com_ref, axis=-1) * 1000
    f_err = np.linalg.norm(feet_fk - feet_ref, axis=-1) * 1000

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    fig.suptitle(f"Open Duck Mini V2: gait references -> WBIK (kinematic; env={env})")

    axs[0, 0].plot(t, com_ref[:, 0], label="CoM x ref")
    axs[0, 0].plot(t, com_fk[:, 0], "--", label="CoM x FK")
    axs[0, 0].plot(t, trace["zmp_ref"][:, env, 0], ":", label="ZMP x ref")
    axs[0, 0].set_ylabel("X (m)")
    axs[0, 0].legend(loc="best")

    axs[0, 1].plot(t, com_ref[:, 1], label="CoM y ref")
    axs[0, 1].plot(t, com_fk[:, 1], "--", label="CoM y FK")
    axs[0, 1].plot(t, trace["zmp_ref"][:, env, 1], ":", label="ZMP y ref")
    axs[0, 1].set_ylabel("Y (m)")
    axs[0, 1].legend(loc="best")

    for foot, label in [(0, "Left"), (1, "Right")]:
        axs[1, 0].plot(t, feet_ref[:, foot, 2], label=f"{label} z ref")
        axs[1, 0].plot(t, feet_fk[:, foot, 2], "--", label=f"{label} z FK")
    axs[1, 0].set_ylabel("Foot Z (m)")
    axs[1, 0].legend(loc="best", fontsize="small")

    axs[1, 1].plot(t, com_err, label="CoM 3D error")
    axs[1, 1].plot(t, f_err[:, 0], label="Left foot error")
    axs[1, 1].plot(t, f_err[:, 1], label="Right foot error")
    axs[1, 1].set_ylabel("Tracking error (mm)")
    axs[1, 1].legend(loc="best", fontsize="small")

    axs[2, 0].plot(t, np.rad2deg(q))
    axs[2, 0].set_ylabel("Joint angles (deg)")
    axs[2, 0].set_xlabel("Time (s)")

    axs[2, 1].step(t, contact[:, 0].astype(float), where="post", label="Contact L")
    axs[2, 1].step(t, contact[:, 1].astype(float), where="post", label="Contact R")
    axs[2, 1].plot(t, trace["j"][:, env] / max(1, trace["j"][:, env].max()),
                   ":", label="Normalized step index")
    axs[2, 1].set_ylabel("Contact / phase")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].legend(loc="best")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def replay_mujoco(model, trace, base_adr, env=0, speed=1.0):
    """Chi phat lai dong hoc. Khong tich phan contact/dynamics."""
    if speed <= 0:
        raise ValueError("--speed must be > 0")
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("mujoco.viewer not available. Try `pip install mujoco`.") from exc
    data = mujoco.MjData(model)
    print("[VIEWER] MuJoCo playback of IK configuration ONLY; no mj_step().")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for k in range(len(trace["t"])):
            if not viewer.is_running():
                break
            set_mujoco_pose(
                model, data, base_adr,
                trace["base_pos"][k, env],
                trace["base_rot"][k, env],
                trace["q"][k, env],
            )
            data.time = float(trace["t"][k, env])
            viewer.sync()
            time.sleep(DT / speed)


def main():
    parser = argparse.ArgumentParser(description="Integration: gait planning -> WBIK -> MuJoCo FK")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds, multiple of 0.02")
    parser.add_argument("--vx", type=float, default=0.08, help="Desired forward speed [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="Desired lateral speed [m/s]")
    parser.add_argument("--swing-height", type=float, default=SWING_HEIGHT)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--iter-first", type=int, default=8)
    parser.add_argument("--iter-warm", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--xml", type=str, default=None, help="Optional path to v5 scene XML")
    parser.add_argument("--cp", action="store_true", help="Test CP using IK FK as proxy, NOT real feedback")
    parser.add_argument("--plot", action="store_true", help="Show Matplotlib tracking plots")
    parser.add_argument("--viewer", action="store_true", help="Kinematic playback in MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=1.0, help="MuJoCo viewer playback speed")
    parser.add_argument("--save", type=str, default=None, help="Optional .npz trace output")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA not available; use --device cpu")
    device = torch.device(args.device)
    xml_path = find_mjcf(args.xml)
    print(f"[MODEL] {xml_path}")
    print(f"[CONFIG] {args.duration:.2f}s, vx={args.vx:.3f}, vy={args.vy:.3f}, "
          f"CP={args.cp}, envs={args.envs}, warm_iterations={args.iter_warm}")
    torch.manual_seed(20261114)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    trace, base_adr = run_pipeline(args, model, device)
    print_report(trace)
    if args.save:
        path = Path(args.save).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **trace)
        print(f"[SAVED] {path}")
    if args.plot:
        plot_trace(trace)
    if args.viewer:
        replay_mujoco(model, trace, base_adr, speed=args.speed)


if __name__ == "__main__":
    main()
