"""Visualize CP footstep correction and swing trajectories for walker-fk-com.

Standalone visual wrapper around the ORIGINAL BatchedZmpWalker and RobotKinematics.
No changes to the planner, MPC, CP, IK, or load-compensation logic.

Examples:
    python run_walker_visualize.py --model v5 --duration 10 --vx 0.1
    python run_walker_visualize.py --model source --duration 8 --vx 0
    python run_walker_visualize.py --model v5 --duration 4 --vx 0 --viz-frame plan

Key: yellow = nominal landing, cyan = CP-corrected landing, magenta =
planned swing path, green = current swing-foot reference, red = physical
swing-foot site, faint cyan = past corrected landing points (this step).
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reference_walker"
if not (SOURCE / "open_duck_isaac" / "walker.py").is_file():
    raise FileNotFoundError("Clone 'reference_walker' branch 'walker-fk-com' beside this file.")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "scripts"))
from check_walker_mujoco import build_model
from open_duck_isaac.constants import FOOT_SITES, JOINT_NAMES, start_pose
from open_duck_isaac.kinematics import RobotKinematics, quat_to_mat
from open_duck_isaac.walker import BatchedZmpWalker, WalkParams


def as_tensor(v):
    return torch.as_tensor(np.asarray(v).copy(), dtype=torch.float32)[None]


def v_np(x):
    return x.detach().cpu().numpy().copy()


def named_id(m, kind, name):
    out = mujoco.mj_name2id(m, kind, name)
    if out < 0:
        raise ValueError(f"MuJoCo model lacks {name!r} ({kind}). Check V5's L6_IK/R6_IK sites.")
    return out


def scene_v5():
    for p in (ROOT / "xmls" / "scene_flat_terrian_v5.xml",
              ROOT.parent / "xmls" / "scene_flat_terrian_v5.xml"):
        if p.is_file():
            return p
    raise FileNotFoundError("Cannot find xmls/scene_flat_terrian_v5.xml")


class LandingAndSwingViz:
    """Draw the ORIGINAL walker's plan state; never modify the controller."""
    YELLOW = np.array([1.0, 0.72, 0.08, 1.0], np.float32)
    CYAN = np.array([0.05, 0.93, 1.0, 1.0], np.float32)
    MAGENTA = np.array([1.0, 0.19, 0.69, 0.94], np.float32)
    GREEN = np.array([0.25, 1.0, 0.27, 1.0], np.float32)
    RED = np.array([1.0, 0.18, 0.20, 1.0], np.float32)
    WHITE = np.array([0.96, 0.96, 0.96, 1.0], np.float32)
    FAINT = np.array([0.10, 0.75, 1.0, 0.45], np.float32)

    def __init__(self, viewer, frame="stance", samples=25):
        self.viewer = viewer
        self.frame = frame
        self.samples = samples
        self.step = -99
        self.landing_history = []

    @staticmethod
    def _smooth(s):
        return 3.0 * s * s - 2.0 * s * s * s

    def _sphere(self, scene, p, rgba, radius=0.006):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            [radius, 0, 0], p, np.eye(3).ravel(), rgba,
        )
        scene.ngeom += 1

    def _segment(self, scene, a, b, rgba, width=0.0015):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(b - a) < 1e-8:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            [width, width, width], np.zeros(3), np.eye(3).ravel(), rgba,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1

    def update(self, walker, data, foot_ids):
        """Call after walker.step + mj_step, before viewer.sync."""
        j = int(walker.last_step[0].item())
        scene = self.viewer.user_scn
        if j < 0:
            with self.viewer.lock():
                scene.ngeom = 0
            return

        st = j % 2                      # 0 left support, 1 right support
        sw = 1 - st                    # opposite foot swings
        feet_start = v_np(walker.feet_start[0])
        start = feet_start[sw]
        nominal = v_np(walker.nominal_landing[0])
        corrected = v_np(walker.landing[0])
        foot_ref = v_np(walker.out_feet[0, sw])
        foot_real = np.asarray(data.site_xpos[foot_ids[sw]], dtype=float).copy()
        stance_real = np.asarray(data.site_xpos[foot_ids[st]], dtype=float).copy()

        # The walker uses a plan frame. For easier visual comparison, optionally
        # register that frame to the CURRENT PHYSICAL stance site (XY only).
        # This is a DISPLAY-ONLY translation: no control value is changed.
        offset = np.zeros(3)
        if self.frame == "stance":
            offset[:2] = stance_real[:2] - feet_start[st, :2]

        if j != self.step:
            self.step = j
            self.landing_history = []
        if not self.landing_history or np.linalg.norm(
            corrected[:2] - self.landing_history[-1][:2]
        ) > 0.00015:
            self.landing_history.append(corrected.copy())
            self.landing_history = self.landing_history[-40:]

        samples = np.linspace(0.0, 1.0, self.samples)
        path = []
        for s in samples:
            pt = start + (corrected - start) * self._smooth(s)
            # EXACT '_feet_at' vertical profile on flat ground.
            pt[2] = start[2] + walker.p.swing_height * 64.0 * s**3 * (1.0 - s)**3
            path.append(pt + offset)

        marker_lift = np.array([0, 0, 0.009])
        with self.viewer.lock():
            scene.ngeom = 0
            for a, b in zip(path[:-1], path[1:]):
                self._segment(scene, a, b, self.MAGENTA)
            # Points where the foot was commanded to land during the CURRENT step
            for p in self.landing_history:
                self._sphere(scene, p + offset + marker_lift, self.FAINT, 0.0023)
            self._sphere(scene, start + offset + marker_lift, self.WHITE, 0.006)
            self._sphere(scene, nominal + offset + marker_lift, self.YELLOW, 0.008)
            self._sphere(scene, corrected + offset + marker_lift, self.CYAN, 0.006)
            self._sphere(scene, foot_ref + offset, self.GREEN, 0.006)
            self._sphere(scene, foot_real, self.RED, 0.005)
            # Arrow/segment: nominal footstep -> CP-corrected footstep
            self._segment(
                scene, nominal + offset + marker_lift,
                corrected + offset + marker_lift, self.CYAN, 0.0028,
            )


def main():
    parser = argparse.ArgumentParser(description="CP landing and swing trajectory visualization")
    parser.add_argument("--model", choices=("source", "v5"), default="v5")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--vx", type=float, default=0.1)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cp-off", action="store_true",
                        help="Disable CP footstep correction for A/B visualization")
    parser.add_argument("--viz-frame", choices=("stance", "plan"), default="stance")
    parser.add_argument("--viz-samples", type=int, default=25)
    args = parser.parse_args()

    if args.model == "source":
        m = build_model()
    else:
        sc = scene_v5()
        print("Loading:", sc)
        m = mujoco.MjModel.from_xml_path(str(sc))
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[named_id(m, mujoco.mjtObj.mjOBJ_JOINT, x)]
                     for x in JOINT_NAMES])
    aids = np.array([named_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in JOINT_NAMES])
    fid = np.array([named_id(m, mujoco.mjtObj.mjOBJ_SITE, x) for x in FOOT_SITES])
    base = named_id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    if args.model == "source":
        d.qpos[3] = 1.0
        d.qpos[qadr] = start_pose(10.0)
        d.ctrl[aids] = d.qpos[qadr]
    else:
        key = named_id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.ctrl[aids] = d.qpos[qadr]
    mujoco.mj_forward(m, d)
    d.qpos[2] += 0.002 - np.min(d.site_xpos[fid, 2])
    mujoco.mj_forward(m, d)

    params = WalkParams(dt=0.02)      # Same parameters as original source.
    if args.cp_off:
        params.step_gain = (0.0, 0.0)   # Only for an explicit A/B test.
    substeps = int(round(params.dt / m.opt.timestep))
    if not np.isclose(substeps * m.opt.timestep, params.dt):
        raise ValueError("Model timestep must divide the 0.02-second control period")
    for _ in range(int(round(0.3 / m.opt.timestep))):
        mujoco.mj_step(m, d)

    kin = RobotKinematics(m)
    walker = BatchedZmpWalker(kin, 1, params)
    walker.reset(
        torch.tensor([0], dtype=torch.long),
        as_tensor(d.qpos[:3]),
        quat_to_mat(as_tensor(d.qpos[3:7])),
        as_tensor(d.qpos[qadr]),
    )

    columns = ["t", "step", "phase", "swing", "nom_x", "nom_y", "nom_z",
               "cp_x", "cp_y", "cp_z", "cp_dx_mm", "cp_dy_mm",
               "dcm_err_x", "dcm_err_y", "swing_ref_x", "swing_ref_y",
               "swing_ref_z", "swing_real_x", "swing_real_y", "swing_real_z"]
    out = ROOT / "walker_cp_swing_visual_trace.csv"
    viewer = None if args.headless else mujoco.viewer.launch_passive(m, d)
    viz = LandingAndSwingViz(viewer, args.viz_frame, max(5, args.viz_samples)) if viewer else None

    if viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base
            viewer.cam.distance = 0.82
            viewer.cam.elevation = -25.0
            viewer.cam.azimuth = 130.0
            viewer.cam.lookat[:] = d.xpos[base].copy()

    print("\nVisualizer colors:")
    print("YELLOW=nominal landing; CYAN=CP landing; MAGENTA=swing curve")
    print("GREEN=planned foot now; RED=physical foot site; WHITE=swing start")
    print("Faint CYAN=previous CP landing positions in the current step")
    print(f"Frame={args.viz_frame}; CP gain={params.step_gain}; Walker source files are NOT modified.\n")

    last_phase = None
    try:
        with out.open("w", newline="", encoding="utf-8") as f:
            log = csv.writer(f)
            log.writerow(columns)
            for k in range(int(round(args.duration / params.dt))):
                start_wall = time.perf_counter()
                now = k * params.dt
                ramp = np.clip((now - 2.0) / 1.0, 0.0, 1.0)
                cmd = torch.tensor([[args.vx * ramp, args.vy * ramp]], dtype=torch.float32)
                with torch.no_grad():
                    q = v_np(walker.step(
                        cmd, quat_to_mat(as_tensor(d.qpos[3:7])),
                        as_tensor(d.qpos[qadr]),
                    )[0])
                if not np.isfinite(q).all() or bool(walker.bad[0].item()):
                    print(f"Walker non-finite at {now:.2f}s")
                    break
                d.ctrl[aids] = q
                for _ in range(substeps):
                    mujoco.mj_step(m, d)

                j = int(walker.last_step[0].item())
                contact = v_np(walker.contact[0])
                if j < 0:
                    phase = "SHIFT"
                elif not contact[1]:
                    phase = "SWING_R"
                elif not contact[0]:
                    phase = "SWING_L"
                else:
                    phase = "DSP"
                sw = 1 - (j % 2) if j >= 0 else 1
                nom = v_np(walker.nominal_landing[0])
                cp = v_np(walker.landing[0])
                ref = v_np(walker.out_feet[0, sw])
                actual = d.site_xpos[fid[sw]].copy()
                delta = (cp - nom) * 1000.0
                err = v_np(walker.dcm_err[0])
                log.writerow([now + params.dt, j, phase, "R" if sw == 1 else "L",
                              *nom, *cp, delta[0], delta[1], *err,
                              *ref, *actual])

                if viz:
                    viz.update(walker, d, fid)
                    viewer.sync()
                    if not viewer.is_running():
                        break
                    remain = params.dt - (time.perf_counter() - start_wall)
                    if remain > 0:
                        time.sleep(remain)

                if phase != last_phase or k % 25 == 24:
                    print(f"t={now + params.dt:5.2f}  j={j:3d}  {phase:8s}  "
                          f"landing adjustment=({delta[0]:+6.1f}, {delta[1]:+6.1f}) mm  "
                          f"swing Z desired/real={ref[2]:.3f}/{actual[2]:.3f} m")
                last_phase = phase

                R = d.xmat[base].reshape(3, 3)
                roll = np.arctan2(R[2, 1], R[2, 2])
                pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
                if d.xpos[base, 2] < 0.12 or abs(roll) > np.deg2rad(40) or abs(pitch) > np.deg2rad(40):
                    print(f"FALL at {now + params.dt:.2f}s")
                    break
    finally:
        if viewer:
            viewer.close()
    print("CSV:", out)


if __name__ == "__main__":
    main()
