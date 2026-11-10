"""Visualize full 14-point footstep plan and scheduled reanchor_x events.

Place NEXT TO run_walker_reference.py in the user's main/ folder.
No edits to the upstream walker or MuJoCo model are needed.

Examples:
    python run_walker_reanchor_viz.py --model v5 --duration 6 --vx 0.1
    python run_walker_reanchor_viz.py --model v5 --duration 6 --vx 0.1 --event-pause 1.0
    python run_walker_reanchor_viz.py --model v5 --duration 6 --vx 0.1 --reanchor-off
    python run_walker_reanchor_viz.py --model v5 --duration 6 --vx 0.1 --cp-off
    python run_walker_reanchor_viz.py --model v5 --duration 6 --vx 0.1 --headless --plot

Reference-frame convention:
- W = MuJoCo world (physical site_xpos, for plotting only).
- B = moving robot base (the kinematic tree is rooted at B).
- plan = internal plan coordinates initialized from world at reset, subsequently
  translated by reanchor_x. There is NO explicit {P} transform in the repo.
- FK uses base_pos=0 and measured world-aligned base_rot; site_new-site_old
  is thus a RELATIVE foot displacement expressed along world-aligned axes.
- Visualization registers the PRE-EVENT plan to the observed OLD stance foot.
  That display-only registration is NOT an estimate of absolute world position.

Event capture occurs BEFORE mj_step for the same control tick. The event is
triggered by the scheduler, NOT by MuJoCo contact detection.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Uses the already-working baseline provided in user's repository.
from run_walker_reference import (
    ROOT, build_model, find_scene, get_id, tensor,
    FOOT_SITES, JOINT_NAMES, start_pose, quat_to_mat,
    RobotKinematics, BatchedZmpWalker, WalkParams,
)


def npcopy(t):
    return t.detach().cpu().numpy().copy()


class ReanchorViewer:
    """Draw event snapshots in MuJoCo. Pure visualization; never changes controls."""
    C = {
        "old_stance": np.array([0.25, 0.45, 1.00, 1.0], dtype=float),  # blue
        "planned":    np.array([1.00, 0.72, 0.12, 1.0], dtype=float),  # yellow
        "fk":         np.array([0.10, 0.95, 0.92, 1.0], dtype=float),  # cyan
        "updated":    np.array([0.20, 0.95, 0.30, 1.0], dtype=float),  # green
        "real":       np.array([1.00, 0.18, 0.20, 1.0], dtype=float),  # red
        "com_before": np.array([0.95, 0.35, 0.95, 1.0], dtype=float),  # pink
        "com_after":  np.array([0.66, 0.43, 1.00, 1.0], dtype=float),  # violet
        "next_before":np.array([1.00, 0.78, 0.32, 0.7], dtype=float),
        "next_after": np.array([0.30, 0.98, 0.40, 0.8], dtype=float),
        "sequence_before": np.array([1.00, 0.67, 0.15, 0.48], dtype=float),
        "sequence_after": np.array([0.20, 0.90, 0.40, 0.92], dtype=float),
        "sequence_current": np.array([0.05, 0.85, 1.00, 1.00], dtype=float),
        "sequence_shift": np.array([0.80, 0.42, 0.95, 0.55], dtype=float),
    }

    def __init__(self, viewer):
        self.viewer = viewer
        self.event = None
        self.live = None

    @staticmethod
    def sphere(scn, xyz, rgba, size=0.008):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            [size, 0, 0], np.asarray(xyz, dtype=float),
                            np.eye(3).ravel(), rgba)
        scn.ngeom += 1

    @staticmethod
    def segment(scn, a, b, rgba, width=0.0025):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if scn.ngeom >= scn.maxgeom or np.linalg.norm(b - a) < 1e-6:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            [width, width, width], np.zeros(3),
                            np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1

    def draw(self):
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            ev = self.event
            if ev is None:
                # Live 50-Hz footstep sequence, using the CURRENT stance
                # solely for display registration to the physical robot.
                live = self.live
                if live is None:
                    return
                pts = live["points"] + live["world_offset"] + np.array([0, 0, 0.015])
                for i in range(len(pts)-1):
                    self.segment(scn, pts[i], pts[i+1],
                                 self.C["sequence_after"], 0.0017)
                for i in range(len(pts)):
                    self.sphere(scn, pts[i], self.C["sequence_after"], 0.0058)
                self.sphere(scn, pts[1], self.C["old_stance"], 0.009)  # P_0
                self.sphere(scn, pts[2], self.C["sequence_current"], 0.010)  # P_1
                return

            # All displayed plan markers are aligned to OLD stance at the
            # event. The same pre-event registration is used before and after.
            anchor = ev["world_offset"]
            lift = np.array([0.0, 0.0, 0.012])
            high = np.array([0.0, 0.0, 0.040])

            old = ev["old_foot_world"] + lift
            actual_new = ev["new_foot_world"] + lift
            plan = ev["new_foot_plan_before"] + anchor + lift

            # Only the X component changes in reanchor_x. Hold planned Y/Z
            # fixed for the 'FK estimate' and 'after' markers: this makes
            # the X correction geometrically unambiguous.
            fk = plan.copy()
            fk[0] = ev["x_fk_est"] + anchor[0]
            updated = plan.copy()
            updated[0] = ev["x_plan_after"] + anchor[0]

            self.segment(scn, old, plan, self.C["old_stance"], 0.0012)
            self.segment(scn, plan, fk, self.C["fk"], 0.003)
            self.sphere(scn, old, self.C["old_stance"], 0.009)
            self.sphere(scn, actual_new, self.C["real"], 0.007)
            self.sphere(scn, plan, self.C["planned"], 0.012)
            self.sphere(scn, fk, self.C["fk"], 0.008)
            # After plan and FK coincide when reanchor is enabled:
            # green dot sits at center of cyan dot.
            self.sphere(scn, updated, self.C["updated"], 0.0045)

            cb = ev["com_before"] + anchor + high
            ca = ev["com_after"] + anchor + high
            self.segment(scn, cb, ca, self.C["com_after"], 0.0028)
            self.sphere(scn, cb, self.C["com_before"], 0.010)
            self.sphere(scn, ca, self.C["com_after"], 0.006)

            # The ACTUAL _footsteps() output at the event: P_-1, P_0, ..., P_12.
            # Both snapshots use the SAME display-only transform (old stance),
            # so the visible horizontal displacement equals the reanchor shift.
            # BEFORE is reconstructed from the exact AFTER sequence by undoing
            # the uniform X translation; it is NOT a second optimizer run.
            before = ev["sequence_before"] + anchor + lift
            after = ev["sequence_after"] + anchor + lift
            npts = len(after)

            # Draw 14-point polylines first, then individual footstep markers.
            for i in range(npts - 1):
                self.segment(scn, before[i], before[i + 1],
                             self.C["sequence_before"], 0.0009)
                self.segment(scn, after[i], after[i + 1],
                             self.C["sequence_after"], 0.0017)
            for i in range(npts):
                self.sphere(scn, before[i], self.C["sequence_before"], 0.0048)
                self.sphere(scn, after[i], self.C["sequence_after"], 0.0058)
                if abs(ev["dx_applied"]) > 0.0001 and i in (1, 2, 4, 7, 10, 13):
                    self.segment(scn, before[i], after[i],
                                 self.C["sequence_shift"], 0.001)
            # P_0 is the newly activated stance; P_1 is the current landing
            # that CP may have changed. This highlight is on the POST sequence.
            self.sphere(scn, after[1], self.C["old_stance"], 0.008)
            self.sphere(scn, after[2], self.C["sequence_current"], 0.009)

            # A second target illustrates translation of NEXT nominal step.
            nb = ev["next_nominal_before"] + anchor + lift
            na = ev["next_nominal_after"] + anchor + lift
            self.segment(scn, nb, na, self.C["next_after"], 0.0018)
            self.sphere(scn, nb, self.C["next_before"], 0.006)
            self.sphere(scn, na, self.C["next_after"], 0.004)


def make_event_before(walker, kin, model_data, foot_ids, base_rot, q, j):
    """Read pre-event state with the SAME FK inputs used by walker._measure()."""
    new_st = j % 2           # stance that is being activated
    old_st = 1 - new_st      # previous stance

    feet_before = npcopy(walker.feet_start[0])
    previous_landing = npcopy(walker.landing[0])
    com_before = npcopy(walker.com_state[0, :, 0])

    with torch.no_grad():
        ks = kin.fk(torch.zeros((1, 3), dtype=q.dtype, device=q.device), base_rot, q)
    fk_sites = npcopy(ks.site_pos[0])
    foot_relative = fk_sites[new_st] - fk_sites[old_st]
    # Same scalar as source's: fs[old_st] + rel[old_st] - rel[new_st]
    x_fk_est = float(feet_before[old_st, 0] + foot_relative[0])
    x_plan_before = float(previous_landing[0])
    dx_needed = x_fk_est - x_plan_before

    old_world = np.asarray(model_data.site_xpos[foot_ids[old_st]], dtype=float).copy()
    new_world = np.asarray(model_data.site_xpos[foot_ids[new_st]], dtype=float).copy()
    # Register PLAN coordinates to the old physical stance (DISPLAY ONLY).
    world_offset = old_world - feet_before[old_st]

    return {
        "t": float(walker.t[0].item()), "step": j,
        "old_st": old_st, "new_st": new_st,
        "old_foot_world": old_world,
        "new_foot_world": new_world,
        "new_foot_plan_before": previous_landing,
        "old_foot_plan_before": feet_before[old_st],
        "world_offset": world_offset,
        "fk_displacement_x": float(foot_relative[0]),
        "x_fk_est": x_fk_est,
        "x_plan_before": x_plan_before,
        "dx_needed": dx_needed,
        "com_before": np.array([com_before[0], com_before[1],
                                npcopy(walker.com_height[0]).item()], dtype=float),
    }


def finish_event(ev, walker):
    st = ev["new_st"]
    ev["x_plan_after"] = float(walker.feet_start[0, st, 0].item())
    ev["dx_applied"] = ev["x_plan_after"] - ev["x_plan_before"]
    # Walker.step already ran the NEXT MPC prediction. Its com_state is
    # x_(k+1|k), NOT the intermediate state immediately after reanchor.
    # Reconstruct that intermediate state exactly using the applied shift.
    ev["com_after"] = ev["com_before"].copy()
    ev["com_after"][0] += ev["dx_applied"]
    ev["com_mpc_next_x"] = float(walker.com_state[0, 0, 0].item())
    ev["next_nominal_after"] = npcopy(walker.nominal_landing[0])
    ev["next_nominal_before"] = ev["next_nominal_after"].copy()
    ev["next_nominal_before"][0] -= ev["dx_applied"]
    # '_footsteps' is exactly the original planner's 14-point output:
    # index 0 = P_-1, index 1 = P_0, index 2 = P_1, ..., index 13 = P_12.
    # walker.step() incremented walker.t, so explicitly pass the event's j.
    j_at_event = torch.tensor([ev["step"]], dtype=torch.long, device=walker.dev)
    with torch.no_grad():
        points, _ = walker._footsteps(j_at_event)
    ev["sequence_after"] = npcopy(points[0])
    # The source shifts BOTH feet_start and landing by exactly the SAME X.
    # Step offsets and the DCM error are translation-invariant, thus the
    # step sequence immediately BEFORE reanchor is AFTER minus applied dx.
    ev["sequence_before"] = ev["sequence_after"].copy()
    ev["sequence_before"][:, 0] -= ev["dx_applied"]


def optional_plot(events, filepath):
    if not events:
        print("No j>=1 transition, nothing to plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("--plot requires matplotlib (pip install matplotlib). CSV is still saved.")
        return
    t = np.array([e["t"] for e in events]);
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, [e["x_plan_before"] * 1000 for e in events], "o-", label="new foot plan BEFORE")
    ax[0].plot(t, [e["x_fk_est"] * 1000 for e in events], "x-", label="new foot estimated from FK")
    ax[0].plot(t, [e["x_plan_after"] * 1000 for e in events], ".--", label="new foot plan AFTER")
    ax[0].set_ylabel("X in internal plan coordinates (mm)")
    ax[0].grid(True, alpha=0.25); ax[0].legend()
    ax[1].plot(t, [e["dx_needed"] * 1000 for e in events], "o-", label="FK correction required")
    ax[1].plot(t, [e["dx_applied"] * 1000 for e in events], ".--", label="reanchor shift applied")
    ax[1].axhline(0, color="gray", lw=0.7)
    ax[1].set_ylabel("Delta x (mm)")
    ax[1].set_xlabel("Scheduled step transition time (s)")
    ax[1].grid(True, alpha=0.25); ax[1].legend()
    fig.suptitle("reanchor_x: old stance + FK relative displacement -> new stance estimate")
    fig.tight_layout(); fig.savefig(filepath, dpi=160); plt.close(fig)
    print("Plot:", filepath)


def main():
    ap = argparse.ArgumentParser(description="Visualize walker-fk-com reanchor_x events")
    ap.add_argument("--model", choices=("v5", "source"), default="v5")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--vx", type=float, default=0.1)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--reanchor-off", action="store_true", help="Keep reanchor disabled while still logging the FK discrepancy")
    ap.add_argument("--cp-off", action="store_true", help="Disable CP landing correction for comparison")
    ap.add_argument("--event-pause", type=float, default=0.45, help="Wall seconds to pause physics at each event for inspection")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--plot", action="store_true", help="Save a 2-panel event-summary PNG (needs matplotlib)")
    args = ap.parse_args()

    if args.model == "source":
        m = build_model()
    else:
        scene = find_scene()
        print("Loading:", scene)
        m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    qadr = np.array([m.jnt_qposadr[get_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    aids = np.array([get_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
    foot_ids = np.array([get_id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES])
    base_id = get_id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    if args.model == "source":
        d.qpos[3] = 1.0
        d.qpos[qadr] = start_pose(10.0)
        d.ctrl[aids] = d.qpos[qadr]
    else:
        key = get_id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.ctrl[aids] = d.qpos[qadr]
    mujoco.mj_forward(m, d)
    d.qpos[2] += 0.002 - min(d.site_xpos[i, 2] for i in foot_ids)
    mujoco.mj_forward(m, d)

    params = WalkParams(dt=0.02)
    if args.reanchor_off:
        params.reanchor_x = False
    if args.cp_off:
        params.step_gain = (0.0, 0.0)

    substeps = int(round(params.dt / m.opt.timestep))
    if not np.isclose(substeps * m.opt.timestep, params.dt):
        raise ValueError("Physics timestep does not evenly divide control dt")
    for _ in range(int(round(0.3 / m.opt.timestep))):
        mujoco.mj_step(m, d)

    kin = RobotKinematics(m)
    walker = BatchedZmpWalker(kin, 1, params)
    walker.reset(torch.tensor([0], dtype=torch.long), tensor(d.qpos[:3]),
                 quat_to_mat(tensor(d.qpos[3:7])), tensor(d.qpos[qadr]))

    viewer = None if args.headless else mujoco.viewer.launch_passive(m, d)
    viz = ReanchorViewer(viewer) if viewer is not None else None
    if viewer is not None:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
            viewer.cam.distance = 0.90
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 125
            viewer.cam.lookat[:] = d.xpos[base_id].copy()

    print("\nREANCHOR_X VISUALIZER (only scheduler-triggered, starting at t=0.6 s)")
    print("BLUE=old stance; YELLOW=new stance PLAN before; CYAN=FK estimate;")
    print("GREEN=new stance plan after; RED=actual new foot (MuJoCo)")
    print("PINK/VIOLET=CoM plan before/after; small YELLOW/GREEN=next nominal landing")
    print("LIVE (every 20 ms): GREEN=P_-1..P_12, BLUE=P_0, CYAN=P_1")
    print("AT REANCHOR EVENT: AMBER faint=BEFORE; GREEN=AFTER, then resume LIVE")
    print("BLUE large=P_0 new stance; CYAN large=P_1 current landing (after CP)")
    print("NOTE: plan is display-registered to OLD stance; no physical World foot position feedback")
    print("Visual markers aligned to actual OLD stance ONLY FOR DISPLAY (no world pose estimator).")
    print(f"reanchor_x={params.reanchor_x}; CP gain={params.step_gain}")
    print("At event: new stance PLAN is overwritten with previous LANDING, then shifted.")
    print("Sequence BEFORE is the post-step sequence with the uniform reanchor X shift undone.")

    events = []
    out = ROOT / "walker_reanchor_events.csv"
    fields = ["t", "step", "old_stance", "new_stance", "x_old_plan",
              "dx_feet_fk", "x_new_plan_before", "x_new_fk_est",
              "dx_needed", "dx_applied", "x_new_plan_after",
              "com_x_before", "com_x_after_reanchor", "com_x_next_mpc",
              "next_nominal_x_before",
              "next_nominal_x_after", "actual_new_foot_world_x"]
    try:
        for k in range(int(round(args.duration / params.dt))):
            wall_start = time.perf_counter()
            now = k * params.dt
            ramp = np.clip((now - 2.0) / 1.0, 0.0, 1.0)
            cmd = torch.tensor([[args.vx * ramp, args.vy * ramp]], dtype=torch.float32)
            qmeas = tensor(d.qpos[qadr]); rot = quat_to_mat(tensor(d.qpos[3:7]))

            # Same scheduler and BEFORE-physics measured inputs as upstream.
            j_now = int(walker._phase(walker.t)[0][0].item())
            last = int(walker.last_step[0].item())
            is_new_valid_step = j_now >= 1 and j_now != last
            ev = None
            if is_new_valid_step:
                ev = make_event_before(walker, kin, d, foot_ids, rot, qmeas, j_now)

            with torch.no_grad():
                qtarget = npcopy(walker.step(cmd, rot, qmeas)[0])
            if not np.isfinite(qtarget).all() or bool(walker.bad[0].item()):
                print(f"Walker error at t={now:.2f}")
                break

            if ev is not None:
                finish_event(ev, walker)
                events.append(ev)
                print(
                    f"EVENT t={ev['t']:.2f}s old={'L' if ev['old_st']==0 else 'R'} "
                    f"-> new={'L' if ev['new_st']==0 else 'R'} | "
                    f"planBefore={1000*ev['x_plan_before']:+7.1f} mm  "
                    f"FKest={1000*ev['x_fk_est']:+7.1f} mm | "
                    f"needDx={1000*ev['dx_needed']:+6.1f} mm  "
                    f"appliedDx={1000*ev['dx_applied']:+6.1f} mm | "
                    f"planAfter={1000*ev['x_plan_after']:+7.1f} mm"
                )
                # Print first 5 of the 14 positions to clarify indexing.
                first = ", ".join(
                    f"P_{i-1}=({1000*ev['sequence_after'][i,0]:+.0f},"
                    f"{1000*ev['sequence_after'][i,1]:+.0f})mm"
                    for i in range(5)
                )
                print(f"  Steps: {first}; ... through P_12")
                if viz is not None:
                    viz.event = ev
                    viz.draw(); viewer.sync()
                    # Freeze only PHYSICS; user can rotate the camera during pause.
                    if args.event_pause > 0.0:
                        end_wait = time.perf_counter() + args.event_pause
                        while time.perf_counter() < end_wait and viewer.is_running():
                            viz.draw(); viewer.sync(); time.sleep(0.04)

            # IMPORTANT: source control timing x_k -> apply u_k -> x_(k+1).
            d.ctrl[aids] = qtarget
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            if viewer is not None:
                # Always show the latest 14-step plan between reanchor events.
                # The scheduled event snapshot is drawn only during the pause.
                with torch.no_grad():
                    pts_live, _ = walker._footsteps(
                        torch.tensor([j_now], dtype=torch.long, device=walker.dev)
                    )
                stance = (max(j_now, 0) % 2)
                anchor_world = np.asarray(d.site_xpos[foot_ids[stance]]).copy()
                anchor_plan = npcopy(walker.feet_start[0, stance])
                viz.live = {
                    "points": npcopy(pts_live[0]),
                    "world_offset": anchor_world-anchor_plan,
                }
                viz.event = None
                viz.draw(); viewer.sync()
                if not viewer.is_running():
                    break
                remaining = params.dt - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
            R = d.xmat[base_id].reshape(3, 3)
            roll = np.arctan2(R[2, 1], R[2, 2]); pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
            if d.xpos[base_id, 2] < 0.12 or abs(roll) > np.deg2rad(40) or abs(pitch) > np.deg2rad(40):
                print(f"FALL at t={now + params.dt:.2f}s")
                break
    finally:
        if viewer is not None:
            viewer.close()
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for ev in events:
                w.writerow({
                    "t": ev["t"], "step": ev["step"],
                    "old_stance": "L" if ev["old_st"] == 0 else "R",
                    "new_stance": "L" if ev["new_st"] == 0 else "R",
                    "x_old_plan": ev["old_foot_plan_before"][0],
                    "dx_feet_fk": ev["fk_displacement_x"],
                    "x_new_plan_before": ev["x_plan_before"],
                    "x_new_fk_est": ev["x_fk_est"],
                    "dx_needed": ev["dx_needed"],
                    "dx_applied": ev["dx_applied"],
                    "x_new_plan_after": ev["x_plan_after"],
                    "com_x_before": ev["com_before"][0],
                    "com_x_after_reanchor": ev["com_after"][0],
                    "com_x_next_mpc": ev["com_mpc_next_x"],
                    "next_nominal_x_before": ev["next_nominal_before"][0],
                    "next_nominal_x_after": ev["next_nominal_after"][0],
                    "actual_new_foot_world_x": ev["new_foot_world"][0],
                })
        # Detailed event-by-event FOOTSTEP SEQUENCE. Positions in the
        # walker's internal plan coordinates, NOT physical MuJoCo World XYZ.
        seq_out = ROOT / "walker_reanchor_footstep_sequence.csv"
        with seq_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "step", "point", "role", "x_before", "y_before",
                        "z_before", "x_after", "y_after", "z_after", "dx_applied"])
            for ev in events:
                for i, (pb, pa) in enumerate(zip(ev["sequence_before"], ev["sequence_after"])):
                    point = f"P_{i-1}"
                    role = "old_stance" if i == 0 else (
                        "new_stance" if i == 1 else (
                        "current_landing" if i == 2 else "future_footstep"))
                    w.writerow([ev["t"], ev["step"], point, role,
                                *[float(v) for v in pb], *[float(v) for v in pa],
                                ev["dx_applied"]])
        print("Full 14-step sequence CSV:", seq_out)
        print(f"\nEvents logged: {len(events)}; CSV: {out}")
        if args.plot:
            optional_plot(events, ROOT / "walker_reanchor_events.png")


if __name__ == "__main__":
    main()
