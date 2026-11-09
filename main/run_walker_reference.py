
from pathlib import Path
import sys
import csv
import time
import argparse

import torch
import numpy as np
import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reference_walker"

if not (SOURCE / "open_duck_isaac" / "walker.py").is_file():
    raise FileNotFoundError(
        "Chua clone reference_walker (branch walker-fk-com)"
    )

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "scripts"))

# Import TRUC TIEP code goc
from check_walker_mujoco import build_model
from open_duck_isaac.constants import (
    FOOT_SITES, JOINT_NAMES, start_pose
)
from open_duck_isaac.kinematics import (
    RobotKinematics, quat_to_mat
)
from open_duck_isaac.walker import (
    BatchedZmpWalker, WalkParams
)


def tensor(x):
    return torch.as_tensor(
        np.asarray(x).copy(),
        dtype=torch.float32
    ).unsqueeze(0)


def find_scene():
    candidates = [
        ROOT / "xmls" / "scene_flat_terrian_v5.xml",
        ROOT.parent / "xmls" / "scene_flat_terrian_v5.xml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Khong tim thay scene_flat_terrian_v5.xml"
    )


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)

    if idx < 0:
        raise ValueError(
            f"Model thieu: {name}. "
            "Kiem tra ten joint, actuator hoac site."
        )

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["source", "v5"],
        default="source"
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--vx", type=float, default=0.1)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # 1. Load MuJoCo model
    if args.model == "source":
        model = build_model()
    else:
        scene = find_scene()
        print("Loading:", scene)
        model = mujoco.MjModel.from_xml_path(str(scene))

    data = mujoco.MjData(model)

    qadr = np.array([
        model.jnt_qposadr[
            get_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        ]
        for n in JOINT_NAMES
    ])

    aids = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in JOINT_NAMES
    ])

    foot_ids = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        for n in FOOT_SITES
    ])

    base_id = get_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )

    # 2. Initial pose
    if args.model == "source":
        data.qpos[3] = 1.0
        data.qpos[qadr] = start_pose(10.0)
        data.ctrl[aids] = data.qpos[qadr]
    else:
        key_id = get_id(
            model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.ctrl[aids] = data.qpos[qadr]

    mujoco.mj_forward(model, data)

    # Place soles slightly above ground
    min_z = min(data.site_xpos[i, 2] for i in foot_ids)
    data.qpos[2] += 0.002 - min_z
    mujoco.mj_forward(model, data)

    dt = 0.02
    physics_dt = model.opt.timestep
    substeps = int(round(dt / physics_dt))

    if not np.isclose(substeps * physics_dt, dt):
        raise ValueError("Physics timestep incompatible with 50 Hz")

    # Stand for 0.3 seconds (same source procedure)
    for _ in range(int(round(0.3 / physics_dt))):
        mujoco.mj_step(model, data)

    # 3. Original FK and Walker
    kin = RobotKinematics(model)
    params = WalkParams(dt=dt)
    walker = BatchedZmpWalker(kin, 1, params)

    walker.reset(
        torch.tensor([0], dtype=torch.long),
        tensor(data.qpos[:3]),
        quat_to_mat(tensor(data.qpos[3:7])),
        tensor(data.qpos[qadr])
    )

    print("=" * 65)
    print("OPEN DUCK WALKER - walker-fk-com")
    print("Model:", args.model)
    print("Control dt:", dt)
    print("Physics dt:", physics_dt)
    print("Start CoM:", data.subtree_com[base_id])
    print("Start feet:", data.site_xpos[foot_ids])
    print("=" * 65)

    # 4. Simulation + CSV
    log_path = ROOT / "walker_reference_trace.csv"

    columns = [
        "time", "phase",
        "com_plan_x", "com_plan_y",
        "com_real_x", "com_real_y",
        "zmp_x", "zmp_y",
        "dcm_error_x", "dcm_error_y",
        "roll_deg", "pitch_deg",
        "base_z", "contact_L", "contact_R",
        "max_joint_error_deg"
    ]

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    try:
        with log_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)

            steps = int(round(args.duration / dt))

            for k in range(steps):
                t = k * dt
                wall_start = time.perf_counter()

                # x_k: actual measured joint angles and base orientation
                q_meas = tensor(data.qpos[qadr])
                base_rot = quat_to_mat(
                    tensor(data.qpos[3:7])
                )

                # Same velocity ramp as original source:
                # walking in place first, then forward after 2 seconds
                ramp = np.clip((t - 2.0) / 1.0, 0.0, 1.0)

                command = torch.tensor(
                    [[args.vx * ramp, args.vy * ramp]],
                    dtype=torch.float32
                )

                # 5. ORIGINAL CP + ZMP-MPC + WBIK + LOAD COMP
                with torch.no_grad():
                    q_target = walker.step(
                        command,
                        base_rot,
                        q_meas
                    )[0].cpu().numpy()

                if (
                    not np.all(np.isfinite(q_target))
                    or bool(walker.bad[0])
                ):
                    print("Walker error at:", t)
                    break

                # Apply u_k, then advance physics to x_(k+1)
                data.ctrl[aids] = q_target

                for _ in range(substeps):
                    mujoco.mj_step(model, data)

                # Read post-step state
                R = data.xmat[base_id].reshape(3, 3)

                roll = np.degrees(
                    np.arctan2(R[2, 1], R[2, 2])
                )
                pitch = np.degrees(
                    -np.arcsin(np.clip(R[2, 0], -1, 1))
                )

                com_real = data.subtree_com[base_id].copy()
                com_plan = walker.out_com[0].cpu().numpy()
                zmp = walker.out_zmp[0].cpu().numpy()
                dcm = walker.dcm_err[0].cpu().numpy()
                contact = walker.contact[0].cpu().numpy()

                q_err = np.degrees(
                    np.max(np.abs(q_target - data.qpos[qadr]))
                )

                if t + dt < params.start_shift - 1e-8:
                    phase = "SHIFT"
                elif not contact[1]:
                    phase = "SWING_R"
                elif not contact[0]:
                    phase = "SWING_L"
                else:
                    phase = "DSP"

                writer.writerow([
                    t + dt, phase,
                    com_plan[0], com_plan[1],
                    com_real[0], com_real[1],
                    zmp[0], zmp[1],
                    dcm[0], dcm[1],
                    roll, pitch,
                    data.xpos[base_id, 2],
                    int(contact[0]), int(contact[1]),
                    q_err
                ])

                if k % 10 == 9:
                    print(
                        f"t={t + dt:.2f}s | {phase:8s} | "
                        f"CoM_y plan/real="
                        f"{com_plan[1]:+.4f}/{com_real[1]:+.4f} | "
                        f"roll={roll:+.1f} deg | "
                        f"q_err={q_err:.1f} deg"
                    )

                if viewer is not None:
                    viewer.sync()

                    elapsed = time.perf_counter() - wall_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                    if not viewer.is_running():
                        break

                if (
                    data.xpos[base_id, 2] < 0.12
                    or abs(roll) > 40.0
                    or abs(pitch) > 40.0
                ):
                    print(
                        f"FALL at t={t + dt:.2f}s | "
                        f"roll={roll:.1f}, pitch={pitch:.1f}"
                    )
                    break

    finally:
        if viewer is not None:
            viewer.close()

    print("Finished. CSV:", log_path)


if __name__ == "__main__":
    main()
