"""Mo phong MuJoCo Open Duck Mini V2 co GUI - LOAD ON.

File doc lap cho che do LOAD ON. Khong sua cac module dieu khien.
Dat file trong thu muc capture_point/ cung walk_controller.py va cac module khac.

Chay:
    python run_walk_load_on.py
    python run_walk_load_on.py --duration 8 --vx 0.06 --vy 0
    python run_walk_load_on.py --duration 12 --vx 0.10 --follow-camera
    python run_walk_load_on.py --duration 3 --no-gui

- Mac dinh mo MuJoCo Viewer va mo phong dong luc hoc that bang mj_step().
- Sau khi mo phong xong, cua so van mo de xem tu the cuoi; dong GUI de thoat.
- --exit-on-finish: tu dong dong cua so sau khi ket thuc mo phong.
- Khong phai chi la FK/mj_forward de hien thi tu the.
- Hai file ON/OFF su dung cung keyframe 'home' va cung tham so dieu khien.
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from walk_controller import WalkConfig, WalkController, JOINT_NAMES


# Che do duoc co dinh trong file nay; giu nguyen de doi chieu ON/OFF.
ENABLE_LOAD_COMPENSATION = True

HERE = Path(__file__).resolve().parent


def find_mjcf(xml_path: str | None) -> Path:
    if xml_path is not None:
        path = Path(xml_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Khong tim thay MJCF: {path}")
        return path

    # Ho tro chay truc tiep tu capture_point/ hoac tu thu muc khac.
    for root in (HERE, *HERE.parents):
        path = root / "xmls" / "scene_flat_terrian_v5.xml"
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Khong tim thay xmls/scene_flat_terrian_v5.xml. "
        "Hay dung --xml <duong_dan_den_file_scene>."
    )


def initialize_simulation(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    home_id = next(
        (i for i in range(model.nkey) if model.key(i).name == "home"),
        None,
    )
    if home_id is None:
        raise ValueError("MJCF phai co keyframe ten 'home'.")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mo phong Open Duck Mini V2 voi MuJoCo Viewer - LOAD ON"
    )
    parser.add_argument("--duration", type=float, default=8.0, help="Thoi gian mo phong [s]")
    parser.add_argument("--vx", type=float, default=0.06, help="Van toc X dat [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="Van toc Y dat [m/s]")
    parser.add_argument("--speed", type=float, default=1.0, help="Toc do xem: 1=real time, 2=nhanh 2 lan")
    parser.add_argument("--xml", type=str, default=None, help="Duong dan toi MJCF scene")
    parser.add_argument("--follow-camera", action="store_true", help="Camera di theo robot theo phuong X")
    parser.add_argument("--exit-on-finish", action="store_true", help="Dong viewer ngay khi mo phong xong")
    parser.add_argument("--no-gui", action="store_true", help="Chay headless de kiem tra/ghi log")
    args = parser.parse_args()
    if not all(math.isfinite(x) for x in (args.duration, args.vx, args.vy, args.speed)):
        parser.error("Tat ca gia tri tham so phai huu han")
    if args.duration <= 0 or args.speed <= 0:
        parser.error("--duration va --speed phai > 0")
    return args


def run():
    args = parse_args()
    xml = find_mjcf(args.xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    initialize_simulation(model, data)

    config = WalkConfig(enable_load_compensation=ENABLE_LOAD_COMPENSATION)
    controller = WalkController(model, num_envs=1, config=config, device="cpu")
    controller.reset(data)
    controller.set_load_compensation(ENABLE_LOAD_COMPENSATION)

    act_ids = np.asarray([model.actuator(name).id for name in JOINT_NAMES], dtype=int)
    if len(np.unique(act_ids)) != 10:
        raise ValueError("Khong tim thay 10 actuator vi tri rieng biet")

    physics_dt = float(model.opt.timestep)
    nsub_float = config.dt / physics_dt
    nsub = int(round(nsub_float))
    if not math.isclose(nsub_float, nsub, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("dt dieu khien phai la boi so nguyen cua timestep MuJoCo")

    cycles = math.ceil(args.duration / config.dt)
    base_id = controller.estimator.base_id
    print("=" * 65)
    print("Open Duck Mini V2 - LOAD ON (MuJoCo GUI)")
    print(f"MJCF: {xml}")
    print(f"vx={args.vx:.3f} m/s, vy={args.vy:.3f} m/s, T={args.duration:.2f} s")
    print(f"control dt={config.dt:.4f} s; physics dt={physics_dt:.4f} s; steps/cmd={nsub}")
    print(f"Load enabled: {controller.load_enabled}, gain={config.load_gain:.2f}")
    print("=" * 65)

    viewer_ctx = nullcontext(None)
    if not args.no_gui:
        # Dung passive viewer: script van dieu khien physics trong main thread.
        from mujoco import viewer as mj_viewer
        viewer_ctx = mj_viewer.launch_passive(model, data)

    num_valid = 0
    num_run = 0
    com_xy_err_squared = 0.0
    min_base_z = math.inf
    max_actuator_torque = 0.0
    max_delta_q_deg = 0.0
    max_reanchor_dx = 0.0
    last_report = -1

    with viewer_ctx as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.azimuth = 145.0
                viewer.cam.elevation = -20.0
                viewer.cam.distance = 1.0
                viewer.cam.lookat[:] = [0.0, 0.0, 0.17]
            viewer.sync()

        for k in range(cycles):
            if viewer is not None and not viewer.is_running():
                print("Nguoi dung da dong viewer truoc khi het mo phong.")
                break

            wall_start = time.perf_counter()
            out = controller.step(data, [args.vx, args.vy])
            q_target = out.q_cmd[0].detach().cpu().numpy().astype(np.float64)
            if not np.all(np.isfinite(q_target)):
                raise FloatingPointError("q_cmd khong huu han")

            # Ghi goc dat vao 10 position actuators; giu nguyen trong 40 substeps.
            data.ctrl[act_ids] = q_target
            for _ in range(nsub):
                mujoco.mj_step(model, data)

            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                raise FloatingPointError("Trang thai MuJoCo khong huu han")

            com_xy = np.asarray(data.subtree_com[base_id, :2], dtype=float)
            com_des_xy = out.com_des[0, :2].detach().cpu().numpy()
            err = com_xy - com_des_xy
            com_xy_err_squared += float(np.dot(err, err))
            min_base_z = min(min_base_z, float(data.xpos[base_id, 2]))
            max_actuator_torque = max(max_actuator_torque, float(np.abs(data.qfrc_actuator).max()))
            max_delta_q_deg = max(
                max_delta_q_deg,
                float(np.rad2deg(out.delta_q_load[0].abs().max().item())),
            )
            max_reanchor_dx = max(max_reanchor_dx, float(out.reanchor_dx.abs().max()))
            num_valid += int(bool(out.valid[0]))
            num_run += 1

            if viewer is not None:
                if args.follow_camera:
                    with viewer.lock():
                        viewer.cam.lookat[0] = float(data.xpos[base_id, 0])
                viewer.sync()
                # Canh thoi gian de viewer chay gan real time.
                sleep_s = config.dt / args.speed - (time.perf_counter() - wall_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            sec = int(data.time)
            if sec > last_report and (sec % 2 == 0):
                last_report = sec
                print(f"t={data.time:5.2f} s | base z={data.xpos[base_id, 2]:.3f} m | "
                      f"CoM XY err={np.linalg.norm(err):.4f} m | max dQ load={max_delta_q_deg:.3f} deg")

        if num_run:
            print("\n--- Summary LOAD ON ---")
            print(f"Simulated: {num_run * config.dt:.2f} s / requested {args.duration:.2f} s")
            print(f"CoM XY RMSE: {math.sqrt(com_xy_err_squared / num_run):.5f} m")
            print(f"Min base z:  {min_base_z:.5f} m")
            print(f"Peak actuator torque proxy |qfrc_actuator|: {max_actuator_torque:.5f} Nm")
            print(f"Max load correction: {max_delta_q_deg:.5f} deg")
            print(f"Max |reanchor_x|: {max_reanchor_dx:.5f} m")
            print(f"Valid controller ticks: {num_valid}/{num_run}")

        if viewer is not None and viewer.is_running() and not args.exit_on_finish:
            print("\nMo phong da ket thuc. GUI giu tu the cuoi. Dong cua so MuJoCo de thoat.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


if __name__ == "__main__":
    run()
