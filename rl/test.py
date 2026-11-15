"""Smoke test for Open Duck Mini V2 scene on MJLab main.

Run from repository root:
    python rl/test_scene_cfg.py --device auto --duration 0.5
    python rl/test_scene_cfg.py --compile-only
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from rl.robot_cfg import FOOT_GEOM_NAMES, FOOT_SITE_NAMES, HOME_KEYFRAME, JOINT_NAMES
from rl.scene_cfg import get_scene_cfg


def check_compiled_scene(scene: Scene, model: mujoco.MjModel) -> None:
    print("\n=== SCENE / MODEL ===")
    print(f"njnt={model.njnt}, nq={model.nq}, nv={model.nv}, nu={model.nu}, nsensor={model.nsensor}")
    assert (model.njnt, model.nq, model.nv, model.nu) == (11, 17, 16, 10)

    for name in JOINT_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{name}") >= 0
    for name in FOOT_GEOM_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}") >= 0
    for name in FOOT_SITE_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{name}") >= 0
    print("10 joints/actuators, foot geoms and sites: OK")

    key = model.key("init_state")
    expected = np.array([HOME_KEYFRAME.joint_pos[name] for name in JOINT_NAMES])
    np.testing.assert_allclose(key.qpos[:3], HOME_KEYFRAME.pos, atol=1e-6)
    np.testing.assert_allclose(key.qpos[3:7], HOME_KEYFRAME.rot, atol=1e-6)
    np.testing.assert_allclose(key.qpos[7:17], expected, atol=1e-6)
    np.testing.assert_allclose(key.ctrl[:10], expected, atol=1e-6)
    print("Home keyframe (qpos and ctrl): OK")

    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert ground_id >= 0
    np.testing.assert_allclose(
        model.geom_friction[ground_id], [0.8, 0.005, 0.0001], rtol=1e-6, atol=1e-8
    )
    assert int(model.geom_condim[ground_id]) == 3
    assert int(model.geom_contype[ground_id]) == 1
    assert int(model.geom_conaffinity[ground_id]) == 1
    print("Terrain friction, condim and collision bits: OK")

    sensor = scene["feet_contact"]
    primary_names = sensor.primary_names
    for foot in FOOT_GEOM_NAMES:
        assert sum(name == foot or name.endswith("/" + foot) for name in primary_names) == 1
    assert len(primary_names) == 2
    assert sensor.cfg.fields == ("found",)
    assert sensor.cfg.track_air_time
    print("Contact switch primaries:", primary_names)
    print("Contact switch config: OK")


def run_warp_test(scene: Scene, model: mujoco.MjModel, device: str, duration: float) -> None:
    # Explicitly restore the original V5 physics settings; MuJoCo's XML options
    # are not automatically propagated when the robot is attached to a Scene.
    physics_dt = 0.0005
    sim = Simulation(
        num_envs=1,
        cfg=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=physics_dt,
                integrator="implicitfast",
                solver="newton",
                cone="elliptic",
                tolerance=1e-10,
            ),
        ),
        model=model,
        device=device,
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)

    robot = scene["robot"]
    sensor = scene["feet_contact"]

    # MJLab's init_state must be written explicitly for this standalone test;
    # sim.reset() alone is not a substitute for the RL environment's reset flow.
    sim.reset()
    scene.reset()
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.write_data_to_sim()
    sim.forward()

    print(f"\n=== MUJOCO WARP | {device} ===")
    print(f"Physics timestep: {sim.mj_model.opt.timestep:.6f} s")
    print(f"Initial base z: {robot.data.root_link_pos_w[0, 2].item():.4f} m")
    assert abs(robot.data.root_link_pos_w[0, 2].item() - 0.275) < 1e-4

    nsteps = max(1, round(duration / physics_dt))
    log_every = max(1, round(0.1 / physics_dt))
    ever_contact = torch.zeros((2,), dtype=torch.bool, device=device)

    for step in range(1, nsteps + 1):
        scene.write_data_to_sim()  # Apply the constant home position targets.
        sim.step()
        scene.update(dt=physics_dt)  # Updates contact/air-time tracking.

        contact_data = sensor.data
        assert contact_data.found is not None
        assert contact_data.found.shape == (1, 2)
        assert contact_data.current_air_time is not None
        assert contact_data.current_contact_time is not None
        assert contact_data.current_air_time.shape == (1, 2)
        assert contact_data.current_contact_time.shape == (1, 2)

        switch = contact_data.found[0] > 0
        ever_contact |= switch

        if step == 1 or step % log_every == 0 or step == nsteps:
            base_z = robot.data.root_link_pos_w[0, 2].item()
            max_err_deg = torch.rad2deg(
                (robot.data.joint_pos - robot.data.default_joint_pos).abs().max()
            ).item()
            left = int(switch[0].item())
            right = int(switch[1].item())
            print(
                f"t={step * physics_dt:5.3f} s | base z={base_z:.4f} m | "
                f"contact (L,R)=({left},{right}) | max joint err={max_err_deg:.2f} deg"
            )
            assert np.isfinite(base_z)
            assert np.isfinite(max_err_deg)

    print("Ever detected contact (L,R):", [bool(x) for x in ever_contact.tolist()])
    if not ever_contact.all().item():
        print("WARNING: At least one foot never contacted the terrain. Inspect pose/collision.")
    print("MuJoCo Warp stepping and contact sensor: OK")
    print("Note: this is a smoke test, not proof of stable standing or RL readiness.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be > 0")

    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available() and not args.compile_only:
        parser.error("CUDA is not available; run --device cpu or fix the driver")

    cfg = get_scene_cfg(num_envs=1)
    assert cfg.num_envs == 1
    assert cfg.terrain is not None and cfg.terrain.terrain_type == "plane"
    scene = Scene(cfg, device="cpu" if args.compile_only else device)
    model = scene.compile()
    check_compiled_scene(scene, model)
    if args.compile_only:
        print("\nCOMPILE-ONLY TEST PASSED")
        return
    run_warp_test(scene, model, device, args.duration)
    print("\nALL CHECKS COMPLETED")


if __name__ == "__main__":
    main()
