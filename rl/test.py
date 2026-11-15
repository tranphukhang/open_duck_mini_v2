"""MJLab/MuJoCo Warp standing diagnostic for the installed CWC reward.

Run from the repository root (the installed mjlab environment):
    python -m rl.test_cwc_reward_live --device cuda:0 --duration 1.0
    python -m rl.test_cwc_reward_live --device cpu --duration 1.0

Does not train PPO, change actuators, or edit any repository file.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from mjlab.scene import Scene
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from rl.cwc_reward import cwc_diagnostics
from rl.scene_cfg import get_scene_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("--duration and --interval must be > 0")
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu")

    physics_dt = 0.0005
    scene = Scene(get_scene_cfg(num_envs=1), device=device)
    compiled = scene.compile()
    sim = Simulation(
        num_envs=1,
        cfg=SimulationCfg(mujoco=MujocoCfg(
            timestep=physics_dt, integrator="implicitfast",
            solver="newton", cone="elliptic", tolerance=1e-10,
        )),
        model=compiled,
        device=device,
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.add_data_listener(scene.invalidate_sensor_caches)

    robot = scene["robot"]
    sim.reset()
    scene.reset()
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.write_data_to_sim()
    sim.forward()

    # The reward only reads scene and sim. It does not need the PPO managers.
    env_view = SimpleNamespace(scene=scene, sim=sim)
    total_steps = max(1, round(args.duration / physics_dt))
    interval_steps = max(1, round(args.interval / physics_dt))
    print("=== CWC LIVE SENSOR TEST ===")
    print("Foot order: left, right. Threshold: f_z > 5 N. Soft margin: 0.2")
    print("P_CWC is nonnegative. PPO config multiplies it by weight=-0.1")

    with torch.no_grad():
        for step in range(1, total_steps + 1):
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=physics_dt)
            if step != 1 and step % interval_steps != 0 and step != total_steps:
                continue
            sim.forward()  # refresh derived positions, contacts and sensors
            v = cwc_diagnostics(env_view)
            print(f"\nt={step * physics_dt:.3f} s | active={v.active[0].tolist()}"
                  f" | total P_CWC={v.penalty[0].item():.6f}")
            for i, name in enumerate(("L", "R")):
                print(
                    f"  {name}: friction={v.friction_margin[0, i].item():+.4f},"
                    f" CoP={v.cop_margin[0, i].item():+.4f},"
                    f" yaw={v.yaw_margin[0, i].item():+.4f}"
                    f" | P=(f:{v.friction_penalty[0, i].item():.5f},"
                    f" CoP:{v.cop_penalty[0, i].item():.5f},"
                    f" yaw:{v.yaw_penalty[0, i].item():.5f})"
                )
    print("\nLIVE TEST COMPLETE (physics only, no PPO training)")


if __name__ == "__main__":
    main()
