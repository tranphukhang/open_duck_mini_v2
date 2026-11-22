"""Run an Open Duck Mini V2 PPO policy and collect motion/contact traces.

Usage from the repository root:
    python -m rl.run --device cuda:0 --vx 0.08            # Native MuJoCo GUI
    python -m rl.run --device cuda:0 --vx 0.08 --no-gui   # Fast evaluation
    python -m rl.run --checkpoint /path/to/model_XXX.pt --no-gui

Simulation lasts 10 s (500 policy steps at 0.02 s). With --gui,
Figures 1 (base motion) and 3 (foot contacts) update live, while Figure 2
(joint positions) opens only after the simulation. With --no-gui, no viewer
or Matplotlib figures open until evaluation finishes. No files are saved.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from rl.env_cfg import EPISODE_LENGTH_S, POLICY_DT
from rl.play import ContactViewer, find_checkpoint, prepare_evaluation_cfg
from rl.plotter import Plotter
from rl.ppo_cfg import get_ppo_cfg
from rl.robot_cfg import FOOT_GEOM_NAMES, JOINT_NAMES


NORMAL_FORCE_THRESHOLD_N = 5.0


def quaternion_wxyz_to_rpy_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    """ZYX Euler angles (roll, pitch, yaw) for base relative to World [deg]."""
    w, x, y, z = quat_wxyz.detach().cpu().tolist()
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Invalid zero-norm base quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def get_joint_indices(robot) -> tuple[int, ...]:
    """Map JOINT_NAMES onto indices of robot.data.joint_pos."""
    name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
    missing = [name for name in JOINT_NAMES if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing robot joints: {missing}")
    if len(JOINT_NAMES) != 10:
        raise RuntimeError("Expected five left and five right joints")
    return tuple(name_to_idx[name] for name in JOINT_NAMES)


def get_foot_wrench_indices(env) -> tuple[int, int]:
    """Map left/right collision geoms to feet_wrench primary sensor ordering."""
    sensor = env.scene["feet_wrench"]
    if sensor.cfg.reduce != "netforce":
        raise RuntimeError("feet_wrench must use reduce='netforce'")
    names = sensor.primary_names
    ids: list[int] = []
    for geom_name in FOOT_GEOM_NAMES:  # left, right
        matches = [
            i for i, name in enumerate(names)
            if name == geom_name or name.endswith("/" + geom_name)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one feet_wrench primary for '{geom_name}', got {names}"
            )
        ids.append(matches[0])
    return ids[0], ids[1]


class EvaluationSampler:
    """Collect desired/actual velocities, base motion, joints and foot contacts."""

    def __init__(self, env: ManagerBasedRlEnv, plotter: Plotter):
        self.env = env
        self.plotter = plotter
        self.joint_indices = get_joint_indices(env.scene["robot"])
        self.foot_indices = get_foot_wrench_indices(env)

    def read_command(self) -> tuple[float, float]:
        """Read the command in body frame before stepping/resampling."""
        command = self.env.command_manager.get_command("twist")
        if command is None:
            raise RuntimeError("Command 'twist' is missing")
        return tuple(command[0, :2].detach().cpu().tolist())

    def record(self, command_xy: tuple[float, float], *, reset: bool = False) -> None:
        """Record one policy sample, with NaN at an auto/manual reset boundary."""
        if reset:
            vx_actual = vy_actual = math.nan
            roll_deg = pitch_deg = yaw_deg = math.nan
            wx_rad_s = wy_rad_s = wz_rad_s = math.nan
            left_joint_pos = [math.nan] * 5
            right_joint_pos = [math.nan] * 5
            contact_left = contact_right = math.nan
        else:
            robot = self.env.scene["robot"]
            data = robot.data
            vx_actual, vy_actual = (
                data.root_link_lin_vel_b[0, :2].detach().cpu().tolist()
            )
            roll_deg, pitch_deg, yaw_deg = quaternion_wxyz_to_rpy_deg(
                data.root_link_quat_w[0]
            )
            wx_rad_s, wy_rad_s, wz_rad_s = (
                data.root_link_ang_vel_b[0, :3].detach().cpu().tolist()
            )
            joints = (
                data.joint_pos[0, list(self.joint_indices)].detach().cpu().tolist()
            )
            left_joint_pos, right_joint_pos = joints[:5], joints[5:]

            wrench = self.env.scene["feet_wrench"].data
            if wrench.force is None or wrench.found is None:
                raise RuntimeError("feet_wrench must provide 'force' and 'found'")

            # Sensor primary=foot, secondary=terrain, reduce='netforce'.
            # In this repository, force is FOOT-on-TERRAIN (World frame).
            # Negating it gives the ground reaction on the foot; its world Z
            # component is the normal force on the horizontal ground.
            normal_force = -wrench.force[0, list(self.foot_indices), 2]
            found = wrench.found[0, list(self.foot_indices)] > 0
            contacts = (found & (normal_force > NORMAL_FORCE_THRESHOLD_N))
            contact_left, contact_right = (
                contacts.to(torch.int32).detach().cpu().tolist()
            )

        self.plotter.add_sample(
            vx_desired=command_xy[0],
            vx_actual=vx_actual,
            vy_desired=command_xy[1],
            vy_actual=vy_actual,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            wx_rad_s=wx_rad_s,
            wy_rad_s=wy_rad_s,
            wz_rad_s=wz_rad_s,
            joint_pos_left=left_joint_pos,
            joint_pos_right=right_joint_pos,
            contact_left=contact_left,
            contact_right=contact_right,
        )


class PlottingContactViewer(ContactViewer):
    """Native MuJoCo viewer that records once per successful policy step."""

    def __init__(
        self,
        env,
        policy,
        sampler: EvaluationSampler,
        *,
        force_scale: float = 1.0,
    ):
        super().__init__(env, policy, force_scale=force_scale)
        self.sampler = sampler
        self._break_after_manual_reset = False

    def reset_environment(self) -> None:
        super().reset_environment()
        self._break_after_manual_reset = True

    def _execute_step(self) -> bool:
        command_xy = self.sampler.read_command()
        if not super()._execute_step():
            return False
        reset = (
            self.env.unwrapped.episode_length_buf[0].item() == 0
            or self._break_after_manual_reset
        )
        self.sampler.record(command_xy, reset=reset)
        self._break_after_manual_reset = False
        self.sampler.plotter.refresh()  # Figures 1 and 3, in GUI mode.
        return True


def run_without_gui(
    wrapped: RslRlVecEnvWrapper,
    policy,
    sampler: EvaluationSampler,
    max_policy_steps: int,
) -> None:
    """Policy evaluation with no Viewer, plotting, or wall-clock pacing."""
    reset_fn = getattr(policy, "reset", None)
    with torch.inference_mode():
        for _ in range(max_policy_steps):
            command_xy = sampler.read_command()
            observations = wrapped.get_observations()
            actions = policy(observations)
            _, _, dones, _ = wrapped.step(actions)
            if reset_fn is not None:
                reset_fn(dones)
            reset = sampler.env.episode_length_buf[0].item() == 0
            sampler.record(command_xy, reset=reset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vx", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument(
        "--gui", dest="gui", action="store_true",
        help="Show MuJoCo Viewer and live Figures 1/3 (default)",
    )
    gui_group.add_argument(
        "--no-gui", dest="gui", action="store_false",
        help="Run without viewer/live plots, display results after simulation",
    )
    parser.set_defaults(gui=True)
    parser.add_argument(
        "--refresh-interval", type=float, default=0.10,
        help="Live Figures 1/3 refresh interval in wall-clock seconds",
    )
    parser.add_argument(
        "--force-scale", type=float, default=1.0,
        help="Contact-force arrow scale in GUI mode (visual only)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.vx <= 0.1:
        parser.error("--vx must lie in [0.0, 0.1] m/s")
    if args.refresh_interval < 0 or args.force_scale <= 0:
        parser.error("Require --refresh-interval >= 0 and --force-scale > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; check the environment or use --device cpu")
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        cuda_id = torch.device(args.device).index or 0
        torch.cuda.set_device(cuda_id)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(cuda_id))
    configure_torch_backends()

    ppo_cfg = get_ppo_cfg()
    checkpoint = find_checkpoint(
        args.checkpoint, Path(args.log_root), ppo_cfg.experiment_name
    )
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Fixed body-frame command: vx={args.vx:.3f}, vy=0.000 m/s")
    print(f"[INFO] Policy sampling interval: {POLICY_DT:.3f} s")
    print(f"[INFO] MuJoCo GUI: {'ON' if args.gui else 'OFF'}")

    cfg = prepare_evaluation_cfg(num_envs=1, vx=args.vx, training_commands=False)
    cfg.seed = args.seed
    if not args.gui:
        cfg.commands["twist"].debug_vis = False

    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        if not math.isclose(env.step_dt, POLICY_DT, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Policy dt mismatch: env={env.step_dt}, cfg={POLICY_DT}")
        if (env.observation_manager.group_obs_dim.get("actor") != (44,)
                or env.action_manager.total_action_dim != 10):
            raise RuntimeError("Unexpected actor/action dimensions; verify rl/env_cfg.py")

        num_steps_exact = EPISODE_LENGTH_S / env.step_dt
        max_policy_steps = round(num_steps_exact)
        if not math.isclose(num_steps_exact, max_policy_steps, abs_tol=1e-10):
            raise RuntimeError("Episode duration must be a multiple of policy_dt")
        print(
            f"[INFO] Simulation horizon: {EPISODE_LENGTH_S:g} s "
            f"= {max_policy_steps} policy steps"
        )

        wrapped = RslRlVecEnvWrapper(env, clip_actions=ppo_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(
            wrapped, asdict(ppo_cfg), log_dir=None, device=args.device
        )
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()

        env.reset()
        plotter = Plotter(policy_dt=POLICY_DT, refresh_interval=args.refresh_interval)
        sampler = EvaluationSampler(env, plotter)
        sampler.record(sampler.read_command())  # Initial sample at t=0.

        if args.gui:
            plotter.start()
            plotter.refresh(force=True)
            print("[INFO] GUI: Figures 1/3 live; Figure 2 deferred")
            viewer = PlottingContactViewer(
                wrapped, policy, sampler, force_scale=args.force_scale
            )
            viewer.run(num_steps=max_policy_steps)
        else:
            print("[INFO] No GUI during evaluation; collecting data in RAM")
            run_without_gui(wrapped, policy, sampler, max_policy_steps)

        print(f"[INFO] Recorded {len(plotter.time_s)} samples in RAM")
        print("[INFO] Displaying three figures (no files saved)")
        plotter.show()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
