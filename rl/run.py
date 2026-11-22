"""Run the trained PPO policy with X/Y velocity, base RPY, and body angular-rate plots.

Run from the repository root:
    python -m rl.run --device cuda:0 --vx 0.08
    python -m rl.run --checkpoint logs/rsl_rl/open_duck_mini_v2/<run>/model_XXX.pt

Evaluation lasts 10 seconds of simulated time (500 policy steps at 0.02 s).
This file reuses rl.play for PPO loading and native MuJoCo contact rendering.
Only one environment is plotted. No CSV or figure files are written.
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

from rl.env_cfg import POLICY_DT, EPISODE_LENGTH_S
from rl.play import ContactViewer, find_checkpoint, prepare_evaluation_cfg
from rl.plotter import Plotter
from rl.ppo_cfg import get_ppo_cfg


def quaternion_wxyz_to_rpy_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    """Convert root-link orientation relative to World into ZYX Euler angles [deg]."""
    w, x, y, z = quat_wxyz.detach().cpu().tolist()
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Invalid zero-norm base quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class PlottingContactViewer(ContactViewer):
    """Append one complete base-state sample per policy step, not per frame."""

    def __init__(self, env, policy, plotter: Plotter, *, force_scale: float = 1.0):
        super().__init__(env, policy, force_scale=force_scale)
        self.plotter = plotter
        self._break_after_manual_reset = False

    def reset_environment(self) -> None:
        """Keep plot history but prevent joining curves across a manual reset."""
        super().reset_environment()
        self._break_after_manual_reset = True

    def _execute_step(self) -> bool:
        """Sample after every env.step() with a fixed policy-step time base."""
        base_env = self.env.unwrapped

        # Preserve the command used by this step. It may be resampled on reset.
        command = base_env.command_manager.get_command("twist")
        if command is None:
            raise RuntimeError("Command 'twist' is missing")
        vx_desired, vy_desired = command[0, :2].detach().cpu().tolist()

        # BaseViewer executes observation -> policy -> action -> physics.
        if not super()._execute_step():
            return False

        # When auto-reset occurs, env.step() returns the NEW episode's
        # initial state. Insert NaNs instead of mislabeling it as the previous
        # episode's terminal state. This also handles a manual reset marker.
        auto_reset = base_env.episode_length_buf[0].item() == 0
        if auto_reset or self._break_after_manual_reset:
            vx_actual = vy_actual = math.nan
            roll_deg = pitch_deg = yaw_deg = math.nan
            wx_rad_s = wy_rad_s = wz_rad_s = math.nan
            self._break_after_manual_reset = False
        else:
            robot_data = base_env.scene["robot"].data
            vx_actual, vy_actual = (
                robot_data.root_link_lin_vel_b[0, :2].detach().cpu().tolist()
            )
            roll_deg, pitch_deg, yaw_deg = quaternion_wxyz_to_rpy_deg(
                robot_data.root_link_quat_w[0]
            )
            # Angular velocity expressed in the base/body frame [rad/s].
            # These are physical angular rates, not Euler-angle derivatives.
            wx_rad_s, wy_rad_s, wz_rad_s = (
                robot_data.root_link_ang_vel_b[0, :3].detach().cpu().tolist()
            )

        self.plotter.add_sample(
            vx_desired=vx_desired,
            vx_actual=vx_actual,
            vy_desired=vy_desired,
            vy_actual=vy_actual,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            wx_rad_s=wx_rad_s,
            wy_rad_s=wy_rad_s,
            wz_rad_s=wz_rad_s,
        )
        self.plotter.refresh()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vx", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-interval", type=float, default=0.10,
                        help="Matplotlib refresh interval in wall-clock seconds")
    parser.add_argument("--force-scale", type=float, default=1.0,
                        help="Viewer contact-force arrow scaling (visual only)")
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

    cfg = prepare_evaluation_cfg(num_envs=1, vx=args.vx, training_commands=False)
    cfg.seed = args.seed
    # Keep training-like auto-reset on falls before the 10-second horizon.

    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        if not math.isclose(env.step_dt, POLICY_DT, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Policy dt mismatch: env={env.step_dt}, cfg={POLICY_DT}")
        if (env.observation_manager.group_obs_dim.get("actor") != (44,)
                or env.action_manager.total_action_dim != 10):
            raise RuntimeError("Unexpected actor/action dimensions; verify rl/env_cfg.py")

        # Derive step limit from the episode duration in the env configuration.
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

        # Record initial state at t=0 before the 500 policy steps.
        env.reset()
        plotter = Plotter(
            policy_dt=POLICY_DT,
            refresh_interval=args.refresh_interval,
        )
        command = env.command_manager.get_command("twist")
        if command is None:
            raise RuntimeError("Command 'twist' is missing after reset")
        robot_data = env.scene["robot"].data
        vx_desired, vy_desired = command[0, :2].detach().cpu().tolist()
        vx_actual, vy_actual = (
            robot_data.root_link_lin_vel_b[0, :2].detach().cpu().tolist()
        )
        roll_deg, pitch_deg, yaw_deg = quaternion_wxyz_to_rpy_deg(
            robot_data.root_link_quat_w[0]
        )
        wx_rad_s, wy_rad_s, wz_rad_s = (
            robot_data.root_link_ang_vel_b[0, :3].detach().cpu().tolist()
        )
        plotter.add_sample(
            vx_desired=vx_desired, vx_actual=vx_actual,
            vy_desired=vy_desired, vy_actual=vy_actual,
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            wx_rad_s=wx_rad_s, wy_rad_s=wy_rad_s, wz_rad_s=wz_rad_s,
        )
        plotter.start()
        plotter.refresh(force=True)

        print("[OK] Policy loaded. MuJoCo viewer and 4 plots started")
        print("[INFO] MuJoCo: Space=pause; avoid Enter/reset during the 10s run")
        viewer = PlottingContactViewer(
            wrapped, policy, plotter, force_scale=args.force_scale
        )
        viewer.run(num_steps=max_policy_steps)

        # Keep final Matplotlib window open after the 10-second simulation.
        print(f"[INFO] Recorded {len(plotter.time_s)} samples in RAM")
        plotter.show()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
