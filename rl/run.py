"""Run the trained PPO policy and plot body-frame X/Y velocities.

From the repository root:
    python -m rl.run --device cuda:0 --vx 0.08
    # Evaluation runs for 10 seconds of simulated time (500 policy steps).
    python -m rl.run --checkpoint logs/rsl_rl/open_duck_mini_v2/<run>/model_XXX.pt

This file reuses rl.play for PPO loading and the native MuJoCo contact viewer.
Only one environment is used for plotting. Data remains in RAM (no CSV/images).
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


class PlottingContactViewer(ContactViewer):
    """Extend the existing viewer with exactly one sample per policy step."""

    def __init__(self, env, policy, plotter: Plotter, *, force_scale: float = 1.0):
        super().__init__(env, policy, force_scale=force_scale)
        self.plotter = plotter
        self._break_after_manual_reset = False

    def reset_environment(self) -> None:
        """Keep the plot history while avoiding a line across a manual reset."""
        super().reset_environment()
        self._break_after_manual_reset = True

    def _execute_step(self) -> bool:
        """Sample after each env.step(), not after each rendering frame."""
        base_env = self.env.unwrapped

        # Capture the command USED for the upcoming step. The command manager
        # may resample it inside env.step() (particularly on auto-reset).
        command = base_env.command_manager.get_command("twist")
        if command is None:
            raise RuntimeError("Command 'twist' is missing")
        vx_desired, vy_desired = command[0, :2].detach().cpu().tolist()

        # Native viewer handles observations, deterministic policy inference,
        # physics, terminations, playback controls, and contact rendering.
        if not super()._execute_step():
            return False

        # MJLab auto-reset replaces the terminal state before returning from
        # env.step(). Do not plot the new episode's initial velocity as though
        # it were the previous episode's last measured velocity.
        auto_reset = base_env.episode_length_buf[0].item() == 0
        if auto_reset or self._break_after_manual_reset:
            vx_actual = math.nan
            vy_actual = math.nan
            self._break_after_manual_reset = False
        else:
            actual = base_env.scene["robot"].data.root_link_lin_vel_b
            vx_actual, vy_actual = actual[0, :2].detach().cpu().tolist()

        # This call occurs once per policy step: dt = POLICY_DT.
        self.plotter.add_sample(
            vx_desired=vx_desired,
            vx_actual=vx_actual,
            vy_desired=vy_desired,
            vy_actual=vy_actual,
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

    # Reuse the SAME inference environment settings as rl.play.
    cfg = prepare_evaluation_cfg(num_envs=1, vx=args.vx, training_commands=False)
    cfg.seed = args.seed
    # Keep training-like auto-reset if the robot falls before the 10s horizon.
    # run() still stops after exactly 500 total policy steps.

    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        if not math.isclose(env.step_dt, POLICY_DT, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"Policy dt mismatch: env={env.step_dt}, cfg={POLICY_DT}")
        if (env.observation_manager.group_obs_dim.get("actor") != (44,)
                or env.action_manager.total_action_dim != 10):
            raise RuntimeError("Unexpected actor/action dimensions; verify rl/env_cfg.py")

        # Derive the policy step cap from the environment episode duration.
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

        # Ensure initial command/state are valid and register t=0.
        env.reset()
        plotter = Plotter(
            policy_dt=POLICY_DT,
            refresh_interval=args.refresh_interval,
        )
        command = env.command_manager.get_command("twist")
        actual = env.scene["robot"].data.root_link_lin_vel_b
        if command is None:
            raise RuntimeError("Command 'twist' is missing after reset")
        desired_xy = command[0, :2].detach().cpu().tolist()
        actual_xy = actual[0, :2].detach().cpu().tolist()
        plotter.add_sample(
            vx_desired=desired_xy[0], vx_actual=actual_xy[0],
            vy_desired=desired_xy[1], vy_actual=actual_xy[1],
        )
        plotter.start()
        plotter.refresh(force=True)

        print("[OK] Policy loaded. Native MuJoCo viewer and velocity plots started")
        print("[INFO] MuJoCo: Space=pause; avoid Enter/reset during the 10s run")
        viewer = PlottingContactViewer(
            wrapped, policy, plotter, force_scale=args.force_scale
        )
        viewer.run(num_steps=max_policy_steps)

        # After 500 steps (10s simulated), the viewer closes automatically.
        # The final plotting window stays open for inspection.
        print(f"[INFO] Recorded {len(plotter.time_s)} samples in RAM")
        plotter.show()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
