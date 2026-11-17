"""Play an Open Duck Mini V2 PPO policy with MuJoCo contact visualization.

Run from the repository root after copying this file to rl/play.py:

  python -m rl.play --device cuda:0 --vx 0.08
  python -m rl.play --device cuda:0 --vx 0.08 --force-scale 2
  python -m rl.play --device cuda:0 --vx 0 --force-labels
  python -m rl.play --checkpoint /path/to/model_XXX.pt --vx 0.08

In the native viewer: Space = pause, Enter = reset, P = reward plots.
Contact points and contact-force arrows are enabled by default.

Note: MJLab simulates contacts on GPU via MuJoCo Warp. The native viewer
reconstructs contacts with CPU MuJoCo for visualization. Displayed forces may
therefore differ slightly from GPU sensor values used in the RL reward.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

from rl.env_cfg import get_env_cfg
from rl.ppo_cfg import get_ppo_cfg


class ContactViewer(NativeMujocoViewer):
    """Native MJLab viewer showing contact positions and contact-force arrows."""

    def __init__(
        self,
        env,
        policy,
        *,
        force_scale: float = 1.0,
        split_forces: bool = False,
        force_labels: bool = False,
    ) -> None:
        super().__init__(env, policy)
        self.force_scale = force_scale
        self.split_forces = split_forces
        self.force_labels = force_labels

    def setup(self) -> None:
        # MJLab's native viewer already synchronizes Warp simulation state to
        # CPU MjData and recomputes contacts through mj_forward every frame.
        super().setup()
        assert self.viewer is not None
        assert self.mjm is not None

        with self.viewer.lock():
            flags = self.viewer.opt.flags
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = int(self.split_forces)

            # Visualization parameters only; do not modify simulated forces.
            self.mjm.vis.map.force *= self.force_scale
            if self.force_labels:
                self.viewer.opt.label = mujoco.mjtLabel.mjLABEL_CONTACTFORCE

        self.viewer.sync(state_only=False)
        print(
            "[INFO] Contact points: ON | Contact force arrows: ON | "
            f"scale={self.force_scale:g} | "
            f"split={self.split_forces} | labels={self.force_labels}"
        )


def find_checkpoint(checkpoint: str, log_root: Path, experiment_name: str) -> Path:
    """Use an explicit checkpoint or newest modified model_*.pt."""
    if checkpoint != "latest":
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    search_dir = log_root.expanduser().resolve() / experiment_name
    candidates = list(search_dir.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model_*.pt found under {search_dir}. "
            "Use --checkpoint /path/to/model_XXX.pt."
        )
    return max(candidates, key=lambda file: file.stat().st_mtime)


def prepare_evaluation_cfg(num_envs: int, vx: float, training_commands: bool):
    cfg = get_env_cfg(num_envs=num_envs)

    # Freeze curriculum during inference; retain last-stage reward weights for
    # informational reward plots. Neither action nor policy is affected.
    for term in cfg.curriculum.values():
        reward_name = term.params.get("reward_name")
        stages = term.params.get("stages", ())
        if reward_name in cfg.rewards and stages:
            last_stage = max(stages, key=lambda stage: stage["step"])
            cfg.rewards[reward_name].weight = last_stage["weight"]
    cfg.curriculum = {}

    if not training_commands:
        twist = cfg.commands["twist"]
        twist.ranges.lin_vel_x = (vx, vx)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)
        twist.rel_standing_envs = 0.0
        twist.rel_heading_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.rel_world_envs = 0.0
        twist.heading_command = False
        twist.init_velocity_prob = 0.0

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vx", type=float, default=0.08)
    parser.add_argument("--training-commands", action="store_true")
    parser.add_argument("--steps", type=int, default=0,
                        help="Number of policy steps; 0 = run until viewer closes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-scale", type=float, default=1.0,
                        help="Arrow length multiplier (visual only)")
    parser.add_argument("--split-forces", action="store_true",
                        help="Show normal and tangential forces separately")
    parser.add_argument("--force-labels", action="store_true",
                        help="Display contact-force magnitudes in the native viewer")
    args = parser.parse_args()

    if args.num_envs < 1 or args.steps < 0 or args.force_scale <= 0:
        parser.error("Require --num-envs >= 1, --steps >= 0, --force-scale > 0")
    if not args.training_commands and not (0.0 <= args.vx <= 0.1):
        parser.error("--vx must be in [0.0, 0.1] m/s (training command range)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA not available; use --device cpu or check installation")

    ppo_cfg = get_ppo_cfg()
    checkpoint = find_checkpoint(
        args.checkpoint, Path(args.log_root), ppo_cfg.experiment_name
    )
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Device: {args.device}; num_envs: {args.num_envs}")
    if args.training_commands:
        print("[INFO] Using original training velocity command distribution")
    else:
        print(f"[INFO] Fixed command: vx={args.vx:.3f} m/s, vy=0, wz=0")

    if args.device.startswith("cuda"):
        cuda_id = torch.device(args.device).index or 0
        torch.cuda.set_device(cuda_id)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(cuda_id))
    configure_torch_backends()

    cfg = prepare_evaluation_cfg(args.num_envs, args.vx, args.training_commands)
    cfg.seed = args.seed

    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        actor_dim = env.observation_manager.group_obs_dim["actor"]
        action_dim = env.action_manager.total_action_dim
        if actor_dim != (44,) or action_dim != 10:
            raise RuntimeError(
                f"Expected actor (44,) and action 10, got {actor_dim}, {action_dim}"
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

        print("[OK] Deterministic PPO policy loaded; no learning updates")
        print("[INFO] MuJoCo keys: Space=pause, Enter=reset, P=reward plot, R=debug")
        viewer = ContactViewer(
            wrapped,
            policy,
            force_scale=args.force_scale,
            split_forces=args.split_forces,
            force_labels=args.force_labels,
        )
        viewer.run(num_steps=None if args.steps == 0 else args.steps)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
