"""Play a trained Open Duck Mini V2 PPO policy in MJLab/MuJoCo.

Run from the open_duck_mini_v2 repository root:

  python -m rl.play --device cuda:0
  python -m rl.play --device cuda:0 --vx 0.0
  python -m rl.play --device cuda:0 --vx 0.08
  python -m rl.play --checkpoint logs/rsl_rl/open_duck_mini_v2/<run>/model_XXX.pt

The script loads ONLY the trained actor (plus observation normalizer), does not
train PPO, and keeps episode reset behavior intact. It uses the native MuJoCo
viewer provided by MJLab; press Space to pause and Enter to reset.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from rl.env_cfg import get_env_cfg
from rl.ppo_cfg import get_ppo_cfg


def find_checkpoint(checkpoint: str, log_root: Path, experiment_name: str) -> Path:
    """Choose an explicit checkpoint or the newest checkpoint on disk."""
    if checkpoint != "latest":
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    search_dir = log_root.expanduser().resolve() / experiment_name
    candidates = list(search_dir.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model_*.pt checkpoints in {search_dir}. "
            "Use --checkpoint <path-to-model_XXX.pt>."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prepare_evaluation_cfg(num_envs: int, vx: float, training_commands: bool):
    env_cfg = get_env_cfg(num_envs=num_envs)

    # No curriculum progression while watching a frozen, trained policy.
    # Keep the reward readouts at their last training curriculum weights.
    for term in env_cfg.curriculum.values():
        reward_name = term.params.get("reward_name")
        stages = term.params.get("stages", [])
        if reward_name in env_cfg.rewards and stages:
            final_stage = max(stages, key=lambda stage: stage["step"])
            if "weight" in final_stage:
                env_cfg.rewards[reward_name].weight = final_stage["weight"]
    env_cfg.curriculum = {}

    # Use either the original training command distribution or a fixed
    # forward velocity for a reproducible standing/walking visual test.
    if not training_commands:
        twist = env_cfg.commands["twist"]
        twist.ranges.lin_vel_x = (vx, vx)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)
        twist.rel_standing_envs = 0.0
        twist.rel_heading_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.rel_world_envs = 0.0
        twist.heading_command = False
        twist.init_velocity_prob = 0.0

    return env_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest",
                        help="Checkpoint path or 'latest' (default).")
    parser.add_argument("--log-root", default="logs/rsl_rl",
                        help="Same log root used by rl.train.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vx", type=float, default=0.08,
                        help="Fixed forward command (m/s). 0 = stand, 0.08 = walk.")
    parser.add_argument("--training-commands", action="store_true",
                        help="Use original random training commands instead of fixed vx.")
    parser.add_argument("--viewer", choices=("native", "viser"), default="native",
                        help="Native MuJoCo window or browser-based Viser viewer.")
    parser.add_argument("--steps", type=int, default=0,
                        help="Policy steps to run; 0 means until viewer is closed.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_envs < 1 or args.steps < 0:
        parser.error("--num-envs must be >=1; --steps must be >=0.")
    if not args.training_commands and not (0.0 <= args.vx <= 0.1):
        parser.error("--vx must be between 0 and 0.1 m/s (training range).")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu or fix CUDA.")

    ppo_cfg = get_ppo_cfg()
    ckpt = find_checkpoint(args.checkpoint, Path(args.log_root), ppo_cfg.experiment_name)
    print(f"[INFO] Loading trained policy from: {ckpt}")
    print(f"[INFO] Evaluation device={args.device}, num_envs={args.num_envs}")
    if args.training_commands:
        print("[INFO] Commands: original training distribution")
    else:
        print(f"[INFO] Command: vx={args.vx:.3f} m/s, vy=0, yaw_rate=0")

    if args.device.startswith("cuda"):
        cuda_idx = torch.device(args.device).index or 0
        torch.cuda.set_device(cuda_idx)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(cuda_idx))
    configure_torch_backends()

    env_cfg = prepare_evaluation_cfg(args.num_envs, args.vx, args.training_commands)
    env_cfg.seed = args.seed
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
        actor_dim = env.observation_manager.group_obs_dim["actor"]
        num_actions = env.action_manager.total_action_dim
        if actor_dim != (44,) or num_actions != 10:
            raise RuntimeError(
                f"Unexpected actor/action dimensions: actor={actor_dim}, "
                f"actions={num_actions}; expected (44,), 10."
            )

        wrapped = RslRlVecEnvWrapper(env, clip_actions=ppo_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(
            wrapped, asdict(ppo_cfg), log_dir=None, device=args.device
        )
        runner.load(
            str(ckpt),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        policy.eval()
        print("[OK] Policy loaded. Inference is deterministic (no PPO updates).")
        print("[INFO] Viewer keys: Space=pause, Enter=reset, P=reward plots, "
              "R=toggle debug, +/-=speed.")

        viewer_cls = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
        viewer = viewer_cls(wrapped, policy)
        viewer.run(num_steps=None if args.steps == 0 else args.steps)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
