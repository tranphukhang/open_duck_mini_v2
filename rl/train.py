"""Train Open Duck Mini V2 locomotion with MJLab + RSL-RL PPO.

Run at the repository root (with your existing rl/env_cfg.py and MJLab installed):

    # First validate the pipeline with a small environment count:
    python -m rl.train --device cuda:0 --num-envs 4 --max-iterations 2 --hours 0

    # Main 90-minute run, 1024 parallel environments:
    python -m rl.train --device cuda:0 --num-envs 1024 --hours 1.5

    # Resume a saved checkpoint for an additional hour:
    python -m rl.train --device cuda:0 --num-envs 1024 --hours 1 \\
        --resume logs/rsl_rl/open_duck_mini_v2/<run>/model_XXX.pt
    
    python -m rl.train \
    --device cuda:0 \
    --num-envs 1024 \
    --hours 0 \
    --max-iterations 3000

The wall-clock limit is checked BETWEEN complete PPO iterations. This script
reuses MJLab's PPO, runner, logging, TensorBoard and checkpoint formats.
No initial state/episode-length randomization is applied.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.utils import check_nan

from rl.env_cfg import get_env_cfg
from rl.ppo_cfg import get_ppo_cfg


class TimedMjlabOnPolicyRunner(MjlabOnPolicyRunner):
    """RSL-RL 5.5.2 training loop with a safe wall-clock stop at iteration end.

    The PPO implementation, logger, model and checkpoint code remain from
    RSL-RL / MJLab. Only an optional time-limit check is introduced here.
    """

    def learn(
        self,
        num_learning_iterations: int,
        *,
        max_seconds: float | None = None,
    ) -> None:
        if num_learning_iterations <= 0:
            return
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive or None")

        # Do NOT use init_at_random_ep_len: user requested deterministic reset.
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        t_start = time.monotonic()
        last_saved_path = None

        try:
            for it in range(start_it, total_it):
                # No check inside rollout or PPO update; leave each iteration intact.
                if max_seconds is not None and time.monotonic() - t_start >= max_seconds:
                    break

                start = time.time()
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(
                            actions.to(self.env.device)
                        )
                        if self.cfg.get("check_for_nan", True):
                            check_nan(obs, rewards, dones)
                        obs = obs.to(self.device)
                        rewards = rewards.to(self.device)
                        dones = dones.to(self.device)
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        self.logger.process_env_step(
                            rewards, dones, extras,
                            self.alg.intrinsic_rewards
                            if self.cfg["algorithm"].get("rnd_cfg") else None,
                        )

                    collect_time = time.time() - start
                    start_update = time.time()
                    self.alg.compute_returns(obs)

                loss_dict = self.alg.update()
                learn_time = time.time() - start_update
                self.current_learning_iteration = it

                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collect_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=(self.alg.rnd.weight
                                if self.cfg["algorithm"].get("rnd_cfg") else None),
                )

                if (self.logger.writer is not None
                        and it % self.cfg["save_interval"] == 0):
                    last_saved_path = Path(self.logger.log_dir) / f"model_{it}.pt"
                    self.save(str(last_saved_path))

            # Save at the end even when the wall-clock budget has expired.
            if self.logger.writer is not None:
                final_path = Path(self.logger.log_dir) / (
                    f"model_{self.current_learning_iteration}.pt"
                )
                if final_path != last_saved_path or not final_path.is_file():
                    self.save(str(final_path))
                print(f"[INFO] Final checkpoint: {final_path}")

        except BaseException:
            self.logger.abort_logging_writer()
            raise
        else:
            self.logger.stop_logging_writer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (cuda:0) or cpu")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--hours", type=float, default=1.5,
                        help="Training time in hours. 0 disables the time limit.")
    parser.add_argument("--max-iterations", type=int, default=30_000,
                        help="Total iteration cap (including resumed iterations).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a previous model_*.pt checkpoint")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.max_iterations <= 0 or args.hours < 0:
        parser.error("--num-envs and --max-iterations must be >0; --hours must be >=0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --device cpu or fix CUDA installation.")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Checkpoint not found: {args.resume}")
    return args


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        # Keeping device index explicit also helps single-GPU Warp setups.
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(torch.device(args.device).index or 0))
    configure_torch_backends()

    env_cfg = get_env_cfg(num_envs=args.num_envs)
    agent_cfg = get_ppo_cfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = args.seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False

    log_dir = (
        Path(args.log_root).expanduser().resolve()
        / agent_cfg.experiment_name
        / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + agent_cfg.run_name)
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={args.device}, environments={args.num_envs}")
    print(f"[INFO] policy_dt={env_cfg.sim.mujoco.timestep * env_cfg.decimation:.4f} s")
    print(f"[INFO] PPO rollout={agent_cfg.num_steps_per_env}, minibatches={agent_cfg.algorithm.num_mini_batches}")
    print(f"[INFO] Time limit={args.hours} h (0 = no limit)")
    print(f"[INFO] Logs/checkpoints: {log_dir}")

    env = None
    runner = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
        if (env.observation_manager.group_obs_dim.get("actor") != (44,)
                or env.observation_manager.group_obs_dim.get("critic") != (50,)
                or env.action_manager.total_action_dim != 10):
            raise RuntimeError(
                "Unexpected observation/action dimensions. Expected actor 44, "
                "critic 50, action 10; check rl/env_cfg.py."
            )
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = TimedMjlabOnPolicyRunner(
            wrapped, asdict(agent_cfg), str(log_dir), args.device
        )
        runner.add_git_repo_to_log(__file__)

        if args.resume is not None:
            runner.load(str(args.resume.resolve()), map_location=args.device)
            print(f"[INFO] Resumed from {args.resume}")
            print(f"[INFO] Restored common_step_counter={env.common_step_counter}")
            # Immediately restore curriculum reward weights to the loaded step.
            # Without this, stage-0 weights would persist until an env resets.
            env.curriculum_manager.compute()
            for reward_name in ("action_rate_l2", "joint_torques_l2"):
                weight = env.reward_manager.get_term_cfg(reward_name).weight
                print(f"[INFO] Restored curriculum: {reward_name}={weight}")

        remaining = args.max_iterations - runner.current_learning_iteration
        if remaining <= 0:
            print("[INFO] No iterations remain under --max-iterations.")
            return
        max_seconds = args.hours * 3600 if args.hours > 0 else None
        print(f"[INFO] PPO start iteration={runner.current_learning_iteration}; "
              f"remaining cap={remaining}")
        runner.learn(remaining, max_seconds=max_seconds)

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by Ctrl+C.")
        if runner is not None:
            interrupted = log_dir / f"model_interrupted_{runner.current_learning_iteration}.pt"
            try:
                runner.save(str(interrupted))
                print(f"[INFO] Saved interrupt checkpoint: {interrupted}")
            except Exception as save_exc:
                print(f"[ERROR] Could not save interrupt checkpoint: {save_exc}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
