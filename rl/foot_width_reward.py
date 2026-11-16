"""Lateral foot-width penalty for Open Duck Mini V2 (MJLab).

Matches the intent of capture_point/CapturePoint.correct_landing:
keep left foot on the robot-left side and right foot on robot-right side,
with at least min_width of lateral separation. Unlike the CP planner's hard
landing constraint, this is a continuous soft reward penalty.
"""

from __future__ import annotations

import torch


def signed_lateral_foot_width(
    foot_sites_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Signed left-minus-right foot separation along body-yaw lateral axis.

    Args:
        foot_sites_pos_w: (N, 2, 3), in (L6_IK, R6_IK) order, world {W}.
        root_quat_w: (N, 4), scalar-first (w, x, y, z), world orientation.

    Returns:
        (N,) lateral width [m]. Positive means left foot is left of right.
    """
    if foot_sites_pos_w.ndim != 3 or foot_sites_pos_w.shape[1:] != (2, 3):
        raise ValueError("foot_sites_pos_w must have shape (N, 2, 3)")
    if root_quat_w.shape != (foot_sites_pos_w.shape[0], 4):
        raise ValueError("root_quat_w must have shape (N, 4)")

    w, x, y, z = root_quat_w.unbind(dim=-1)
    # Yaw direction in the world plane (quaternion uses wxyz convention).
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    normalizer = torch.sqrt(sin_yaw.square() + cos_yaw.square()).clamp_min(1e-6)

    diff_w = foot_sites_pos_w[:, 0, :2] - foot_sites_pos_w[:, 1, :2]
    # Body's horizontal +y axis is [-sin(yaw), +cos(yaw)] in {W}.
    return (-sin_yaw * diff_w[:, 0] + cos_yaw * diff_w[:, 1]) / normalizer


def normalized_foot_width_penalty(
    width: torch.Tensor,
    min_width: float = 0.12,
) -> torch.Tensor:
    """Zero at/above min_width; quadratically penalize smaller/crossed feet."""
    if min_width <= 0:
        raise ValueError("min_width must be > 0")
    return ((min_width - width).clamp_min(0.0) / min_width).square()


class foot_width_penalty:
    """MJLab class-based reward term; caches the two MJCF site IDs."""

    def __init__(self, cfg, env) -> None:
        del cfg
        import mujoco

        from rl.robot_cfg import FOOT_SITE_NAMES

        if len(FOOT_SITE_NAMES) != 2:
            raise ValueError("FOOT_SITE_NAMES must be ordered as (left, right)")

        model = env.sim.mj_model
        site_ids = []
        for site_name in FOOT_SITE_NAMES:
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{site_name}"
            )
            if site_id < 0:
                raise RuntimeError(f"Foot site not found: robot/{site_name}")
            site_ids.append(site_id)
        self.site_ids = tuple(site_ids)

    def lateral_width(self, env) -> torch.Tensor:
        sites_w = env.sim.data.site_xpos[:, list(self.site_ids), :]
        quat_w = env.scene["robot"].data.root_link_quat_w
        return signed_lateral_foot_width(sites_w, quat_w)

    def __call__(self, env, min_width: float = 0.12) -> torch.Tensor:
        return normalized_foot_width_penalty(self.lateral_width(env), min_width)
