"""Contact Wrench Cone (CWC) soft-margin penalty for Open Duck Mini V2.

Assumptions agreed for the current flat-ground MJLab environment:
  * One net wrench per foot from scene['feet_wrench'], reduce='netforce'.
  * Sensor primary = foot, secondary = terrain. The tested world-frame
    contact wrench is sign-reversed to obtain terrain-on-foot GRF.
  * Each effective support patch is a 67 x 6 mm rectangle.
  * Contact frame {F_i}: origin at rectangle centre, +z world up,
    +x the horizontal projection of the foot COLLISION geom's local +x,
    +y = +z cross +x. Hence roll/pitch tilt is removed, yaw retained.
  * Both offsets are in these levelled axes relative to L6_IK/R6_IK:
    +11.5 mm along x, +7 mm y (left), -7 mm y (right).
  * Evaluate a foot only if its net normal GRF is > 5 N.
  * Caron's circumscribed Coulomb-friction pyramid approximation.
  * Return a NONNEGATIVE penalty. Set a NEGATIVE weight in RewardTermCfg.

No gait phase/contact reference is used. Code is fully batched for MJLab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rl.robot_cfg import FOOT_GEOM_NAMES, FOOT_SITE_NAMES

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Half dimensions of the assumed, effective support rectangle [m].
HALF_LENGTH_X = 0.0335
HALF_WIDTH_Y = 0.003

# Rectangle centre relative to the foot IK site's *horizontal position* [m].
OFFSET_X = 0.0115
OFFSET_Y_LEFT_RIGHT = (0.007, -0.007)


@dataclass(frozen=True)
class CWCValues:
    """Batched diagnostic values; all per-foot tensors have shape (B,2)."""

    penalty: torch.Tensor       # (B,) mean penalty over active feet
    active: torch.Tensor        # (B,2) bool, GRF_z > 5 N and found contact
    friction_margin: torch.Tensor
    cop_margin: torch.Tensor
    yaw_margin: torch.Tensor
    friction_penalty: torch.Tensor
    cop_penalty: torch.Tensor
    yaw_penalty: torch.Tensor


@dataclass(frozen=True)
class _Ids:
    model_id: int
    sensor_name: str
    foot_indices: tuple[int, int]
    site_ids: tuple[int, int]
    geom_ids: tuple[int, int]


def _resolve_ids(env: ManagerBasedRlEnv, sensor_name: str) -> _Ids:
    """Resolve static MJCF IDs only once per environment and compiled model."""
    model = env.sim.mj_model
    cached = getattr(env, "_open_duck_cwc_ids", None)
    if isinstance(cached, _Ids) and cached.model_id == id(model) and cached.sensor_name == sensor_name:
        return cached

    import mujoco

    sensor = env.scene[sensor_name]
    names = sensor.primary_names
    foot_indices = []
    site_ids = []
    geom_ids = []
    for geom_name, site_name in zip(FOOT_GEOM_NAMES, FOOT_SITE_NAMES, strict=True):
        matches = [
            idx for idx, name in enumerate(names)
            if name == geom_name or name.endswith("/" + geom_name)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one sensor primary for {geom_name}, got {names}"
            )
        foot_indices.append(matches[0])
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{site_name}")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{geom_name}")
        if sid < 0 or gid < 0:
            raise RuntimeError(f"Missing robot site or collision geom: {site_name}, {geom_name}")
        site_ids.append(sid)
        geom_ids.append(gid)

    result = _Ids(
        model_id=id(model),
        sensor_name=sensor_name,
        foot_indices=(foot_indices[0], foot_indices[1]),
        site_ids=(site_ids[0], site_ids[1]),
        geom_ids=(geom_ids[0], geom_ids[1]),
    )
    setattr(env, "_open_duck_cwc_ids", result)
    return result


def net_wrench_to_contact_frame(
    sensor_force_w: torch.Tensor,
    sensor_torque_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    ik_site_pos_w: torch.Tensor,
    foot_geom_xmat_w: torch.Tensor,
    *,
    ground_height: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MJLab netforce to GRF wrench at rectangle centre in frame {F}.

    Each input except foot_geom_xmat_w is (B,2,3), and the geom orientation
    is (B,2,3,3) (or (B,2,9)), world from geom. Outputs are (B,2,3).

    Use collision geom x-axis rather than IK site orientation: in the current
    MJCF, L6_IK/R6_IK sites have calibration Euler rotations. The collision
    geoms use the foot-link axes, with +x along the longitudinal direction.
    """
    if sensor_force_w.ndim != 3 or sensor_force_w.shape[-2:] != (2, 3):
        raise ValueError("sensor_force_w must be (B,2,3)")
    for name, x in (
        ("sensor_torque_w", sensor_torque_w),
        ("sensor_pos_w", sensor_pos_w),
        ("ik_site_pos_w", ik_site_pos_w),
    ):
        if x.shape != sensor_force_w.shape:
            raise ValueError(f"{name} must match sensor_force_w shape")
    b = sensor_force_w.shape[0]
    rot = foot_geom_xmat_w.reshape(b, 2, 3, 3)

    # Rotate away the foot's roll and pitch while retaining its heading.
    # R[:, :, :, 0] = local +x in world coordinates.
    foot_x = rot[..., :, 0]
    horizontal = torch.stack(
        (foot_x[..., 0], foot_x[..., 1], torch.zeros_like(foot_x[..., 0])), dim=-1
    )
    length = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    world_x = torch.zeros_like(horizontal)
    world_x[..., 0] = 1.0
    ex_w = torch.where(length > 1.0e-6, horizontal / length.clamp_min(1.0e-6), world_x)
    ey_w = torch.stack((-ex_w[..., 1], ex_w[..., 0], torch.zeros_like(ex_w[..., 0])), dim=-1)
    ez_w = torch.zeros_like(ex_w)
    ez_w[..., 2] = 1.0

    side_y = sensor_force_w.new_tensor(OFFSET_Y_LEFT_RIGHT).view(1, 2, 1)
    origin_w = ik_site_pos_w + OFFSET_X * ex_w + side_y * ey_w
    # The contact origin must be on the flat ground plane, NOT at site z.
    origin_w = torch.stack(
        (origin_w[..., 0], origin_w[..., 1],
         torch.full_like(origin_w[..., 2], ground_height)), dim=-1
    )

    # The sign was checked in the user's standing test: primary=foot,
    # secondary=terrain. Both force and torque must change sign together.
    force_w = -sensor_force_w
    torque_about_sensor_pos_w = -sensor_torque_w
    torque_about_origin_w = (
        torque_about_sensor_pos_w
        + torch.cross(sensor_pos_w - origin_w, force_w, dim=-1)
    )

    def components(vector_w: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                (vector_w * ex_w).sum(dim=-1),
                (vector_w * ey_w).sum(dim=-1),
                (vector_w * ez_w).sum(dim=-1),
            ), dim=-1
        )

    return components(force_w), components(torque_about_origin_w)


def compute_cwc_values(
    force_f: torch.Tensor,
    moment_f: torch.Tensor,
    found: torch.Tensor,
    *,
    mu: float = 0.8,
    min_normal_force: float = 5.0,
    safe_margin: float = 0.2,
    weight_friction: float = 1.0,
    weight_cop: float = 1.0,
    weight_yaw: float = 1.0,
) -> CWCValues:
    """Pure batched CWC evaluation; inputs in contact frame {F_i}.

    force_f, moment_f: (B,2,3); found: (B,2).
    The return value is a nonnegative mean penalty (B,).
    """
    if force_f.ndim != 3 or force_f.shape[-2:] != (2, 3) or moment_f.shape != force_f.shape:
        raise ValueError("force_f and moment_f must both be (B,2,3)")
    if found.shape != force_f.shape[:2]:
        raise ValueError("found must have shape (B,2)")
    if mu <= 0 or min_normal_force < 0 or not 0 <= safe_margin <= 1:
        raise ValueError("Require mu>0, min_normal_force>=0, safe_margin in [0,1]")
    if min(weight_friction, weight_cop, weight_yaw) < 0:
        raise ValueError("Internal penalty weights must be nonnegative")

    fx, fy, fz = force_f.unbind(dim=-1)
    mx, my, mz = moment_f.unbind(dim=-1)
    finite = torch.isfinite(force_f).all(dim=-1) & torch.isfinite(moment_f).all(dim=-1)
    active = (found > 0) & (fz > min_normal_force) & finite
    # Safe denominator even for swing/unloaded feet (they will be masked out).
    fz_safe = fz.clamp_min(max(min_normal_force, 1.0e-6))

    # Circumscribed friction pyramid: |fx|,|fy| <= mu*fz.
    margin_friction = 1.0 - torch.maximum(fx.abs(), fy.abs()) / (mu * fz_safe)

    # Foot CoP inside [-lx,lx] x [-ly,ly] at the *contact* origin.
    margin_cop_x = 1.0 - my.abs() / (HALF_LENGTH_X * fz_safe)
    margin_cop_y = 1.0 - mx.abs() / (HALF_WIDTH_Y * fz_safe)
    margin_cop = torch.minimum(margin_cop_x, margin_cop_y)

    # Caron et al. (2015), circumscribed-pyramid yaw inequalities.
    reference_yaw = mu * (HALF_LENGTH_X + HALF_WIDTH_Y) * fz_safe
    yaw_lower = (
        -reference_yaw
        + (HALF_WIDTH_Y * fx - mu * mx).abs()
        + (HALF_LENGTH_X * fy - mu * my).abs()
    )
    yaw_upper = (
        +reference_yaw
        - (HALF_WIDTH_Y * fx + mu * mx).abs()
        - (HALF_LENGTH_X * fy + mu * my).abs()
    )
    margin_yaw = torch.minimum(mz - yaw_lower, yaw_upper - mz) / reference_yaw

    pen_f = torch.relu(safe_margin - margin_friction).square()
    pen_c = torch.relu(safe_margin - margin_cop).square()
    pen_y = torch.relu(safe_margin - margin_yaw).square()
    pen_foot = weight_friction * pen_f + weight_cop * pen_c + weight_yaw * pen_y

    # Avoid propagating NaN/Inf from inactive bad-physics frames.
    pen_foot = torch.where(active, torch.nan_to_num(pen_foot, nan=0.0, posinf=0.0),
                           torch.zeros_like(pen_foot))
    num_active = active.sum(dim=1).clamp_min(1)
    penalty = pen_foot.sum(dim=1) / num_active
    return CWCValues(
        penalty=penalty,
        active=active,
        friction_margin=torch.where(active, margin_friction, torch.zeros_like(margin_friction)),
        cop_margin=torch.where(active, margin_cop, torch.zeros_like(margin_cop)),
        yaw_margin=torch.where(active, margin_yaw, torch.zeros_like(margin_yaw)),
        friction_penalty=torch.where(active, pen_f, torch.zeros_like(pen_f)),
        cop_penalty=torch.where(active, pen_c, torch.zeros_like(pen_c)),
        yaw_penalty=torch.where(active, pen_y, torch.zeros_like(pen_y)),
    )


def cwc_diagnostics(
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str = "feet_wrench",
    mu: float = 0.8,
    min_normal_force: float = 5.0,
    safe_margin: float = 0.2,
    weight_friction: float = 1.0,
    weight_cop: float = 1.0,
    weight_yaw: float = 1.0,
) -> CWCValues:
    """Read sensor, transform to {F}, compute CWC values; useful for logging."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    if sensor.cfg.reduce != "netforce":
        raise RuntimeError("CWC requires feet_wrench with reduce='netforce'")
    if any(getattr(data, key) is None for key in ("force", "torque", "pos", "found")):
        raise RuntimeError("CWC sensor must provide found, force, torque, pos")

    ids = _resolve_ids(env, sensor_name)
    order = list(ids.foot_indices)
    force_sensor_w = data.force[:, order, :]
    torque_sensor_w = data.torque[:, order, :]
    pos_sensor_w = data.pos[:, order, :]
    found = data.found[:, order]

    # Batched MJLab world-frame position/rotation arrays (GPU compatible).
    site_pos_w = env.sim.data.site_xpos[:, list(ids.site_ids), :]
    foot_rot_w = env.sim.data.geom_xmat[:, list(ids.geom_ids)]
    force_f, moment_f = net_wrench_to_contact_frame(
        force_sensor_w, torque_sensor_w, pos_sensor_w, site_pos_w, foot_rot_w
    )
    return compute_cwc_values(
        force_f, moment_f, found,
        mu=mu, min_normal_force=min_normal_force, safe_margin=safe_margin,
        weight_friction=weight_friction, weight_cop=weight_cop, weight_yaw=weight_yaw,
    )


def cwc_penalty(
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str = "feet_wrench",
    mu: float = 0.8,
    min_normal_force: float = 5.0,
    safe_margin: float = 0.2,
    weight_friction: float = 1.0,
    weight_cop: float = 1.0,
    weight_yaw: float = 1.0,
) -> torch.Tensor:
    """Reward-manager term: return nonnegative CWC penalty, shape (num_envs,)."""
    return cwc_diagnostics(
        env, sensor_name=sensor_name, mu=mu,
        min_normal_force=min_normal_force, safe_margin=safe_margin,
        weight_friction=weight_friction, weight_cop=weight_cop,
        weight_yaw=weight_yaw,
    ).penalty
