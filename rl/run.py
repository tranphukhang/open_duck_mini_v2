"""Evaluate Open Duck Mini V2 PPO and plot base, joint, contact and ZMP data.

Run from the repository root:
    python -m rl.run --device cuda:0 --vx 0.08 --gui
    python -m rl.run --device cuda:0 --vx 0.08 --no-gui
    python -m rl.run --checkpoint /path/to/model_XXX.pt --no-gui

Simulation lasts 10 s (500 policy steps at 0.02 s). ALL Matplotlib figures
are created only after the rollout, regardless of --gui/--no-gui.
When --gui is selected, the MuJoCo native viewer displays a transparent robot,
current and historical support-polygon outlines, and measured ZMP trajectory.
With --no-gui no viewer opens during evaluation. No data files are written.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from rl.cwc_reward import (
    HALF_LENGTH_X,
    HALF_WIDTH_Y,
    OFFSET_X,
    OFFSET_Y_LEFT_RIGHT,
)
from rl.env_cfg import EPISODE_LENGTH_S, POLICY_DT
from rl.play import ContactViewer, find_checkpoint, prepare_evaluation_cfg
from rl.plotter import Plotter
from rl.ppo_cfg import get_ppo_cfg
from rl.robot_cfg import FOOT_GEOM_NAMES, FOOT_SITE_NAMES, JOINT_NAMES


NORMAL_FORCE_THRESHOLD_N = 5.0


def quaternion_wxyz_to_rpy_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    """ZYX Euler roll/pitch/yaw of the robot base wrt World [deg]."""
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
    """Map JOINT_NAMES onto indices of robot.data.joint_pos once at startup."""
    name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}
    missing = [name for name in JOINT_NAMES if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing robot joints: {missing}")
    if len(JOINT_NAMES) != 10:
        raise RuntimeError("Expected five left and five right joints")
    return tuple(name_to_idx[name] for name in JOINT_NAMES)


def get_foot_wrench_indices(env) -> tuple[int, int]:
    """Map collision geometry names to feet_wrench primary sensor indices."""
    sensor = env.scene["feet_wrench"]
    if sensor.cfg.reduce != "netforce":
        raise RuntimeError("feet_wrench must use reduce='netforce'")
    names = sensor.primary_names
    ids: list[int] = []
    for geom_name in FOOT_GEOM_NAMES:
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


def get_foot_site_indices(env) -> tuple[int, int]:
    """Resolve L6_IK and R6_IK IDs in the MuJoCo compiled model."""
    model = env.sim.mj_model
    ids = []
    for site_name in FOOT_SITE_NAMES:
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{site_name}"
        )
        if site_id < 0:
            raise RuntimeError(f"Missing IK site: robot/{site_name}")
        ids.append(site_id)
    return ids[0], ids[1]


def convex_hull_xy(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Small 2D convex hull (monotonic chain), no SciPy dependency."""
    points = sorted(set(points))
    if len(points) < 3:
        return tuple(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


class EvaluationSampler:
    """Collect one synchronized sample of robot state and contact wrench per step."""

    def __init__(self, env: ManagerBasedRlEnv, plotter: Plotter):
        self.env = env
        self.plotter = plotter
        self.joint_indices = get_joint_indices(env.scene["robot"])
        self.foot_indices = get_foot_wrench_indices(env)
        self.foot_site_indices = get_foot_site_indices(env)

    def read_command(self) -> tuple[float, float]:
        """Read command in body frame before policy step/resampling."""
        command = self.env.command_manager.get_command("twist")
        if command is None:
            raise RuntimeError("Command 'twist' is missing")
        return tuple(command[0, :2].detach().cpu().tolist())

    def read_contact_support_zmp(
        self,
    ) -> tuple[
        tuple[int, int],
        tuple[tuple[float, float], ...],
        tuple[float, float],
    ]:
        """Read contact state, effective support polygon and net ZMP in World XY.

        Wrench convention matches rl/cwc_reward.py: feet_wrench reports
        foot-on-ground net force/torque in World, at sensor.data.pos.
        Negating both gives terrain-on-foot GRF (f) and contact moment (M).
        Transform each moment to World origin O: M_O = M_at_pos + pos cross f.
        For horizontal z=0: x_ZMP = -M_O,y/f_z; y_ZMP = M_O,x/f_z.

        Only contacts with found=True and normal GRF > 5 N contribute.
        Effective rectangles and offsets are imported from rl/cwc_reward.py.
        """
        wrench = self.env.scene["feet_wrench"].data
        if any(getattr(wrench, field) is None for field in (
            "force", "torque", "pos", "found"
        )):
            raise RuntimeError(
                "feet_wrench must provide force, torque, pos and found"
            )

        order = list(self.foot_indices)
        # (2,3) World-frame force, torque and wrench reference positions.
        force_w = -wrench.force[0, order, :]
        torque_w = -wrench.torque[0, order, :]
        pos_w = wrench.pos[0, order, :]
        found = wrench.found[0, order] > 0

        finite = (
            torch.isfinite(force_w).all(dim=-1)
            & torch.isfinite(torque_w).all(dim=-1)
            & torch.isfinite(pos_w).all(dim=-1)
        )
        active = found & (force_w[:, 2] > NORMAL_FORCE_THRESHOLD_N) & finite
        active_list = active.detach().cpu().tolist()
        contact_pair = (int(active_list[0]), int(active_list[1]))

        # Build current support polygon from the effective CWC rectangles.
        # The IK site's +X (not the collision geom's +X) defines foot length.
        vertices: list[tuple[float, float]] = []
        if any(active_list):
            site_ids = list(self.foot_site_indices)
            site_pos_xy = self.env.sim.data.site_xpos[
                0, site_ids, :2
            ].detach().cpu().tolist()
            site_rot = self.env.sim.data.site_xmat[
                0, site_ids
            ].reshape(2, 3, 3).detach().cpu().tolist()

            for foot_idx, is_active in enumerate(active_list):
                if not is_active:
                    continue
                # First column: world direction of IK site's local X axis.
                foot_x = site_rot[foot_idx]
                ex_x, ex_y = foot_x[0][0], foot_x[1][0]
                norm_xy = math.hypot(ex_x, ex_y)
                if norm_xy > 1.0e-6:
                    ex_x /= norm_xy
                    ex_y /= norm_xy
                else:
                    ex_x, ex_y = 1.0, 0.0
                ey_x, ey_y = -ex_y, ex_x

                center_x = (
                    site_pos_xy[foot_idx][0]
                    + OFFSET_X * ex_x
                    + OFFSET_Y_LEFT_RIGHT[foot_idx] * ey_x
                )
                center_y = (
                    site_pos_xy[foot_idx][1]
                    + OFFSET_X * ex_y
                    + OFFSET_Y_LEFT_RIGHT[foot_idx] * ey_y
                )
                for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                    vertices.append((
                        center_x + sx * HALF_LENGTH_X * ex_x
                                 + sy * HALF_WIDTH_Y * ey_x,
                        center_y + sx * HALF_LENGTH_X * ex_y
                                 + sy * HALF_WIDTH_Y * ey_y,
                    ))

        support_polygon = convex_hull_xy(vertices) if vertices else ()

        if not any(active_list):
            return contact_pair, support_polygon, (math.nan, math.nan)

        # Sum contact wrenches about WORLD ORIGIN, not about the foot sites.
        moment_about_world = torque_w + torch.cross(pos_w, force_w, dim=-1)
        zeros_f = torch.zeros_like(force_w)
        zeros_m = torch.zeros_like(moment_about_world)
        total_force = torch.where(active[:, None], force_w, zeros_f).sum(dim=0)
        total_moment = torch.where(
            active[:, None], moment_about_world, zeros_m
        ).sum(dim=0)

        fz = float(total_force[2].item())
        if not math.isfinite(fz) or fz <= NORMAL_FORCE_THRESHOLD_N:
            return contact_pair, support_polygon, (math.nan, math.nan)
        mx, my = total_moment[:2].detach().cpu().tolist()
        zmp_x, zmp_y = -my / fz, mx / fz
        if not (math.isfinite(zmp_x) and math.isfinite(zmp_y)):
            zmp_x = zmp_y = math.nan

        return contact_pair, support_polygon, (zmp_x, zmp_y)

    def record(self, command_xy: tuple[float, float], *, reset: bool = False) -> None:
        """Record a policy sample; break actual traces at reset boundaries."""
        if reset:
            vx_actual = vy_actual = math.nan
            roll_deg = pitch_deg = yaw_deg = math.nan
            wx_rad_s = wy_rad_s = wz_rad_s = math.nan
            left_joint_pos = [math.nan] * 5
            right_joint_pos = [math.nan] * 5
            contact_left = contact_right = math.nan
            polygon_xy: tuple[tuple[float, float], ...] = ()
            zmp_x = zmp_y = math.nan
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
            joints = data.joint_pos[
                0, list(self.joint_indices)
            ].detach().cpu().tolist()
            left_joint_pos, right_joint_pos = joints[:5], joints[5:]
            (contact_left, contact_right), polygon_xy, (zmp_x, zmp_y) = (
                self.read_contact_support_zmp()
            )

        self.plotter.add_sample(
            vx_desired=command_xy[0], vx_actual=vx_actual,
            vy_desired=command_xy[1], vy_actual=vy_actual,
            roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg,
            wx_rad_s=wx_rad_s, wy_rad_s=wy_rad_s, wz_rad_s=wz_rad_s,
            joint_pos_left=left_joint_pos, joint_pos_right=right_joint_pos,
            contact_left=contact_left, contact_right=contact_right,
            support_polygon_xy=polygon_xy, zmp_x_w=zmp_x, zmp_y_w=zmp_y,
        )


class PlottingContactViewer(ContactViewer):
    """Native viewer: transparent robot + 3D support/ZMP; no live Matplotlib.

    The user_scn decorations are refreshed in _render_other_env_geoms(), a
    hook called by MJLab AFTER it clears prior-frame debug geoms and BEFORE
    its final viewer.sync(). This avoids a second viewer.sync per frame.
    """

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

    def setup(self) -> None:
        """Enable MuJoCo's visual-only transparency on dynamic robot geoms."""
        super().setup()
        assert self.viewer is not None
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        # setup() of the parent has already synchronized once; the following
        # render frame propagates transparency without changing mjModel RGBA.
        print("[INFO] Robot transparency ON; 3D support boundary + ZMP ON")

    @staticmethod
    def _add_segment(scene, point_a, point_b, color, width: float) -> bool:
        """Add a decorative capsule segment in World coordinates."""
        if scene.ngeom >= scene.maxgeom:
            return False
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return False
        if np.linalg.norm(b - a) < 1e-8:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).ravel(),
            rgba=np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
            width=width,
            from_=a,
            to=b,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
        return True

    @staticmethod
    def _add_marker(scene, xy, *, z: float, radius: float, color) -> bool:
        """Add a decorative sphere at the measured ZMP."""
        if scene.ngeom >= scene.maxgeom:
            return False
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE.value,
            size=np.array([radius, radius, radius]),
            pos=np.array([xy[0], xy[1], z], dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.asarray(color, dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
        return True

    @classmethod
    def _add_polygon_outline(cls, scene, polygon, *, z, color, width):
        """Draw only the CLOSED EXTERIOR hull, never edges inside DSP."""
        if len(polygon) < 3:
            return
        for a, b in zip(polygon, (*polygon[1:], polygon[0])):
            if not cls._add_segment(
                scene, (a[0], a[1], z), (b[0], b[1], z), color, width
            ):
                return

    def _render_other_env_geoms(self, viewer, sim, sim_data) -> None:
        """Append overlay after the base MJLab debug visuals but before sync."""
        super()._render_other_env_geoms(viewer, sim, sim_data)
        if not viewer.is_running():
            return
        traces = self.sampler.plotter
        if not traces.time_s:
            return
        scene = viewer.user_scn

        # Last sample may be the automatic 10-second reset marker. Retain the
        # last actual support state rather than drawing the newly reset pose.
        last = len(traces.time_s) - 1
        while last >= 0 and (
            math.isnan(traces.contact_left[last])
            or math.isnan(traces.contact_right[last])
        ):
            last -= 1
        if last < 0:
            return

        # Past support boundaries: sparse samples (0.5s) AND contact mode
        # transitions. Each outline is independent, never joined across time.
        # Very low alpha/width avoids obscuring the current outline or robot.
        stride = max(1, round(0.50 / traces.policy_dt))
        previous_mode = None
        with viewer.lock():
            for k in range(last):
                polygon = traces.support_polygon_xy[k]
                if len(polygon) < 3:
                    previous_mode = None
                    continue
                mode = (traces.contact_left[k], traces.contact_right[k])
                mode_changed = mode != previous_mode
                previous_mode = mode
                if k % stride and not mode_changed:
                    continue
                self._add_polygon_outline(
                    scene, polygon, z=0.006, width=0.0007,
                    color=(0.15, 0.60, 0.95, 0.28),
                )

            # ZMP history: decimate the drawn trajectory only. All full-rate
            # samples remain in RAM for Figure 4. NaN resets break the line.
            prev = None
            trail_stride = max(1, round(0.06 / traces.policy_dt))
            for k in range(last + 1):
                x, y = traces.zmp_x_w[k], traces.zmp_y_w[k]
                if not (math.isfinite(x) and math.isfinite(y)):
                    prev = None
                    continue
                if k % trail_stride and k != last:
                    continue
                curr = (x, y, 0.010)
                if prev is not None:
                    self._add_segment(
                        scene, prev, curr,
                        color=(0.90, 0.12, 0.18, 0.85), width=0.001,
                    )
                prev = curr

            # CURRENT support hull is one bright, CLOSED outer boundary.
            # In DSP it is the convex hull of the 8 contact-patch vertices.
            self._add_polygon_outline(
                scene, traces.support_polygon_xy[last], z=0.004,
                color=(0.02, 0.85, 0.98, 1.0), width=0.004,
            )
            self._add_marker(
                scene, (traces.zmp_x_w[last], traces.zmp_y_w[last]),
                z=0.014, radius=0.004, color=(1.0, 0.05, 0.05, 1.0),
            )

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
        return True


def run_without_gui(
    wrapped: RslRlVecEnvWrapper,
    policy,
    sampler: EvaluationSampler,
    max_policy_steps: int,
) -> None:
    """Policy evaluation without MuJoCo Viewer or Matplotlib during rollout."""
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
        help="Open transparent MuJoCo Viewer with live 3D support/ZMP (default)",
    )
    gui_group.add_argument(
        "--no-gui", dest="gui", action="store_false",
        help="Skip MuJoCo Viewer; show all four figures after simulation",
    )
    parser.set_defaults(gui=True)
    parser.add_argument(
        "--refresh-interval", type=float, default=0.10,
        help="Reserved for Plotter compatibility; no live Matplotlib during rollout",
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
            print("[INFO] GUI: transparent robot, live 3D support polygon and ZMP")
            print("[INFO] Matplotlib: all 4 figures deferred until rollout ends")
            viewer = PlottingContactViewer(
                wrapped, policy, sampler, force_scale=args.force_scale
            )
            viewer.run(num_steps=max_policy_steps)
        else:
            print("[INFO] No GUI during evaluation; collecting data in RAM")
            run_without_gui(wrapped, policy, sampler, max_policy_steps)

        print(f"[INFO] Recorded {len(plotter.time_s)} samples in RAM")
        print("[INFO] Plotting all four figures now (no files saved)")
        plotter.show()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
