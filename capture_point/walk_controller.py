"""Bo dieu phoi gait controller cho Open Duck Mini V2, 50 Hz.

Pipeline: MuJoCo GT -> Scheduler -> Footstep/reanchor/CP -> ZMP/LIPM-MPC
          -> Swing -> Whole-Body IK -> (optional) Load Compensation.

Chay trong thu muc capture_point/ cua repo tranphukhang/open_duck_mini_v2.
Khong goi mujoco.mj_step() trong controller; vong lap mo phong thuc hien
nhieu buoc vat ly giua hai lenh 50 Hz.

Tai reset() chi doc goc base va goc 10 khop *mot lan* de khoi tao
nghiem WBIK/home; phep do CoM/chan trong moi tick chi lay Ground Truth
qua GroundTruthStateEstimator (KHONG doc joint state/khong FK measurement).

Moi toa do tham chieu nam trong {P}, voi cac truc {P}//{W} cua MuJoCo.
Reanchor_x dich toa do *ke hoach*, CoM measured duoc quy chieu lai
qua chan ho tro, nhu walker.py cua anhoang-21/open_duck_isaac.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner
from capture_point import CapturePoint
from zmp_generator import ZMPGenerator
from lipm_mpc import LIPMMPC
from swing_trajectory import SwingTrajectoryGenerator
from robot_kinematics import RobotKinematics
from whole_body_IK import WholeBodyIK
from load_compensation import LoadCompensation, LoadCompensationParams
from state_estimate import GroundTruthStateEstimator


JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)


@dataclass
class WalkConfig:
    dt: float = 0.02
    period_phase: float = 0.4
    shift: float = 0.4
    double_support: float = 0.1
    swing_height: float = 0.02
    horizon: int = 80
    num_footsteps: int = 14

    # CP and ZMP preview MPC (author's walker-fk-com defaults).
    step_gain: tuple[float, float] = (0.4, 0.3)
    step_limit: tuple[float, float] = (0.04, 0.03)
    min_step_width: float = 0.12
    zmp_offset_x: float = 0.005
    zmp_offset_y_out: float = 0.007
    q_zmp: float = 1.0
    q_vel: float = 1e-3
    r_jerk: float = 1e-6
    mpc_fb_pos: tuple[float, float] = (0.0, 0.0)
    mpc_fb_vel: tuple[float, float] = (0.0, 0.2)
    reanchor_x: bool = True

    ik_initial_iterations: int = 8
    ik_warm_iterations: int = 1

    # Co the so sanh 2 nhom mo phong chi bang cong tac nay.
    enable_load_compensation: bool = True
    load_gain: float = 1.0        # alpha: 0..1
    servo_kp: float = 13.37       # N*m/rad, MJCF v5
    cop_moment: bool = True
    gravity: float = 9.81

    def __post_init__(self):
        if self.dt <= 0 or self.period_phase <= 0 or self.shift <= 0:
            raise ValueError("dt, period_phase and shift must be > 0")
        if not (0 < self.double_support < 0.5):
            raise ValueError("double_support must be in (0, 0.5)")
        if self.horizon < 2 or self.num_footsteps < 4:
            raise ValueError("horizon >= 2 and num_footsteps >= 4 are required")
        if self.ik_initial_iterations < 1 or self.ik_warm_iterations < 1:
            raise ValueError("IK iteration counts must be >= 1")
        if not (0.0 <= self.load_gain <= 1.0):
            raise ValueError("load_gain must lie in [0, 1]")
        if self.servo_kp <= 0 or self.gravity <= 0:
            raise ValueError("servo_kp and gravity must be positive")
        if len(self.mpc_fb_pos) != 2 or len(self.mpc_fb_vel) != 2:
            raise ValueError("MPC feedback vectors must each have 2 elements")


@dataclass
class WalkOutput:
    """Outputs for command at t_(k+1), all batched. Foot order L,R."""
    q_cmd: torch.Tensor           # (B,10), commanded rad, clipped to joint limits
    q_ik: torch.Tensor            # (B,10), WBIK before load compensation
    delta_q_load: torch.Tensor    # (B,10), un-clipped offset rad; zero when disabled
    tau_load: torch.Tensor        # (B,10), N*m; zero when disabled
    com_des: torch.Tensor         # (B,3), planned CoM in {P}
    feet_des: torch.Tensor        # (B,2,3), planned feet at t_(k+1) in {P}
    contact: torch.Tensor         # (B,2), planned contact at t_(k+1)
    zmp: torch.Tensor             # (B,2), MPC-estimated ZMP at t_(k+1)
    zmp_ref: torch.Tensor         # (B,N,2), ZMP preview for MPC
    dcm_error: torch.Tensor       # (B,2), measured - planned at t_k
    reanchor_dx: torch.Tensor     # (B,), shift of planned x at new step
    j_now: torch.Tensor          # (B,), step index at t_k
    load_enabled: bool           # active flag for this tick
    valid: torch.Tensor          # (B,), finite GT and IK result


class WalkController:
    """Orchestrate existing gait modules without rewriting their algorithms.

    Usage:
        ctrl = WalkController(model, num_envs=1, config=WalkConfig())
        ctrl.reset(data)                    # after setting MuJoCo initial pose
        out = ctrl.step(data, cmd=[0.08, 0.0])
        data.ctrl[:] = out.q_cmd[0].cpu().numpy()  # v5 actuator order

    Runtime toggle:
        ctrl.set_load_compensation(False)  # WBIK only
        ctrl.set_load_compensation(True)   # WBIK + alpha*tau/Kp
    """

    def __init__(
        self,
        model,
        num_envs: int = 1,
        config: Optional[WalkConfig] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.model = model
        self.n = int(num_envs)
        self.device = torch.device(device)
        self.cfg = config if config is not None else WalkConfig()
        c = self.cfg

        self.kin = RobotKinematics(model, device=self.device)
        self.estimator = GroundTruthStateEstimator(model, device=self.device)
        self.scheduler = ContactScheduler(
            dt=c.dt, period_phase=c.period_phase, shift=c.shift,
            double_support=c.double_support, num_envs=self.n, device=self.device,
        )
        self.planner = FootstepPlanner(
            num_envs=self.n, period_phase=c.period_phase,
            num_footsteps=c.num_footsteps, device=self.device,
        )
        self.zmp_generator = ZMPGenerator(
            zmp_offset_x=c.zmp_offset_x, zmp_offset_y_out=c.zmp_offset_y_out,
            horizon=c.horizon,
        )
        self.swing = SwingTrajectoryGenerator(
            period_phase=c.period_phase, double_support=c.double_support,
            swing_height=c.swing_height,
        )
        self.ik = WholeBodyIK(self.kin)
        self.load_comp = LoadCompensation(
            self.kin,
            LoadCompensationParams(
                gravity=c.gravity, load=c.load_gain, servo_kp=c.servo_kp,
                cop_moment=c.cop_moment, zmp_offset_out=c.zmp_offset_y_out,
            ),
        )
        self.capture_point: CapturePoint | None = None
        self.mpc: LIPMMPC | None = None
        self.lipm_height: float | None = None
        self.initialized = False
        self.load_enabled = bool(c.enable_load_compensation)
        self._joint_qpos_adr = [
            int(model.jnt_qposadr[model.joint(name).id]) for name in JOINT_NAMES
        ]

        self.com_state: torch.Tensor | None = None
        self.com_z: torch.Tensor | None = None
        self.q_home: torch.Tensor | None = None
        self.foot_rot0: torch.Tensor | None = None
        self.ik_base_pos: torch.Tensor | None = None
        self.ik_base_rot: torch.Tensor | None = None
        self.ik_q: torch.Tensor | None = None
        self.ik_ks = None
        self.last_step: torch.Tensor | None = None
        self.contact: torch.Tensor | None = None
        self.last_q_cmd: torch.Tensor | None = None

    def set_load_compensation(self, enabled: bool) -> None:
        """Switch load compensation ON/OFF, keeping the same planner/MPC/IK."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.load_enabled = enabled

    @staticmethod
    def _remove_yaw(foot_rot: torch.Tensor) -> torch.Tensor:
        """Remove world-z yaw; match author's remove_yaw(ks.site_rot)."""
        yaw = torch.atan2(foot_rot[..., 1, 0], foot_rot[..., 0, 0])
        cs, sn = torch.cos(yaw), torch.sin(yaw)
        zero, one = torch.zeros_like(cs), torch.ones_like(cs)
        rz = torch.stack((
            cs, sn, zero,
            -sn, cs, zero,
            zero, zero, one,
        ), dim=-1).reshape(*yaw.shape, 3, 3)
        return rz @ foot_rot

    def _data_list(self, data):
        ds = list(data) if isinstance(data, (list, tuple)) else [data]
        if len(ds) != self.n:
            raise ValueError(f"Expected {self.n} MuJoCo MjData objects, got {len(ds)}")
        return ds

    def _command(self, cmd) -> torch.Tensor:
        """Accept (2,) or (B,2) desired horizontal velocity in m/s."""
        val = torch.as_tensor(cmd, dtype=torch.float32, device=self.device)
        if val.shape == (2,):
            val = val.unsqueeze(0).expand(self.n, -1)
        if val.shape != (self.n, 2) or not torch.isfinite(val).all():
            raise ValueError(f"cmd must be finite and have shape (2,) or ({self.n},2)")
        return val

    def reset(self, data, q_home=None) -> None:
        """Initialize from an already positioned MuJoCo pose (typically key 'home').

        q_home is optional (B,10) rad. If omitted, reads the 10 joint angles
        exactly once at reset for the WBIK seed; the runtime state estimator
        continues to read ONLY Ground Truth CoM and two foot positions.
        """
        ds = self._data_list(data)
        ground = self.estimator.update(ds)
        if not bool(ground.valid.all()):
            raise ValueError("Invalid Ground Truth during reset")

        base_pos = torch.as_tensor(
            np.stack([d.xpos[self.estimator.base_id].copy() for d in ds]),
            dtype=torch.float32, device=self.device,
        )
        base_rot = torch.as_tensor(
            np.stack([d.xmat[self.estimator.base_id].copy().reshape(3, 3) for d in ds]),
            dtype=torch.float32, device=self.device,
        )
        if q_home is None:
            q0 = torch.as_tensor(
                np.stack([d.qpos[self._joint_qpos_adr].copy() for d in ds]),
                dtype=torch.float32, device=self.device,
            )
        else:
            q0 = torch.as_tensor(q_home, dtype=torch.float32, device=self.device)
            if q0.shape == (10,) and self.n == 1:
                q0 = q0.unsqueeze(0)
            if q0.shape != (self.n, 10):
                raise ValueError(f"q_home must have shape ({self.n},10)")
        if not bool(torch.isfinite(q0).all()):
            raise ValueError("q_home is non-finite")
        ks = self.kin.forward_kinematics(base_pos, base_rot, q0)

        # Follow original reset: base/world CoM xy is initial MPC state;
        # LIPM height h is CoM height ABOVE the mean foot site, NOT CoM world z.
        h_batch = ground.com_pos[:, 2] - ground.feet_pos[:, :, 2].mean(dim=1)
        if not bool(torch.isfinite(h_batch).all()) or bool((h_batch <= 0).any()):
            raise ValueError("CoM must be higher than the mean foot site")
        h = float(h_batch.mean())  # one shared ZMP-MPC matrix for all envs
        self.lipm_height = h
        self.capture_point = CapturePoint(
            com_height=h, step_gain=self.cfg.step_gain,
            step_limit=self.cfg.step_limit,
            min_step_width=self.cfg.min_step_width, device=self.device,
        )
        self.mpc = LIPMMPC(
            com_height=h, dt=self.cfg.dt, horizon=self.cfg.horizon,
            q_zmp=self.cfg.q_zmp, q_vel=self.cfg.q_vel,
            r_jerk=self.cfg.r_jerk, device=self.device,
        )
        self.mpc.build_model()
        self.mpc.build_state_prediction()
        self.mpc.build_output_prediction()
        self.mpc.precompute_gains()

        self.scheduler.reset()
        env_ids = torch.arange(self.n, device=self.device)
        self.planner.reset(env_ids, ground.feet_pos)
        self.com_state = torch.zeros(self.n, 2, 3, device=self.device)
        self.com_state[:, :, 0] = ground.com_pos[:, :2]
        # Match author's initial MPC velocity/acceleration = 0.
        self.com_z = ks.com[:, 2].clone()
        self.q_home = q0.clone()
        self.foot_rot0 = self._remove_yaw(ks.site_rot)
        self.ik_base_pos = base_pos
        self.ik_base_rot = base_rot
        self.ik_q = q0.clone()
        self.ik_ks = ks
        self.last_step = torch.full((self.n,), -3, dtype=torch.long, device=self.device)
        self.contact = torch.ones((self.n, 2), dtype=torch.bool, device=self.device)
        self.last_q_cmd = q0.clone()
        self._first_ik = True
        self.initialized = True

    @torch.no_grad()
    def step(self, data, cmd) -> WalkOutput:
        """Compute joint position targets q_cmd for time t+dt (no physics step)."""
        if not self.initialized:
            raise RuntimeError("Call reset(data) before step(data, cmd)")
        ds = self._data_list(data)
        vel_cmd = self._command(cmd)
        cfg = self.cfg
        B = self.n
        b = torch.arange(B, device=self.device)
        t = self.scheduler.t

        # 1. Ground Truth at current time t_k: world xyz, two feet and relative CoM.
        gt = self.estimator.update(ds)

        # 2. Contact scheduler + commit new landing + x reanchor.
        j_now = self.scheduler.step_index(t)
        stance = self.scheduler.stance_feet(j_now)
        new_step = self.scheduler.new_step_event(j_now, self.last_step)
        self.planner.update_new_step(j_now, new_step, stance, vel_cmd)
        if cfg.reanchor_x:
            reanchor_dx = self.planner.reanchor_x(
                j_now, new_step, stance, gt.com_relative_to_feet,
                gt.valid, self.com_state,
            )
        else:
            reanchor_dx = torch.zeros(B, device=self.device)
        self.last_step = torch.where(j_now >= 0, j_now, torch.full_like(j_now, -1))

        # 3. Express measured CoM in the *planner's* coordinate convention.
        # Planned support foot + (GT CoM - GT foot); use prior tick's contact,
        # exactly as author's walker. This remains consistent after reanchor.
        plan_feet = torch.where(
            (j_now >= 0)[:, None, None], self.planner.feet_start, self.planner.feet0,
        )
        support = self.contact.float()
        measured_com_plan = (
            ((plan_feet + gt.com_relative_to_feet) * support[..., None]).sum(dim=1)
            / support.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        plan_com_fallback = torch.cat(
            (self.com_state[:, :, 0], self.com_z[:, None]), dim=-1
        )
        measured_com_plan = torch.where(gt.valid[:, None], measured_com_plan, plan_com_fallback)
        # With an invalid measurement, use the predicted horizontal velocity
        # to avoid injecting a false MPC/CP velocity error.
        vel_fallback = torch.cat((
            self.com_state[:, :, 1], torch.zeros(B, 1, device=self.device),
        ), dim=1)
        measured_vel = torch.where(gt.valid[:, None], gt.com_vel, vel_fallback)
        _, _, dcm_error = self.capture_point.compute_dcm_error(
            measured_com_plan, measured_vel, self.com_state,
        )
        pos_err = measured_com_plan[:, :2] - self.com_state[:, :, 0]
        vel_err = measured_vel[:, :2] - self.com_state[:, :, 1]

        # 4. CP correction, only while landing adjustment is still possible.
        correction = self.capture_point.compute_correction(dcm_error)
        nominal = self.planner.nominal_landing
        stance_y = self.planner.feet_start[b, stance, 1]
        corrected = self.capture_point.correct_landing(nominal, correction, stance, stance_y)
        adjust = self.scheduler.allow_footstep_adjust(t, j_now) & gt.valid
        self.planner.set_landing(corrected, adjust)

        # 5. Generate horizon ZMP and update LIPM CoM at time t_(k+1).
        footsteps, j_base = self.planner.footstep_sequence(j_now, vel_cmd)
        zmp_ref = self.zmp_generator.generate(
            footsteps, self.planner.feet0, j_base, self.scheduler,
        )
        fb_pos = torch.as_tensor(cfg.mpc_fb_pos, device=self.device, dtype=self.com_state.dtype)
        fb_vel = torch.as_tensor(cfg.mpc_fb_vel, device=self.device, dtype=self.com_state.dtype)
        self.com_state[:, :, 0] += fb_pos * pos_err
        self.com_state[:, :, 1] += fb_vel * vel_err
        vref = vel_cmd[:, None, :].expand(-1, cfg.horizon, -1)
        self.com_state = self.mpc.step(self.com_state, zmp_ref, vref)

        # 6. Swing trajectory and planned contact at t_(k+1).
        j_next = self.scheduler.step_index(t + cfg.dt)
        local_next = self.scheduler.local_time_phase(t + cfg.dt, j_next)
        stance_next = self.scheduler.stance_feet(j_next)
        _, _, ss_next = self.scheduler.phase_state(j_next, local_next)
        contact_next = self.scheduler.contact_state(stance_next, ss_next)
        feet_des, contact_next = self.swing.generate(
            footsteps, self.planner.feet0, j_base, j_next, local_next, contact_next,
        )
        com_des = torch.cat((self.com_state[:, :, 0], self.com_z[:, None]), dim=1)

        # 7. WBIK. Always retain UNCOMPENSATED warm start.
        n_iter = cfg.ik_initial_iterations if self._first_ik else cfg.ik_warm_iterations
        base_pos, base_rot, q_ik, ks = self.ik.solve(
            self.ik_base_pos, self.ik_base_rot, self.ik_q,
            com_des, feet_des, self.foot_rot0, self.q_home,
            iterations=n_iter, ks0=None if self._first_ik else self.ik_ks,
        )
        ik_finite = (
            torch.isfinite(base_pos).all(dim=1)
            & torch.isfinite(base_rot).all(dim=(1, 2))
            & torch.isfinite(q_ik).all(dim=1)
        )
        valid = gt.valid & ik_finite
        base_pos = torch.where(ik_finite[:, None], base_pos, self.ik_base_pos)
        base_rot = torch.where(ik_finite[:, None, None], base_rot, self.ik_base_rot)
        q_ik = torch.where(ik_finite[:, None], q_ik, self.ik_q)
        if not bool(ik_finite.all()):
            ks = self.kin.forward_kinematics(base_pos, base_rot, q_ik)
        self.ik_base_pos, self.ik_base_rot = base_pos, base_rot
        self.ik_q, self.ik_ks = q_ik, ks
        self._first_ik = False

        # 8. Optional feedforward load compensation.
        com_acc = self.com_state[:, :, 2]
        zmp = self.com_state[:, :, 0] - com_acc * (self.lipm_height / cfg.gravity)
        if self.load_enabled and cfg.load_gain > 0:
            loaded = self.load_comp.compute(
                q_ik=q_ik, ks=ks, contact=contact_next,
                feet=feet_des, com_acc=com_acc, zmp=zmp,
            )
            q_target = loaded.q_cmd
            delta_q_load = loaded.delta_q
            tau_load = loaded.joint_torques
        else:
            q_target = q_ik
            delta_q_load = torch.zeros_like(q_ik)
            tau_load = torch.zeros_like(q_ik)

        # Respect actuator's joint-position limits, do not send NaNs.
        q_target = torch.minimum(
            torch.maximum(q_target, self.kin.qmin[None, :]), self.kin.qmax[None, :]
        )
        q_cmd = torch.where(valid[:, None], q_target, self.last_q_cmd)
        self.last_q_cmd = q_cmd.clone()
        self.contact = contact_next.clone()
        self.scheduler.count()  # exactly one time increment per controller tick

        return WalkOutput(
            q_cmd=q_cmd, q_ik=q_ik, delta_q_load=delta_q_load,
            tau_load=tau_load, com_des=com_des, feet_des=feet_des,
            contact=contact_next, zmp=zmp, zmp_ref=zmp_ref,
            dcm_error=dcm_error, reanchor_dx=reanchor_dx,
            j_now=j_now, load_enabled=self.load_enabled, valid=valid,
        )
