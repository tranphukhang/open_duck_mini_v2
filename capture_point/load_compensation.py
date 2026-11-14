"""Model-based load compensation for Open Duck Mini V2 (10 leg joints).

Reference implementation:
  anhoang-21/open_duck_isaac, branch walker-fk-com
  - walker.py: _load_torques() and q += load * tau / servo_kp
  - kinematics.py: leg_torques()

All forces, moments, joint axes and positions are expressed in planning frame {P};
its +z axis points upward. Foot order is [left, right].

The algorithm is a *quasi-static feedforward approximation*, not full inverse
robot dynamics nor a Contact Wrench Cone feasibility solver.

Requires: torch. For the real robot use the existing RobotKinematics/KinState
from capture_point/robot_kinematics.py; the module does not need to import
MuJoCo itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from robot_kinematics import KinState, RobotKinematics


@dataclass
class LoadCompensationParams:
    """Parameters aligned with the author's WalkParams defaults (SI units)."""

    gravity: float = 9.81               # m/s^2
    load: float = 1.0                  # feedforward scaling, 0..1
    servo_kp: float = 13.37            # N*m/rad, MJCF v5 position actuator
    cop_moment: bool = True           # include foot-site CoP moment
    zmp_box_back: float = 0.022        # m: negative x bound relative to foot site
    zmp_box_front: float = 0.045       # m: positive x bound
    zmp_offset_out: float = 0.007      # m: +y for left, -y for right
    zmp_box_half_width: float = 0.003  # m: lateral half-width about outward offset


@dataclass
class LoadCompensationResult:
    """Outputs and diagnostics for one control tick.

    Shapes (B = batch size):
      q_cmd, delta_q, joint_torques: (B, 10)
      contact_forces: (B, 2, 3)
      contact_moments: (B, 2, 3) or None if disabled
      total_force: (B, 3)
    """

    q_cmd: torch.Tensor
    delta_q: torch.Tensor
    joint_torques: torch.Tensor
    contact_forces: torch.Tensor
    contact_moments: torch.Tensor | None
    total_force: torch.Tensor


class LoadCompensation:
    """Author-matched load-to-position-target feedforward compensation.

    The kinematics object exposes `.total_mass`, `.mass`, `.desc`, while `ks`
    is the final KinState returned by WholeBodyIK.solve(...).
    """

    def __init__(
        self,
        kin: RobotKinematics,
        params: LoadCompensationParams | None = None,
    ) -> None:
        self.kin = kin
        self.params = params if params is not None else LoadCompensationParams()
        p = self.params

        if p.gravity <= 0:
            raise ValueError("gravity must be positive")
        if p.servo_kp <= 0:
            raise ValueError("servo_kp must be positive")
        if not 0.0 <= p.load <= 1.0:
            raise ValueError("load must be in [0, 1]")
        if min(p.zmp_box_back, p.zmp_box_front, p.zmp_box_half_width) < 0:
            raise ValueError("CoP bounds must be nonnegative")
        if getattr(kin, "desc").shape[0] != 10 or getattr(kin, "mass").ndim != 1:
            raise ValueError("Expected a 10-joint robot with 1D body masses")
        if kin.desc.shape[1] != kin.mass.numel():
            raise ValueError("kin.desc and kin.mass have incompatible shapes")

        # 10 joint indices -> 2 feet: first 5 left, last 5 right.
        self.joint_to_foot = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)

    def compute_total_force(self, com_acc: torch.Tensor) -> torch.Tensor:
        """Compute total ground reaction force from horizontal CoM acceleration.

        com_acc: (B,2) [ddot(x), ddot(y)] in frame {P}, m/s^2.
        return: (B,3) [m*ax, m*ay, m*g] in N.
        """
        if com_acc.ndim != 2 or com_acc.shape[1] != 2:
            raise ValueError("com_acc must have shape (B, 2)")
        vertical = torch.full_like(com_acc[:, 0], self.params.gravity)
        return float(self.kin.total_mass) * torch.stack(
            (com_acc[:, 0], com_acc[:, 1], vertical), dim=-1
        )

    def distribute_contact_forces(
        self,
        total_force: torch.Tensor,
        contact: torch.Tensor,
        feet: torch.Tensor,
        zmp: torch.Tensor,
    ) -> torch.Tensor:
        """Distribute total force using lateral ZMP and the planned contact flags.

        total_force (B,3), contact (B,2), feet (B,2,3), zmp (B,2).
        contact True means foot in support. Returns (B,2,3) in N.

        Both feet: lambda = clamp((zmp_y - y_right)/(y_left - y_right)).
        One foot: assign all total force to that foot.
        Neither: return zero forces (same behavior as the reference code).
        """
        if total_force.ndim != 2 or total_force.shape[1] != 3:
            raise ValueError("total_force must have shape (B, 3)")
        b = total_force.shape[0]
        if contact.shape != (b, 2) or feet.shape != (b, 2, 3) or zmp.shape != (b, 2):
            raise ValueError("Expected contact (B,2), feet (B,2,3), zmp (B,2)")
        contact = contact.to(torch.bool)
        both = contact[:, 0] & contact[:, 1]

        span = feet[:, 0, 1] - feet[:, 1, 1]
        safe_span = torch.where(span.abs() > 1e-6, span, torch.ones_like(span))
        lam = ((zmp[:, 1] - feet[:, 1, 1]) / safe_span).clamp(0.0, 1.0)

        double_share = torch.stack((lam, 1.0 - lam), dim=1)
        share = torch.where(both[:, None], double_share, contact.to(total_force.dtype))
        return share[..., None] * total_force[:, None, :]

    def compute_contact_moments(
        self,
        contact_forces: torch.Tensor,
        contact: torch.Tensor,
        feet: torch.Tensor,
        zmp: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute CoP-induced moments at each foot site (B,2,3), N*m.

        CoP displacement is clamped to the same outward narrow rectangular
        strip as the reference walker. During double support, its y component
        is forced to the outward offset before clamping.
        """
        if not self.params.cop_moment:
            return None
        if contact_forces.ndim != 3 or contact_forces.shape[1:] != (2, 3):
            raise ValueError("contact_forces must have shape (B, 2, 3)")
        b = contact_forces.shape[0]
        if contact.shape != (b, 2) or feet.shape != (b, 2, 3) or zmp.shape != (b, 2):
            raise ValueError("Expected contact (B,2), feet (B,2,3), zmp (B,2)")

        contact = contact.to(torch.bool)
        both = contact[:, 0] & contact[:, 1]
        d = zmp[:, None, :] - feet[:, :, :2]
        out = d.new_tensor([self.params.zmp_offset_out, -self.params.zmp_offset_out])
        dx = d[..., 0].clamp(-self.params.zmp_box_back, self.params.zmp_box_front)
        dy_pre = torch.where(both[:, None], out[None, :].expand_as(d[..., 1]), d[..., 1])
        dy = (dy_pre - out[None, :]).clamp(
            -self.params.zmp_box_half_width, self.params.zmp_box_half_width
        ) + out[None, :]
        r = torch.stack((dx, dy, torch.zeros_like(dx)), dim=-1)
        return torch.cross(r, contact_forces, dim=-1) * contact[..., None]

    def compute_joint_torques(
        self,
        ks: KinState,
        contact_forces: torch.Tensor,
        contact_moments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute quasi-static torque for 10 joints, matching leg_torques().

        tau_j = a_j dot [sum_{body in descendants(j)} m_b*(c_b-p_j) x (g*ez)
                           - (p_foot-p_j) x f_foot - M_foot].
        All terms are expressed in the same plan frame {P}.
        """
        if ks.axis.ndim != 3 or ks.axis.shape[1:] != (10, 3):
            raise ValueError("ks.axis must have shape (B, 10, 3)")
        b = ks.axis.shape[0]
        if (ks.anchor.shape != (b, 10, 3)
                or ks.com_body.shape != (b, self.kin.mass.numel(), 3)
                or ks.site_pos.shape != (b, 2, 3)
                or contact_forces.shape != (b, 2, 3)):
            raise ValueError("Incompatible KinState / contact force shapes")
        if contact_moments is not None and contact_moments.shape != (b, 2, 3):
            raise ValueError("contact_moments must have shape (B, 2, 3)")

        # Aggregate mass and mass-weighted CoM for descendants of each joint.
        # (10, nbodies), (B,10,3) and (1,10,1).
        wd = self.kin.desc.to(device=ks.com_body.device, dtype=ks.com_body.dtype) \
             * self.kin.mass.to(device=ks.com_body.device, dtype=ks.com_body.dtype)[None, :]
        msum = wd.sum(dim=1)[None, :, None]
        mc = torch.einsum("jb,nbk->njk", wd, ks.com_body)
        up = ks.com_body.new_tensor([0.0, 0.0, self.params.gravity])
        grav = torch.cross(mc - msum * ks.anchor, up.expand_as(mc), dim=-1)
        tau_gravity = (ks.axis * grav).sum(dim=-1)

        idx = self.joint_to_foot.to(device=ks.site_pos.device)
        p_foot = ks.site_pos[:, idx, :]
        f_foot = contact_forces[:, idx, :]
        lever = torch.cross(p_foot - ks.anchor, f_foot, dim=-1)
        if contact_moments is not None:
            lever = lever + contact_moments[:, idx, :]
        return tau_gravity - (ks.axis * lever).sum(dim=-1)

    def apply_compensation(
        self, q_ik: torch.Tensor, joint_torques: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (q_cmd, delta_q) in radians; no joint limit clamp here.

        q_cmd = q_ik + load * tau_load / servo_kp.
        Joint target clipping belongs to the downstream actuator/controller.
        """
        if q_ik.ndim != 2 or q_ik.shape[1] != 10 or joint_torques.shape != q_ik.shape:
            raise ValueError("q_ik and joint_torques must both have shape (B, 10)")
        delta_q = (self.params.load / self.params.servo_kp) * joint_torques
        return q_ik + delta_q, delta_q

    def compute(
        self,
        q_ik: torch.Tensor,
        ks: KinState,
        contact: torch.Tensor,
        feet: torch.Tensor,
        com_acc: torch.Tensor,
        zmp: torch.Tensor,
    ) -> LoadCompensationResult:
        """Run the entire compensation pipeline for one 50 Hz control tick.

        Inputs:
          q_ik:    (B,10) IK joint targets, rad.
          ks:      final KinState returned by WholeBodyIK.solve(...).
          contact: (B,2) [left,right] support flags at t+dt.
          feet:    (B,2,3) planned foot positions at t+dt in {P} (not FK).
          com_acc: (B,2) CoM acceleration from LIPM-MPC in {P}, m/s^2.
          zmp:     (B,2) computed as com_xy - com_acc/(g/h), in {P}.

        Note: use `feet` for force distribution / CoP; use `ks.site_pos`
        (actual IK FK) for the joint moment arms, exactly like the author.
        """
        total_force = self.compute_total_force(com_acc)
        forces = self.distribute_contact_forces(total_force, contact, feet, zmp)
        moments = self.compute_contact_moments(forces, contact, feet, zmp)
        tau = self.compute_joint_torques(ks, forces, moments)
        q_cmd, delta_q = self.apply_compensation(q_ik, tau)
        return LoadCompensationResult(
            q_cmd=q_cmd,
            delta_q=delta_q,
            joint_torques=tau,
            contact_forces=forces,
            contact_moments=moments,
            total_force=total_force,
        )
