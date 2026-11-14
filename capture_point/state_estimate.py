"""Ground-truth CoM reader for Open Duck Mini V2 (MuJoCo).

No IMU estimation, joint-state reading, forward kinematics, or filtering.

Outputs are batched Torch tensors in the planning frame {P}:
    com_pos: (B, 3) [m]
    com_vel: (B, 3) [m/s]
    valid:   (B,)   finite-value indicator

By default {P} coincides with MuJoCo world {W}. An optional, *stationary*
rigid transform can be supplied to express both quantities in another frame.
This is a coordinate change; the class does not track moving-frame motion.

Before calling update(), run mujoco.mj_forward(model, data) after changing
qpos/qvel manually, or call it after a completed mujoco.mj_step().
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch


@dataclass
class GroundTruthState:
    """Measured CoM state expressed in the planning frame {P}."""

    com_pos: torch.Tensor  # (B, 3), m
    com_vel: torch.Tensor  # (B, 3), m/s
    valid: torch.Tensor    # (B,), bool


class GroundTruthStateEstimator:
    """Read robot-subtree CoM position and velocity from MuJoCo Ground Truth.

    The subtree rooted at `base_link` includes the robot but not the floor.
    The returned values are copies (not views of MuJoCo's mutable buffers).
    """

    def __init__(self, model: mujoco.MjModel, device="cpu", base_body="base_link"):
        self.model = model
        self.device = torch.device(device)
        try:
            self.base_id = int(model.body(base_body).id)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Body {base_body!r} was not found in MuJoCo model") from exc

    @staticmethod
    def _as_data_list(data):
        """Accept one MjData or a list/tuple of MjData (batched output)."""
        if isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("data list cannot be empty")
            return data
        return [data]

    def get_com_position(self, data):
        """Return CoM position in world {W}, shape (B, 3).

        MuJoCo must already have up-to-date forward-kinematics data.
        """
        states = self._as_data_list(data)
        values = [d.subtree_com[self.base_id].copy() for d in states]
        return torch.as_tensor(np.stack(values, axis=0), device=self.device, dtype=torch.float32).clone()

    def get_com_velocity(self, data):
        """Return CoM velocity in world {W}, shape (B, 3).

        mj_subtreeVel explicitly updates subtree_linvel; mj_forward/mj_step
        need not run the subtree-velocity stage when no sensor requests it.
        """
        states = self._as_data_list(data)
        values = []
        for d in states:
            mujoco.mj_subtreeVel(self.model, d)
            values.append(d.subtree_linvel[self.base_id].copy())
        return torch.as_tensor(np.stack(values, axis=0), device=self.device, dtype=torch.float32).clone()

    def update(self, data, origin_world=None, rotation_world_from_plan=None):
        """Read Ground Truth and express it in planning frame {P}.

        Args:
            data: one MuJoCo MjData, or list/tuple of MjData.
            origin_world: location of planning-frame origin in world coordinates,
                (3,) or (B, 3), meters; default zero.
            rotation_world_from_plan: rotation R_WP, mapping {P} vectors to
                {W}, (3,3) or (B,3,3); default identity.

        For a stationary planning frame:
            p_P = R_WP.T @ (p_W - origin_world)
            v_P = R_WP.T @ v_W

        If {P} moves continuously, its origin velocity and angular velocity
        must also be accounted for externally before using these velocities
        as relative derivatives. Re-anchoring the planner does not by itself
        require a new physical-velocity estimate.
        """
        com_world = self.get_com_position(data)
        vel_world = self.get_com_velocity(data)
        B = com_world.shape[0]
        dtype, device = com_world.dtype, com_world.device

        if origin_world is None:
            origin = torch.zeros((B, 3), dtype=dtype, device=device)
        else:
            origin = torch.as_tensor(origin_world, dtype=dtype, device=device)
            if origin.shape == (3,):
                origin = origin.unsqueeze(0).expand(B, -1)
            if origin.shape != (B, 3):
                raise ValueError(f"origin_world must have shape (3,) or ({B}, 3)")

        if rotation_world_from_plan is None:
            R_wp = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
        else:
            R_wp = torch.as_tensor(rotation_world_from_plan, dtype=dtype, device=device)
            if R_wp.shape == (3, 3):
                R_wp = R_wp.unsqueeze(0).expand(B, -1, -1)
            if R_wp.shape != (B, 3, 3):
                raise ValueError(f"rotation_world_from_plan must have shape (3,3) or ({B},3,3)")

        R_pw = R_wp.transpose(-1, -2)
        com_pos = (R_pw @ (com_world - origin).unsqueeze(-1)).squeeze(-1)
        com_vel = (R_pw @ vel_world.unsqueeze(-1)).squeeze(-1)
        valid = torch.isfinite(com_pos).all(dim=1) & torch.isfinite(com_vel).all(dim=1)

        return GroundTruthState(com_pos=com_pos, com_vel=com_vel, valid=valid)
