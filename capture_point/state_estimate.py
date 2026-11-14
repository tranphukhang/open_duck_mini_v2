"""Ground Truth CoM + feet for Open Duck Mini V2 (MuJoCo).

No IMU estimation, joint-state reading, FK, differentiation or filtering.

Outputs (Torch, planning frame {P}):
    com_pos:             (B, 3)    CoM position [m]
    com_vel:             (B, 3)    CoM velocity [m/s]
    feet_pos:            (B, 2, 3) left/right IK foot-site positions [m]
    com_relative_to_feet:(B, 2, 3) com_pos[:,None,:] - feet_pos [m]
    valid:               (B,)      all data finite

By default the planning frame {P} equals MuJoCo world {W}.
Optional origin_world and rotation_world_from_plan describe a stationary
frame transformation, applied consistently to both CoM and feet.

Run mj_forward() after manually changing qpos/qvel, or mj_step() before update().
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch


@dataclass
class GroundTruthState:
    """Ground Truth expressed in frame {P}; foot order: left, right."""

    com_pos: torch.Tensor               # (B, 3)
    com_vel: torch.Tensor               # (B, 3)
    feet_pos: torch.Tensor              # (B, 2, 3)
    com_relative_to_feet: torch.Tensor  # (B, 2, 3)
    valid: torch.Tensor                 # (B,), bool


class GroundTruthStateEstimator:
    """Read robot-subtree CoM and two foot sites directly from MuJoCo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        device="cpu",
        base_body="base_link",
        foot_sites=("L6_IK", "R6_IK"),
    ):
        self.model = model
        self.device = torch.device(device)

        try:
            self.base_id = int(model.body(base_body).id)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Body {base_body!r} was not found in MuJoCo model") from exc

        if len(foot_sites) != 2 or foot_sites[0] == foot_sites[1]:
            raise ValueError("foot_sites must be (left_site_name, right_site_name)")

        try:
            self.foot_site_ids = tuple(int(model.site(name).id) for name in foot_sites)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Required foot sites not found: {foot_sites}") from exc

    @staticmethod
    def _as_data_list(data):
        if isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("data list cannot be empty")
            return data
        return [data]

    def _to_tensor(self, values):
        """Copy MuJoCo's mutable NumPy buffers before making Torch tensors."""
        return torch.as_tensor(
            np.stack(values, axis=0), dtype=torch.float32, device=self.device
        ).clone()

    def get_com_position(self, data):
        """Read subtree CoM in {W}; output (B,3)."""
        return self._to_tensor([
            d.subtree_com[self.base_id].copy() for d in self._as_data_list(data)
        ])

    def get_com_velocity(self, data):
        """Read subtree CoM linear velocity in {W}; output (B,3)."""
        values = []
        for d in self._as_data_list(data):
            # Explicitly refresh subtree velocities (not always done in mj_step).
            mujoco.mj_subtreeVel(self.model, d)
            values.append(d.subtree_linvel[self.base_id].copy())
        return self._to_tensor(values)

    def get_foot_position(self, data):
        """Read L6_IK, R6_IK positions in {W}; output (B,2,3)."""
        return self._to_tensor([
            d.site_xpos[list(self.foot_site_ids)].copy()
            for d in self._as_data_list(data)
        ])

    def update(self, data, origin_world=None, rotation_world_from_plan=None):
        """Read Ground Truth and express all measurements in planning frame {P}.

        Args:
            data: MuJoCo MjData or list/tuple of MjData; batch size B.
            origin_world: origin of {P} expressed in {W}, (3,) or (B,3).
            rotation_world_from_plan: rotation R_WP, (3,3) or (B,3,3).

        For a stationary frame:
            p_P = R_WP.T @ (p_W - origin_world)
            v_P = R_WP.T @ v_W
            r_{com/foot,P} = p_{com,P} - p_{foot,P}

        This function does not compute transport terms for a moving frame.
        """
        com_world = self.get_com_position(data)
        vel_world = self.get_com_velocity(data)
        feet_world = self.get_foot_position(data)
        B = com_world.shape[0]
        dtype, device = com_world.dtype, com_world.device

        if origin_world is None:
            origin = torch.zeros((B, 3), dtype=dtype, device=device)
        else:
            origin = torch.as_tensor(origin_world, dtype=dtype, device=device)
            if origin.shape == (3,):
                origin = origin.unsqueeze(0).expand(B, -1)
            if origin.shape != (B, 3):
                raise ValueError(f"origin_world must have shape (3,) or ({B},3)")

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
        feet_pos = (R_pw[:, None] @ (feet_world - origin[:, None, :]).unsqueeze(-1)).squeeze(-1)
        com_relative_to_feet = com_pos[:, None, :] - feet_pos

        valid = (
            torch.isfinite(com_pos).all(dim=1)
            & torch.isfinite(com_vel).all(dim=1)
            & torch.isfinite(feet_pos).all(dim=(1, 2))
        )
        return GroundTruthState(
            com_pos=com_pos,
            com_vel=com_vel,
            feet_pos=feet_pos,
            com_relative_to_feet=com_relative_to_feet,
            valid=valid,
        )
