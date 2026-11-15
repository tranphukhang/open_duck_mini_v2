from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity import mdp as velocity_mdp
from rl.gait_observation import gait_reference_contact

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def gait_contact_matching(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_contact",
) -> torch.Tensor:
    """
    Reward matching between actual and reference foot contacts.

    Reference:
        [left_contact, right_contact]

    Reward:
        +1.0 if contact matches reference
        -0.3 otherwise

    Output:
        (num_envs,)
    """

    # Reference contacts from gait scheduler
    contact_ref = gait_reference_contact(
        env,
        command_name=command_name,
    )

    # Actual contacts from contact sensor
    contact_actual = velocity_mdp.foot_contact(
        env,
        sensor_name=sensor_name,
    )

    # Compare actual and reference contact states
    contact_match = (
        contact_actual.bool() == contact_ref.bool()
    )

    # Reward each foot
    reward = torch.where(
        contact_match,
        torch.ones_like(contact_actual),
        -0.3 * torch.ones_like(contact_actual),
    )

    # Average reward over two feet
    return torch.mean(reward, dim=1)