from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ============================================================
# Gait Parameters
# ============================================================

GAIT_CYCLE_S = 0.40
STEP_S = 0.20
DOUBLE_SUPPORT_S = 0.04

MOVING_THRESHOLD = 0.01


# ============================================================
# Gait Observation
# ============================================================

def gait_observation(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
) -> torch.Tensor:
    """
    Return gait phase and reference foot contacts.

    Output:
        [sin(phase), cos(phase), left_contact, right_contact]

    Shape:
        (num_envs, 4)
    """

    # --------------------------------------------------------
    # 1. Get velocity command
    # --------------------------------------------------------

    command = env.command_manager.get_command(command_name)

    if command is None:
        raise ValueError(f"Command '{command_name}' not found")

    # --------------------------------------------------------
    # 2. Convert gait durations to policy steps
    # --------------------------------------------------------

    dt = env.step_dt

    cycle_ticks = round(GAIT_CYCLE_S / dt)
    step_ticks = round(STEP_S / dt)
    ds_ticks = round(DOUBLE_SUPPORT_S / dt)

    # --------------------------------------------------------
    # 3. Determine current gait step
    # --------------------------------------------------------

    tick = env.episode_length_buf

    tick_in_cycle = torch.remainder(tick, cycle_ticks)

    step_index = torch.div(
        tick_in_cycle,
        step_ticks,
        rounding_mode="floor",
    )

    local_tick = torch.remainder(
        tick_in_cycle,
        step_ticks,
    )

    # --------------------------------------------------------
    # 4. Determine contact phase
    # --------------------------------------------------------

    is_ds = local_tick < ds_ticks

    # Even step: left stance, right swing
    # Odd step: right stance, left swing

    is_left_stance = torch.remainder(step_index, 2) == 0

    left_contact = is_ds | is_left_stance
    right_contact = is_ds | (~is_left_stance)

    # --------------------------------------------------------
    # 5. Determine whether robot should walk
    # --------------------------------------------------------

    command_speed = torch.linalg.vector_norm(
        command[:, :2],
        dim=-1,
    )

    moving = command_speed > MOVING_THRESHOLD

    # --------------------------------------------------------
    # 6. Calculate gait phase
    # --------------------------------------------------------

    phase = (
        2.0
        * math.pi
        * tick_in_cycle.float()
        / cycle_ticks
    )

    sin_phase = torch.where(
        moving,
        torch.sin(phase),
        torch.zeros_like(phase),
    )

    cos_phase = torch.where(
        moving,
        torch.cos(phase),
        torch.ones_like(phase),
    )

    # --------------------------------------------------------
    # 7. Standing command: both feet in contact
    # --------------------------------------------------------

    left_contact = left_contact | (~moving)
    right_contact = right_contact | (~moving)

    # --------------------------------------------------------
    # 8. Assemble observation
    # --------------------------------------------------------

    observation = torch.stack(
        (
            sin_phase,
            cos_phase,
            left_contact.float(),
            right_contact.float(),
        ),
        dim=-1,
    )

    return observation