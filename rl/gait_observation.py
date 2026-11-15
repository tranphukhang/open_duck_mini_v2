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
# Gait State
# ============================================================

def _gait_state(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gait phase and reference foot contacts."""

    # --------------------------------------------------------
    # 1. Get velocity command
    # --------------------------------------------------------

    command = env.command_manager.get_command(command_name)

    if command is None:
        raise ValueError(
            f"Command '{command_name}' not found"
        )

    # --------------------------------------------------------
    # 2. Convert durations to policy ticks
    # --------------------------------------------------------

    dt = env.step_dt

    cycle_ticks = round(GAIT_CYCLE_S / dt)
    step_ticks = round(STEP_S / dt)
    ds_ticks = round(DOUBLE_SUPPORT_S / dt)

    assert cycle_ticks == 2 * step_ticks
    assert 0 < ds_ticks < step_ticks

    # --------------------------------------------------------
    # 3. Get current time within gait cycle
    # --------------------------------------------------------

    tick = env.episode_length_buf

    tick_in_cycle = torch.remainder(
        tick,
        cycle_ticks,
    )

    # --------------------------------------------------------
    # 4. Calculate gait phase
    # --------------------------------------------------------

    phase = (
        2.0
        * math.pi
        * tick_in_cycle.float()
        / cycle_ticks
    )

    # --------------------------------------------------------
    # 5. Check walking command
    # --------------------------------------------------------

    command_speed = torch.linalg.vector_norm(
        command[:, :2],
        dim=-1,
    )

    moving = command_speed > MOVING_THRESHOLD

    # --------------------------------------------------------
    # 6. Phase observation
    # --------------------------------------------------------

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

    phase_obs = torch.stack(
        (sin_phase, cos_phase),
        dim=-1,
    )

    # --------------------------------------------------------
    # 7. Reference contact schedule
    # --------------------------------------------------------

    step_index = torch.div(
        tick_in_cycle,
        step_ticks,
        rounding_mode="floor",
    )

    local_tick = torch.remainder(
        tick_in_cycle,
        step_ticks,
    )

    is_ds = local_tick < ds_ticks

    # Even step: left stance, right swing
    # Odd step: right stance, left swing

    is_left_stance = step_index == 0

    left_contact = (
        is_ds | is_left_stance | (~moving)
    )

    right_contact = (
        is_ds | (~is_left_stance) | (~moving)
    )

    contact_ref = torch.stack(
        (
            left_contact,
            right_contact,
        ),
        dim=-1,
    ).float()

    return phase_obs, contact_ref


# ============================================================
# Actor: Gait Phase Observation
# ============================================================

def gait_observation(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
) -> torch.Tensor:
    """
    Return gait phase observation.

    Output:
        [sin(phase), cos(phase)]

    Shape:
        (num_envs, 2)
    """

    phase_obs, _ = _gait_state(
        env,
        command_name,
    )

    return phase_obs


# ============================================================
# Critic: Reference Contact Observation
# ============================================================

def gait_reference_contact(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
) -> torch.Tensor:
    """
    Return reference contact states.

    Output:
        [left_contact, right_contact]

    Shape:
        (num_envs, 2)
    """

    _, contact_ref = _gait_state(
        env,
        command_name,
    )

    return contact_ref