"""Gait contact matching reward for Open Duck Mini V2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity import mdp as velocity_mdp
from rl.gait_observation import gait_reference_contact

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ============================================================
# Gait Contact Score
# ============================================================

def gait_contact_score(
    contact_actual: torch.Tensor,
    contact_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate bilateral foot contact matching.

    Reward:
        Both feet match:       +1.0
        One foot mismatches:   -0.5
        Both feet mismatch:    -1.0

    Args:
        contact_actual: (B, 2)
        contact_ref:    (B, 2)

    Returns:
        reward: (B,)
    """

    if (
        contact_actual.shape != contact_ref.shape
        or contact_ref.ndim != 2
        or contact_ref.shape[1] != 2
    ):
        raise ValueError(
            "Contact tensors must have shape (num_envs, 2)"
        )

    # Convert contact signals to boolean
    actual = contact_actual > 0.5
    reference = contact_ref > 0.5

    # Count mismatched feet
    mismatch_count = (
        actual != reference
    ).sum(dim=1)

    # Calculate reward
    reward = 1.0 - 1.5 * mismatch_count.float()

    # Limit the minimum reward
    reward = reward.clamp_min(-1.0)

    return reward


# ============================================================
# Gait Contact Matching Reward
# ============================================================

def gait_contact_matching(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "feet_contact",
) -> torch.Tensor:
    """
    Reward matching between actual and reference foot contacts.

    Reference contact:
        [left_contact, right_contact]

    Actual contact:
        [left_contact, right_contact]

    Output:
        (num_envs,)
    """

    # --------------------------------------------------------
    # 1. Reference contact from gait scheduler
    # --------------------------------------------------------

    contact_ref = gait_reference_contact(
        env,
        command_name=command_name,
    )

    # --------------------------------------------------------
    # 2. Actual contact from MuJoCo
    # --------------------------------------------------------

    contact_actual = velocity_mdp.foot_contact(
        env,
        sensor_name=sensor_name,
    )

    # --------------------------------------------------------
    # 3. Compute gait reward
    # --------------------------------------------------------

    reward = gait_contact_score(
        contact_actual,
        contact_ref,
    )

    # --------------------------------------------------------
    # 4. Contact diagnostics
    # --------------------------------------------------------

    ref_bool = contact_ref > 0.5
    actual_bool = contact_actual > 0.5

    # Both feet match the reference
    exact_match = (
        actual_bool == ref_bool
    ).all(dim=1)

    # Single Support phase
    is_ss = ref_bool.sum(dim=1) == 1

    # Double Support phase
    is_ds = ref_bool.sum(dim=1) == 2

    # Calculate conditional rate
    def masked_rate(
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:

        numerator = (
            mask & condition
        ).float().sum()

        denominator = (
            mask.float().sum()
        ).clamp_min(1.0)

        return numerator / denominator

    # --------------------------------------------------------
    # 5. TensorBoard diagnostics
    # --------------------------------------------------------

    logs = env.extras.get("log")

    if isinstance(logs, dict):

        # Overall contact matching rate
        logs["Metrics/gait_exact_match_rate"] = (
            exact_match.float().mean()
        )

        # Contact matching rate during SSP
        logs["Metrics/gait_ss_match_rate"] = (
            masked_rate(
                is_ss,
                exact_match,
            )
        )

        # Contact matching rate during DSP
        logs["Metrics/gait_ds_match_rate"] = (
            masked_rate(
                is_ds,
                exact_match,
            )
        )

        # Both feet contacting during SSP
        logs["Metrics/gait_ss_double_contact_rate"] = (
            masked_rate(
                is_ss,
                actual_bool.all(dim=1),
            )
        )

    # --------------------------------------------------------
    # 6. Return reward
    # --------------------------------------------------------

    return reward