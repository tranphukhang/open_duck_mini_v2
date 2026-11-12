"""Integration-style unit tests for FootstepPlanner.update_new_step().

Place this file next to footstep_planner.py and run:
    python test_footstep_planner_update_new_step.py

All coordinates are illustrative foot-site positions in the planning frame {P}.
No MuJoCo or real FK is required.
"""

import unittest

import torch
from footstep_planner import FootstepPlanner


def make_planner(num_envs=1):
    planner = FootstepPlanner(num_envs=num_envs, period_phase=0.4, device="cpu")
    foot = torch.tensor(
        [[[0.00, 0.06, 0.01], [0.00, -0.06, 0.02]]],
        dtype=torch.float32,
    ).repeat(num_envs, 1, 1)
    planner.reset(torch.arange(num_envs), foot)
    return planner, foot


def check(actual, expected, message=""):
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, dtype=actual.dtype, device=actual.device),
        rtol=0,
        atol=1e-6,
        msg=message or None,
    )


class TestUpdateNewStep(unittest.TestCase):

    def test_01_first_step(self):
        """j=0: left stance, right swing, nominal landing from feet0."""
        p, foot = make_planner()
        p.update_new_step(
            j_now=torch.tensor([0]),
            new_step=torch.tensor([True]),
            stance=torch.tensor([0]),
            cmd=torch.tensor([[0.10, 0.00]]),
        )
        check(p.feet_start, foot)
        check(p.nominal_landing, [[0.02, -0.06, 0.02]])
        check(p.updated_landing, [[0.02, -0.06, 0.02]])
        check(p.feet0, foot)
        check(p.feet_offset0, [[-0.12, 0.01]])

    def test_02_later_step_uses_corrected_previous_landing(self):
        """j=1: previous corrected right landing becomes right stance."""
        p, foot = make_planner()
        p.update_new_step(
            torch.tensor([0]), torch.tensor([True]),
            torch.tensor([0]), torch.tensor([[0.10, 0.00]]),
        )
        # Simulate a CP correction previously accepted for the right foot.
        p.updated_landing[0] = torch.tensor([0.035, -0.068, 0.025])
        p.update_new_step(
            j_now=torch.tensor([1]),
            new_step=torch.tensor([True]),
            stance=torch.tensor([1]),
            cmd=torch.tensor([[0.10, 0.00]]),
        )
        check(p.feet_start[0, 0], foot[0, 0])
        check(p.feet_start[0, 1], [0.035, -0.068, 0.025])
        # Left swings: x +0.02, y +0.12, z -0.01.
        check(p.nominal_landing[0], [0.055, 0.052, 0.015])
        check(p.updated_landing, p.nominal_landing)

        # j=2: the corrected/accepted left landing becomes left stance.
        p.updated_landing[0] = torch.tensor([0.060, 0.050, 0.018])
        p.update_new_step(
            torch.tensor([2]), torch.tensor([True]),
            torch.tensor([0]), torch.tensor([[0.10, 0.00]]),
        )
        check(p.feet_start[0, 0], [0.060, 0.050, 0.018])
        check(p.feet_start[0, 1], [0.035, -0.068, 0.025])
        # Right swings: x +0.02, y -0.12, z +0.01.
        check(p.nominal_landing[0], [0.080, -0.070, 0.028])
        check(p.updated_landing, p.nominal_landing)

    def test_03_no_event_preserves_state(self):
        """Outside new-step events no stored positions should be overwritten."""
        p, _ = make_planner()
        p.updated_landing[0] = torch.tensor([0.035, -0.068, 0.025])
        before = {name: getattr(p, name).clone() for name in (
            "feet0", "feet_offset0", "feet_start",
            "nominal_landing", "updated_landing",
        )}
        for j, st in ((-1, 0), (0, 0), (1, 1)):
            p.update_new_step(
                j_now=torch.tensor([j]),
                new_step=torch.tensor([False]),
                stance=torch.tensor([st]),
                cmd=torch.tensor([[0.25, -0.08]]),
            )
        for name, expected in before.items():
            check(getattr(p, name), expected, name)

    def test_04_batch_mixed_events(self):
        """Only environments with new_step=True change their state."""
        p, foot = make_planner(num_envs=4)
        p.updated_landing[2] = torch.tensor([0.120, 0.090, 0.015])
        p.updated_landing[1] = torch.tensor([0.200, -0.100, 0.030])
        p.updated_landing[3] = torch.tensor([0.400, -0.050, 0.030])
        old = {name: getattr(p, name).clone() for name in (
            "feet0", "feet_offset0", "feet_start",
            "nominal_landing", "updated_landing",
        )}
        p.update_new_step(
            j_now=torch.tensor([0, -1, 2, 1]),
            new_step=torch.tensor([True, False, True, False]),
            stance=torch.tensor([0, 0, 0, 1]),
            cmd=torch.tensor([
                [0.10, 0.00], [0.20, 0.10], [-0.05, 0.02], [0.00, 0.00],
            ]),
        )
        # Env 0: first step, left stance -> right swing.
        check(p.nominal_landing[0], [0.020, -0.060, 0.020])
        check(p.feet_start[0], foot[0])
        # Env 2: later step, left stance takes previously updated landing.
        check(p.feet_start[2, 0], [0.120, 0.090, 0.015])
        check(p.feet_start[2, 1], foot[2, 1])
        check(p.nominal_landing[2], [0.110, -0.026, 0.025])
        check(p.updated_landing[0], p.nominal_landing[0])
        check(p.updated_landing[2], p.nominal_landing[2])
        # Envs 1 and 3 must be exactly unchanged.
        for env in (1, 3):
            for name, expected in old.items():
                check(getattr(p, name)[env], expected[env], f"{name}[{env}]")
        check(p.feet0, old["feet0"])
        check(p.feet_offset0, old["feet_offset0"])

    def test_05_first_step_reinitializes_old_step_data(self):
        """j=0 must use feet0, even if feet_start is stale."""
        p, foot = make_planner()
        p.feet_start[:] = 7.0
        p.nominal_landing[:] = 8.0
        p.updated_landing[:] = 9.0
        p.update_new_step(
            torch.tensor([0]), torch.tensor([True]),
            torch.tensor([0]), torch.tensor([[0.0, 0.0]]),
        )
        check(p.feet_start, foot)
        check(p.nominal_landing, [[0.0, -0.06, 0.02]])
        check(p.updated_landing, p.nominal_landing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
