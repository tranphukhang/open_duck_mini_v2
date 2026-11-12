"""Kiem thu step_offset() trong FootstepPlanner.

Dat file cung thu muc voi footstep_planner.py, sau do chay:
    python test_footstep_planner_step_offset.py

Quy uoc:
    swing = 0: chan trai Swing
    swing = 1: chan phai Swing
    feet_pos[:, 0, :] = chan trai; feet_pos[:, 1, :] = chan phai
"""

import unittest

import torch

from footstep_planner import FootstepPlanner


class TestStepOffset(unittest.TestCase):
    def make_planner(self, feet_pos, period_phase=0.4):
        """Khoi tao planner tu toa do foot site gia lap trong he {P}."""
        num_envs = feet_pos.shape[0]
        planner = FootstepPlanner(num_envs=num_envs, period_phase=period_phase)
        env_ids = torch.arange(num_envs, dtype=torch.long)
        planner.reset(env_ids, feet_pos)
        return planner

    def assert_offset(self, actual, expected):
        torch.testing.assert_close(
            actual,
            torch.tensor(expected, dtype=torch.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_1_right_swing(self):
        """Chan phai Swing: y_R - y_L = -0.12, z_R - z_L = +0.01."""
        feet_pos = torch.tensor([
            [[0.0, 0.06, 0.005], [0.0, -0.06, 0.015]],
        ])
        planner = self.make_planner(feet_pos)

        swing = torch.tensor([1], dtype=torch.long)
        cmd = torch.tensor([[0.10, 0.00]])
        actual = planner.step_offset(swing, cmd)

        # dx = 0.10 * 0.20 = 0.02; dy = -0.12; dz = 0.01
        self.assert_offset(actual, [[0.02, -0.12, 0.01]])

    def test_2_left_swing(self):
        """Doi chan Swing thi dau hinh hoc Y va Z phai dao nguoc."""
        feet_pos = torch.tensor([
            [[0.0, 0.06, 0.005], [0.0, -0.06, 0.015]],
        ])
        planner = self.make_planner(feet_pos)

        swing = torch.tensor([0], dtype=torch.long)
        cmd = torch.tensor([[0.10, 0.00]])
        actual = planner.step_offset(swing, cmd)

        # dx = 0.02; dy = +0.12; dz = -0.01
        self.assert_offset(actual, [[0.02, 0.12, -0.01]])

    def test_3_nonzero_lateral_velocity(self):
        """Kiem tra ca vx, vy va hai chieu Swing."""
        feet_pos = torch.tensor([
            [[0.0, 0.06, 0.005], [0.0, -0.06, 0.015]],
            [[0.0, 0.06, 0.005], [0.0, -0.06, 0.015]],
        ])
        planner = self.make_planner(feet_pos)

        swing = torch.tensor([1, 0], dtype=torch.long)
        cmd = torch.tensor([
            [0.15, -0.05],
            [0.15, -0.05],
        ])
        actual = planner.step_offset(swing, cmd)

        # dx = 0.03; vy * T = -0.01
        self.assert_offset(actual, [
            [0.03, -0.13,  0.01],  # Right Swing
            [0.03,  0.11, -0.01],  # Left Swing
        ])

    def test_4_batch_independent_environments(self):
        """Moi robot co feet0, swing va cmd rieng, khong lan du lieu."""
        feet_pos = torch.tensor([
            [[0.00, 0.06,  0.00], [0.00, -0.06, 0.01]],
            [[0.02, 0.05,  0.03], [0.02, -0.07, 0.00]],
            [[0.00, 0.08, -0.02], [0.00, -0.04, 0.01]],
        ])
        planner = self.make_planner(feet_pos)

        swing = torch.tensor([1, 0, 1], dtype=torch.long)
        cmd = torch.tensor([
            [ 0.10,  0.00],
            [-0.05,  0.02],
            [ 0.20, -0.03],
        ])
        actual = planner.step_offset(swing, cmd)

        self.assertEqual(tuple(actual.shape), (3, 3))
        self.assert_offset(actual, [
            [ 0.02, -0.120, 0.01],
            [-0.01,  0.124, 0.03],
            [ 0.04, -0.126, 0.03],
        ])

    def test_5_custom_gait_period(self):
        """step_offset phai dung half_period_phase cua planner."""
        feet_pos = torch.tensor([
            [[0.0, 0.06, 0.0], [0.0, -0.06, 0.0]],
        ])
        planner = self.make_planner(feet_pos, period_phase=0.6)

        actual = planner.step_offset(
            torch.tensor([1]), torch.tensor([[0.10, 0.02]])
        )
        # half = 0.30; dx = 0.03; dy = -0.12 + 0.006
        self.assert_offset(actual, [[0.03, -0.114, 0.0]])

    def test_6_step_offset_does_not_modify_planner_state(self):
        """step_offset chi tra offset, khong duoc tu cap nhat state."""
        feet_pos = torch.tensor([
            [[0.0, 0.06, 0.0], [0.0, -0.06, 0.01]],
        ])
        planner = self.make_planner(feet_pos)
        state_names = (
            "feet0", "feet_offset0", "feet_start",
            "nominal_landing", "updated_landing",
        )
        before = {name: getattr(planner, name).clone() for name in state_names}

        planner.step_offset(torch.tensor([1]), torch.tensor([[0.1, -0.02]]))

        for name in state_names:
            self.assertTrue(
                torch.equal(getattr(planner, name), before[name]),
                msg=f"step_offset() da lam thay doi {name}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
