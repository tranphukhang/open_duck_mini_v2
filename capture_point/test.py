"""Kiem thu FootstepPlanner.footstep_sequence() cho Open Duck Mini V2.

Dat file trong thu muc capture_point/ cung footstep_planner.py, sau do chay:
    python test_footstep_planner_footstep_sequence.py

Quy uoc:
    0 = chan trai (L), 1 = chan phai (R)
    j = -1: shift ban dau, 2 chan van tiep xuc
    j =  0: chan trai stance, chan phai swing
    j =  1: chan phai stance, chan trai swing
    Tat ca vi tri (x, y, z) trong he lap ke hoach {P}; don vi met.

Test tinh gia tri mong doi doc lap (khong dung footstep_sequence
hoac step_offset de tinh expected), ngoai cac phep so sanh batch.
"""

import unittest
import torch

from footstep_planner import FootstepPlanner


class TestFootstepSequence(unittest.TestCase):
    @staticmethod
    def default_feet():
        # L: y = +0.09, z = 0.005; R: y = -0.09, z = 0.015.
        return torch.tensor([[[0.0, 0.09, 0.005],
                              [0.0, -0.09, 0.015]]], dtype=torch.float32)

    def make_planner(self, feet=None, period_phase=0.4, num_footsteps=14):
        if feet is None:
            feet = self.default_feet()
        planner = FootstepPlanner(
            num_envs=feet.shape[0], period_phase=period_phase,
            num_footsteps=num_footsteps, device="cpu"
        )
        planner.reset(torch.arange(feet.shape[0]), feet)
        return planner

    def assert_tensor(self, actual, expected, atol=1e-6):
        expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=0)

    def test_01_shift_initial_sequence(self):
        """j=-1: p[-1]=R0, p[0]=L0, p[1] du doan tu lenh van toc."""
        planner = self.make_planner()
        cmd = torch.tensor([[0.10, 0.02]])  # T_step=0.2; dx=0.02; dy_cmd=0.004
        points, j_base = planner.footstep_sequence(torch.tensor([-1]), cmd)

        self.assertEqual(tuple(points.shape), (1, 14, 3))
        self.assertEqual(j_base.tolist(), [0])
        # Thu tu: p[-1], p[0], p[1], p[2], p[3], p[4].
        self.assert_tensor(points[0, :6], [
            [0.00, -0.090, 0.015],  # p[-1]: R bat dau
            [0.00, +0.090, 0.005],  # p[0]: L stance
            [0.02, -0.086, 0.015],  # p[1]: R du doan
            [0.04, +0.098, 0.005],  # p[2]: L du doan
            [0.06, -0.078, 0.015],  # p[3]: R du doan
            [0.08, +0.106, 0.005],  # p[4]: L du doan
        ])

    def test_02_shift_ignores_updated_landing(self):
        """Trong shift, p[1] phai tinh danh dinh, khong lay updated_landing."""
        planner = self.make_planner()
        planner.set_landing(
            torch.tensor([[0.75, -0.50, 0.02]]),
            torch.tensor([True])
        )
        points, _ = planner.footstep_sequence(
            torch.tensor([-1]), torch.tensor([[0.10, 0.00]])
        )
        self.assert_tensor(points[0, 2], [0.02, -0.09, 0.015])
        self.assert_tensor(planner.updated_landing[0], [0.75, -0.50, 0.02])

    def test_03_first_step_matches_shift_prediction(self):
        """Chuyen shift -> j=0, voi cmd khong doi: chuoi trung nhau."""
        planner = self.make_planner()
        cmd = torch.tensor([[0.10, 0.02]])
        shift_points, _ = planner.footstep_sequence(torch.tensor([-1]), cmd)
        planner.update_new_step(
            j_now=torch.tensor([0]), new_step=torch.tensor([True]),
            stance=torch.tensor([0]), cmd=cmd
        )
        walking_points, j_base = planner.footstep_sequence(torch.tensor([0]), cmd)
        self.assertEqual(j_base.tolist(), [0])
        self.assert_tensor(walking_points, shift_points)
        self.assert_tensor(walking_points[0, 2], planner.updated_landing[0])

    def test_04_cp_correction_propagates_to_all_future_steps(self):
        """j=0: cap nhat CP p[1], cac vi tri p[2]... duoc tinh tu p[1]."""
        planner = self.make_planner()
        cmd = torch.tensor([[0.10, 0.0]])
        planner.update_new_step(torch.tensor([0]), torch.tensor([True]),
                                torch.tensor([0]), cmd)
        before, _ = planner.footstep_sequence(torch.tensor([0]), cmd)
        nominal_before = planner.nominal_landing.clone()

        correction = torch.tensor([[+0.013, -0.007, 0.0]])
        planner.set_landing(planner.nominal_landing + correction, torch.tensor([True]))
        after, _ = planner.footstep_sequence(torch.tensor([0]), cmd)

        self.assert_tensor(after[:, :2], before[:, :2])  # p[-1] va p[0] giu nguyen
        self.assert_tensor(after[:, 2:] - before[:, 2:],
                           correction[:, None, :].expand(1, 12, 3))
        self.assert_tensor(planner.nominal_landing, nominal_before)
        self.assert_tensor(after[:, 2], planner.updated_landing)

    def test_05_second_step_switches_stance_and_swing(self):
        """j=1: chan phai stance; chan trai swing; p[1] la landing chan trai."""
        planner = self.make_planner()
        cmd = torch.tensor([[0.10, 0.00]])
        planner.update_new_step(torch.tensor([0]), torch.tensor([True]),
                                torch.tensor([0]), cmd)
        # Gia lap landing chan phai (bước j=0) duoc CP hieu chinh theo X.
        planner.set_landing(torch.tensor([[0.03, -0.09, 0.015]]),
                            torch.tensor([True]))
        planner.update_new_step(torch.tensor([1]), torch.tensor([True]),
                                torch.tensor([1]), cmd)
        points, j_base = planner.footstep_sequence(torch.tensor([1]), cmd)

        self.assertEqual(j_base.tolist(), [1])
        self.assert_tensor(points[0, :5], [
            [0.00, +0.09, 0.005],  # p[-1]: chan trai, swing moi
            [0.03, -0.09, 0.015],  # p[0]: chan phai, stance moi
            [0.05, +0.09, 0.005],  # p[1]: landing chan trai
            [0.07, -0.09, 0.015],  # p[2]: landing chan phai
            [0.09, +0.09, 0.005],  # p[3]: landing chan trai
        ])

    def test_06_zero_command_preserves_alternating_initial_feet(self):
        """Khi vx=vy=0, chuoi chi luan phien vi tri L va R."""
        planner = self.make_planner()
        points, _ = planner.footstep_sequence(
            torch.tensor([-1]), torch.tensor([[0.0, 0.0]])
        )
        for i in range(14):
            expected = planner.feet0[0, 1 if i % 2 == 0 else 0]
            self.assert_tensor(points[0, i], expected)

    def test_07_custom_num_footsteps_and_period(self):
        """So diem va khoang buoc phai theo tham so, khong hard-code 14/0.2."""
        planner = self.make_planner(period_phase=0.6, num_footsteps=5)
        cmd = torch.tensor([[0.10, 0.00]])  # T_step = 0.3, dx = 0.03
        points, j_base = planner.footstep_sequence(torch.tensor([-1]), cmd)
        self.assertEqual(tuple(points.shape), (1, 5, 3))
        self.assertEqual(j_base.tolist(), [0])
        self.assert_tensor(points[0], [
            [0.00, -0.09, 0.015],
            [0.00, +0.09, 0.005],
            [0.03, -0.09, 0.015],
            [0.06, +0.09, 0.005],
            [0.09, -0.09, 0.015],
        ])

    def test_08_batch_independent_environments(self):
        """Ba moi truong co j, lenh, hinh hoc va landing khac nhau."""
        feet = torch.tensor([
            [[0.00, +0.09, 0.005], [0.00, -0.09, 0.015]],
            [[0.05, +0.07, 0.020], [0.04, -0.08, 0.000]],
            [[0.10, +0.10, 0.010], [0.12, -0.09, 0.020]],
        ])
        cmd = torch.tensor([[0.10, 0.02], [-0.05, -0.02], [0.20, 0.01]])
        j_now = torch.tensor([-1, 0, 1], dtype=torch.long)
        planner = self.make_planner(feet=feet)
        # Cau hinh gia lap sau update_new_step va CP feedback tai tung env.
        planner.updated_landing[1] = torch.tensor([0.025, -0.088, 0.000])
        planner.updated_landing[2] = torch.tensor([0.170, +0.110, 0.010])
        planner.feet_start[2, 1] = torch.tensor([0.145, -0.09, 0.020])

        batch_points, batch_jbase = planner.footstep_sequence(j_now, cmd)
        self.assertEqual(tuple(batch_points.shape), (3, 14, 3))
        self.assertEqual(batch_jbase.tolist(), [0, 0, 1])
        for i in range(3):
            single = self.make_planner(feet=feet[i:i+1])
            single.feet_start[:] = planner.feet_start[i:i+1]
            single.updated_landing[:] = planner.updated_landing[i:i+1]
            single_points, single_jbase = single.footstep_sequence(j_now[i:i+1], cmd[i:i+1])
            self.assert_tensor(batch_points[i:i+1], single_points)
            self.assert_tensor(batch_jbase[i:i+1], single_jbase)

    def test_09_function_does_not_modify_planner_state(self):
        """footstep_sequence chi tinh chuoi, khong sua cac bien noi bo."""
        planner = self.make_planner()
        cmd = torch.tensor([[0.10, -0.04]])
        planner.update_new_step(torch.tensor([0]), torch.tensor([True]),
                                torch.tensor([0]), cmd)
        names = ("feet0", "feet_offset0", "feet_start",
                 "nominal_landing", "updated_landing")
        before = {name: getattr(planner, name).clone() for name in names}

        _ = planner.footstep_sequence(torch.tensor([0]), cmd)
        _ = planner.footstep_sequence(torch.tensor([-1]), cmd)

        for name in names:
            self.assertTrue(
                torch.equal(getattr(planner, name), before[name]),
                msg=f"footstep_sequence() da thay doi {name}"
            )

    def test_10_last_prediction_and_step_alternation(self):
        """Kiem tra diem cuoi p[12] va thu tu 14 diem khong bi lech index."""
        planner = self.make_planner()
        cmd = torch.tensor([[0.10, 0.00]])
        points, _ = planner.footstep_sequence(torch.tensor([-1]), cmd)
        # 14 diem p[-1]..p[12], chi so tensor 13 -> p[12].
        self.assert_tensor(points[0, 13], [0.24, +0.09, 0.005])
        # p1..p12: x tang 0.02 tung buoc; y/z luan phien.
        for k in range(1, 13):
            p = points[0, k + 1]  # p_k
            self.assertAlmostEqual(p[0].item(), 0.02 * k, delta=1e-6)
            if k % 2 == 1:
                self.assertAlmostEqual(p[1].item(), -0.09, delta=1e-6)
                self.assertAlmostEqual(p[2].item(), 0.015, delta=1e-6)
            else:
                self.assertAlmostEqual(p[1].item(), +0.09, delta=1e-6)
                self.assertAlmostEqual(p[2].item(), 0.005, delta=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
