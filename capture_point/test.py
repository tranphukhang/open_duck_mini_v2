"""Kiem thu FootstepPlanner.reanchor_x() - Open Duck Mini V2.

Dat file nay cung thu muc voi footstep_planner.py, sau do chay:
    python test_footstep_planner_reanchor_x.py

Quy uoc:
    L = 0 (chan trai), R = 1 (chan phai)
    stance: chan stance cua buoc VUA BAT DAU,
            tuc la chan swing cua buoc VUA HOAN THANH.
    swing = 1 - stance: chan stance cua buoc VUA HOAN THANH.
    feet_start, nominal_landing, updated_landing: toa do trong {P}.
    com_relative_to_feet[b, i] = p_CoM - p_foot_i, tu FK voi truc cung huong {P}.
    Gia su cac truc {K} va {P} duoc can chinh de cong truc tiep delta X.

Ky vong: reanchor chi xay ra khi new_step=True, j_now>0 va fk_valid=True.
Chung ta chi test FootstepPlanner; vi tri CoM trong MPC do module dieu phoi cap nhat
bang delta_x tra ve.
"""

import unittest
import torch

from footstep_planner import FootstepPlanner


class TestReanchorX(unittest.TestCase):
    def make_planner(self, num_envs=1):
        planner = FootstepPlanner(num_envs=num_envs, period_phase=0.4)
        feet_pos = torch.tensor(
            [[[0.0, 0.06, 0.005], [0.0, -0.06, 0.015]]],
            dtype=torch.float32,
        ).repeat(num_envs, 1, 1)
        planner.reset(torch.arange(num_envs), feet_pos)
        return planner

    def assert_tensor(self, actual, expected, message=""):
        torch.testing.assert_close(
            actual,
            torch.as_tensor(expected, dtype=actual.dtype, device=actual.device),
            rtol=0,
            atol=1e-6,
            msg=message or None,
        )

    def test_1_reanchor_after_completed_step_with_cp_adjusted_landing(self):
        """j=1: landing da hieu chinh tu CP, FK cho vi tri thuc te khac plan."""
        planner = self.make_planner()
        vx_cmd = torch.tensor([[0.20, 0.00]])

        # Bat dau buoc j=0: trai stance, phai swing. Landing danh dinh R_x=0.04 m.
        planner.update_new_step(
            j_now=torch.tensor([0]),
            new_step=torch.tensor([True]),
            stance=torch.tensor([0]),
            cmd=vx_cmd,
        )
        self.assert_tensor(planner.nominal_landing[:, 0], [0.04])

        # Gia lap CP chinh landing cua chan phai len 0.05 m truoc khi tiep dat.
        planner.updated_landing[:, 0] = 0.05

        # Bat dau buoc j=1: phai stance, trai chuan bi swing.
        # update_new_step() phai duoc goi TRUOC reanchor_x().
        planner.update_new_step(
            j_now=torch.tensor([1]),
            new_step=torch.tensor([True]),
            stance=torch.tensor([1]),
            cmd=vx_cmd,
        )
        self.assert_tensor(planner.feet_start[0, :, 0], [0.0, 0.05])
        self.assert_tensor(planner.nominal_landing[:, 0], [0.09])

        # FK: khoang cach R-L = r_L - r_R = 0.100 - 0.055 = 0.045 m.
        # Vi tri R uoc luong = 0 + 0.045 = 0.045 m; plan = 0.05 m.
        com_relative_to_feet = torch.tensor(
            [[[0.100, 0.02, 0.03], [0.055, 0.02, 0.03]]]
        )
        yz_feet_before = planner.feet_start[:, :, 1:].clone()
        yz_nominal_before = planner.nominal_landing[:, 1:].clone()
        yz_updated_before = planner.updated_landing[:, 1:].clone()
        feet0_before = planner.feet0.clone()
        rel_before = com_relative_to_feet.clone()

        delta_x = planner.reanchor_x(
            j_now=torch.tensor([1]),
            new_step=torch.tensor([True]),
            stance=torch.tensor([1]),
            com_relative_to_feet=com_relative_to_feet,
            fk_valid=torch.tensor([True]),
        )

        self.assert_tensor(delta_x, [-0.005])
        self.assert_tensor(planner.feet_start[0, :, 0], [-0.005, 0.045])
        self.assert_tensor(planner.nominal_landing[:, 0], [0.085])
        self.assert_tensor(planner.updated_landing[:, 0], [0.085])

        # Reanchor chi dich X; Y, Z va du lieu FK khong duoc thay doi.
        self.assertTrue(torch.equal(planner.feet_start[:, :, 1:], yz_feet_before))
        self.assertTrue(torch.equal(planner.nominal_landing[:, 1:], yz_nominal_before))
        self.assertTrue(torch.equal(planner.updated_landing[:, 1:], yz_updated_before))
        self.assertTrue(torch.equal(planner.feet0, feet0_before))
        self.assertTrue(torch.equal(com_relative_to_feet, rel_before))

    def test_2_positive_delta_x_when_left_becomes_stance(self):
        """j=2: trai la stance; vi tri trai FK lon hon vi tri plan 2 cm."""
        planner = self.make_planner()
        planner.feet_start[0, :, 0] = torch.tensor([0.04, 0.08])
        planner.nominal_landing[0, 0] = 0.10
        planner.updated_landing[0, 0] = 0.11

        # Trai vua tiep dat, phai la stance truoc do.
        # r_R - r_L = 0.07 - 0.09 = -0.02 => landed_L=0.08-0.02=0.06.
        rel = torch.tensor([[[0.09, 0., 0.], [0.07, 0., 0.]]])
        delta_x = planner.reanchor_x(
            torch.tensor([2]), torch.tensor([True]), torch.tensor([0]),
            rel, torch.tensor([True]),
        )
        self.assert_tensor(delta_x, [0.02])
        self.assert_tensor(planner.feet_start[0, :, 0], [0.06, 0.10])
        self.assert_tensor(planner.nominal_landing[:, 0], [0.12])
        self.assert_tensor(planner.updated_landing[:, 0], [0.13])

        # Phep dich dong bo khong lam thay doi khoang cach hai chan trong plan.
        self.assert_tensor(
            planner.feet_start[:, 1, 0] - planner.feet_start[:, 0, 0],
            [0.04],
        )

    def test_3_no_reanchor_at_first_step_j0(self):
        """j=0 chi bat dau buoc dau tien, chua co buoc truoc de reanchor."""
        planner = self.make_planner()
        planner.update_new_step(
            torch.tensor([0]), torch.tensor([True]), torch.tensor([0]),
            torch.tensor([[0.20, 0.0]]),
        )
        before = self.snapshot(planner)
        rel = torch.tensor([[[0.10, 0., 0.], [0.01, 0., 0.]]])
        delta_x = planner.reanchor_x(
            torch.tensor([0]), torch.tensor([True]), torch.tensor([0]),
            rel, torch.tensor([True]),
        )
        self.assert_tensor(delta_x, [0.0])
        self.assert_unchanged(planner, before)

    def test_4_no_reanchor_without_new_step(self):
        """Trong buoc hien tai, ham co the duoc goi moi dt=0.02 s nhung khong dich."""
        planner = self.make_planner()
        planner.feet_start[0, :, 0] = torch.tensor([0.00, 0.04])
        before = self.snapshot(planner)
        rel = torch.tensor([[[0.10, 0., 0.], [0.01, 0., 0.]]])
        delta_x = planner.reanchor_x(
            torch.tensor([1]), torch.tensor([False]), torch.tensor([1]),
            rel, torch.tensor([True]),
        )
        self.assert_tensor(delta_x, [0.0])
        self.assert_unchanged(planner, before)

    def test_5_no_reanchor_with_invalid_fk(self):
        """FK khong hop le thi giu nguyen toan bo vi tri du da co new_step."""
        planner = self.make_planner()
        planner.feet_start[0, :, 0] = torch.tensor([0.00, 0.04])
        before = self.snapshot(planner)
        # Khong dung NaN tai day de kiem tra ca dau ra delta_x la huu han.
        rel = torch.tensor([[[0.10, 0., 0.], [0.01, 0., 0.]]])
        delta_x = planner.reanchor_x(
            torch.tensor([1]), torch.tensor([True]), torch.tensor([1]),
            rel, torch.tensor([False]),
        )
        self.assert_tensor(delta_x, [0.0])
        self.assert_unchanged(planner, before)

    def test_6_batch_independence_mixed_conditions(self):
        """Nhieu robot: j=1, j=2, j=0, no event, invalid FK cung mot batch."""
        planner = self.make_planner(num_envs=5)
        planner.feet_start[:, :, 0] = torch.tensor([
            [0.00, 0.04],  # env0: j1, delta_x=-0.01
            [0.04, 0.08],  # env1: j2, delta_x=+0.02
            [0.00, 0.04],  # env2: j0, khong reanchor
            [0.00, 0.04],  # env3: khong co new_step
            [0.00, 0.04],  # env4: FK khong hop le
        ])
        planner.nominal_landing[:, 0] = 0.10
        planner.updated_landing[:, 0] = 0.11

        before = self.snapshot(planner)
        rel = torch.zeros(5, 2, 3)
        rel[:, 0, 0] = torch.tensor([0.08, 0.09, 0.10, 0.10, 0.10])
        rel[:, 1, 0] = torch.tensor([0.05, 0.07, 0.01, 0.01, 0.01])
        delta_x = planner.reanchor_x(
            j_now=torch.tensor([1, 2, 0, 1, 1]),
            new_step=torch.tensor([True, True, True, False, True]),
            stance=torch.tensor([1, 0, 0, 1, 1]),
            com_relative_to_feet=rel,
            fk_valid=torch.tensor([True, True, True, True, False]),
        )
        self.assert_tensor(delta_x, [-0.01, 0.02, 0.0, 0.0, 0.0])
        self.assert_tensor(planner.feet_start[:, :, 0], [
            [-0.01, 0.03],
            [0.06, 0.10],
            [0.00, 0.04],
            [0.00, 0.04],
            [0.00, 0.04],
        ])
        self.assert_tensor(planner.nominal_landing[:, 0], [0.09, 0.12, 0.10, 0.10, 0.10])
        self.assert_tensor(planner.updated_landing[:, 0], [0.10, 0.13, 0.11, 0.11, 0.11])
        self.assertTrue(torch.equal(planner.feet_start[:, :, 1:], before["feet_start"][:, :, 1:]))
        self.assertTrue(torch.equal(planner.nominal_landing[:, 1:], before["nominal_landing"][:, 1:]))
        self.assertTrue(torch.equal(planner.updated_landing[:, 1:], before["updated_landing"][:, 1:]))
        self.assertTrue(torch.equal(planner.feet0, before["feet0"]))

    def test_7_relative_fk_is_invariant_to_common_translation(self):
        """Tinh vi tri tu CHENH LECH FK: cong mot offset cho hai chan khong doi delta_x."""
        planner_a = self.make_planner()
        planner_b = self.make_planner()
        for planner in (planner_a, planner_b):
            planner.feet_start[0, :, 0] = torch.tensor([0.0, 0.04])
        rel = torch.tensor([[[0.08, 0.01, 0.02], [0.05, -0.03, 0.02]]])
        same_translation = torch.tensor([[[0.37, -0.10, 0.15]]])
        args = (torch.tensor([1]), torch.tensor([True]), torch.tensor([1]))

        delta_a = planner_a.reanchor_x(*args, rel, torch.tensor([True]))
        delta_b = planner_b.reanchor_x(*args, rel + same_translation, torch.tensor([True]))
        self.assert_tensor(delta_a, [-0.01])
        self.assert_tensor(delta_b, [-0.01])
        self.assertTrue(torch.allclose(planner_a.feet_start, planner_b.feet_start))

    @staticmethod
    def snapshot(planner):
        return {
            name: getattr(planner, name).clone()
            for name in ("feet0", "feet_offset0", "feet_start",
                         "nominal_landing", "updated_landing")
        }

    def assert_unchanged(self, planner, snapshot):
        for name, before in snapshot.items():
            self.assertTrue(
                torch.equal(getattr(planner, name), before),
                msg=f"{name} bi thay doi trong khi delta_x phai bang 0",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
