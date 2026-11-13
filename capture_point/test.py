"""Test ham ZMPGenerator.zmp_point() trong capture_point/zmp_generator.py.

Dat file nay cung thu muc voi zmp_generator.py, sau do chay:
    python test_zmp_generator_zmp_point.py

Khong mo do thi, khong tao CSV va khong luu anh.
"""

import unittest
import torch

from zmp_generator import ZMPGenerator


class TestZMPPoint(unittest.TestCase):
    def setUp(self):
        self.generator = ZMPGenerator()

    def assertTensorClose(self, actual, expected, atol=1e-7):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

    def test_01_default_offsets(self):
        """Gia tri default giong walker.py cua tac gia."""
        self.assertEqual(self.generator.zmp_offset_x, 0.005)
        self.assertEqual(self.generator.zmp_offset_y_out, 0.007)

    def test_02_left_foot_positive_y(self):
        """Chan trai (side=0): +5 mm theo X, +7 mm theo Y."""
        foot = torch.tensor([[0.0, 0.09, 0.03]])
        side = torch.tensor([0], dtype=torch.long)
        expected = torch.tensor([[0.005, 0.097]])
        self.assertTensorClose(self.generator.zmp_point(foot, side), expected)

    def test_03_right_foot_negative_y(self):
        """Chan phai (side=1): +5 mm theo X, -7 mm theo Y."""
        foot = torch.tensor([[0.0, -0.09, 0.03]])
        side = torch.tensor([1], dtype=torch.long)
        expected = torch.tensor([[0.005, -0.097]])
        self.assertTensorClose(self.generator.zmp_point(foot, side), expected)

    def test_04_both_feet_in_one_batch(self):
        """Hai chan co the xu ly cung luc, khong bi dao dau offset Y."""
        foot = torch.tensor([
            [0.10,  0.09, 0.00],
            [0.12, -0.09, 0.00],
        ])
        side = torch.tensor([0, 1], dtype=torch.long)
        expected = torch.tensor([
            [0.105,  0.097],
            [0.125, -0.097],
        ])
        self.assertTensorClose(self.generator.zmp_point(foot, side), expected)

    def test_05_custom_offsets(self):
        """Thay doi tham so offset ma khong sua ham zmp_point()."""
        gen = ZMPGenerator(zmp_offset_x=0.012, zmp_offset_y_out=0.020)
        foot = torch.tensor([[0.20, 0.08, 0.0], [0.20, -0.08, 0.0]])
        side = torch.tensor([0, 1], dtype=torch.long)
        expected = torch.tensor([[0.212, 0.100], [0.212, -0.100]])
        self.assertTensorClose(gen.zmp_point(foot, side), expected)

    def test_06_zero_offsets(self):
        """Khi offset = 0, ZMP dung tai vi tri foot site tren mat phang XY."""
        gen = ZMPGenerator(zmp_offset_x=0.0, zmp_offset_y_out=0.0)
        foot = torch.tensor([[0.15, 0.07, 0.02], [-0.10, -0.07, 0.05]])
        side = torch.tensor([0, 1])
        self.assertTensorClose(gen.zmp_point(foot, side), foot[:, :2])

    def test_07_z_coordinate_does_not_affect_zmp(self):
        """Thay doi Z cua ban chan khong lam thay doi ZMP tham chieu XY."""
        a = torch.tensor([[0.2, 0.1, -0.10], [0.3, -0.1, 0.01]])
        b = a.clone()
        b[:, 2] = torch.tensor([0.50, 1.20])
        side = torch.tensor([0, 1])
        self.assertTensorClose(self.generator.zmp_point(a, side), self.generator.zmp_point(b, side))

    def test_08_batched_horizon_80_samples(self):
        """Kich thuoc (B,N,3) -> (B,N,2), B=3, N=80."""
        batch, horizon = 3, 80
        foot = torch.zeros(batch, horizon, 3)
        foot[..., 0] = torch.arange(horizon).float()[None, :] * 0.01
        foot[..., 1] = torch.tensor([0.09, -0.09, 0.05])[:, None]
        side = (torch.arange(horizon)[None, :] + torch.arange(batch)[:, None]) % 2
        actual = self.generator.zmp_point(foot, side)
        expected = foot[..., :2].clone()
        expected[..., 0] += 0.005
        expected[..., 1] += torch.where(side == 0, 0.007, -0.007)
        self.assertTensorClose(actual, expected)
        self.assertEqual(tuple(actual.shape), (3, 80, 2))

    def test_09_single_foot_scalar_side(self):
        """Ho tro tensor mot ban chan (3,) va chi so chan vo huong ()."""
        foot = torch.tensor([0.04, -0.10, 0.02])
        side = torch.tensor(1)
        expected = torch.tensor([0.045, -0.107])
        self.assertTensorClose(self.generator.zmp_point(foot, side), expected)

    def test_10_does_not_modify_inputs(self):
        """Khong thay doi tensor foot va side ban dau."""
        foot = torch.tensor([[0.0, 0.09, 0.01], [0.0, -0.09, 0.02]])
        side = torch.tensor([0, 1])
        original_foot, original_side = foot.clone(), side.clone()
        _ = self.generator.zmp_point(foot, side)
        self.assertTrue(torch.equal(foot, original_foot))
        self.assertTrue(torch.equal(side, original_side))

    def test_11_preserves_floating_dtype(self):
        """Tinh toan cho foot float32 va float64."""
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                foot = torch.tensor([[0.1, 0.09, 0.0], [0.2, -0.09, 0.0]], dtype=dtype)
                side = torch.tensor([0, 1])
                actual = self.generator.zmp_point(foot, side)
                expected = torch.tensor([[0.105, 0.097], [0.205, -0.097]], dtype=dtype)
                self.assertEqual(actual.dtype, dtype)
                self.assertTensorClose(actual, expected)

    def test_12_gradient_through_foot_position(self):
        """Gia tri ZMP cho phep lan truyen dao ham theo toa do XY."""
        foot = torch.tensor([[0.1, 0.09, 0.02]], requires_grad=True)
        side = torch.tensor([0])
        self.generator.zmp_point(foot, side).sum().backward()
        expected_grad = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertTensorClose(foot.grad, expected_grad)

    def test_13_reference_formula_random_batch(self):
        """Doi chieu 100 mau voi cong thuc zmp = foot_xy + [dx, +/-dy]."""
        g = torch.Generator().manual_seed(2026)
        foot = torch.randn(100, 3, generator=g)
        side = torch.randint(0, 2, (100,), generator=g)
        expected = torch.stack([
            foot[:, 0] + 0.005,
            foot[:, 1] + torch.where(side == 0, 0.007, -0.007),
        ], dim=-1)
        self.assertTensorClose(self.generator.zmp_point(foot, side), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA khong kha dung")
    def test_14_cuda(self):
        """Kiem tra batch tren GPU khi may co CUDA."""
        foot = torch.tensor([[0.10, 0.09, 0.0], [0.10, -0.09, 0.0]], device="cuda")
        side = torch.tensor([0, 1], device="cuda")
        expected = torch.tensor([[0.105, 0.097], [0.105, -0.097]], device="cuda")
        self.assertTensorClose(self.generator.zmp_point(foot, side), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
