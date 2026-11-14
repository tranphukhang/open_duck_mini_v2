"""Kiểm thử CapturePoint.compute_dcm_error() theo repo open_duck_mini_v2.

Đặt file này cạnh capture_point.py rồi chạy:
    python test_capture_point_compute_dcm_error.py
    python test_capture_point_compute_dcm_error.py --no-plot

Chỉ hiển thị biểu đồ Matplotlib; không lưu hình ảnh hay CSV.
Các tín hiệu minh họa là dữ liệu tổng hợp, không phải Ground Truth MuJoCo thật.
"""

import argparse
import math
import unittest

import torch

from capture_point import CapturePoint


DTYPE = torch.float32


def make_states(pos_xy, vel_xy, *, z=0.15, vz=0.0, acc_xy=None):
    """Chuyển dữ liệu (B, 2) thành input Ground Truth và MPC (B, 2, 3).

    pos_xy, vel_xy là trạng thái kế hoạch; dữ liệu đo được mặc định bằng kế hoạch.
    """
    pos_xy = torch.as_tensor(pos_xy, dtype=DTYPE)
    vel_xy = torch.as_tensor(vel_xy, dtype=DTYPE)
    assert pos_xy.shape == vel_xy.shape and pos_xy.ndim == 2 and pos_xy.shape[-1] == 2
    b = pos_xy.shape[0]
    com_meas = torch.cat((pos_xy, torch.full((b, 1), z, dtype=DTYPE)), dim=1)
    vcom_meas = torch.cat((vel_xy, torch.full((b, 1), vz, dtype=DTYPE)), dim=1)
    com_state = torch.zeros((b, 2, 3), dtype=DTYPE)
    com_state[:, :, 0] = pos_xy
    com_state[:, :, 1] = vel_xy
    if acc_xy is not None:
        com_state[:, :, 2] = torch.as_tensor(acc_xy, dtype=DTYPE)
    return com_meas, vcom_meas, com_state


class TestComputeDCMError(unittest.TestCase):
    def setUp(self):
        self.cp = CapturePoint(com_height=0.15)

    def assert_close(self, actual, expected, *, atol=1e-6):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-6)

    def test_01_initialize_omega_and_parameters(self):
        self.assertAlmostEqual(self.cp.omega, math.sqrt(9.81 / 0.15), places=10)
        self.assert_close(self.cp.step_gain, torch.tensor([0.4, 0.3]))
        self.assert_close(self.cp.step_limit, torch.tensor([0.04, 0.03]))
        self.assertEqual(self.cp.min_step_width, 0.12)

    def test_02_manual_reference_example(self):
        com = torch.tensor([[0.03, 0.01, 0.15]])
        vel = torch.tensor([[0.10, 0.00, 0.00]])
        state = torch.tensor([[[0.02, 0.08, 0.00], [0.01, 0.00, 0.00]]])
        d_meas, d_plan, err = self.cp.compute_dcm_error(com, vel, state)
        w = math.sqrt(9.81 / 0.15)
        self.assert_close(d_meas, torch.tensor([[0.03 + 0.10 / w, 0.01]]))
        self.assert_close(d_plan, torch.tensor([[0.02 + 0.08 / w, 0.01]]))
        self.assert_close(err, torch.tensor([[0.01 + 0.02 / w, 0.0]]))

    def test_03_shapes_dtype_device(self):
        com, vel, state = make_states(torch.zeros(5, 2), torch.zeros(5, 2))
        out = self.cp.compute_dcm_error(com, vel, state)
        for tensor in out:
            self.assertEqual(tensor.shape, (5, 2))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, 'cpu')

    def test_04_equal_measured_and_planned_states(self):
        pos = torch.tensor([[0.02, 0.01], [-0.01, 0.04], [0.12, -0.07]])
        vel = torch.tensor([[0.1, -0.1], [0.03, 0.05], [-0.2, 0.02]])
        com, vcom, state = make_states(pos, vel)
        d_meas, d_plan, err = self.cp.compute_dcm_error(com, vcom, state)
        self.assert_close(d_meas, d_plan)
        self.assert_close(err, torch.zeros_like(err))

    def test_05_position_error_only(self):
        com, vel, state = make_states([[0.02, -0.02]], [[0.1, -0.04]])
        com[:, 0] += 0.015
        com[:, 1] -= 0.012
        _, _, err = self.cp.compute_dcm_error(com, vel, state)
        self.assert_close(err, torch.tensor([[0.015, -0.012]]))

    def test_06_velocity_error_only(self):
        com, vel, state = make_states([[0.02, -0.02]], [[0.1, -0.04]])
        vel[:, :2] += torch.tensor([[0.081, -0.047]])
        _, _, err = self.cp.compute_dcm_error(com, vel, state)
        expected = torch.tensor([[0.081, -0.047]]) / self.cp.omega
        self.assert_close(err, expected)

    def test_07_sign_measured_minus_planned(self):
        com, vel, state = make_states([[0.03, 0.02]], [[0.1, -0.03]])
        com[:, :2] -= torch.tensor([[0.02, 0.01]])
        _, _, err = self.cp.compute_dcm_error(com, vel, state)
        self.assertTrue((err < 0).all().item())
        self.assert_close(err, torch.tensor([[-0.02, -0.01]]))

    def test_08_z_and_acceleration_not_used(self):
        com, vel, state = make_states([[0.01, -0.05]], [[0.06, 0.09]])
        baseline = self.cp.compute_dcm_error(com, vel, state)
        com[:, 2] = 3.0
        vel[:, 2] = -7.0
        state[:, :, 2] = torch.tensor([[100., -50.]])
        changed = self.cp.compute_dcm_error(com, vel, state)
        for x, y in zip(baseline, changed):
            self.assert_close(x, y)

    def test_09_xy_independence(self):
        com, vel, state = make_states([[0.01, 0.01]], [[0.02, 0.03]])
        old = self.cp.compute_dcm_error(com, vel, state)[2]
        com[:, 0] += 0.04
        vel[:, 0] += 0.05
        new = self.cp.compute_dcm_error(com, vel, state)[2]
        self.assert_close(old[:, 1], new[:, 1])
        self.assertGreater(abs((new-old)[0, 0].item()), 0.04)

    def test_10_translation_invariance_of_error(self):
        com, vel, state = make_states([[0.02, -0.03]], [[0.01, -0.05]])
        com[:, :2] += torch.tensor([[0.01, 0.02]])
        ref_err = self.cp.compute_dcm_error(com, vel, state)[2]
        shift = torch.tensor([[0.8, -0.5]])
        com[:, :2] += shift
        state[:, :, 0] += shift
        shifted_err = self.cp.compute_dcm_error(com, vel, state)[2]
        self.assert_close(ref_err, shifted_err)

    def test_11_batch_independence(self):
        g = torch.Generator().manual_seed(2026)
        b = 8
        pos = 0.1 * torch.randn(b, 2, generator=g)
        v = 0.2 * torch.randn(b, 2, generator=g)
        com, vel, state = make_states(pos, v)
        com[:, :2] += 0.02 * torch.randn(b, 2, generator=g)
        vel[:, :2] += 0.03 * torch.randn(b, 2, generator=g)
        grouped = self.cp.compute_dcm_error(com, vel, state)
        for i in range(b):
            single = self.cp.compute_dcm_error(com[i:i+1], vel[i:i+1], state[i:i+1])
            for a, s in zip(grouped, single):
                self.assert_close(a[i:i+1], s)

    def test_12_height_affects_velocity_component(self):
        com, vel, state = make_states([[0.02, 0.03]], [[0.0, 0.0]])
        vel[:, 0] += 0.1
        cp_short = CapturePoint(com_height=0.10)
        cp_tall = CapturePoint(com_height=0.25)
        error_short = cp_short.compute_dcm_error(com, vel, state)[2][0, 0]
        error_tall = cp_tall.compute_dcm_error(com, vel, state)[2][0, 0]
        self.assertGreater(error_tall.item(), error_short.item())
        self.assertAlmostEqual((error_tall/error_short).item(), math.sqrt(0.25/0.10), places=5)

    def test_13_no_input_mutation(self):
        com, vel, state = make_states([[0.02, 0.01]], [[0.1, -0.01]])
        originals = [x.clone() for x in (com, vel, state)]
        self.cp.compute_dcm_error(com, vel, state)
        for old, new in zip(originals, (com, vel, state)):
            self.assertTrue(torch.equal(old, new))

    def test_14_cuda_if_available(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA không khả dụng trong môi trường hiện tại')
        cp = CapturePoint(com_height=0.15, device='cuda')
        com, vel, state = make_states([[0.02, 0.01]], [[0.1, 0.02]])
        com, vel, state = com.cuda(), vel.cuda(), state.cuda()
        outputs = cp.compute_dcm_error(com, vel, state)
        for out in outputs:
            self.assertEqual(out.device.type, 'cuda')
            self.assertTrue(torch.isfinite(out).all())


def plot_dcm_demo():
    import matplotlib.pyplot as plt

    cp = CapturePoint(com_height=0.15)
    dt = 0.02
    t = torch.arange(0, 4.0 + 1e-6, dt)
    n = t.numel()

    # Dữ liệu tổng hợp có đạo hàm vận tốc nhất quán với vị trí.
    x_plan = 0.02 + 0.035 * t
    y_plan = 0.012 * torch.sin(2.0 * math.pi * 0.7 * t)
    vx_plan = torch.full_like(t, 0.035)
    vy_plan = 0.012 * (2.0 * math.pi * 0.7) * torch.cos(2.0 * math.pi * 0.7 * t)

    x_offset = 0.010 * torch.sin(2.0 * math.pi * 0.35 * t)
    y_offset = -0.008 * torch.cos(2.0 * math.pi * 0.40 * t)
    vx_offset = 0.010 * (2.0 * math.pi * 0.35) * torch.cos(2.0 * math.pi * 0.35 * t)
    vy_offset = 0.008 * (2.0 * math.pi * 0.40) * torch.sin(2.0 * math.pi * 0.40 * t)

    state = torch.zeros(n, 2, 3)
    state[:, 0, 0], state[:, 1, 0] = x_plan, y_plan
    state[:, 0, 1], state[:, 1, 1] = vx_plan, vy_plan
    com_meas = torch.stack([x_plan + x_offset, y_plan + y_offset, torch.full_like(t, 0.15)], dim=-1)
    vcom_meas = torch.stack([vx_plan + vx_offset, vy_plan + vy_offset, torch.zeros_like(t)], dim=-1)
    d_meas, d_plan, err = cp.compute_dcm_error(com_meas, vcom_meas, state)

    ts = t.numpy()
    d_meas, d_plan, err = [v.numpy() for v in (d_meas, d_plan, err)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, idx, letter in [(axes[0, 0], 0, 'X'), (axes[0, 1], 1, 'Y')]:
        axis.plot(ts, d_meas[:, idx], label='DCM measured (synthetic)', linewidth=2)
        axis.plot(ts, d_plan[:, idx], '--', label='DCM planned', linewidth=1.7)
        axis.set_ylabel(f'DCM {letter} (m)')
        axis.set_title(f'DCM: measured vs planned — {letter}')
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best')
    for axis, idx, letter in [(axes[1, 0], 0, 'X'), (axes[1, 1], 1, 'Y')]:
        axis.plot(ts, err[:, idx], linewidth=2)
        axis.axhline(0, color='black', linewidth=0.8)
        axis.set_xlabel('Time (s)')
        axis.set_ylabel(f'DCM error {letter} (m)')
        axis.set_title(f'Error = measured - planned — {letter}')
        axis.grid(True, alpha=0.3)
    fig.suptitle('CapturePoint.compute_dcm_error() — synthetic test', fontsize=13)
    fig.tight_layout()
    print(f'\nĐồ thị minh họa: {n} mẫu, dt={dt:.3f} s; '
          f'|e_x|_max={abs(err[:, 0]).max():.6f} m, '
          f'|e_y|_max={abs(err[:, 1]).max():.6f} m')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Unit tests for CapturePoint.compute_dcm_error')
    parser.add_argument('--no-plot', action='store_true', help='Chỉ chạy test, không mở biểu đồ')
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestComputeDCMError)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if not args.no_plot:
        plot_dcm_demo()


if __name__ == '__main__':
    main()
