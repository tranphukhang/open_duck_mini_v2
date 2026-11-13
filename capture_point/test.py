"""Kiểm thử zmp_reference() và trực quan hóa quỹ đạo ZMP.

Đặt file này cùng thư mục với:
    zmp_generator.py, contact_scheduler.py, footstep_planner.py

Chạy:
    python test_zmp_generator_reference.py

Chỉ in kết quả kiểm thử và HIỂN THỊ đồ thị; không xuất CSV, không lưu ảnh.
"""

import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt

from zmp_generator import ZMPGenerator
from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner


DT = 0.02
PERIOD = 0.4
SHIFT = 0.4
DS_RATIO = 0.1
DS_DURATION = PERIOD * DS_RATIO  # 0.04 s
N = 80


def create_setup(num_envs=2, zmp_offset_x=0.005, zmp_offset_y_out=0.007):
    """Khởi tạo 3 module với hai cấu hình chân có hình học khác nhau."""
    scheduler = ContactScheduler(
        dt=DT, period_phase=PERIOD, shift=SHIFT,
        double_support=DS_RATIO, num_envs=num_envs, device="cpu"
    )
    planner = FootstepPlanner(
        num_envs=num_envs, period_phase=PERIOD,
        num_footsteps=14, device="cpu"
    )
    generator = ZMPGenerator(
        zmp_offset_x=zmp_offset_x, zmp_offset_y_out=zmp_offset_y_out
    )

    # Môi trường 0: hai chân đối xứng; môi trường 1: chân lệch cả X, Y, Z.
    initial = torch.tensor([
        [[0.00,  0.090, 0.00], [0.00, -0.090, 0.00]],
        [[0.03,  0.085, 0.01], [-0.02, -0.080, 0.01]],
    ], dtype=torch.float32)
    if num_envs > 2:
        initial = torch.cat([initial, initial[:1].repeat(num_envs - 2, 1, 1)], dim=0)
    feet0 = initial[:num_envs].clone()
    planner.reset(torch.arange(num_envs), feet0)

    cmd = torch.tensor([[0.20, 0.00], [0.12, -0.02]], dtype=torch.float32)
    if num_envs > 2:
        cmd = torch.cat([cmd, cmd[:1].repeat(num_envs - 2, 1)], dim=0)
    cmd = cmd[:num_envs]
    return scheduler, planner, generator, cmd


def sample_reference(scheduler, planner, generator, cmd, times):
    """Tạo ZMP theo j_future và local_future từ ContactScheduler."""
    times = torch.as_tensor(times, dtype=torch.float32)
    if times.ndim == 1:
        times = times[None, :].expand(planner.feet0.shape[0], -1)
    if times.shape[0] != planner.feet0.shape[0]:
        raise ValueError("Số môi trường của times không khớp với planner")

    current_time = torch.zeros(times.shape[0], dtype=times.dtype)
    j_now = scheduler.step_index(current_time)
    footsteps, j_base = planner.footstep_sequence(j_now, cmd)
    j_future = scheduler.step_index(times)
    local_future = scheduler.local_time_phase(times, j_future)

    zref = generator.zmp_reference(
        footsteps, planner.feet0, j_base, j_future, local_future,
        SHIFT, DS_DURATION
    )
    return zref, footsteps, j_base, j_future, local_future


def at(t, zref, time_grid, env=0):
    """Lấy mẫu đúng thời gian; chỉ dùng trong test."""
    idx = torch.argmin(torch.abs(time_grid - float(t))).item()
    assert abs(float(time_grid[idx]) - t) < 1e-5
    return zref[env, idx]


class TestZMPReference(unittest.TestCase):
    def setUp(self):
        self.sch, self.pl, self.zmp, self.cmd = create_setup()
        self.times = torch.arange(0, 81, dtype=torch.float32) * DT  # 0..1.60 s
        (self.ref, self.steps, self.jbase,
         self.jfuture, self.local) = sample_reference(
            self.sch, self.pl, self.zmp, self.cmd, self.times
        )

    def close(self, actual, expected, atol=1e-6):
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=actual.dtype),
            atol=atol, rtol=0.0
        )

    def test_01_shape_dtype_and_all_finite(self):
        self.assertEqual(tuple(self.ref.shape), (2, 81, 2))
        self.assertEqual(self.ref.dtype, torch.float32)
        self.assertTrue(torch.isfinite(self.ref).all().item())

    def test_02_shift_start_mid_end(self):
        # Chân L: (0.005, +0.097); chân R: (0.005, -0.097).
        self.close(at(0.0, self.ref, self.times), [0.005, 0.0])
        self.close(at(0.20, self.ref, self.times), [0.005, 0.0485])
        # 0.40 s: vừa chuyển sang bước j=0, ZMP tại chân trái.
        self.close(at(0.40, self.ref, self.times), [0.005, 0.097])

    def test_03_shift_monotonic_y(self):
        mask = self.jfuture[0] < 0
        shift_y = self.ref[0, mask, 1]
        self.assertTrue(torch.all(torch.diff(shift_y) >= -1e-7).item())
        self.assertTrue(torch.all(shift_y >= -1e-7).item())

    def test_04_first_step_no_extra_ds(self):
        self.close(at(0.40, self.ref, self.times), [0.005, 0.097])
        self.close(at(0.42, self.ref, self.times), [0.005, 0.097])
        self.close(at(0.58, self.ref, self.times), [0.005, 0.097])

    def test_05_ds_j1_left_to_right(self):
        # P1 (chân phải) tại X=0.2 * 0.2=0.04 m.
        self.close(at(0.60, self.ref, self.times), [0.005, 0.097])
        self.close(at(0.62, self.ref, self.times), [0.025, 0.0])
        self.close(at(0.64, self.ref, self.times), [0.045, -0.097])

    def test_06_ds_j2_right_to_left(self):
        # P2 (chân trái) tại X=0.08 m.
        self.close(at(0.80, self.ref, self.times), [0.045, -0.097])
        self.close(at(0.82, self.ref, self.times), [0.065, 0.0])
        self.close(at(0.84, self.ref, self.times), [0.085, 0.097])

    def test_07_single_support_remains_fixed(self):
        self.close(at(0.50, self.ref, self.times), [0.005, 0.097])
        self.close(at(0.70, self.ref, self.times), [0.045, -0.097])
        self.close(at(0.90, self.ref, self.times), [0.085, 0.097])

    def test_08_reference_selects_correct_subfunction(self):
        z_shift = self.zmp.zmp_shift(self.pl.feet0, self.local, SHIFT)
        z_walk = self.zmp.zmp_walk(
            self.steps, self.jbase, self.jfuture, self.local, DS_DURATION
        )
        expected = torch.where((self.jfuture < 0)[..., None], z_shift, z_walk)
        torch.testing.assert_close(self.ref, expected, atol=0, rtol=0)
        self.assertTrue((self.jfuture[0] < 0).any().item())
        self.assertTrue((self.jfuture[0] >= 0).any().item())

    def test_09_multienvironment_independence(self):
        # Ở môi trường 1, ZMP trước Shift là trung điểm ZMP hai chân.
        left = torch.tensor([0.035, 0.092])
        right = torch.tensor([-0.015, -0.087])
        midpoint = (left + right) / 2
        torch.testing.assert_close(at(0.0, self.ref, self.times, 1), midpoint, atol=1e-6, rtol=0)
        torch.testing.assert_close(at(0.40, self.ref, self.times, 1), left, atol=1e-6, rtol=0)
        self.assertFalse(torch.allclose(self.ref[0], self.ref[1]))

    def test_10_zero_and_custom_offsets(self):
        # Offset = 0: ZMP Shift bắt đầu ở trung điểm hai foot sites.
        sch, pl, g, cmd = create_setup(1, zmp_offset_x=0.0, zmp_offset_y_out=0.0)
        result, *_ = sample_reference(sch, pl, g, cmd, [0.0, 0.4])
        self.close(result[0, 0], [0.0, 0.0])
        self.close(result[0, 1], [0.0, 0.09])

        # Offset tùy chỉnh phản ánh trực tiếp ở ZMP tham chiếu.
        sch2, pl2, g2, cmd2 = create_setup(1, 0.012, 0.020)
        result2, *_ = sample_reference(sch2, pl2, g2, cmd2, [0.0, 0.4])
        self.close(result2[0, 0], [0.012, 0.0])
        self.close(result2[0, 1], [0.012, 0.11])

    def test_11_no_input_mutation(self):
        footsteps0 = self.steps.clone()
        feet0 = self.pl.feet0.clone()
        base0 = self.jbase.clone()
        future0 = self.jfuture.clone()
        local0 = self.local.clone()
        _ = self.zmp.zmp_reference(
            self.steps, self.pl.feet0, self.jbase,
            self.jfuture, self.local, SHIFT, DS_DURATION
        )
        for current, original in (
            (self.steps, footsteps0), (self.pl.feet0, feet0),
            (self.jbase, base0), (self.jfuture, future0), (self.local, local0)
        ):
            self.assertTrue(torch.equal(current, original))

    def test_12_mpc_horizon_N80(self):
        future_time = DT * torch.arange(1, N + 1, dtype=torch.float32)
        ref, *_ = sample_reference(self.sch, self.pl, self.zmp, self.cmd, future_time)
        self.assertEqual(tuple(ref.shape), (2, 80, 2))
        self.close(ref[0, 0], [0.005, 0.097 * (3 * 0.05**2 - 2 * 0.05**3)])
        self.close(ref[0, -1], [0.205, -0.097])  # t=1.60 vừa bắt đầu DS j=6, ZMP vẫn ở P5 (chân phải)

    def test_13_nonzero_jbase(self):
        # Khởi tạo footstep planner qua nhiều bước, giống trình tự của walker.
        sch, pl, g, cmd = create_setup(1)
        last_step = torch.tensor([-3], dtype=torch.long)
        for tick in range(51):  # t=0...1.00 s, qua j=3
            t = torch.tensor([tick * DT], dtype=torch.float32)
            j = sch.step_index(t)
            new = sch.new_step_event(j, last_step)
            stance = sch.stance_feet(j)
            pl.update_new_step(j, new, stance, cmd)
            last_step = torch.where(j >= 0, j, torch.full_like(j, -1))
        now = torch.tensor([1.02], dtype=torch.float32)
        j_now = sch.step_index(now)
        self.assertEqual(int(j_now.item()), 3)
        steps, base = pl.footstep_sequence(j_now, cmd)
        self.assertEqual(int(base.item()), 3)
        horizon_time = now[:, None] + DT * torch.arange(1, 81)[None, :]
        future_j = sch.step_index(horizon_time)
        future_local = sch.local_time_phase(horizon_time, future_j)
        ref = g.zmp_reference(steps, pl.feet0, base, future_j,
                              future_local, SHIFT, DS_DURATION)
        self.assertEqual(tuple(ref.shape), (1, 80, 2))
        # t=1.04 thuộc j=3, kết thúc DS => stance là chân phải P0.
        expected = g.zmp_point(steps[:, 1], torch.tensor([1]))
        torch.testing.assert_close(ref[:, 0], expected, atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(ref).all().item())


def plot_reference():
    """Vẽ quỹ đạo tham chiếu, không ghi file."""
    scheduler, planner, generator, cmd = create_setup(num_envs=2)
    times = torch.arange(0, 81, dtype=torch.float32) * DT
    ref, footsteps, _, j_future, _ = sample_reference(
        scheduler, planner, generator, cmd, times
    )
    t = times.numpy()
    z0 = ref[0].detach().numpy()
    z1 = ref[1].detach().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("ZMP Reference: Shift -> Double Support -> Single Support", fontsize=13)

    ax = axes[0, 0]
    ax.plot(t, z0[:, 0], label="ZMP X - env 0", linewidth=2)
    ax.plot(t, z1[:, 0], label="ZMP X - env 1", linestyle="--", linewidth=1.5)
    ax.set(xlabel="Time (s)", ylabel="ZMP X (m)", title="ZMP X theo thời gian")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(t, z0[:, 1], label="ZMP Y - env 0", linewidth=2)
    ax.plot(t, z1[:, 1], label="ZMP Y - env 1", linestyle="--", linewidth=1.5)
    ax.set(xlabel="Time (s)", ylabel="ZMP Y (m)", title="ZMP Y theo thời gian")
    ax.legend(fontsize=9)

    for ax in axes[0]:
        ax.axvspan(0.0, SHIFT, color="gray", alpha=0.12, label="Shift")
        for start in np.arange(SHIFT + PERIOD / 2, t[-1] + 1e-9, PERIOD / 2):
            ax.axvspan(start, min(start + DS_DURATION, float(t[-1])),
                       color="orange", alpha=0.12)
        ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(z0[:, 0], z0[:, 1], marker=".", markersize=3, label="ZMP ref (env 0)")
    left_pt = generator.zmp_point(planner.feet0[0, 0], torch.tensor(0)).numpy()
    right_pt = generator.zmp_point(planner.feet0[0, 1], torch.tensor(1)).numpy()
    ax.scatter(left_pt[0], left_pt[1], marker="^", s=90, label="Initial left ZMP")
    ax.scatter(right_pt[0], right_pt[1], marker="v", s=90, label="Initial right ZMP")
    ax.scatter(z0[0, 0], z0[0, 1], marker="o", s=55, label="Start (t=0)")
    ax.scatter(z0[-1, 0], z0[-1, 1], marker="x", s=80, label="End (t=1.6)")
    ax.set(xlabel="ZMP X (m)", ylabel="ZMP Y (m)", title="Quỹ đạo ZMP trên mặt phẳng X-Y")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    mask = (t >= 0.56 - 1e-6) & (t <= 0.86 + 1e-6)
    ax.plot(t[mask], z0[mask, 1], marker="o", label="ZMP Y - env 0")
    for x, label in [(0.60, "DS j=1"), (0.64, "SS j=1"),
                     (0.80, "DS j=2"), (0.84, "SS j=2")]:
        ax.axvline(x, linestyle="--", alpha=0.35)
        ax.annotate(label, (x, 0.0), rotation=90, fontsize=8, va="center", ha="right")
    ax.set(xlabel="Time (s)", ylabel="ZMP Y (m)", title="Phóng to chuyển tiếp DS và SS")
    ax.grid(True, alpha=0.25)

    plt.show()  # Chỉ hiển thị, không savefig()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestZMPReference)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("\nTất cả test PASS. Mở đồ thị ZMP...\n")
        plot_reference()
    else:
        print("\nCó test FAIL: vui lòng kiểm tra trước khi vẽ đồ thị.")
        raise SystemExit(1)
