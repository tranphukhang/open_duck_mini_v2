"""Kiểm thử zmp_shift() và hiển thị quỹ đạo ZMP, không lưu dữ liệu/ảnh.

Chạy tại thư mục capture_point:
    python test_zmp_generator_shift.py

Yêu cầu: torch, numpy, matplotlib.
"""

from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

# Cho phép chạy file từ bất kỳ working directory nào.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zmp_generator import ZMPGenerator

DT = 0.02
SHIFT_DURATION = 0.4
TOL = 1e-6


def make_feet(dtype=torch.float32):
    """Hai môi trường có vị trí đặt chân khác nhau, đơn vị mét.

    Môi trường 1: hai chân đối xứng theo Y, cùng tọa độ X.
    Môi trường 2: hai chân lệch nhau cả X và Y.
    Trục 1: [chân trái, chân phải].
    """
    return torch.tensor(
        [
            [[0.00, 0.09, 0.00], [0.00, -0.09, 0.00]],
            [[0.04, 0.08, 0.00], [-0.02, -0.10, 0.00]],
        ],
        dtype=dtype,
    )


def reference_points(generator, feet0):
    n = feet0.shape[0]
    left = torch.zeros(n, dtype=torch.long, device=feet0.device)
    right = torch.ones(n, dtype=torch.long, device=feet0.device)
    z_left = generator.zmp_point(feet0[:, 0], left)
    z_right = generator.zmp_point(feet0[:, 1], right)
    return (z_left + z_right) / 2.0, z_left, z_right


def check(label, condition, detail=None):
    if not condition:
        raise AssertionError(f"{label} FAILED" + (f": {detail}" if detail else ""))
    print(f"[PASS] {label}")


def run_tests():
    generator = ZMPGenerator()  # Offset mặc định theo tác giả
    feet0 = make_feet()
    times = torch.arange(21, dtype=torch.float32) * DT  # 0, 0.02, ..., 0.40
    local_future = times.unsqueeze(0).expand(feet0.shape[0], -1).clone()

    feet_before = feet0.clone()
    times_before = local_future.clone()
    zmp = generator.zmp_shift(feet0, local_future, SHIFT_DURATION)
    z_mid, z_left, z_right = reference_points(generator, feet0)

    check("1. Kích thước (B, N, 2) và dtype", zmp.shape == (2, 21, 2) and zmp.dtype == feet0.dtype)
    check("2. Tại t=0: ZMP ở trung điểm", torch.allclose(zmp[:, 0], z_mid, atol=TOL))
    check("3. Tại t=T/2: ZMP ở giữa trung điểm và chân trái",
          torch.allclose(zmp[:, 10], (z_mid + z_left) / 2, atol=TOL))
    check("4. Tại t=T: ZMP tới chân trái", torch.allclose(zmp[:, -1], z_left, atol=TOL))

    s = (local_future / SHIFT_DURATION).clamp(0, 1)
    sigma = 3 * s**2 - 2 * s**3
    manual = (1 - sigma[..., None]) * z_mid[:, None, :] + sigma[..., None] * z_left[:, None, :]
    check("5. Cả quỹ đạo khớp công thức smoothstep", torch.allclose(zmp, manual, atol=TOL))

    extra_times = torch.tensor([[-0.10, 0.00, 0.40, 0.50]]).expand(feet0.shape[0], -1)
    z_extra = generator.zmp_shift(feet0, extra_times, SHIFT_DURATION)
    check("6. Clamp ngoài khoảng Shift",
          torch.allclose(z_extra[:, 0], z_mid, atol=TOL)
          and torch.allclose(z_extra[:, -1], z_left, atol=TOL))

    custom_generator = ZMPGenerator(zmp_offset_x=0.012, zmp_offset_y_out=0.004)
    z_custom = custom_generator.zmp_shift(feet0, local_future, SHIFT_DURATION)
    custom_mid, custom_left, _ = reference_points(custom_generator, feet0)
    check("7. Thay đổi hai offset ZMP",
          torch.allclose(z_custom[:, 0], custom_mid, atol=TOL)
          and torch.allclose(z_custom[:, -1], custom_left, atol=TOL))

    check("8. Không sửa tensor đầu vào",
          torch.equal(feet0, feet_before) and torch.equal(local_future, times_before))

    horizon_time = torch.arange(1, 81, dtype=torch.float32)[None, :] * DT
    z_horizon = generator.zmp_shift(feet0[:1], horizon_time, SHIFT_DURATION)
    check("9. Chấp nhận horizon N=80", z_horizon.shape == (1, 80, 2)
          and torch.allclose(z_horizon[0, -1], z_left[0], atol=TOL))

    # Tính liên tục, đơn điệu của phép nội suy tại các mẫu theo thời gian.
    delta = (zmp - z_mid[:, None, :])
    move_vec = z_left - z_mid
    progress = (delta * move_vec[:, None, :]).sum(-1) / move_vec.square().sum(-1)[:, None]
    check("10. Nội suy tiến một chiều từ trung điểm đến chân trái",
          torch.all(torch.diff(progress, dim=1) >= -TOL).item()
          and torch.allclose(progress[:, 0], torch.zeros(2), atol=TOL)
          and torch.allclose(progress[:, -1], torch.ones(2), atol=TOL))

    # Kiểm tra đạo hàm bằng autograd: vận tốc ZMP về 0 tại đầu và cuối Shift.
    time_grad = torch.tensor([[0.0, SHIFT_DURATION]], dtype=torch.float64, requires_grad=True)
    z_grad = generator.zmp_shift(feet0[:1].to(torch.float64), time_grad, SHIFT_DURATION)
    derivatives = torch.autograd.grad(z_grad[..., 1].sum(), time_grad)[0]
    check("11. Đạo hàm đầu và cuối Shift xấp xỉ 0",
          torch.allclose(derivatives, torch.zeros_like(derivatives), atol=1e-9))

    print("\nKẾT QUẢ: 11/11 TEST PASS")
    return times.numpy(), feet0.numpy(), zmp.detach().numpy(), z_mid.numpy(), z_left.numpy(), z_right.numpy()


def plot_results(times, feet0, zmp, z_mid, z_left, z_right):
    titles = ["Môi trường 1: hai chân đối xứng", "Môi trường 2: hai chân lệch X/Y"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for b in range(2):
        ax_time = axes[b, 0]
        ax_xy = axes[b, 1]

        ax_time.plot(times, zmp[b, :, 0], "-o", markersize=3, label="ZMP X")
        ax_time.plot(times, zmp[b, :, 1], "-o", markersize=3, label="ZMP Y")
        ax_time.axvline(SHIFT_DURATION, linestyle="--", linewidth=1, label="Kết thúc Shift")
        ax_time.set(title=f"{titles[b]} — ZMP theo thời gian", xlabel="Thời gian (s)", ylabel="ZMP (m)")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()

        ax_xy.plot(zmp[b, :, 0], zmp[b, :, 1], "-o", markersize=4, label="Quỹ đạo ZMP")
        ax_xy.scatter(*z_mid[b], marker="s", s=70, label="Trung điểm ban đầu", zorder=3)
        ax_xy.scatter(*z_left[b], marker="^", s=90, label="ZMP chân trái", zorder=3)
        ax_xy.scatter(*z_right[b], marker="v", s=90, label="ZMP chân phải", zorder=3)
        ax_xy.scatter(*feet0[b, 0, :2], marker="x", s=80, label="Foot site trái", zorder=3)
        ax_xy.scatter(*feet0[b, 1, :2], marker="x", s=80, label="Foot site phải", zorder=3)
        ax_xy.set(title=f"{titles[b]} — quỹ đạo X–Y", xlabel="X (m)", ylabel="Y (m)")
        ax_xy.axis("equal")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.legend(fontsize=8, loc="best")

    fig.suptitle("Kiểm thử ZMPGenerator.zmp_shift() — Shift 0.4 s, dt = 0.02 s", fontsize=13)
    # Chỉ hiển thị trên màn hình; KHÔNG lưu CSV hay ảnh.
    plt.show()


if __name__ == "__main__":
    values = run_tests()
    plot_results(*values)
