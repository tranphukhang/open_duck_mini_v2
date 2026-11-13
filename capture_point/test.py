"""Kiem thu + ve do thi ZMPGenerator.smoothstep().

Dat file nay o MOT trong hai vi tri:
  A. open_duck_mini_v2/capture_point/test_zmp_generator_smoothstep.py
  B. open_duck_mini_v2/test_zmp_generator_smoothstep.py

Chay:
    python test_zmp_generator_smoothstep.py

Yeu cau: torch, matplotlib. Khong can pytest, MuJoCo hay Isaac Lab.
Chi hien thi do thi; KHONG luu CSV, anh hoac file ket qua.
"""

from pathlib import Path
import importlib.util
import unittest

import torch


def load_zmp_generator():
    """Tim zmp_generator.py khi test nam o thu muc goc hoac capture_point/."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "zmp_generator.py",
        here / "capture_point" / "zmp_generator.py",
        here.parent / "capture_point" / "zmp_generator.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("zmp_generator_local", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"Dang kiem thu: {path}", flush=True)
            return module.ZMPGenerator
    raise FileNotFoundError(
        "Khong tim thay zmp_generator.py. Hay dat file test trong thu muc "
        "capture_point/ (cung cap zmp_generator.py) hoac tai thu muc goc repo "
        "open_duck_mini_v2/.\nDa tim tai:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


ZMPGenerator = load_zmp_generator()


class TestSmoothstep(unittest.TestCase):
    def test_01_gia_tri_bien(self):
        s = torch.tensor([0.0, 1.0], dtype=torch.float64)
        expected = torch.tensor([0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(ZMPGenerator.smoothstep(s), expected)

    def test_02_cac_diem_dac_trung(self):
        s = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        expected = torch.tensor([0.0, 0.15625, 0.5, 0.84375, 1.0], dtype=torch.float64)
        torch.testing.assert_close(ZMPGenerator.smoothstep(s), expected, rtol=0, atol=1e-12)

    def test_03_clamp_ngoai_khoang(self):
        s = torch.tensor([-2.0, -0.1, 0.0, 1.0, 1.1, 2.0])
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(ZMPGenerator.smoothstep(s), expected)

    def test_04_tensor_scalar(self):
        result = ZMPGenerator.smoothstep(torch.tensor(0.5))
        self.assertEqual(result.ndim, 0)
        self.assertAlmostEqual(result.item(), 0.5, places=7)

    def test_05_tensor_batch_horizon(self):
        # Kiem thu dang du lieu se dung o MPC: B=3, N=80.
        s = torch.linspace(-0.2, 1.2, 80, dtype=torch.float64).repeat(3, 1)
        result = ZMPGenerator.smoothstep(s)
        s_clip = s.clamp(0.0, 1.0)
        expected = 3.0 * s_clip**2 - 2.0 * s_clip**3
        self.assertEqual(tuple(result.shape), (3, 80))
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)

    def test_06_don_dieu_va_gioi_han(self):
        s = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
        y = ZMPGenerator.smoothstep(s)
        self.assertTrue(bool(torch.all(y[1:] >= y[:-1] - 1e-12)))
        self.assertTrue(bool(torch.all((y >= 0.0) & (y <= 1.0))))

    def test_07_tinh_doi_xung(self):
        s = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        y = ZMPGenerator.smoothstep(s) + ZMPGenerator.smoothstep(1.0 - s)
        torch.testing.assert_close(y, torch.ones_like(y), rtol=0, atol=1e-12)

    def test_08_dao_ham_hai_dau(self):
        s = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
        y = ZMPGenerator.smoothstep(s)
        grad = torch.autograd.grad(y.sum(), s)[0]
        expected = torch.tensor([0.0, 1.5, 0.0], dtype=torch.float64)
        torch.testing.assert_close(grad, expected, rtol=0, atol=1e-12)

    def test_09_khong_thay_doi_input(self):
        s = torch.tensor([[-0.5, 0.3], [0.8, 1.5]])
        before = s.clone()
        _ = ZMPGenerator.smoothstep(s)
        torch.testing.assert_close(s, before)

    @unittest.skipUnless(torch.cuda.is_available(), "May khong co CUDA")
    def test_10_cuda(self):
        s = torch.tensor([0.0, 0.5, 1.0], device="cuda")
        result = ZMPGenerator.smoothstep(s)
        self.assertEqual(result.device.type, "cuda")
        torch.testing.assert_close(result.cpu(), torch.tensor([0.0, 0.5, 1.0]))


def plot_results():
    """Ve ba do thi de quan sat; khong ghi file."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (1) Do thi noi suy va clamp o ngoai [0, 1].
    s = torch.linspace(-0.2, 1.2, 500, dtype=torch.float64)
    y = ZMPGenerator.smoothstep(s)
    axes[0].plot(s.numpy(), y.numpy(), linewidth=2, label="Smoothstep")
    axes[0].plot([0, 1], [0, 1], "--", label="Linear interpolation")
    axes[0].axvline(0, linestyle=":", alpha=0.7)
    axes[0].axvline(1, linestyle=":", alpha=0.7)
    axes[0].set_title("1. Smoothstep & clamp")
    axes[0].set_xlabel("s - normalized time")
    axes[0].set_ylabel("sigma(s)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # (2) Dao ham: sigma'(s) = 6s - 6s^2; tai hai dau bang 0.
    s2 = torch.linspace(0, 1, 500, dtype=torch.float64)
    slope = 6 * s2 * (1 - s2)
    axes[1].plot(s2.numpy(), slope.numpy(), linewidth=2, label="d sigma / ds")
    axes[1].scatter([0, 1], [0, 0], zorder=3, label="Zero slope endpoints")
    axes[1].set_title("2. Derivative of smoothstep")
    axes[1].set_xlabel("s - normalized time")
    axes[1].set_ylabel("d sigma / ds")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # (3) Cung cong thuc, nhung so mau khac nhau vi T_shift != T_DSP.
    dt = 0.02
    for label, duration in (("Shift (0.4 s)", 0.4), ("DSP (0.04 s)", 0.04)):
        num_intervals = round(duration / dt)
        time = torch.arange(num_intervals + 1, dtype=torch.float64) * dt
        phase = time / duration
        y_phase = ZMPGenerator.smoothstep(phase)
        axes[2].plot(phase.numpy(), y_phase.numpy(), "o-", label=f"{label}: {len(time)} samples")
    axes[2].set_title("3. MPC sampling, dt = 0.02 s")
    axes[2].set_xlabel("t / T_phase")
    axes[2].set_ylabel("sigma(t / T_phase)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle("Test ZMPGenerator.smoothstep()", fontsize=13)
    fig.tight_layout()
    # Chi hien thi trong cua so Matplotlib; KHONG goi savefig/to_csv.
    plt.show()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSmoothstep)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("\nTests PASS. Mo cua so do thi Matplotlib...", flush=True)
    plot_results()
