"""Kiem thu LIPMMPC.build_state_prediction() theo ban GitHub hien tai.

Dat file nay cung thu muc voi lipm_mpc.py, sau do chay:
    python test_lipm_mpc_build_state_prediction.py

Can: torch, matplotlib. Chuong trinh khong ghi CSV, khong luu anh.
Mac dinh HIEN THI do thi qua plt.show(); dung --no-plot de chi chay test.

Mo hinh CoM 1D: x = [vi tri, van toc, gia toc]; u = jerk.
Test ca batch B moi truong va 2 truc X,Y; chi la kiem thu du doan,
CHUA phai toi uu jerk cua MPC.
"""

import argparse
from pathlib import Path
import sys

import torch

# Tai file test ve va dat cung thu muc voi lipm_mpc.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipm_mpc import LIPMMPC


H = 0.15  # Chi la gia tri minh hoa, khong phai chieu cao CoM cua robot.
DT = 0.02
N = 80


def make_mpc(dt=DT, horizon=N, com_height=H, device="cpu", **kwargs):
    mpc = LIPMMPC(
        com_height=com_height, dt=dt, horizon=horizon, device=device, **kwargs
    )
    mpc.build_model()
    mpc.build_state_prediction()
    return mpc


def close(actual, expected, message, atol=1e-5, rtol=2e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=message)


def condensed_predict(mpc, x0, jerk):
    """x0 (B,2,3), jerk (B,2,N) -> states (B,2,N,3)."""
    return (
        torch.einsum("kij,baj->baki", mpc.Phi, x0)
        + torch.einsum("kin,ban->baki", mpc.Gamma, jerk)
    )


def recursive_predict(mpc, x0, jerk):
    """Du doan doc lap bang cach chay A x + B u tung chu ky."""
    x = x0.clone()
    sequence = []
    for k in range(jerk.shape[-1]):
        x = x @ mpc.A.T + jerk[..., k, None] * mpc.B
        sequence.append(x)
    return torch.stack(sequence, dim=-2)


def example_signals(batch=3, horizon=N, device="cpu"):
    torch.manual_seed(20261113)
    x0 = torch.tensor(
        [
            [[0.00, 0.12, 0.00], [0.04, 0.00, 0.00]],
            [[0.10, 0.03, -0.08], [-0.05, 0.02, 0.12]],
            [[-0.02, -0.06, 0.05], [0.09, -0.03, -0.10]],
        ],
        dtype=torch.float32,
        device=device,
    )[:batch]
    t = DT * torch.arange(horizon, device=device, dtype=torch.float32)
    u = torch.empty(batch, 2, horizon, device=device)
    for b in range(batch):
        u[b, 0] = 0.6 * torch.sin(2.1 * t + 0.2 * b) + 0.12
        u[b, 1] = -0.4 * torch.cos(3.4 * t + 0.3 * b) + 0.08
    return x0, u


def test_shapes_types():
    mpc = make_mpc()
    assert mpc.Phi.shape == (N, 3, 3)
    assert mpc.Gamma.shape == (N, 3, N)
    assert mpc.Phi.dtype == torch.float32
    assert mpc.Gamma.dtype == torch.float32
    assert mpc.Phi.device.type == "cpu"
    assert mpc.Gamma.device.type == "cpu"
    assert torch.isfinite(mpc.Phi).all() and torch.isfinite(mpc.Gamma).all()


def test_phi_powers():
    mpc = make_mpc()
    for k in (0, 1, 2, 9, 39, 79):
        reference = torch.linalg.matrix_power(mpc.A, k + 1)
        close(mpc.Phi[k], reference, f"Phi[{k}] phai bang A^{k+1}", atol=5e-6)
    close(mpc.Phi[0], mpc.A, "Phi[0] != A")


def test_gamma_causality_and_columns():
    mpc = make_mpc()
    for k in range(N):
        close(mpc.Gamma[k, :, k], mpc.B, f"Gamma[{k},:,{k}] != B")
        if k + 1 < N:
            assert torch.count_nonzero(mpc.Gamma[k, :, k+1:]) == 0, (
                f"Gia tri Gamma o tuong lai phai bang 0, k={k}"
            )
    for i, j in ((0, 0), (1, 0), (2, 0), (6, 3), (25, 5), (79, 55)):
        expected = torch.linalg.matrix_power(mpc.A, i-j) @ mpc.B
        close(mpc.Gamma[i, :, j], expected, f"Gamma[{i},:,{j}] sai")


def test_recurrence_relations():
    mpc = make_mpc()
    for k in range(1, N):
        close(mpc.Phi[k], mpc.A @ mpc.Phi[k-1], f"Phi recurrence k={k}")
        close(
            mpc.Gamma[k, :, :k],
            mpc.A @ mpc.Gamma[k-1, :, :k],
            f"Gamma recurrence k={k}",
        )


def test_small_horizons():
    for n in (1, 2, 3, 5):
        mpc = make_mpc(horizon=n)
        assert mpc.Phi.shape == (n, 3, 3)
        assert mpc.Gamma.shape == (n, 3, n)
        close(mpc.Gamma[0, :, 0], mpc.B, f"N={n}, Gamma step dau sai")
    mpc = make_mpc(horizon=2)
    close(mpc.Phi[1], mpc.A @ mpc.A, "N=2: Phi second sai")
    close(mpc.Gamma[1, :, 0], mpc.A @ mpc.B, "N=2: Gamma[1,:,0] sai")
    close(mpc.Gamma[1, :, 1], mpc.B, "N=2: Gamma[1,:,1] sai")


def test_zero_jerk():
    mpc = make_mpc()
    x0, jerk = example_signals()
    jerk.zero_()
    predicted = condensed_predict(mpc, x0, jerk)
    reference = torch.einsum("kij,baj->baki", mpc.Phi, x0)
    close(predicted, reference, "Zero jerk != Phi @ x0")
    close(predicted, recursive_predict(mpc, x0, jerk), "Zero jerk sequential sai")


def test_full_horizon_batch_xy():
    mpc = make_mpc()
    x0, jerk = example_signals()
    condensed = condensed_predict(mpc, x0, jerk)
    sequential = recursive_predict(mpc, x0, jerk)
    assert condensed.shape == (3, 2, N, 3)
    close(condensed, sequential, "80-step condensed vs sequential sai", atol=1e-5)
    max_abs_error = (condensed - sequential).abs().max().item()
    print(f"    Sai lech max: {max_abs_error:.3e} (trang thai CoM)")


def test_first_last_step():
    mpc = make_mpc()
    x0, jerk = example_signals()
    predicted = condensed_predict(mpc, x0, jerk)
    expected_first = x0 @ mpc.A.T + jerk[..., 0, None] * mpc.B
    close(predicted[..., 0, :], expected_first, "Trang thai du doan dau sai")
    expected_last = recursive_predict(mpc, x0, jerk)[..., -1, :]
    close(predicted[..., -1, :], expected_last, "Trang thai du doan cuoi sai")


def test_superposition():
    mpc = make_mpc(horizon=12)
    torch.manual_seed(1902)
    a = torch.randn(2, 2, 3)
    b = torch.randn(2, 2, 3)
    u = torch.randn(2, 2, 12)
    v = torch.randn(2, 2, 12)
    lhs = condensed_predict(mpc, a+b, u+v)
    rhs = condensed_predict(mpc, a, u) + condensed_predict(mpc, b, v)
    close(lhs, rhs, "Tinh tuyen tinh du doan khong dung")


def test_parameter_variations():
    for dt, n in ((0.01, 20), (0.02, 80), (0.04, 40)):
        mpc = make_mpc(dt=dt, horizon=n)
        torch.manual_seed(61)
        x0 = torch.randn(2, 2, 3) * 0.1
        u = torch.randn(2, 2, n) * 0.3
        p = condensed_predict(mpc, x0, u)
        r = recursive_predict(mpc, x0, u)
        close(p, r, f"Du doan sai khi dt={dt}, N={n}")


def test_independent_of_height_and_weights():
    a = make_mpc(com_height=.13)
    b = make_mpc(com_height=.20, q_zmp=10., q_vel=.5, r_jerk=.01)
    close(a.Phi, b.Phi, "Phi phu thuoc h/cost (khong duoc)")
    close(a.Gamma, b.Gamma, "Gamma phu thuoc h/cost (khong duoc)")
    assert a.C[2] != b.C[2], "C phai phu thuoc com_height"


def test_idempotent_rebuild():
    mpc = make_mpc()
    phi0, gam0 = mpc.Phi.clone(), mpc.Gamma.clone()
    mpc.build_state_prediction()
    close(mpc.Phi, phi0, "Phi thay doi khi build lai")
    close(mpc.Gamma, gam0, "Gamma thay doi khi build lai")


def test_optional_cuda():
    if not torch.cuda.is_available():
        return "SKIP: moi truong khong co CUDA"
    mpc = make_mpc(device="cuda")
    assert mpc.Phi.is_cuda and mpc.Gamma.is_cuda
    x0, jerk = example_signals(device="cuda")
    close(condensed_predict(mpc, x0, jerk), recursive_predict(mpc, x0, jerk),
          "Du doan GPU sai")


def plot_demo():
    import matplotlib.pyplot as plt

    mpc = make_mpc()
    x0, jerk = example_signals()
    predicted = condensed_predict(mpc, x0, jerk)[0].detach().cpu().numpy()  # (2,N,3)
    sequential = recursive_predict(mpc, x0, jerk)[0].detach().cpu().numpy()
    error = abs(predicted-sequential)
    times = DT * torch.arange(1, N+1).numpy()
    components = ("CoM position (m)", "CoM velocity (m/s)", "CoM acceleration (m/s^2)")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for idx, ax in enumerate((axs[0,0], axs[0,1], axs[1,0])):
        for a, axis in enumerate(("X", "Y")):
            line, = ax.plot(times, predicted[a, :, idx], label=f"{axis}: Phi, Gamma", linewidth=2)
            ax.plot(times, sequential[a, :, idx], "--", color=line.get_color(),
                    label=f"{axis}: sequential", linewidth=1.5)
        ax.set_title(components[idx])
        ax.set_xlabel("Prediction time (s)")
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)

    ax = axs[1,1]
    ax.plot(times, error[0].max(axis=1), label="X: max error among 3 states")
    ax.plot(times, error[1].max(axis=1), label="Y: max error among 3 states")
    ax.set_title("Absolute prediction error")
    ax.set_xlabel("Prediction time (s)")
    ax.set_ylabel("Absolute difference")
    ax.grid(True, alpha=.3)
    ax.legend(fontsize=8)

    fig.suptitle("LIPM state prediction: condensed vs sequential (N=80, dt=0.02)")
    fig.tight_layout()
    plt.show()  # Khong savefig, khong ghi CSV/anh.


def main():
    parser = argparse.ArgumentParser(description="Kiem thu build_state_prediction")
    parser.add_argument("--no-plot", action="store_true", help="Chi chay test (khong mo cua so do thi)")
    args = parser.parse_args()
    tests = [
        ("Shape, dtype, device", test_shapes_types),
        ("Phi = A^(k+1)", test_phi_powers),
        ("Gamma causality, diagonal, powers", test_gamma_causality_and_columns),
        ("Recurrence Phi/Gamma", test_recurrence_relations),
        ("Short horizons N=1/2/3/5", test_small_horizons),
        ("Zero jerk", test_zero_jerk),
        ("80-step prediction, batch X/Y", test_full_horizon_batch_xy),
        ("First/last prediction", test_first_last_step),
        ("Superposition / linearity", test_superposition),
        ("Different dt and N", test_parameter_variations),
        ("Independent of com_height/cost", test_independent_of_height_and_weights),
        ("Rebuild without change", test_idempotent_rebuild),
        ("CUDA (optional)", test_optional_cuda),
    ]
    passed = skipped = failed = 0
    for title, fn in tests:
        try:
            status = fn()
            if status and status.startswith("SKIP"):
                skipped += 1
                print(f"[SKIP] {title}: {status}")
            else:
                passed += 1
                print(f"[PASS] {title}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {title}: {type(e).__name__}: {e}")
    print(f"\nRESULT: {passed} PASS | {skipped} SKIP | {failed} FAIL")
    if failed:
        raise SystemExit(1)
    if not args.no_plot:
        plot_demo()


if __name__ == "__main__":
    main()
