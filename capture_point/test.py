"""Kiem thu LIPMMPC.build_output_prediction() trong Open Duck Mini V2.

Dat file nay CUNG THU MUC voi capture_point/lipm_mpc.py, sau do chay:
    python test_lipm_mpc_build_output_prediction.py

Can thu vien: torch, matplotlib (matplotlib chi can neu ve do thi).
Mac dinh mo cua so Matplotlib; --no-plot chi chay test.
Khong luu anh hay CSV.

Luu y: jerk trong bai test la tin hieu THU NGHIEM, chua phai jerk toi uu MPC.
Chieu cao CoM (0.15 m) chi mang tinh minh hoa, khong phai so lieu robot.
"""

import argparse
import sys
from pathlib import Path

import torch

# Cho phep import khi dat test cung thu muc voi lipm_mpc.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipm_mpc import LIPMMPC


DT = 0.02
N = 80
H = 0.15


def make_mpc(dt=DT, horizon=N, com_height=H, device="cpu", **kwargs):
    mpc = LIPMMPC(
        com_height=com_height,
        dt=dt,
        horizon=horizon,
        device=device,
        **kwargs,
    )
    mpc.build_model()
    mpc.build_state_prediction()
    mpc.build_output_prediction()
    return mpc


def check_close(actual, expected, name, atol=2e-5, rtol=2e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)


def test_signals(n=N, dt=DT, batch=3, device="cpu"):
    """Tao vi du cho B robot, 2 truc (X/Y): x0 (B,2,3), jerk (B,2,N)."""
    x0 = torch.tensor(
        [
            [[0.02, 0.10, -0.03], [0.04, -0.03, 0.07]],
            [[-0.05, 0.06, 0.11], [0.10, 0.02, -0.02]],
            [[0.06, -0.08, 0.03], [-0.02, 0.04, -0.09]],
        ],
        dtype=torch.float32,
        device=device,
    )[:batch]
    t = dt * torch.arange(n, device=device, dtype=torch.float32)
    jerk = torch.empty((batch, 2, n), dtype=torch.float32, device=device)
    for b in range(batch):
        jerk[b, 0] = 0.5 * torch.sin(2.1 * t + 0.3 * b) + 0.10
        jerk[b, 1] = -0.4 * torch.cos(3.2 * t + 0.2 * b) - 0.04
    return x0, jerk


def state_from_matrices(mpc, x0, jerk):
    """Du doan trang thai x: (B,2,N,3)."""
    return (
        torch.einsum("nij,baj->bani", mpc.Phi, x0)
        + torch.einsum("nij,baj->bani", mpc.Gamma, jerk)
    )


def outputs_from_matrices(mpc, x0, jerk):
    """Du doan ZMP va van toc, moi dau ra co shape (B,2,N)."""
    z = torch.einsum("nj,baj->ban", mpc.Z_x, x0) + torch.einsum(
        "nj,baj->ban", mpc.Z_u, jerk
    )
    v = torch.einsum("nj,baj->ban", mpc.V_x, x0) + torch.einsum(
        "nj,baj->ban", mpc.V_u, jerk
    )
    return z, v


def recursive_reference(mpc, x0, jerk):
    """Doc lap: dung cong thuc x(t+1)=A x(t)+B u(t), sau do z=c-h/g*a."""
    x = x0.clone()
    x_history = []
    z_history = []
    v_history = []
    for i in range(jerk.shape[-1]):
        x = x @ mpc.A.T + jerk[..., i, None] * mpc.B
        x_history.append(x)
        z_history.append(x[..., 0] - (mpc.com_height / mpc.g) * x[..., 2])
        v_history.append(x[..., 1])
    return (
        torch.stack(x_history, dim=-2),
        torch.stack(z_history, dim=-1),
        torch.stack(v_history, dim=-1),
    )


def test_shapes_dtype_device():
    mpc = make_mpc()
    for name, shape in (
        ("Z_x", (N, 3)), ("Z_u", (N, N)),
        ("V_x", (N, 3)), ("V_u", (N, N)),
    ):
        value = getattr(mpc, name)
        assert value.shape == shape, f"{name}: {value.shape} != {shape}"
        assert value.dtype == torch.float32, f"{name}: dtype sai"
        assert value.device.type == "cpu", f"{name}: device sai"
        assert torch.isfinite(value).all(), f"{name}: gia tri NaN/Inf"


def test_zmp_projection():
    mpc = make_mpc()
    ref_zx = torch.stack([mpc.C @ mpc.Phi[i] for i in range(N)])
    ref_zu = torch.stack([mpc.C @ mpc.Gamma[i] for i in range(N)])
    check_close(mpc.Z_x, ref_zx, "Z_x khac C @ Phi")
    check_close(mpc.Z_u, ref_zu, "Z_u khac C @ Gamma")


def test_velocity_projection():
    mpc = make_mpc()
    cv = torch.tensor([0.0, 1.0, 0.0], device=mpc.device)
    ref_vx = torch.stack([cv @ mpc.Phi[i] for i in range(N)])
    ref_vu = torch.stack([cv @ mpc.Gamma[i] for i in range(N)])
    check_close(mpc.V_x, ref_vx, "V_x khac Cv @ Phi")
    check_close(mpc.V_u, ref_vu, "V_u khac Cv @ Gamma")
    assert mpc.V_x.data_ptr() != mpc.Phi.data_ptr(), "V_x can la clone"
    assert mpc.V_u.data_ptr() != mpc.Gamma.data_ptr(), "V_u can la clone"


def test_first_prediction_step():
    mpc = make_mpc()
    check_close(mpc.Z_x[0], mpc.C @ mpc.A, "Z_x[0] != C A")
    check_close(mpc.Z_u[0, 0], mpc.C @ mpc.B, "Z_u[0,0] != C B")
    check_close(mpc.V_x[0], mpc.A[1], "V_x[0] != hang van toc A")
    check_close(mpc.V_u[0, 0], mpc.B[1], "V_u[0,0] != B[1]")


def test_causality():
    mpc = make_mpc()
    for name in ("Z_u", "V_u"):
        matrix = getattr(mpc, name)
        assert torch.count_nonzero(torch.triu(matrix, diagonal=1)) == 0, (
            f"{name} phai bang 0 phia tren duong cheo (nhan qua)"
        )


def test_outputs_from_full_state():
    mpc = make_mpc()
    x0, jerk = test_signals()
    x = state_from_matrices(mpc, x0, jerk)
    z, v = outputs_from_matrices(mpc, x0, jerk)
    check_close(z, x[..., 0] - (H / mpc.g) * x[..., 2], "ZMP sai so voi trang thai du doan")
    check_close(v, x[..., 1], "Van toc sai so voi trang thai du doan")


def test_sequential_reference_batched_xy():
    mpc = make_mpc()
    x0, jerk = test_signals()
    _, z_ref, v_ref = recursive_reference(mpc, x0, jerk)
    z, v = outputs_from_matrices(mpc, x0, jerk)
    assert z.shape == (3, 2, N) and v.shape == (3, 2, N)
    check_close(z, z_ref, "ZMP tu ma tran khong khop mo phong tuan tu")
    check_close(v, v_ref, "Van toc tu ma tran khong khop mo phong tuan tu")
    print(f"    Sai lech ZMP lon nhat : {(z-z_ref).abs().max().item():.3e} m")
    print(f"    Sai lech van toc lon nhat: {(v-v_ref).abs().max().item():.3e} m/s")


def test_zero_jerk():
    mpc = make_mpc()
    x0, jerk = test_signals()
    jerk.zero_()
    z, v = outputs_from_matrices(mpc, x0, jerk)
    z_ref, v_ref = (
        torch.einsum("nj,baj->ban", mpc.Z_x, x0),
        torch.einsum("nj,baj->ban", mpc.V_x, x0),
    )
    check_close(z, z_ref, "ZMP phai chi phu thuoc x0 khi jerk=0")
    check_close(v, v_ref, "Van toc phai chi phu thuoc x0 khi jerk=0")
    _, z_seq, v_seq = recursive_reference(mpc, x0, jerk)
    check_close(z, z_seq, "ZMP khi jerk=0 sai")
    check_close(v, v_seq, "Van toc khi jerk=0 sai")


def test_zero_initial_state():
    mpc = make_mpc()
    x0, jerk = test_signals()
    x0.zero_()
    z, v = outputs_from_matrices(mpc, x0, jerk)
    check_close(z, torch.einsum("nj,baj->ban", mpc.Z_u, jerk), "ZMP zero x0 sai")
    check_close(v, torch.einsum("nj,baj->ban", mpc.V_u, jerk), "V zero x0 sai")


def test_height_changes_zmp_only():
    a = make_mpc(com_height=0.12)
    b = make_mpc(com_height=0.20)
    assert not torch.allclose(a.Z_x, b.Z_x), "Z_x phai thay doi theo h"
    assert not torch.allclose(a.Z_u, b.Z_u), "Z_u phai thay doi theo h"
    check_close(a.V_x, b.V_x, "V_x khong duoc thay doi khi doi h")
    check_close(a.V_u, b.V_u, "V_u khong duoc thay doi khi doi h")
    x0, jerk = test_signals()
    za, va = outputs_from_matrices(a, x0, jerk)
    zb, vb = outputs_from_matrices(b, x0, jerk)
    check_close(va, vb, "Van toc khong phu thuoc h")
    assert not torch.allclose(za, zb), "ZMP phai phu thuoc h"


def test_independent_of_cost_weights():
    a = make_mpc()
    b = make_mpc(q_zmp=10.0, q_vel=0.2, r_jerk=0.01)
    for name in ("Z_x", "Z_u", "V_x", "V_u"):
        check_close(getattr(a, name), getattr(b, name), f"{name} khong phu thuoc q/r")


def test_parameter_variants():
    for dt, n in ((0.01, 20), (0.02, 80), (0.04, 40), (0.05, 1)):
        mpc = make_mpc(dt=dt, horizon=n)
        x0, jerk = test_signals(n=n, dt=dt)
        z, v = outputs_from_matrices(mpc, x0, jerk)
        _, z_ref, v_ref = recursive_reference(mpc, x0, jerk)
        check_close(z, z_ref, f"ZMP sai: dt={dt}, N={n}")
        check_close(v, v_ref, f"Van toc sai: dt={dt}, N={n}")


def test_x_y_axes_are_independent():
    mpc = make_mpc()
    x0, jerk = test_signals()
    z, v = outputs_from_matrices(mpc, x0, jerk)
    # Hoan doi tru X va Y: ket qua phai hoan doi tuong ung.
    zswap, vswap = outputs_from_matrices(mpc, x0.flip(1), jerk.flip(1))
    check_close(zswap, z.flip(1), "Truc X/Y khong doc lap (ZMP)")
    check_close(vswap, v.flip(1), "Truc X/Y khong doc lap (van toc)")


def test_rebuild_is_deterministic():
    mpc = make_mpc()
    old = {name: getattr(mpc, name).clone() for name in ("Z_x", "Z_u", "V_x", "V_u")}
    mpc.build_output_prediction()
    for name in old:
        check_close(getattr(mpc, name), old[name], f"{name} thay doi khi build lai")


def test_optional_cuda():
    if not torch.cuda.is_available():
        return "SKIP: Khong co CUDA trong moi truong hien tai"
    mpc = make_mpc(device="cuda")
    x0, jerk = test_signals(device="cuda")
    z, v = outputs_from_matrices(mpc, x0, jerk)
    _, z_ref, v_ref = recursive_reference(mpc, x0, jerk)
    assert z.is_cuda and v.is_cuda
    check_close(z, z_ref, "ZMP CUDA sai")
    check_close(v, v_ref, "Van toc CUDA sai")


def plot_demo():
    import matplotlib.pyplot as plt

    mpc = make_mpc()
    x0, jerk = test_signals(batch=1)
    x = state_from_matrices(mpc, x0, jerk)[0].detach().cpu()
    z, v = outputs_from_matrices(mpc, x0, jerk)
    _, z_ref, v_ref = recursive_reference(mpc, x0, jerk)
    z, v = z[0].detach().cpu(), v[0].detach().cpu()
    z_ref, v_ref = z_ref[0].detach().cpu(), v_ref[0].detach().cpu()
    t = mpc.dt * torch.arange(1, mpc.horizon + 1)
    time = t.numpy()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for axis, title in ((0, "X"), (1, "Y")):
        ax = axs[0, axis]
        ax.plot(time, z[axis].numpy(), label="ZMP - prediction matrices", linewidth=2)
        ax.plot(time, z_ref[axis].numpy(), "--", label="ZMP - sequential", linewidth=1.7)
        ax.plot(time, x[axis, :, 0].numpy(), ":", label="CoM position", linewidth=1.5)
        ax.set(title=f"Axis {title}: ZMP vs CoM position", xlabel="Time (s)", ylabel="Position (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axs[1, axis]
        ax.plot(time, v[axis].numpy(), label="CoM velocity - prediction matrices", linewidth=2)
        ax.plot(time, v_ref[axis].numpy(), "--", label="CoM velocity - sequential", linewidth=1.7)
        ax.set(title=f"Axis {title}: predicted CoM velocity", xlabel="Time (s)", ylabel="Velocity (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("LIPM MPC: output prediction verification (jerk for testing only)")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test LIPMMPC.build_output_prediction()")
    parser.add_argument("--no-plot", action="store_true", help="Chi kiem thu, khong hien do thi")
    args = parser.parse_args()

    tests = [
        test_shapes_dtype_device,
        test_zmp_projection,
        test_velocity_projection,
        test_first_prediction_step,
        test_causality,
        test_outputs_from_full_state,
        test_sequential_reference_batched_xy,
        test_zero_jerk,
        test_zero_initial_state,
        test_height_changes_zmp_only,
        test_independent_of_cost_weights,
        test_parameter_variants,
        test_x_y_axes_are_independent,
        test_rebuild_is_deterministic,
        test_optional_cuda,
    ]

    print("=" * 68)
    print("TEST build_output_prediction() -- LIPMMPC")
    print("=" * 68)
    passed = skipped = failed = 0
    for fn in tests:
        try:
            result = fn()
            if isinstance(result, str) and result.startswith("SKIP"):
                print(f"[SKIP] {fn.__name__}: {result}")
                skipped += 1
            else:
                print(f"[PASS] {fn.__name__}")
                passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            failed += 1

    print("-" * 68)
    print(f"RESULT: {passed} PASS | {skipped} SKIP | {failed} FAIL")
    if failed:
        raise SystemExit(1)
    if not args.no_plot:
        plot_demo()


if __name__ == "__main__":
    main()
