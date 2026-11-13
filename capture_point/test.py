"""Kiem thu ham LIPMMPC.solve() - Open Duck Mini V2.

Dat file cung thu muc voi capture_point/lipm_mpc.py, sau do chay:
    python test_lipm_mpc_solve.py
    python test_lipm_mpc_solve.py --no-plot

Yeu cau: torch, numpy; matplotlib chi can de ve do thi.
Khong luu anh, CSV hay bat ky du lieu dau ra nao.

Luu y: cac quy dao tham chieu la tin hieu kiem thu thuat toan,
KHONG phai ket qua robot di bo thuc te. solve() dung mo hinh open-loop;
khong su dung du lieu phan hoi tu cam bien.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Import dung class LIPMMPC tu file lipm_mpc.py cung thu muc.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipm_mpc import LIPMMPC


DT = 0.02
N = 80
H_COM = 0.15  # Vi du de kiem thu, KHONG phai h danh dinh da do cua robot.


def create_mpc(horizon=N, dt=DT, com_height=H_COM, device="cpu", **weights):
    mpc = LIPMMPC(com_height=com_height, dt=dt, horizon=horizon,
                  device=device, **weights)
    mpc.build_model()
    mpc.build_state_prediction()
    mpc.build_output_prediction()
    mpc.precompute_gains()
    return mpc


def make_signals(mpc, batch=4):
    """Tao x0(B,3), zref(B,N), vref(B,N) co nhieu gia tri khac nhau."""
    n = mpc.horizon
    t = torch.arange(1, n + 1, dtype=torch.float32,
                     device=mpc.device) * mpc.dt
    b = torch.arange(batch, dtype=torch.float32,
                     device=mpc.device)
    x0 = torch.stack([
        0.025 * torch.sin(0.9 * (b + 1)),
        0.06 * torch.cos(0.8 * (b + 1)),
        0.025 * torch.sin(1.3 * (b + 1)),
    ], dim=1)
    zref = (x0[:, 0:1]
            + 0.025 * torch.sin(2.1 * t[None, :] + 0.2 * b[:, None])
            + 0.012 * torch.sin(0.8 * t[None, :]))
    vref = (0.035 * torch.cos(1.4 * t[None, :] + 0.4 * b[:, None])
            + 0.02 * (b[:, None] % 2 - 0.5))
    return x0, zref, vref


def predict_iteratively(mpc, x0, U):
    """Kiem chuan doc lap: tinh tung buoc x_{k+1}=A*x_k+B*u_k."""
    x = x0.clone()
    states = []
    for k in range(mpc.horizon):
        x = x @ mpc.A.T + U[:, k:k+1] * mpc.B
        states.append(x.clone())
    return torch.stack(states, dim=1)


def hessian_rhs(mpc, x0, zref, vref):
    """Dung float64 de tinh he phuong trinh toi uu doc lap voi K_z, K_v."""
    zu = mpc.Z_u.double()
    vu = mpc.V_u.double()
    zfree = x0.double() @ mpc.Z_x.double().T
    vfree = x0.double() @ mpc.V_x.double().T
    ez = zref.double() - zfree
    ev = vref.double() - vfree
    h = (mpc.q_zmp * (zu.T @ zu)
         + mpc.q_vel * (vu.T @ vu)
         + mpc.r_jerk * torch.eye(mpc.horizon, dtype=torch.float64,
                                  device=mpc.device))
    rhs = mpc.q_zmp * (ez @ zu) + mpc.q_vel * (ev @ vu)
    return h, rhs


def objective(mpc, x0, zref, vref, U):
    """Gia tri ham chi phi cho moi mau trong batch (float64)."""
    x = x0.double()
    u = U.double()
    z = x @ mpc.Z_x.double().T + u @ mpc.Z_u.double().T
    v = x @ mpc.V_x.double().T + u @ mpc.V_u.double().T
    return (mpc.q_zmp * ((z - zref.double()) ** 2).sum(dim=1)
            + mpc.q_vel * ((v - vref.double()) ** 2).sum(dim=1)
            + mpc.r_jerk * (u ** 2).sum(dim=1))


def test_01_shapes_dtypes_devices_and_finite():
    m = create_mpc()
    for b in (1, 2, 4):
        x0, zr, vr = make_signals(m, batch=b)
        U, X = m.solve(x0, zr, vr)
        assert U.shape == (b, N), (U.shape, (b, N))
        assert X.shape == (b, N, 3), (X.shape, (b, N, 3))
        assert U.dtype == X.dtype == x0.dtype == torch.float32
        assert U.device == X.device == m.device
        assert torch.isfinite(U).all() and torch.isfinite(X).all()


def test_02_compare_independent_float64_linear_solve():
    """Dung ma tran H tu mo hinh, KHONG dung K_z/K_v de tim U tham chieu."""
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, _ = m.solve(x0, zr, vr)
    H, rhs = hessian_rhs(m, x0, zr, vr)
    U_ref = torch.linalg.solve(H, rhs.T).T
    torch.testing.assert_close(U.double(), U_ref, atol=2e-5, rtol=2e-5)
    print(f"    max|U_solve - U_direct| = {(U.double()-U_ref).abs().max().item():.3e}")


def test_03_compare_author_reference_formula():
    """Kiem tra bieu thuc solve() cua ZmpPreview goc."""
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, X = m.solve(x0, zr, vr)
    ez = zr - x0 @ m.Z_x.T
    ev = vr - x0 @ m.V_x.T
    u_author = ez @ m.K_z.T + ev @ m.K_v.T
    x_author = (torch.einsum('kij,bj->bki', m.Phi, x0)
                + torch.einsum('kin,bn->bki', m.Gamma, u_author))
    torch.testing.assert_close(U, u_author, atol=0, rtol=0)
    torch.testing.assert_close(X, x_author, atol=0, rtol=0)


def test_04_state_prediction_vs_step_by_step():
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, X = m.solve(x0, zr, vr)
    X_seq = predict_iteratively(m, x0, U)
    torch.testing.assert_close(X, X_seq, atol=2e-6, rtol=2e-5)
    print(f"    max|X_PhiGamma - X_iterative| = {(X-X_seq).abs().max().item():.3e}")


def test_05_first_and_final_states():
    m = create_mpc()
    x0, zr, vr = make_signals(m, batch=2)
    U, X = m.solve(x0, zr, vr)
    first = x0 @ m.A.T + U[:, 0:1] * m.B
    last = X[:, -2] @ m.A.T + U[:, -1:] * m.B
    torch.testing.assert_close(X[:, 0], first, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(X[:, -1], last, atol=2e-6, rtol=1e-5)


def test_06_zmp_and_velocity_from_state_prediction():
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, X = m.solve(x0, zr, vr)
    z_from_states = X[:, :, 0] - (m.com_height / m.g) * X[:, :, 2]
    z_from_matrices = x0 @ m.Z_x.T + U @ m.Z_u.T
    v_from_states = X[:, :, 1]
    v_from_matrices = x0 @ m.V_x.T + U @ m.V_u.T
    torch.testing.assert_close(z_from_states, z_from_matrices, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(v_from_states, v_from_matrices, atol=2e-6, rtol=2e-5)


def test_07_first_jerk_matches_cached_kz0_kv0():
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, _ = m.solve(x0, zr, vr)
    ez = zr - x0 @ m.Z_x.T
    ev = vr - x0 @ m.V_x.T
    u0 = (ez * m.K_z0).sum(dim=1) + (ev * m.K_v0).sum(dim=1)
    torch.testing.assert_close(U[:, 0], u0, atol=1e-6, rtol=1e-5)


def test_08_optimality_gradient():
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, _ = m.solve(x0, zr, vr)
    H, rhs = hessian_rhs(m, x0, zr, vr)
    residual = U.double() @ H.T - rhs
    relative = residual.norm() / rhs.norm().clamp_min(1e-14)
    assert relative.item() < 1e-5, f"gradient khac 0: {relative.item():.3e}"
    print(f"    relative ||H U - rhs|| = {relative.item():.3e}")


def test_09_cost_better_than_zero_and_perturbed_jerk():
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    U, _ = m.solve(x0, zr, vr)
    Jopt = objective(m, x0, zr, vr, U)
    Jzero = objective(m, x0, zr, vr, torch.zeros_like(U))
    gen = torch.Generator().manual_seed(20261113)
    noise = torch.randn(U.shape, dtype=torch.float32, generator=gen).to(U.device) * 0.8
    Jnoise = objective(m, x0, zr, vr, U + noise)
    assert (Jopt <= Jzero + 1e-9).all(), 'J toi uu > J jerk=0'
    assert (Jopt <= Jnoise + 1e-9).all(), 'J toi uu > J jerk nhieu'
    print(f"    mean J(U*)={Jopt.mean().item():.6g}; "
          f"J(0)={Jzero.mean().item():.6g}; J(U*+noise)={Jnoise.mean().item():.6g}")


def test_10_zero_initial_state_and_references():
    m = create_mpc()
    x0 = torch.zeros((3, 3), dtype=torch.float32)
    zref = torch.zeros((3, N), dtype=torch.float32)
    vref = torch.zeros_like(zref)
    U, X = m.solve(x0, zref, vref)
    assert torch.count_nonzero(U) == 0
    assert torch.count_nonzero(X) == 0


def test_11_constant_position_equilibrium():
    m = create_mpc()
    positions = torch.tensor([0.025, -0.037], dtype=torch.float32)
    x0 = torch.stack([positions, torch.zeros_like(positions),
                      torch.zeros_like(positions)], dim=1)
    zref = positions[:, None].expand(-1, N).clone()
    vref = torch.zeros_like(zref)
    U, X = m.solve(x0, zref, vref)
    assert U.abs().max().item() < 1e-4
    torch.testing.assert_close(X[:, :, 0], zref, atol=5e-6, rtol=0)
    assert X[:, :, 1:].abs().max().item() < 5e-6


def test_12_batch_equals_separate_solves_and_axis_independence():
    m = create_mpc()
    x0, zr, vr = make_signals(m, batch=4)
    U, X = m.solve(x0, zr, vr)
    for i in range(4):
        ui, xi = m.solve(x0[i:i+1], zr[i:i+1], vr[i:i+1])
        torch.testing.assert_close(U[i:i+1], ui, atol=1e-5, rtol=2e-5)
        torch.testing.assert_close(X[i:i+1], xi, atol=2e-6, rtol=1e-5)
    # Xem B=2 la hai truc X/Y: thay doi tham chieu X khong anh huong Y.
    zr_new = zr.clone()
    vr_new = vr.clone()
    zr_new[0] += 0.04
    vr_new[0] -= 0.03
    U_new, X_new = m.solve(x0, zr_new, vr_new)
    torch.testing.assert_close(U[1:], U_new[1:], atol=0, rtol=0)
    torch.testing.assert_close(X[1:], X_new[1:], atol=0, rtol=0)
    assert (U[0] - U_new[0]).abs().max().item() > 1e-3


def test_13_input_immutability_and_repeatability():
    m = create_mpc()
    x0, zr, vr = make_signals(m)
    before = (x0.clone(), zr.clone(), vr.clone())
    U, X = m.solve(x0, zr, vr)
    U2, X2 = m.solve(x0, zr, vr)
    torch.testing.assert_close(U, U2, atol=0, rtol=0)
    torch.testing.assert_close(X, X2, atol=0, rtol=0)
    for original, snapshot in zip((x0, zr, vr), before):
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)


def test_14_short_horizons_and_custom_dt():
    for n, dt in ((1, 0.01), (2, 0.02), (5, 0.03), (20, 0.015)):
        m = create_mpc(horizon=n, dt=dt)
        x0, zr, vr = make_signals(m, batch=2)
        U, X = m.solve(x0, zr, vr)
        assert U.shape == (2, n) and X.shape == (2, n, 3)
        torch.testing.assert_close(X, predict_iteratively(m, x0, U),
                                   atol=2e-6, rtol=2e-5)


def test_15_cuda_if_available():
    if not torch.cuda.is_available():
        return 'SKIP: khong co GPU CUDA'
    m = create_mpc(device='cuda')
    x0, zr, vr = make_signals(m, batch=2)
    U, X = m.solve(x0, zr, vr)
    assert U.is_cuda and X.is_cuda
    assert torch.isfinite(U).all() and torch.isfinite(X).all()
    torch.testing.assert_close(X, predict_iteratively(m, x0, U),
                               atol=2e-6, rtol=2e-5)


def plot_demo():
    import matplotlib.pyplot as plt

    m = create_mpc()
    # Batched=2 tuong ung X va Y; 2 bo tham chieu doc lap.
    x0, zref, vref = make_signals(m, batch=2)
    U, X = m.solve(x0, zref, vref)
    X_seq = predict_iteratively(m, x0, U)
    z = X[:, :, 0] - (m.com_height / m.g) * X[:, :, 2]
    t = np.arange(1, N + 1) * DT

    def arr(tensor):
        return tensor.detach().cpu().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle('LIPM-MPC solve(): kiem thu du doan CoM va ZMP (khong phai gait thuc te)')

    for i, axis in enumerate(('X', 'Y')):
        axs[0, i].plot(t, arr(zref[i]), '--', label=f'ZMP ref {axis}')
        axs[0, i].plot(t, arr(z[i]), label=f'ZMP predicted {axis}')
        axs[0, i].set_title(f'ZMP - truc {axis}')
        axs[0, i].set_ylabel('ZMP (m)')

    for i, axis in enumerate(('X', 'Y')):
        axs[0, 2].plot(t, arr(X[i, :, 0]), label=f'CoM {axis}')
    axs[0, 2].set_title('Vi tri CoM du doan')
    axs[0, 2].set_ylabel('CoM (m)')

    for i, axis in enumerate(('X', 'Y')):
        axs[1, 0].plot(t, arr(X[i, :, 1]), label=f'CoM vel {axis}')
        axs[1, 0].plot(t, arr(vref[i]), '--', label=f'vref {axis}')
    axs[1, 0].set_title('Van toc CoM')
    axs[1, 0].set_ylabel('Van toc (m/s)')

    for i, axis in enumerate(('X', 'Y')):
        axs[1, 1].plot(t, arr(X[i, :, 2]), label=f'CoM acc {axis}')
    axs[1, 1].set_title('Gia toc CoM')
    axs[1, 1].set_ylabel('Gia toc (m/s^2)')

    for i, axis in enumerate(('X', 'Y')):
        axs[1, 2].plot(t, arr(U[i]), label=f'jerk {axis}')
    axs[1, 2].set_title('Chuoi jerk toi uu')
    axs[1, 2].set_ylabel('Jerk (m/s^3)')

    for ax in axs.flat:
        ax.set_xlabel('Thoi gian du doan (s)')
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    max_err = (X - X_seq).abs().max().item()
    print(f'\n[DO THI] Sai lech max giua du doan va tich phan tung buoc: {max_err:.3e}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test LIPMMPC.solve()')
    parser.add_argument('--no-plot', action='store_true',
                        help='Chi kiem thu, khong mo do thi')
    args = parser.parse_args()

    tests = [
        test_01_shapes_dtypes_devices_and_finite,
        test_02_compare_independent_float64_linear_solve,
        test_03_compare_author_reference_formula,
        test_04_state_prediction_vs_step_by_step,
        test_05_first_and_final_states,
        test_06_zmp_and_velocity_from_state_prediction,
        test_07_first_jerk_matches_cached_kz0_kv0,
        test_08_optimality_gradient,
        test_09_cost_better_than_zero_and_perturbed_jerk,
        test_10_zero_initial_state_and_references,
        test_11_constant_position_equilibrium,
        test_12_batch_equals_separate_solves_and_axis_independence,
        test_13_input_immutability_and_repeatability,
        test_14_short_horizons_and_custom_dt,
        test_15_cuda_if_available,
    ]
    passed = skipped = failed = 0
    print('=' * 72)
    print('TEST LIPMMPC.solve() - OPEN DUCK MINI V2')
    print('=' * 72)
    for i, fn in enumerate(tests, 1):
        try:
            result = fn()
            if isinstance(result, str) and result.startswith('SKIP'):
                skipped += 1
                print(f'[{i:02d}] SKIP  {fn.__name__}: {result}')
            else:
                passed += 1
                print(f'[{i:02d}] PASS  {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[{i:02d}] FAIL  {fn.__name__}: {type(exc).__name__}: {exc}')

    print('-' * 72)
    print(f'KET QUA: {passed} PASS, {skipped} SKIP, {failed} FAIL')
    print('=' * 72)
    if failed:
        return 1
    if not args.no_plot:
        plot_demo()
    return 0


if __name__ == '__main__':
    sys.exit(main())
