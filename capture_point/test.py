"""Kiem thu LIPMMPC.build_model() tren ma nguon hien tai.

Dat file nay cung thu muc voi lipm_mpc.py, sau do chay:
    python test_lipm_mpc_build_model.py

Chuong trinh chi in ket qua va HIEN THI do thi Matplotlib.
Khong ghi CSV, khong luu anh, khong thay doi ma nguon MPC.
"""

import math
import sys
from pathlib import Path

import torch

# Ho tro chay truc tiep tu thu muc capture_point hoac tu thu muc bat ky.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lipm_mpc import LIPMMPC


DT = 0.02
H = 0.15  # Vi du kiem thu, KHONG phai chieu cao CoM da do cua robot.
G = 9.81


def check_close(actual, expected, message, atol=2e-6, rtol=1e-6):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=message)


def test_default_matrices():
    mpc = LIPMMPC(com_height=H)
    mpc.build_model()
    A_ref = torch.tensor([
        [1., DT, DT**2 / 2],
        [0., 1., DT],
        [0., 0., 1.],
    ])
    B_ref = torch.tensor([DT**3 / 6, DT**2 / 2, DT])
    C_ref = torch.tensor([1., 0., -H / G])
    assert tuple(mpc.A.shape) == (3, 3)
    assert tuple(mpc.B.shape) == (3,)
    assert tuple(mpc.C.shape) == (3,)
    for x in (mpc.A, mpc.B, mpc.C):
        assert x.dtype == torch.float32 and x.device.type == 'cpu'
    check_close(mpc.A, A_ref, 'Ma tran A khong dung')
    check_close(mpc.B, B_ref, 'Vector B khong dung')
    check_close(mpc.C, C_ref, 'Vector C khong dung')


def test_constant_jerk_closed_form():
    # Doi chieu truc tiep voi phep tich phan giai tich trong mot chu ky.
    torch.manual_seed(2026)
    for dt in (0.005, 0.01, 0.02, 0.05):
        mpc = LIPMMPC(com_height=H, dt=dt)
        mpc.build_model()
        x0 = torch.randn(200, 3)
        u = torch.randn(200)
        actual = x0 @ mpc.A.T + u[:, None] * mpc.B
        expected = torch.stack([
            x0[:, 0] + dt*x0[:, 1] + .5*dt**2*x0[:, 2] + dt**3/6*u,
            x0[:, 1] + dt*x0[:, 2] + .5*dt**2*u,
            x0[:, 2] + dt*u,
        ], dim=1)
        check_close(actual, expected, f'Roi rac hoa sai dt={dt}')


def test_zero_jerk():
    mpc = LIPMMPC(com_height=H)
    mpc.build_model()
    x0 = torch.tensor([0.12, -0.3, 1.5])
    x1 = mpc.A @ x0 + mpc.B * 0.0
    expected = torch.tensor([
        0.12 - 0.3*DT + 0.5*1.5*DT**2,
        -0.3 + 1.5*DT,
        1.5,
    ])
    check_close(x1, expected, 'Zero jerk khong dung')


def test_zmp_output():
    mpc = LIPMMPC(com_height=H)
    mpc.build_model()
    x = torch.tensor([0.15, 0.24, 0.9])
    z = mpc.C @ x
    assert math.isclose(z.item(), 0.15 - H/G*0.9, rel_tol=1e-6, abs_tol=1e-7)
    # Van toc khong anh huong truc tiep den ZMP hien tai.
    x_other_v = x.clone()
    x_other_v[1] = 100.0
    check_close(mpc.C @ x_other_v, z, 'ZMP bi anh huong boi van toc')


def test_height_dependency():
    small = LIPMMPC(com_height=0.12)
    tall = LIPMMPC(com_height=0.20)
    small.build_model()
    tall.build_model()
    check_close(small.A, tall.A, 'Chieu cao khong duoc doi A')
    check_close(small.B, tall.B, 'Chieu cao khong duoc doi B')
    assert small.C[2] != tall.C[2]
    check_close(small.C[2], torch.tensor(-0.12/G), 'Chieu cao sai trong C')
    check_close(tall.C[2], torch.tensor(-0.20/G), 'Chieu cao sai trong C')


def test_dt_dependency():
    slow = LIPMMPC(com_height=H, dt=.05)
    fast = LIPMMPC(com_height=H, dt=.01)
    slow.build_model()
    fast.build_model()
    assert not torch.allclose(slow.A, fast.A)
    assert not torch.allclose(slow.B, fast.B)
    check_close(slow.C, fast.C, 'dt khong duoc doi C')


def test_shared_xy_model():
    # B=2 moi truong, 2 truc (x,y), moi truc 3 trang thai.
    mpc = LIPMMPC(com_height=H)
    mpc.build_model()
    x = torch.tensor([
        [[0., 0.10, 0.0], [0.05, 0., 0.0]],
        [[0., 0.00, 0.3], [-0.05, 0., -0.2]],
    ])
    u = torch.tensor([[0.8, -0.5], [0.3, 0.7]])
    result = x @ mpc.A.T + u[..., None] * mpc.B
    assert result.shape == (2, 2, 3)
    for b in range(2):
        for axis in range(2):
            check_close(result[b, axis], mpc.A @ x[b, axis] + mpc.B*u[b, axis],
                        'Khong dung khi xu ly batch va hai truc')


def test_independence_from_cost_and_horizon():
    m0 = LIPMMPC(com_height=H, dt=DT, horizon=80)
    m1 = LIPMMPC(com_height=H, dt=DT, horizon=20,
                 q_zmp=2.0, q_vel=0.05, r_jerk=1e-3)
    m0.build_model()
    m1.build_model()
    for name in ('A', 'B', 'C'):
        check_close(getattr(m0, name), getattr(m1, name),
                    f'Ma tran {name} khong duoc phu thuoc cost/horizon')


def test_rebuild():
    mpc = LIPMMPC(com_height=H)
    mpc.build_model()
    old = [mpc.A.clone(), mpc.B.clone(), mpc.C.clone()]
    mpc.build_model()
    for name, x in zip(('A', 'B', 'C'), old):
        check_close(getattr(mpc, name), x, f'Goi lai build_model lam doi {name}')


def test_optional_cuda():
    if not torch.cuda.is_available():
        return 'SKIP (khong co GPU CUDA)'
    mpc = LIPMMPC(com_height=H, device='cuda')
    mpc.build_model()
    assert all(t.device.type == 'cuda' for t in (mpc.A, mpc.B, mpc.C))


def show_demo_plot():
    # Minh hoa rieng mo hinh: jerk dat thu nghiem, KHONG phai jerk toi uu MPC.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Khong co matplotlib: bo qua do thi (pip install matplotlib)')
        return
    mpc = LIPMMPC(com_height=H, dt=DT)
    mpc.build_model()
    t = torch.arange(0, 1.0 + DT/2, DT)
    states = torch.zeros(len(t), 3)
    states[0] = torch.tensor([0.0, 0.05, 0.0])
    jerks = torch.zeros(len(t)-1)
    for k in range(len(t)-1):
        jerks[k] = 1.5 if t[k] < .5 else -1.5
        states[k+1] = mpc.A @ states[k] + mpc.B * jerks[k]
    zmp = states @ mpc.C
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle('Kiem chung build_model(): cap nhat CoM voi jerk thu nghiem (khong phai MPC)')
    for ax, data, title, ylabel in [
        (axs[0, 0], states[:, 0], 'Vi tri CoM', 'c (m)'),
        (axs[0, 1], states[:, 1], 'Van toc CoM', 'dc/dt (m/s)'),
        (axs[1, 0], states[:, 2], 'Gia toc CoM', 'd2c/dt2 (m/s2)'),
        (axs[1, 1], zmp, 'ZMP tu C @ x', 'ZMP (m)'),
    ]:
        ax.plot(t.numpy(), data.numpy(), linewidth=2)
        ax.set(title=title, xlabel='Thoi gian (s)', ylabel=ylabel)
        ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.show()


def main():
    tests = [
        ('A, B, C dung gia tri / shape / dtype', test_default_matrices),
        ('200 mau jerk cho 4 gia tri dt', test_constant_jerk_closed_form),
        ('Jerk bang 0', test_zero_jerk),
        ('ZMP = c - h/g * gia toc', test_zmp_output),
        ('Thay doi chieu cao CoM', test_height_dependency),
        ('Thay doi chu ky lay mau', test_dt_dependency),
        ('Xu ly dong thoi hai truc X/Y', test_shared_xy_model),
        ('Khong phu thuoc horizon / cost', test_independence_from_cost_and_horizon),
        ('Goi lai build_model', test_rebuild),
        ('Kiem tra CUDA neu co', test_optional_cuda),
    ]
    print('\n' + '='*64)
    print('TEST LIPMMPC.build_model()')
    print('='*64)
    passed, skipped = 0, 0
    for name, fn in tests:
        result = fn()
        if result and result.startswith('SKIP'):
            skipped += 1
            print(f'[SKIP] {name}: {result}')
        else:
            passed += 1
            print(f'[PASS] {name}')
    print('-'*64)
    print(f'Ket qua: {passed} PASS, {skipped} SKIP, {len(tests)} tong kiem tra')
    mpc = LIPMMPC(com_height=H)
    mpc.build_model()
    print('\nMa tran A:\n', mpc.A)
    print('Vector B:', mpc.B)
    print('Vector C:', mpc.C)
    print('\nMo do thi kiem chung (khong luu tep)...')
    show_demo_plot()


if __name__ == '__main__':
    main()
