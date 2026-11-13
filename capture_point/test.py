"""Kiem thu ham ZMPGenerator.zmp_walk() va hien thi ZMP reference.

Dat file nay cung thu muc voi:
    zmp_generator.py
    contact_scheduler.py
    footstep_planner.py

Chay:
    python test_zmp_generator_walk.py

Khong ghi CSV, khong luu anh. Chi in PASS/FAIL va mo do thi Matplotlib.
"""

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
HORIZON = 80


def make_footsteps(j_base=0, num_envs=1):
    """Tao du lieu chan mau theo thu tu p_-1, p0, ..., p12 (B,14,3).

    p0 la chan tru cua j_base; moi lan doi chan, x tang 0.04 m.
    """
    pts = torch.zeros(num_envs, 14, 3, dtype=torch.float32)
    for k in range(14):
        relative_step = k - 1  # chi so 0 trong mang la p_-1
        step_index = j_base + relative_step
        pts[:, k, 0] = 0.04 * max(relative_step, 0)
        pts[:, k, 1] = 0.09 if step_index % 2 == 0 else -0.09
    return pts


def check_close(actual, expected, name, atol=2e-6):
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0, msg=name)


def test_j0_no_double_support():
    """Tai j=0, ZMP nam o chan trai ngay ca khi local < T_DS."""
    gen = ZMPGenerator()
    pts = make_footsteps(j_base=0)
    j = torch.zeros(1, 5, dtype=torch.long)
    local = torch.tensor([[0.0, 0.01, 0.02, 0.03, 0.08]])
    out = gen.zmp_walk(pts, torch.tensor([0]), j, local, DS_DURATION)
    assert out.shape == (1, 5, 2)
    expected = torch.tensor([[0.005, 0.097]]).expand(5, 2)
    check_close(out[0], expected, "Buoc j=0 phai giu ZMP tai chan trai")


def test_j1_ds_then_ss():
    """Tai j=1: DS tu trai sang phai; SS giu tai phai."""
    gen = ZMPGenerator()
    pts = make_footsteps(j_base=0)
    j = torch.ones(1, 5, dtype=torch.long)
    local = torch.tensor([[0.0, 0.01, 0.02, 0.04, 0.12]])
    out = gen.zmp_walk(pts, torch.tensor([0]), j, local, DS_DURATION)[0]
    left = torch.tensor([0.005, 0.097])   # p0
    right = torch.tensor([0.045, -0.097])  # p1
    check_close(out[0], left, "DS j=1, bat dau o chan trai")
    s_quarter = 3 * .25**2 - 2 * .25**3
    check_close(out[1], (1 - s_quarter) * left + s_quarter * right, "DS j=1, 25%")
    check_close(out[2], 0.5 * (left + right), "DS j=1, giua pha")
    check_close(out[3], right, "DS j=1 ket thuc")
    check_close(out[4], right, "SS j=1")


def test_j2_change_foot_direction():
    """Tai j=2: DS tu chan phai sang chan trai, ke ca dich chuyen X."""
    gen = ZMPGenerator()
    pts = make_footsteps(j_base=0)
    j = torch.full((1, 3), 2, dtype=torch.long)
    local = torch.tensor([[0.0, 0.02, 0.10]])
    out = gen.zmp_walk(pts, torch.tensor([0]), j, local, DS_DURATION)[0]
    old = torch.tensor([0.045, -0.097])
    new = torch.tensor([0.085, 0.097])
    check_close(out[0], old, "DS j=2 bat dau o chan phai")
    check_close(out[1], (old + new) / 2, "DS j=2 giua pha")
    check_close(out[2], new, "SS j=2 o chan trai")


def test_nonzero_j_base():
    """Khong duoc nham j_future tuyet doi voi vi tri tuong doi trong footsteps."""
    gen = ZMPGenerator()
    base = 5  # buoc j=5: chan phai la stance
    pts = make_footsteps(j_base=base)
    j = torch.tensor([[5, 6, 6, 7]], dtype=torch.long)
    local = torch.tensor([[0.10, 0.0, 0.10, 0.10]])
    out = gen.zmp_walk(pts, torch.tensor([base]), j, local, DS_DURATION)[0]
    check_close(out[0], [0.005, -0.097], "j_base=5, j=5 phai")
    check_close(out[1], [0.005, -0.097], "j_base=5, j=6 DS bat dau o phai")
    check_close(out[2], [0.045, 0.097], "j_base=5, j=6 SS o trai")
    check_close(out[3], [0.085, -0.097], "j_base=5, j=7 SS o phai")


def test_custom_and_zero_offsets():
    pts = make_footsteps()
    j = torch.tensor([[0, 1, 2]], dtype=torch.long)
    local = torch.full((1, 3), .1)
    base = torch.tensor([0])
    zero = ZMPGenerator(zmp_offset_x=0.0, zmp_offset_y_out=0.0)
    out0 = zero.zmp_walk(pts, base, j, local, DS_DURATION)[0]
    check_close(out0, [[0.0, 0.09], [0.04, -0.09], [0.08, 0.09]], "offset=0")
    custom = ZMPGenerator(zmp_offset_x=.01, zmp_offset_y_out=.02)
    out = custom.zmp_walk(pts, base, j, local, DS_DURATION)[0]
    check_close(out, [[.01, .11], [.05, -.11], [.09, .11]], "offset tuy chinh")


def test_multiple_environments():
    gen = ZMPGenerator()
    pts = torch.cat([make_footsteps(0), make_footsteps(3)], dim=0)
    j_base = torch.tensor([0, 3])
    j = torch.tensor([[0, 1, 2], [3, 4, 5]])
    local = torch.tensor([[.1, .02, .1], [.1, .02, .1]])
    out = gen.zmp_walk(pts, j_base, j, local, DS_DURATION)
    assert out.shape == (2, 3, 2)
    check_close(out[0, 0], [.005, .097], "env0")
    check_close(out[1, 0], [.005, -.097], "env1")
    check_close(out[1, 1], [.025, 0.0], "env1 j=4 DS giua pha")


def test_no_input_mutation():
    gen = ZMPGenerator()
    pts = make_footsteps()
    jb = torch.tensor([0])
    jf = torch.tensor([[0, 1, 2]])
    local = torch.tensor([[.1, .02, .1]])
    originals = [x.clone() for x in (pts, jb, jf, local)]
    _ = gen.zmp_walk(pts, jb, jf, local, DS_DURATION)
    for name, now, old in zip(("footsteps", "j_base", "j_future", "local_future"),
                              (pts, jb, jf, local), originals):
        assert torch.equal(now, old), f"Ham da thay doi input {name}"


def build_horizon_example():
    """Dung scheduler + footstep planner that de lap horizon cho zmp_walk()."""
    scheduler = ContactScheduler(dt=DT, period_phase=PERIOD,
                                 shift=SHIFT, double_support=DS_RATIO)
    planner = FootstepPlanner(num_envs=1, period_phase=PERIOD, num_footsteps=14)
    generator = ZMPGenerator()

    feet0 = torch.tensor([[[0.0, 0.09, 0.0],
                           [0.0, -0.09, 0.0]]], dtype=torch.float32)
    planner.reset(torch.tensor([0]), feet0)
    cmd = torch.tensor([[0.20, 0.0]], dtype=torch.float32)

    # Khong co CP correction, khong reanchor trong bai test ZMP WALK.
    t_now = torch.tensor([SHIFT], dtype=torch.float32)
    j_now = scheduler.step_index(t_now)
    assert j_now.item() == 0
    stance = scheduler.stance_feet(j_now)
    new_step = scheduler.new_step_event(j_now, torch.tensor([-1], dtype=torch.long))
    planner.update_new_step(j_now, new_step, stance, cmd)
    footsteps, j_base = planner.footstep_sequence(j_now, cmd)

    # MPC du doan tu t+dt den t+N*dt, khong bao gom t hien tai.
    times = t_now[:, None] + DT * torch.arange(1, HORIZON + 1)[None, :]
    j_future = scheduler.step_index(times)
    local_future = scheduler.local_time_phase(times, j_future)
    zmp = generator.zmp_walk(footsteps, j_base, j_future, local_future, DS_DURATION)
    assert zmp.shape == (1, HORIZON, 2)
    assert torch.isfinite(zmp).all()
    assert j_future.max().item() == 8

    # Gia tri du kien tai 3 mau DS cua buoc j=1: t = 0.60, 0.62, 0.64.
    for t_check, expected in [(0.60, [0.005, 0.097]),
                              (0.62, [0.025, 0.0]),
                              (0.64, [0.045, -0.097])]:
        ix = torch.argmin((times[0] - t_check).abs()).item()
        assert abs(times[0, ix].item() - t_check) < 1e-5
        check_close(zmp[0, ix], expected, f"Horizon t={t_check}", atol=1e-5)

    return times[0].numpy(), zmp[0].numpy(), j_future[0].numpy(), footsteps[0].numpy()


def test_real_scheduler_and_planner():
    build_horizon_example()


def test_horizon_end_clamping():
    """Gioi han chi so ngan vuot mang, khong co nghia du doan du buoc."""
    gen = ZMPGenerator()
    pts = make_footsteps()
    j = torch.tensor([[11, 12, 14, 20]], dtype=torch.long)
    local = torch.full((1, 4), .1)
    out = gen.zmp_walk(pts, torch.tensor([0]), j, local, DS_DURATION)
    assert out.shape == (1, 4, 2) and torch.isfinite(out).all()


def plot_walk_reference():
    times, zmp, j, footsteps = build_horizon_example()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_x, ax_y, ax_xy, ax_zoom = axes.flat

    ax_x.plot(times, zmp[:, 0], color="tab:blue", linewidth=2, label="ZMP X")
    ax_y.plot(times, zmp[:, 1], color="tab:orange", linewidth=2, label="ZMP Y")
    for ax in (ax_x, ax_y):
        ax.set_xlabel("Thoi gian (s)")
        ax.set_ylabel("ZMP (m)")
        ax.grid(True, alpha=.3)
        ax.legend()
        for k in range(1, 9):
            start = SHIFT + k * (PERIOD / 2)
            end = start + DS_DURATION
            if start <= times[-1]:
                ax.axvspan(start, min(end, times[-1]), color="gray", alpha=.13)
    ax_x.set_title("ZMP X theo thoi gian | v_x = 0.20 m/s")
    ax_y.set_title("ZMP Y theo thoi gian | DS to mau xam")

    ax_xy.plot(zmp[:, 0], zmp[:, 1], ".-", markersize=3, linewidth=1.6,
               color="tab:green", label="Quy dao ZMP tham chieu")
    ax_xy.scatter(footsteps[1::2, 0] + .005, footsteps[1::2, 1] + .007,
                  c="tab:blue", marker="x", s=45, label="ZMP stance trai")
    ax_xy.scatter(footsteps[2::2, 0] + .005, footsteps[2::2, 1] - .007,
                  c="tab:orange", marker="x", s=45, label="ZMP stance phai")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Quy dao ZMP tren mat phang X-Y")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=.3)
    ax_xy.legend(fontsize=8)

    mask = (times >= .58 - 1e-5) & (times <= .68 + 1e-5)
    ax_zoom.plot(times[mask], zmp[mask, 0], "o-", label="ZMP X")
    ax_zoom.plot(times[mask], zmp[mask, 1], "s-", label="ZMP Y")
    ax_zoom.axvspan(.60, .64, color="gray", alpha=.16, label="DS j=1")
    ax_zoom.set_xlabel("Thoi gian (s)")
    ax_zoom.set_ylabel("ZMP (m)")
    ax_zoom.set_title("Phong to chuyen ZMP: trai → phai (j=1)")
    ax_zoom.grid(True, alpha=.3)
    ax_zoom.legend(fontsize=8)

    fig.suptitle("TEST ZMPGenerator.zmp_walk() | N=80, dt=0.02 s, T_DS=0.04 s", fontsize=13)
    plt.show()  # Chi hien thi, KHONG luu file


def main():
    tests = [
        test_j0_no_double_support,
        test_j1_ds_then_ss,
        test_j2_change_foot_direction,
        test_nonzero_j_base,
        test_custom_and_zero_offsets,
        test_multiple_environments,
        test_no_input_mutation,
        test_real_scheduler_and_planner,
        test_horizon_end_clamping,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} TESTS PASS")
    print("Dang mo do thi ZMP theo thoi gian va trong mat phang X-Y...")
    plot_walk_reference()


if __name__ == "__main__":
    main()
