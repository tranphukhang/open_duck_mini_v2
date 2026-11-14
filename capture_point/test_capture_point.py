"""Test tích hợp 5 module của pipeline Capture Point / ZMP / LIPM-MPC.

Đặt file cạnh 5 file sau trong thư mục capture_point/:
    contact_scheduler.py, footstep_planner.py, capture_point.py,
    zmp_generator.py, lipm_mpc.py

Chạy:
    python test_5_module_integration.py
    python test_5_module_integration.py --plot
    python test_5_module_integration.py --duration 11 --plot

Pipeline mỗi tick 0.02 s:
    ContactScheduler -> FootstepPlanner.update_new_step/reanchor_x
    -> CapturePoint.compute_dcm_error/compute_correction/correct_landing
    -> ContactScheduler.allow_footstep_adjust -> FootstepPlanner.set_landing
    -> FootstepPlanner.footstep_sequence -> ZMPGenerator.generate
    -> LIPMMPC.step -> thời điểm kế tiếp.

QUAN TRỌNG:
- Thử nghiệm tổng hợp, KHÔNG phải MuJoCo, FK đo thực, WBIK hay servo.
- CoM đo được tạo từ CoM kế hoạch cộng sai lệch vị trí/vận tốc đã biết,
  cùng hệ {P} và cùng thời điểm t_k. reanchor_x nhận tín hiệu FK GIẢ LẬP.
- KHÔNG có feedback trực tiếp trạng thái đo vào MPC; KHÔNG có Raibert.
- Không lưu ảnh, CSV hoặc kết quả ra tệp. Chỉ in và tùy chọn vẽ đồ thị.
"""

import argparse
import math

import numpy as np
import torch

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner
from capture_point import CapturePoint
from zmp_generator import ZMPGenerator
from lipm_mpc import LIPMMPC

DT = 0.02
PERIOD = 0.40
SHIFT = 0.40
DOUBLE_SUPPORT = 0.10
HORIZON = 80
N_FOOTSTEPS = 14
H = 0.15                  # Chỉ là chiều cao CoM giả lập [m]
MIN_WIDTH = 0.12          # Đặt khoảng cách ban đầu 0.13 m để test giới hạn Y
FEET_LEFT_Y = 0.065
FEET_RIGHT_Y = -0.065
VX, VY = 0.12, 0.0
REANCHOR_DX = 0.002       # Dịch x nhân tạo tại đầu mỗi bước j>0

# Các trường hợp dùng CÙNG lệnh và vị trí ban đầu, chỉ khác sai số đo.
CASE_NAMES = (
    "Baseline (e_DCM=0)",
    "CP: nhiễu vừa, thay đổi theo thời gian",
    "CP: bão hòa dương X/Y",
    "CP: bão hòa âm X/Y",
)


def check_close(actual, expected, message, atol=3e-5, rtol=1e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        delta = (actual - expected).abs().max().item()
        raise AssertionError(f"{message}: max|delta|={delta:.6g}")


def synthetic_relative(planner, stance, delta_x):
    """Tạo (CoM-foot)_FK giả lập sao cho reanchor_x thu đúng delta_x."""
    b = torch.arange(stance.shape[0], device=stance.device)
    other = 1 - stance
    rel = torch.zeros_like(planner.feet_start)
    rel[b, other, 0] = (
        planner.feet_start[b, stance, 0]
        - planner.feet_start[b, other, 0]
        + delta_x
    )
    return rel


def measurement_errors(ids, t, device):
    """Trả về (e_pos, e_vel) (B,2) giả lập, KHÔNG sử dụng MuJoCo."""
    b = len(ids)
    pos = torch.zeros(b, 2, dtype=torch.float32, device=device)
    vel = torch.zeros_like(pos)
    for k, case_id in enumerate(ids):
        if case_id == 1:
            pos[k, 0] = 0.018 + 0.005 * math.sin(2 * math.pi * t / 0.8)
            pos[k, 1] = -0.008 + 0.004 * math.cos(2 * math.pi * t / 0.8)
            vel[k] = torch.tensor((0.030, 0.010), device=device)
        elif case_id == 2:
            pos[k] = torch.tensor((0.25, 0.20), device=device)
        elif case_id == 3:
            pos[k] = torch.tensor((-0.25, -0.20), device=device)
        elif case_id != 0:
            raise ValueError(f"Unknown case_id: {case_id}")
    return pos, vel


def expected_landing(nominal, correction, stance, stance_y, width):
    """Oracle độc lập cho ràng buộc bề rộng chân (theo từng env)."""
    expected = nominal.clone()
    expected[:, :2] += correction
    for b in range(nominal.shape[0]):
        y = float(expected[b, 1])
        sy = float(stance_y[b])
        if int(stance[b]) == 0:
            expected[b, 1] = min(y, sy - width)
        else:
            expected[b, 1] = max(y, sy + width)
    return expected


@torch.no_grad()
def simulate(case_ids, duration, device="cpu", verbose=True):
    """Chạy trọn pipeline cho một batch env với các case_id cho trước."""
    B = len(case_ids)
    ticks = round(duration / DT)
    if duration < 2.0 or not math.isclose(ticks * DT, duration, abs_tol=1e-8):
        raise ValueError("--duration phải >= 2 s và chia hết cho dt=0.02 s")

    scheduler = ContactScheduler(
        dt=DT, period_phase=PERIOD, shift=SHIFT,
        double_support=DOUBLE_SUPPORT, num_envs=B, device=device,
    )
    planner = FootstepPlanner(
        num_envs=B, period_phase=PERIOD,
        num_footsteps=N_FOOTSTEPS, device=device,
    )
    cp = CapturePoint(
        com_height=H, step_gain=(0.4, 0.3),
        step_limit=(0.04, 0.03), min_step_width=MIN_WIDTH, device=device,
    )
    zmp_gen = ZMPGenerator(horizon=HORIZON)
    mpc = LIPMMPC(com_height=H, dt=DT, horizon=HORIZON, device=device)
    mpc.build_model()
    mpc.build_state_prediction()
    mpc.build_output_prediction()
    mpc.precompute_gains()

    feet = torch.tensor(
        [[[0.0, FEET_LEFT_Y, 0.0], [0.0, FEET_RIGHT_Y, 0.0]]],
        device=device, dtype=torch.float32,
    ).repeat(B, 1, 1)
    cmd = torch.tensor([[VX, VY]], device=device, dtype=torch.float32).repeat(B, 1)
    planner.reset(torch.arange(B, device=device), feet)
    scheduler.reset()

    # CoM plan tại t=0: giữa hai chân; (B,2,3)=[position,velocity,acceleration].
    com_state = torch.zeros(B, 2, 3, dtype=torch.float32, device=device)
    com_state[:, :, 0] = feet[:, :, :2].mean(dim=1)
    last_step = torch.full((B,), -1, dtype=torch.long, device=device)

    events = [[] for _ in range(B)]
    footprints = [[] for _ in range(B)]
    counts = {name: 0 for name in (
        "shift", "ds", "ss", "adjust", "freeze", "saturated", "width_limited",
        "reanchor", "preview_changed", "mpc_changed",
    )}
    max_step_solve_diff = 0.0
    track = {k: [] for k in (
        "time", "com", "zmp", "zmp_ref", "vel", "dcm_error",
        "correction", "landing", "nominal", "allow", "stance",
    )}

    for k in range(ticks):
        t = scheduler.t.clone()
        t0 = float(t[0].item())
        check_close(t, torch.full_like(t, k * DT), "Thời gian scheduler", atol=5e-6)
        j_now = scheduler.step_index(t)
        local = scheduler.local_time_phase(t, j_now)
        shift, ds, ss = scheduler.phase_state(j_now, local)
        stance = scheduler.stance_feet(j_now)
        contacts = scheduler.contact_state(stance, ss)
        new_step = scheduler.new_step_event(j_now, last_step)

        assert torch.all(shift.int() + ds.int() + ss.int() == 1), "Các pha phải loại trừ nhau"
        assert torch.equal(contacts.sum(1), torch.where(ss, 1, 2)), "Sai contact schedule"
        assert torch.equal(stance, j_now.clamp_min(0).remainder(2)), "Sai chân stance"
        counts["shift"] += int(shift[0])
        counts["ds"] += int(ds[0])
        counts["ss"] += int(ss[0])

        # 1. Xác nhận hoàn thành bước cũ, tạo landing danh định mới.
        planner.update_new_step(j_now, new_step, stance, cmd)
        for b in range(B):
            if bool(new_step[b]):
                jb = int(j_now[b])
                events[b].append(jb)
                foot = planner.feet_start[b, int(stance[b])].clone()
                footprints[b].append(foot.cpu().numpy())

        # 2. reanchor_x trong cùng hệ {P}, trước CP.
        before = com_state.clone()
        fk_relative = synthetic_relative(planner, stance, REANCHOR_DX)
        fk_valid = torch.ones(B, dtype=torch.bool, device=device)
        dx = planner.reanchor_x(j_now, new_step, stance, fk_relative, fk_valid, com_state)
        expected_dx = torch.where(new_step & (j_now > 0),
                                  torch.full_like(dx, REANCHOR_DX), torch.zeros_like(dx))
        check_close(dx, expected_dx, "reanchor đúng ở đầu bước j>0", atol=1e-6)
        check_close(com_state[:, 0, 0] - before[:, 0, 0], dx,
                    "reanchor phải dịch cả CoM-plan theo X", atol=1e-6)
        check_close(com_state[:, 1], before[:, 1], "reanchor không đổi trục Y", atol=1e-6)
        counts["reanchor"] += int((dx.abs() > 1e-7).sum())

        # 3. Dữ liệu đo GIẢ LẬP tại t_k trong hệ {P}.
        pos_err, vel_err = measurement_errors(case_ids, t0, device)
        com_meas = torch.zeros(B, 3, dtype=torch.float32, device=device)
        vel_meas = torch.zeros_like(com_meas)
        com_meas[:, :2] = com_state[:, :, 0] + pos_err
        com_meas[:, 2] = H
        vel_meas[:, :2] = com_state[:, :, 1] + vel_err

        dcm_meas, dcm_plan, dcm_err = cp.compute_dcm_error(
            com_meas, vel_meas, com_state
        )
        expected_error = pos_err + vel_err / cp.omega
        check_close(dcm_err, expected_error, "Sai số DCM giả lập", atol=5e-6)
        check_close(dcm_meas - dcm_plan, dcm_err, "Dấu DCM measured - planned")
        if 0 in case_ids:
            b0 = case_ids.index(0)
            check_close(dcm_err[b0], torch.zeros(2, device=device), "CP baseline=0", atol=1e-6)

        # 4. Hiệu chỉnh CP và khóa cập nhật ở chu kỳ sát thời điểm đặt chân.
        correction = cp.compute_correction(dcm_err)
        expected_corr = torch.maximum(
            torch.minimum(cp.step_gain * expected_error, cp.step_limit),
            -cp.step_limit,
        )
        check_close(correction, expected_corr, "Gain + clamp CP", atol=2e-6)
        counts["saturated"] += int(((cp.step_gain * dcm_err).abs() > cp.step_limit + 1e-6).any(1).sum())

        batch = torch.arange(B, device=device)
        stance_y = planner.feet_start[batch, stance, 1]
        nominal = planner.nominal_landing.clone()
        correct = cp.correct_landing(nominal, correction, stance, stance_y)
        expect = expected_landing(nominal, correction, stance, stance_y, MIN_WIDTH)
        check_close(correct, expect, "Landing sau CP + giới hạn bề rộng chân", atol=2e-6)
        check_close(planner.nominal_landing, nominal, "CP không mutate nominal_landing")
        check_close(correct[:, 2], nominal[:, 2], "CP không đổi landing Z")
        raw_y = nominal[:, 1] + correction[:, 1]
        counts["width_limited"] += int(((correct[:, 1] - raw_y).abs() > 1e-6).sum())

        allow = scheduler.allow_footstep_adjust(t, j_now)
        next_j = scheduler.step_index(t + DT)
        assert torch.equal(allow, (next_j >= 0) & (next_j == j_now)), "Sai gate allow_adjust"
        updated_before = planner.updated_landing.clone()
        planner.set_landing(correct, allow)
        check_close(planner.updated_landing,
                    torch.where(allow[:, None], correct, updated_before),
                    "Gate set_landing", atol=1e-6)
        counts["adjust"] += int(allow.sum())
        counts["freeze"] += int(((j_now >= 0) & ~allow).sum())
        if bool((j_now >= 0).any()):
            delta_y = planner.updated_landing[:, 1] - stance_y
            ok_width = torch.where(stance == 0, delta_y <= -MIN_WIDTH + 2e-6,
                                   delta_y >= MIN_WIDTH - 2e-6)
            assert torch.all(ok_width), "Landing cuối cùng vi phạm min_step_width"

        # 5. Chuỗi bước và ZMP tham chiếu cho t_{k+1} .. t_{k+N}.
        footsteps, j_base = planner.footstep_sequence(j_now, cmd)
        assert footsteps.shape == (B, N_FOOTSTEPS, 3)
        assert torch.equal(j_base, j_now.clamp_min(0))
        check_close(footsteps[:, 0], planner.feet_start[batch, 1 - stance], "P_-1")
        check_close(footsteps[:, 1], planner.feet_start[batch, stance], "P_0")
        walking = j_now >= 0
        if walking.any():
            check_close(footsteps[walking, 2], planner.updated_landing[walking], "P_1=updated_landing")
        zmp_ref = zmp_gen.generate(footsteps, planner.feet0, j_base, scheduler)
        assert zmp_ref.shape == (B, HORIZON, 2) and bool(torch.isfinite(zmp_ref).all()), "ZMP preview không hợp lệ"

        # Khi CP thay đổi landing, dự báo ZMP xa hơn phải thay đổi theo.
        if allow.all() and 0 in case_ids and 1 in case_ids:
            base, perturbed = case_ids.index(0), case_ids.index(1)
            if not torch.allclose(planner.updated_landing[base], planner.updated_landing[perturbed], atol=1e-5):
                if not torch.allclose(zmp_ref[base], zmp_ref[perturbed], atol=1e-7):
                    counts["preview_changed"] += 1

        # 6. MPC dự đoán t_{k+1} từ trạng thái t_k (KHÔNG feedback đo trực tiếp).
        old = com_state.clone()
        vel_ref = cmd[:, None, :].expand(-1, HORIZON, -1)
        com_next = mpc.step(old, zmp_ref, vel_ref)
        assert com_next.shape == (B, 2, 3) and bool(torch.isfinite(com_next).all()), "MPC không hợp lệ"
        check_close(com_state, old, "MPC.step không mutate com_state")
        jerk = (com_next[:, :, 2] - old[:, :, 2]) / DT
        model_next = old @ mpc.A.T + jerk[..., None] * mpc.B
        check_close(com_next, model_next, "Mô hình LIPM: x_next=Ax+Bu", atol=5e-5)

        if k % 25 == 0 or k == ticks - 1:
            x_flat = old.reshape(B * 2, 3)
            z_flat = zmp_ref.transpose(1, 2).reshape(B * 2, HORIZON)
            v_flat = vel_ref.transpose(1, 2).reshape(B * 2, HORIZON)
            U, X = mpc.solve(x_flat, z_flat, v_flat)
            from_solve = X[:, 0, :].reshape(B, 2, 3)
            err = float((from_solve - com_next).abs().max())
            max_step_solve_diff = max(err, max_step_solve_diff)
            check_close(com_next, from_solve, "MPC.step vs MPC.solve first step", atol=1e-4)
            check_close(jerk, U[:, 0].reshape(B, 2), "Jerk đầu MPC", atol=2e-4)

        if allow.all() and 0 in case_ids and 1 in case_ids:
            b0, b1 = case_ids.index(0), case_ids.index(1)
            if not torch.allclose(com_next[b0], com_next[b1], atol=1e-7):
                counts["mpc_changed"] += 1

        zmp_out = com_next[:, :, 0] - (H / 9.81) * com_next[:, :, 2]
        for name, array in (
            ("time", t), ("com", com_next[:, :, 0]), ("zmp", zmp_out),
            ("zmp_ref", zmp_ref[:, 0]), ("vel", com_next[:, :, 1]),
            ("dcm_error", dcm_err), ("correction", correction),
            ("landing", planner.updated_landing), ("nominal", planner.nominal_landing),
            ("allow", allow), ("stance", stance),
        ):
            track[name].append(array.detach().cpu().numpy().copy())

        com_state = com_next
        last_step = torch.where(j_now >= 0, j_now, last_step)
        scheduler.count()

    assert counts["shift"] == int(round(SHIFT / DT)), "Sai thời gian Shift"
    assert counts["adjust"] > 0 and counts["freeze"] > 0, "Chưa test cả cho phép và khóa landing"
    if 2 in case_ids or 3 in case_ids:
        assert counts["saturated"] > 0, "Chưa kích hoạt bão hòa CP"
        assert counts["width_limited"] > 0, "Chưa kích hoạt giới hạn bề rộng bước chân"
    if 0 in case_ids and 1 in case_ids:
        assert counts["preview_changed"] > 0, "CP không lan truyền vào ZMP reference"
        assert counts["mpc_changed"] > 0, "CP không lan truyền đến MPC"

    for b in range(B):
        assert events[b] == list(range(len(events[b]))), "Sự kiện bước thiếu hoặc lặp"
        assert len(events[b]) >= 5, "Không đủ bước để test tích hợp"
    assert counts["reanchor"] == sum(max(0, len(x) - 1) for x in events), "Sai số sự kiện reanchor"
    traces = {name: np.asarray(value) for name, value in track.items()}
    assert all(np.isfinite(traces[k]).all() for k in ("com", "zmp", "zmp_ref", "vel", "dcm_error"))

    if verbose:
        print(f"[PASS] B={B}, {ticks} tick, t={duration:.2f}s, events={list(map(len, events))}")
        print("       counts:", counts)
        print(f"       max |MPC.step - MPC.solve()[0]| = {max_step_solve_diff:.3e}")
        for b, case in enumerate(case_ids):
            rmse = np.sqrt(np.mean((traces["zmp"][:, b] - traces["zmp_ref"][:, b])**2, axis=0))
            print(f"       {CASE_NAMES[case]}: RMSE ZMP X/Y = {rmse.round(5).tolist()} m")
    return {"traces": traces, "events": events, "footprints": footprints, "counts": counts}


def test_batch_independence(result, duration, device):
    """Hai env được chọn phải khớp khi chạy lại riêng B=1."""
    full = result["traces"]
    for batch_index, case in ((0, 0), (2, 2)):
        separate = simulate([case], duration, device, verbose=False)["traces"]
        for key in ("com", "zmp", "zmp_ref", "vel", "dcm_error", "correction", "landing"):
            got = full[key][:, batch_index]
            want = separate[key][:, 0]
            if not np.allclose(got, want, atol=5e-5, rtol=1e-5):
                e = np.max(np.abs(got - want))
                raise AssertionError(f"Batch independence: env={case}, {key}, max_err={e:.3e}")
    print("[PASS] Batch independence: B=4 đối chiếu B=1 (baseline và CP bão hòa)")


def plot_results(result):
    import matplotlib.pyplot as plt
    tr = result["traces"]
    t = tr["time"][:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for idx in range(4):
        axes[0, 0].plot(t, tr["com"][:, idx, 0], label=CASE_NAMES[idx])
        axes[0, 1].plot(t, tr["com"][:, idx, 1], label=CASE_NAMES[idx])
        axes[1, 0].plot(t, tr["correction"][:, idx, 0], label=CASE_NAMES[idx])
        axes[1, 1].plot(t, tr["correction"][:, idx, 1], label=CASE_NAMES[idx])
    for ax, title, ylabel in zip(
        axes.flat,
        ("CoM kế hoạch - X", "CoM kế hoạch - Y", "CP correction - X", "CP correction - Y"),
        ("CoM X [m]", "CoM Y [m]", "Δp X [m]", "Δp Y [m]"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Thời gian [s]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Test 5 module - DCM/CP tác động lên CoM kế hoạch (dữ liệu tổng hợp)")
    fig.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, case in zip(axes2, (0, 1)):
        com = tr["com"][:, case]
        zref = tr["zmp_ref"][:, case]
        zmp = tr["zmp"][:, case]
        ax.plot(com[:, 0], com[:, 1], label="CoM kế hoạch", linewidth=1.7)
        ax.plot(zref[:, 0], zref[:, 1], label="ZMP tham chiếu", alpha=0.8)
        ax.plot(zmp[:, 0], zmp[:, 1], "--", label="ZMP từ LIPM", alpha=0.8)
        feet = np.asarray(result["footprints"][case])
        if len(feet):
            ax.scatter(feet[:, 0], feet[:, 1], s=18, label="Chân stance (kế hoạch)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(CASE_NAMES[case])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig2.suptitle("Footstep → ZMP reference → LIPM-MPC (không mô phỏng động lực học robot)")
    fig2.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Tích hợp 5 module Contact/Footstep/CP/ZMP/LIPM-MPC")
    p.add_argument("--duration", type=float, default=6.0, help="Thời gian chạy [s], chia hết cho 0.02")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--plot", action="store_true", help="Hiển thị đồ thị (không lưu file)")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("Máy không có CUDA; hãy dùng --device cpu")
    torch.manual_seed(2026)
    if args.device == "cpu":
        torch.set_num_threads(1)
    print("=== TEST TÍCH HỢP 5 MODULE: ContactScheduler + FootstepPlanner + CapturePoint + ZMPGenerator + LIPMMPC ===")
    print("CoM đo và FK/reanchor đều GIẢ LẬP; không có MuJoCo, WBIK, Raibert hoặc MPC state feedback.\n")
    result = simulate([0, 1, 2, 3], args.duration, args.device)
    test_batch_independence(result, args.duration, args.device)
    print("\n[PASS] Các kiểm tra tích hợp hoàn tất.")
    print("Lưu ý: RMSE ZMP chỉ là chỉ số toán học của bộ lập kế hoạch, không chứng minh robot đi ổn định.")
    if args.plot:
        plot_results(result)


if __name__ == "__main__":
    main()
