"""Kiểm thử tích hợp 4 module trong capture_point (CHƯA có CP/WBIK/FK thực).

Chạy sau khi đặt file này trong cùng thư mục với 4 file .py:
    python test_integrated_contact_footstep_zmp_lipm.py
    python test_integrated_contact_footstep_zmp_lipm.py --no-plot

Luồng kiểm thử:
    ContactScheduler -> FootstepPlanner -> ZMPGenerator -> LIPMMPC.step()

- Vị trí bàn chân và chiều cao CoM ban đầu là DỮ LIỆU GIẢ LẬP, không phải FK robot.
- Có kịch bản tùy chọn dùng CP correction / reanchor GIẢ LẬP chỉ để kiểm tra
  kết nối dữ liệu. KHÔNG phải thuật toán Capture Point và KHÔNG phải FK.
- Không gọi WBIK hoặc MuJoCo; không kiểm chứng ổn định vật lý/contact thực.
- Không ghi file CSV hoặc file ảnh. Chỉ in kết quả và hiện Matplotlib.
"""

import argparse
import math
import sys

import numpy as np
import torch

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner
from zmp_generator import ZMPGenerator
from lipm_mpc import LIPMMPC


DT = 0.02
PERIOD_PHASE = 0.4
SHIFT = 0.4
DOUBLE_SUPPORT = 0.1
N = 80
NUM_FOOTSTEPS = 14
COM_HEIGHT = 0.15  # m, giá trị thử nghiệm; KHÔNG phải chiều cao FK thực

# Chân trái (y>0) và chân phải (y<0), trong hệ {P} giả lập.
FEET_0 = torch.tensor([
    [[0.00, +0.09, 0.0], [0.00, -0.09, 0.0]],
    [[0.02, +0.11, 0.0], [0.02, -0.11, 0.0]],
], dtype=torch.float32)
COMMANDS = torch.tensor([
    [0.15, 0.00],  # env 0: đi tới
    [0.09, 0.02],  # env 1: đi tới và lệch theo y
], dtype=torch.float32)


def check_close(actual, expected, name, atol=2e-5, rtol=2e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        err = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: max |error| = {err:.3e}")


def make_pipeline(num_envs, device):
    scheduler = ContactScheduler(
        dt=DT, period_phase=PERIOD_PHASE, shift=SHIFT,
        double_support=DOUBLE_SUPPORT, num_envs=num_envs, device=device,
    )
    planner = FootstepPlanner(
        num_envs=num_envs, period_phase=PERIOD_PHASE,
        num_footsteps=NUM_FOOTSTEPS, device=device,
    )
    generator = ZMPGenerator(
        zmp_offset_x=0.005, zmp_offset_y_out=0.007, horizon=N,
    )
    mpc = LIPMMPC(com_height=COM_HEIGHT, dt=DT, horizon=N, device=device)
    # Hiện tại __init__ chỉ lưu tham số, cần gọi bốn hàm theo đúng thứ tự.
    mpc.build_model()
    mpc.build_state_prediction()
    mpc.build_output_prediction()
    mpc.precompute_gains()
    return scheduler, planner, generator, mpc


def synthetic_relative_for_reanchor(planner, stance, imposed_dx):
    """Sinh dữ liệu giống chữ ký FK để KÍCH HOẠT API reanchor_x.

    Dữ liệu được tạo đại số nhằm thu được delta_x = imposed_dx.
    Đây KHÔNG phải tọa độ CoM tương đối đo/ước lượng từ robot.
    """
    relative = torch.zeros_like(planner.feet_start)
    batch = torch.arange(stance.shape[0], device=stance.device)
    swing = 1 - stance
    relative[batch, swing, 0] = (
        planner.feet_start[batch, stance, 0]
        - planner.feet_start[batch, swing, 0]
        + imposed_dx
    )
    return relative


def run_case(label, feet_init, commands, duration, device, synthetic_adjust=False):
    B = feet_init.shape[0]
    ticks = int(round(duration / DT))
    assert ticks > int(round((SHIFT + PERIOD_PHASE) / DT)), "Thời gian chạy quá ngắn"
    scheduler, planner, generator, mpc = make_pipeline(B, device)
    feet_init, commands = feet_init.to(device), commands.to(device)
    planner.reset(torch.arange(B, device=device), feet_init)
    scheduler.reset()

    # Thay FK bằng trung điểm hai bàn chân, chỉ dùng cho kiểm thử tích hợp.
    com_state = torch.zeros(B, 2, 3, dtype=torch.float32, device=device)
    com_state[:, :, 0] = feet_init[:, :, :2].mean(dim=1)

    last_step = torch.full((B,), -1, dtype=torch.long, device=device)
    previous_j = torch.full((B,), -1, dtype=torch.long, device=device)
    events = [[] for _ in range(B)]
    reanchor_sum = torch.zeros(B, device=device)
    reanchor_count = torch.zeros(B, dtype=torch.long, device=device)
    cp_apply_count = torch.zeros(B, dtype=torch.long, device=device)
    max_solve_step_error = 0.0
    max_jerk_error = 0.0
    max_model_error = 0.0

    traces = {key: [] for key in (
        "time", "com", "zmp", "zmp_ref", "velocity", "acceleration",
        "jerk", "command", "stance", "step_index", "contacts"
    )}
    previews = {}
    foot_landing = [[] for _ in range(B)]

    # Không tạo các module khác: CP, FK và WBIK chưa nằm trong vòng kín này.
    for tick in range(ticks):
        t = scheduler.t.clone()
        j_now = scheduler.step_index(t)
        local = scheduler.local_time_phase(t, j_now)
        stance = scheduler.stance_feet(j_now)
        new_step = scheduler.new_step_event(j_now, last_step)

        shift, ds, ss = scheduler.phase_state(j_now, local)
        contacts = scheduler.contact_state(stance, ss)
        assert contacts.shape == (B, 2)
        assert torch.all((shift.int() + ds.int() + ss.int()) == 1)
        assert torch.all(contacts.sum(dim=1) == torch.where(ss, 1, 2))
        assert torch.all(j_now >= previous_j), "Chỉ số bước đi lùi"
        assert torch.all((j_now - previous_j) <= 1), "Bỏ qua chỉ số bước"

        # 1) Bắt đầu bước mới, cập nhật landing.
        planner.update_new_step(j_now, new_step, stance, commands)

        # 2) reanchor_x chỉ có dữ liệu thực sau FK. Mặc định không kích hoạt.
        #    Kịch bản synthetic_adjust chủ động tạo 'FK' giả lập để thử API.
        before_reanchor = com_state.clone()
        synthetic_dx = 0.003 if synthetic_adjust else 0.0
        relative = synthetic_relative_for_reanchor(planner, stance, synthetic_dx)
        fk_valid = torch.full((B,), synthetic_adjust, dtype=torch.bool, device=device)
        delta = planner.reanchor_x(
            j_now, new_step, stance, relative, fk_valid, com_state
        )
        check_close(com_state[:, 0, 0] - before_reanchor[:, 0, 0],
                    delta, "CoM x phải đổi cùng gốc với bàn chân", atol=2e-6)
        check_close(com_state[:, 1, :], before_reanchor[:, 1, :],
                    "reanchor_x không được thay đổi trục Y", atol=2e-6)
        if not synthetic_adjust:
            assert torch.all(delta == 0), "Không có FK thì reanchor phải tắt"
        reanchor_sum += delta
        reanchor_count += (delta.abs() > 1e-7).long()

        # 3) Hiệu chỉnh landing giả lập để kiểm thử kết nối set_landing().
        #    KHÔNG phải công thức CP thực tế.
        allow = scheduler.allow_footstep_adjust(t, j_now) & (j_now >= 0)
        if synthetic_adjust:
            corrected = planner.nominal_landing.clone()
            corrected[:, 0] += 0.008
            corrected[:, 1] += torch.where(
                stance == 0, -0.004, +0.004
            )
            old_landing = planner.updated_landing.clone()
            planner.set_landing(corrected, allow)
            check_close(planner.updated_landing,
                        torch.where(allow[:, None], corrected, old_landing),
                        "Áp dụng landing giả lập")
            cp_apply_count += allow.long()

        # 4) Sinh chuỗi bước chân từ planner và tham chiếu ZMP horizon.
        footsteps, j_base = planner.footstep_sequence(j_now, commands)
        assert footsteps.shape == (B, NUM_FOOTSTEPS, 3)
        check_close(j_base, j_now.clamp_min(0), "j_base")
        zmp_ref = generator.generate(footsteps, planner.feet0, j_base, scheduler)
        assert zmp_ref.shape == (B, N, 2)
        assert torch.isfinite(zmp_ref).all(), "ZMP ref chứa NaN/Inf"

        # 5) Vận tốc tham chiếu lấy từ lệnh người dùng và lặp trong horizon.
        vel_ref = commands[:, None, :].expand(-1, N, -1)

        # 6) Một bước MPC. Không phản hồi từ CoM đo; trạng thái được lưu ngoài MPC.
        old_state = com_state.clone()
        next_state = mpc.step(old_state, zmp_ref, vel_ref)
        assert next_state.shape == (B, 2, 3)
        assert torch.isfinite(next_state).all(), "Trạng thái CoM chứa NaN/Inf"
        check_close(com_state, old_state, "step() không được mutate input")

        # 7) Đối chiếu định kỳ với solve() (giải full horizon cho từng trục).
        if tick % 10 == 0 or tick == ticks - 1:
            x_flat = old_state.reshape(B * 2, 3)
            z_flat = zmp_ref.transpose(1, 2).reshape(B * 2, N)
            v_flat = vel_ref.transpose(1, 2).reshape(B * 2, N)
            U, X = mpc.solve(x_flat, z_flat, v_flat)
            from_solve = X[:, 0, :].reshape(B, 2, 3)
            err = (next_state - from_solve).abs().max().item()
            max_solve_step_error = max(max_solve_step_error, err)
            check_close(next_state, from_solve, "step() vs solve()[0]", atol=3e-5)
            jerk_solve = U[:, 0].reshape(B, 2)
            jerk_actual = (next_state[:, :, 2] - old_state[:, :, 2]) / DT
            jerk_err = (jerk_actual - jerk_solve).abs().max().item()
            max_jerk_error = max(max_jerk_error, jerk_err)
            check_close(jerk_actual, jerk_solve, "jerk đầu tiên vs solve", atol=1e-4)

        # Xác minh phương trình A*x + B*u và ZMP=c-h/g*a.
        u0 = (next_state[:, :, 2] - old_state[:, :, 2]) / DT
        model_next = old_state @ mpc.A.T + u0[..., None] * mpc.B
        model_err = (next_state - model_next).abs().max().item()
        max_model_error = max(max_model_error, model_err)
        check_close(next_state, model_next, "Động lực học LIPM", atol=3e-6)
        predicted_zmp = next_state[:, :, 0] - (COM_HEIGHT / 9.81) * next_state[:, :, 2]

        for b in range(B):
            if bool(new_step[b]):
                events[b].append(int(j_now[b]))
                # Ghi vị trí chân stance tại sự kiện; không phải vị trí thực tế.
                foot_landing[b].append(planner.feet_start[b, int(stance[b])].detach().cpu().numpy().copy())

        if tick in (0, 19, 29, 70, 140):
            previews[tick] = (float(t[0]), zmp_ref[0].detach().cpu().numpy().copy())

        traces["time"].append((t + DT).cpu().numpy().copy())
        traces["com"].append(next_state[:, :, 0].detach().cpu().numpy().copy())
        traces["zmp"].append(predicted_zmp.detach().cpu().numpy().copy())
        traces["zmp_ref"].append(zmp_ref[:, 0, :].detach().cpu().numpy().copy())
        traces["velocity"].append(next_state[:, :, 1].detach().cpu().numpy().copy())
        traces["acceleration"].append(next_state[:, :, 2].detach().cpu().numpy().copy())
        traces["jerk"].append(u0.detach().cpu().numpy().copy())
        traces["command"].append(commands.detach().cpu().numpy().copy())
        traces["stance"].append(stance.cpu().numpy().copy())
        traces["step_index"].append(j_now.cpu().numpy().copy())
        traces["contacts"].append(contacts.cpu().numpy().copy())

        com_state = next_state
        last_step = torch.where(j_now >= 0, j_now, last_step)
        previous_j = j_now
        scheduler.count()

    for key, samples in traces.items():
        traces[key] = np.asarray(samples)
    for b in range(B):
        assert len(events[b]) >= 8, "Chưa đủ sự kiện bước để kiểm tra walking"
        assert events[b] == list(range(len(events[b]))), "Sự kiện bước bị lặp/mất bước"
        if synthetic_adjust:
            check_close(reanchor_sum[b:b+1],
                        torch.tensor([(len(events[b]) - 1) * 0.003], device=device),
                        "Tổng dịch chuyển reanchor", atol=2e-4)
            assert cp_apply_count[b] > 0, "set_landing chưa được kích hoạt"

    err_zmp = traces["zmp"] - traces["zmp_ref"]
    err_vel = traces["velocity"] - traces["command"]
    rmse_zmp = np.sqrt(np.mean(err_zmp ** 2, axis=0))
    rmse_vel = np.sqrt(np.mean(err_vel ** 2, axis=0))
    assert np.isfinite(rmse_zmp).all() and np.isfinite(rmse_vel).all()
    assert np.isfinite(traces["jerk"]).all()
    print(f"[PASS] {label}: B={B}, ticks={ticks}, events={list(map(len, events))}")
    print("       max|step - solve[0]|=%.3e, max|jerk - solve jerk[0]|=%.3e" %
          (max_solve_step_error, max_jerk_error))
    print("       max|x_next - (Ax+Bu)|=%.3e, reanchor_sum=%s" %
          (max_model_error, reanchor_sum.detach().cpu().numpy().round(4).tolist()))
    print("       RMSE ZMP [m] :", np.round(rmse_zmp, 5).tolist())
    print("       RMSE vel [m/s]:", np.round(rmse_vel, 5).tolist())
    return {**traces, "label": label, "previews": previews,
            "events": events, "landing": foot_landing,
            "rmse_zmp": rmse_zmp, "rmse_vel": rmse_vel}


def test_batch_independence(batched, feet_init, commands, duration, device):
    """B=2 phải tương đương hai lần chạy B=1 độc lập."""
    for b in range(feet_init.shape[0]):
        single = run_case(
            f"độc lập env {b}", feet_init[b:b+1], commands[b:b+1],
            duration, device, synthetic_adjust=False
        )
        for key in ("com", "zmp", "zmp_ref", "velocity", "acceleration", "jerk"):
            got = batched[key][:, b, :]
            want = single[key][:, 0, :]
            max_err = np.max(np.abs(got - want))
            assert np.allclose(got, want, atol=3e-5, rtol=1e-5), (
                f"Batch khác chạy độc lập: env={b}, {key}, max error={max_err}"
            )
    print("[PASS] Kiểm tra B=2 tương đương hai môi trường B=1 độc lập")


def plot_results(baseline, adjusted):
    import matplotlib.pyplot as plt
    data = baseline
    t = data["time"][:, 0]
    # Chỉ vẽ env 0; env 1 vẫn được kiểm thử bằng assertions.
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    labels = ["X", "Y"]
    for d in range(2):
        ax = axs[0, d]
        ax.plot(t, data["zmp_ref"][:, 0, d], label="ZMP tham chiếu", linewidth=1.7)
        ax.plot(t, data["zmp"][:, 0, d], label="ZMP từ LIPM", linestyle="--")
        ax.plot(t, data["com"][:, 0, d], label="CoM dự đoán", alpha=0.8)
        ax.set_title(f"ZMP và CoM - trục {labels[d]}")
        ax.set_ylabel("Vị trí [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axs[1, d]
        ax.plot(t, data["velocity"][:, 0, d], label="Vận tốc CoM")
        ax.plot(t, data["command"][:, 0, d], "--", label="Vận tốc đặt")
        ax.set_ylabel("Vận tốc [m/s]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axs[2, d]
        ax.plot(t, data["jerk"][:, 0, d], label="Jerk tối ưu")
        ax.set_ylabel("Jerk [m/s³]")
        ax.set_xlabel("Thời gian [s]")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Test tích hợp cơ sở: Contact → Footstep → ZMP → LIPM-MPC (env 0)")
    fig.tight_layout()

    fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, result, name in zip(axes, (baseline, adjusted), ("Không hiệu chỉnh", "Hiệu chỉnh GIẢ LẬP")):
        com = result["com"][:, 0]
        z_ref = result["zmp_ref"][:, 0]
        z = result["zmp"][:, 0]
        ax.plot(com[:, 0], com[:, 1], label="Quỹ đạo CoM", linewidth=1.8)
        ax.plot(z_ref[:, 0], z_ref[:, 1], label="ZMP tham chiếu", alpha=0.75)
        ax.plot(z[:, 0], z[:, 1], linestyle="--", label="ZMP từ LIPM", alpha=0.75)
        feet = np.array(result["landing"][0])
        if len(feet):
            ax.scatter(feet[:, 0], feet[:, 1], s=18, label="Vị trí chân stance (kế hoạch)")
        ax.set_title(name)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(fontsize=8)
    fig2.suptitle("Mặt phẳng XY - chỉ dùng mô hình/dữ liệu giả lập")
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(11, 4))
    for tick, (t0, preview) in baseline["previews"].items():
        lookahead_t = t0 + DT * np.arange(1, N + 1)
        ax3.plot(lookahead_t, preview[:, 1], label=f"Preview tại t={t0:.2f}s")
    ax3.plot(t, baseline["zmp_ref"][:, 0, 1], "k--", linewidth=1.1,
             label="ZMP ref thực thi theo từng chu kỳ")
    ax3.set_title("Các horizon ZMP tham chiếu (trục Y, env 0)")
    ax3.set_xlabel("Thời gian [s]")
    ax3.set_ylabel("ZMP Y [m]")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test tích hợp 4 module Capture Point/ZMP/LIPM-MPC")
    parser.add_argument("--no-plot", action="store_true", help="Chỉ chạy assertions, không vẽ đồ thị")
    parser.add_argument("--duration", type=float, default=5.0, help="Thời gian giả lập, giây (mặc định: 5)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA không khả dụng; hãy sử dụng --device cpu")
    torch.manual_seed(2026)
    torch.set_num_threads(1)  # tránh overhead thread CPU cho các ma trận nhỏ (N=80)

    print("\n=== INTEGRATION TEST: ContactScheduler + FootstepPlanner + ZMPGenerator + LIPMMPC ===")
    print("Dữ liệu feet0/h là giả lập; chưa kiểm chứng MuJoCo, contact thực hoặc WBIK.\n")
    baseline = run_case("Baseline", FEET_0, COMMANDS, args.duration, args.device)
    test_batch_independence(baseline, FEET_0, COMMANDS, args.duration, args.device)
    adjusted = run_case("Synthetic landing + reanchor", FEET_0, COMMANDS,
                        args.duration, args.device, synthetic_adjust=True)
    print("\n[PASS] Toàn bộ kiểm thử tích hợp đã hoàn thành.")
    print("Lưu ý: Đây là MPC không ràng buộc ZMP; RMSE không chứng minh contact ổn định.")
    if not args.no_plot:
        plot_results(baseline, adjusted)


if __name__ == "__main__":
    main()
