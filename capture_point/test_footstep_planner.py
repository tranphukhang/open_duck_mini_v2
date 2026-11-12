"""Integration test: ContactScheduler + FootstepPlanner, mo phong 11 giay.

Dat file nay cung thu muc voi:
    capture_point/contact_scheduler.py
    capture_point/footstep_planner.py

Chay:
    python test_contact_scheduler_footstep_planner_11s.py

Bon case:
    1. delta_x = 0.0 m, khong hieu chinh CP
    2. delta_x = 0.2 m, khong hieu chinh CP
    3. delta_x = 0.0 m, co hieu chinh CP co dinh
    4. delta_x = 0.2 m, co hieu chinh CP co dinh

Quan trong:
  * delta_x la gia tri gia lap cho MOI su kien reanchor khi j > 0
    (moi 0.2 s), khong phai moi tick 0.02 s.
  * Khong chay mo hinh FK / MPC / CP that. Tuong duong dau vao duoc
    tao co chu dich de kiem thu giao tiep giua hai module.
  * Da hieu chinh CP: landing = nominal_landing + [CP_DX, CP_DY, 0]
    tai cac tick ContactScheduler cho phep. Khong cong don theo tick.
  * Chi hien thi matplotlib sau khi chay; khong luu anh hay CSV.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner

# ---------------------------------------------------------------------------
# Tham so thu nghiem (co the sua truc tiep tai day)
# ---------------------------------------------------------------------------
DT = 0.02                       # Chu ky dieu khien [s]
DURATION = 11.0                 # Thoi gian mo phong [s]
PERIOD_PHASE = 0.4              # Chu ky hai buoc (moi buoc 0.2 s)
SHIFT = 0.4                     # Chuyen trong luong ban dau [s]
DOUBLE_SUPPORT = 0.1           # Ty le DSP so voi PERIOD_PHASE
NUM_FOOTSTEPS = 14             # Chieu dai chuoi du doan

VX, VY = 0.20, 0.00             # Van toc yeu cau [m/s]
CP_DX, CP_DY = 0.020, 0.015     # Hieu chinh CP CO DINH cho moi buoc [m]
FEET_INITIAL = (                # Vi tri foot site trong he {P} [m]
    (0.0,  0.09, 0.0),          # Chan trai (L = 0)
    (0.0, -0.09, 0.0),          # Chan phai (R = 1)
)

# ---------------------------------------------------------------------------
# Ham ho tro
# ---------------------------------------------------------------------------

def check(condition, message):
    """Bao loi co thong tin khi mot dieu kien tich hop khong dung."""
    if not bool(condition):
        raise AssertionError(message)


def synthetic_fk_relative(planner, stance, delta_x_target):
    """Gia lap (CoM - foot)_FK sao cho reanchor_x() tra delta_x_target.

    Cong thuc cua FootstepPlanner:
        landed_x = x_old_stance + rel_old_stance - rel_new_stance
        delta_x = landed_x - x_new_stance

    Trong do:
        new_stance = stance (chan swing truoc khi hoan thanh buoc)
        old_stance = 1 - stance (chan stance cua buoc truoc)

    Chon rel_new_stance = 0 va:
        rel_old_stance = x_new_stance - x_old_stance + delta_x_target

    Day la DAU VAO GIA LAP, khong phai du lieu do tu FK that.
    """
    rel = torch.zeros_like(planner.feet_start)
    old_stance = 1 - stance
    b = torch.arange(stance.numel(), device=stance.device)
    rel[b, old_stance, 0] = (
        planner.feet_start[b, stance, 0]
        - planner.feet_start[b, old_stance, 0]
        + delta_x_target
    )
    return rel


def simulate_case(case_id, delta_x_target, use_cp):
    """Chay 11 giay, kiem tra logic va thu thap cac vi tri dat chan X-Y."""
    sched = ContactScheduler(
        dt=DT,
        period_phase=PERIOD_PHASE,
        shift=SHIFT,
        double_support=DOUBLE_SUPPORT,
        num_envs=1,
        device="cpu",
    )
    planner = FootstepPlanner(
        num_envs=1,
        period_phase=PERIOD_PHASE,
        num_footsteps=NUM_FOOTSTEPS,
        device="cpu",
    )
    sched.reset()
    feet0 = torch.tensor([FEET_INITIAL], dtype=torch.float32)
    planner.reset(torch.tensor([0], dtype=torch.long), feet0)

    cmd = torch.tensor([[VX, VY]], dtype=torch.float32)
    cp_offset = torch.tensor([[CP_DX, CP_DY, 0.0]], dtype=torch.float32)
    com_state = torch.zeros(1, 2, 3, dtype=torch.float32)
    fk_valid = torch.tensor([True])
    last_step = torch.tensor([-3], dtype=torch.long)

    # Du lieu ve do thi: toa do hai chan ban dau va cac lan tiep dat
    history = {
        0: [feet0[0, 0, :2].numpy().copy()],  # chan trai
        1: [feet0[0, 1, :2].numpy().copy()],  # chan phai
    }
    touchdown = []             # Cap (j, chan stance, x, y)
    dx_events = []             # Gia tri delta_x tai cac su kien bat dau buoc
    step_indices = []
    allow_count = 0
    shift_count = 0
    new_count = 0
    last_pts = None
    last_j = None
    half_step = PERIOD_PHASE / 2.0
    n_ticks = round(DURATION / DT)
    check(math.isclose(n_ticks * DT, DURATION, abs_tol=1e-9),
          "DURATION khong chia het cho DT")

    for k in range(n_ticks + 1):
        # t = k * DT, bat dau tu 0.0 va gom ca moc 11.0 s.
        t = sched.t
        j = sched.step_index(t)
        local = sched.local_time_phase(t, j)
        is_shift, is_ds, is_ss = sched.phase_state(j, local)
        stance = sched.stance_feet(j)
        contact = sched.contact_state(stance, is_ss)
        new_step = sched.new_step_event(j, last_step)
        j_int = int(j.item())
        step_indices.append(j_int)

        check(int(sched.tick.item()) == k, f"Case {case_id}: sai tick tai k={k}")
        check(abs(float(t.item()) - k * DT) < 2e-5,
              f"Case {case_id}: sai thoi gian tai k={k}")
        check(bool(is_shift.item()) == (j_int == -1),
              f"Case {case_id}: shift khong khop voi step_index")
        check(bool(is_shift.item()) or (bool(is_ds.item()) != bool(is_ss.item())),
              f"Case {case_id}: DSP/SSP khong hop le")
        if j_int == -1:
            shift_count += 1
            check(bool(contact.all().item()),
                  f"Case {case_id}: giai doan Shift phai tiep xuc ca hai chan")

        # (1) Vua bat dau mot buoc moi: chuyen landing cu thanh chan stance.
        if bool(new_step.item()):
            new_count += 1
            planner.update_new_step(j, new_step, stance, cmd)

            # (2) Gia lap sai lech FK va reanchor: j=0 khong reanchor.
            fk_rel = synthetic_fk_relative(planner, stance, delta_x_target)
            com_x_before = float(com_state[0, 0, 0].item())
            dx = planner.reanchor_x(
                j, new_step, stance, fk_rel, fk_valid, com_state
            )
            dx_value = float(dx.item())
            expected = delta_x_target if j_int > 0 else 0.0
            check(abs(dx_value - expected) < 2e-5,
                  f"Case {case_id}, j={j_int}: delta_x={dx_value}, expected={expected}")
            check(abs(float(com_state[0, 0, 0].item()) - com_x_before - dx_value) < 2e-5,
                  f"Case {case_id}, j={j_int}: CoM MPC chua dong bo")
            dx_events.append(dx_value)

            # Moi lan j>0: chan swing cua buoc truoc vua thanh chan stance.
            # Luu vi tri SAU reanchor cho do thi X-Y.
            side = int(stance.item())
            xy = planner.feet_start[0, side, :2].detach().numpy().copy()
            touchdown.append((j_int, side, xy[0], xy[1]))
            if j_int > 0:  # Khong lap lai chan trai ban dau tai j=0.
                history[side].append(xy)

        # (3) CP gia lap: cong offset CO DINH vao nominal, khong cong don.
        allow = sched.allow_footstep_adjust(t, j)
        corrected_landing = planner.nominal_landing.clone()
        if use_cp:
            corrected_landing += cp_offset
        planner.set_landing(corrected_landing, allow)
        if bool(allow.item()):
            allow_count += 1
            cp_expected = cp_offset if use_cp else torch.zeros_like(cp_offset)
            check(torch.allclose(planner.updated_landing - planner.nominal_landing,
                                 cp_expected, atol=2e-5, rtol=0.0),
                  f"Case {case_id}, t={float(t.item()):.2f}: sai hieu chinh CP")

        # (4) Footstep preview phai phan anh landing da duoc CP chinh.
        pts, j_base = planner.footstep_sequence(j, cmd)
        check(pts.shape == (1, NUM_FOOTSTEPS, 3),
              f"Case {case_id}, t={float(t.item()):.2f}: sai shape chuoi footstep")
        check(int(j_base.item()) == max(j_int, 0),
              f"Case {case_id}: sai j_base")
        st = int(stance.item())
        sw = 1 - st
        check(torch.allclose(pts[0, 0], planner.feet_start[0, sw], atol=2e-5),
              f"Case {case_id}: P_-1 khong khop chan swing dau buoc")
        check(torch.allclose(pts[0, 1], planner.feet_start[0, st], atol=2e-5),
              f"Case {case_id}: P_0 khong khop chan stance dau buoc")
        if j_int >= 0:
            check(torch.allclose(pts[0, 2], planner.updated_landing[0], atol=2e-5),
                  f"Case {case_id}: P_1 khong khop updated_landing")
        else:
            expected_shift_p1 = (
                planner.feet_start[0, st] + planner.step_offset(stance.new_tensor([sw]), cmd)[0]
            )
            check(torch.allclose(pts[0, 2], expected_shift_p1, atol=2e-5),
                  f"Case {case_id}: P_1 khi Shift khong dung gia tri danh dinh")

        # Kiem tra 11 vector gia tang du doan sau P1, luan phien hai chan.
        for idx in range(3, NUM_FOOTSTEPS):
            swing_side = st if (idx - 3) % 2 == 0 else sw
            off = planner.step_offset(stance.new_tensor([swing_side]), cmd)[0]
            check(torch.allclose(pts[0, idx] - pts[0, idx-1], off, atol=2e-5),
                  f"Case {case_id}: sai luan phien footstep tai index={idx}")

        last_pts = pts.detach().numpy().copy()
        last_j = j_int
        last_step = torch.where(j >= 0, j, torch.full_like(j, -1))
        if k < n_ticks:
            sched.count()

    # Kiem tra so buoc va tong delta_x.
    expected_steps = int(round((DURATION - SHIFT) / half_step)) + 1
    check(new_count == expected_steps,
          f"Case {case_id}: new_count={new_count}, expected={expected_steps}")
    check([int(x[0]) for x in touchdown] == list(range(expected_steps)),
          f"Case {case_id}: chi so buoc khong lien tuc")
    check(shift_count == round(SHIFT / DT),
          f"Case {case_id}: so tick Shift khong dung")
    check(all(abs(v) < 2e-5 for v in dx_events[:1]),
          f"Case {case_id}: j=0 khong duoc reanchor")
    check(all(abs(v - delta_x_target) < 2e-5 for v in dx_events[1:]),
          f"Case {case_id}: delta_x khong on dinh")
    expected_com_x = (expected_steps - 1) * delta_x_target
    check(abs(float(com_state[0, 0, 0].item()) - expected_com_x) < 2e-4,
          f"Case {case_id}: tong x_CoM={float(com_state[0,0,0]):.4f} ",
          )

    # Luu y: history la vi tri KE HOACH (footstep/stance) sau reanchor,
    # KHONG phai duong di CoM hay vi tri chan do tu mo phong dong luc hoc.
    return {
        "id": case_id,
        "delta_x": delta_x_target,
        "cp": use_cp,
        "history": {side: np.asarray(data) for side, data in history.items()},
        "touchdown": np.asarray(touchdown, dtype=float),
        "steps": new_count,
        "cp_enabled_ticks": allow_count,
        "com_x": float(com_state[0, 0, 0].item()),
        "last_j": last_j,
        "preview_shape": last_pts.shape,
    }


def plot_results(results):
    """Ve 4 do thi X-Y; KHONG luu file anh."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for ax, result in zip(axes.flat, results):
        h = result["history"]
        left = h[0]
        right = h[1]
        touchdowns = result["touchdown"]

        # Duong lien tiep cac lan tiep dat theo thu tu thoi gian.
        ax.plot(touchdowns[:, 2], touchdowns[:, 3],
                linestyle="--", linewidth=1.0, alpha=0.45,
                label="Thu tu dat chan")
        # Hai chuoi vi tri chan trai/phai trong he {P}.
        ax.plot(left[:, 0], left[:, 1], marker="o", markersize=3.5,
                linewidth=1.4, label="Chan trai (L)")
        ax.plot(right[:, 0], right[:, 1], marker="s", markersize=3.5,
                linewidth=1.4, label="Chan phai (R)")

        # Vi tri hai chan ban dau va vi tri stance cuoi (khong luu anh).
        ax.scatter([left[0, 0], right[0, 0]], [left[0, 1], right[0, 1]],
                   marker="x", s=85, linewidths=2.0, zorder=6,
                   label="Vi tri ban dau")
        ax.scatter([touchdowns[-1, 2]], [touchdowns[-1, 3]],
                   marker="*", s=145, zorder=7, label="Vi tri stance cuoi")

        cp_label = f"CP = (+{CP_DX:.3f}, +{CP_DY:.3f}) m" if result["cp"] else "CP = 0"
        ax.set_title(
            f"Case {result['id']}: reanchor = {result['delta_x']:.1f} m/buoc, {cp_label}",
            fontsize=11
        )
        ax.set_xlabel("X trong he {P} (m)")
        ax.set_ylabel("Y trong he {P} (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.margins(x=0.06, y=0.12)

    fig.suptitle(
        "KIEM THU TICH HOP CONTACT SCHEDULER + FOOTSTEP PLANNER | 11 s\n"
        "Quy dao X-Y cua cac vi tri dat chan duoc lap ke hoach",
        fontsize=14, fontweight="bold"
    )
    plt.show()


def main():
    cases = [
        (1, 0.0, False),
        (2, 0.2, False),
        (3, 0.0, True),
        (4, 0.2, True),
    ]
    results = [simulate_case(*case) for case in cases]

    print("\n=== TEST TICH HOP 11 GIAY: 4 CASES ===")
    print(f"dt={DT:.2f}s | shift={SHIFT:.2f}s | thoi gian/buoc={PERIOD_PHASE/2:.2f}s")
    print(f"len(footstep_sequence)={NUM_FOOTSTEPS} | cmd=({VX:.2f}, {VY:.2f})m/s")
    print(f"CP gia lap: (+{CP_DX:.3f}, +{CP_DY:.3f}) m so voi nominal_landing")
    print("delta_x gia lap duoc ap dung tai MOI su kien j>0, khong phai moi tick")
    for res in results:
        print(
            f"Case {res['id']}: PASS | delta_x={res['delta_x']:.1f}m/buoc "
            f"| CP={'ON' if res['cp'] else 'OFF'} "
            f"| so buoc={res['steps']} | j_cuoi={res['last_j']} "
            f"| x_CoM_MPC={res['com_x']:.3f}m "
            f"| preview={res['preview_shape']}"
        )
    print("4/4 CASES PASS. Dang hien thi 4 do thi X-Y (khong luu tep).\n")
    plot_results(results)


if __name__ == "__main__":
    main()
