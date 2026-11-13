"""Integration test: ContactScheduler + FootstepPlanner + ZMPGenerator.

Run from the capture_point directory (or with the script placed there):
    python test_integrated_contact_footstep_zmp.py

Only prints assertions and shows Matplotlib plots. No CSV or image is saved.
The CP and reanchor signals are SYNTHETIC; this is not a robot dynamics test.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner
from zmp_generator import ZMPGenerator


# ------------------------- Test configuration -------------------------
SIM_DURATION = 11.0   # s
DT = 0.02             # s
PERIOD_PHASE = 0.4    # s; one step = 0.2 s
SHIFT_DURATION = 0.4  # s
DOUBLE_SUPPORT = 0.1 # fraction of PERIOD_PHASE -> 0.04 s
HORIZON = 80
NUM_FOOTSTEPS = 14
CMD_XY = (0.20, 0.00)       # m/s
CP_OFFSET_XY = (0.020, 0.015)  # m, fixed per step (not accumulated every tick)
INITIAL_FEET = ((0.0, 0.09, 0.0), (0.0, -0.09, 0.0))  # left/right, plan frame

CASES = [
    ("Case 1: dx=0, CP off", 0.0, False),
    ("Case 2: dx=0.2, CP off", 0.2, False),
    ("Case 3: dx=0, CP on", 0.0, True),
    ("Case 4: dx=0.2, CP on", 0.2, True),
]


def assert_close(actual, expected, name, atol=2e-5):
    got = np.asarray(actual, dtype=np.float64)
    want = np.asarray(expected, dtype=np.float64)
    if not np.allclose(got, want, atol=atol, rtol=0.0):
        raise AssertionError(f"{name}: got={got}, expected={want}")


def make_modules(num_envs=1):
    scheduler = ContactScheduler(
        dt=DT, period_phase=PERIOD_PHASE, shift=SHIFT_DURATION,
        double_support=DOUBLE_SUPPORT, num_envs=num_envs, device="cpu"
    )
    planner = FootstepPlanner(
        num_envs=num_envs, period_phase=PERIOD_PHASE,
        num_footsteps=NUM_FOOTSTEPS, device="cpu"
    )
    generator = ZMPGenerator(
        zmp_offset_x=0.005, zmp_offset_y_out=0.007,
        horizon=HORIZON
    )
    return scheduler, planner, generator


def synthetic_com_relative_to_feet(planner, stance, delta_x):
    """Build synthetic FK-relative CoM data to make reanchor_x return delta_x.

    From current code: shift = feet_start[swing].x + rel[swing].x
                                    - rel[stance].x - feet_start[stance].x
    Thus choose rel[stance].x=0 and
                rel[swing].x=feet_start[stance].x-feet_start[swing].x+delta_x.
    This is deliberately not a physical FK measurement.
    """
    b = torch.arange(stance.shape[0], device=stance.device)
    swing = 1 - stance
    rel = torch.zeros_like(planner.feet_start)
    rel[b, swing, 0] = (
        planner.feet_start[b, stance, 0]
        - planner.feet_start[b, swing, 0]
        + float(delta_x)
    )
    return rel


def simulate_case(label, imposed_delta_x, use_cp):
    scheduler, planner, generator = make_modules()
    feet0 = torch.tensor([INITIAL_FEET], dtype=torch.float32)
    planner.reset(torch.tensor([0]), feet0)
    scheduler.reset()

    command = torch.tensor([CMD_XY], dtype=torch.float32)
    com_state = torch.zeros(1, 2, 3, dtype=torch.float32)  # only reanchor is tested
    fk_valid = torch.ones(1, dtype=torch.bool)
    last_step = torch.full((1,), -3, dtype=torch.long)

    t_samples, zmp_samples, step_samples = [], [], []
    step_times, step_positions, step_sides = [], [], []
    reanchor_sum, reanchor_count, cp_count = 0.0, 0, 0
    previews = {}
    n_ticks = int(round(SIM_DURATION / DT))

    for tick in range(n_ticks):
        t = scheduler.t.clone()
        j_now = scheduler.step_index(t)
        stance = scheduler.stance_feet(j_now)
        new_step = scheduler.new_step_event(j_now, last_step)

        # (1) Commit previous corrected landing and begin a new step.
        planner.update_new_step(j_now, new_step, stance, command)

        # (2) Synthetic reanchor, only active when new_step and j_now > 0.
        rel = synthetic_com_relative_to_feet(planner, stance, imposed_delta_x)
        old_com_x = com_state[:, 0, 0].clone()
        actual_shift = planner.reanchor_x(
            j_now, new_step, stance, rel, fk_valid, com_state
        )
        assert torch.allclose(
            com_state[:, 0, 0] - old_com_x, actual_shift, atol=2e-6
        ), "MPC CoM x not synchronized with footstep reanchor."

        if bool(new_step[0]):
            index = int(j_now[0])
            step_times.append(float(t[0]))
            step_positions.append(planner.feet_start[0, stance[0]].clone().numpy())
            step_sides.append(int(stance[0]))
            if index > 0:
                reanchor_sum += float(actual_shift[0])
                reanchor_count += 1
                assert_close(
                    actual_shift[0].item(), imposed_delta_x,
                    f"{label}: reanchor on step {index}", atol=1e-5
                )
            else:
                assert_close(actual_shift[0].item(), 0.0,
                             f"{label}: reanchor must be disabled on j=0")

        # (3) Optional fixed Capture Point correction, nominal-based, not cumulative per tick.
        allow_adjust = scheduler.allow_footstep_adjust(t, j_now)
        if use_cp:
            corrected = planner.nominal_landing.clone()
            corrected[:, 0] += CP_OFFSET_XY[0]
            corrected[:, 1] += CP_OFFSET_XY[1]
            before = planner.updated_landing.clone()
            planner.set_landing(corrected, allow_adjust)
            if bool(allow_adjust[0]):
                assert torch.allclose(planner.updated_landing, corrected, atol=1e-6)
                cp_count += 1
            else:
                assert torch.allclose(planner.updated_landing, before, atol=1e-6)

        # (4) Plan 14 future footsteps; generate 80 ZMP reference samples for MPC.
        footsteps, j_base = planner.footstep_sequence(j_now, command)
        assert footsteps.shape == (1, NUM_FOOTSTEPS, 3)
        assert torch.equal(j_base, j_now.clamp_min(0))

        current_t = scheduler.t.clone()
        input_copy = footsteps.clone()
        zref = generator.generate(footsteps, planner.feet0, j_base, scheduler)
        assert zref.shape == (1, HORIZON, 2)
        assert torch.isfinite(zref).all(), f"Non-finite ZMP at t={float(t[0])}"
        assert torch.equal(input_copy, footsteps), "generate() modified footsteps!"
        assert torch.equal(current_t, scheduler.t), "generate() changed scheduler time!"

        # Save first predicted point z(t+dt), not z(t).
        t_samples.append(float(t[0] + DT))
        zmp_samples.append(zref[0, 0].detach().numpy().copy())
        step_samples.append(int(j_now[0]))
        if tick in (0, 19, 100, 250):
            previews[tick] = zref[0].detach().numpy().copy()

        last_step = torch.where(j_now >= 0, j_now, torch.full_like(j_now, -1))
        scheduler.count()

    assert_close(scheduler.t[0].item(), SIM_DURATION, f"{label}: final clock")
    assert len(t_samples) == n_ticks
    assert len(step_times) == 53, "Expected 53 step-start events t=0.4..10.8"
    assert step_sides[0] == 0, "First stance should be left"
    assert all(step_sides[i] != step_sides[i-1] for i in range(1,len(step_sides))), \
        "Stance legs must alternate"
    assert reanchor_count == 52
    assert_close(reanchor_sum, 52 * imposed_delta_x,
                 f"{label}: accumulated reanchor", atol=1e-4)
    assert (cp_count > 0) == use_cp

    print(f"[PASS] {label:25s}  tick={n_ticks}, steps={len(step_times)}, "
          f"reanchor_sum={reanchor_sum:.3f} m, CP updates={cp_count}")
    return {
        "label":label,
        "t":np.array(t_samples),
        "zmp":np.stack(zmp_samples),
        "steps":np.array(step_samples),
        "step_times":np.array(step_times),
        "feet":np.stack(step_positions),
        "sides":np.array(step_sides),
        "previews":previews,
        "total_shift":reanchor_sum,
    }


def test_analytical_baseline(result):
    """Independent expected values for shift and DS/SS with no CP/reanchor.

    Feet are initially (+/-0.09 m), velocity is +0.20 m/s, and a step is 0.20 s.
    Thus first right landing x=0.04, second left landing x=0.08.
    """
    t = result["t"]
    z = result["zmp"]

    def at(sec):
        k = int(round(sec / DT)) - 1
        assert_close(t[k], sec, f"time index at {sec}")
        return z[k]

    # Shift: initial midpoint -> left stance, with Y-left=0.09+0.007=0.097.
    s = 3*(0.5**2) - 2*(0.5**3)
    assert_close(at(0.20), [0.005, s*0.097], "Mid shift t=0.20")
    assert_close(at(0.40), [0.005, 0.097], "End shift t=0.40")

    # j=0 has no additional DS interpolation.
    assert_close(at(0.42), [0.005, 0.097], "j=0 t=0.42")
    assert_close(at(0.58), [0.005, 0.097], "j=0 t=0.58")

    # j=1: left -> right; right landing at x=0.04, y=-0.09.
    assert_close(at(0.60), [0.005, 0.097], "j=1 DS start")
    assert_close(at(0.62), [0.025, 0.0], "j=1 DS midpoint")
    assert_close(at(0.64), [0.045, -0.097], "j=1 DS end")
    assert_close(at(0.78), [0.045, -0.097], "j=1 SS")

    # j=2: right -> left; left landing x=0.08, y=+0.09.
    assert_close(at(0.80), [0.045, -0.097], "j=2 DS start")
    assert_close(at(0.82), [0.065, 0.0], "j=2 DS midpoint")
    assert_close(at(0.84), [0.085, 0.097], "j=2 DS end")
    print("[PASS] Analytic landmarks: shift, j=0, DS left->right and right->left, SS")


def test_multi_environment():
    """Quick vectorized B=2 integration check with different initial feet and cmd."""
    scheduler, planner, gen = make_modules(num_envs=2)
    feet = torch.tensor([
        INITIAL_FEET,
        ((0., 0.12, 0.), (0., -0.12, 0.)),
    ], dtype=torch.float32)
    cmds = torch.tensor([[0.2, 0.0], [0.1, 0.02]], dtype=torch.float32)
    planner.reset(torch.arange(2), feet)
    j = scheduler.step_index(scheduler.t)
    footsteps, jb = planner.footstep_sequence(j, cmds)
    zmp = gen.generate(footsteps, planner.feet0, jb, scheduler)
    assert zmp.shape == (2, HORIZON, 2)
    assert torch.isfinite(zmp).all()
    assert not torch.allclose(zmp[0,:,1], zmp[1,:,1])

    # Advance to first walking step, exercise batched new-step logic and generate again.
    for _ in range(20):
        scheduler.count()
    j = scheduler.step_index(scheduler.t)
    st = scheduler.stance_feet(j)
    new = scheduler.new_step_event(j, torch.full_like(j, -1))
    assert new.all() and (j == 0).all()
    planner.update_new_step(j, new, st, cmds)
    footsteps, jb = planner.footstep_sequence(j, cmds)
    zmp = gen.generate(footsteps, planner.feet0, jb, scheduler)
    assert zmp.shape == (2, HORIZON, 2)
    assert torch.isfinite(zmp).all()
    print("[PASS] Batched environments B=2, N=80, initial shift and walking")


def plot_results(results):
    # Figure 1: sample-by-sample first ZMP point from the MPC horizon.
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    for ax, res in zip(axes, results):
        ax.plot(res["t"], res["zmp"][:,0], label="ZMP X", linewidth=1.5)
        ax.plot(res["t"], res["zmp"][:,1], label="ZMP Y", linewidth=1.2)
        ax.axvspan(0, SHIFT_DURATION, alpha=0.12, color="grey", label="Initial shift")
        ax.set_title(res["label"])
        ax.set_ylabel("ZMP (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)  [first MPC sample at t + dt]")
    fig.suptitle("Integrated test (11 s): ZMP X, Y over time", fontsize=14)

    # Figure 2: planar ZMP trajectory and footstep sites committed at step starts.
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, res in zip(axes2.flat, results):
        ax.plot(res["zmp"][:,0], res["zmp"][:,1], "-", lw=1.3, label="ZMP ref")
        l = res["sides"] == 0
        r = ~l
        ax.scatter(res["feet"][l,0], res["feet"][l,1], marker="o", s=20,
                   label="Left stance foot", alpha=0.6)
        ax.scatter(res["feet"][r,0], res["feet"][r,1], marker="s", s=20,
                   label="Right stance foot", alpha=0.6)
        ax.scatter([res["zmp"][0,0]],[res["zmp"][0,1]], marker="*", s=100,
                   color="black", label="Start")
        ax.scatter([res["zmp"][-1,0]],[res["zmp"][-1,1]], marker="x", s=75,
                   color="black", label="End")
        ax.set_title(res["label"])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig2.suptitle("Integrated test (11 s): ZMP reference and committed foot sites", fontsize=14)

    # Figure 3: visualize an actual horizon, to distinguish 80 future points from one applied sample.
    base = results[0]
    fig3, ax3 = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for tick, name in [(0, "t=0.00 s"), (250, "t=5.00 s")]:
        preview = base["previews"][tick]
        tk = tick * DT + DT * np.arange(1, HORIZON+1)
        ax3[0].plot(tk, preview[:,0], label=name)
        ax3[1].plot(tk, preview[:,1], label=name)
    ax3[0].set_ylabel("Predicted ZMP X (m)")
    ax3[1].set_ylabel("Predicted ZMP Y (m)")
    ax3[1].set_xlabel("Future time (s)")
    for ax in ax3:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig3.suptitle("Case 1: 80 future ZMP samples at two planning instants", fontsize=14)

    # Only show figures; do not save any image or CSV.
    plt.show()


def main():
    torch.set_grad_enabled(False)
    print("="*72)
    print("INTEGRATION TEST: ContactScheduler + FootstepPlanner + ZMPGenerator")
    print(f"Time={SIM_DURATION}s, dt={DT}s, horizon={HORIZON}, "
          f"step duration={PERIOD_PHASE/2:.2f}s, DS={DOUBLE_SUPPORT*PERIOD_PHASE:.2f}s")
    print("NOTE: 0.2 m reanchor is deliberately injected PER COMPLETED STEP.")
    print("="*72)

    results = [simulate_case(*case) for case in CASES]
    test_analytical_baseline(results[0])
    test_multi_environment()

    # Reanchor should affect X, not Y, for equal CP configurations.
    assert_close(results[0]["zmp"][:,1], results[1]["zmp"][:,1],
                 "Y of cases 1 and 2 independent of reanchor", atol=2e-5)
    assert_close(results[2]["zmp"][:,1], results[3]["zmp"][:,1],
                 "Y of cases 3 and 4 independent of reanchor", atol=2e-5)
    assert not np.allclose(results[0]["zmp"], results[2]["zmp"]), \
        "Fixed CP correction did not affect ZMP reference"
    assert_close(results[1]["total_shift"], 52*0.2, "Case 2 total shift", atol=1e-4)
    assert_close(results[3]["total_shift"], 52*0.2, "Case 4 total shift", atol=1e-4)
    print("[PASS] Cross-case checks: CP changes ZMP; reanchor only translates X")
    print("\nALL INTEGRATION CHECKS PASSED. Opening plots (no files are saved).")
    plot_results(results)


if __name__ == "__main__":
    main()
