"""Integration test: ContactScheduler + FootstepPlanner + CapturePoint + SwingTrajectoryGenerator.

Put this file alongside these four modules inside capture_point/:
    contact_scheduler.py
    footstep_planner.py
    capture_point.py
    swing_trajectory.py

Run:
    python test_swing_integration.py
    python test_swing_integration.py --duration 11
    python test_swing_integration.py --duration 11 --plot

Uses synthetic CoM plan/measurements in frame {P}. No ZMP Generator, MPC,
MuJoCo, WBIK, real FK measurement, or contact sensor. CP corrections are
computed by the actual CapturePoint class; they are NOT injected directly.
The optional `reanchor_x` operation is deliberately excluded because it
requires a separate FK measurement (not available in this 4-module test).
Nothing is saved to disk by this test; --plot only displays figures.
"""

import argparse
import math
import unittest

import torch

from contact_scheduler import ContactScheduler
from footstep_planner import FootstepPlanner
from capture_point import CapturePoint
from swing_trajectory import SwingTrajectoryGenerator


DT = 0.02
PERIOD = 0.4
SHIFT = 0.4
DOUBLE_SUPPORT = 0.1
SWING_HEIGHT = 0.02
COM_HEIGHT = 0.15  # m; synthetic value for the CP formula, not a measured CoM height
STEP_WIDTH = 0.12
NUM_FOOTSTEPS = 14
CASE_NAMES = ("Baseline", "CP moderate", "CP + saturation", "CP - saturation")
B = len(CASE_NAMES)


def check_close(actual, expected, label, atol=3e-6):
    try:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{label}: {exc}") from exc


def synthetic_measurement(t, planned, omega):
    """Produce (B,3) measured CoM position/velocity from a known DCM error.

    Four cases share the same planned CoM and commands; only sensor-like
    disturbances differ. These numbers test CP interfaces, not robot physics.
    """
    pos = torch.zeros(B, 2, dtype=torch.float32, device=planned.device)
    vel = torch.zeros_like(pos)
    pos[1, 0] = 0.015 + 0.004 * math.sin(2.0 * math.pi * t / 0.8)
    pos[1, 1] = 0.005 + 0.002 * math.cos(2.0 * math.pi * t / 0.8)
    vel[1] = torch.tensor([0.020, -0.006], device=planned.device)
    pos[2] = torch.tensor([0.30, 0.30], device=planned.device)
    pos[3] = torch.tensor([-0.30, -0.30], device=planned.device)

    com_meas = torch.zeros(B, 3, dtype=torch.float32, device=planned.device)
    vcom_meas = torch.zeros_like(com_meas)
    com_meas[:, :2] = planned[:, :, 0] + pos
    vcom_meas[:, :2] = planned[:, :, 1] + vel
    com_meas[:, 2] = COM_HEIGHT
    return com_meas, vcom_meas, pos + vel / omega


@torch.no_grad()
def simulate(duration=4.0, device="cpu"):
    """Run the four actual modules on each 20-ms tick, returning a trace."""
    n_ticks = round(duration / DT)
    if duration < 1.2 or abs(n_ticks * DT - duration) > 1e-8:
        raise ValueError("--duration must be >= 1.2 s and divisible by 0.02 s")

    sched = ContactScheduler(
        dt=DT, period_phase=PERIOD, shift=SHIFT,
        double_support=DOUBLE_SUPPORT, num_envs=B, device=device,
    )
    planner = FootstepPlanner(
        num_envs=B, period_phase=PERIOD, num_footsteps=NUM_FOOTSTEPS, device=device,
    )
    cp = CapturePoint(
        com_height=COM_HEIGHT, step_gain=(0.4, 0.3),
        step_limit=(0.04, 0.03), min_step_width=STEP_WIDTH, device=device,
    )
    swing = SwingTrajectoryGenerator(
        period_phase=PERIOD, double_support=DOUBLE_SUPPORT, swing_height=SWING_HEIGHT,
    )
    # Left foot y > 0, right foot y < 0, in the local planning frame {P}.
    feet0 = torch.tensor(
        [[0.0, 0.09, 0.0], [0.0, -0.09, 0.0]],
        dtype=torch.float32, device=device,
    )[None].repeat(B, 1, 1)
    sched.reset()
    planner.reset(torch.arange(B, device=device), feet0)
    cmd = torch.tensor([[0.20, 0.00]], dtype=torch.float32, device=device).repeat(B, 1)
    last_step = torch.full((B,), -3, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)
    store = {name: [] for name in (
        "t_next", "j_now", "j_next", "local_next", "contact",
        "feet", "feet_start", "nominal", "landing", "correction",
        "allow", "new_step", "swinging", "dcm_error", "predicted_p1",
    )}
    counts = {"events": 0, "allowed": 0, "blocked": 0, "shift": 0,
              "double_support": 0, "single_support": 0, "transitions": 0}

    for k in range(n_ticks):
        t = sched.t
        t_scalar = float(k * DT)
        j_now = sched.step_index(t)
        stance_now = sched.stance_feet(j_now)
        new_step = sched.new_step_event(j_now, last_step)

        # (1) New-step event updates the stance foot and nominal landing.
        planner.update_new_step(j_now, new_step, stance_now, cmd)
        counts["events"] += int(new_step[0].item())

        # (2) Synthetic planned CoM and synthetic measured CoM, at SAME t_k.
        # com_state shape (B, 2, 3): [axis=(x,y), state=(pos,vel,acc)].
        com_state = torch.zeros(B, 2, 3, dtype=torch.float32, device=device)
        com_state[:, 0, 0] = cmd[:, 0] * t_scalar
        com_state[:, 1, 0] = cmd[:, 1] * t_scalar
        com_state[:, :, 1] = cmd
        com_meas, vcom_meas, known_dcm_error = synthetic_measurement(
            t_scalar, com_state, cp.omega,
        )
        _, _, dcm_error = cp.compute_dcm_error(com_meas, vcom_meas, com_state)
        check_close(dcm_error, known_dcm_error, f"DCM at tick {k}")
        correction = cp.compute_correction(dcm_error)
        expected_correction = torch.maximum(
            torch.minimum(known_dcm_error * cp.step_gain, cp.step_limit),
            -cp.step_limit,
        )
        check_close(correction, expected_correction, f"CP gain/limit at tick {k}")

        # (3) CP landing correction, then scheduler-controlled freeze.
        stance_y = planner.feet_start[batch_idx, stance_now, 1]
        proposed_landing = cp.correct_landing(
            planner.nominal_landing, correction, stance_now, stance_y,
        )
        assert bool(torch.all(proposed_landing[stance_now == 0, 1]
                              <= stance_y[stance_now == 0] - STEP_WIDTH + 2e-6))
        assert bool(torch.all(proposed_landing[stance_now == 1, 1]
                              >= stance_y[stance_now == 1] + STEP_WIDTH - 2e-6))
        allow = sched.allow_footstep_adjust(t, j_now)
        old_landing = planner.updated_landing.clone()
        planner.set_landing(proposed_landing, allow)
        expected_landing = torch.where(allow[:, None], proposed_landing, old_landing)
        check_close(planner.updated_landing, expected_landing, f"landing gate tick {k}")
        if j_now[0].item() >= 0:
            if allow[0].item():
                counts["allowed"] += 1
            else:
                counts["blocked"] += 1

        # (4) Footstep sequence at t_k; CP update must be in P_1.
        footsteps, j_base = planner.footstep_sequence(j_now, cmd)
        assert footsteps.shape == (B, NUM_FOOTSTEPS, 3)
        check_close(
            footsteps[j_now >= 0, 2], planner.updated_landing[j_now >= 0],
            f"CP -> P_1 tick {k}",
        )

        # (5) Target feet and schedule at t_(k+1), not at t_k.
        t_next = t + DT
        j_next = sched.step_index(t_next)
        local_next = sched.local_time_phase(t_next, j_next)
        is_shift, is_ds, is_ss = sched.phase_state(j_next, local_next)
        stance_next = sched.stance_feet(j_next)
        swing_next = 1 - stance_next
        contact_next = sched.contact_state(stance_next, is_ss)
        feet_target, contact_out = swing.generate(
            footsteps, planner.feet0, j_base, j_next, local_next, contact_next,
        )
        assert feet_target.shape == (B, 2, 3)
        assert contact_out.shape == (B, 2) and contact_out.dtype == torch.bool
        assert bool(torch.isfinite(feet_target).all())
        assert torch.equal(contact_out, contact_next)
        assert bool(torch.all(contact_next[batch_idx, stance_next]))
        assert bool(torch.all(contact_next[batch_idx, swing_next] == ~is_ss))
        assert bool(torch.all(is_shift | is_ds | is_ss))

        # Independent geometric oracle for selected positions and phase handling.
        i = (j_next - j_base).clamp(0, NUM_FOOTSTEPS - 3)
        start = footsteps[batch_idx, i]
        stance_pos = footsteps[batch_idx, i + 1]
        end = footsteps[batch_idx, i + 2]
        s = ((local_next - DOUBLE_SUPPORT * PERIOD) /
             (PERIOD / 2 - DOUBLE_SUPPORT * PERIOD)).clamp(0.0, 1.0)
        sigma = 3 * s**2 - 2 * s**3
        target_expected = start + (end - start) * sigma[:, None]
        target_expected[:, 2] = (
            start[:, 2] + 64 * SWING_HEIGHT * s**3 * (1 - s)**3
        )
        for b in range(B):
            if is_shift[b].item():
                check_close(feet_target[b], feet0[b], f"shift feet tick {k}, env {b}")
            else:
                st = int(stance_next[b].item())
                sw = 1 - st
                check_close(feet_target[b, st], stance_pos[b],
                            f"stance feet tick {k}, env {b}")
                expected_swing = (target_expected[b] if is_ss[b].item() else start[b])
                check_close(feet_target[b, sw], expected_swing,
                            f"swing feet tick {k}, env {b}")

        counts["shift"] += int(is_shift[0].item())
        counts["double_support"] += int(is_ds[0].item())
        counts["single_support"] += int(is_ss[0].item())
        counts["transitions"] += int((j_next[0] != j_now[0]).item())
        last_step = torch.where(new_step, j_now, last_step)
        for name, val in (
            ("t_next", t_next), ("j_now", j_now), ("j_next", j_next),
            ("local_next", local_next), ("contact", contact_out),
            ("feet", feet_target), ("feet_start", planner.feet_start),
            ("nominal", planner.nominal_landing), ("landing", planner.updated_landing),
            ("correction", correction), ("allow", allow), ("new_step", new_step),
            ("swinging", is_ss), ("dcm_error", dcm_error),
            ("predicted_p1", footsteps[:, 2]),
        ):
            store[name].append(val.detach().cpu().clone())
        sched.count()

    trace = {k: torch.stack(v, dim=0) for k, v in store.items()}
    return trace, counts


class FourModuleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace, cls.counts = simulate(duration=ARGS.duration, device=ARGS.device)

    def test_01_batch_and_clock(self):
        x = self.trace
        self.assertEqual(x["feet"].shape[1:], (B, 2, 3))
        self.assertEqual(x["contact"].shape[1:], (B, 2))
        self.assertAlmostEqual(float(x["t_next"][-1, 0]), ARGS.duration, places=5)
        self.assertTrue(torch.all(x["j_next"] >= x["j_now"]).item())
        self.assertTrue(torch.all(x["j_next"] - x["j_now"] <= 1).item())

    def test_02_shift_keeps_initial_feet(self):
        x = self.trace
        mask = x["j_next"][:, 0] < 0
        self.assertGreater(int(mask.sum()), 0)
        start = x["feet"][0, 0].clone()
        check_close(x["feet"][mask, 0], start[None].expand(int(mask.sum()), -1, -1),
                    "shift hold")
        self.assertTrue(x["contact"][mask].all().item())

    def test_03_double_support_vs_single_support(self):
        x = self.trace
        is_ds = (x["j_next"] >= 0) & x["contact"].all(dim=-1)
        is_ss = (x["j_next"] >= 0) & (x["contact"].sum(dim=-1) == 1)
        self.assertTrue(is_ds[:, 0].any().item())
        self.assertTrue(is_ss[:, 0].any().item())
        self.assertEqual(int(self.counts["double_support"] + self.counts["single_support"] +
                             self.counts["shift"]), len(x["feet"]))
        self.assertTrue((x["feet"][:, :, :, 2] >= -1e-6).all().item())

    def test_04_left_and_right_swing(self):
        x = self.trace
        for foot in (0, 1):
            airborne = ~x["contact"][:, 0, foot]
            self.assertTrue(airborne.any().item(), f"foot {foot} never enters swing")
            self.assertGreater(float(x["feet"][airborne, 0, foot, 2].max()), 0.0199)
            self.assertLessEqual(float(x["feet"][airborne, 0, foot, 2].max()),
                                 SWING_HEIGHT + 1e-5)

    def test_05_all_step_transitions_exist(self):
        x = self.trace
        jump = (x["j_next"][:, 0] != x["j_now"][:, 0])
        self.assertGreaterEqual(int(jump.sum()), 4)
        self.assertTrue(x["contact"][jump].all().item())
        self.assertGreaterEqual(self.counts["events"], 3)

    def test_06_baseline_has_zero_cp_correction(self):
        x = self.trace
        self.assertTrue(torch.equal(x["correction"][:, 0],
                                    torch.zeros_like(x["correction"][:, 0])))
        walking = x["j_now"][:, 0] >= 0
        check_close(x["landing"][walking, 0], x["nominal"][walking, 0],
                    "baseline landing == nominal")

    def test_07_cp_changes_current_swing_path(self):
        x = self.trace
        # During the first right-foot swing, all cases have exactly the same start.
        # Therefore the foot target difference must equal smoothstep(s) times
        # the difference in the CP-updated landing of the current step.
        mask = (
            (x["j_now"][:, 0] == 0) & (x["j_next"][:, 0] == 0)
            & (x["local_next"][:, 0] > 0.07)
            & (x["local_next"][:, 0] < 0.17)
        )
        self.assertTrue(mask.any().item())
        s = ((x["local_next"][mask, 0] - DOUBLE_SUPPORT * PERIOD)
             / (PERIOD / 2 - DOUBLE_SUPPORT * PERIOD)).clamp(0, 1)
        sigma = 3 * s**2 - 2 * s**3
        delta_landing = x["landing"][mask, 1, :2] - x["landing"][mask, 0, :2]
        delta_target = x["feet"][mask, 1, 1, :2] - x["feet"][mask, 0, 1, :2]
        check_close(delta_target, delta_landing * sigma[:, None],
                    "CP updated landing -> swing XY")
        self.assertGreater(float(delta_target[:, 0].abs().max()), 0.003)

    def test_08_cp_saturation_and_minimum_width(self):
        x = self.trace
        walking = x["j_now"][:, 0] >= 0
        corr = x["correction"][walking]
        check_close(corr[:, 2, 0], torch.full_like(corr[:, 2, 0], 0.04), "X +limit")
        check_close(corr[:, 3, 0], torch.full_like(corr[:, 3, 0], -0.04), "X -limit")
        check_close(corr[:, 2, 1], torch.full_like(corr[:, 2, 1], 0.03), "Y +limit")
        check_close(corr[:, 3, 1], torch.full_like(corr[:, 3, 1], -0.03), "Y -limit")
        self.assertTrue(torch.isfinite(x["feet"]).all().item())

    def test_09_last_tick_landing_is_frozen(self):
        x = self.trace
        final_tick = (x["j_now"][:, 0] >= 0) & ~x["allow"][:, 0]
        self.assertGreater(int(final_tick.sum()), 0)
        self.assertGreater(self.counts["blocked"], 0)
        # At a blocked tick, no NEW landing can be injected. The pipeline
        # already checked this against old_landing inside simulate().
        self.assertTrue((x["j_next"][final_tick, 0] >
                         x["j_now"][final_tick, 0]).all().item())

    def test_10_batch_cases_are_distinct(self):
        x = self.trace
        self.assertTrue((x["feet"][:, 0] - x["feet"][:, 1]).abs().max() > 0.003)
        self.assertTrue((x["landing"][:, 2] - x["landing"][:, 3]).abs().max() > 0.05)


def plot_results(trace):
    import matplotlib.pyplot as plt

    t = trace["t_next"][:, 0].numpy()
    feet = trace["feet"].numpy()
    land = trace["landing"].numpy()
    nominal = trace["nominal"].numpy()
    contact = trace["contact"].numpy()

    fig1, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, feet[:, 0, 0, 0], label="Left foot X - baseline")
    ax.plot(t, feet[:, 0, 1, 0], label="Right foot X - baseline")
    ax.plot(t, feet[:, 1, 1, 0], "--", label="Right foot X - CP moderate")
    ax.set(title="Foot targets at t + dt (planning frame P)",
           xlabel="Time (s)", ylabel="X (m)")
    ax.grid(True); ax.legend(); fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, feet[:, 0, 0, 2], label="Left foot Z")
    ax.plot(t, feet[:, 0, 1, 2], label="Right foot Z")
    ax.step(t, (~contact[:, 0, 0]).astype(float) * SWING_HEIGHT,
            where="post", linestyle=":", label="Left swing schedule (scaled)")
    ax.step(t, (~contact[:, 0, 1]).astype(float) * SWING_HEIGHT,
            where="post", linestyle=":", label="Right swing schedule (scaled)")
    ax.set(title="Swing height and nominal contact schedule - baseline",
           xlabel="Time (s)", ylabel="Z (m)")
    ax.grid(True); ax.legend(); fig2.tight_layout()

    fig3, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, nominal[:, 1, 0], label="Nominal landing X - CP moderate")
    ax.plot(t, land[:, 1, 0], label="Updated landing X - CP moderate")
    ax.plot(t, land[:, 0, 0], "--", label="Landing X - baseline")
    ax.set(title="Capture Point adjustment of landing", xlabel="Time (s)", ylabel="X (m)")
    ax.grid(True); ax.legend(); fig3.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Duration in seconds, multiple of 0.02 (default: 4)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--plot", action="store_true", help="Display figures; save nothing")
    args, unittest_args = parser.parse_known_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available on this machine")
    global ARGS
    ARGS = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FourModuleIntegrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        t = FourModuleIntegrationTests.trace
        counts = FourModuleIntegrationTests.counts
        print("\n[4-MODULE INTEGRATION] PASS")
        print(f"Duration: {args.duration:.2f}s; ticks: {len(t['feet'])}; dt: {DT:.3f}s")
        print("Cases:", ", ".join(CASE_NAMES))
        print("Counts (baseline):", counts)
        print("Max swing height: left=%.5f m, right=%.5f m" % (
            float(t["feet"][:, 0, 0, 2].max()),
            float(t["feet"][:, 0, 1, 2].max()),
        ))
        print("No ZMP/MPC, MuJoCo, WBIK, FK measurement, or physical contact feedback.")
        if args.plot:
            plot_results(t)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
