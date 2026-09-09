# lipm_mpc/test.py

from __future__ import annotations

import numpy as np

if __package__:
    from .lipm_model import LIPMModel1D
else:
    from lipm_model import LIPMModel1D


# ============================================================
# Temporary test parameters
# ============================================================
# These parameters will later be moved to run.py.

MPC_TIMESTEP = 0.03       # [s]
COM_HEIGHT = 0.205        # [m]
GRAVITY = 9.81            # [m/s^2]

ATOL = 1e-12
RTOL = 1e-10


def create_model() -> LIPMModel1D:
    return LIPMModel1D(
        timestep=MPC_TIMESTEP,
        com_height=COM_HEIGHT,
        gravity=GRAVITY,
    )


# ============================================================
# Test 1: Model matrices
# ============================================================

def test_model_matrices() -> None:
    model = create_model()

    T = MPC_TIMESTEP

    A_expected = np.array(
        [
            [1.0, T, T**2 / 2.0],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    B_expected = np.array(
        [
            T**3 / 6.0,
            T**2 / 2.0,
            T,
        ],
        dtype=float,
    )

    C_zmp_expected = np.array(
        [
            1.0,
            0.0,
            -COM_HEIGHT / GRAVITY,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        model.A,
        A_expected,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        model.B,
        B_expected,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        model.C_zmp,
        C_zmp_expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print("[PASS] A, B and C_zmp matrices")


# ============================================================
# Test 2: One-step propagation
# ============================================================

def test_one_step_propagation() -> None:
    model = create_model()

    T = MPC_TIMESTEP

    state = np.array(
        [
            0.10,   # position [m]
            0.20,   # velocity [m/s]
            -0.30,  # acceleration [m/s^2]
        ],
        dtype=float,
    )

    jerk = 0.50  # [m/s^3]

    next_state = model.propagate(
        state=state,
        jerk=jerk,
    )

    p0 = state[0]
    v0 = state[1]
    a0 = state[2]

    expected = np.array(
        [
            p0
            + T * v0
            + 0.5 * T**2 * a0
            + (T**3 / 6.0) * jerk,

            v0
            + T * a0
            + 0.5 * T**2 * jerk,

            a0
            + T * jerk,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        next_state,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print("[PASS] One-step state propagation")


# ============================================================
# Test 3: ZMP equation
# ============================================================

def test_zmp_equation() -> None:
    model = create_model()

    state = np.array(
        [
            0.10,  # CoM position [m]
            0.20,  # CoM velocity [m/s]
            0.30,  # CoM acceleration [m/s^2]
        ],
        dtype=float,
    )

    zmp = model.compute_zmp(state)

    expected = (
        state[0]
        - (COM_HEIGHT / GRAVITY) * state[2]
    )

    np.testing.assert_allclose(
        zmp,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print("[PASS] LIPM ZMP equation")


# ============================================================
# Test 4: Zero jerk
# ============================================================

def test_zero_jerk() -> None:
    model = create_model()

    state = np.array(
        [
            0.05,
            0.10,
            0.20,
        ],
        dtype=float,
    )

    next_state = model.propagate(
        state=state,
        jerk=0.0,
    )

    T = MPC_TIMESTEP

    expected = np.array(
        [
            state[0]
            + T * state[1]
            + 0.5 * T**2 * state[2],

            state[1]
            + T * state[2],

            state[2],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        next_state,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print("[PASS] Zero-jerk propagation")


# ============================================================
# Test 5: Multiple steps with constant jerk
# ============================================================

def test_constant_jerk_multiple_steps() -> None:
    model = create_model()

    jerk = 1.50  # [m/s^3]

    state = np.zeros(3, dtype=float)

    number_of_steps = 10

    for _ in range(number_of_steps):
        state = model.propagate(
            state=state,
            jerk=jerk,
        )

    total_time = number_of_steps * MPC_TIMESTEP

    expected = np.array(
        [
            jerk * total_time**3 / 6.0,
            jerk * total_time**2 / 2.0,
            jerk * total_time,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        state,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print("[PASS] Constant-jerk multi-step propagation")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 60)
    print("LIPM MODEL TEST")
    print("=" * 60)

    print(f"MPC timestep : {MPC_TIMESTEP:.6f} s")
    print(f"CoM height   : {COM_HEIGHT:.6f} m")
    print(f"Gravity      : {GRAVITY:.6f} m/s^2")
    print()

    model = create_model()

    print("A =")
    print(model.A)

    print("\nB =")
    print(model.B)

    print("\nC_zmp =")
    print(model.C_zmp)

    print()

    test_model_matrices()
    test_one_step_propagation()
    test_zmp_equation()
    test_zero_jerk()
    test_constant_jerk_multiple_steps()

    print()
    print("=" * 60)
    print("ALL LIPM MODEL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()