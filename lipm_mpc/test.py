# lipm_mpc/test.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# IMPORTS
# ============================================================

if __package__:
    from .lipm_model import LIPMModel1D
    from .support_preview import (
        build_support_preview_x,
        compute_sole_corners,
        compute_support_bounds_x,
        compute_support_polygon,
    )
else:
    from lipm_model import LIPMModel1D
    from support_preview import (
        build_support_preview_x,
        compute_sole_corners,
        compute_support_bounds_x,
        compute_support_polygon,
    )


from footstep_planning.walking_fsm import (
    WalkingFSM,
)


# ============================================================
# TEMPORARY TEST PARAMETERS
# ============================================================
#
# These parameters will later be moved to lipm_mpc/run.py.
#
# ============================================================

MPC_TIMESTEP = 0.03
MPC_HORIZON_STEPS = 16

COM_HEIGHT = 0.205
GRAVITY = 9.81

SINGLE_SUPPORT_DURATION = 0.18
DOUBLE_SUPPORT_DURATION = 0.09

STEP_LENGTH = 0.04
FEET_SPACING = 0.16

ZMP_SUPPORT_SCALE = 0.9

# Same approximate sole geometry currently used
# by walking_visualization.
FOOT_TOE = 0.0645
FOOT_HEEL = 0.0386
FOOT_HALF_WIDTH = 0.02065


# Representative settled foot-site positions.
P_LEFT_INITIAL = np.array(
    [
        -0.0295957703,
        +0.0839454760,
        0.0023691268,
    ],
    dtype=float,
)

P_RIGHT_INITIAL = np.array(
    [
        -0.0299839476,
        -0.0842981118,
        0.0025529409,
    ],
    dtype=float,
)


# Straight walking:
# foot yaw remains aligned with world x-y axes.
R_LEFT_INITIAL = np.eye(
    3,
    dtype=float,
)

R_RIGHT_INITIAL = np.eye(
    3,
    dtype=float,
)


ATOL = 1e-12
RTOL = 1e-10


# ============================================================
# FACTORIES
# ============================================================

def create_model() -> LIPMModel1D:

    return LIPMModel1D(
        timestep=MPC_TIMESTEP,
        com_height=COM_HEIGHT,
        gravity=GRAVITY,
    )


def create_fsm() -> WalkingFSM:

    return WalkingFSM(
        p_left_initial=P_LEFT_INITIAL,
        p_right_initial=P_RIGHT_INITIAL,
        step_length=STEP_LENGTH,
        feet_spacing=FEET_SPACING,
        single_support_duration=(
            SINGLE_SUPPORT_DURATION
        ),
        double_support_duration=(
            DOUBLE_SUPPORT_DURATION
        ),
        first_swing_side="right",
    )


# ============================================================
# LIPM TEST 1
# ============================================================

def test_model_matrices() -> None:

    model = create_model()

    T = MPC_TIMESTEP

    A_expected = np.array(
        [
            [
                1.0,
                T,
                T**2 / 2.0,
            ],
            [
                0.0,
                1.0,
                T,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
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

    print(
        "[PASS] A, B and C_zmp matrices"
    )


# ============================================================
# LIPM TEST 2
# ============================================================

def test_one_step_propagation() -> None:

    model = create_model()

    T = MPC_TIMESTEP

    state = np.array(
        [
            0.10,
            0.20,
            -0.30,
        ],
        dtype=float,
    )

    jerk = 0.50

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
            + T**3 / 6.0 * jerk,

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

    print(
        "[PASS] One-step state propagation"
    )


# ============================================================
# LIPM TEST 3
# ============================================================

def test_zmp_equation() -> None:

    model = create_model()

    state = np.array(
        [
            0.10,
            0.20,
            0.30,
        ],
        dtype=float,
    )

    zmp = model.compute_zmp(
        state
    )

    expected = (
        state[0]
        -
        COM_HEIGHT
        /
        GRAVITY
        *
        state[2]
    )

    np.testing.assert_allclose(
        zmp,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "[PASS] LIPM ZMP equation"
    )


# ============================================================
# LIPM TEST 4
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

    print(
        "[PASS] Zero-jerk propagation"
    )


# ============================================================
# LIPM TEST 5
# ============================================================

def test_constant_jerk_multiple_steps() -> None:

    model = create_model()

    jerk = 1.50

    state = np.zeros(
        3,
        dtype=float,
    )

    number_of_steps = 10

    for _ in range(
        number_of_steps
    ):
        state = model.propagate(
            state=state,
            jerk=jerk,
        )

    total_time = (
        number_of_steps
        *
        MPC_TIMESTEP
    )

    expected = np.array(
        [
            jerk
            *
            total_time**3
            /
            6.0,

            jerk
            *
            total_time**2
            /
            2.0,

            jerk
            *
            total_time,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        state,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "[PASS] Constant-jerk "
        "multi-step propagation"
    )


# ============================================================
# SUPPORT TEST 1
# ============================================================

def test_sole_corners() -> None:

    position = np.array(
        [
            0.10,
            0.08,
            0.0,
        ],
        dtype=float,
    )

    rotation = np.eye(
        3,
        dtype=float,
    )

    corners = compute_sole_corners(
        position=position,
        rotation=rotation,
        foot_toe=FOOT_TOE,
        foot_heel=FOOT_HEEL,
        foot_half_width=(
            FOOT_HALF_WIDTH
        ),
    )

    assert corners.shape == (4, 3)

    expected_x_min = (
        position[0]
        -
        FOOT_HEEL
    )

    expected_x_max = (
        position[0]
        +
        FOOT_TOE
    )

    expected_y_min = (
        position[1]
        -
        FOOT_HALF_WIDTH
    )

    expected_y_max = (
        position[1]
        +
        FOOT_HALF_WIDTH
    )

    np.testing.assert_allclose(
        np.min(corners[:, 0]),
        expected_x_min,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        np.max(corners[:, 0]),
        expected_x_max,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        np.min(corners[:, 1]),
        expected_y_min,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        np.max(corners[:, 1]),
        expected_y_max,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "[PASS] Sole-corner geometry"
    )


# ============================================================
# SUPPORT TEST 2
# ============================================================

def test_single_support_bounds() -> None:

    polygon = compute_support_polygon(
        left_position=P_LEFT_INITIAL,
        left_rotation=R_LEFT_INITIAL,
        right_position=P_RIGHT_INITIAL,
        right_rotation=R_RIGHT_INITIAL,
        phase="SINGLE_SUPPORT",
        support_side="left",
        foot_toe=FOOT_TOE,
        foot_heel=FOOT_HEEL,
        foot_half_width=(
            FOOT_HALF_WIDTH
        ),
    )

    x_min, x_max = (
        compute_support_bounds_x(
            polygon=polygon,
            zmp_scale=(
                ZMP_SUPPORT_SCALE
            ),
        )
    )

    raw_min = (
        P_LEFT_INITIAL[0]
        -
        FOOT_HEEL
    )

    raw_max = (
        P_LEFT_INITIAL[0]
        +
        FOOT_TOE
    )

    center = 0.5 * (
        raw_min
        +
        raw_max
    )

    expected_min = (
        center
        +
        ZMP_SUPPORT_SCALE
        *
        (
            raw_min
            -
            center
        )
    )

    expected_max = (
        center
        +
        ZMP_SUPPORT_SCALE
        *
        (
            raw_max
            -
            center
        )
    )

    np.testing.assert_allclose(
        x_min,
        expected_min,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        x_max,
        expected_max,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "[PASS] Single-support x bounds"
    )


# ============================================================
# SUPPORT TEST 3
# ============================================================

def test_double_support_bounds() -> None:

    polygon = compute_support_polygon(
        left_position=P_LEFT_INITIAL,
        left_rotation=R_LEFT_INITIAL,
        right_position=P_RIGHT_INITIAL,
        right_rotation=R_RIGHT_INITIAL,
        phase="DOUBLE_SUPPORT",
        support_side="both",
        foot_toe=FOOT_TOE,
        foot_heel=FOOT_HEEL,
        foot_half_width=(
            FOOT_HALF_WIDTH
        ),
    )

    x_min, x_max = (
        compute_support_bounds_x(
            polygon=polygon,
            zmp_scale=(
                ZMP_SUPPORT_SCALE
            ),
        )
    )

    assert x_min < x_max

    raw_min = min(
        P_LEFT_INITIAL[0],
        P_RIGHT_INITIAL[0],
    ) - FOOT_HEEL

    raw_max = max(
        P_LEFT_INITIAL[0],
        P_RIGHT_INITIAL[0],
    ) + FOOT_TOE

    center = 0.5 * (
        raw_min
        +
        raw_max
    )

    expected_min = (
        center
        +
        ZMP_SUPPORT_SCALE
        *
        (
            raw_min
            -
            center
        )
    )

    expected_max = (
        center
        +
        ZMP_SUPPORT_SCALE
        *
        (
            raw_max
            -
            center
        )
    )

    np.testing.assert_allclose(
        x_min,
        expected_min,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        x_max,
        expected_max,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "[PASS] Double-support x bounds"
    )


# ============================================================
# SUPPORT TEST 4
# ============================================================

def test_support_preview_schedule() -> None:

    fsm = create_fsm()

    preview = build_support_preview_x(
        fsm=fsm,
        left_rotation=R_LEFT_INITIAL,
        right_rotation=R_RIGHT_INITIAL,
        timestep=MPC_TIMESTEP,
        horizon_steps=(
            MPC_HORIZON_STEPS
        ),
        foot_toe=FOOT_TOE,
        foot_heel=FOOT_HEEL,
        foot_half_width=(
            FOOT_HALF_WIDTH
        ),
        zmp_scale=(
            ZMP_SUPPORT_SCALE
        ),
    )

    assert (
        preview.horizon_steps
        ==
        MPC_HORIZON_STEPS
    )

    assert preview.x_min.shape == (
        MPC_HORIZON_STEPS,
    )

    assert preview.x_max.shape == (
        MPC_HORIZON_STEPS,
    )

    assert np.all(
        preview.x_min
        <
        preview.x_max
    )

    expected_phase = (
        ["INITIAL_DOUBLE_SUPPORT"] * 3
        +
        ["SINGLE_SUPPORT"] * 6
        +
        ["DOUBLE_SUPPORT"] * 3
        +
        ["SINGLE_SUPPORT"] * 4
    )

    expected_support = (
        ["both"] * 3
        +
        ["left"] * 6
        +
        ["both"] * 3
        +
        ["right"] * 4
    )

    assert list(
        preview.phase
    ) == expected_phase

    assert list(
        preview.support_side
    ) == expected_support

    print(
        "[PASS] 16-step support preview schedule"
    )


# ============================================================
# SUPPORT TEST 5
# ============================================================

def test_preview_does_not_modify_fsm() -> None:

    fsm = create_fsm()

    state_before = (
        fsm.get_state()
    )

    phase_before = (
        state_before.phase
    )

    phase_time_before = (
        state_before.phase_time
    )

    left_before = (
        state_before
        .left_contact_position
        .copy()
    )

    right_before = (
        state_before
        .right_contact_position
        .copy()
    )

    build_support_preview_x(
        fsm=fsm,
        left_rotation=R_LEFT_INITIAL,
        right_rotation=R_RIGHT_INITIAL,
        timestep=MPC_TIMESTEP,
        horizon_steps=(
            MPC_HORIZON_STEPS
        ),
        foot_toe=FOOT_TOE,
        foot_heel=FOOT_HEEL,
        foot_half_width=(
            FOOT_HALF_WIDTH
        ),
        zmp_scale=(
            ZMP_SUPPORT_SCALE
        ),
    )

    state_after = (
        fsm.get_state()
    )

    assert (
        state_after.phase
        ==
        phase_before
    )

    np.testing.assert_allclose(
        state_after.phase_time,
        phase_time_before,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        state_after.left_contact_position,
        left_before,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        state_after.right_contact_position,
        right_before,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        "[PASS] Support preview "
        "does not modify real FSM"
    )


# ============================================================
# PRINT SUPPORT PREVIEW
# ============================================================

def print_support_preview() -> None:

    fsm = create_fsm()

    preview = build_support_preview_x(
        fsm=fsm,
        left_rotation=R_LEFT_INITIAL,
        right_rotation=R_RIGHT_INITIAL,
        timestep=MPC_TIMESTEP,
        horizon_steps=(
            MPC_HORIZON_STEPS
        ),
        foot_toe=FOOT_TOE,
        foot_heel=FOOT_HEEL,
        foot_half_width=(
            FOOT_HALF_WIDTH
        ),
        zmp_scale=(
            ZMP_SUPPORT_SCALE
        ),
    )

    print()
    print(
        "SUPPORT PREVIEW"
    )
    print(
        "-" * 82
    )

    print(
        f"{'k':>2} "
        f"{'t [s]':>8} "
        f"{'phase':>24} "
        f"{'support':>8} "
        f"{'x_min [m]':>12} "
        f"{'x_max [m]':>12}"
    )

    print(
        "-" * 82
    )

    for k in range(
        preview.horizon_steps
    ):

        print(
            f"{k:2d} "
            f"{preview.time[k]:8.3f} "
            f"{preview.phase[k]:>24} "
            f"{preview.support_side[k]:>8} "
            f"{preview.x_min[k]:12.6f} "
            f"{preview.x_max[k]:12.6f}"
        )

    print(
        "-" * 82
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 60
    )
    print(
        "LIPM-MPC MODEL AND SUPPORT PREVIEW TEST"
    )
    print(
        "=" * 60
    )

    print(
        f"MPC timestep       : "
        f"{MPC_TIMESTEP:.6f} s"
    )

    print(
        f"MPC horizon steps  : "
        f"{MPC_HORIZON_STEPS}"
    )

    print(
        f"MPC horizon time   : "
        f"{MPC_TIMESTEP * MPC_HORIZON_STEPS:.6f} s"
    )

    print(
        f"Single support     : "
        f"{SINGLE_SUPPORT_DURATION:.6f} s"
    )

    print(
        f"Double support     : "
        f"{DOUBLE_SUPPORT_DURATION:.6f} s"
    )

    print(
        f"ZMP support scale  : "
        f"{ZMP_SUPPORT_SCALE:.3f}"
    )

    print()

    # --------------------------------------------------------
    # LIPM
    # --------------------------------------------------------

    model = create_model()

    print(
        "A ="
    )
    print(
        model.A
    )

    print(
        "\nB ="
    )
    print(
        model.B
    )

    print(
        "\nC_zmp ="
    )
    print(
        model.C_zmp
    )

    print()

    test_model_matrices()
    test_one_step_propagation()
    test_zmp_equation()
    test_zero_jerk()
    test_constant_jerk_multiple_steps()

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    print()

    test_sole_corners()
    test_single_support_bounds()
    test_double_support_bounds()
    test_support_preview_schedule()
    test_preview_does_not_modify_fsm()

    # --------------------------------------------------------
    # DEBUG TABLE
    # --------------------------------------------------------

    print_support_preview()

    print()
    print(
        "=" * 60
    )
    print(
        "ALL TESTS PASSED"
    )
    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()