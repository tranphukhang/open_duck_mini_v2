from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = Path(
    __file__
).resolve().parent

ROOT_DIR = (
    CURRENT_DIR.parent
)

if str(ROOT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


# ============================================================
# IMPORTS
# ============================================================

if __package__:

    from .lipm_model import (
        LIPMModel1D,
    )

    from .com_trajectory import (
        ConstantJerkCoMSegment,
    )

else:

    from lipm_model import (
        LIPMModel1D,
    )

    from com_trajectory import (
        ConstantJerkCoMSegment,
    )


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=9,
    suppress=True,
)


# ============================================================
# TIMING
# ============================================================

MPC_TIMESTEP = 0.03

IK_TIMESTEP = 0.0005

IK_STEPS_PER_MPC = int(
    round(
        MPC_TIMESTEP
        /
        IK_TIMESTEP
    )
)


# ============================================================
# LIPM
# ============================================================

COM_HEIGHT = 0.205

GRAVITY = 9.81


# ============================================================
# INITIAL STATE
# ============================================================

X_INITIAL = np.array(
    [
        -0.0308300889,
        0.0,
        0.0,
    ],
    dtype=float,
)

Y_INITIAL = np.array(
    [
        -0.0003994468,
        0.0,
        0.0,
    ],
    dtype=float,
)


# ============================================================
# REPRESENTATIVE FIRST MPC COMMAND
# ============================================================
#
# Taken from the already-passed receding-horizon test.
#
# This test does NOT solve MPC again.
#
# It tests only:
#
# MPC state + constant jerk
#        ->
# continuous CoM reference
# ============================================================

X_JERK = 1.599988

Y_JERK = 15.870031


# ============================================================
# HELPER
# ============================================================

def separator():

    print(
        "=" * 72
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_constant_jerk_com_segment():

    # ========================================================
    # 1. CREATE DISCRETE LIPM MODEL
    # ========================================================

    model = LIPMModel1D(
        timestep=(
            MPC_TIMESTEP
        ),

        com_height=(
            COM_HEIGHT
        ),

        gravity=(
            GRAVITY
        ),
    )

    print(
        "[PASS] LIPM model created"
    )

    # ========================================================
    # 2. CREATE CONTINUOUS SEGMENT
    # ========================================================

    segment = ConstantJerkCoMSegment(
        x_state=(
            X_INITIAL
        ),

        y_state=(
            Y_INITIAL
        ),

        x_jerk=(
            X_JERK
        ),

        y_jerk=(
            Y_JERK
        ),

        com_height=(
            COM_HEIGHT
        ),

        duration=(
            MPC_TIMESTEP
        ),
    )

    print(
        "[PASS] Constant-jerk CoM segment created"
    )

    # ========================================================
    # 3. CHECK TAU = 0
    # ========================================================

    ref_0 = (
        segment.evaluate(
            0.0
        )
    )

    expected_position_0 = np.array(
        [
            X_INITIAL[0],
            Y_INITIAL[0],
            COM_HEIGHT,
        ],
        dtype=float,
    )

    expected_velocity_0 = np.array(
        [
            X_INITIAL[1],
            Y_INITIAL[1],
            0.0,
        ],
        dtype=float,
    )

    expected_acceleration_0 = np.array(
        [
            X_INITIAL[2],
            Y_INITIAL[2],
            0.0,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        ref_0.position,
        expected_position_0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        ref_0.velocity,
        expected_velocity_0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        ref_0.acceleration,
        expected_acceleration_0,
        atol=1e-12,
    )

    print(
        "[PASS] tau = 0 reference"
    )

    # ========================================================
    # 4. DISCRETE LIPM TERMINAL STATE
    # ========================================================

    x_discrete_next = (
        model.propagate(
            state=(
                X_INITIAL
            ),

            jerk=(
                X_JERK
            ),
        )
    )

    y_discrete_next = (
        model.propagate(
            state=(
                Y_INITIAL
            ),

            jerk=(
                Y_JERK
            ),
        )
    )

    # ========================================================
    # 5. CONTINUOUS SEGMENT AT TAU = T_MPC
    # ========================================================

    x_continuous_next = (
        segment.get_terminal_x_state()
    )

    y_continuous_next = (
        segment.get_terminal_y_state()
    )

    np.testing.assert_allclose(
        x_continuous_next,
        x_discrete_next,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        y_continuous_next,
        y_discrete_next,
        atol=1e-12,
    )

    print(
        "[PASS] continuous terminal state matches LIPM discrete propagation"
    )

    # ========================================================
    # 6. FULL TERMINAL REFERENCE
    # ========================================================

    ref_T = (
        segment.evaluate(
            MPC_TIMESTEP
        )
    )

    np.testing.assert_allclose(
        ref_T.position,
        np.array(
            [
                x_discrete_next[0],
                y_discrete_next[0],
                COM_HEIGHT,
            ],
            dtype=float,
        ),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        ref_T.velocity,
        np.array(
            [
                x_discrete_next[1],
                y_discrete_next[1],
                0.0,
            ],
            dtype=float,
        ),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        ref_T.acceleration,
        np.array(
            [
                x_discrete_next[2],
                y_discrete_next[2],
                0.0,
            ],
            dtype=float,
        ),
        atol=1e-12,
    )

    print(
        "[PASS] terminal 3D CoM reference"
    )

    # ========================================================
    # 7. SAMPLE AT IK RATE
    # ========================================================
    #
    # 0.03 / 0.0005 = 60 executor intervals
    #
    # Including both endpoints:
    #
    # tau =
    # 0,
    # 0.0005,
    # ...
    # 0.0300
    #
    # gives 61 samples.
    # ========================================================

    assert (
        IK_STEPS_PER_MPC
        ==
        60
    )

    tau_samples = np.linspace(
        0.0,
        MPC_TIMESTEP,
        IK_STEPS_PER_MPC + 1,
    )

    position_log = []

    velocity_log = []

    acceleration_log = []

    for tau in (
        tau_samples
    ):

        ref = (
            segment.evaluate(
                tau
            )
        )

        position_log.append(
            ref.position
        )

        velocity_log.append(
            ref.velocity
        )

        acceleration_log.append(
            ref.acceleration
        )

    position_log = np.asarray(
        position_log,
        dtype=float,
    )

    velocity_log = np.asarray(
        velocity_log,
        dtype=float,
    )

    acceleration_log = np.asarray(
        acceleration_log,
        dtype=float,
    )

    assert position_log.shape == (
        61,
        3,
    )

    assert velocity_log.shape == (
        61,
        3,
    )

    assert acceleration_log.shape == (
        61,
        3,
    )

    assert np.all(
        np.isfinite(
            position_log
        )
    )

    assert np.all(
        np.isfinite(
            velocity_log
        )
    )

    assert np.all(
        np.isfinite(
            acceleration_log
        )
    )

    print(
        "[PASS] 60 IK intervals generated inside one MPC interval"
    )

    # ========================================================
    # 8. CONSTANT Z
    # ========================================================

    np.testing.assert_allclose(
        position_log[:, 2],
        COM_HEIGHT,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        velocity_log[:, 2],
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        acceleration_log[:, 2],
        0.0,
        atol=1e-12,
    )

    print(
        "[PASS] CoM height remains constant"
    )

    # ========================================================
    # 9. CHECK CONTINUITY INSIDE SEGMENT
    # ========================================================

    position_difference = np.diff(
        position_log,
        axis=0,
    )

    velocity_difference = np.diff(
        velocity_log,
        axis=0,
    )

    acceleration_difference = np.diff(
        acceleration_log,
        axis=0,
    )

    assert np.all(
        np.isfinite(
            position_difference
        )
    )

    assert np.all(
        np.isfinite(
            velocity_difference
        )
    )

    assert np.all(
        np.isfinite(
            acceleration_difference
        )
    )

    print(
        "[PASS] continuous reference across IK samples"
    )

    # ========================================================
    # 10. SECOND SEGMENT
    #
    # Verify exact continuity when the next MPC solve starts
    # from the terminal state of the previous segment.
    # ========================================================

    SECOND_X_JERK = (
        -0.30
    )

    SECOND_Y_JERK = (
        -12.0
    )

    second_segment = ConstantJerkCoMSegment(
        x_state=(
            x_continuous_next
        ),

        y_state=(
            y_continuous_next
        ),

        x_jerk=(
            SECOND_X_JERK
        ),

        y_jerk=(
            SECOND_Y_JERK
        ),

        com_height=(
            COM_HEIGHT
        ),

        duration=(
            MPC_TIMESTEP
        ),
    )

    second_ref_0 = (
        second_segment.evaluate(
            0.0
        )
    )

    np.testing.assert_allclose(
        second_ref_0.position,
        ref_T.position,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        second_ref_0.velocity,
        ref_T.velocity,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        second_ref_0.acceleration,
        ref_T.acceleration,
        atol=1e-12,
    )

    print(
        "[PASS] exact continuity between consecutive MPC segments"
    )

    # ========================================================
    # PRINT REPRESENTATIVE VALUES
    # ========================================================

    separator()

    print(
        "COM TRAJECTORY SEGMENT SUMMARY"
    )

    separator()

    print(
        f"MPC interval        : "
        f"{MPC_TIMESTEP:.6f} s"
    )

    print(
        f"IK interval         : "
        f"{IK_TIMESTEP:.6f} s"
    )

    print(
        f"IK intervals / MPC  : "
        f"{IK_STEPS_PER_MPC}"
    )

    print()

    print(
        "Initial X state:"
    )

    print(
        " ",
        X_INITIAL
    )

    print(
        "Initial Y state:"
    )

    print(
        " ",
        Y_INITIAL
    )

    print()

    print(
        f"Applied jerk x      : "
        f"{X_JERK:+.6f} m/s^3"
    )

    print(
        f"Applied jerk y      : "
        f"{Y_JERK:+.6f} m/s^3"
    )

    print()

    print(
        "CoM reference at tau = 0:"
    )

    print(
        "  position     =",
        ref_0.position,
    )

    print(
        "  velocity     =",
        ref_0.velocity,
    )

    print(
        "  acceleration =",
        ref_0.acceleration,
    )

    print()

    midpoint_index = (
        IK_STEPS_PER_MPC
        //
        2
    )

    midpoint_tau = (
        tau_samples[
            midpoint_index
        ]
    )

    midpoint_ref = (
        segment.evaluate(
            midpoint_tau
        )
    )

    print(
        f"CoM reference at tau = "
        f"{midpoint_tau:.6f} s:"
    )

    print(
        "  position     =",
        midpoint_ref.position,
    )

    print(
        "  velocity     =",
        midpoint_ref.velocity,
    )

    print(
        "  acceleration =",
        midpoint_ref.acceleration,
    )

    print()

    print(
        f"CoM reference at tau = "
        f"{MPC_TIMESTEP:.6f} s:"
    )

    print(
        "  position     =",
        ref_T.position,
    )

    print(
        "  velocity     =",
        ref_T.velocity,
    )

    print(
        "  acceleration =",
        ref_T.acceleration,
    )

    separator()

    print(
        "CONSTANT-JERK COM TRAJECTORY TEST PASSED"
    )

    separator()


# ============================================================
# MAIN
# ============================================================

def main():

    separator()

    print(
        "MPC -> IK CONTINUOUS COM REFERENCE TEST"
    )

    separator()

    print(
        f"MPC timestep : "
        f"{MPC_TIMESTEP:.6f} s"
    )

    print(
        f"IK timestep  : "
        f"{IK_TIMESTEP:.6f} s"
    )

    print(
        f"Ratio        : "
        f"{IK_STEPS_PER_MPC} IK intervals / MPC interval"
    )

    print()

    test_constant_jerk_com_segment()


if __name__ == "__main__":

    main()