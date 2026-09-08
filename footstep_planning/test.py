import numpy as np

from com_reference import (
    compute_com_reference,
)

from swing_trajectory import (
    compute_swing_trajectory,
)


np.set_printoptions(
    precision=8,
    suppress=True,
)


# ============================================================
# PARAMETERS
# ============================================================

SS_DURATION = 0.18

SWING_HEIGHT = 0.04

COM_HEIGHT = 0.205

# Settled CoM obtained previously from MuJoCo.
COM_Y0 = -0.000399


# ============================================================
# SETTLED FOOT POSITIONS
# ============================================================

P_LEFT_INITIAL = np.array(
    [
        -0.029596,
        +0.083945,
        +0.002369,
    ],
    dtype=float,
)


P_RIGHT_INITIAL = np.array(
    [
        -0.029984,
        -0.084298,
        +0.002553,
    ],
    dtype=float,
)


# First half-step target from WalkingFSM
P_RIGHT_TARGET = np.array(
    [
        -0.009596,
        -0.080176,
        +0.002461,
    ],
    dtype=float,
)


# ============================================================
# HELPERS
# ============================================================

def separator():

    print("=" * 90)


def assert_close(
    actual,
    expected,
    tol=1e-9,
    message="",
):

    if not np.allclose(
        actual,
        expected,
        atol=tol,
        rtol=0.0,
    ):

        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


# ============================================================
# TEST 1
# INITIAL DOUBLE SUPPORT
# ============================================================

separator()
print("TEST 1 - INITIAL DOUBLE SUPPORT")
separator()


p_com, v_com = compute_com_reference(
    left_foot_position=P_LEFT_INITIAL,
    right_foot_position=P_RIGHT_INITIAL,
    com_y=COM_Y0,
    com_height=COM_HEIGHT,
)


expected_x = 0.5 * (
    P_LEFT_INITIAL[0]
    +
    P_RIGHT_INITIAL[0]
)


expected_p = np.array(
    [
        expected_x,
        COM_Y0,
        COM_HEIGHT,
    ]
)


print(
    "p_com_ref =",
    p_com,
)

print(
    "v_com_ref =",
    v_com,
)


assert_close(
    p_com,
    expected_p,
    message=(
        "Initial CoM reference is wrong."
    ),
)


assert_close(
    v_com,
    np.zeros(3),
    message=(
        "CoM velocity during static DS "
        "must be zero."
    ),
)


print("[OK]")


# ============================================================
# TEST 2
# MID-SWING
# ============================================================

separator()
print("TEST 2 - MID-SWING")
separator()


p_right_ref, v_right_ref = (
    compute_swing_trajectory(
        start_position=P_RIGHT_INITIAL,
        target_position=P_RIGHT_TARGET,
        phase_time=(
            0.5 * SS_DURATION
        ),
        duration=SS_DURATION,
        swing_height=SWING_HEIGHT,
    )
)


# Left foot is support -> fixed
p_left_ref = (
    P_LEFT_INITIAL.copy()
)

v_left_ref = np.zeros(
    3
)


p_com, v_com = compute_com_reference(
    left_foot_position=p_left_ref,
    right_foot_position=p_right_ref,

    left_foot_velocity=v_left_ref,
    right_foot_velocity=v_right_ref,

    com_y=COM_Y0,
    com_height=COM_HEIGHT,
)


expected_x = 0.5 * (
    p_left_ref[0]
    +
    p_right_ref[0]
)


expected_vx = 0.5 * (
    v_left_ref[0]
    +
    v_right_ref[0]
)


print(
    "left foot ref  =",
    p_left_ref,
)

print(
    "right foot ref =",
    p_right_ref,
)

print(
    "right foot vel =",
    v_right_ref,
)

print(
    "\np_com_ref =",
    p_com,
)

print(
    "v_com_ref =",
    v_com,
)


assert_close(
    p_com[0],
    expected_x,
    message=(
        "CoM X midpoint is wrong."
    ),
)


assert_close(
    v_com[0],
    expected_vx,
    message=(
        "CoM X velocity is wrong."
    ),
)


print("[OK]")


# ============================================================
# TEST 3
# Y MUST REMAIN CONSTANT
# ============================================================

separator()
print("TEST 3 - CONSTANT COM Y")
separator()


foot_midpoint_y = 0.5 * (
    p_left_ref[1]
    +
    p_right_ref[1]
)


print(
    f"foot midpoint y = "
    f"{foot_midpoint_y:+.8f}"
)

print(
    f"CoM y reference = "
    f"{p_com[1]:+.8f}"
)


assert_close(
    p_com[1],
    COM_Y0,
    message=(
        "CoM Y must remain equal "
        "to the initial reference."
    ),
)


# This test deliberately verifies that
# CoM Y is NOT generated from the feet midpoint.
if abs(
    foot_midpoint_y
    -
    COM_Y0
) > 1e-8:

    if np.isclose(
        p_com[1],
        foot_midpoint_y,
        atol=1e-9,
    ):

        raise AssertionError(
            "CoM Y incorrectly follows "
            "the feet midpoint."
        )


print(
    "[OK] CoM Y remains constant."
)


# ============================================================
# TEST 4
# CONSTANT COM HEIGHT
# ============================================================

separator()
print("TEST 4 - CONSTANT COM HEIGHT")
separator()


assert_close(
    p_com[2],
    COM_HEIGHT,
    message=(
        "CoM height is wrong."
    ),
)


assert_close(
    v_com[2],
    0.0,
    message=(
        "Vertical CoM velocity "
        "must be zero."
    ),
)


print(
    f"CoM height = "
    f"{p_com[2]:.6f} m"
)

print("[OK]")


# ============================================================
# TEST 5
# END OF FIRST SWING
# ============================================================

separator()
print("TEST 5 - END OF FIRST SWING")
separator()


p_right_end, v_right_end = (
    compute_swing_trajectory(
        start_position=P_RIGHT_INITIAL,
        target_position=P_RIGHT_TARGET,
        phase_time=SS_DURATION,
        duration=SS_DURATION,
        swing_height=SWING_HEIGHT,
    )
)


p_com_end, v_com_end = (
    compute_com_reference(
        left_foot_position=(
            P_LEFT_INITIAL
        ),

        right_foot_position=(
            p_right_end
        ),

        left_foot_velocity=(
            np.zeros(3)
        ),

        right_foot_velocity=(
            v_right_end
        ),

        com_y=COM_Y0,

        com_height=COM_HEIGHT,
    )
)


expected_end_x = 0.5 * (
    P_LEFT_INITIAL[0]
    +
    P_RIGHT_TARGET[0]
)


print(
    "p_com_end =",
    p_com_end,
)

print(
    "v_com_end =",
    v_com_end,
)


assert_close(
    p_com_end[0],
    expected_end_x,
)


assert_close(
    v_com_end,
    np.zeros(3),
    message=(
        "CoM velocity must return "
        "to zero at swing landing."
    ),
)


print("[OK]")


# ============================================================
# TEST 6
# WHOLE SWING TRAJECTORY
# ============================================================

separator()
print("TEST 6 - WHOLE SWING COM TRAJECTORY")
separator()


times = np.linspace(
    0.0,
    SS_DURATION,
    1001,
)


com_positions = []

com_velocities = []


for t in times:

    p_right, v_right = (
        compute_swing_trajectory(
            start_position=(
                P_RIGHT_INITIAL
            ),

            target_position=(
                P_RIGHT_TARGET
            ),

            phase_time=t,

            duration=(
                SS_DURATION
            ),

            swing_height=(
                SWING_HEIGHT
            ),
        )
    )

    p_com, v_com = (
        compute_com_reference(
            left_foot_position=(
                P_LEFT_INITIAL
            ),

            right_foot_position=(
                p_right
            ),

            left_foot_velocity=(
                np.zeros(3)
            ),

            right_foot_velocity=(
                v_right
            ),

            com_y=(
                COM_Y0
            ),

            com_height=(
                COM_HEIGHT
            ),
        )
    )

    com_positions.append(
        p_com
    )

    com_velocities.append(
        v_com
    )


com_positions = np.asarray(
    com_positions
)

com_velocities = np.asarray(
    com_velocities
)


# ------------------------------------------------------------
# X monotonicity
# ------------------------------------------------------------

dx = np.diff(
    com_positions[:, 0]
)


if np.any(
    dx < -1e-10
):

    raise AssertionError(
        "CoM X reference is not monotonic."
    )


# ------------------------------------------------------------
# Constant Y
# ------------------------------------------------------------

assert_close(
    com_positions[:, 1],
    COM_Y0,
    message=(
        "CoM Y changed during swing."
    ),
)


# ------------------------------------------------------------
# Constant Z
# ------------------------------------------------------------

assert_close(
    com_positions[:, 2],
    COM_HEIGHT,
    message=(
        "CoM height changed during swing."
    ),
)


print(
    f"CoM x start = "
    f"{com_positions[0,0]:+.8f} m"
)

print(
    f"CoM x end   = "
    f"{com_positions[-1,0]:+.8f} m"
)

print(
    f"Max vx      = "
    f"{np.max(com_velocities[:,0]):.8f} m/s"
)


print("[OK]")


# ============================================================
# TEST 7
# DOUBLE SUPPORT AFTER LANDING
# ============================================================

separator()
print("TEST 7 - DOUBLE SUPPORT AFTER LANDING")
separator()


p_com_ds, v_com_ds = (
    compute_com_reference(
        left_foot_position=(
            P_LEFT_INITIAL
        ),

        right_foot_position=(
            P_RIGHT_TARGET
        ),

        com_y=(
            COM_Y0
        ),

        com_height=(
            COM_HEIGHT
        ),
    )
)


assert_close(
    v_com_ds,
    np.zeros(3),
    message=(
        "CoM velocity during DS "
        "must be zero."
    ),
)


print(
    "p_com_ds =",
    p_com_ds,
)

print(
    "v_com_ds =",
    v_com_ds,
)

print("[OK]")


# ============================================================
# FINAL RESULT
# ============================================================

separator()
print(
    "ALL COM REFERENCE TESTS PASSED"
)
separator()


print("""
Verified:

  1. CoM X is the midpoint of the two foot X references
  2. CoM X velocity is the midpoint of foot X velocities
  3. CoM Y remains fixed at the initial settled value
  4. CoM Y does NOT follow the foot Y midpoint
  5. CoM Z remains fixed at 0.205 m
  6. CoM vertical velocity is zero
  7. CoM X moves smoothly during swing
  8. CoM velocity returns to zero at landing
  9. CoM remains stationary during double support

This is a KINEMATIC CoM reference only.

No LIPM dynamics are used.
No ZMP constraints are used.
No MPC is used.
No differential IK is used yet.
""")