import numpy as np

try:

    from .walking_fsm import (
        WalkingFSM,
        WalkingPhase,
        StepType,
    )

except ImportError:

    from walking_fsm import (
        WalkingFSM,
        WalkingPhase,
        StepType,
    )


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# GAIT PARAMETERS
# ============================================================

STEP_LENGTH = 0.04

FEET_SPACING = 0.16

SS_DURATION = 0.18

DS_DURATION = 0.09


# ============================================================
# SETTLED FOOT POSITIONS FROM PREVIOUS TEST
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


def make_fsm():

    return WalkingFSM(
        p_left_initial=(
            P_LEFT_INITIAL
        ),

        p_right_initial=(
            P_RIGHT_INITIAL
        ),

        step_length=(
            STEP_LENGTH
        ),

        feet_spacing=(
            FEET_SPACING
        ),

        single_support_duration=(
            SS_DURATION
        ),

        double_support_duration=(
            DS_DURATION
        ),

        first_swing_side="right",
    )


# ============================================================
# TEST 1
# INITIAL STATE
# ============================================================

separator()
print("TEST 1 - INITIAL DOUBLE SUPPORT")
separator()

fsm = make_fsm()

state = fsm.get_state()

print(
    f"phase = "
    f"{state.phase.value}"
)

if (
    state.phase
    !=
    WalkingPhase.INITIAL_DOUBLE_SUPPORT
):

    raise AssertionError(
        "FSM must start in INITIAL_DOUBLE_SUPPORT."
    )

print("[OK]")


# ============================================================
# TEST 2
# START HALF STEP
# ============================================================

separator()
print("TEST 2 - START HALF STEP")
separator()

fsm.update(
    DS_DURATION
)

state = fsm.get_state()

print(
    f"phase   = "
    f"{state.phase.value}"
)

print(
    f"type    = "
    f"{state.step_type.value}"
)

print(
    f"support = "
    f"{state.support_side}"
)

print(
    f"swing   = "
    f"{state.swing_side}"
)

print(
    f"start   = "
    f"{state.swing_start}"
)

print(
    f"target  = "
    f"{state.swing_target}"
)


if (
    state.phase
    !=
    WalkingPhase.SINGLE_SUPPORT
):

    raise AssertionError(
        "Expected SINGLE_SUPPORT."
    )


if (
    state.step_type
    !=
    StepType.START_HALF_STEP
):

    raise AssertionError(
        "First step must be START_HALF_STEP."
    )


if state.swing_side != "right":

    raise AssertionError(
        "First swing foot must be RIGHT."
    )


expected_half_x = (
    P_LEFT_INITIAL[0]
    +
    0.5 * STEP_LENGTH
)


assert_close(
    state.swing_target[0],
    expected_half_x,
    message=(
        "Start half-step X target is wrong."
    ),
)


print(
    f"\nExpected half step = "
    f"{0.5 * STEP_LENGTH:.6f} m"
)

print("[OK]")


# ============================================================
# TEST 3
# LAND FIRST HALF STEP
# ============================================================

separator()
print("TEST 3 - FIRST HALF STEP LANDING")
separator()

first_target = (
    state.swing_target.copy()
)

fsm.update(
    SS_DURATION
)

state = fsm.get_state()

if (
    state.phase
    !=
    WalkingPhase.DOUBLE_SUPPORT
):

    raise AssertionError(
        "After first swing robot must enter DS."
    )


assert_close(
    state.right_contact_position,
    first_target,
    message=(
        "Right foot was not committed "
        "at first half-step target."
    ),
)


print(
    "right landed =",
    state.right_contact_position,
)

print("[OK]")


# ============================================================
# TEST 4
# FIRST NORMAL STEP
# ============================================================

separator()
print("TEST 4 - FIRST NORMAL STEP")
separator()

fsm.update(
    DS_DURATION
)

state = fsm.get_state()

print(
    f"type    = "
    f"{state.step_type.value}"
)

print(
    f"support = "
    f"{state.support_side}"
)

print(
    f"swing   = "
    f"{state.swing_side}"
)

print(
    f"target  = "
    f"{state.swing_target}"
)


if (
    state.step_type
    !=
    StepType.NORMAL_STEP
):

    raise AssertionError(
        "Expected NORMAL_STEP."
    )


if state.swing_side != "left":

    raise AssertionError(
        "Second swing foot must be LEFT."
    )


expected_x = (
    first_target[0]
    +
    STEP_LENGTH
)


assert_close(
    state.swing_target[0],
    expected_x,
    message=(
        "First normal step target is wrong."
    ),
)


print("[OK]")


# ============================================================
# TEST 5
# NORMAL ALTERNATING STEP
# ============================================================

separator()
print("TEST 5 - ALTERNATING NORMAL STEPS")
separator()

# Finish LEFT normal step.
fsm.update(
    SS_DURATION
)

fsm.update(
    DS_DURATION
)

state = fsm.get_state()


if (
    state.step_type
    !=
    StepType.NORMAL_STEP
):

    raise AssertionError(
        "Expected another NORMAL_STEP."
    )


if state.swing_side != "right":

    raise AssertionError(
        "Swing feet are not alternating."
    )


print(
    f"next swing = "
    f"{state.swing_side}"
)

print(
    f"target     = "
    f"{state.swing_target}"
)

print("[OK]")


# ============================================================
# TEST 6
# PRESS F DURING SINGLE SUPPORT
# ============================================================

separator()
print("TEST 6 - STOP REQUEST DURING WALKING")
separator()

# Move halfway through current RIGHT swing.
fsm.update(
    0.5 * SS_DURATION
)

before_stop = fsm.get_state()

fsm.request_stop()

after_stop = fsm.get_state()


print(
    f"phase before/after stop = "
    f"{before_stop.phase.value}"
)

print(
    f"stop_requested = "
    f"{after_stop.stop_requested}"
)


# Request must NOT interrupt current step.
if (
    after_stop.phase
    !=
    WalkingPhase.SINGLE_SUPPORT
):

    raise AssertionError(
        "request_stop() interrupted "
        "the current swing."
    )


if (
    after_stop.step_index
    !=
    before_stop.step_index
):

    raise AssertionError(
        "Current step changed immediately "
        "after stop request."
    )


print(
    "[OK] Current swing continues after F."
)


# ============================================================
# TEST 7
# FINISH CURRENT STEP
# ============================================================

separator()
print("TEST 7 - FINISH CURRENT STEP")
separator()

fsm.update(
    0.5 * SS_DURATION
)

state = fsm.get_state()


if (
    state.phase
    !=
    WalkingPhase.DOUBLE_SUPPORT
):

    raise AssertionError(
        "Robot must enter DS after "
        "finishing current step."
    )


print(
    "left  contact =",
    state.left_contact_position,
)

print(
    "right contact =",
    state.right_contact_position,
)

print("[OK]")


# ============================================================
# TEST 8
# CLOSING STEP
# ============================================================

separator()
print("TEST 8 - CLOSING STEP")
separator()

# Finish DS after stop request.
fsm.update(
    DS_DURATION
)

state = fsm.get_state()


if (
    state.phase
    !=
    WalkingPhase.SINGLE_SUPPORT
):

    raise AssertionError(
        "Expected closing SINGLE_SUPPORT."
    )


if (
    state.step_type
    !=
    StepType.CLOSING_STEP
):

    raise AssertionError(
        "Expected CLOSING_STEP."
    )


print(
    f"support = "
    f"{state.support_side}"
)

print(
    f"swing   = "
    f"{state.swing_side}"
)

print(
    f"start   = "
    f"{state.swing_start}"
)

print(
    f"target  = "
    f"{state.swing_target}"
)


# Target X must equal stance-foot X.
if state.support_side == "left":

    support_x = (
        state.left_contact_position[0]
    )

else:

    support_x = (
        state.right_contact_position[0]
    )


assert_close(
    state.swing_target[0],
    support_x,
    message=(
        "Closing step must align "
        "the two feet in X."
    ),
)


print(
    "\n[OK] Closing step target "
    "aligns both feet longitudinally."
)


# ============================================================
# TEST 9
# FINAL DOUBLE SUPPORT
# ============================================================

separator()
print("TEST 9 - FINAL DOUBLE SUPPORT")
separator()

fsm.update(
    SS_DURATION
)

state = fsm.get_state()


if (
    state.phase
    !=
    WalkingPhase.FINAL_DOUBLE_SUPPORT
):

    raise AssertionError(
        "Expected FINAL_DOUBLE_SUPPORT."
    )


print(
    "left final  =",
    state.left_contact_position,
)

print(
    "right final =",
    state.right_contact_position,
)


assert_close(
    state.left_contact_position[0],
    state.right_contact_position[0],
    message=(
        "Final feet are not aligned in X."
    ),
)


final_spacing = (
    state.left_contact_position[1]
    -
    state.right_contact_position[1]
)


assert_close(
    final_spacing,
    FEET_SPACING,
    message=(
        "Final feet spacing is wrong."
    ),
)


print(
    f"\nFinal X alignment = "
    f"{state.left_contact_position[0]:+.6f} m"
)

print(
    f"Final feet spacing = "
    f"{final_spacing:.6f} m"
)

print("[OK]")


# ============================================================
# TEST 10
# FINISHED
# ============================================================

separator()
print("TEST 10 - FINISHED")
separator()

fsm.update(
    DS_DURATION
)

state = fsm.get_state()


print(
    f"phase    = "
    f"{state.phase.value}"
)

print(
    f"finished = "
    f"{state.finished}"
)


if (
    state.phase
    !=
    WalkingPhase.FINISHED
):

    raise AssertionError(
        "FSM did not reach FINISHED."
    )


if not state.finished:

    raise AssertionError(
        "finished flag is False."
    )


print("[OK]")


# ============================================================
# TEST 11
# STOP DURING START HALF STEP
# ============================================================

separator()
print(
    "TEST 11 - STOP DURING START HALF STEP"
)
separator()

fsm2 = make_fsm()

# Initial DS -> start half step
fsm2.update(
    DS_DURATION
)

fsm2.update(
    0.5 * SS_DURATION
)

fsm2.request_stop()

# Finish start half step
fsm2.update(
    0.5 * SS_DURATION
)

# Finish DS
fsm2.update(
    DS_DURATION
)

state = fsm2.get_state()


if (
    state.step_type
    !=
    StepType.CLOSING_STEP
):

    raise AssertionError(
        "Stopping after start half-step "
        "must produce a closing step."
    )


# Finish closing step
fsm2.update(
    SS_DURATION
)

state = fsm2.get_state()


assert_close(
    state.left_contact_position[0],
    state.right_contact_position[0],
    message=(
        "Start-stop sequence did not "
        "return feet to aligned X."
    ),
)


print(
    "Final aligned X =",
    state.left_contact_position[0],
)

print("[OK]")


# ============================================================
# FINAL RESULT
# ============================================================

separator()
print(
    "ALL WALKING FSM TESTS PASSED"
)
separator()

print("""
Verified:

  1. Initial double support
  2. First step is a HALF STEP
  3. Normal steps use full STEP_LENGTH
  4. Swing feet alternate LEFT / RIGHT
  5. request_stop() does not interrupt current swing
  6. Current step finishes normally after F
  7. Robot enters double support
  8. A closing step is generated
  9. Closing step aligns x_left = x_right
 10. Final feet spacing remains FEET_SPACING
 11. Robot enters FINAL_DOUBLE_SUPPORT
 12. FSM ends in FINISHED
 13. Stop during the initial half-step also works

No swing trajectory is implemented yet.
No CoM reference is implemented yet.
No differential IK is implemented yet.
""")