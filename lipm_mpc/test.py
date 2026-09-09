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

    from .lipm_model import (
        LIPMModel1D,
    )

    from .support_preview import (
        build_support_preview,
    )

    from .mpc_1d import (
        LIPMMPC1D,
    )

else:

    from lipm_model import (
        LIPMModel1D,
    )

    from support_preview import (
        build_support_preview,
    )

    from mpc_1d import (
        LIPMMPC1D,
    )


from footstep_planning.walking_fsm import (
    WalkingFSM,
)


# ============================================================
# TEMPORARY MPC PARAMETERS
# ============================================================

MPC_TIMESTEP = 0.03

# ------------------------------------------------------------
# Important change:
#
# 48 * 0.03 = 1.44 s preview horizon
# ------------------------------------------------------------

MPC_HORIZON_STEPS = 48

COM_HEIGHT = 0.205

GRAVITY = 9.81


# ------------------------------------------------------------
# Keep exactly the same weights as previous test.
#
# We want to study ONLY the effect of horizon length.
# ------------------------------------------------------------

TERMINAL_WEIGHT = 1.0

CONTROL_WEIGHT = 2e-5


# ============================================================
# WALKING PARAMETERS
# ============================================================

STEP_LENGTH = 0.04

FEET_SPACING = 0.16


# ============================================================
# GAIT TIMING
# ============================================================

SINGLE_SUPPORT_DURATION = 0.18

DOUBLE_SUPPORT_DURATION = 0.27

INITIAL_DOUBLE_SUPPORT_DURATION = 0.36


# ============================================================
# SUPPORT REGION
# ============================================================

ZMP_SUPPORT_SCALE = 0.9

FOOT_TOE = 0.0645

FOOT_HEEL = 0.0386

FOOT_HALF_WIDTH = 0.02065


# ============================================================
# SETTLED INITIAL CONFIGURATION
# ============================================================

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


P_COM_INITIAL = np.array(
    [
        -0.0308300889,
        -0.0003994468,
        0.2048464890,
    ],
    dtype=float,
)


# ============================================================
# FOOT ORIENTATION
# ============================================================

R_LEFT_INITIAL = np.eye(
    3,
    dtype=float,
)

R_RIGHT_INITIAL = np.eye(
    3,
    dtype=float,
)


# ============================================================
# SOLVER
# ============================================================

SOLVER_OPTIONS = {
    "ftol": 1e-10,
    "maxiter": 2000,
}


# ============================================================
# CREATE FSM
# ============================================================

def create_fsm():

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
            SINGLE_SUPPORT_DURATION
        ),

        double_support_duration=(
            DOUBLE_SUPPORT_DURATION
        ),

        first_swing_side="right",

        initial_double_support_duration=(
            INITIAL_DOUBLE_SUPPORT_DURATION
        ),
    )


# ============================================================
# CREATE ONE GENERIC 1D MPC
# ============================================================

def create_mpc():

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

    mpc = LIPMMPC1D(
        model=model,

        horizon_steps=(
            MPC_HORIZON_STEPS
        ),

        terminal_weight=(
            TERMINAL_WEIGHT
        ),

        control_weight=(
            CONTROL_WEIGHT
        ),
    )

    return (
        model,
        mpc,
    )


# ============================================================
# CHECK ONE MPC AXIS
# ============================================================

def check_axis_result(
    axis_name,
    model,
    result,
    lower_bounds,
    upper_bounds,
):

    assert result.success

    assert result.control.shape == (
        MPC_HORIZON_STEPS,
    )

    assert result.state.shape == (
        MPC_HORIZON_STEPS + 1,
        3,
    )

    assert result.zmp.shape == (
        MPC_HORIZON_STEPS,
    )

    # --------------------------------------------------------
    # Dynamics consistency
    # --------------------------------------------------------

    for k in range(
        MPC_HORIZON_STEPS
    ):

        expected_next = (
            model.propagate(
                state=(
                    result.state[k]
                ),

                jerk=(
                    result.control[k]
                ),
            )
        )

        np.testing.assert_allclose(
            result.state[k + 1],
            expected_next,
            atol=1e-8,
        )

    # --------------------------------------------------------
    # ZMP constraints
    # --------------------------------------------------------

    tolerance = 1e-6

    assert np.all(
        result.zmp
        >=
        lower_bounds
        -
        tolerance
    )

    assert np.all(
        result.zmp
        <=
        upper_bounds
        +
        tolerance
    )

    print(
        f"[PASS] {axis_name}-MPC dynamics"
    )

    print(
        f"[PASS] {axis_name}-MPC ZMP constraints"
    )


# ============================================================
# PRINT AXIS SUMMARY
# ============================================================

def print_axis_summary(
    axis_name,
    current_state,
    goal_state,
    result,
):

    initial_position_error = abs(
        current_state[0]
        -
        goal_state[0]
    )

    terminal_position_error = abs(
        result.state[-1, 0]
        -
        goal_state[0]
    )

    max_jerk = np.max(
        np.abs(
            result.control
        )
    )

    max_jerk_index = int(
        np.argmax(
            np.abs(
                result.control
            )
        )
    )

    max_jerk_time = (
        max_jerk_index
        *
        MPC_TIMESTEP
    )

    print()

    print(
        "-" * 72
    )

    print(
        f"{axis_name}-MPC SUMMARY"
    )

    print(
        "-" * 72
    )

    print(
        f"Current CoM {axis_name.lower():<7}: "
        f"{current_state[0]: .6f} m"
    )

    print(
        f"Goal {axis_name.lower():<14}: "
        f"{goal_state[0]: .6f} m"
    )

    print(
        f"Terminal CoM {axis_name.lower():<6}: "
        f"{result.state[-1, 0]: .6f} m"
    )

    print(
        f"Terminal velocity : "
        f"{result.state[-1, 1]: .6f} m/s"
    )

    print(
        f"Terminal accel.   : "
        f"{result.state[-1, 2]: .6f} m/s^2"
    )

    print(
        f"First jerk        : "
        f"{result.first_control: .6f} m/s^3"
    )

    print(
        f"Max |jerk|        : "
        f"{max_jerk: .6f} m/s^3"
    )

    print(
        f"Max jerk index    : "
        f"{max_jerk_index}"
    )

    print(
        f"Max jerk time     : "
        f"{max_jerk_time:.3f} s"
    )

    print(
        f"Initial pos error : "
        f"{initial_position_error:.6f} m"
    )

    print(
        f"Terminal pos error: "
        f"{terminal_position_error:.6f} m"
    )

    print(
        f"Objective         : "
        f"{result.objective:.10f}"
    )

    print(
        f"Solver iterations : "
        f"{result.iterations}"
    )


# ============================================================
# PRINT PREVIEW TABLE
# ============================================================

def print_preview_table(
    preview,
    x_result,
    y_result,
):

    print()

    print(
        "=" * 138
    )

    print(
        "2D ZMP PREVIEW RESULT"
    )

    print(
        "=" * 138
    )

    print(
        f"{'k':>2} "
        f"{'t':>6} "
        f"{'phase':>24} "
        f"{'support':>8} "
        f"{'x_Z':>10} "
        f"{'x_min':>10} "
        f"{'x_max':>10} "
        f"{'y_Z':>10} "
        f"{'y_min':>10} "
        f"{'y_max':>10}"
    )

    print(
        "-" * 138
    )

    for k in range(
        MPC_HORIZON_STEPS
    ):

        print(
            f"{k:2d} "
            f"{preview.time[k]:6.3f} "
            f"{preview.phase[k]:>24} "
            f"{preview.support_side[k]:>8} "
            f"{x_result.zmp[k]:10.6f} "
            f"{preview.x_min[k]:10.6f} "
            f"{preview.x_max[k]:10.6f} "
            f"{y_result.zmp[k]:10.6f} "
            f"{preview.y_min[k]:10.6f} "
            f"{preview.y_max[k]:10.6f}"
        )

    print(
        "-" * 138
    )


# ============================================================
# X/Y MPC INTEGRATION TEST
# ============================================================

def test_xy_mpc_with_support_preview():

    # ========================================================
    # 1. FSM
    # ========================================================

    fsm = create_fsm()

    swing_target = (
        fsm.get_next_swing_target()
    )

    assert (
        swing_target
        is not None
    )

    print(
        "[PASS] FSM next swing target"
    )

    # ========================================================
    # 2. SUPPORT PREVIEW
    # ========================================================

    preview = (
        build_support_preview(
            fsm=fsm,

            left_rotation=(
                R_LEFT_INITIAL
            ),

            right_rotation=(
                R_RIGHT_INITIAL
            ),

            timestep=(
                MPC_TIMESTEP
            ),

            horizon_steps=(
                MPC_HORIZON_STEPS
            ),

            foot_toe=(
                FOOT_TOE
            ),

            foot_heel=(
                FOOT_HEEL
            ),

            foot_half_width=(
                FOOT_HALF_WIDTH
            ),

            zmp_scale=(
                ZMP_SUPPORT_SCALE
            ),
        )
    )

    assert (
        preview.horizon_steps
        ==
        MPC_HORIZON_STEPS
    )

    print(
        "[PASS] 2D support preview"
    )

    # ========================================================
    # 3. INITIAL LIPM STATES
    # ========================================================

    x_current = np.array(
        [
            P_COM_INITIAL[0],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    y_current = np.array(
        [
            P_COM_INITIAL[1],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # ========================================================
    # 4. TERMINAL GOALS
    # ========================================================
    #
    # Terminal state:
    #
    #     [p_goal, 0, 0]
    #
    # If terminal acceleration is zero:
    #
    #     p_Z,N = p_G,N
    #
    # Therefore the terminal CoM position should be
    # consistent with the support region at the END
    # of the prediction horizon.
    #
    # For the current decoupled x/y formulation, use the
    # center of the final safe support bounds.
    # ========================================================

    x_terminal_center = 0.5 * (
        preview.x_min[-1]
        +
        preview.x_max[-1]
    )

    y_terminal_center = 0.5 * (
        preview.y_min[-1]
        +
        preview.y_max[-1]
    )


    x_goal = np.array(
        [
            x_terminal_center,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    y_goal = np.array(
        [
            y_terminal_center,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    print(
        f"Terminal support phase : "
        f"{preview.phase[-1]}"
    )

    print(
        f"Terminal support side  : "
        f"{preview.support_side[-1]}"
    )

    print(
        f"Terminal goal x        : "
        f"{x_terminal_center:.6f} m"
    )

    print(
        f"Terminal goal y        : "
        f"{y_terminal_center:.6f} m"
    )

    # ========================================================
    # 5. CREATE TWO INSTANCES OF SAME 1D MPC
    # ========================================================

    x_model, x_mpc = (
        create_mpc()
    )

    y_model, y_mpc = (
        create_mpc()
    )

    assert (
        type(x_mpc)
        is
        type(y_mpc)
    )

    print(
        "[PASS] Same LIPMMPC1D used for x and y"
    )

    # ========================================================
    # 6. SOLVE X
    # ========================================================

    x_result = (
        x_mpc.solve(
            current_state=(
                x_current
            ),

            goal_state=(
                x_goal
            ),

            lower_bounds=(
                preview.x_min
            ),

            upper_bounds=(
                preview.x_max
            ),

            solver_options=(
                SOLVER_OPTIONS
            ),
        )
    )

    check_axis_result(
        axis_name="X",

        model=x_model,

        result=x_result,

        lower_bounds=(
            preview.x_min
        ),

        upper_bounds=(
            preview.x_max
        ),
    )

    # ========================================================
    # 7. SOLVE Y
    # ========================================================

    y_result = (
        y_mpc.solve(
            current_state=(
                y_current
            ),

            goal_state=(
                y_goal
            ),

            lower_bounds=(
                preview.y_min
            ),

            upper_bounds=(
                preview.y_max
            ),

            solver_options=(
                SOLVER_OPTIONS
            ),
        )
    )

    check_axis_result(
        axis_name="Y",

        model=y_model,

        result=y_result,

        lower_bounds=(
            preview.y_min
        ),

        upper_bounds=(
            preview.y_max
        ),
    )

    # ========================================================
    # 8. PRINT RESULTS
    # ========================================================

    print_axis_summary(
        axis_name="X",

        current_state=x_current,

        goal_state=x_goal,

        result=x_result,
    )

    print_axis_summary(
        axis_name="Y",

        current_state=y_current,

        goal_state=y_goal,

        result=y_result,
    )

    print_preview_table(
        preview=preview,

        x_result=x_result,

        y_result=y_result,
    )

    return (
        preview,
        x_result,
        y_result,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "LIPM-MPC X/Y LONG-HORIZON TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"Initial DS : "
        f"{INITIAL_DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"Normal DS  : "
        f"{DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"SS         : "
        f"{SINGLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"MPC timestep: "
        f"{MPC_TIMESTEP:.3f} s"
    )

    print(
        f"MPC steps   : "
        f"{MPC_HORIZON_STEPS}"
    )

    print(
        f"MPC horizon : "
        f"{MPC_HORIZON_STEPS * MPC_TIMESTEP:.3f} s"
    )

    print()

    test_xy_mpc_with_support_preview()

    print()

    print(
        "=" * 60
    )

    print(
        "LONG-HORIZON X/Y MPC TEST PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()