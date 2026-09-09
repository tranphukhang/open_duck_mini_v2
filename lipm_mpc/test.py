from __future__ import annotations

from types import SimpleNamespace

import numpy as np


# ============================================================
# IMPORTS
# ============================================================

if __package__:

    from .pinocchio_model import (
        LEFT_FOOT_FRAME,
        RIGHT_FOOT_FRAME,
    )

    from .differential_ik import (
        TRUNK_FRAME,
        solve_single_support_ik,
        solve_double_support_ik,
    )

else:

    from pinocchio_model import (
        LEFT_FOOT_FRAME,
        RIGHT_FOOT_FRAME,
    )

    from differential_ik import (
        TRUNK_FRAME,
        solve_single_support_ik,
        solve_double_support_ik,
    )


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=9,
    suppress=True,
)


# ============================================================
# TEST ROBOT
# ============================================================

class FakeRobot:
    """
    Synthetic 16-DoF robot used only to verify hierarchy.

    Coordinates:

        q0 ... q4:
            left-foot task

        q5 ... q7:
            CoM task

        q8 ... q12:
            extra directions available to swing foot

        q13:
            trunk pitch

        q14, q15:
            remaining nullspace

    The right-foot Jacobian contains q5...q7 as well as
    q8...q12.

    Therefore when CoM has higher priority, the swing-foot
    task must use its remaining directions without changing
    the previously satisfied CoM task.
    """

    def __init__(
        self,
    ):

        self.walking_velocity_indices = (
            np.arange(
                16,
                dtype=int,
            )
        )

        self.model = (
            SimpleNamespace(
                nv=16
            )
        )

        # ----------------------------------------------------
        # LEFT FOOT: rank 5
        # q0 ... q4
        # ----------------------------------------------------

        self.J_left = np.zeros(
            (
                5,
                16,
            ),
            dtype=float,
        )

        self.J_left[
            :,
            0:5,
        ] = np.eye(
            5
        )

        # ----------------------------------------------------
        # COM: rank 3
        # q5, q6, q7
        # ----------------------------------------------------

        self.J_com = np.zeros(
            (
                3,
                16,
            ),
            dtype=float,
        )

        self.J_com[
            0,
            5,
        ] = 1.0

        self.J_com[
            1,
            6,
        ] = 1.0

        self.J_com[
            2,
            7,
        ] = 1.0

        # ----------------------------------------------------
        # RIGHT FOOT: rank 5
        #
        # First three rows depend partly on CoM coordinates,
        # but also have independent swing directions.
        # ----------------------------------------------------

        self.J_right = np.zeros(
            (
                5,
                16,
            ),
            dtype=float,
        )

        self.J_right[
            0,
            5,
        ] = 1.0

        self.J_right[
            0,
            8,
        ] = 1.0

        self.J_right[
            1,
            6,
        ] = 1.0

        self.J_right[
            1,
            9,
        ] = 1.0

        self.J_right[
            2,
            7,
        ] = 1.0

        self.J_right[
            2,
            10,
        ] = 1.0

        self.J_right[
            3,
            11,
        ] = 1.0

        self.J_right[
            4,
            12,
        ] = 1.0

        # ----------------------------------------------------
        # TRUNK LOCAL JACOBIAN
        # ----------------------------------------------------

        self.J_trunk = np.zeros(
            (
                6,
                16,
            ),
            dtype=float,
        )

        # local angular Y
        self.J_trunk[
            4,
            13,
        ] = 1.0


    # ========================================================
    # REQUIRED ROBOT API
    # ========================================================

    def update(
        self,
        q_pin,
    ):

        return None


    def get_left_foot_pose(
        self,
    ):

        return (
            np.zeros(
                3,
                dtype=float,
            ),

            np.eye(
                3,
                dtype=float,
            ),
        )


    def get_right_foot_pose(
        self,
    ):

        return (
            np.zeros(
                3,
                dtype=float,
            ),

            np.eye(
                3,
                dtype=float,
            ),
        )


    def get_com(
        self,
    ):

        return np.zeros(
            3,
            dtype=float,
        )


    def get_foot_task_jacobian(
        self,
        frame_name,
        active_only=True,
    ):

        if frame_name == (
            LEFT_FOOT_FRAME
        ):

            return (
                self.J_left.copy()
            )

        if frame_name == (
            RIGHT_FOOT_FRAME
        ):

            return (
                self.J_right.copy()
            )

        raise ValueError(
            f"Unknown foot frame: "
            f"{frame_name}"
        )


    def get_com_jacobian(
        self,
        active_only=True,
    ):

        return (
            self.J_com.copy()
        )


    def get_frame_pose(
        self,
        frame_name,
    ):

        if frame_name != (
            TRUNK_FRAME
        ):

            raise ValueError(
                f"Unknown frame: "
                f"{frame_name}"
            )

        return (
            np.zeros(
                3,
                dtype=float,
            ),

            np.eye(
                3,
                dtype=float,
            ),
        )


    def get_frame_jacobian_local(
        self,
        frame_name,
    ):

        if frame_name != (
            TRUNK_FRAME
        ):

            raise ValueError(
                f"Unknown frame: "
                f"{frame_name}"
            )

        return (
            self.J_trunk.copy()
        )


# ============================================================
# HELPERS
# ============================================================

def separator():

    print(
        "=" * 76
    )


def print_diagnostics(
    diagnostics,
):

    print()

    print(
        f"{'task':<24}"
        f"{'rank':>8}"
        f"{'residual':>16}"
        f"{'nullity':>12}"
    )

    print(
        "-" * 60
    )

    for item in (
        diagnostics
    ):

        if item["name"] == (
            "tracking_errors"
        ):

            continue

        print(
            f"{item['name']:<24}"
            f"{item['reduced_rank']:>8d}"
            f"{item['residual_after_norm']:>16.3e}"
            f"{item['remaining_nullity']:>12d}"
        )


# ============================================================
# SINGLE SUPPORT TEST
# ============================================================

def test_single_support_priority():

    robot = (
        FakeRobot()
    )

    q_pin = np.zeros(
        16,
        dtype=float,
    )

    # ========================================================
    # REFERENCES
    # ========================================================

    support_position_ref = np.zeros(
        3,
        dtype=float,
    )

    swing_position_ref = np.array(
        [
            +0.010,
            -0.020,
            +0.030,
        ],
        dtype=float,
    )

    swing_velocity_ref = np.array(
        [
            +0.020,
            -0.010,
            +0.015,
        ],
        dtype=float,
    )

    com_position_ref = np.array(
        [
            +0.020,
            -0.010,
            +0.030,
        ],
        dtype=float,
    )

    com_velocity_ref = np.array(
        [
            +0.100,
            +0.200,
            -0.100,
        ],
        dtype=float,
    )

    trunk_rotation_ref = np.eye(
        3,
        dtype=float,
    )

    SUPPORT_GAIN = 25.0

    SWING_GAIN = 20.0

    COM_GAIN = 10.0

    # ========================================================
    # SOLVE
    # ========================================================

    (
        qdot,
        diagnostics,
        Z,
    ) = solve_single_support_ik(
        robot=(
            robot
        ),

        q_pin=(
            q_pin
        ),

        support_side=(
            "left"
        ),

        support_position_ref=(
            support_position_ref
        ),

        swing_position_ref=(
            swing_position_ref
        ),

        swing_linear_velocity_ref=(
            swing_velocity_ref
        ),

        com_position_ref=(
            com_position_ref
        ),

        com_velocity_ref=(
            com_velocity_ref
        ),

        trunk_rotation_ref=(
            trunk_rotation_ref
        ),

        support_position_gain=(
            SUPPORT_GAIN
        ),

        swing_position_gain=(
            SWING_GAIN
        ),

        com_position_gain=(
            COM_GAIN
        ),

        trunk_orientation_gain=(
            10.0
        ),

        damping=(
            1e-10
        ),

        rcond=(
            1e-12
        ),
    )

    # ========================================================
    # EXPECTED TASK ORDER
    # ========================================================

    task_names = [
        item["name"]
        for item in diagnostics[:-1]
    ]

    expected_names = [
        "left_support",
        "com",
        "right_swing",
        "trunk_pitch",
    ]

    assert (
        task_names
        ==
        expected_names
    )

    print(
        "[PASS] single-support hierarchy order"
    )

    # ========================================================
    # EXPECTED NULLSPACE SEQUENCE
    # ========================================================

    expected_nullities = [
        11,
        8,
        3,
        2,
    ]

    actual_nullities = [
        item["remaining_nullity"]
        for item in diagnostics[:-1]
    ]

    assert (
        actual_nullities
        ==
        expected_nullities
    )

    assert (
        Z.shape
        ==
        (
            16,
            2,
        )
    )

    print(
        "[PASS] single-support final nullity = 2"
    )

    # ========================================================
    # EXPECTED COMMANDS
    # ========================================================

    expected_support_cmd = np.zeros(
        5,
        dtype=float,
    )

    expected_com_cmd = (
        com_velocity_ref
        +
        COM_GAIN
        *
        com_position_ref
    )

    expected_swing_linear = (
        swing_velocity_ref
        +
        SWING_GAIN
        *
        swing_position_ref
    )

    expected_swing_cmd = np.concatenate(
        [
            expected_swing_linear,

            np.zeros(
                2,
                dtype=float,
            ),
        ]
    )

    # ========================================================
    # ACTUAL FINAL TASK VELOCITIES
    # ========================================================

    support_velocity = (
        robot.J_left
        @
        qdot
    )

    com_velocity = (
        robot.J_com
        @
        qdot
    )

    swing_velocity = (
        robot.J_right
        @
        qdot
    )

    # ========================================================
    # HIGH-PRIORITY SUPPORT
    # ========================================================

    np.testing.assert_allclose(
        support_velocity,
        expected_support_cmd,
        atol=1e-8,
    )

    print(
        "[PASS] swing/CoM cannot corrupt support task"
    )

    # ========================================================
    # SECOND-PRIORITY COM
    # ========================================================

    np.testing.assert_allclose(
        com_velocity,
        expected_com_cmd,
        atol=1e-8,
    )

    print(
        "[PASS] swing task cannot corrupt CoM task"
    )

    # ========================================================
    # SWING STILL TRACKABLE
    # ========================================================

    np.testing.assert_allclose(
        swing_velocity,
        expected_swing_cmd,
        atol=1e-8,
    )

    print(
        "[PASS] swing task remains achievable"
    )

    # ========================================================
    # FINITE SOLUTION
    # ========================================================

    assert np.all(
        np.isfinite(
            qdot
        )
    )

    print(
        "[PASS] single-support qdot finite"
    )

    print_diagnostics(
        diagnostics
    )


# ============================================================
# DOUBLE SUPPORT TEST
# ============================================================

def test_double_support_priority():

    robot = (
        FakeRobot()
    )

    q_pin = np.zeros(
        16,
        dtype=float,
    )

    left_ref = np.zeros(
        3,
        dtype=float,
    )

    right_ref = np.zeros(
        3,
        dtype=float,
    )

    com_ref = np.array(
        [
            +0.020,
            -0.010,
            +0.030,
        ],
        dtype=float,
    )

    com_velocity_ref = np.array(
        [
            +0.050,
            +0.020,
            -0.010,
        ],
        dtype=float,
    )

    trunk_ref = np.eye(
        3,
        dtype=float,
    )

    COM_GAIN = 10.0

    (
        qdot,
        diagnostics,
        Z,
    ) = solve_double_support_ik(
        robot=(
            robot
        ),

        q_pin=(
            q_pin
        ),

        left_position_ref=(
            left_ref
        ),

        right_position_ref=(
            right_ref
        ),

        com_position_ref=(
            com_ref
        ),

        com_velocity_ref=(
            com_velocity_ref
        ),

        trunk_rotation_ref=(
            trunk_ref
        ),

        foot_position_gain=(
            25.0
        ),

        com_position_gain=(
            COM_GAIN
        ),

        trunk_orientation_gain=(
            10.0
        ),

        damping=(
            1e-10
        ),

        rcond=(
            1e-12
        ),
    )

    # ========================================================
    # ORDER
    # ========================================================

    task_names = [
        item["name"]
        for item in diagnostics[:-1]
    ]

    expected_names = [
        "double_support_feet",
        "com",
        "trunk_pitch",
    ]

    assert (
        task_names
        ==
        expected_names
    )

    print()

    print(
        "[PASS] double-support hierarchy unchanged"
    )

    # ========================================================
    # FINAL NULLITY
    # ========================================================

    assert (
        Z.shape
        ==
        (
            16,
            2,
        )
    )

    print(
        "[PASS] double-support final nullity = 2"
    )

    # ========================================================
    # BOTH FEET REMAIN FIXED
    # ========================================================

    both_feet_velocity = (
        np.vstack(
            [
                robot.J_left,
                robot.J_right,
            ]
        )
        @
        qdot
    )

    np.testing.assert_allclose(
        both_feet_velocity,
        0.0,
        atol=1e-8,
    )

    print(
        "[PASS] CoM cannot corrupt double-support feet"
    )

    # ========================================================
    # COM
    # ========================================================

    expected_com_cmd = (
        com_velocity_ref
        +
        COM_GAIN
        *
        com_ref
    )

    actual_com_cmd = (
        robot.J_com
        @
        qdot
    )

    np.testing.assert_allclose(
        actual_com_cmd,
        expected_com_cmd,
        atol=1e-8,
    )

    print(
        "[PASS] double-support CoM task achievable"
    )

    print_diagnostics(
        diagnostics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    separator()

    print(
        "DIFFERENTIAL IK PRIORITY TEST"
    )

    separator()

    print()

    print(
        "Single support target hierarchy:"
    )

    print(
        "  P1 support"
    )

    print(
        "  P2 CoM"
    )

    print(
        "  P3 swing"
    )

    print(
        "  P4 trunk"
    )

    print()

    test_single_support_priority()

    print()

    separator()

    test_double_support_priority()

    print()

    separator()

    print(
        "DIFFERENTIAL IK PRIORITY TEST PASSED"
    )

    separator()


if __name__ == "__main__":

    main()