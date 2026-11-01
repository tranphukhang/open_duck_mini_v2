import numpy as np


from foothold_generator import (
    HOME_NOMINAL_STEP_WIDTH,
    get_kinematic_foothold_bounds,
    get_opposite_foot,
    generate_nominal_foothold,
    apply_foothold_correction,
    clip_foothold_to_kinematic_workspace,
    generate_foothold
)


# ============================================================
# TEST PARAMETERS
# ============================================================

ATOL = 1e-12


W_P_L_HOME = np.array(
    [
        -0.016060,
        +0.054817,
        0.000037
    ],
    dtype=float
)


W_P_R_HOME = np.array(
    [
        -0.016060,
        -0.054817,
        0.000037
    ],
    dtype=float
)


# ============================================================
# PRINT HELPERS
# ============================================================

def print_test_header(
    number,
    title
):
    print("\n")
    print("=" * 70)
    print(
        f"TEST {number}: {title}"
    )
    print("=" * 70)


def print_pass(
    name
):
    print(
        f"{name}: PASS"
    )


def assert_equal(
    actual,
    expected,
    name
):
    if actual != expected:

        raise AssertionError(
            f"{name}: FAILED\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )


    print_pass(
        name
    )


def assert_close(
    actual,
    expected,
    name
):
    if not np.isclose(
        actual,
        expected,
        atol=ATOL,
        rtol=1e-12
    ):

        raise AssertionError(
            f"{name}: FAILED\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )


    print_pass(
        name
    )


def assert_array_close(
    actual,
    expected,
    name
):
    if not np.allclose(
        actual,
        expected,
        atol=ATOL,
        rtol=1e-12
    ):

        raise AssertionError(
            f"{name}: FAILED\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )


    print_pass(
        name
    )


# ============================================================
# TEST 1
#
# MEASURED KINEMATIC BOUNDS
# ============================================================

def test_kinematic_bounds():

    print_test_header(
        1,
        "Measured kinematic foothold bounds"
    )


    left = (
        get_kinematic_foothold_bounds(
            "L"
        )
    )


    right = (
        get_kinematic_foothold_bounds(
            "R"
        )
    )


    assert_close(
        left["dx_min"],
        -0.055,
        "Left dx_min"
    )


    assert_close(
        left["dx_max"],
        +0.065,
        "Left dx_max"
    )


    assert_close(
        left["dy_min"],
        -0.025,
        "Left dy_min"
    )


    assert_close(
        left["dy_max"],
        +0.045,
        "Left dy_max"
    )


    assert_close(
        right["dx_min"],
        -0.055,
        "Right dx_min"
    )


    assert_close(
        right["dx_max"],
        +0.065,
        "Right dx_max"
    )


    assert_close(
        right["dy_min"],
        -0.045,
        "Right dy_min"
    )


    assert_close(
        right["dy_max"],
        +0.025,
        "Right dy_max"
    )


    print(
        "\nTEST 1 RESULT: PASS"
    )


# ============================================================
# TEST 2
#
# LEFT / RIGHT FOOT LOGIC
# ============================================================

def test_opposite_foot():

    print_test_header(
        2,
        "Opposite-foot logic"
    )


    assert_equal(
        get_opposite_foot(
            "L"
        ),
        "R",
        "Opposite of L"
    )


    assert_equal(
        get_opposite_foot(
            "R"
        ),
        "L",
        "Opposite of R"
    )


    print(
        "\nTEST 2 RESULT: PASS"
    )


# ============================================================
# TEST 3
#
# HOME STEP WIDTH
#
# Starting from right support, zero step length should place
# left foot at the home lateral position.
# ============================================================

def test_home_width_left_nominal():

    print_test_header(
        3,
        "Nominal left foothold reproduces home width"
    )


    W_p_nominal = (
        generate_nominal_foothold(

            W_p_support=W_P_R_HOME,

            swing_foot="L",

            step_length=0.0,

            step_width=(
                HOME_NOMINAL_STEP_WIDTH
            ),

            ground_height=(
                W_P_L_HOME[2]
            )
        )
    )


    assert_array_close(
        W_p_nominal,
        W_P_L_HOME,
        "Right support -> left home foothold"
    )


    print(
        "\nTEST 3 RESULT: PASS"
    )


# ============================================================
# TEST 4
#
# RIGHT HOME REPRODUCTION
# ============================================================

def test_home_width_right_nominal():

    print_test_header(
        4,
        "Nominal right foothold reproduces home width"
    )


    W_p_nominal = (
        generate_nominal_foothold(

            W_p_support=W_P_L_HOME,

            swing_foot="R",

            step_length=0.0,

            step_width=(
                HOME_NOMINAL_STEP_WIDTH
            ),

            ground_height=(
                W_P_R_HOME[2]
            )
        )
    )


    assert_array_close(
        W_p_nominal,
        W_P_R_HOME,
        "Left support -> right home foothold"
    )


    print(
        "\nTEST 4 RESULT: PASS"
    )


# ============================================================
# TEST 5
#
# FORWARD NOMINAL STEP
# ============================================================

def test_forward_nominal_step():

    print_test_header(
        5,
        "Forward nominal foothold"
    )


    W_p_nominal = (
        generate_nominal_foothold(

            W_p_support=W_P_R_HOME,

            swing_foot="L",

            step_length=0.05,

            step_width=(
                HOME_NOMINAL_STEP_WIDTH
            ),

            ground_height=0.0
        )
    )


    expected = np.array(
        [
            W_P_R_HOME[0] + 0.05,
            W_P_L_HOME[1],
            0.0
        ]
    )


    assert_array_close(
        W_p_nominal,
        expected,
        "Forward 0.05 m left foothold"
    )


    print(
        "\nTEST 5 RESULT: PASS"
    )


# ============================================================
# TEST 6
#
# PLANNER CORRECTION
# ============================================================

def test_planner_correction():

    print_test_header(
        6,
        "Apply planner correction"
    )


    W_p_nominal = np.array(
        [
            0.0,
            0.05,
            0.0
        ]
    )


    W_p_candidate = (
        apply_foothold_correction(

            W_p_nominal=W_p_nominal,

            correction_xy=np.array(
                [
                    0.01,
                    -0.005
                ]
            )
        )
    )


    assert_array_close(
        W_p_candidate,
        np.array(
            [
                0.01,
                0.045,
                0.0
            ]
        ),
        "Planner XY correction"
    )


    print(
        "\nTEST 6 RESULT: PASS"
    )


# ============================================================
# TEST 7
#
# INSIDE SAFE WORKSPACE -> NO CLIPPING
# ============================================================

def test_no_kinematic_clipping():

    print_test_header(
        7,
        "Candidate inside safe workspace"
    )


    candidate = (
        W_P_L_HOME
        +
        np.array(
            [
                0.04,
                0.02,
                0.0
            ]
        )
    )


    (
        plan,
        diagnostics

    ) = clip_foothold_to_kinematic_workspace(

        W_p_candidate=candidate,

        W_p_kinematic_reference=(
            W_P_L_HOME
        ),

        swing_foot="L",

        ground_height=(
            W_P_L_HOME[2]
        )
    )


    assert_array_close(
        plan,
        candidate,
        "Inside candidate unchanged"
    )


    assert_equal(
        diagnostics["clipped"],
        False,
        "No clipping flag"
    )


    print(
        "\nTEST 7 RESULT: PASS"
    )


# ============================================================
# TEST 8
#
# FORWARD LIMIT
# ============================================================

def test_forward_clipping():

    print_test_header(
        8,
        "Forward kinematic clipping"
    )


    candidate = (
        W_P_L_HOME
        +
        np.array(
            [
                0.10,
                0.0,
                0.0
            ]
        )
    )


    (
        plan,
        diagnostics

    ) = clip_foothold_to_kinematic_workspace(

        W_p_candidate=candidate,

        W_p_kinematic_reference=(
            W_P_L_HOME
        ),

        swing_foot="L",

        ground_height=(
            W_P_L_HOME[2]
        )
    )


    assert_close(
        plan[0]
        -
        W_P_L_HOME[0],
        0.065,
        "Forward displacement clipped to +0.065 m"
    )


    assert_equal(
        diagnostics["clipped_x"],
        True,
        "Forward clipping flag"
    )


    print(
        "\nTEST 8 RESULT: PASS"
    )


# ============================================================
# TEST 9
#
# LEFT LATERAL LIMIT
# ============================================================

def test_left_lateral_clipping():

    print_test_header(
        9,
        "Left-foot lateral clipping"
    )


    candidate = (
        W_P_L_HOME
        +
        np.array(
            [
                0.0,
                0.10,
                0.0
            ]
        )
    )


    (
        plan,
        diagnostics

    ) = clip_foothold_to_kinematic_workspace(

        W_p_candidate=candidate,

        W_p_kinematic_reference=(
            W_P_L_HOME
        ),

        swing_foot="L",

        ground_height=(
            W_P_L_HOME[2]
        )
    )


    assert_close(
        plan[1]
        -
        W_P_L_HOME[1],
        0.045,
        "Left outward displacement clipped to +0.045 m"
    )


    assert_equal(
        diagnostics["clipped_y"],
        True,
        "Left lateral clipping flag"
    )


    print(
        "\nTEST 9 RESULT: PASS"
    )


# ============================================================
# TEST 10
#
# RIGHT LATERAL MIRROR
# ============================================================

def test_right_lateral_clipping():

    print_test_header(
        10,
        "Right-foot mirrored lateral clipping"
    )


    candidate = (
        W_P_R_HOME
        +
        np.array(
            [
                0.0,
                -0.10,
                0.0
            ]
        )
    )


    (
        plan,
        diagnostics

    ) = clip_foothold_to_kinematic_workspace(

        W_p_candidate=candidate,

        W_p_kinematic_reference=(
            W_P_R_HOME
        ),

        swing_foot="R",

        ground_height=(
            W_P_R_HOME[2]
        )
    )


    assert_close(
        plan[1]
        -
        W_P_R_HOME[1],
        -0.045,
        "Right outward displacement clipped to -0.045 m"
    )


    assert_equal(
        diagnostics["clipped_y"],
        True,
        "Right lateral clipping flag"
    )


    print(
        "\nTEST 10 RESULT: PASS"
    )


# ============================================================
# TEST 11
#
# COMPLETE GENERATOR
# ============================================================

def test_complete_generator():

    print_test_header(
        11,
        "Complete foothold generation pipeline"
    )


    (
        W_p_plan,
        diagnostics

    ) = generate_foothold(

        W_p_support=W_P_R_HOME,

        W_p_kinematic_reference=(
            W_P_L_HOME
        ),

        swing_foot="L",

        step_length=0.05,

        step_width=(
            HOME_NOMINAL_STEP_WIDTH
        ),

        correction_xy=np.array(
            [
                0.01,
                0.01
            ]
        ),

        ground_height=(
            W_P_L_HOME[2]
        )
    )


    # Nominal:
    #
    # dx relative to left home = 0.05
    #
    # correction:
    #
    # +0.01
    #
    # total:
    #
    # dx = 0.06
    #
    # still inside +0.065 safe limit.


    expected = (
        W_P_L_HOME
        +
        np.array(
            [
                0.06,
                0.01,
                0.0
            ]
        )
    )


    assert_array_close(
        W_p_plan,
        expected,
        "Complete planned foothold"
    )


    assert_equal(
        diagnostics[
            "kinematic"
        ][
            "clipped"
        ],
        False,
        "Complete generator no clipping"
    )


    print(
        "\nW_p_nominal ="
    )

    print(
        diagnostics[
            "W_p_nominal"
        ]
    )


    print(
        "\nW_p_candidate ="
    )

    print(
        diagnostics[
            "W_p_candidate"
        ]
    )


    print(
        "\nW_p_plan ="
    )

    print(
        W_p_plan
    )


    print(
        "\nTEST 11 RESULT: PASS"
    )


# ============================================================
# TEST 12
#
# COMPLETE GENERATOR WITH CLIPPING
# ============================================================

def test_complete_generator_with_clipping():

    print_test_header(
        12,
        "Complete generator rejects unreachable request"
    )


    (
        W_p_plan,
        diagnostics

    ) = generate_foothold(

        W_p_support=W_P_R_HOME,

        W_p_kinematic_reference=(
            W_P_L_HOME
        ),

        swing_foot="L",

        step_length=0.10,

        step_width=(
            HOME_NOMINAL_STEP_WIDTH
        ),

        correction_xy=np.zeros(
            2
        ),

        ground_height=(
            W_P_L_HOME[2]
        )
    )


    assert_close(
        W_p_plan[0]
        -
        W_P_L_HOME[0],
        0.065,
        "Unreachable forward foothold clipped"
    )


    assert_equal(
        diagnostics[
            "kinematic"
        ][
            "clipped"
        ],
        True,
        "Clipping detected"
    )


    print(
        "\nRequested candidate ="
    )

    print(
        diagnostics[
            "W_p_candidate"
        ]
    )


    print(
        "\nBounded plan ="
    )

    print(
        W_p_plan
    )


    print(
        "\nTEST 12 RESULT: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("FOOTHOLD GENERATOR TESTS")
    print("=" * 70)


    test_kinematic_bounds()

    test_opposite_foot()

    test_home_width_left_nominal()

    test_home_width_right_nominal()

    test_forward_nominal_step()

    test_planner_correction()

    test_no_kinematic_clipping()

    test_forward_clipping()

    test_left_lateral_clipping()

    test_right_lateral_clipping()

    test_complete_generator()

    test_complete_generator_with_clipping()


    print("\n")
    print("=" * 70)
    print("ALL FOOTHOLD GENERATOR TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()