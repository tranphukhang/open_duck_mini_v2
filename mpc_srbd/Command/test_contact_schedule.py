import numpy as np

from contact_schedule import (
    N_DS,
    N_SS,
    N_CYCLE,
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
    get_contact_schedule,
    get_current_contact_info
)


# ============================================================
# TEST PARAMETERS
# ============================================================

DT_MPC = 0.025

N = 10


# ============================================================
# TEST 1
#
# GAIT CONSTANTS
# ============================================================

def test_gait_constants():

    assert N_DS == 2

    assert N_SS == 8

    assert N_CYCLE == (
        2 * N_DS
        +
        2 * N_SS
    )


    T_DS = (
        N_DS
        *
        DT_MPC
    )


    T_SS = (
        N_SS
        *
        DT_MPC
    )


    T_CYCLE = (
        N_CYCLE
        *
        DT_MPC
    )


    assert np.isclose(
        T_DS,
        0.05
    )


    assert np.isclose(
        T_SS,
        0.20
    )


    assert np.isclose(
        T_CYCLE,
        0.50
    )


    print(
        "TEST 1 PASSED: Gait Constants"
    )


# ============================================================
# TEST 2
#
# CONTACT STATE CONSTANTS
# ============================================================

def test_contact_state_constants():

    assert np.array_equal(
        DOUBLE_SUPPORT,
        np.array(
            [1, 1]
        )
    )


    assert np.array_equal(
        LEFT_SINGLE_SUPPORT,
        np.array(
            [1, 0]
        )
    )


    assert np.array_equal(
        RIGHT_SINGLE_SUPPORT,
        np.array(
            [0, 1]
        )
    )


    print(
        "TEST 2 PASSED: Contact State Constants"
    )


# ============================================================
# TEST 3
#
# FIRST DOUBLE SUPPORT
#
# phase:
#
#     [0, 2)
#
# time:
#
#     [0, 0.05)
# ============================================================

def test_first_double_support():

    schedule = get_contact_schedule(
        current_time=0.0,
        N=1,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        schedule[0],
        DOUBLE_SUPPORT
    )


    print(
        "TEST 3 PASSED: First Double Support"
    )


# ============================================================
# TEST 4
#
# LEFT SINGLE SUPPORT START
#
# t = N_DS * dt_mpc
#
#   = 2 * 0.025
#
#   = 0.05 s
#
# Expected:
#
#     [1, 0]
#
# Right foot swing.
# ============================================================

def test_left_single_support_start():

    t = (
        N_DS
        *
        DT_MPC
    )


    schedule = get_contact_schedule(
        current_time=t,
        N=1,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        schedule[0],
        LEFT_SINGLE_SUPPORT
    )


    print(
        "TEST 4 PASSED: Left Single Support Start"
    )


# ============================================================
# TEST 5
#
# SECOND DOUBLE SUPPORT START
#
# t =
#
#     (N_DS + N_SS) * dt_mpc
#
#   =
#
#     10 * 0.025
#
#   =
#
#     0.25 s
# ============================================================

def test_second_double_support_start():

    t = (
        N_DS
        +
        N_SS
    ) * DT_MPC


    schedule = get_contact_schedule(
        current_time=t,
        N=1,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        schedule[0],
        DOUBLE_SUPPORT
    )


    print(
        "TEST 5 PASSED: Second Double Support Start"
    )


# ============================================================
# TEST 6
#
# RIGHT SINGLE SUPPORT START
#
# t =
#
#     (2*N_DS + N_SS) * dt_mpc
#
#   =
#
#     12 * 0.025
#
#   =
#
#     0.30 s
#
# Expected:
#
#     [0, 1]
#
# Left foot swing.
# ============================================================

def test_right_single_support_start():

    t = (
        2 * N_DS
        +
        N_SS
    ) * DT_MPC


    schedule = get_contact_schedule(
        current_time=t,
        N=1,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        schedule[0],
        RIGHT_SINGLE_SUPPORT
    )


    print(
        "TEST 6 PASSED: Right Single Support Start"
    )


# ============================================================
# TEST 7
#
# COMPLETE CYCLE REPEAT
#
# t =
#
#     N_CYCLE * dt_mpc
#
#   =
#
#     20 * 0.025
#
#   =
#
#     0.50 s
#
# Expected:
#
#     cycle restarts at Double Support
# ============================================================

def test_cycle_repeat():

    t = (
        N_CYCLE
        *
        DT_MPC
    )


    schedule = get_contact_schedule(
        current_time=t,
        N=1,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        schedule[0],
        DOUBLE_SUPPORT
    )


    print(
        "TEST 7 PASSED: Complete Cycle Repeat"
    )


# ============================================================
# TEST 8
#
# HORIZON SHAPE
# ============================================================

def test_schedule_shape():

    schedule = get_contact_schedule(
        current_time=0.0,
        N=N,
        dt_mpc=DT_MPC
    )


    assert schedule.shape == (
        N,
        2
    )


    assert np.all(
        np.isin(
            schedule,
            [0, 1]
        )
    )


    print(
        "TEST 8 PASSED: Schedule Shape"
    )


# ============================================================
# TEST 9
#
# CURRENT CONTACT INFO - DOUBLE SUPPORT
# ============================================================

def test_current_contact_double_support():

    (
        current_contact,
        swing_foot,
        delta_T

    ) = get_current_contact_info(
        current_time=0.0,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        current_contact,
        DOUBLE_SUPPORT
    )


    assert swing_foot is None

    assert delta_T is None


    print(
        "TEST 9 PASSED: Current Contact Double Support"
    )


# ============================================================
# TEST 10
#
# CURRENT CONTACT INFO - LSS
#
# At beginning of LSS:
#
#     delta_T = N_SS * dt_mpc
#
#             = 0.20 s
# ============================================================

def test_current_contact_LSS():

    t = (
        N_DS
        *
        DT_MPC
    )


    (
        current_contact,
        swing_foot,
        delta_T

    ) = get_current_contact_info(
        current_time=t,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        current_contact,
        LEFT_SINGLE_SUPPORT
    )


    assert swing_foot == "R"


    expected_delta_T = (
        N_SS
        *
        DT_MPC
    )


    assert np.isclose(
        delta_T,
        expected_delta_T,
        atol=1e-12
    )


    print(
        "TEST 10 PASSED: Current Contact LSS"
    )


# ============================================================
# TEST 11
#
# delta_T MUST DECREASE DURING LSS
# ============================================================

def test_delta_T_decreases_LSS():

    t_start = (
        N_DS
        *
        DT_MPC
    )


    t_middle = (
        t_start
        +
        0.075
    )


    (
        _,
        swing_foot_start,
        delta_T_start

    ) = get_current_contact_info(
        current_time=t_start,
        dt_mpc=DT_MPC
    )


    (
        _,
        swing_foot_middle,
        delta_T_middle

    ) = get_current_contact_info(
        current_time=t_middle,
        dt_mpc=DT_MPC
    )


    assert swing_foot_start == "R"

    assert swing_foot_middle == "R"


    assert delta_T_middle < delta_T_start


    expected_delta_T_middle = (
        N_SS
        *
        DT_MPC
        -
        0.075
    )


    assert np.isclose(
        delta_T_middle,
        expected_delta_T_middle,
        atol=1e-12
    )


    print(
        "TEST 11 PASSED: delta_T Decreases During LSS"
    )


# ============================================================
# TEST 12
#
# CURRENT CONTACT INFO - RSS
#
# At beginning of RSS:
#
#     delta_T = N_SS * dt_mpc
#
#             = 0.20 s
# ============================================================

def test_current_contact_RSS():

    t = (
        2 * N_DS
        +
        N_SS
    ) * DT_MPC


    (
        current_contact,
        swing_foot,
        delta_T

    ) = get_current_contact_info(
        current_time=t,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        current_contact,
        RIGHT_SINGLE_SUPPORT
    )


    assert swing_foot == "L"


    expected_delta_T = (
        N_SS
        *
        DT_MPC
    )


    assert np.isclose(
        delta_T,
        expected_delta_T,
        atol=1e-12
    )


    print(
        "TEST 12 PASSED: Current Contact RSS"
    )


# ============================================================
# TEST 13
#
# FLOATING-POINT BOUNDARY
#
# Ensure values extremely close to a phase boundary
# are handled correctly.
# ============================================================

def test_floating_point_boundary():

    t = (
        (
            2 * N_DS
            +
            N_SS
        )
        *
        DT_MPC
        -
        1e-15
    )


    schedule = get_contact_schedule(
        current_time=t,
        N=1,
        dt_mpc=DT_MPC
    )


    assert np.array_equal(
        schedule[0],
        RIGHT_SINGLE_SUPPORT
    )


    print(
        "TEST 13 PASSED: Floating-Point Boundary"
    )


# ============================================================
# TEST 14
#
# INVALID INPUTS
# ============================================================

def test_invalid_inputs():

    # --------------------------------------------------------
    # Negative time
    # --------------------------------------------------------

    try:

        get_contact_schedule(
            current_time=-0.1,
            N=1,
            dt_mpc=DT_MPC
        )


        raise AssertionError(
            "Negative current_time phải gây lỗi."
        )


    except ValueError:

        pass


    # --------------------------------------------------------
    # Invalid horizon
    # --------------------------------------------------------

    try:

        get_contact_schedule(
            current_time=0.0,
            N=0,
            dt_mpc=DT_MPC
        )


        raise AssertionError(
            "N = 0 phải gây lỗi."
        )


    except ValueError:

        pass


    # --------------------------------------------------------
    # Invalid dt_mpc
    # --------------------------------------------------------

    try:

        get_contact_schedule(
            current_time=0.0,
            N=1,
            dt_mpc=0.0
        )


        raise AssertionError(
            "dt_mpc = 0 phải gây lỗi."
        )


    except ValueError:

        pass


    print(
        "TEST 14 PASSED: Invalid Inputs"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n============================================"
    )

    print(
        " TEST CONTACT SCHEDULE"
    )

    print(
        "============================================\n"
    )


    print(
        f"N_DS = {N_DS}"
    )


    print(
        f"N_SS = {N_SS}"
    )


    print(
        f"N_CYCLE = {N_CYCLE}"
    )


    print(
        f"dt_mpc = {DT_MPC:.6f} s"
    )


    print(
        f"T_DS = {N_DS * DT_MPC:.6f} s"
    )


    print(
        f"T_SS = {N_SS * DT_MPC:.6f} s"
    )


    print(
        f"T_CYCLE = {N_CYCLE * DT_MPC:.6f} s\n"
    )


    # ========================================================
    # RUN TESTS
    # ========================================================

    test_gait_constants()

    test_contact_state_constants()

    test_first_double_support()

    test_left_single_support_start()

    test_second_double_support_start()

    test_right_single_support_start()

    test_cycle_repeat()

    test_schedule_shape()

    test_current_contact_double_support()

    test_current_contact_LSS()

    test_delta_T_decreases_LSS()

    test_current_contact_RSS()

    test_floating_point_boundary()

    test_invalid_inputs()


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n--------------------------------------------"
    )

    print(
        "ALL TESTS PASSED"
    )

    print(
        "--------------------------------------------\n"
    )


if __name__ == "__main__":

    main()