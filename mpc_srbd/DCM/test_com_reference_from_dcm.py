from pathlib import Path
import sys

import numpy as np


# ============================================================
# PATH
# ============================================================

DCM_DIR = (
    Path(
        __file__
    ).resolve().parent
)


MPC_SRBD_DIR = (
    DCM_DIR.parent
)


if str(
    MPC_SRBD_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            MPC_SRBD_DIR
        )
    )


if str(
    DCM_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            DCM_DIR
        )
    )


# ============================================================
# MODULES
# ============================================================

from Command.gait_timing import (
    build_gait_timing
)


from dcm_model import (
    compute_omega0
)


from dcm_reference_horizon import (
    build_dcm_reference_horizon
)


from com_reference_from_dcm import (
    compute_com_velocity_xy,
    integrate_com_xy_heun_step,
    build_com_reference_from_dcm
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=10,
    suppress=True,
    linewidth=200
)


# ============================================================
# TEST HELPERS
# ============================================================

def print_header(
    number,
    title
):

    print("\n")
    print("=" * 70)

    print(
        f"TEST {number}: {title}"
    )

    print("=" * 70)


def assert_close(
    actual,
    expected,
    name,
    atol=1e-10
):

    actual = np.asarray(
        actual,
        dtype=float
    )


    expected = np.asarray(
        expected,
        dtype=float
    )


    if not np.allclose(
        actual,
        expected,
        atol=atol,
        rtol=0.0
    ):

        raise AssertionError(
            f"{name}: FAILED\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )


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


    print(
        f"{name}: PASS"
    )


def assert_true(
    condition,
    name
):

    if not condition:

        raise AssertionError(
            f"{name}: FAILED"
        )


    print(
        f"{name}: PASS"
    )


def assert_raises(
    function,
    name
):

    try:

        function()

    except (
        ValueError,
        TypeError,
        RuntimeError
    ):

        print(
            f"{name}: PASS"
        )

        return


    raise AssertionError(
        f"{name}: FAILED - exception was not raised."
    )


# ============================================================
# COMMON NOMINAL WALKING CASE
# ============================================================

def build_test_case():

    timing = build_gait_timing(

        T_step=0.5,

        horizon=10,

        N_SS=8,

        N_DS=2
    )


    omega0 = compute_omega0(
        0.25
    )


    foot_y = 0.054817

    step_length = 0.04


    support_points = np.array(
        [
            [
                0.00,
                +foot_y
            ],

            [
                0.04,
                -foot_y
            ],

            [
                0.08,
                +foot_y
            ],

            [
                0.12,
                -foot_y
            ],

            [
                0.16,
                +foot_y
            ],
        ],
        dtype=float
    )


    dcm = build_dcm_reference_horizon(

        current_phase="LSS",

        t_phase=0.0,

        timing=timing,

        support_points=(
            support_points
        ),

        omega0=omega0
    )


    # --------------------------------------------------------
    # NOMINAL CoM state at the beginning of this test.
    #
    # This is deliberately a nominal state.
    #
    # It is NOT measured from MuJoCo.
    #
    # Startup / Initial Transfer will later initialize this
    # state in the higher-level walking integration.
    # --------------------------------------------------------

    W_p_CoM_nominal_current = np.array(
        [
            0.0,
            0.0,
            0.25
        ],
        dtype=float
    )


    return (
        timing,
        omega0,
        support_points,
        foot_y,
        dcm,
        W_p_CoM_nominal_current
    )


# ============================================================
# TEST 1
#
# DCM -> CoM VELOCITY
# ============================================================

def test_com_velocity_relation():

    print_header(
        1,
        "Nominal CoM velocity from DCM"
    )


    W_p_CoM_xy = np.array(
        [
            0.10,
            -0.02
        ],
        dtype=float
    )


    W_xi_xy = np.array(
        [
            0.14,
            0.01
        ],
        dtype=float
    )


    omega0 = 5.0


    expected = (

        omega0

        *

        (
            W_xi_xy
            -
            W_p_CoM_xy
        )
    )


    actual = compute_com_velocity_xy(

        W_p_CoM_xy=(
            W_p_CoM_xy
        ),

        W_xi_xy=(
            W_xi_xy
        ),

        omega0=omega0
    )


    print(
        "\nW_v_CoM_xy =",
        actual
    )


    assert_close(
        actual,
        expected,
        "v = omega0*(xi-p)"
    )


# ============================================================
# TEST 2
#
# ONE HEUN STEP
# ============================================================

def test_one_heun_step():

    print_header(
        2,
        "One nominal CoM Heun integration step"
    )


    W_p_start = np.array(
        [
            0.02,
            -0.01
        ],
        dtype=float
    )


    W_xi_start = np.array(
        [
            0.04,
            0.01
        ],
        dtype=float
    )


    W_xi_end = np.array(
        [
            0.045,
            0.005
        ],
        dtype=float
    )


    omega0 = 5.0

    dt = 0.05


    # --------------------------------------------------------
    # Manual Heun calculation
    # --------------------------------------------------------

    v_start = (

        omega0

        *

        (
            W_xi_start
            -
            W_p_start
        )
    )


    p_predict = (

        W_p_start

        +

        dt
        *
        v_start
    )


    v_end_predict = (

        omega0

        *

        (
            W_xi_end
            -
            p_predict
        )
    )


    p_expected = (

        W_p_start

        +

        0.5
        *
        dt
        *
        (
            v_start
            +
            v_end_predict
        )
    )


    v_expected = (

        omega0

        *

        (
            W_xi_end
            -
            p_expected
        )
    )


    (
        p_actual,
        v_actual

    ) = integrate_com_xy_heun_step(

        W_p_CoM_start_xy=(
            W_p_start
        ),

        W_xi_start_xy=(
            W_xi_start
        ),

        W_xi_end_xy=(
            W_xi_end
        ),

        omega0=omega0,

        dt=dt
    )


    assert_close(
        p_actual,
        p_expected,
        "Heun nominal position"
    )


    assert_close(
        v_actual,
        v_expected,
        "Heun nominal velocity"
    )


# ============================================================
# TEST 3
#
# COMPLETE REFERENCE SHAPES
# ============================================================

def test_reference_shapes():

    print_header(
        3,
        "Nominal CoM reference shapes"
    )


    (
        timing,
        omega0,
        _,
        _,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    print(
        "\nW_p_CoM_ref =\n",
        result[
            "W_p_CoM_ref"
        ]
    )


    print(
        "\nW_v_CoM_ref =\n",
        result[
            "W_v_CoM_ref"
        ]
    )


    assert_equal(
        result[
            "W_p_CoM_ref"
        ].shape,
        (
            10,
            3
        ),
        "W_p_CoM_ref shape"
    )


    assert_equal(
        result[
            "W_v_CoM_ref"
        ].shape,
        (
            10,
            3
        ),
        "W_v_CoM_ref shape"
    )


    assert_equal(
        result[
            "W_p_CoM_sequence"
        ].shape,
        (
            11,
            3
        ),
        "Full nominal CoM sequence shape"
    )


    assert_equal(
        result[
            "W_xi_sequence"
        ].shape,
        (
            11,
            2
        ),
        "Full nominal DCM sequence shape"
    )


# ============================================================
# TEST 4
#
# CURRENT NOMINAL STATE IS PRESERVED
# ============================================================

def test_current_nominal_state():

    print_header(
        4,
        "Current nominal CoM state is preserved"
    )


    (
        timing,
        omega0,
        _,
        _,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    assert_close(
        result[
            "W_p_CoM_sequence"
        ][
            0
        ],
        W_p_CoM_nominal_current,
        "Nominal current CoM position preserved"
    )


    expected_velocity_xy = (

        omega0

        *

        (
            dcm[
                "W_xi_nominal_current_xy"
            ]

            -

            W_p_CoM_nominal_current[
                0:2
            ]
        )
    )


    assert_close(
        result[
            "W_v_CoM_sequence"
        ][
            0,
            0:2
        ],
        expected_velocity_xy,
        "Nominal current CoM velocity"
    )


# ============================================================
# TEST 5
#
# FLAT-GROUND CONSTANT HEIGHT
# ============================================================

def test_constant_height():

    print_header(
        5,
        "Flat-ground constant nominal CoM height"
    )


    (
        timing,
        omega0,
        _,
        _,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    assert_close(
        result[
            "W_p_CoM_sequence"
        ][
            :,
            2
        ],
        np.full(
            11,
            0.25
        ),
        "z_CoM nominal remains constant"
    )


    assert_close(
        result[
            "W_v_CoM_sequence"
        ][
            :,
            2
        ],
        np.zeros(
            11
        ),
        "Nominal v_z remains zero"
    )


# ============================================================
# TEST 6
#
# DCM-CoM CONSISTENCY
# ============================================================

def test_dcm_com_consistency():

    print_header(
        6,
        "Nominal DCM-CoM consistency over horizon"
    )


    (
        timing,
        omega0,
        _,
        _,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    expected_velocity_xy = (

        omega0

        *

        (
            result[
                "W_xi_sequence"
            ]

            -

            result[
                "W_p_CoM_sequence"
            ][
                :,
                0:2
            ]
        )
    )


    actual_velocity_xy = (
        result[
            "W_v_CoM_sequence"
        ][
            :,
            0:2
        ]
    )


    max_error = np.max(
        np.abs(
            expected_velocity_xy
            -
            actual_velocity_xy
        )
    )


    print(
        "\nMaximum consistency error =",
        max_error
    )


    assert_close(
        actual_velocity_xy,
        expected_velocity_xy,
        "v_nom = omega0*(xi_nom-p_nom)"
    )


# ============================================================
# TEST 7
#
# EQUILIBRIUM
# ============================================================

def test_equilibrium():

    print_header(
        7,
        "Nominal DCM-CoM equilibrium"
    )


    W_p_CoM_nominal_current = np.array(
        [
            0.10,
            -0.03,
            0.25
        ],
        dtype=float
    )


    W_xi_nominal_current_xy = (
        W_p_CoM_nominal_current[
            0:2
        ].copy()
    )


    N = 10


    W_xi_ref = np.tile(

        W_xi_nominal_current_xy,

        (
            N,
            1
        )
    )


    omega0 = compute_omega0(
        0.25
    )


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            W_xi_nominal_current_xy
        ),

        W_xi_ref=W_xi_ref,

        omega0=omega0,

        dt_mpc=0.05
    )


    expected_positions = np.tile(

        W_p_CoM_nominal_current,

        (
            N,
            1
        )
    )


    assert_close(
        result[
            "W_p_CoM_ref"
        ],
        expected_positions,
        "Nominal equilibrium position"
    )


    assert_close(
        result[
            "W_v_CoM_ref"
        ],
        np.zeros(
            (
                N,
                3
            )
        ),
        "Nominal equilibrium velocity"
    )


# ============================================================
# TEST 8
#
# RECEDING-HORIZON CONSISTENCY
#
# Generate:
#
#     k -> k+N
#
# then take nominal state at k+1 as new current state.
#
# Reintegrating the remaining DCM samples must recover:
#
#     original k+2 ... k+N
#
# This checks that the nominal trajectory can be propagated
# without resetting to the measured robot state.
# ============================================================

def test_receding_horizon_consistency():

    print_header(
        8,
        "Nominal receding-horizon consistency"
    )


    (
        timing,
        omega0,
        _,
        _,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    full = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    # --------------------------------------------------------
    # Shift current nominal state to k+1.
    # --------------------------------------------------------

    shifted = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            full[
                "W_p_CoM_ref"
            ][
                0
            ]
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_ref"
            ][
                0
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ][
                1:
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    assert_close(
        shifted[
            "W_p_CoM_ref"
        ],
        full[
            "W_p_CoM_ref"
        ][
            1:
        ],
        "Receding nominal CoM position consistency"
    )


    assert_close(
        shifted[
            "W_v_CoM_ref"
        ],
        full[
            "W_v_CoM_ref"
        ][
            1:
        ],
        "Receding nominal CoM velocity consistency"
    )


# ============================================================
# TEST 9
#
# OPEN DUCK LATERAL SANITY
# ============================================================

def test_open_duck_lateral_sanity():

    print_header(
        9,
        "Open Duck nominal CoM lateral sanity"
    )


    (
        timing,
        omega0,
        _,
        foot_y,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    max_abs_y = np.max(
        np.abs(
            result[
                "W_p_CoM_ref"
            ][
                :,
                1
            ]
        )
    )


    max_abs_vy = np.max(
        np.abs(
            result[
                "W_v_CoM_ref"
            ][
                :,
                1
            ]
        )
    )


    print(
        "\nmax |CoM_y| =",
        max_abs_y,
        "m"
    )


    print(
        "foot |y|    =",
        foot_y,
        "m"
    )


    print(
        "max |v_y|   =",
        max_abs_vy,
        "m/s"
    )


    print(
        "\nW_p_CoM_ref =\n",
        result[
            "W_p_CoM_ref"
        ]
    )


    assert_true(
        max_abs_y
        <
        foot_y,
        "Nominal CoM lateral motion remains inside foot-width scale"
    )


    assert_true(
        max_abs_y
        <
        0.10,
        "No nominal CoM lateral divergence"
    )


# ============================================================
# TEST 10
#
# FUTURE-ONLY MPC OUTPUT
# ============================================================

def test_future_only_output():

    print_header(
        10,
        "MPC CoM output contains k+1 ... k+N only"
    )


    (
        timing,
        omega0,
        _,
        _,
        dcm,
        W_p_CoM_nominal_current

    ) = build_test_case()


    result = build_com_reference_from_dcm(

        W_p_CoM_nominal_current=(
            W_p_CoM_nominal_current
        ),

        W_xi_nominal_current_xy=(
            dcm[
                "W_xi_nominal_current_xy"
            ]
        ),

        W_xi_ref=(
            dcm[
                "W_xi_ref"
            ]
        ),

        omega0=omega0,

        dt_mpc=(
            timing[
                "dt_mpc"
            ]
        )
    )


    assert_equal(
        result[
            "W_p_CoM_ref"
        ].shape[
            0
        ],
        timing[
            "horizon"
        ],
        "Future CoM horizon length"
    )


    assert_close(
        result[
            "W_p_CoM_sequence"
        ][
            0
        ],
        W_p_CoM_nominal_current,
        "Current nominal CoM kept only in diagnostic sequence"
    )


# ============================================================
# TEST 11
#
# INVALID INPUT
# ============================================================

def test_invalid_input():

    print_header(
        11,
        "Invalid input rejection"
    )


    assert_raises(

        lambda: build_com_reference_from_dcm(

            W_p_CoM_nominal_current=np.array(
                [
                    0.0,
                    0.0,
                    0.25
                ]
            ),

            W_xi_nominal_current_xy=np.zeros(
                2
            ),

            W_xi_ref=np.zeros(
                (
                    10,
                    3
                )
            ),

            omega0=5.0,

            dt_mpc=0.05
        ),

        "Reject wrong DCM reference dimension"
    )


    assert_raises(

        lambda: build_com_reference_from_dcm(

            W_p_CoM_nominal_current=np.array(
                [
                    0.0,
                    0.0,
                    0.25
                ]
            ),

            W_xi_nominal_current_xy=np.zeros(
                2
            ),

            W_xi_ref=np.zeros(
                (
                    10,
                    2
                )
            ),

            omega0=0.0,

            dt_mpc=0.05
        ),

        "Reject zero omega0"
    )


    assert_raises(

        lambda: build_com_reference_from_dcm(

            W_p_CoM_nominal_current=np.array(
                [
                    0.0,
                    0.0,
                    0.25
                ]
            ),

            W_xi_nominal_current_xy=np.zeros(
                2
            ),

            W_xi_ref=np.zeros(
                (
                    10,
                    2
                )
            ),

            omega0=5.0,

            dt_mpc=0.0
        ),

        "Reject zero dt_mpc"
    )


    assert_raises(

        lambda: build_com_reference_from_dcm(

            W_p_CoM_nominal_current=np.array(
                [
                    0.0,
                    0.0,
                    0.24
                ]
            ),

            W_xi_nominal_current_xy=np.zeros(
                2
            ),

            W_xi_ref=np.zeros(
                (
                    10,
                    2
                )
            ),

            omega0=5.0,

            dt_mpc=0.05,

            com_height=0.25
        ),

        "Reject inconsistent nominal CoM height"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NOMINAL CoM REFERENCE FROM DCM TESTS")
    print("=" * 70)


    test_com_velocity_relation()

    test_one_heun_step()

    test_reference_shapes()

    test_current_nominal_state()

    test_constant_height()

    test_dcm_com_consistency()

    test_equilibrium()

    test_receding_horizon_consistency()

    test_open_duck_lateral_sanity()

    test_future_only_output()

    test_invalid_input()


    print("\n")
    print("=" * 70)
    print("ALL NOMINAL CoM REFERENCE FROM DCM TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()