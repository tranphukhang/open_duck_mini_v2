"""
run_stepping_in_place.py

Temporary diagnostic run for stepping in place.

Purpose
-------
Reuse the existing walking pipeline without modifying the current
production modules, but replace the Raibert foothold calculation by
a fixed foothold captured from the robot home configuration.

The rest of the pipeline is unchanged:

    contact schedule
        -> fixed home foothold
        -> cubic Bezier swing trajectory
        -> 5D numerical IK
        -> swing joint PD
        -> stance MPC wrench control

Important
---------
This file intentionally monkey-patches only the Raibert foothold
function used inside swing_low_level_controller.py.

It is a TEST-ONLY overlay.
No existing production module is modified.

For this stepping-in-place test, the touchdown position is exactly
the physical home foot position:

    x_foothold = x_home
    y_foothold = y_home
    z_foothold = z_home

Therefore the swing foot returns to the same physical position from
which the test was initialized.
"""

import numpy as np

import run_walking as base_run
import swing_low_level_controller as swing_controller


# ============================================================
# FIXED HOME FOOTHOLDS
# ============================================================

_FIXED_HOME_FOOTHOLDS = {
    "L": None,
    "R": None,
}


# ============================================================
# FIXED FOOTHOLD PLANNER
# ============================================================

def compute_fixed_home_foothold(
    W_p_L6_ref,
    W_p_R6_ref,
    W_p_dot_CoM,
    W_p_dot_CoM_ref,
    swing_foot,
    delta_T,
    k_d,
    ground_height=0.0,
):
    """
    Drop-in replacement for compute_raibert_foothold().

    For stepping-in-place testing:

        x_foothold = x_home
        y_foothold = y_home
        z_foothold = z_home

    Therefore the complete home foot position is preserved.

    The following arguments are intentionally ignored:

        W_p_L6_ref
        W_p_R6_ref
        W_p_dot_CoM
        W_p_dot_CoM_ref
        delta_T
        k_d
        ground_height

    They are kept only so that this function has the same call
    interface as the current Raibert foothold planner.
    """

    if swing_foot not in (
        "L",
        "R"
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    foothold = _FIXED_HOME_FOOTHOLDS[
        swing_foot
    ]

    if foothold is None:
        raise RuntimeError(
            "Fixed home foothold chưa được khởi tạo."
        )

    foothold = np.asarray(
        foothold,
        dtype=float
    ).copy()

    if foothold.shape != (
        3,
    ):
        raise RuntimeError(
            "Fixed home foothold phải có shape (3,)."
        )

    # IMPORTANT:
    #
    # Không ép:
    #
    #     foothold[2] = ground_height
    #
    # vì stepping-in-place test cần chân quay về đúng
    # physical home position, bao gồm cả z_home.

    return foothold


# ============================================================
# INITIALIZATION WRAPPER
# ============================================================

_original_initialize_simulation = (
    base_run.initialize_simulation
)


def initialize_simulation_fixed_foothold(
    *args,
    **kwargs
):
    """
    Call the normal simulation initialization, then capture the
    physical L6/R6 positions at the home configuration.

    The complete home positions are preserved:

        [x_home, y_home, z_home]

    These positions are used directly as the fixed touchdown
    targets for stepping in place.
    """

    setup = _original_initialize_simulation(
        *args,
        **kwargs
    )

    model = setup.model
    data = setup.data


    # ========================================================
    # HOME PHYSICAL FOOT POSITIONS
    # ========================================================

    W_p_L6_home = (
        base_run.get_W_position_L6(
            model,
            data
        )
    )

    W_p_R6_home = (
        base_run.get_W_position_R6(
            model,
            data
        )
    )


    W_p_L6_home = np.asarray(
        W_p_L6_home,
        dtype=float
    ).reshape(
        3
    )

    W_p_R6_home = np.asarray(
        W_p_R6_home,
        dtype=float
    ).reshape(
        3
    )


    # ========================================================
    # FIXED STEPPING-IN-PLACE FOOTHOLDS
    #
    # Preserve complete home position:
    #
    #     x = x_home
    #     y = y_home
    #     z = z_home
    #
    # Do NOT force z to GROUND_HEIGHT.
    # ========================================================

    _FIXED_HOME_FOOTHOLDS[
        "L"
    ] = (
        W_p_L6_home.copy()
    )

    _FIXED_HOME_FOOTHOLDS[
        "R"
    ] = (
        W_p_R6_home.copy()
    )


    # ========================================================
    # MONKEY PATCH ONLY THE RAIBERT FOOTHOLD PLANNER
    # ========================================================

    swing_controller.compute_raibert_foothold = (
        compute_fixed_home_foothold
    )


    # ========================================================
    # PRINT TEST CONFIGURATION
    # ========================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "STEPPING-IN-PLACE TEST"
    )

    print(
        "=" * 70
    )


    print(
        "\nRaibert foothold planner: DISABLED"
    )

    print(
        "Fixed foothold planner: ENABLED"
    )


    print(
        "\nHome L6 position [World] =",
        W_p_L6_home
    )

    print(
        "Home R6 position [World] =",
        W_p_R6_home
    )


    print(
        "\nFixed L foothold [World] =",
        _FIXED_HOME_FOOTHOLDS[
            "L"
        ]
    )

    print(
        "Fixed R foothold [World] =",
        _FIXED_HOME_FOOTHOLDS[
            "R"
        ]
    )


    print(
        "\nFoothold rule:"
    )

    print(
        "x = home x, "
        "y = home y, "
        "z = home z"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Touchdown height is the physical home-foot height."
    )

    print(
        "GROUND_HEIGHT is NOT used to overwrite foothold z."
    )


    print(
        "=" * 70
    )


    return setup


# ============================================================
# TEST CONFIGURATION
# ============================================================

# Force stepping-in-place CoM command:
#
#     vx_ref = 0
#     vy_ref = 0
#     vz_ref = 0

base_run.V_COM_CMD = np.zeros(
    3,
    dtype=float
)


# Replace only the initialization entry point used inside
# run_walking.main().

base_run.initialize_simulation = (
    initialize_simulation_fixed_foothold
)


# ============================================================
# MAIN
# ============================================================

def main():

    base_run.main()


if __name__ == "__main__":

    main()