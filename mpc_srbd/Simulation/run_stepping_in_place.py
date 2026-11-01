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

It is a TEST-ONLY overlay.  No existing module is modified.
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
        z_foothold = ground_height

    Therefore this function deliberately ignores:

        W_p_L6_ref
        W_p_R6_ref
        W_p_dot_CoM
        W_p_dot_CoM_ref
        delta_T
        k_d

    The arguments are kept only so that the function has the same
    call interface as the current Raibert planner.
    """

    if swing_foot not in ("L", "R"):
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

    if foothold.shape != (3,):
        raise RuntimeError(
            "Fixed home foothold phải có shape (3,)."
        )

    # Flat-ground touchdown height.
    foothold[2] = float(
        ground_height
    )

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

    Only x and y are used as fixed touchdown references.
    z is forced to GROUND_HEIGHT by compute_fixed_home_foothold().
    """

    setup = _original_initialize_simulation(
        *args,
        **kwargs
    )

    model = setup.model
    data = setup.data

    # Home physical foot positions in World frame.
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
    ).reshape(3)

    W_p_R6_home = np.asarray(
        W_p_R6_home,
        dtype=float
    ).reshape(3)

    # Fixed stepping-in-place touchdown targets.
    _FIXED_HOME_FOOTHOLDS["L"] = (
        W_p_L6_home.copy()
    )

    _FIXED_HOME_FOOTHOLDS["R"] = (
        W_p_R6_home.copy()
    )

    _FIXED_HOME_FOOTHOLDS["L"][2] = (
        base_run.GROUND_HEIGHT
    )

    _FIXED_HOME_FOOTHOLDS["R"][2] = (
        base_run.GROUND_HEIGHT
    )

    # Replace only the Raibert foothold calculation used by
    # swing_low_level_controller.py.
    swing_controller.compute_raibert_foothold = (
        compute_fixed_home_foothold
    )

    print("\n")
    print("=" * 70)
    print("STEPPING-IN-PLACE TEST")
    print("=" * 70)

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
        _FIXED_HOME_FOOTHOLDS["L"]
    )

    print(
        "Fixed R foothold [World] =",
        _FIXED_HOME_FOOTHOLDS["R"]
    )

    print(
        "\nFoothold rule:"
    )

    print(
        "x = home x, y = home y, "
        "z = GROUND_HEIGHT"
    )

    print("=" * 70)

    return setup


# ============================================================
# TEST CONFIGURATION
# ============================================================

# Force stepping-in-place CoM command.
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
