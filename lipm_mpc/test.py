# lipm_mpc/test.py

from __future__ import annotations

import numpy as np


# ============================================================
# IMPORT
# ============================================================

if __package__:

    from .lipm_model import (
        LIPMModel1D,
    )

    from .mpc_1d import (
        LIPMMPC1D,
    )

else:

    from lipm_model import (
        LIPMModel1D,
    )

    from mpc_1d import (
        LIPMMPC1D,
    )


# ============================================================
# TEMPORARY TEST PARAMETERS
# ============================================================

MPC_TIMESTEP = 0.03

MPC_HORIZON_STEPS = 16

COM_HEIGHT = 0.205

GRAVITY = 9.81

TERMINAL_WEIGHT = 1.0

CONTROL_WEIGHT = 0.01


SOLVER_OPTIONS = {
    "ftol": 1e-10,
    "maxiter": 500,
}


# ============================================================
# CREATE MPC
# ============================================================

def create_mpc():

    model = LIPMModel1D(
        timestep=MPC_TIMESTEP,
        com_height=COM_HEIGHT,
        gravity=GRAVITY,
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

    return model, mpc


# ============================================================
# TEST QP MATRICES
# ============================================================

def test_qp_matrices():

    _, mpc = create_mpc()

    current_state = np.zeros(
        3,
        dtype=float,
    )

    goal_state = np.array(
        [
            0.02,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    lower = np.full(
        MPC_HORIZON_STEPS,
        -0.10,
    )

    upper = np.full(
        MPC_HORIZON_STEPS,
        +0.10,
    )

    (
        H,
        f,
        G,
        lb,
        ub,
    ) = mpc.build_qp(
        current_state=current_state,
        goal_state=goal_state,
        lower_bounds=lower,
        upper_bounds=upper,
    )

    assert H.shape == (
        MPC_HORIZON_STEPS,
        MPC_HORIZON_STEPS,
    )

    assert f.shape == (
        MPC_HORIZON_STEPS,
    )

    assert G.shape == (
        MPC_HORIZON_STEPS,
        MPC_HORIZON_STEPS,
    )

    assert lb.shape == (
        MPC_HORIZON_STEPS,
    )

    assert ub.shape == (
        MPC_HORIZON_STEPS,
    )

    np.testing.assert_allclose(
        H,
        H.T,
        atol=1e-12,
    )

    eigenvalues = (
        np.linalg.eigvalsh(
            H
        )
    )

    assert np.all(
        eigenvalues > 0.0
    )

    print(
        "[PASS] QP matrices"
    )


# ============================================================
# TEST MPC SOLVE
# ============================================================

def test_mpc_solve():

    model, mpc = create_mpc()

    current_state = np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    goal_state = np.array(
        [
            0.02,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    lower = np.full(
        MPC_HORIZON_STEPS,
        -0.10,
        dtype=float,
    )

    upper = np.full(
        MPC_HORIZON_STEPS,
        +0.10,
        dtype=float,
    )

    result = mpc.solve(
        current_state=current_state,

        goal_state=goal_state,

        lower_bounds=lower,

        upper_bounds=upper,

        solver_options=(
            SOLVER_OPTIONS
        ),
    )

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
    # Check state dynamics
    # --------------------------------------------------------

    for k in range(
        MPC_HORIZON_STEPS
    ):

        expected_next = (
            model.propagate(
                state=result.state[k],
                jerk=result.control[k],
            )
        )

        np.testing.assert_allclose(
            result.state[k + 1],
            expected_next,
            atol=1e-10,
        )

    # --------------------------------------------------------
    # Check ZMP constraints
    # --------------------------------------------------------

    assert np.all(
        result.zmp
        >=
        lower
        -
        1e-8
    )

    assert np.all(
        result.zmp
        <=
        upper
        +
        1e-8
    )

    # --------------------------------------------------------
    # Terminal state should move toward goal
    # --------------------------------------------------------

    initial_error = (
        np.linalg.norm(
            current_state
            -
            goal_state
        )
    )

    terminal_error = (
        np.linalg.norm(
            result.state[-1]
            -
            goal_state
        )
    )

    assert (
        terminal_error
        <
        initial_error
    )

    print(
        "[PASS] MPC solve"
    )

    print()

    print(
        f"First jerk       : "
        f"{result.first_control:.8f} m/s^3"
    )

    print(
        f"Terminal state   : "
        f"{result.state[-1]}"
    )

    print(
        f"Terminal error   : "
        f"{terminal_error:.8f}"
    )

    print(
        f"Objective        : "
        f"{result.objective:.10f}"
    )

    print(
        f"Solver iterations: "
        f"{result.iterations}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "LIPM MPC 1D TEST"
    )

    print(
        "=" * 60
    )

    test_qp_matrices()

    test_mpc_solve()

    print()

    print(
        "=" * 60
    )

    print(
        "ALL MPC 1D TESTS PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()