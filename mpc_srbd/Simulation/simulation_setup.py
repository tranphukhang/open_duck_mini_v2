from pathlib import Path

import numpy as np
import mujoco


# ============================================================
# STATE
# ============================================================

from get_current_state import (
    get_current_state
)

from mass_CoM import (
    get_mass_CoM
)

from W_inertia_CoM import (
    get_W_inertia_CoM
)


# ============================================================
# COMMAND
# ============================================================

from reference_generator import (
    ReferenceGenerator
)


# ============================================================
# QP
# ============================================================

from QP_solver import (
    QPSolver
)


# ============================================================
# INITIALIZE SIMULATION
# ============================================================

def initialize_simulation(
    xml_path,
    horizon,
    dt_mpc,
    dt_mpc_update,
    home_key_name="home",
    qp_verbose=False,
    qp_eps_abs=1e-5,
    qp_eps_rel=1e-5,
    qp_max_iter=1000,
    qp_polishing=True,
    qp_enable_warm_start=False,
    qp_constraint_tolerance=1e-4
):
    """
    Initialize the common components required by SRBD-MPC
    simulations.

    This function performs:

        1. Load MuJoCo model
        2. Create MuJoCo data
        3. Configure simulation timing
        4. Reset robot to home keyframe
        5. Update MuJoCo kinematics/dynamics
        6. Compute constant SRBD parameters
        7. Read initial state
        8. Initialize reference generator
        9. Initialize QP solver


    Parameters
    ----------
    xml_path : str or pathlib.Path

        Path to the MuJoCo scene XML.


    horizon : int

        MPC prediction horizon N.


    dt_mpc : float

        MPC prediction discretization step.


    dt_mpc_update : float

        Actual MPC control update period.


    home_key_name : str

        Name of MuJoCo keyframe used for initialization.

        Default:

            "home"


    qp_verbose : bool

        Enable OSQP terminal output.


    qp_eps_abs : float

        OSQP absolute tolerance.


    qp_eps_rel : float

        OSQP relative tolerance.


    qp_max_iter : int

        Maximum number of OSQP iterations.


    qp_polishing : bool

        Enable OSQP polishing.


    qp_enable_warm_start : bool

        Enable OSQP warm start.


    qp_constraint_tolerance : float

        Maximum accepted post-solve constraint violation.


    Returns
    -------
    setup : dict

        Dictionary containing:

            model

            data

            dt_sim

            steps_per_mpc_update

            mass_CoM

            W_inertia_CoM_q0

            x_initial

            reference_generator

            qp_solver
    """


    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    xml_path = Path(
        xml_path
    )


    if not xml_path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy XML:\n{xml_path}"
        )


    if horizon <= 0:

        raise ValueError(
            "horizon phải lớn hơn 0."
        )


    if dt_mpc <= 0.0:

        raise ValueError(
            "dt_mpc phải lớn hơn 0."
        )


    if dt_mpc_update <= 0.0:

        raise ValueError(
            "dt_mpc_update phải lớn hơn 0."
        )


    # ========================================================
    # 2. LOAD MUJOCO MODEL
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(
            xml_path
        )
    )


    data = mujoco.MjData(
        model
    )


    # ========================================================
    # 3. SIMULATION TIMING
    # ========================================================

    dt_sim = float(
        model.opt.timestep
    )


    steps_per_mpc_update = int(
        round(
            dt_mpc_update
            /
            dt_sim
        )
    )


    # --------------------------------------------------------
    # MPC update period must correspond to an integer number
    # of MuJoCo integration steps.
    # --------------------------------------------------------

    if not np.isclose(
        steps_per_mpc_update
        *
        dt_sim,
        dt_mpc_update,
        atol=1e-12
    ):

        raise ValueError(
            "dt_mpc_update phải là bội số nguyên "
            "của dt_sim."
        )


    # ========================================================
    # 4. INITIALIZE MUJOCO FROM HOME KEYFRAME
    # ========================================================

    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        home_key_name
    )


    if home_key_id < 0:

        raise RuntimeError(
            f'Không tìm thấy keyframe "{home_key_name}" '
            "trong XML."
        )


    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_key_id
    )


    # --------------------------------------------------------
    # Update all MuJoCo derived quantities at q0.
    #
    # These operations do not advance simulation time.
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data
    )


    mujoco.mj_subtreeVel(
        model,
        data
    )


    # ========================================================
    # 5. CONSTANT SRBD PARAMETERS
    # ========================================================

    mass_CoM = get_mass_CoM(
        model
    )


    # --------------------------------------------------------
    # Current SRBD formulation freezes the CoM inertia tensor
    # at the initial configuration q0.
    # --------------------------------------------------------

    W_inertia_CoM_q0 = (
        get_W_inertia_CoM(
            model,
            data
        )
    )


    # ========================================================
    # 6. INITIAL STATE
    # ========================================================

    x_initial = get_current_state(
        model,
        data
    )


    # ========================================================
    # 7. REFERENCE GENERATOR
    # ========================================================

    reference_generator = ReferenceGenerator(
        x_initial=x_initial,
        horizon=horizon,
        dt_prediction=dt_mpc,
        dt_update=dt_mpc_update
    )


    # ========================================================
    # 8. QP SOLVER
    # ========================================================

    qp_solver = QPSolver(
        verbose=qp_verbose,
        eps_abs=qp_eps_abs,
        eps_rel=qp_eps_rel,
        max_iter=qp_max_iter,
        polishing=qp_polishing,
        enable_warm_start=(
            qp_enable_warm_start
        ),
        constraint_tolerance=(
            qp_constraint_tolerance
        )
    )


    # ========================================================
    # 9. PRINT COMMON INITIALIZATION INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SIMULATION INITIALIZATION")
    print("=" * 70)


    print(
        "XML =",
        xml_path
    )


    print(
        "\ndt_sim             =",
        dt_sim,
        "s"
    )


    print(
        "dt_mpc             =",
        dt_mpc,
        "s"
    )


    print(
        "dt_mpc_update      =",
        dt_mpc_update,
        "s"
    )


    print(
        "MPC update rate    =",
        1.0 / dt_mpc_update,
        "Hz"
    )


    print(
        "MuJoCo steps/MPC   =",
        steps_per_mpc_update
    )


    print(
        "Prediction horizon =",
        horizon * dt_mpc,
        "s"
    )


    print(
        "\nmass_CoM =",
        mass_CoM,
        "kg"
    )


    print(
        "\nW_inertia_CoM(q0) ="
    )


    print(
        W_inertia_CoM_q0
    )


    print(
        "\nx_initial ="
    )


    print(
        x_initial
    )


    print(
        "\nInitial CoM position reference ="
    )


    print(
        reference_generator.get_position_reference()
    )


    # ========================================================
    # 10. RETURN SETUP
    # ========================================================

    setup = {

        "model":
            model,

        "data":
            data,

        "dt_sim":
            dt_sim,

        "steps_per_mpc_update":
            steps_per_mpc_update,

        "mass_CoM":
            mass_CoM,

        "W_inertia_CoM_q0":
            W_inertia_CoM_q0,

        "x_initial":
            x_initial,

        "reference_generator":
            reference_generator,

        "qp_solver":
            qp_solver
    }


    return setup