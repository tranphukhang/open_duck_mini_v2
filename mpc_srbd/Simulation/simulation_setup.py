from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


# ============================================================
# PATH SETUP
#
# simulation_setup.py is a reusable module.
#
# It must be importable directly from:
#
#     Simulation/
#
# without requiring the caller to modify sys.path first.
# ============================================================

SIMULATION_DIR = (
    Path(__file__).resolve().parent
)

MPC_SRBD_DIR = (
    SIMULATION_DIR.parent
)

STATE_DIR = (
    MPC_SRBD_DIR
    /
    "State"
)

QP_DIR = (
    MPC_SRBD_DIR
    /
    "QP"
)


for directory in (
    MPC_SRBD_DIR,
    STATE_DIR,
    QP_DIR,
):

    directory_str = str(
        directory
    )

    if directory_str not in sys.path:

        sys.path.insert(
            0,
            directory_str
        )


# ============================================================
# STATE
# ============================================================

from get_current_state_mpc import (
    get_current_state_mpc,
)

from mass_CoM import (
    get_mass_CoM,
)

from W_inertia_CoM import (
    get_W_inertia_CoM,
)

from get_contact_state import (
    build_contact_state_map,
)


# ============================================================
# QP
# ============================================================

from QP_solver import (
    QPSolver,
)


# ============================================================
# SIMULATION SETUP DATA
# ============================================================

@dataclass
class SimulationSetup:
    """
    Common static resources required by SRBD-MPC simulations.

    This setup is deliberately independent of:

        - standing reference
        - DCM reference
        - foothold generator
        - gait FSM
        - swing trajectory
        - walking controller

    Those components belong to the corresponding higher-level
    simulation/controller modules.
    """

    # --------------------------------------------------------
    # MuJoCo
    # --------------------------------------------------------

    model: object

    data: object


    # --------------------------------------------------------
    # Simulation timing
    # --------------------------------------------------------

    dt_sim: float

    horizon: int

    dt_mpc: float

    dt_mpc_update: float

    steps_per_mpc_update: int

    prediction_horizon_time: float


    # --------------------------------------------------------
    # MuJoCo initialization
    # --------------------------------------------------------

    home_key_id: int

    home_key_name: str


    # --------------------------------------------------------
    # Constant SRBD parameters
    # --------------------------------------------------------

    mass_CoM: float

    W_inertia_CoM_q0: np.ndarray


    # --------------------------------------------------------
    # Initial MPC state
    # --------------------------------------------------------

    x_initial: np.ndarray


    # --------------------------------------------------------
    # Contact detection resources
    #
    # Built ONCE and reused throughout simulation.
    # --------------------------------------------------------

    contact_map: dict


    # --------------------------------------------------------
    # QP
    # --------------------------------------------------------

    qp_solver: QPSolver


# ============================================================
# SCALAR VALIDATION
# ============================================================

def _validate_positive_float(
    value,
    name,
):
    """
    Validate positive finite scalar.
    """

    value = float(
        value
    )


    if not np.isfinite(
        value
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    if value <= 0.0:

        raise ValueError(
            f"{name} phải lớn hơn 0."
        )


    return value


# ============================================================
# INTEGER VALIDATION
# ============================================================

def _validate_positive_integer(
    value,
    name,
):

    # --------------------------------------------------------
    # Avoid silently accepting:
    #
    #     10.5 -> int(10.5) -> 10
    # --------------------------------------------------------

    value_float = float(
        value
    )


    if not np.isfinite(
        value_float
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    value_int = int(
        round(
            value_float
        )
    )


    if not np.isclose(
        value_float,
        value_int,
        atol=1e-12,
        rtol=0.0,
    ):

        raise ValueError(
            f"{name} phải là số nguyên."
        )


    if value_int <= 0:

        raise ValueError(
            f"{name} phải lớn hơn 0."
        )


    return value_int


# ============================================================
# VALIDATE TIMING MULTIPLE
# ============================================================

def _compute_integer_step_ratio(
    larger_dt,
    smaller_dt,
    name,
):
    """
    Compute:

        larger_dt / smaller_dt

    and require the result to be an integer.

    Used for:

        dt_mpc_update / dt_sim

    because actual MPC updates must occur exactly on MuJoCo
    physics-step boundaries.
    """

    larger_dt = _validate_positive_float(
        larger_dt,
        "larger_dt",
    )


    smaller_dt = _validate_positive_float(
        smaller_dt,
        "smaller_dt",
    )


    steps = int(

        round(

            larger_dt

            /

            smaller_dt
        )
    )


    if steps <= 0:

        raise ValueError(
            f"{name}: số bước phải > 0."
        )


    reconstructed_dt = (

        steps

        *

        smaller_dt
    )


    if not np.isclose(
        reconstructed_dt,
        larger_dt,
        atol=1e-12,
        rtol=0.0,
    ):

        raise ValueError(
            f"{name}: {larger_dt} phải là bội số nguyên "
            f"của {smaller_dt}."
        )


    return steps


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
    qp_constraint_tolerance=1e-4,
    print_initialization=True,
):
    """
    Initialize common resources for SRBD-MPC simulation.

    Architecture
    ------------

        XML
         |
         v
        MuJoCo model/data
         |
         v
        home keyframe
         |
         v
        mj_forward + mj_subtreeVel
         |
         +----------------------+
         |                      |
         v                      v
    SRBD constants         contact map
         |
         v
    current MPC state x0
         |
         v
       QP solver


    Important
    ---------

    This function does NOT create:

        - walking reference
        - DCM reference
        - standing reference
        - footholds
        - gait phase
        - contact schedule
        - swing trajectory

    Those signals are scenario-dependent and must be generated
    outside this setup.


    Timing convention
    -----------------

    dt_sim
        MuJoCo physics timestep.


    dt_mpc_update
        Actual MPC solve/update period.

        Must satisfy:

            dt_mpc_update
                =
            integer * dt_sim


    dt_mpc
        MPC prediction discretization interval.

        This is NOT required to equal dt_mpc_update.

        For current walking architecture:

            dt_mpc = T_step / N


    Parameters
    ----------
    xml_path : str or pathlib.Path

        MuJoCo scene XML.


    horizon : int

        MPC horizon N.


    dt_mpc : float

        MPC prediction discretization period [s].


    dt_mpc_update : float

        Actual MPC solve/update period [s].


    home_key_name : str

        MuJoCo home keyframe.


    Returns
    -------
    setup : SimulationSetup
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


    horizon = (
        _validate_positive_integer(
            horizon,
            "horizon",
        )
    )


    dt_mpc = (
        _validate_positive_float(
            dt_mpc,
            "dt_mpc",
        )
    )


    dt_mpc_update = (
        _validate_positive_float(
            dt_mpc_update,
            "dt_mpc_update",
        )
    )


    qp_eps_abs = (
        _validate_positive_float(
            qp_eps_abs,
            "qp_eps_abs",
        )
    )


    qp_eps_rel = (
        _validate_positive_float(
            qp_eps_rel,
            "qp_eps_rel",
        )
    )


    qp_constraint_tolerance = (
        _validate_positive_float(
            qp_constraint_tolerance,
            "qp_constraint_tolerance",
        )
    )


    qp_max_iter = (
        _validate_positive_integer(
            qp_max_iter,
            "qp_max_iter",
        )
    )


    home_key_name = str(
        home_key_name
    )


    if len(
        home_key_name
    ) == 0:

        raise ValueError(
            "home_key_name không được rỗng."
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


    if (
        not np.isfinite(
            dt_sim
        )
        or
        dt_sim <= 0.0
    ):

        raise RuntimeError(
            "MuJoCo model.opt.timestep không hợp lệ."
        )


    # --------------------------------------------------------
    # Actual MPC solve period must align exactly with
    # MuJoCo integration steps.
    # --------------------------------------------------------

    steps_per_mpc_update = (
        _compute_integer_step_ratio(

            larger_dt=(
                dt_mpc_update
            ),

            smaller_dt=(
                dt_sim
            ),

            name=(
                "dt_mpc_update / dt_sim"
            ),
        )
    )


    # --------------------------------------------------------
    # Prediction horizon duration.
    #
    # Note:
    #
    #     horizon * dt_mpc
    #
    # is independent from:
    #
    #     dt_mpc_update
    # --------------------------------------------------------

    prediction_horizon_time = (

        horizon

        *

        dt_mpc
    )


    # ========================================================
    # 4. INITIALIZE MUJOCO FROM HOME KEYFRAME
    # ========================================================

    home_key_id = mujoco.mj_name2id(

        model,

        mujoco.mjtObj.mjOBJ_KEY,

        home_key_name,
    )


    if home_key_id < 0:

        raise RuntimeError(
            f"Không tìm thấy keyframe "
            f"'{home_key_name}' trong XML."
        )


    mujoco.mj_resetDataKeyframe(

        model,

        data,

        home_key_id,
    )


    # --------------------------------------------------------
    # Update all MuJoCo derived quantities at q0.
    #
    # IMPORTANT:
    #
    # These functions do NOT advance simulation time.
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )


    mujoco.mj_subtreeVel(
        model,
        data,
    )


    # ========================================================
    # 5. CONSTANT SRBD PARAMETERS
    # ========================================================

    mass_CoM = float(

        get_mass_CoM(
            model
        )
    )


    if (
        not np.isfinite(
            mass_CoM
        )
        or
        mass_CoM <= 0.0
    ):

        raise RuntimeError(
            "mass_CoM không hợp lệ."
        )


    # --------------------------------------------------------
    # Freeze whole-body inertia at home configuration q0.
    #
    # Current SRBD model treats:
    #
    #     ICoM = ICoM(q0)
    #
    # as constant during prediction.
    # --------------------------------------------------------

    W_inertia_CoM_q0 = np.asarray(

        get_W_inertia_CoM(

            model,

            data,
        ),

        dtype=float,

    )


    if W_inertia_CoM_q0.shape != (
        3,
        3,
    ):

        raise RuntimeError(
            "W_inertia_CoM_q0 phải có shape (3,3)."
        )


    if not np.all(
        np.isfinite(
            W_inertia_CoM_q0
        )
    ):

        raise RuntimeError(
            "W_inertia_CoM_q0 chứa NaN hoặc Inf."
        )


    # --------------------------------------------------------
    # Inertia tensor should be symmetric.
    # --------------------------------------------------------

    if not np.allclose(
        W_inertia_CoM_q0,
        W_inertia_CoM_q0.T,
        atol=1e-10,
        rtol=0.0,
    ):

        raise RuntimeError(
            "W_inertia_CoM_q0 không đối xứng."
        )


    # ========================================================
    # 6. INITIAL MPC STATE
    # ========================================================

    initial = (
        get_current_state_mpc(
            model,
            data,
        )
    )


    x_initial = np.asarray(

        initial.x,

        dtype=float,

    ).reshape(-1)


    if x_initial.shape != (
        13,
    ):

        raise RuntimeError(
            "x_initial phải có shape (13,)."
        )


    if not np.all(
        np.isfinite(
            x_initial
        )
    ):

        raise RuntimeError(
            "x_initial chứa NaN hoặc Inf."
        )


    x_initial = (
        x_initial.copy()
    )


    # ========================================================
    # 7. CONTACT DETECTION MAP
    #
    # Resolve MuJoCo IDs ONCE.
    #
    # The resulting dictionary is reused at every simulation
    # step by:
    #
    #     get_contact_state(...)
    #
    # or:
    #
    #     get_binary_contact_state(...)
    # ========================================================

    contact_map = (
        build_contact_state_map(
            model
        )
    )


    # ========================================================
    # 8. QP SOLVER
    # ========================================================

    qp_solver = QPSolver(

        verbose=(
            qp_verbose
        ),

        eps_abs=(
            qp_eps_abs
        ),

        eps_rel=(
            qp_eps_rel
        ),

        max_iter=(
            qp_max_iter
        ),

        polishing=(
            qp_polishing
        ),

        enable_warm_start=(
            qp_enable_warm_start
        ),

        constraint_tolerance=(
            qp_constraint_tolerance
        ),
    )


    # ========================================================
    # 9. BUILD SETUP OBJECT
    # ========================================================

    setup = SimulationSetup(

        model=(
            model
        ),

        data=(
            data
        ),

        dt_sim=(
            dt_sim
        ),

        horizon=(
            horizon
        ),

        dt_mpc=(
            dt_mpc
        ),

        dt_mpc_update=(
            dt_mpc_update
        ),

        steps_per_mpc_update=(
            steps_per_mpc_update
        ),

        prediction_horizon_time=(
            prediction_horizon_time
        ),

        home_key_id=(
            int(
                home_key_id
            )
        ),

        home_key_name=(
            home_key_name
        ),

        mass_CoM=(
            mass_CoM
        ),

        W_inertia_CoM_q0=(
            W_inertia_CoM_q0.copy()
        ),

        x_initial=(
            x_initial.copy()
        ),

        contact_map=(
            contact_map
        ),

        qp_solver=(
            qp_solver
        ),
    )


    # ========================================================
    # 10. INITIALIZATION INFORMATION
    # ========================================================

    if print_initialization:

        print("\n")
        print("=" * 70)
        print("SIMULATION INITIALIZATION")
        print("=" * 70)


        print(
            "XML =",
            xml_path
        )


        print(
            "\nhome_key_name =",
            home_key_name
        )


        print(
            "home_key_id   =",
            home_key_id
        )


        print(
            "\ndt_sim =",
            dt_sim,
            "s"
        )


        print(
            "dt_mpc =",
            dt_mpc,
            "s"
        )


        print(
            "dt_mpc_update =",
            dt_mpc_update,
            "s"
        )


        print(
            "MuJoCo steps / MPC update =",
            steps_per_mpc_update
        )


        print(
            "MPC update rate =",
            1.0 / dt_mpc_update,
            "Hz"
        )


        print(
            "\nhorizon N =",
            horizon
        )


        print(
            "prediction horizon =",
            prediction_horizon_time,
            "s"
        )


        print(
            "\nmass_CoM =",
            mass_CoM,
            "kg"
        )


        print(
            "\nW_inertia_CoM_q0 ="
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
            "\ncontact map:"
        )


        print(
            "floor geom =",
            contact_map[
                "floor_geom_name"
            ]
        )


        print(
            "left foot body =",
            contact_map[
                "left_foot_body_name"
            ]
        )


        print(
            "right foot body =",
            contact_map[
                "right_foot_body_name"
            ]
        )


        print(
            "\nIMPORTANT:"
        )


        print(
            "Reference generation is NOT part of "
            "simulation_setup.py."
        )


        print(
            "dt_mpc is prediction spacing; "
            "dt_mpc_update is actual MPC update period."
        )


        print(
            "No mj_step() was executed during initialization."
        )


    # ========================================================
    # 11. RETURN
    # ========================================================

    return setup