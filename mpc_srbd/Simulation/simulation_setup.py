from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


SIMULATION_DIR = Path(__file__).resolve().parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
STATE_DIR = MPC_SRBD_DIR / "State"
QP_DIR = MPC_SRBD_DIR / "QP"

for directory in (
    MPC_SRBD_DIR,
    STATE_DIR,
    QP_DIR,
):
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)


from get_current_state_mpc import get_current_state_mpc
from mass_CoM import get_mass_CoM
from W_inertia_CoM import get_B_inertia_CoM
from get_contact_state import build_contact_state_map
from QP_solver import QPSolver


@dataclass
class SimulationSetup:
    """
    Static resources shared by SRBD-MPC simulations.

    HECTOR-style inertia convention
    -------------------------------
    B_inertia_CoM is the constant SRBD inertia expressed in the
    body frame.

    At every MPC update:

        W_I_CoM(k)
            =
        W_R_B(k) @ B_I_CoM @ W_R_B(k).T

    is reconstructed from orientation feedback by
    get_current_state_mpc().
    """

    model: object
    data: object

    dt_sim: float
    horizon: int
    dt_mpc: float
    dt_mpc_update: float
    steps_per_mpc_update: int
    prediction_horizon_time: float

    home_key_id: int
    home_key_name: str

    mass_CoM: float
    B_inertia_CoM: np.ndarray

    x_initial: np.ndarray

    contact_map: dict
    qp_solver: QPSolver


def _validate_positive_float(value, name):
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} chứa NaN hoặc Inf.")

    if value <= 0.0:
        raise ValueError(f"{name} phải > 0.")

    return value


def _validate_positive_integer(value, name):
    value_float = float(value)

    if not np.isfinite(value_float):
        raise ValueError(f"{name} chứa NaN hoặc Inf.")

    value_int = int(round(value_float))

    if not np.isclose(
        value_float,
        value_int,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"{name} phải là số nguyên.")

    if value_int <= 0:
        raise ValueError(f"{name} phải > 0.")

    return value_int


def _compute_integer_step_ratio(
    larger_dt,
    smaller_dt,
    name,
):
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
        raise ValueError(f"{name}: số bước phải > 0.")

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
    Initialize common SRBD-MPC resources.

    No mj_step() is executed here.

    Constant SRBD parameters:
        mass_CoM
        B_inertia_CoM

    Feedback-dependent SRBD quantities are NOT frozen here:
        x_k
        Ew_inverse(k)
        r_L(k)
        r_R(k)
        W_inertia_CoM(k)

    Those are reconstructed at every MPC update.
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy XML:\n{xml_path}"
        )

    horizon = _validate_positive_integer(
        horizon,
        "horizon",
    )

    dt_mpc = _validate_positive_float(
        dt_mpc,
        "dt_mpc",
    )

    dt_mpc_update = _validate_positive_float(
        dt_mpc_update,
        "dt_mpc_update",
    )

    qp_eps_abs = _validate_positive_float(
        qp_eps_abs,
        "qp_eps_abs",
    )

    qp_eps_rel = _validate_positive_float(
        qp_eps_rel,
        "qp_eps_rel",
    )

    qp_constraint_tolerance = _validate_positive_float(
        qp_constraint_tolerance,
        "qp_constraint_tolerance",
    )

    qp_max_iter = _validate_positive_integer(
        qp_max_iter,
        "qp_max_iter",
    )

    home_key_name = str(home_key_name)

    if len(home_key_name) == 0:
        raise ValueError(
            "home_key_name không được rỗng."
        )

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(
        model
    )

    dt_sim = float(
        model.opt.timestep
    )

    if (
        not np.isfinite(dt_sim)
        or
        dt_sim <= 0.0
    ):
        raise RuntimeError(
            "MuJoCo model.opt.timestep không hợp lệ."
        )

    steps_per_mpc_update = _compute_integer_step_ratio(
        larger_dt=dt_mpc_update,
        smaller_dt=dt_sim,
        name="dt_mpc_update / dt_sim",
    )

    prediction_horizon_time = (
        horizon
        *
        dt_mpc
    )

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

    # Update MuJoCo derived quantities at q0 without advancing time.
    mujoco.mj_forward(
        model,
        data,
    )

    mujoco.mj_subtreeVel(
        model,
        data,
    )

    mass_CoM = float(
        get_mass_CoM(
            model
        )
    )

    if (
        not np.isfinite(mass_CoM)
        or
        mass_CoM <= 0.0
    ):
        raise RuntimeError(
            "mass_CoM không hợp lệ."
        )

    # ========================================================
    # CONSTANT SRBD BODY-FRAME INERTIA
    #
    # Identify once at q0:
    #
    #     B_I_CoM
    #
    # During runtime, W_I_CoM(k) is reconstructed only from
    # current body orientation feedback:
    #
    #     W_I_CoM(k)
    #       =
    #     W_R_B(k) @ B_I_CoM @ W_R_B(k).T
    # ========================================================

    B_inertia_CoM = np.asarray(
        get_B_inertia_CoM(
            model=model,
            data=data,
        ),
        dtype=float,
    )

    if B_inertia_CoM.shape != (3, 3):
        raise RuntimeError(
            "B_inertia_CoM phải có shape (3,3)."
        )

    if not np.all(
        np.isfinite(
            B_inertia_CoM
        )
    ):
        raise RuntimeError(
            "B_inertia_CoM chứa NaN hoặc Inf."
        )

    if not np.allclose(
        B_inertia_CoM,
        B_inertia_CoM.T,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "B_inertia_CoM không đối xứng."
        )

    initial = get_current_state_mpc(
        model=model,
        data=data,
        B_inertia_CoM=B_inertia_CoM,
    )

    x_initial = np.asarray(
        initial.x,
        dtype=float,
    ).reshape(-1)

    if x_initial.shape != (13,):
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

    x_initial = x_initial.copy()

    contact_map = build_contact_state_map(
        model
    )

    qp_solver = QPSolver(
        verbose=qp_verbose,
        eps_abs=qp_eps_abs,
        eps_rel=qp_eps_rel,
        max_iter=qp_max_iter,
        polishing=qp_polishing,
        enable_warm_start=qp_enable_warm_start,
        constraint_tolerance=qp_constraint_tolerance,
    )

    setup = SimulationSetup(
        model=model,
        data=data,
        dt_sim=dt_sim,
        horizon=horizon,
        dt_mpc=dt_mpc,
        dt_mpc_update=dt_mpc_update,
        steps_per_mpc_update=steps_per_mpc_update,
        prediction_horizon_time=prediction_horizon_time,
        home_key_id=int(home_key_id),
        home_key_name=home_key_name,
        mass_CoM=mass_CoM,
        B_inertia_CoM=B_inertia_CoM.copy(),
        x_initial=x_initial.copy(),
        contact_map=contact_map,
        qp_solver=qp_solver,
    )

    if print_initialization:
        print("\n")
        print("=" * 70)
        print("SIMULATION INITIALIZATION")
        print("=" * 70)

        print("\nXML =", xml_path)

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
            "\nB_inertia_CoM (constant SRBD body-frame inertia) ="
        )

        print(
            B_inertia_CoM
        )

        print(
            "\nW_inertia_CoM_initial "
            "(reconstructed from orientation feedback) ="
        )

        print(
            initial.W_inertia_CoM
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
            "B_inertia_CoM is constant."
        )

        print(
            "W_inertia_CoM is recomputed from orientation "
            "feedback at each MPC update and frozen over that "
            "prediction horizon."
        )

        print(
            "dt_mpc is prediction spacing; "
            "dt_mpc_update is actual MPC update period."
        )

        print(
            "No mj_step() was executed during initialization."
        )

    return setup
