from pathlib import Path
import sys
import csv
import json
import time

import numpy as np
import mujoco
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

SIMULATION_DIR = Path(__file__).resolve().parent

MPC_SRBD_DIR = SIMULATION_DIR.parent

REPO_ROOT = MPC_SRBD_DIR.parent


KINEMATICS_DIR = (
    MPC_SRBD_DIR
    /
    "Kinematics"
)


if str(KINEMATICS_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(KINEMATICS_DIR)
    )


# ============================================================
# KINEMATICS
# ============================================================

from numerical_IK_hybrid import (
    solve_numerical_IK_hybrid
)


from numerical_IK_params import (
    SIGMA_THRESHOLD,
    ALPHA_NEWTON,
    ETA_GRADIENT,
    IK_TOLERANCE,
    IK_MAX_ITERATIONS
)


# ============================================================
# MODEL
# ============================================================

XML_PATH = (
    REPO_ROOT
    /
    "xmls"
    /
    "scene_flat_terrian_v3.xml"
)


HOME_KEY_NAME = "home"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    SIMULATION_DIR
    /
    "kinematic_workspace_results"
)


# ============================================================
# WORKSPACE SCAN PARAMETERS
#
# These are SEARCH LIMITS only.
#
# They are NOT foothold bounds.
#
# Actual foothold bounds will be extracted from IK results.
# ============================================================

DX_MIN = -0.15
DX_MAX = +0.15

DY_MIN = -0.10
DY_MAX = +0.10


# Grid resolution [m]
GRID_STEP = 0.005


# ============================================================
# IK SETTINGS
#
# Match current swing-leg controller.
# ============================================================

SWING_PITCH_DES = 0.0

IK_ORIENTATION_TOLERANCE = 1e-3


# ============================================================
# SAFE WORKSPACE SETTINGS
#
# A point is "raw reachable" if IK converges.
#
# A point is "safe reachable" only if IK converges AND
# all controlled leg joints stay at least this far from
# their hard joint limits.
# ============================================================

SAFE_JOINT_MARGIN_DEG = 5.0

SAFE_JOINT_MARGIN_RAD = np.deg2rad(
    SAFE_JOINT_MARGIN_DEG
)


# ============================================================
# OPTIONAL MUJOCO VISUALIZATION
#
# False:
#     scan + save results only
#
# True:
#     after scan, visualize selected safe-boundary points
# ============================================================

ENABLE_VIEWER = False

VIEW_HOLD_TIME = 1.0


# ============================================================
# LEG INFORMATION
# ============================================================

LEG_INFO = {

    "L": {

        "site_name":
            "L6_IK",

        "joint_names": (
            "L_hip_yaw",
            "L_hip_roll",
            "L_hip_pitch",
            "L_knee",
            "L_ankle",
        ),
    },


    "R": {

        "site_name":
            "R6_IK",

        "joint_names": (
            "R_hip_yaw",
            "R_hip_roll",
            "R_hip_pitch",
            "R_knee",
            "R_ankle",
        ),
    },
}


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=200
)


# ============================================================
# ANGLE WRAP
# ============================================================

def wrap_to_pi(
    angle
):
    return (
        float(angle)
        +
        np.pi
    ) % (
        2.0
        *
        np.pi
    ) - np.pi


# ============================================================
# GET PITCH FROM ROTATION MATRIX
# ============================================================

def get_pitch_from_rotation_matrix(
    R
):
    R = np.asarray(
        R,
        dtype=float
    ).reshape(
        3,
        3
    )


    denominator = np.sqrt(
        R[0, 0] ** 2
        +
        R[1, 0] ** 2
    )


    pitch = np.arctan2(
        -R[2, 0],
        denominator
    )


    return float(
        pitch
    )


# ============================================================
# GRID AXIS
# ============================================================

def build_grid_axis(
    minimum,
    maximum,
    step
):
    minimum = float(
        minimum
    )

    maximum = float(
        maximum
    )

    step = float(
        step
    )


    if step <= 0.0:

        raise ValueError(
            "GRID_STEP phải > 0."
        )


    count = int(
        round(
            (
                maximum
                -
                minimum
            )
            /
            step
        )
    )


    axis = np.linspace(
        minimum,
        maximum,
        count + 1
    )


    return axis


# ============================================================
# LOAD MODEL AT HOME
# ============================================================

def load_home_model():

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"Không tìm thấy XML:\n{XML_PATH}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(
            XML_PATH
        )
    )


    data = mujoco.MjData(
        model
    )


    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        HOME_KEY_NAME
    )


    if home_key_id < 0:

        raise RuntimeError(
            f"Không tìm thấy keyframe "
            f"'{HOME_KEY_NAME}'."
        )


    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_key_id
    )


    mujoco.mj_forward(
        model,
        data
    )


    return (
        model,
        data
    )


# ============================================================
# RESET TO HOME
# ============================================================

def reset_to_home(
    model,
    data
):
    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        HOME_KEY_NAME
    )


    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_key_id
    )


    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# GET SITE ID
# ============================================================

def get_site_id(
    model,
    site_name
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )


    if site_id < 0:

        raise RuntimeError(
            f"Không tìm thấy site '{site_name}'."
        )


    return int(
        site_id
    )


# ============================================================
# GET HOME FOOT POSITION
# ============================================================

def get_home_foot_position(
    model,
    data,
    leg
):
    site_name = (
        LEG_INFO[
            leg
        ][
            "site_name"
        ]
    )


    site_id = get_site_id(
        model,
        site_name
    )


    return (
        data.site_xpos[
            site_id
        ].copy()
    )


# ============================================================
# GET LEG JOINT INFORMATION
# ============================================================

def get_leg_joint_information(
    model,
    leg
):
    joint_names = (
        LEG_INFO[
            leg
        ][
            "joint_names"
        ]
    )


    joint_ids = []

    qpos_addresses = []

    q_min = []

    q_max = []


    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )


        if joint_id < 0:

            raise RuntimeError(
                f"Không tìm thấy joint "
                f"'{joint_name}'."
            )


        joint_ids.append(
            int(
                joint_id
            )
        )


        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )


        qpos_addresses.append(
            qpos_address
        )


        if not model.jnt_limited[
            joint_id
        ]:

            raise RuntimeError(
                f"Joint '{joint_name}' "
                "không có joint limit."
            )


        q_min.append(
            float(
                model.jnt_range[
                    joint_id,
                    0
                ]
            )
        )


        q_max.append(
            float(
                model.jnt_range[
                    joint_id,
                    1
                ]
            )
        )


    return {

        "joint_names":
            tuple(
                joint_names
            ),

        "joint_ids":
            np.asarray(
                joint_ids,
                dtype=int
            ),

        "qpos_addresses":
            np.asarray(
                qpos_addresses,
                dtype=int
            ),

        "q_min":
            np.asarray(
                q_min,
                dtype=float
            ),

        "q_max":
            np.asarray(
                q_max,
                dtype=float
            ),
    }


# ============================================================
# COMPUTE JOINT MARGIN
# ============================================================

def compute_joint_margin(
    q,
    q_min,
    q_max
):
    q = np.asarray(
        q,
        dtype=float
    ).reshape(
        5
    )


    lower_margin = (
        q
        -
        q_min
    )


    upper_margin = (
        q_max
        -
        q
    )


    joint_margin = np.minimum(
        lower_margin,
        upper_margin
    )


    minimum_margin = float(
        np.min(
            joint_margin
        )
    )


    return (
        joint_margin,
        minimum_margin
    )


# ============================================================
# VERIFY IK SOLUTION WITH MUJOCO FORWARD KINEMATICS
# ============================================================

def verify_ik_solution(
    model,
    data,
    leg,
    q_solution,
    W_p_target,
    pitch_target
):
    joint_info = (
        get_leg_joint_information(
            model,
            leg
        )
    )


    qpos_addresses = (
        joint_info[
            "qpos_addresses"
        ]
    )


    site_name = (
        LEG_INFO[
            leg
        ][
            "site_name"
        ]
    )


    site_id = get_site_id(
        model,
        site_name
    )


    # --------------------------------------------------------
    # Save real state
    # --------------------------------------------------------

    qpos_original = (
        data.qpos.copy()
    )

    qvel_original = (
        data.qvel.copy()
    )


    try:

        # ----------------------------------------------------
        # Apply IK solution
        # ----------------------------------------------------

        data.qpos[
            qpos_addresses
        ] = q_solution


        mujoco.mj_forward(
            model,
            data
        )


        # ----------------------------------------------------
        # Actual foot position
        # ----------------------------------------------------

        W_p_actual = (
            data.site_xpos[
                site_id
            ].copy()
        )


        # ----------------------------------------------------
        # Actual foot orientation
        # ----------------------------------------------------

        W_R_site = (
            data.site_xmat[
                site_id
            ].reshape(
                3,
                3
            ).copy()
        )


        pitch_actual = (
            get_pitch_from_rotation_matrix(
                W_R_site
            )
        )


        # ----------------------------------------------------
        # Independent errors
        # ----------------------------------------------------

        position_error = (
            np.asarray(
                W_p_target,
                dtype=float
            )
            -
            W_p_actual
        )


        position_error_norm = float(
            np.linalg.norm(
                position_error
            )
        )


        pitch_error = wrap_to_pi(
            pitch_target
            -
            pitch_actual
        )


        pitch_error_abs = abs(
            float(
                pitch_error
            )
        )


    finally:

        # ----------------------------------------------------
        # Restore real simulation state
        # ----------------------------------------------------

        data.qpos[:] = (
            qpos_original
        )


        data.qvel[:] = (
            qvel_original
        )


        mujoco.mj_forward(
            model,
            data
        )


    return {

        "W_p_actual":
            W_p_actual,

        "pitch_actual":
            pitch_actual,

        "position_error":
            position_error,

        "position_error_norm":
            position_error_norm,

        "pitch_error":
            pitch_error,

        "pitch_error_abs":
            pitch_error_abs,
    }


# ============================================================
# EVALUATE ONE WORKSPACE POINT
# ============================================================

def evaluate_workspace_point(
    model,
    data,
    leg,
    W_p_home,
    dx,
    dy
):
    W_p_target = (
        W_p_home.copy()
    )


    W_p_target[0] += (
        dx
    )


    W_p_target[1] += (
        dy
    )


    # Flat-ground foothold:
    #
    #     z_des = z_home
    #
    # For the home keyframe this corresponds to the nominal
    # ground-contact height of the IK site.
    W_p_target[2] = (
        W_p_home[2]
    )


    # ========================================================
    # RUN ACTUAL PROJECT IK
    # ========================================================

    try:

        (
            q_solution,
            diagnostics

        ) = solve_numerical_IK_hybrid(

            model=model,

            data=data,

            W_p_swing_des=W_p_target,

            swing_foot=leg,

            pitch_swing_des=SWING_PITCH_DES,

            sigma_threshold=SIGMA_THRESHOLD,

            alpha_newton=ALPHA_NEWTON,

            eta_gradient=ETA_GRADIENT,

            tolerance=IK_TOLERANCE,

            orientation_tolerance=(
                IK_ORIENTATION_TOLERANCE
            ),

            max_iterations=(
                IK_MAX_ITERATIONS
            )
        )


    except Exception as exc:

        return {

            "leg": leg,

            "dx": float(dx),
            "dy": float(dy),

            "target_x":
                float(
                    W_p_target[0]
                ),

            "target_y":
                float(
                    W_p_target[1]
                ),

            "target_z":
                float(
                    W_p_target[2]
                ),

            "ik_converged":
                False,

            "raw_reachable":
                False,

            "safe_reachable":
                False,

            "iterations":
                -1,

            "position_error_norm":
                np.inf,

            "orientation_error_abs":
                np.inf,

            "minimum_joint_margin_deg":
                -np.inf,

            "q_solution":
                np.full(
                    5,
                    np.nan,
                    dtype=float
                ),

            "joint_margin_deg":
                np.full(
                    5,
                    np.nan,
                    dtype=float
                ),

            "error_message":
                str(
                    exc
                ),
        }


    # ========================================================
    # IK DIAGNOSTICS
    # ========================================================

    ik_converged = bool(
        diagnostics[
            "converged"
        ]
    )


    q_solution = np.asarray(
        q_solution,
        dtype=float
    ).reshape(
        5
    )


    # ========================================================
    # INDEPENDENT MUJOCO FK VERIFICATION
    # ========================================================

    verification = verify_ik_solution(

        model=model,

        data=data,

        leg=leg,

        q_solution=q_solution,

        W_p_target=W_p_target,

        pitch_target=SWING_PITCH_DES
    )


    position_error_norm = float(
        verification[
            "position_error_norm"
        ]
    )


    orientation_error_abs = float(
        verification[
            "pitch_error_abs"
        ]
    )


    # ========================================================
    # JOINT LIMIT MARGIN
    # ========================================================

    joint_info = (
        get_leg_joint_information(
            model,
            leg
        )
    )


    (
        joint_margin,
        minimum_joint_margin

    ) = compute_joint_margin(

        q=q_solution,

        q_min=joint_info[
            "q_min"
        ],

        q_max=joint_info[
            "q_max"
        ]
    )


    joint_margin_deg = np.rad2deg(
        joint_margin
    )


    minimum_joint_margin_deg = float(
        np.rad2deg(
            minimum_joint_margin
        )
    )


    # ========================================================
    # RAW REACHABILITY
    #
    # Require:
    #
    #   1. IK convergence
    #   2. independent FK position check
    #   3. independent pitch check
    # ========================================================

    raw_reachable = bool(

        ik_converged

        and

        position_error_norm
        <=
        IK_TOLERANCE

        and

        orientation_error_abs
        <=
        IK_ORIENTATION_TOLERANCE
    )


    # ========================================================
    # SAFE REACHABILITY
    #
    # Additional joint-limit margin.
    # ========================================================

    safe_reachable = bool(

        raw_reachable

        and

        minimum_joint_margin
        >=
        SAFE_JOINT_MARGIN_RAD
    )


    return {

        "leg": leg,

        "dx": float(dx),
        "dy": float(dy),

        "target_x":
            float(
                W_p_target[0]
            ),

        "target_y":
            float(
                W_p_target[1]
            ),

        "target_z":
            float(
                W_p_target[2]
            ),

        "ik_converged":
            ik_converged,

        "raw_reachable":
            raw_reachable,

        "safe_reachable":
            safe_reachable,

        "iterations":
            int(
                diagnostics[
                    "iterations"
                ]
            ),

        "position_error_norm":
            position_error_norm,

        "orientation_error_abs":
            orientation_error_abs,

        "minimum_joint_margin_deg":
            minimum_joint_margin_deg,

        "q_solution":
            q_solution.copy(),

        "joint_margin_deg":
            joint_margin_deg.copy(),

        "error_message":
            "",
    }


# ============================================================
# SCAN ONE LEG
# ============================================================

def scan_leg_workspace(
    model,
    data,
    leg,
    dx_values,
    dy_values
):
    reset_to_home(
        model,
        data
    )


    W_p_home = (
        get_home_foot_position(
            model,
            data,
            leg
        )
    )


    ny = len(
        dy_values
    )

    nx = len(
        dx_values
    )


    raw_mask = np.zeros(
        (
            ny,
            nx
        ),
        dtype=bool
    )


    safe_mask = np.zeros(
        (
            ny,
            nx
        ),
        dtype=bool
    )


    results = []


    print("\n")
    print("=" * 70)
    print(
        f"SCANNING LEG {leg}"
    )
    print("=" * 70)


    print(
        "\nHome foot position ="
    )

    print(
        W_p_home
    )


    print(
        "\nGrid:"
    )

    print(
        "dx =",
        dx_values[0],
        "to",
        dx_values[-1],
        "m"
    )

    print(
        "dy =",
        dy_values[0],
        "to",
        dy_values[-1],
        "m"
    )

    print(
        "number of points =",
        nx * ny
    )


    for iy, dy in enumerate(
        dy_values
    ):

        for ix, dx in enumerate(
            dx_values
        ):

            result = evaluate_workspace_point(

                model=model,

                data=data,

                leg=leg,

                W_p_home=W_p_home,

                dx=float(
                    dx
                ),

                dy=float(
                    dy
                )
            )


            results.append(
                result
            )


            raw_mask[
                iy,
                ix
            ] = result[
                "raw_reachable"
            ]


            safe_mask[
                iy,
                ix
            ] = result[
                "safe_reachable"
            ]


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            iy == 0
            or
            iy == ny - 1
            or
            (
                iy + 1
            ) % 5 == 0
        ):

            print(
                f"row {iy + 1:3d}/{ny:3d} completed"
            )


    return {

        "leg":
            leg,

        "W_p_home":
            W_p_home,

        "results":
            results,

        "raw_mask":
            raw_mask,

        "safe_mask":
            safe_mask,
    }


# ============================================================
# POINTWISE BOUNDS
#
# NOTE:
#
# These min/max values only describe extrema of feasible
# points.
#
# They do NOT guarantee every point inside the resulting
# rectangle is feasible.
# ============================================================

def compute_pointwise_bounds(
    mask,
    dx_values,
    dy_values
):
    indices = np.argwhere(
        mask
    )


    if indices.size == 0:

        return None


    iy = (
        indices[
            :,
            0
        ]
    )


    ix = (
        indices[
            :,
            1
        ]
    )


    return {

        "dx_min":
            float(
                np.min(
                    dx_values[
                        ix
                    ]
                )
            ),

        "dx_max":
            float(
                np.max(
                    dx_values[
                        ix
                    ]
                )
            ),

        "dy_min":
            float(
                np.min(
                    dy_values[
                        iy
                    ]
                )
            ),

        "dy_max":
            float(
                np.max(
                    dy_values[
                        iy
                    ]
                )
            ),

        "point_count":
            int(
                len(
                    indices
                )
            ),
    }


# ============================================================
# RECTANGLE SUM USING PREFIX MATRIX
# ============================================================

def rectangle_sum(
    prefix,
    y0,
    y1,
    x0,
    x1
):
    return (

        prefix[
            y1 + 1,
            x1 + 1
        ]

        -

        prefix[
            y0,
            x1 + 1
        ]

        -

        prefix[
            y1 + 1,
            x0
        ]

        +

        prefix[
            y0,
            x0
        ]
    )


# ============================================================
# FIND LARGEST GUARANTEED SAFE RECTANGLE CONTAINING HOME
#
# Every grid point inside this rectangle must be safe.
#
# This is much more useful for foothold bounds than simply
# taking independent min/max values.
# ============================================================

def find_largest_safe_rectangle_containing_home(
    safe_mask,
    dx_values,
    dy_values
):
    ny, nx = (
        safe_mask.shape
    )


    ix_home = int(
        np.argmin(
            np.abs(
                dx_values
            )
        )
    )


    iy_home = int(
        np.argmin(
            np.abs(
                dy_values
            )
        )
    )


    if not safe_mask[
        iy_home,
        ix_home
    ]:

        return None


    unsafe = (
        ~safe_mask
    ).astype(
        int
    )


    prefix = np.pad(
        unsafe,
        (
            (1, 0),
            (1, 0)
        ),
        mode="constant"
    )


    prefix = np.cumsum(
        np.cumsum(
            prefix,
            axis=0
        ),
        axis=1
    )


    best = None

    best_point_count = -1

    best_area = -1.0


    for x0 in range(
        0,
        ix_home + 1
    ):

        for x1 in range(
            ix_home,
            nx
        ):

            for y0 in range(
                0,
                iy_home + 1
            ):

                for y1 in range(
                    iy_home,
                    ny
                ):

                    unsafe_count = rectangle_sum(

                        prefix=prefix,

                        y0=y0,
                        y1=y1,

                        x0=x0,
                        x1=x1
                    )


                    if unsafe_count != 0:

                        continue


                    point_count = (
                        (
                            x1
                            -
                            x0
                            +
                            1
                        )
                        *
                        (
                            y1
                            -
                            y0
                            +
                            1
                        )
                    )


                    span_x = float(
                        dx_values[
                            x1
                        ]
                        -
                        dx_values[
                            x0
                        ]
                    )


                    span_y = float(
                        dy_values[
                            y1
                        ]
                        -
                        dy_values[
                            y0
                        ]
                    )


                    area = (
                        span_x
                        *
                        span_y
                    )


                    if (
                        point_count
                        >
                        best_point_count
                    ):

                        better = True


                    elif (
                        point_count
                        ==
                        best_point_count
                        and
                        area
                        >
                        best_area
                    ):

                        better = True


                    else:

                        better = False


                    if better:

                        best_point_count = (
                            point_count
                        )

                        best_area = (
                            area
                        )


                        best = {

                            "dx_min":
                                float(
                                    dx_values[
                                        x0
                                    ]
                                ),

                            "dx_max":
                                float(
                                    dx_values[
                                        x1
                                    ]
                                ),

                            "dy_min":
                                float(
                                    dy_values[
                                        y0
                                    ]
                                ),

                            "dy_max":
                                float(
                                    dy_values[
                                        y1
                                    ]
                                ),

                            "point_count":
                                int(
                                    point_count
                                ),

                            "span_x":
                                span_x,

                            "span_y":
                                span_y,

                            "area":
                                area,
                        }


    return best


# ============================================================
# SAFE INTERVAL AT dy = 0
#
# Useful for forward/backward step-length interpretation.
# ============================================================

def compute_safe_dx_interval_at_dy_zero(
    safe_mask,
    dx_values,
    dy_values
):
    iy = int(
        np.argmin(
            np.abs(
                dy_values
            )
        )
    )


    row = safe_mask[
        iy,
        :
    ]


    safe_indices = np.where(
        row
    )[0]


    if safe_indices.size == 0:

        return None


    return {

        "dy":
            float(
                dy_values[
                    iy
                ]
            ),

        "dx_min":
            float(
                dx_values[
                    safe_indices[
                        0
                    ]
                ]
            ),

        "dx_max":
            float(
                dx_values[
                    safe_indices[
                        -1
                    ]
                ]
            ),
    }


# ============================================================
# SAFE INTERVAL AT dx = 0
#
# Useful for lateral foothold interpretation.
# ============================================================

def compute_safe_dy_interval_at_dx_zero(
    safe_mask,
    dx_values,
    dy_values
):
    ix = int(
        np.argmin(
            np.abs(
                dx_values
            )
        )
    )


    column = safe_mask[
        :,
        ix
    ]


    safe_indices = np.where(
        column
    )[0]


    if safe_indices.size == 0:

        return None


    return {

        "dx":
            float(
                dx_values[
                    ix
                ]
            ),

        "dy_min":
            float(
                dy_values[
                    safe_indices[
                        0
                    ]
                ]
            ),

        "dy_max":
            float(
                dy_values[
                    safe_indices[
                        -1
                    ]
                ]
            ),
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_leg_csv(
    leg_result
):
    leg = (
        leg_result[
            "leg"
        ]
    )


    csv_path = (
        OUTPUT_DIR
        /
        f"workspace_{leg}.csv"
    )


    fieldnames = [

        "leg",

        "dx",
        "dy",

        "target_x",
        "target_y",
        "target_z",

        "ik_converged",

        "raw_reachable",

        "safe_reachable",

        "iterations",

        "position_error_norm",

        "orientation_error_abs",

        "minimum_joint_margin_deg",

        "q_hip_yaw_rad",
        "q_hip_roll_rad",
        "q_hip_pitch_rad",
        "q_knee_rad",
        "q_ankle_rad",

        "q_hip_yaw_deg",
        "q_hip_roll_deg",
        "q_hip_pitch_deg",
        "q_knee_deg",
        "q_ankle_deg",

        "margin_hip_yaw_deg",
        "margin_hip_roll_deg",
        "margin_hip_pitch_deg",
        "margin_knee_deg",
        "margin_ankle_deg",

        "error_message",
    ]


    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )


        writer.writeheader()


        for result in leg_result[
            "results"
        ]:

            q = (
                result[
                    "q_solution"
                ]
            )


            q_deg = np.rad2deg(
                q
            )


            margin_deg = (
                result[
                    "joint_margin_deg"
                ]
            )


            row = {

                "leg":
                    result[
                        "leg"
                    ],

                "dx":
                    result[
                        "dx"
                    ],

                "dy":
                    result[
                        "dy"
                    ],

                "target_x":
                    result[
                        "target_x"
                    ],

                "target_y":
                    result[
                        "target_y"
                    ],

                "target_z":
                    result[
                        "target_z"
                    ],

                "ik_converged":
                    result[
                        "ik_converged"
                    ],

                "raw_reachable":
                    result[
                        "raw_reachable"
                    ],

                "safe_reachable":
                    result[
                        "safe_reachable"
                    ],

                "iterations":
                    result[
                        "iterations"
                    ],

                "position_error_norm":
                    result[
                        "position_error_norm"
                    ],

                "orientation_error_abs":
                    result[
                        "orientation_error_abs"
                    ],

                "minimum_joint_margin_deg":
                    result[
                        "minimum_joint_margin_deg"
                    ],

                "q_hip_yaw_rad":
                    q[0],

                "q_hip_roll_rad":
                    q[1],

                "q_hip_pitch_rad":
                    q[2],

                "q_knee_rad":
                    q[3],

                "q_ankle_rad":
                    q[4],

                "q_hip_yaw_deg":
                    q_deg[0],

                "q_hip_roll_deg":
                    q_deg[1],

                "q_hip_pitch_deg":
                    q_deg[2],

                "q_knee_deg":
                    q_deg[3],

                "q_ankle_deg":
                    q_deg[4],

                "margin_hip_yaw_deg":
                    margin_deg[0],

                "margin_hip_roll_deg":
                    margin_deg[1],

                "margin_hip_pitch_deg":
                    margin_deg[2],

                "margin_knee_deg":
                    margin_deg[3],

                "margin_ankle_deg":
                    margin_deg[4],

                "error_message":
                    result[
                        "error_message"
                    ],
            }


            writer.writerow(
                row
            )


    return csv_path


# ============================================================
# PLOT WORKSPACE
# ============================================================

def plot_leg_workspace(
    leg_result,
    dx_values,
    dy_values,
    summary
):
    leg = (
        leg_result[
            "leg"
        ]
    )


    raw_mask = (
        leg_result[
            "raw_mask"
        ]
    )


    safe_mask = (
        leg_result[
            "safe_mask"
        ]
    )


    # --------------------------------------------------------
    # Classification:
    #
    # 0 = unreachable
    # 1 = IK reachable
    # 2 = safe reachable
    # --------------------------------------------------------

    classification = np.zeros(
        raw_mask.shape,
        dtype=int
    )


    classification[
        raw_mask
    ] = 1


    classification[
        safe_mask
    ] = 2


    fig = plt.figure(
        figsize=(
            9,
            7
        )
    )


    ax = fig.add_subplot(
        111
    )


    image = ax.imshow(

        classification,

        origin="lower",

        extent=[
            dx_values[0],
            dx_values[-1],
            dy_values[0],
            dy_values[-1],
        ],

        aspect="equal",

        interpolation="nearest"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax
    )


    colorbar.set_ticks(
        [
            0,
            1,
            2
        ]
    )


    colorbar.set_ticklabels(
        [
            "Unreachable",
            "IK reachable",
            "Safe"
        ]
    )


    # --------------------------------------------------------
    # Home reference
    # --------------------------------------------------------

    ax.plot(
        0.0,
        0.0,
        marker="x",
        markersize=10
    )


    ax.axvline(
        0.0,
        linewidth=1.0
    )


    ax.axhline(
        0.0,
        linewidth=1.0
    )


    # --------------------------------------------------------
    # Guaranteed safe rectangle
    # --------------------------------------------------------

    rectangle = summary[
        "safe_rectangle"
    ]


    if rectangle is not None:

        x0 = rectangle[
            "dx_min"
        ]

        x1 = rectangle[
            "dx_max"
        ]

        y0 = rectangle[
            "dy_min"
        ]

        y1 = rectangle[
            "dy_max"
        ]


        ax.plot(
            [
                x0,
                x1,
                x1,
                x0,
                x0
            ],
            [
                y0,
                y0,
                y1,
                y1,
                y0
            ],
            linewidth=2.0
        )


    ax.set_xlabel(
        r"$\Delta x$ [m]"
    )


    ax.set_ylabel(
        r"$\Delta y$ [m]"
    )


    ax.set_title(
        f"Leg {leg} - Home-based kinematic workspace"
    )


    ax.grid(
        True
    )


    fig.tight_layout()


    png_path = (
        OUTPUT_DIR
        /
        f"workspace_{leg}.png"
    )


    fig.savefig(
        png_path,
        dpi=200
    )


    plt.close(
        fig
    )


    return png_path


# ============================================================
# BUILD LEG SUMMARY
# ============================================================

def build_leg_summary(
    leg_result,
    dx_values,
    dy_values
):
    raw_mask = (
        leg_result[
            "raw_mask"
        ]
    )


    safe_mask = (
        leg_result[
            "safe_mask"
        ]
    )


    raw_bounds = (
        compute_pointwise_bounds(

            mask=raw_mask,

            dx_values=dx_values,

            dy_values=dy_values
        )
    )


    safe_bounds = (
        compute_pointwise_bounds(

            mask=safe_mask,

            dx_values=dx_values,

            dy_values=dy_values
        )
    )


    safe_rectangle = (
        find_largest_safe_rectangle_containing_home(

            safe_mask=safe_mask,

            dx_values=dx_values,

            dy_values=dy_values
        )
    )


    safe_dx_at_zero_dy = (
        compute_safe_dx_interval_at_dy_zero(

            safe_mask=safe_mask,

            dx_values=dx_values,

            dy_values=dy_values
        )
    )


    safe_dy_at_zero_dx = (
        compute_safe_dy_interval_at_dx_zero(

            safe_mask=safe_mask,

            dx_values=dx_values,

            dy_values=dy_values
        )
    )


    total_points = int(
        safe_mask.size
    )


    raw_count = int(
        np.sum(
            raw_mask
        )
    )


    safe_count = int(
        np.sum(
            safe_mask
        )
    )


    return {

        "leg":
            leg_result[
                "leg"
            ],

        "home_position":
            leg_result[
                "W_p_home"
            ].tolist(),

        "total_points":
            total_points,

        "raw_reachable_points":
            raw_count,

        "safe_reachable_points":
            safe_count,

        "raw_reachable_fraction":
            (
                raw_count
                /
                total_points
            ),

        "safe_reachable_fraction":
            (
                safe_count
                /
                total_points
            ),

        "raw_pointwise_bounds":
            raw_bounds,

        "safe_pointwise_bounds":
            safe_bounds,

        "safe_rectangle":
            safe_rectangle,

        "safe_dx_interval_at_dy_zero":
            safe_dx_at_zero_dy,

        "safe_dy_interval_at_dx_zero":
            safe_dy_at_zero_dx,
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_leg_summary(
    summary
):
    leg = (
        summary[
            "leg"
        ]
    )


    print("\n")
    print("=" * 70)
    print(
        f"WORKSPACE SUMMARY - LEG {leg}"
    )
    print("=" * 70)


    print(
        "\nHome position ="
    )

    print(
        np.asarray(
            summary[
                "home_position"
            ]
        )
    )


    print(
        "\nTotal points =",
        summary[
            "total_points"
        ]
    )


    print(
        "Raw reachable points =",
        summary[
            "raw_reachable_points"
        ]
    )


    print(
        "Safe reachable points =",
        summary[
            "safe_reachable_points"
        ]
    )


    print(
        "\nRaw pointwise bounds ="
    )

    print(
        summary[
            "raw_pointwise_bounds"
        ]
    )


    print(
        "\nSafe pointwise bounds ="
    )

    print(
        summary[
            "safe_pointwise_bounds"
        ]
    )


    print(
        "\nGuaranteed safe rectangle containing home ="
    )

    print(
        summary[
            "safe_rectangle"
        ]
    )


    print(
        "\nSafe dx interval at dy = 0 ="
    )

    print(
        summary[
            "safe_dx_interval_at_dy_zero"
        ]
    )


    print(
        "\nSafe dy interval at dx = 0 ="
    )

    print(
        summary[
            "safe_dy_interval_at_dx_zero"
        ]
    )


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

def save_summary_json(
    summaries
):
    output = {

        "model":
            str(
                XML_PATH
            ),

        "home_key":
            HOME_KEY_NAME,

        "scan_parameters": {

            "dx_min":
                DX_MIN,

            "dx_max":
                DX_MAX,

            "dy_min":
                DY_MIN,

            "dy_max":
                DY_MAX,

            "grid_step":
                GRID_STEP,

            "safe_joint_margin_deg":
                SAFE_JOINT_MARGIN_DEG,

            "ik_tolerance":
                IK_TOLERANCE,

            "ik_orientation_tolerance":
                IK_ORIENTATION_TOLERANCE,
        },

        "legs":
            summaries,
    }


    json_path = (
        OUTPUT_DIR
        /
        "workspace_bounds.json"
    )


    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


    return json_path


# ============================================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================================

def save_summary_txt(
    model,
    summaries
):
    txt_path = (
        OUTPUT_DIR
        /
        "workspace_summary.txt"
    )


    with txt_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70
            +
            "\n"
        )

        file.write(
            "HOME-BASED KINEMATIC WORKSPACE SCAN\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )


        file.write(
            f"XML: {XML_PATH}\n"
        )

        file.write(
            f"Home key: {HOME_KEY_NAME}\n\n"
        )


        file.write(
            f"dx search: [{DX_MIN}, {DX_MAX}] m\n"
        )

        file.write(
            f"dy search: [{DY_MIN}, {DY_MAX}] m\n"
        )

        file.write(
            f"grid step: {GRID_STEP} m\n"
        )

        file.write(
            f"safe joint margin: "
            f"{SAFE_JOINT_MARGIN_DEG} deg\n\n"
        )


        file.write(
            "Joint ranges:\n"
        )


        for leg in (
            "L",
            "R"
        ):

            joint_info = (
                get_leg_joint_information(
                    model,
                    leg
                )
            )


            file.write(
                f"\nLeg {leg}\n"
            )


            for (
                joint_name,
                q_min,
                q_max

            ) in zip(

                joint_info[
                    "joint_names"
                ],

                joint_info[
                    "q_min"
                ],

                joint_info[
                    "q_max"
                ]
            ):

                file.write(

                    f"  {joint_name:15s}: "

                    f"["
                    f"{np.rad2deg(q_min):8.3f}, "
                    f"{np.rad2deg(q_max):8.3f}"
                    f"] deg\n"
                )


        for summary in summaries:

            file.write(
                "\n\n"
                +
                "=" * 70
                +
                "\n"
            )

            file.write(
                f"LEG {summary['leg']}\n"
            )

            file.write(
                "=" * 70
                +
                "\n"
            )


            file.write(
                "\nHome position:\n"
            )

            file.write(
                str(
                    summary[
                        "home_position"
                    ]
                )
                +
                "\n"
            )


            file.write(
                "\nRaw pointwise bounds:\n"
            )

            file.write(
                str(
                    summary[
                        "raw_pointwise_bounds"
                    ]
                )
                +
                "\n"
            )


            file.write(
                "\nSafe pointwise bounds:\n"
            )

            file.write(
                str(
                    summary[
                        "safe_pointwise_bounds"
                    ]
                )
                +
                "\n"
            )


            file.write(
                "\nGuaranteed safe rectangle:\n"
            )

            file.write(
                str(
                    summary[
                        "safe_rectangle"
                    ]
                )
                +
                "\n"
            )


            file.write(
                "\nSafe dx interval at dy=0:\n"
            )

            file.write(
                str(
                    summary[
                        "safe_dx_interval_at_dy_zero"
                    ]
                )
                +
                "\n"
            )


            file.write(
                "\nSafe dy interval at dx=0:\n"
            )

            file.write(
                str(
                    summary[
                        "safe_dy_interval_at_dx_zero"
                    ]
                )
                +
                "\n"
            )


    return txt_path


# ============================================================
# FIND RESULT CLOSEST TO REQUESTED OFFSET
# ============================================================

def find_closest_safe_result(
    leg_result,
    dx_target,
    dy_target
):
    safe_results = [

        result

        for result in leg_result[
            "results"
        ]

        if result[
            "safe_reachable"
        ]
    ]


    if len(
        safe_results
    ) == 0:

        return None


    best = min(

        safe_results,

        key=lambda result: (

            (
                result[
                    "dx"
                ]
                -
                dx_target
            ) ** 2

            +

            (
                result[
                    "dy"
                ]
                -
                dy_target
            ) ** 2
        )
    )


    return best


# ============================================================
# OPTIONAL VIEWER
# ============================================================

def visualize_safe_rectangle(
    model,
    data,
    leg_result,
    summary
):
    if not ENABLE_VIEWER:

        return


    safe_rectangle = (
        summary[
            "safe_rectangle"
        ]
    )


    if safe_rectangle is None:

        print(
            f"No safe rectangle for leg "
            f"{leg_result['leg']}."
        )

        return


    import mujoco.viewer


    leg = (
        leg_result[
            "leg"
        ]
    )


    targets = [

        (
            0.0,
            0.0,
            "HOME"
        ),

        (
            safe_rectangle[
                "dx_min"
            ],
            safe_rectangle[
                "dy_min"
            ],
            "CORNER 1"
        ),

        (
            safe_rectangle[
                "dx_max"
            ],
            safe_rectangle[
                "dy_min"
            ],
            "CORNER 2"
        ),

        (
            safe_rectangle[
                "dx_max"
            ],
            safe_rectangle[
                "dy_max"
            ],
            "CORNER 3"
        ),

        (
            safe_rectangle[
                "dx_min"
            ],
            safe_rectangle[
                "dy_max"
            ],
            "CORNER 4"
        ),
    ]


    joint_info = (
        get_leg_joint_information(
            model,
            leg
        )
    )


    qpos_addresses = (
        joint_info[
            "qpos_addresses"
        ]
    )


    reset_to_home(
        model,
        data
    )


    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        for (
            dx_target,
            dy_target,
            label

        ) in targets:

            result = (
                find_closest_safe_result(

                    leg_result=leg_result,

                    dx_target=dx_target,

                    dy_target=dy_target
                )
            )


            if result is None:

                continue


            reset_to_home(
                model,
                data
            )


            data.qpos[
                qpos_addresses
            ] = result[
                "q_solution"
            ]


            mujoco.mj_forward(
                model,
                data
            )


            viewer.sync()


            print(
                f"Visualizing leg {leg}: "
                f"{label}, "
                f"dx={result['dx']:.4f}, "
                f"dy={result['dy']:.4f}"
            )


            time.sleep(
                VIEW_HOLD_TIME
            )


    reset_to_home(
        model,
        data
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HOME-BASED KINEMATIC WORKSPACE SCAN")
    print("=" * 70)


    print(
        "\nXML ="
    )

    print(
        XML_PATH
    )


    # ========================================================
    # 1. OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 2. LOAD ACTUAL MUJOCO MODEL
    # ========================================================

    (
        model,
        data

    ) = load_home_model()


    # ========================================================
    # 3. GRID
    # ========================================================

    dx_values = build_grid_axis(

        minimum=DX_MIN,

        maximum=DX_MAX,

        step=GRID_STEP
    )


    dy_values = build_grid_axis(

        minimum=DY_MIN,

        maximum=DY_MAX,

        step=GRID_STEP
    )


    print(
        "\nNumber of dx points =",
        len(
            dx_values
        )
    )


    print(
        "Number of dy points =",
        len(
            dy_values
        )
    )


    print(
        "Points per leg =",
        len(
            dx_values
        )
        *
        len(
            dy_values
        )
    )


    # ========================================================
    # 4. HOME POSITIONS
    # ========================================================

    reset_to_home(
        model,
        data
    )


    W_p_L_home = (
        get_home_foot_position(
            model,
            data,
            "L"
        )
    )


    W_p_R_home = (
        get_home_foot_position(
            model,
            data,
            "R"
        )
    )


    print(
        "\nW_p_L6_IK_home ="
    )

    print(
        W_p_L_home
    )


    print(
        "\nW_p_R6_IK_home ="
    )

    print(
        W_p_R_home
    )


    # ========================================================
    # 5. SCAN BOTH LEGS
    # ========================================================

    leg_results = []


    for leg in (
        "L",
        "R"
    ):

        result = scan_leg_workspace(

            model=model,

            data=data,

            leg=leg,

            dx_values=dx_values,

            dy_values=dy_values
        )


        leg_results.append(
            result
        )


    # ========================================================
    # 6. BUILD SUMMARIES
    # ========================================================

    summaries = []


    for leg_result in leg_results:

        summary = build_leg_summary(

            leg_result=leg_result,

            dx_values=dx_values,

            dy_values=dy_values
        )


        summaries.append(
            summary
        )


        print_leg_summary(
            summary
        )


    # ========================================================
    # 7. SAVE CSV
    # ========================================================

    csv_paths = []


    for leg_result in leg_results:

        csv_path = save_leg_csv(
            leg_result
        )


        csv_paths.append(
            csv_path
        )


    # ========================================================
    # 8. SAVE FIGURES
    # ========================================================

    png_paths = []


    for (
        leg_result,
        summary

    ) in zip(

        leg_results,
        summaries
    ):

        png_path = plot_leg_workspace(

            leg_result=leg_result,

            dx_values=dx_values,

            dy_values=dy_values,

            summary=summary
        )


        png_paths.append(
            png_path
        )


    # ========================================================
    # 9. SAVE JSON
    # ========================================================

    json_path = save_summary_json(
        summaries
    )


    # ========================================================
    # 10. SAVE TXT
    # ========================================================

    txt_path = save_summary_txt(

        model=model,

        summaries=summaries
    )


    # ========================================================
    # 11. OPTIONAL VISUALIZATION
    # ========================================================

    if ENABLE_VIEWER:

        for (
            leg_result,
            summary

        ) in zip(

            leg_results,
            summaries
        ):

            visualize_safe_rectangle(

                model=model,

                data=data,

                leg_result=leg_result,

                summary=summary
            )


    # ========================================================
    # 12. FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WORKSPACE SCAN COMPLETE")
    print("=" * 70)


    print(
        "\nOutput directory ="
    )

    print(
        OUTPUT_DIR
    )


    print(
        "\nCSV files:"
    )


    for path in csv_paths:

        print(
            "  ",
            path
        )


    print(
        "\nPNG files:"
    )


    for path in png_paths:

        print(
            "  ",
            path
        )


    print(
        "\nJSON summary:"
    )

    print(
        "  ",
        json_path
    )


    print(
        "\nTXT summary:"
    )

    print(
        "  ",
        txt_path
    )


    print("\n")
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)


    print(
        "\nUse 'safe_rectangle' from workspace_bounds.json "
        "as the conservative candidate bound for the "
        "future foothold generator."
    )


    print(
        "\nDo NOT use raw_pointwise_bounds directly as "
        "a rectangular foothold constraint, because "
        "not every point inside those independent extrema "
        "is necessarily reachable."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()