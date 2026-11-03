from pathlib import Path
from datetime import datetime
import csv
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH
# ============================================================

THIS_FILE = Path(__file__).resolve()

SIMULATION_DIR = THIS_FILE.parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent

STATE_DIR = MPC_SRBD_DIR / "State"
SRBD_DIR = MPC_SRBD_DIR / "SRBD"
QP_DIR = MPC_SRBD_DIR / "QP"
REF_DIR = MPC_SRBD_DIR / "Ref_generator"
FOOTSTEP_DIR = MPC_SRBD_DIR / "Footstep_Planner"
TRAJECTORY_DIR = MPC_SRBD_DIR / "Trajectory_Planner"
KINEMATICS_DIR = MPC_SRBD_DIR / "Kinematics"
ACTUATOR_DIR = MPC_SRBD_DIR / "Actuator"


for directory in (
    MPC_SRBD_DIR,
    SIMULATION_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    REF_DIR,
    FOOTSTEP_DIR,
    TRAJECTORY_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
):

    directory_string = str(
        directory
    )

    if directory_string not in sys.path:

        sys.path.insert(
            0,
            directory_string,
        )


# ============================================================
# XML
# ============================================================

XML_PATH = (
    REPO_ROOT
    / "xmls"
    / "scene_flat_terrian_v3.xml"
)


# ============================================================
# PROJECT MODULES
# ============================================================

from simulation_setup import (
    initialize_simulation,
)

from get_current_state_mpc import (
    get_current_state_mpc,
)

from get_contact_state import (
    get_contact_state,
)

from build_SRBD_model import (
    build_SRBD_model,
)

from build_and_solve_QP import (
    build_and_solve_QP,
)

from reference_generator import (
    ReferenceGenerator,
)

from gait_scheduler import (
    GaitScheduler,
)

from raibert_foot_placement import (
    compute_raibert_foothold,
)

from bezier_swing_trajectory import (
    evaluate_bezier_swing_trajectory,
)

from numerical_IK_hybrid import (
    solve_numerical_IK_hybrid,
)

from numerical_IK_params import (
    HIP_YAW_FIXED,
    SIGMA_THRESHOLD,
    ALPHA_NEWTON,
    ETA_GRADIENT,
    IK_TOLERANCE,
    IK_MAX_ITERATIONS,
)

from swing_joint_PD import (
    compute_swing_joint_PD,
)

from swing_joint_PD_params import (
    KP_SWING,
    KD_SWING,
)

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench,
)

from actuator_mapping import (
    ACTUATOR_NAMES,
    get_actuator_ids,
    apply_joint_torque,
)


# ============================================================
# DISPLAY
# ============================================================

np.set_printoptions(
    precision=7,
    suppress=True,
    linewidth=240,
)


# ============================================================
# BASIC CONSTANTS
# ============================================================

LEFT = 0
RIGHT = 1

LEG_INDEX = {
    "L": LEFT,
    "R": RIGHT,
}


LEG_SLICE = {
    "L": slice(0, 5),
    "R": slice(5, 10),
}


LEG_JOINT_NAMES = {

    "L": [
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee",
        "L_ankle",
    ],

    "R": [
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee",
        "R_ankle",
    ],
}


ANKLE_SITE_NAMES = {
    "L": "L5_ankle",
    "R": "R5_ankle",
}


BASE_BODY_NAME = "base_link"


NX = 13
NU = 12


# ============================================================
# CONTROL / MPC TIMING
# ============================================================

CONTROL_DT = 0.001

MPC_UPDATE_DT = 0.005

MPC_DT = 0.040

MPC_HORIZON = 10

REFERENCE_HORIZON = (
    MPC_HORIZON
    +
    1
)


MPC_UPDATE_DECIMATION = int(
    round(
        MPC_UPDATE_DT
        /
        CONTROL_DT
    )
)


if not np.isclose(
    MPC_UPDATE_DECIMATION
    *
    CONTROL_DT,
    MPC_UPDATE_DT,
):

    raise ValueError(
        "MPC_UPDATE_DT must be an integer multiple "
        "of CONTROL_DT."
    )


# ============================================================
# GAIT
#
# Dynamic walking test.
#
# SSP:
#     6 * 0.04 = 0.24 s
#
# DSP:
#     5 * 0.04 = 0.20 s
#
#
# START:
#
# startup DS
# ->
# DSP2
# ->
# SSP_L / RIGHT swing
# ->
# DSP1
# ->
# SSP_R / LEFT swing
# ->
# DSP2
# ->
# repeat
#
# IMPORTANT:
#
# We are NOT trying to create static SSP.
# SSP is allowed to be dynamic.
# ============================================================

SSP_DURATIONS = (
    8,
    8,
)

DSP_DURATIONS = (
    5,
    5,
)

STARTUP_HOLD_TIME = 0.250

FIRST_SWING_LEG = "R"


# ============================================================
# COMMAND
#
# Start slowly.
#
# Increase later only after first integrated test is stable.
# ============================================================

V_DES_BODY_TARGET = np.array(
    [
        0.025,
        0.00,
        0.00,
    ],
    dtype=float,
)

YAW_RATE_DES_TARGET = 0.0


# ============================================================
# IMPORTANT:
#
# No artificial static lateral CoM hold in this first
# integrated dynamic test.
#
# y reference stays governed by ReferenceGenerator.
#
# MPC is allowed to generate lateral momentum naturally.
# ============================================================

USE_ARTIFICIAL_LATERAL_LOAD_TRANSFER = False


# ============================================================
# SWING
# ============================================================

SWING_HEIGHT = 0.004

CP1_COEF = 0.333

CP2_COEF = 0.666


# ============================================================
# DESIRED JOINT VELOCITY
# ============================================================

Q_DOT_DES_FILTER_TAU = 0.005

Q_DOT_DES_LIMIT = np.array(
    [
        6.0,
        6.0,
        8.0,
        10.0,
        8.0,
    ],
    dtype=float,
)


# ============================================================
# ANKLE COMPENSATION
# ============================================================

SLOPE_PITCH = 0.0


# ============================================================
# EARLY TOUCHDOWN
#
# Important:
#
# At liftoff the foot can remain in physical contact for
# several samples.
#
# Therefore:
#
# 1. the foot must first LOSE contact,
# 2. swing phase must exceed minimum,
# 3. contact must return for N consecutive samples.
#
# Only then:
#
#     SSP -> DSP early
# ============================================================

ENABLE_EARLY_TOUCHDOWN = True

EARLY_TOUCHDOWN_MIN_PHASE = 0.30

EARLY_TOUCHDOWN_DEBOUNCE_STEPS = 2


# ============================================================
# SIMULATION
# ============================================================

SIMULATION_TIME = 2.00


# ============================================================
# QP
# ============================================================

QP_EPS_ABS = 1e-4

QP_EPS_REL = 1e-4

QP_MAX_ITER = 5000

QP_CONSTRAINT_TOLERANCE = 1e-4


# ============================================================
# FALL DETECTION
# ============================================================

MIN_COM_HEIGHT = 0.080

MAX_ABS_ROLL = np.deg2rad(
    60.0
)

MAX_ABS_PITCH = np.deg2rad(
    60.0
)


# ============================================================
# GUI
# ============================================================

USE_GUI = True

VIEWER_SLEEP = 0.0


# ============================================================
# DEBUG PRINT
# ============================================================

STATUS_PRINT_DT = 0.050

STATUS_PRINT_DECIMATION = max(
    1,
    int(
        round(
            STATUS_PRINT_DT
            /
            CONTROL_DT
        )
    ),
)


# ============================================================
# LOG DIRECTORY
# ============================================================

LOG_DIR = (
    SIMULATION_DIR
    /
    "dynamic_walk_logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MUJOCO ID HELPERS
# ============================================================

def get_joint_id(
    model,
    name,
):

    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if object_id < 0:

        raise ValueError(
            f"Joint '{name}' not found."
        )

    return int(
        object_id
    )


def get_site_id(
    model,
    name,
):

    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    if object_id < 0:

        raise ValueError(
            f"Site '{name}' not found."
        )

    return int(
        object_id
    )


def get_body_id(
    model,
    name,
):

    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if object_id < 0:

        raise ValueError(
            f"Body '{name}' not found."
        )

    return int(
        object_id
    )


# ============================================================
# CURRENT LEG JOINT STATE
# ============================================================

def get_leg_joint_state(
    model,
    data,
    leg,
):

    q = np.zeros(
        5,
        dtype=float,
    )

    q_dot = np.zeros(
        5,
        dtype=float,
    )


    for i, joint_name in enumerate(
        LEG_JOINT_NAMES[
            leg
        ]
    ):

        joint_id = get_joint_id(
            model,
            joint_name,
        )

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        dof_address = int(
            model.jnt_dofadr[
                joint_id
            ]
        )


        q[
            i
        ] = data.qpos[
            qpos_address
        ]


        q_dot[
            i
        ] = data.qvel[
            dof_address
        ]


    return (
        q,
        q_dot,
    )


# ============================================================
# ANKLE POSITION
# ============================================================

def get_ankle_position(
    model,
    data,
    leg,
):

    site_id = get_site_id(
        model,
        ANKLE_SITE_NAMES[
            leg
        ],
    )


    return np.asarray(
        data.site_xpos[
            site_id
        ],
        dtype=float,
    ).copy()


# ============================================================
# BASE POSE
# ============================================================

def get_base_pose(
    model,
    data,
):

    body_id = get_body_id(
        model,
        BASE_BODY_NAME,
    )


    W_p_base = np.asarray(
        data.xpos[
            body_id
        ],
        dtype=float,
    ).copy()


    W_R_B = np.asarray(
        data.xmat[
            body_id
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    ).copy()


    return (
        W_p_base,
        W_R_B,
    )


# ============================================================
# RAIBERT ANCHOR OFFSET
# ============================================================

def compute_raibert_anchor_offset_B(
    model,
    data,
    leg,
):

    W_p_ankle = get_ankle_position(
        model,
        data,
        leg,
    )


    (
        W_p_base,
        W_R_B,

    ) = get_base_pose(
        model,
        data,
    )


    B_r_anchor = (
        W_R_B.T
        @
        (
            W_p_ankle
            -
            W_p_base
        )
    )


    B_r_anchor[
        2
    ] = 0.0


    return B_r_anchor


# ============================================================
# CURRENT RAIBERT ANCHOR
# ============================================================

def compute_raibert_anchor(
    model,
    data,
    B_r_anchor,
):

    (
        W_p_base,
        W_R_B,

    ) = get_base_pose(
        model,
        data,
    )


    return (
        W_p_base
        +
        W_R_B
        @
        B_r_anchor
    )


# ============================================================
# ANKLE DESIRED
# ============================================================

def compute_ankle_desired(
    q_actual,
):

    q_actual = np.asarray(
        q_actual,
        dtype=float,
    ).reshape(
        5
    )


    return float(
        -SLOPE_PITCH
        -
        q_actual[
            2
        ]
        -
        q_actual[
            3
        ]
    )


# ============================================================
# JOINT REFERENCE MEMORY
# ============================================================

def create_joint_reference_memory():

    return {

        "q_des_previous":
            None,

        "q_dot_des_filtered":
            np.zeros(
                5,
                dtype=float,
            ),
    }


def reset_joint_reference_memory(
    memory,
):

    memory[
        "q_des_previous"
    ] = None


    memory[
        "q_dot_des_filtered"
    ] = np.zeros(
        5,
        dtype=float,
    )


# ============================================================
# QDOT DES
# ============================================================

def update_q_dot_des(
    q_des,
    memory,
):

    q_des = np.asarray(
        q_des,
        dtype=float,
    ).reshape(
        5
    )


    previous = memory[
        "q_des_previous"
    ]


    if previous is None:

        q_dot_des = np.zeros(
            5,
            dtype=float,
        )


    else:

        q_dot_raw = (
            q_des
            -
            previous
        ) / CONTROL_DT


        q_dot_raw = np.clip(
            q_dot_raw,
            -Q_DOT_DES_LIMIT,
            +Q_DOT_DES_LIMIT,
        )


        alpha = (
            CONTROL_DT
            /
            (
                Q_DOT_DES_FILTER_TAU
                +
                CONTROL_DT
            )
        )


        q_dot_des = (
            (
                1.0
                -
                alpha
            )
            *
            memory[
                "q_dot_des_filtered"
            ]

            +

            alpha
            *
            q_dot_raw
        )


        q_dot_des = np.clip(
            q_dot_des,
            -Q_DOT_DES_LIMIT,
            +Q_DOT_DES_LIMIT,
        )


    memory[
        "q_des_previous"
    ] = q_des.copy()


    memory[
        "q_dot_des_filtered"
    ] = q_dot_des.copy()


    return q_dot_des


# ============================================================
# IK
# ============================================================

def solve_leg_cartesian_target(
    model,
    data,
    leg,
    W_p_ankle_des,
):

    (
        q_ik,
        diagnostics,

    ) = solve_numerical_IK_hybrid(

        model=model,

        data=data,

        W_p_ankle_des=W_p_ankle_des,

        swing_foot=leg,

        sigma_threshold=SIGMA_THRESHOLD,

        alpha_newton=ALPHA_NEWTON,

        eta_gradient=ETA_GRADIENT,

        tolerance=IK_TOLERANCE,

        max_iterations=IK_MAX_ITERATIONS,

        hip_yaw_fixed=HIP_YAW_FIXED,
    )


    if q_ik is None:

        raise RuntimeError(
            f"IK returned None for leg {leg}."
        )


    (
        q_actual,
        q_dot_actual,

    ) = get_leg_joint_state(
        model,
        data,
        leg,
    )


    q_ankle_des = compute_ankle_desired(
        q_actual
    )


    q_des = np.array(
        [
            q_ik[
                0
            ],
            q_ik[
                1
            ],
            q_ik[
                2
            ],
            q_ik[
                3
            ],
            q_ankle_des,
        ],
        dtype=float,
    )


    return {

        "q_des":
            q_des,

        "q_actual":
            q_actual,

        "q_dot_actual":
            q_dot_actual,

        "diagnostics":
            diagnostics,
    }


# ============================================================
# SWING CONTEXT
# ============================================================

def create_swing_context():

    return {

        "active":
            False,

        "W_p_liftoff":
            None,

        "foothold_z":
            None,

        "had_clearance":
            False,

        "contact_debounce":
            0,

        "last_target":
            None,

        "last_q_des":
            None,

        "last_q_dot_des":
            None,

        "swing_samples":
            0,

        "sum_cart_error_sq":
            0.0,

        "max_cart_error":
            0.0,

        "completed_swings":
            0,

        "early_touchdowns":
            0,

        "planned_touchdowns":
            0,

        "missed_planned_touchdowns":
            0,
    }


# ============================================================
# DYNAMIC SWING TARGET
#
# IMPORTANT:
#
# Raibert receives ACTUAL CoM velocity from x_k.
#
# It no longer receives prescribed-base velocity.
# ============================================================

def compute_dynamic_swing_target(
    model,
    data,
    leg,
    swing_phase,
    T_swing,
    context,
    B_r_anchor,
    x_current,
    ref_output,
):

    s = float(
        np.clip(
            swing_phase,
            0.0,
            1.0,
        )
    )


    T_swing = float(
        T_swing
    )


    T_remaining = max(
        (
            1.0
            -
            s
        )
        *
        T_swing,
        0.0,
    )


    W_p_anchor = compute_raibert_anchor(

        model=model,

        data=data,

        B_r_anchor=B_r_anchor,
    )


    W_v_actual = np.asarray(
        x_current[
            6:9
        ],
        dtype=float,
    ).copy()


    W_v_ref = np.asarray(
        ref_output[
            "v_des_world"
        ],
        dtype=float,
    ).copy()


    # ========================================================
    # RAIBERT
    # ========================================================

    raibert_result = compute_raibert_foothold(

        W_p_hip=W_p_anchor,

        W_v_actual=W_v_actual,

        W_v_ref=W_v_ref,

        T_remaining=T_remaining,

        foothold_z=context[
            "foothold_z"
        ],

        leg=leg,
    )


    W_p_foothold = np.asarray(
        raibert_result[
            "W_p_foothold"
        ],
        dtype=float,
    )


    # ========================================================
    # BEZIER
    # ========================================================

    bezier_result = evaluate_bezier_swing_trajectory(

        W_p_liftoff=context[
            "W_p_liftoff"
        ],

        W_p_foothold=W_p_foothold,

        swing_phase=s,

        T_swing=T_swing,

        swing_height=SWING_HEIGHT,

        cp1_coef=CP1_COEF,

        cp2_coef=CP2_COEF,
    )


    W_p_des = np.asarray(
        bezier_result[
            "W_p_swing_des"
        ],
        dtype=float,
    )


    # ========================================================
    # IK
    # ========================================================

    joint_target = solve_leg_cartesian_target(

        model=model,

        data=data,

        leg=leg,

        W_p_ankle_des=W_p_des,
    )


    W_p_actual = get_ankle_position(
        model,
        data,
        leg,
    )


    cart_error = float(
        np.linalg.norm(
            W_p_des
            -
            W_p_actual
        )
    )


    return {

        "phase":
            s,

        "T_remaining":
            T_remaining,

        "W_p_anchor":
            W_p_anchor,

        "W_v_actual":
            W_v_actual,

        "W_v_ref":
            W_v_ref,

        "W_p_foothold":
            W_p_foothold,

        "W_p_des":
            W_p_des,

        "W_p_actual":
            W_p_actual,

        "cart_error":
            cart_error,

        "q_des":
            joint_target[
                "q_des"
            ],

        "q_actual":
            joint_target[
                "q_actual"
            ],

        "q_dot_actual":
            joint_target[
                "q_dot_actual"
            ],

        "diagnostics":
            joint_target[
                "diagnostics"
            ],

        "raibert":
            raibert_result,
    }


# ============================================================
# EARLY TOUCHDOWN:
# FORCE GAIT PHASE TO FOLLOWING DSP
#
# R swing occurs in SSP_L:
#
#     SSP_L -> DSP1
#
# L swing occurs in SSP_R:
#
#     SSP_R -> DSP2
#
# Test-only access to scheduler internal phase.
# ============================================================

def force_touchdown_to_dsp(
    gait,
    swing_leg,
):

    if swing_leg == "R":

        target_gait_step = int(
            gait._b1
        )


    elif swing_leg == "L":

        target_gait_step = int(
            gait._b3
        )


    else:

        raise ValueError(
            "swing_leg must be L or R."
        )


    gait._startup_elapsed = (
        gait.startup_hold_time
    )


    gait._gait_phase = (
        float(
            target_gait_step
        )
        /
        float(
            gait.gait_cycle_length
        )
    )


# ============================================================
# CONTACT FORCE DIAGNOSTIC
#
# mj_contactForce returns contact-frame wrench.
#
# Component 0 = contact normal force.
# ============================================================

def get_contact_normal_force_sum(
    model,
    data,
    contacts,
):

    total_normal_force = 0.0


    for entry in contacts:

        contact_index = int(
            entry[
                "contact_index"
            ]
        )


        wrench = np.zeros(
            6,
            dtype=float,
        )


        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            wrench,
        )


        total_normal_force += max(
            0.0,
            float(
                wrench[
                    0
                ]
            ),
        )


    return float(
        total_normal_force
    )


# ============================================================
# TORQUE CLIPPING
#
# Explicit clipping allows us to log the torque actually
# written to MuJoCo instead of relying silently on XML limits.
# ============================================================

def clip_full_joint_torque(
    model,
    tau_raw,
):

    tau_raw = np.asarray(
        tau_raw,
        dtype=float,
    ).reshape(
        10
    )


    actuator_ids = get_actuator_ids(
        model
    )


    tau_applied = tau_raw.copy()


    clipped = np.zeros(
        10,
        dtype=bool,
    )


    lower_limits = np.full(
        10,
        -np.inf,
        dtype=float,
    )


    upper_limits = np.full(
        10,
        +np.inf,
        dtype=float,
    )


    for i, actuator_id in enumerate(
        actuator_ids
    ):

        if bool(
            model.actuator_ctrllimited[
                actuator_id
            ]
        ):

            lower = float(
                model.actuator_ctrlrange[
                    actuator_id,
                    0
                ]
            )

            upper = float(
                model.actuator_ctrlrange[
                    actuator_id,
                    1
                ]
            )


            lower_limits[
                i
            ] = lower

            upper_limits[
                i
            ] = upper


            clipped_value = float(
                np.clip(
                    tau_raw[
                        i
                    ],
                    lower,
                    upper,
                )
            )


            clipped[
                i
            ] = not np.isclose(
                clipped_value,
                tau_raw[
                    i
                ],
            )


            tau_applied[
                i
            ] = clipped_value


    return {
        "tau_applied":
            tau_applied,

        "clipped":
            clipped,

        "lower":
            lower_limits,

        "upper":
            upper_limits,
    }


# ============================================================
# REFERENCE COMMAND
# ============================================================

def get_reference_command(
    t,
):

    if t < STARTUP_HOLD_TIME:

        return (
            np.zeros(
                3,
                dtype=float,
            ),
            0.0,
        )


    return (
        V_DES_BODY_TARGET.copy(),
        float(
            YAW_RATE_DES_TARGET
        ),
    )


# ============================================================
# FALL CHECK
# ============================================================

def detect_fall(
    x_current,
):

    x_current = np.asarray(
        x_current,
        dtype=float,
    ).reshape(
        NX
    )


    if not np.all(
        np.isfinite(
            x_current
        )
    ):

        return (
            True,
            "STATE_NAN_OR_INF",
        )


    if x_current[
        2
    ] < MIN_COM_HEIGHT:

        return (
            True,
            "COM_HEIGHT_TOO_LOW",
        )


    if abs(
        x_current[
            3
        ]
    ) > MAX_ABS_ROLL:

        return (
            True,
            "ROLL_TOO_LARGE",
        )


    if abs(
        x_current[
            4
        ]
    ) > MAX_ABS_PITCH:

        return (
            True,
            "PITCH_TOO_LARGE",
        )


    return (
        False,
        "",
    )


# ============================================================
# SMALL LOG HELPER
# ============================================================

def add_vector_to_row(
    row,
    prefix,
    vector,
):

    vector = np.asarray(
        vector,
        dtype=float,
    ).reshape(
        -1
    )


    for i, value in enumerate(
        vector
    ):

        row[
            f"{prefix}_{i}"
        ] = float(
            value
        )


# ============================================================
# PRINT EVENT
# ============================================================

def print_event_header(
    title,
    t,
):

    print()

    print(
        "#" * 130
    )

    print(
        f"{title} @ t = {t:.3f} s"
    )

    print(
        "#" * 130
    )


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(
    setup,
    viewer,
):

    model = setup.model
    data = setup.data


    # ========================================================
    # PHYSICS TIMING
    # ========================================================

    physics_dt = float(
        model.opt.timestep
    )


    physics_substeps = int(
        round(
            CONTROL_DT
            /
            physics_dt
        )
    )


    if (
        physics_substeps <= 0
        or
        not np.isclose(
            physics_substeps
            *
            physics_dt,
            CONTROL_DT,
        )
    ):

        raise RuntimeError(
            "CONTROL_DT must be an integer multiple "
            "of MuJoCo timestep."
        )


    print(
        f"MuJoCo dt                 : "
        f"{physics_dt:.6f} s"
    )

    print(
        f"Physics substeps          : "
        f"{physics_substeps}"
    )


    # ========================================================
    # INITIAL FORWARD
    # ========================================================

    mujoco.mj_forward(
        model,
        data,
    )


    # ========================================================
    # INITIAL FEEDBACK
    # ========================================================

    initial_feedback = get_current_state_mpc(

        model=model,

        data=data,

        B_inertia_CoM=setup.B_inertia_CoM,
    )


    Z_DES = float(
        initial_feedback.x[
            2
        ]
    )


    print(
        f"Initial CoM z             : "
        f"{Z_DES:.6f} m"
    )


    # ========================================================
    # GAIT
    # ========================================================

    gait = GaitScheduler(

        mpc_horizon=MPC_HORIZON,

        dsp_durations=DSP_DURATIONS,

        ssp_durations=SSP_DURATIONS,

        control_dt=CONTROL_DT,

        dt_mpc=MPC_DT,

        startup_hold_time=STARTUP_HOLD_TIME,

        first_swing_leg=FIRST_SWING_LEG,
    )


    # ========================================================
    # REFERENCE GENERATOR
    #
    # N + 1 because condensed MPC predicts:
    #
    # x_(k+1) ... x_(k+N)
    # ========================================================

    reference_generator = ReferenceGenerator(

        horizon_length=REFERENCE_HORIZON,

        dt_mpc=MPC_DT,

        reference_update_dt=MPC_UPDATE_DT,
    )


    # ========================================================
    # RAIBERT ANCHOR
    # ========================================================

    B_r_anchor = {

        "L":
            compute_raibert_anchor_offset_B(
                model,
                data,
                "L",
            ),

        "R":
            compute_raibert_anchor_offset_B(
                model,
                data,
                "R",
            ),
    }


    # ========================================================
    # SWING CONTEXT
    # ========================================================

    swing_context = {

        "L":
            create_swing_context(),

        "R":
            create_swing_context(),
    }


    joint_memory = {

        "L":
            create_joint_reference_memory(),

        "R":
            create_joint_reference_memory(),
    }


    # ========================================================
    # HELD MPC CONTROL
    # ========================================================

    held_u0 = np.zeros(
        NU,
        dtype=float,
    )


    ref_output = None

    X_ref_full = None


    previous_phase_name = None

    previous_active_swing_leg = None


    # ========================================================
    # STATISTICS
    # ========================================================

    stats = {

        "mpc_solves":
            0,

        "mpc_failures":
            0,

        "mpc_rejections":
            0,

        "contact_mismatch_samples":
            0,

        "max_abs_vy":
            0.0,

        "min_com_z":
            np.inf,

        "max_abs_roll":
            0.0,

        "max_abs_pitch":
            0.0,

        "max_fz_mpc":
            0.0,

        "peak_tau":
            np.zeros(
                10,
                dtype=float,
            ),

        "clip_count":
            np.zeros(
                10,
                dtype=int,
            ),

        "fall":
            False,

        "fall_reason":
            "",
    }


    # ========================================================
    # CSV
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    csv_path = (
        LOG_DIR
        /
        f"dynamic_walk_debug_{timestamp}.csv"
    )


    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )


    csv_writer = None


    print(
        f"Debug CSV                 : "
        f"{csv_path}"
    )


    # ========================================================
    # LOOP
    # ========================================================

    number_of_steps = int(
        round(
            SIMULATION_TIME
            /
            CONTROL_DT
        )
    )


    for k in range(
        number_of_steps
        +
        1
    ):

        t = (
            k
            *
            CONTROL_DT
        )


        event_string = ""


        # ====================================================
        # 1. READ x_k
        # ====================================================

        feedback = get_current_state_mpc(

            model=model,

            data=data,

            B_inertia_CoM=setup.B_inertia_CoM,
        )


        x_current = feedback.x.copy()


        # ====================================================
        # 2. PHYSICAL CONTACT AT x_k
        # ====================================================

        physical_contact = get_contact_state(

            model=model,

            data=data,

            contact_map=setup.contact_map,
        )


        actual_c_L = int(
            physical_contact[
                "c_L"
            ]
        )

        actual_c_R = int(
            physical_contact[
                "c_R"
            ]
        )


        actual_contact_vector = np.array(
            [
                actual_c_L,
                actual_c_R,
            ],
            dtype=int,
        )


        normal_force_L = (
            get_contact_normal_force_sum(
                model,
                data,
                physical_contact[
                    "left_contacts"
                ],
            )
        )


        normal_force_R = (
            get_contact_normal_force_sum(
                model,
                data,
                physical_contact[
                    "right_contacts"
                ],
            )
        )


        # ====================================================
        # 3. PLANNED GAIT
        # ====================================================

        gait_output = gait.get_output()


        active_swing_leg = gait_output[
            "active_swing_leg"
        ]


        # ====================================================
        # 4. EARLY TOUCHDOWN DETECTION
        #
        # Read physical contact BEFORE controller calculation.
        #
        # If valid early TD occurs:
        #
        #     modify gait phase
        #     recompute gait_output
        #
        # then current control uses DSP immediately.
        # ====================================================

        early_touchdown_leg = None


        if (
            ENABLE_EARLY_TOUCHDOWN
            and
            active_swing_leg is not None
            and
            swing_context[
                active_swing_leg
            ][
                "active"
            ]
        ):

            leg_index = LEG_INDEX[
                active_swing_leg
            ]


            swing_phase = float(
                gait_output[
                    "swing_phase"
                ][
                    leg_index
                ]
            )


            physical_contact_value = int(
                actual_contact_vector[
                    leg_index
                ]
            )


            context = swing_context[
                active_swing_leg
            ]


            # ------------------------------------------------
            # First require real clearance
            # ------------------------------------------------

            if physical_contact_value == 0:

                context[
                    "had_clearance"
                ] = True

                context[
                    "contact_debounce"
                ] = 0


            elif (
                context[
                    "had_clearance"
                ]
                and
                swing_phase
                >=
                EARLY_TOUCHDOWN_MIN_PHASE
            ):

                context[
                    "contact_debounce"
                ] += 1


                if (
                    context[
                        "contact_debounce"
                    ]
                    >=
                    EARLY_TOUCHDOWN_DEBOUNCE_STEPS
                ):

                    early_touchdown_leg = (
                        active_swing_leg
                    )


                    force_touchdown_to_dsp(
                        gait,
                        active_swing_leg,
                    )


                    gait_output = gait.get_output()


                    active_swing_leg = gait_output[
                        "active_swing_leg"
                    ]


                    event_string = (
                        f"EARLY_TD_{early_touchdown_leg}"
                    )


        # ====================================================
        # 5. HANDLE SWING -> CONTACT TRANSITION
        # ====================================================

        if (
            previous_active_swing_leg
            is not None
            and
            active_swing_leg
            !=
            previous_active_swing_leg
        ):

            touchdown_leg = (
                previous_active_swing_leg
            )


            context = swing_context[
                touchdown_leg
            ]


            if context[
                "active"
            ]:

                leg_index = LEG_INDEX[
                    touchdown_leg
                ]


                actual_contact_at_td = int(
                    actual_contact_vector[
                        leg_index
                    ]
                )


                actual_ankle = get_ankle_position(
                    model,
                    data,
                    touchdown_leg,
                )


                if (
                    touchdown_leg
                    ==
                    early_touchdown_leg
                ):

                    touchdown_reason = (
                        "EARLY_TOUCHDOWN"
                    )


                    context[
                        "early_touchdowns"
                    ] += 1


                else:

                    touchdown_reason = (
                        "PLANNED_TOUCHDOWN"
                    )


                    context[
                        "planned_touchdowns"
                    ] += 1


                    if actual_contact_at_td == 0:

                        context[
                            "missed_planned_touchdowns"
                        ] += 1


                print_event_header(
                    touchdown_reason,
                    t,
                )


                print(
                    f"leg                       : "
                    f"{touchdown_leg}"
                )

                print(
                    f"actual contact            : "
                    f"{actual_contact_at_td}"
                )

                print(
                    f"actual ankle              : "
                    f"{actual_ankle}"
                )


                if context[
                    "last_target"
                ] is not None:

                    planned_foothold = np.asarray(
                        context[
                            "last_target"
                        ][
                            "W_p_foothold"
                        ],
                        dtype=float,
                    )


                    print(
                        f"last Raibert foothold     : "
                        f"{planned_foothold}"
                    )

                    print(
                        f"touchdown position error  : "
                        f"{1000.0*np.linalg.norm(actual_ankle-planned_foothold):.3f} mm"
                    )


                context[
                    "active"
                ] = False


                context[
                    "completed_swings"
                ] += 1


                reset_joint_reference_memory(
                    joint_memory[
                        touchdown_leg
                    ]
                )


                if event_string:

                    event_string += "|"


                event_string += (
                    f"{touchdown_reason}_{touchdown_leg}"
                )


        # ====================================================
        # 6. HANDLE CONTACT -> SWING TRANSITION
        # ====================================================

        if (
            active_swing_leg
            is not None
            and
            active_swing_leg
            !=
            previous_active_swing_leg
        ):

            W_p_liftoff = get_ankle_position(
                model,
                data,
                active_swing_leg,
            )


            context = swing_context[
                active_swing_leg
            ]


            context[
                "active"
            ] = True


            context[
                "W_p_liftoff"
            ] = W_p_liftoff.copy()


            context[
                "foothold_z"
            ] = float(
                W_p_liftoff[
                    2
                ]
            )


            context[
                "had_clearance"
            ] = False


            context[
                "contact_debounce"
            ] = 0


            context[
                "last_target"
            ] = None


            context[
                "last_q_des"
            ] = None


            context[
                "last_q_dot_des"
            ] = None


            reset_joint_reference_memory(
                joint_memory[
                    active_swing_leg
                ]
            )


            print_event_header(
                "LIFTOFF",
                t,
            )


            print(
                f"leg                       : "
                f"{active_swing_leg}"
            )

            print(
                f"W_p_liftoff               : "
                f"{W_p_liftoff}"
            )

            print(
                f"CoM position              : "
                f"{x_current[0:3]}"
            )

            print(
                f"CoM velocity              : "
                f"{x_current[6:9]}"
            )

            print(
                f"actual contacts           : "
                f"{actual_contact_vector}"
            )


            if event_string:

                event_string += "|"


            event_string += (
                f"LIFTOFF_{active_swing_leg}"
            )


        # ====================================================
        # 7. GAIT TRANSITION LOG
        # ====================================================

        phase_name = str(
            gait_output[
                "phase_name"
            ]
        )


        if (
            phase_name
            !=
            previous_phase_name
        ):

            print()

            print(
                "=" * 130
            )

            print(
                f"GAIT TRANSITION @ "
                f"t={t:.3f} s"
            )

            print(
                "=" * 130
            )

            print(
                f"phase                     : "
                f"{phase_name}"
            )

            print(
                f"planned contact           : "
                f"{gait_output['contact_state']}"
            )

            print(
                f"actual contact            : "
                f"{actual_contact_vector}"
            )

            print(
                f"active swing              : "
                f"{active_swing_leg}"
            )


            previous_phase_name = (
                phase_name
            )


        # ====================================================
        # 8. COMMAND
        # ====================================================

        (
            v_des_body,
            yaw_rate_des,

        ) = get_reference_command(
            t
        )


        # ====================================================
        # 9. MPC UPDATE
        #
        # 200 Hz
        # ====================================================

        is_mpc_tick = (
            k
            %
            MPC_UPDATE_DECIMATION
            ==
            0
        )


        solver_info = {}

        post_diagnostics = {}


        if is_mpc_tick:

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            ref_output = reference_generator.update(

                x_actual=x_current,

                v_des_body=v_des_body,

                yaw_rate_des=yaw_rate_des,

                z_des=Z_DES,

                roll_des=0.0,

                pitch_des=0.0,
            )


            X_ref_full = np.asarray(
                ref_output[
                    "X_ref"
                ],
                dtype=float,
            ).copy()


            if X_ref_full.shape != (
                REFERENCE_HORIZON,
                NX,
            ):

                raise RuntimeError(
                    f"Unexpected X_ref shape: "
                    f"{X_ref_full.shape}"
                )


            # ------------------------------------------------
            # No artificial static lateral transfer here.
            # ------------------------------------------------

            X_ref_mpc = X_ref_full[
                1:
                MPC_HORIZON
                +
                1,
                :
            ].copy()


            X_ref_flat = X_ref_mpc.reshape(
                MPC_HORIZON
                *
                NX
            )


            # ------------------------------------------------
            # SRBD model from CURRENT feedback x_k
            # ------------------------------------------------

            (
                Phi,
                Gamma,

            ) = build_SRBD_model(

                Ew_inverse=feedback.Ew_inverse,

                mass_CoM=setup.mass_CoM,

                W_inertia_CoM=feedback.W_inertia_CoM,

                W_r_L=feedback.W_r_L,

                W_r_R=feedback.W_r_R,

                dt_mpc=MPC_DT,

                horizon=MPC_HORIZON,
            )


            contact_schedule = np.asarray(
                gait_output[
                    "mpc_contact_schedule"
                ],
                dtype=int,
            )


            (
                U_opt,
                u0_opt,
                solver_info,
                post_diagnostics,

            ) = build_and_solve_QP(

                qp_solver=setup.qp_solver,

                Phi=Phi,

                Gamma=Gamma,

                x_current=x_current,

                X_ref=X_ref_flat,

                N=MPC_HORIZON,

                L6_R_W=feedback.L6_IK_R_W,

                R6_R_W=feedback.R6_IK_R_W,

                contact_schedule=contact_schedule,

                print_diagnostics=False,
            )


            stats[
                "mpc_solves"
            ] += 1


            if U_opt is None:

                stats[
                    "mpc_failures"
                ] += 1


                print_event_header(
                    "MPC HARD FAILURE",
                    t,
                )

                print(
                    solver_info
                )


                stats[
                    "fall"
                ] = True

                stats[
                    "fall_reason"
                ] = "MPC_HARD_FAILURE"


                break


            if u0_opt is None:

                stats[
                    "mpc_rejections"
                ] += 1


                print_event_header(
                    "MPC POST-CHECK REJECTION",
                    t,
                )

                print(
                    f"solver info               : "
                    f"{solver_info}"
                )

                print(
                    f"post diagnostics          : "
                    f"{post_diagnostics}"
                )


                stats[
                    "fall"
                ] = True

                stats[
                    "fall_reason"
                ] = (
                    "MPC_POSTCHECK_REJECTION"
                )


                break


            held_u0 = np.asarray(
                u0_opt,
                dtype=float,
            ).reshape(
                NU
            ).copy()


        # ====================================================
        # 10. REFERENCE MUST EXIST
        # ====================================================

        if ref_output is None:

            raise RuntimeError(
                "Reference output is None."
            )


        # ====================================================
        # 11. MPC WRENCH -> STANCE JOINT TORQUE
        #
        # Mapping is recomputed at CURRENT q_k every 1 ms.
        # Wrench itself is held between MPC updates.
        # ====================================================

        (
            tau_mpc_full,
            tau_mpc_L,
            tau_mpc_R,

        ) = get_joint_torque_from_MPC_wrench(

            model=model,

            data=data,

            u0_MPC=held_u0,
        )


        tau_command_raw = np.asarray(
            tau_mpc_full,
            dtype=float,
        ).copy()


        tau_swing_full = np.zeros(
            10,
            dtype=float,
        )


        q_des_full = np.full(
            10,
            np.nan,
            dtype=float,
        )


        q_dot_des_full = np.full(
            10,
            np.nan,
            dtype=float,
        )


        active_target = None


        # ====================================================
        # 12. SWING LEG
        #
        # Replace MPC torque of the swing leg with:
        #
        # Raibert -> Bezier -> IK -> PD
        #
        # MPC constrains swing-foot wrench to zero anyway.
        # ====================================================

        if active_swing_leg is not None:

            leg = active_swing_leg

            leg_index = LEG_INDEX[
                leg
            ]

            leg_slice = LEG_SLICE[
                leg
            ]


            swing_phase = float(
                gait_output[
                    "swing_phase"
                ][
                    leg_index
                ]
            )


            T_swing = float(
                gait_output[
                    "T_swing"
                ][
                    leg_index
                ]
            )


            active_target = compute_dynamic_swing_target(

                model=model,

                data=data,

                leg=leg,

                swing_phase=swing_phase,

                T_swing=T_swing,

                context=swing_context[
                    leg
                ],

                B_r_anchor=B_r_anchor[
                    leg
                ],

                x_current=x_current,

                ref_output=ref_output,
            )


            q_dot_des = update_q_dot_des(

                q_des=active_target[
                    "q_des"
                ],

                memory=joint_memory[
                    leg
                ],
            )


            tau_swing_leg = compute_swing_joint_PD(

                q_des=active_target[
                    "q_des"
                ],

                q=active_target[
                    "q_actual"
                ],

                q_dot=active_target[
                    "q_dot_actual"
                ],

                Kp=KP_SWING,

                Kd=KD_SWING,

                q_dot_des=q_dot_des,
            )


            tau_swing_full[
                leg_slice
            ] = tau_swing_leg


            # ------------------------------------------------
            # REPLACE stance/MPC torque on swing leg
            # ------------------------------------------------

            tau_command_raw[
                leg_slice
            ] = tau_swing_leg


            q_des_full[
                leg_slice
            ] = active_target[
                "q_des"
            ]


            q_dot_des_full[
                leg_slice
            ] = q_dot_des


            context = swing_context[
                leg
            ]


            context[
                "last_target"
            ] = active_target


            context[
                "last_q_des"
            ] = active_target[
                "q_des"
            ].copy()


            context[
                "last_q_dot_des"
            ] = q_dot_des.copy()


            context[
                "swing_samples"
            ] += 1


            context[
                "sum_cart_error_sq"
            ] += (
                active_target[
                    "cart_error"
                ]
                **
                2
            )


            context[
                "max_cart_error"
            ] = max(
                context[
                    "max_cart_error"
                ],
                active_target[
                    "cart_error"
                ],
            )


        # ====================================================
        # 13. TORQUE CLIP
        # ====================================================

        clip_result = clip_full_joint_torque(

            model=model,

            tau_raw=tau_command_raw,
        )


        tau_applied = clip_result[
            "tau_applied"
        ]


        clipped = clip_result[
            "clipped"
        ]


        # ====================================================
        # 14. APPLY tau_k
        #
        # This happens BEFORE mj_step().
        # ====================================================

        data.ctrl[:] = 0.0


        apply_joint_torque(

            model=model,

            data=data,

            tau=tau_applied,
        )


        # ====================================================
        # 15. CURRENT JOINT STATES
        # ====================================================

        (
            q_L,
            q_dot_L,

        ) = get_leg_joint_state(
            model,
            data,
            "L",
        )


        (
            q_R,
            q_dot_R,

        ) = get_leg_joint_state(
            model,
            data,
            "R",
        )


        q_full = np.concatenate(
            [
                q_L,
                q_R,
            ]
        )


        q_dot_full = np.concatenate(
            [
                q_dot_L,
                q_dot_R,
            ]
        )


        # ====================================================
        # 16. CONTACT MISMATCH
        # ====================================================

        planned_contact_vector = np.asarray(
            gait_output[
                "contact_state"
            ],
            dtype=int,
        ).reshape(
            2
        )


        contact_mismatch = int(
            not np.array_equal(
                planned_contact_vector,
                actual_contact_vector,
            )
        )


        stats[
            "contact_mismatch_samples"
        ] += contact_mismatch


        # ====================================================
        # 17. STATS
        # ====================================================

        stats[
            "max_abs_vy"
        ] = max(
            stats[
                "max_abs_vy"
            ],
            abs(
                float(
                    x_current[
                        7
                    ]
                )
            ),
        )


        stats[
            "min_com_z"
        ] = min(
            stats[
                "min_com_z"
            ],
            float(
                x_current[
                    2
                ]
            ),
        )


        stats[
            "max_abs_roll"
        ] = max(
            stats[
                "max_abs_roll"
            ],
            abs(
                float(
                    x_current[
                        3
                    ]
                )
            ),
        )


        stats[
            "max_abs_pitch"
        ] = max(
            stats[
                "max_abs_pitch"
            ],
            abs(
                float(
                    x_current[
                        4
                    ]
                )
            ),
        )


        stats[
            "max_fz_mpc"
        ] = max(
            stats[
                "max_fz_mpc"
            ],
            float(
                held_u0[
                    2
                ]
                +
                held_u0[
                    8
                ]
            ),
        )


        stats[
            "peak_tau"
        ] = np.maximum(
            stats[
                "peak_tau"
            ],
            np.abs(
                tau_applied
            ),
        )


        stats[
            "clip_count"
        ] += clipped.astype(
            int
        )


        # ====================================================
        # 18. FALL CHECK
        # ====================================================

        (
            fallen,
            fall_reason,

        ) = detect_fall(
            x_current
        )


        if fallen:

            stats[
                "fall"
            ] = True

            stats[
                "fall_reason"
            ] = fall_reason


            print_event_header(
                "FALL DETECTED",
                t,
            )

            print(
                f"reason                    : "
                f"{fall_reason}"
            )

            print(
                f"state                     : "
                f"{x_current}"
            )

            print(
                f"planned contact           : "
                f"{planned_contact_vector}"
            )

            print(
                f"actual contact            : "
                f"{actual_contact_vector}"
            )

            print(
                f"u0 MPC                    : "
                f"{held_u0}"
            )

            print(
                f"tau MPC                   : "
                f"{tau_mpc_full}"
            )

            print(
                f"tau applied               : "
                f"{tau_applied}"
            )


            if event_string:

                event_string += "|"


            event_string += (
                f"FALL_{fall_reason}"
            )


        # ====================================================
        # 19. BUILD CSV ROW
        # ====================================================

        row = {

            "time":
                float(
                    t
                ),

            "phase":
                phase_name,

            "active_swing":
                (
                    ""
                    if active_swing_leg is None
                    else active_swing_leg
                ),

            "event":
                event_string,

            "planned_c_L":
                int(
                    planned_contact_vector[
                        0
                    ]
                ),

            "planned_c_R":
                int(
                    planned_contact_vector[
                        1
                    ]
                ),

            "actual_c_L":
                actual_c_L,

            "actual_c_R":
                actual_c_R,

            "contact_mismatch":
                contact_mismatch,

            "left_contact_count":
                int(
                    physical_contact[
                        "left_contact_count"
                    ]
                ),

            "right_contact_count":
                int(
                    physical_contact[
                        "right_contact_count"
                    ]
                ),

            "normal_force_L":
                normal_force_L,

            "normal_force_R":
                normal_force_R,

            "mpc_tick":
                int(
                    is_mpc_tick
                ),

            "mpc_status":
                str(
                    solver_info.get(
                        "status",
                        "",
                    )
                ),

            "mpc_iterations":
                float(
                    solver_info.get(
                        "iterations",
                        np.nan,
                    )
                ),

            "mpc_max_violation":
                float(
                    post_diagnostics.get(
                        "max_violation",
                        np.nan,
                    )
                ),

            "torque_clip_any":
                int(
                    np.any(
                        clipped
                    )
                ),

            "fall":
                int(
                    fallen
                ),

            "fall_reason":
                fall_reason,
        }


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        state_labels = [
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
            "one",
        ]


        for i, label in enumerate(
            state_labels
        ):

            row[
                f"state_{label}"
            ] = float(
                x_current[
                    i
                ]
            )


        # ----------------------------------------------------
        # CURRENT REFERENCE KNOT
        # ----------------------------------------------------

        if X_ref_full is not None:

            reference_now = X_ref_full[
                0
            ]


            for i, label in enumerate(
                state_labels
            ):

                row[
                    f"ref_{label}"
                ] = float(
                    reference_now[
                        i
                    ]
                )


        else:

            for label in state_labels:

                row[
                    f"ref_{label}"
                ] = np.nan


        # ----------------------------------------------------
        # MPC WRENCH
        # ----------------------------------------------------

        wrench_labels = [
            "FLx",
            "FLy",
            "FLz",
            "MLx",
            "MLy",
            "MLz",
            "FRx",
            "FRy",
            "FRz",
            "MRx",
            "MRy",
            "MRz",
        ]


        for i, label in enumerate(
            wrench_labels
        ):

            row[
                f"mpc_{label}"
            ] = float(
                held_u0[
                    i
                ]
            )


        # ----------------------------------------------------
        # TORQUE
        # ----------------------------------------------------

        for i, actuator_name in enumerate(
            ACTUATOR_NAMES
        ):

            row[
                f"tau_mpc_{actuator_name}"
            ] = float(
                tau_mpc_full[
                    i
                ]
            )


            row[
                f"tau_swing_{actuator_name}"
            ] = float(
                tau_swing_full[
                    i
                ]
            )


            row[
                f"tau_raw_{actuator_name}"
            ] = float(
                tau_command_raw[
                    i
                ]
            )


            row[
                f"tau_applied_{actuator_name}"
            ] = float(
                tau_applied[
                    i
                ]
            )


            row[
                f"tau_clip_{actuator_name}"
            ] = int(
                clipped[
                    i
                ]
            )


            row[
                f"q_{actuator_name}"
            ] = float(
                q_full[
                    i
                ]
            )


            row[
                f"qdot_{actuator_name}"
            ] = float(
                q_dot_full[
                    i
                ]
            )


            row[
                f"qdes_{actuator_name}"
            ] = float(
                q_des_full[
                    i
                ]
            )


            row[
                f"qdot_des_{actuator_name}"
            ] = float(
                q_dot_des_full[
                    i
                ]
            )


        # ----------------------------------------------------
        # SWING DEBUG
        # ----------------------------------------------------

        row[
            "swing_phase"
        ] = np.nan

        row[
            "swing_cart_error"
        ] = np.nan

        row[
            "swing_ik_error"
        ] = np.nan


        for axis in (
            "x",
            "y",
            "z",
        ):

            row[
                f"swing_liftoff_{axis}"
            ] = np.nan

            row[
                f"swing_foothold_{axis}"
            ] = np.nan

            row[
                f"swing_des_{axis}"
            ] = np.nan

            row[
                f"swing_actual_{axis}"
            ] = np.nan

            row[
                f"raibert_raw_correction_{axis}"
            ] = np.nan

            row[
                f"raibert_clipped_correction_{axis}"
            ] = np.nan


        if (
            active_swing_leg is not None
            and
            active_target is not None
        ):

            context = swing_context[
                active_swing_leg
            ]


            row[
                "swing_phase"
            ] = float(
                active_target[
                    "phase"
                ]
            )


            row[
                "swing_cart_error"
            ] = float(
                active_target[
                    "cart_error"
                ]
            )


            row[
                "swing_ik_error"
            ] = float(
                active_target[
                    "diagnostics"
                ].get(
                    "final_position_error_norm",
                    np.nan,
                )
            )


            vectors = {

                "swing_liftoff":
                    context[
                        "W_p_liftoff"
                    ],

                "swing_foothold":
                    active_target[
                        "W_p_foothold"
                    ],

                "swing_des":
                    active_target[
                        "W_p_des"
                    ],

                "swing_actual":
                    active_target[
                        "W_p_actual"
                    ],

                "raibert_raw_correction":
                    active_target[
                        "raibert"
                    ][
                        "raw_correction"
                    ],

                "raibert_clipped_correction":
                    active_target[
                        "raibert"
                    ][
                        "clipped_correction"
                    ],
            }


            for prefix, vector in vectors.items():

                vector = np.asarray(
                    vector,
                    dtype=float,
                )


                row[
                    f"{prefix}_x"
                ] = float(
                    vector[
                        0
                    ]
                )

                row[
                    f"{prefix}_y"
                ] = float(
                    vector[
                        1
                    ]
                )

                row[
                    f"{prefix}_z"
                ] = float(
                    vector[
                        2
                    ]
                )


        # ====================================================
        # 20. WRITE CSV
        # ====================================================

        if csv_writer is None:

            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=list(
                    row.keys()
                ),
            )


            csv_writer.writeheader()


        csv_writer.writerow(
            row
        )


        if (
            k
            %
            100
            ==
            0
            or
            event_string
        ):

            csv_file.flush()


        # ====================================================
        # 21. CONSOLE STATUS
        # ====================================================

        if (
            k
            %
            STATUS_PRINT_DECIMATION
            ==
            0
        ):

            print()

            print(
                "-" * 130
            )

            print(
                f"STATUS @ t={t:.3f} s"
            )

            print(
                "-" * 130
            )

            print(
                f"phase / swing             : "
                f"{phase_name} / "
                f"{active_swing_leg}"
            )

            print(
                f"planned / actual contact  : "
                f"{planned_contact_vector} / "
                f"{actual_contact_vector}"
            )

            print(
                f"CoM p                     : "
                f"{x_current[0:3]}"
            )

            print(
                f"CoM v                     : "
                f"{x_current[6:9]}"
            )

            print(
                f"Euler                     : "
                f"{np.rad2deg(x_current[3:6])} deg"
            )

            print(
                f"physical normal force L/R : "
                f"{normal_force_L:.3f} / "
                f"{normal_force_R:.3f} N"
            )

            print(
                f"MPC Fz L/R                : "
                f"{held_u0[2]:+.3f} / "
                f"{held_u0[8]:+.3f} N"
            )

            print(
                f"peak |tau current|        : "
                f"{np.max(np.abs(tau_applied)):.4f} N.m"
            )

            print(
                f"torque clip               : "
                f"{clipped.astype(int)}"
            )


            if active_target is not None:

                print(
                    f"swing phase               : "
                    f"{active_target['phase']:.3f}"
                )

                print(
                    f"Raibert foothold          : "
                    f"{active_target['W_p_foothold']}"
                )

                print(
                    f"swing desired             : "
                    f"{active_target['W_p_des']}"
                )

                print(
                    f"swing actual              : "
                    f"{active_target['W_p_actual']}"
                )

                print(
                    f"swing cart error          : "
                    f"{1000.0*active_target['cart_error']:.3f} mm"
                )

                print(
                    f"Raibert correction       : "
                    f"{active_target['raibert']['clipped_correction']}"
                )


        # ====================================================
        # 22. VIEWER
        # ====================================================

        if viewer is not None:

            if not viewer.is_running():

                print(
                    "Viewer closed by user."
                )

                break


            viewer.sync()


            if VIEWER_SLEEP > 0.0:

                time.sleep(
                    VIEWER_SLEEP
                )


        # ====================================================
        # 23. STOP AFTER FALL
        # ====================================================

        if fallen:

            break


        if k == number_of_steps:

            break


        # ====================================================
        # 24. PHYSICS
        #
        # CRITICAL ORDER:
        #
        # x_k
        # ->
        # compute u_k / tau_k
        # ->
        # write data.ctrl
        # ->
        # mj_step()
        # ->
        # x_(k+1)
        #
        # SAME tau_k is held during physics substeps.
        # ====================================================

        for _ in range(
            physics_substeps
        ):

            mujoco.mj_step(
                model,
                data,
            )


        # ====================================================
        # 25. STORE CURRENT SWING STATE
        # ====================================================

        previous_active_swing_leg = (
            active_swing_leg
        )


        # ====================================================
        # 26. ADVANCE GAIT
        #
        # AFTER physics only.
        # ====================================================

        gait.advance()


    # ========================================================
    # END
    # ========================================================

    csv_file.flush()

    csv_file.close()


    return {
        "stats":
            stats,

        "swing_context":
            swing_context,

        "csv_path":
            csv_path,
    }


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results,
):

    stats = results[
        "stats"
    ]

    swing_context = results[
        "swing_context"
    ]


    print()

    print(
        "=" * 140
    )

    print(
        "DYNAMIC WALKING TEST SUMMARY"
    )

    print(
        "=" * 140
    )


    print(
        f"MPC solves                  : "
        f"{stats['mpc_solves']}"
    )

    print(
        f"MPC hard failures           : "
        f"{stats['mpc_failures']}"
    )

    print(
        f"MPC post-check rejections   : "
        f"{stats['mpc_rejections']}"
    )


    print()

    print(
        f"Contact mismatch samples    : "
        f"{stats['contact_mismatch_samples']}"
    )


    print(
        f"Minimum CoM z               : "
        f"{stats['min_com_z']:.6f} m"
    )

    print(
        f"Maximum |vy|                : "
        f"{stats['max_abs_vy']:.6f} m/s"
    )

    print(
        f"Maximum |roll|              : "
        f"{np.rad2deg(stats['max_abs_roll']):.3f} deg"
    )

    print(
        f"Maximum |pitch|             : "
        f"{np.rad2deg(stats['max_abs_pitch']):.3f} deg"
    )

    print(
        f"Maximum MPC Fz sum          : "
        f"{stats['max_fz_mpc']:.6f} N"
    )


    print()

    print(
        f"FALL                        : "
        f"{stats['fall']}"
    )

    print(
        f"Fall / stop reason          : "
        f"{stats['fall_reason']}"
    )


    print()

    print(
        "PEAK APPLIED JOINT TORQUE"
    )

    print(
        "-" * 90
    )


    for i, actuator_name in enumerate(
        ACTUATOR_NAMES
    ):

        print(
            f"{actuator_name:<16} : "
            f"{stats['peak_tau'][i]:.6f} N.m "
            f"| clip count = "
            f"{stats['clip_count'][i]}"
        )


    for leg in (
        "R",
        "L",
    ):

        context = swing_context[
            leg
        ]


        samples = int(
            context[
                "swing_samples"
            ]
        )


        if samples > 0:

            rmse = np.sqrt(
                context[
                    "sum_cart_error_sq"
                ]
                /
                samples
            )


        else:

            rmse = np.nan


        print()

        print(
            f"LEG {leg}"
        )

        print(
            "-" * 90
        )

        print(
            f"completed swings             : "
            f"{context['completed_swings']}"
        )

        print(
            f"early touchdowns             : "
            f"{context['early_touchdowns']}"
        )

        print(
            f"planned touchdowns           : "
            f"{context['planned_touchdowns']}"
        )

        print(
            f"missed planned touchdowns    : "
            f"{context['missed_planned_touchdowns']}"
        )

        print(
            f"swing samples                : "
            f"{samples}"
        )

        print(
            f"swing Cartesian RMSE         : "
            f"{1000.0*rmse:.3f} mm"
        )

        print(
            f"maximum swing error          : "
            f"{1000.0*context['max_cart_error']:.3f} mm"
        )


    print()

    print(
        "=" * 140
    )

    print(
        f"DEBUG CSV:"
    )

    print(
        results[
            "csv_path"
        ]
    )

    print(
        "=" * 140
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 140
    )

    print(
        "TEST: FULL DYNAMIC WALKING"
    )

    print(
        "=" * 140
    )


    print(
        f"XML                         : "
        f"{XML_PATH}"
    )

    print(
        "Floating base               : FREE / PHYSICAL"
    )

    print(
        "SRBD-MPC                    : YES"
    )

    print(
        "MPC wrench -> stance torque : YES"
    )

    print(
        "Raibert foot placement      : YES"
    )

    print(
        "Bezier swing trajectory     : YES"
    )

    print(
        "Numerical IK                : YES"
    )

    print(
        "Swing PD                    : YES"
    )

    print(
        "Actual MuJoCo contact       : YES"
    )

    print(
        f"Early touchdown             : "
        f"{ENABLE_EARLY_TOUCHDOWN}"
    )

    print(
        "Artificial static SSP       : NO"
    )


    print()

    print(
        f"CONTROL_DT                  : "
        f"{CONTROL_DT:.6f} s"
    )

    print(
        f"MPC update                  : "
        f"{MPC_UPDATE_DT:.6f} s"
    )

    print(
        f"MPC prediction dt           : "
        f"{MPC_DT:.6f} s"
    )

    print(
        f"MPC horizon                 : "
        f"{MPC_HORIZON}"
    )

    print(
        f"prediction horizon          : "
        f"{MPC_HORIZON*MPC_DT:.3f} s"
    )


    print()

    print(
        f"SSP duration                : "
        f"{SSP_DURATIONS[0]*MPC_DT:.3f} s"
    )

    print(
        f"DSP duration                : "
        f"{DSP_DURATIONS[0]*MPC_DT:.3f} s"
    )

    print(
        f"startup hold                : "
        f"{STARTUP_HOLD_TIME:.3f} s"
    )

    print(
        f"vx desired                  : "
        f"{V_DES_BODY_TARGET[0]:.3f} m/s"
    )

    print(
        f"swing height                : "
        f"{1000.0*SWING_HEIGHT:.1f} mm"
    )


    print()

    print(
        "CONTROL ORDER:"
    )

    print(
        "x_k -> gait/contact -> reference -> MPC/Raibert/IK/PD "
        "-> tau_k -> mj_step() -> x_(k+1) -> gait.advance()"
    )


    # ========================================================
    # SETUP
    # ========================================================

    setup = initialize_simulation(

        xml_path=XML_PATH,

        horizon=MPC_HORIZON,

        dt_mpc=MPC_DT,

        dt_mpc_update=MPC_UPDATE_DT,

        home_key_name="home",

        qp_verbose=False,

        qp_eps_abs=QP_EPS_ABS,

        qp_eps_rel=QP_EPS_REL,

        qp_max_iter=QP_MAX_ITER,

        qp_polishing=True,

        qp_enable_warm_start=False,

        qp_constraint_tolerance=QP_CONSTRAINT_TOLERANCE,

        print_initialization=True,
    )


    # ========================================================
    # VIEWER
    # ========================================================

    if USE_GUI:

        with mujoco.viewer.launch_passive(
            setup.model,
            setup.data,
        ) as viewer:

            results = run_simulation(
                setup,
                viewer,
            )


    else:

        results = run_simulation(
            setup,
            None,
        )


    print_summary(
        results
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()