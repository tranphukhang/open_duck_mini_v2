from pathlib import Path
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

XML_PATH = (
    REPO_ROOT
    / "xmls"
    / "open_duck_mini_v3.xml"
)

if str(MPC_SRBD_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MPC_SRBD_DIR),
    )


# ============================================================
# PROJECT MODULES
# ============================================================

from Ref_generator.gait_scheduler import (
    GaitScheduler,
)

from Ref_generator.reference_generator import (
    ReferenceGenerator,
)

from Footstep_Planner.raibert_foot_placement import (
    compute_raibert_foothold,
)

from Trajectory_Planner.bezier_swing_trajectory import (
    evaluate_bezier_swing_trajectory,
)

from Kinematics.numerical_IK_hybrid import (
    solve_numerical_IK_hybrid,
)

from Kinematics.numerical_IK_params import (
    HIP_YAW_FIXED,
    SIGMA_THRESHOLD,
    ALPHA_NEWTON,
    ETA_GRADIENT,
    IK_TOLERANCE,
    IK_MAX_ITERATIONS,
)

from Actuator.swing_joint_PD import (
    compute_swing_joint_PD,
)

from Actuator.swing_joint_PD_params import (
    KP_SWING,
    KD_SWING,
)


# ============================================================
# LEG
# ============================================================

LEFT = 0
RIGHT = 1

LEG_INDEX = {
    "L": LEFT,
    "R": RIGHT,
}


# ============================================================
# TIMING
# ============================================================

CONTROL_DT = 0.001

REFERENCE_UPDATE_DT = 0.005

MPC_DT = 0.040

MPC_HORIZON = 10


REFERENCE_DECIMATION = int(
    round(
        REFERENCE_UPDATE_DT
        /
        CONTROL_DT
    )
)

if REFERENCE_DECIMATION <= 0:
    raise ValueError(
        "REFERENCE_DECIMATION must be > 0."
    )

if not np.isclose(
    REFERENCE_DECIMATION * CONTROL_DT,
    REFERENCE_UPDATE_DT,
):
    raise ValueError(
        "REFERENCE_UPDATE_DT must be an integer "
        "multiple of CONTROL_DT."
    )


# ============================================================
# GAIT
# ============================================================

SSP_DURATIONS = (
    6,
    6,
)

DSP_DURATIONS = (
    3,
    3,
)

STARTUP_HOLD_TIME = 0.250

FIRST_SWING_LEG = "R"


# ============================================================
# SIMULATION
#
# Timeline:
#
# 0.000 STARTUP_DS
# 0.250 DSP2
# 0.370 R swing
# 0.610 R touchdown
# 0.730 L swing
# 0.970 L touchdown
# 1.090 R swing
# 1.330 R touchdown
# 1.450 L swing
# 1.690 L touchdown
# ============================================================

SIMULATION_TIME = 1.70


# ============================================================
# REFERENCE COMMAND
#
# During STARTUP_DS:
#
#     v_des = 0
#
# After startup:
#
#     vx_des = 0.08 m/s
# ============================================================

V_DES_BODY_TARGET = np.array(
    [
        0.08,
        0.00,
        0.00,
    ],
    dtype=float,
)

YAW_RATE_DES_TARGET = 0.0

Z_DES = 0.280

ROLL_DES = 0.0

PITCH_DES = 0.0


# ============================================================
# SWING TRAJECTORY
# ============================================================

SWING_HEIGHT = 0.008

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
# INITIAL JOINT CONFIGURATION
#
# Order:
#
# hip_yaw
# hip_roll
# hip_pitch
# knee
# ankle
# ============================================================

INITIAL_Q_LEFT = np.array(
    [
        0.0,
        0.0,
        0.626,
        -1.3963,
        0.7854,
    ],
    dtype=float,
)

INITIAL_Q_RIGHT = np.array(
    [
        0.0,
        0.0,
        -0.626,
        1.3963,
        -0.7854,
    ],
    dtype=float,
)


# ============================================================
# ANKLE COMPENSATION
# ============================================================

SLOPE_PITCH = 0.0


# ============================================================
# MUJOCO NAMES
# ============================================================

FLOATING_BASE_JOINT = "floating_base"

BASE_BODY_NAME = "base_link"


ANKLE_SITE_NAMES = {
    "L": "L5_ankle",
    "R": "R5_ankle",
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


# ============================================================
# GUI
# ============================================================

USE_GUI = True

VIEWER_SLEEP = 0.001


# ============================================================
# PRINT CHECKPOINTS
# ============================================================

SWING_PRINT_PHASES = (
    0.00,
    0.25,
    0.50,
    0.75,
)


# ============================================================
# MUJOCO HELPERS
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


def get_actuator_id(
    model,
    name,
):

    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if object_id < 0:
        raise ValueError(
            f"Actuator '{name}' not found."
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
# EULER -> QUATERNION
#
# Euler convention:
#
# roll, pitch, yaw
#
# Rotation:
#
# R_WB = Rz(yaw) Ry(pitch) Rx(roll)
#
# MuJoCo quaternion:
#
# [w, x, y, z]
# ============================================================

def euler_to_quaternion_wxyz(
    euler,
):

    euler = np.asarray(
        euler,
        dtype=float,
    ).reshape(
        3
    )

    roll = float(
        euler[0]
    )

    pitch = float(
        euler[1]
    )

    yaw = float(
        euler[2]
    )


    cr = np.cos(
        0.5 * roll
    )

    sr = np.sin(
        0.5 * roll
    )

    cp = np.cos(
        0.5 * pitch
    )

    sp = np.sin(
        0.5 * pitch
    )

    cy = np.cos(
        0.5 * yaw
    )

    sy = np.sin(
        0.5 * yaw
    )


    quaternion = np.array(
        [
            cr * cp * cy
            +
            sr * sp * sy,

            sr * cp * cy
            -
            cr * sp * sy,

            cr * sp * cy
            +
            sr * cp * sy,

            cr * cp * sy
            -
            sr * sp * cy,
        ],
        dtype=float,
    )


    quaternion /= np.linalg.norm(
        quaternion
    )

    return quaternion


# ============================================================
# LEG JOINT STATE
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
# SET LEG JOINT POSITION
# ============================================================

def set_leg_joint_position(
    model,
    data,
    leg,
    q_leg,
):

    q_leg = np.asarray(
        q_leg,
        dtype=float,
    ).reshape(
        5
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

        data.qpos[
            qpos_address
        ] = q_leg[
            i
        ]


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
# INITIALIZE MUJOCO
# ============================================================

def set_initial_state(
    model,
    data,
):

    mujoco.mj_resetData(
        model,
        data,
    )


    set_leg_joint_position(
        model,
        data,
        "L",
        INITIAL_Q_LEFT,
    )


    set_leg_joint_position(
        model,
        data,
        "R",
        INITIAL_Q_RIGHT,
    )


    data.qvel[:] = 0.0

    data.ctrl[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


# ============================================================
# PRESCRIBE FLOATING BASE
#
# IMPORTANT:
#
# This is ONLY an integration/visualization test.
#
# The base is not moved by SRBD-MPC.
#
# Instead:
#
# Reference Generator
#        ->
# prescribed floating-base state
#
# Later this function will be removed.
#
# SRBD-MPC + contact wrench will then move the base physically.
# ============================================================

def prescribe_floating_base(
    model,
    data,
    x_base_ref,
):

    x_base_ref = np.asarray(
        x_base_ref,
        dtype=float,
    ).reshape(
        13
    )


    freejoint_id = get_joint_id(
        model,
        FLOATING_BASE_JOINT,
    )


    qpos_address = int(
        model.jnt_qposadr[
            freejoint_id
        ]
    )

    dof_address = int(
        model.jnt_dofadr[
            freejoint_id
        ]
    )


    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    data.qpos[
        qpos_address
        :
        qpos_address
        +
        3
    ] = x_base_ref[
        0:3
    ]


    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    quaternion = (
        euler_to_quaternion_wxyz(
            x_base_ref[
                3:6
            ]
        )
    )


    data.qpos[
        qpos_address
        +
        3
        :
        qpos_address
        +
        7
    ] = quaternion


    # --------------------------------------------------------
    # LINEAR VELOCITY
    # --------------------------------------------------------

    data.qvel[
        dof_address
        :
        dof_address
        +
        3
    ] = x_base_ref[
        6:9
    ]


    # --------------------------------------------------------
    # ANGULAR VELOCITY
    # --------------------------------------------------------

    data.qvel[
        dof_address
        +
        3
        :
        dof_address
        +
        6
    ] = x_base_ref[
        9:12
    ]


    mujoco.mj_forward(
        model,
        data,
    )


# ============================================================
# REFERENCE COMMAND
#
# Startup:
#
#     v = 0
#
# After STARTUP_HOLD_TIME:
#
#     v = target
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
# PROPAGATE PRESCRIBED BASE
#
# ReferenceGenerator updates every 5 ms.
#
# Between updates:
#
# use current reference velocities to propagate at 1 ms.
# ============================================================

def propagate_prescribed_base_state(
    x_base_ref,
    X_ref,
):

    x_next = (
        x_base_ref.copy()
    )


    reference_now = np.asarray(
        X_ref[
            0
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    x_next[
        0:3
    ] += (

        reference_now[
            6:9
        ]

        *

        CONTROL_DT
    )


    x_next[
        2
    ] = reference_now[
        2
    ]


    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    x_next[
        3
    ] = reference_now[
        3
    ]

    x_next[
        4
    ] = reference_now[
        4
    ]

    x_next[
        5
    ] += (

        reference_now[
            11
        ]

        *

        CONTROL_DT
    )


    # --------------------------------------------------------
    # VELOCITIES
    # --------------------------------------------------------

    x_next[
        6:9
    ] = reference_now[
        6:9
    ]


    x_next[
        9:12
    ] = reference_now[
        9:12
    ]


    x_next[
        12
    ] = 1.0


    return x_next


# ============================================================
# ANKLE COMPENSATION
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


    q_ankle_des = (

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


    return float(
        q_ankle_des
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
# DESIRED JOINT VELOCITY
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


    # --------------------------------------------------------
    # FIRST SAMPLE
    # --------------------------------------------------------

    if previous is None:

        q_dot_des = np.zeros(
            5,
            dtype=float,
        )


    # --------------------------------------------------------
    # FINITE DIFFERENCE
    # --------------------------------------------------------

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
# NUMERICAL IK TARGET
#
# Works for:
#
# swing target
#
# or
#
# temporary fixed world stance contact
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

        model=
            model,

        data=
            data,

        W_p_ankle_des=
            W_p_ankle_des,

        swing_foot=
            leg,

        sigma_threshold=
            SIGMA_THRESHOLD,

        alpha_newton=
            ALPHA_NEWTON,

        eta_gradient=
            ETA_GRADIENT,

        tolerance=
            IK_TOLERANCE,

        max_iterations=
            IK_MAX_ITERATIONS,

        hip_yaw_fixed=
            HIP_YAW_FIXED,
    )


    if q_ik is None:

        raise RuntimeError(
            f"IK returned None for leg {leg}."
        )


    if not diagnostics[
        "converged"
    ]:

        raise RuntimeError(

            f"IK did not converge for leg {leg}. "
            f"Target = {W_p_ankle_des}, "
            f"error = "
            f"{diagnostics['final_position_error_norm']:.6e}"
        )


    (
        q_actual,
        q_dot_actual,

    ) = get_leg_joint_state(
        model,
        data,
        leg,
    )


    q_ankle_des = (
        compute_ankle_desired(
            q_actual
        )
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
# RAIBERT NOMINAL ANCHOR OFFSET
#
# Computed from initial base/ankle geometry.
#
# Vertical component removed.
# ============================================================

def compute_raibert_anchor_offset_B(
    model,
    data,
    leg,
):

    W_p_ankle = (
        get_ankle_position(
            model,
            data,
            leg,
        )
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


    W_p_anchor = (

        W_p_base

        +

        W_R_B

        @

        B_r_anchor
    )


    return W_p_anchor


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

        "printed_checkpoints":
            set(),
    }


# ============================================================
# COMPUTE SWING TARGET
#
# Gait
#   ->
# Raibert
#   ->
# Bezier
#   ->
# IK
# ============================================================

def compute_swing_target(
    model,
    data,
    leg,
    swing_phase,
    T_swing,
    context,
    B_r_anchor,
    x_base_ref,
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


    # --------------------------------------------------------
    # REMAINING SWING TIME
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RAIBERT ANCHOR
    # --------------------------------------------------------

    W_p_anchor = (
        compute_raibert_anchor(

            model=
                model,

            data=
                data,

            B_r_anchor=
                B_r_anchor,
        )
    )


    # --------------------------------------------------------
    # ACTUAL / REFERENCE VELOCITY
    #
    # Since the base is prescribed,
    #
    # actual velocity = prescribed reference velocity.
    # --------------------------------------------------------

    W_v_actual = (
        x_base_ref[
            6:9
        ].copy()
    )


    W_v_ref = np.asarray(
        ref_output[
            "v_des_world"
        ],
        dtype=float,
    ).copy()


    # --------------------------------------------------------
    # RAIBERT
    # --------------------------------------------------------

    raibert_result = (
        compute_raibert_foothold(

            W_p_hip=
                W_p_anchor,

            W_v_actual=
                W_v_actual,

            W_v_ref=
                W_v_ref,

            T_remaining=
                T_remaining,

            foothold_z=
                context[
                    "foothold_z"
                ],

            leg=
                leg,
        )
    )


    W_p_foothold = np.asarray(
        raibert_result[
            "W_p_foothold"
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # BEZIER
    # --------------------------------------------------------

    bezier_result = (
        evaluate_bezier_swing_trajectory(

            W_p_liftoff=
                context[
                    "W_p_liftoff"
                ],

            W_p_foothold=
                W_p_foothold,

            swing_phase=
                s,

            T_swing=
                T_swing,

            swing_height=
                SWING_HEIGHT,

            cp1_coef=
                CP1_COEF,

            cp2_coef=
                CP2_COEF,
        )
    )


    W_p_des = np.asarray(
        bezier_result[
            "W_p_swing_des"
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # IK
    # --------------------------------------------------------

    joint_target = (
        solve_leg_cartesian_target(

            model=
                model,

            data=
                data,

            leg=
                leg,

            W_p_ankle_des=
                W_p_des,
        )
    )


    # --------------------------------------------------------
    # ACTUAL POSITION
    # --------------------------------------------------------

    W_p_actual = (
        get_ankle_position(
            model,
            data,
            leg,
        )
    )


    cartesian_error = float(
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
            cartesian_error,

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
# TORQUE CLIPPING
# ============================================================

def clip_leg_torque(
    model,
    leg,
    tau_raw,
):

    tau_raw = np.asarray(
        tau_raw,
        dtype=float,
    ).reshape(
        5
    )


    tau_applied = (
        tau_raw.copy()
    )


    clipped = np.zeros(
        5,
        dtype=bool,
    )


    for i, actuator_name in enumerate(
        LEG_JOINT_NAMES[
            leg
        ]
    ):

        actuator_id = (
            get_actuator_id(
                model,
                actuator_name,
            )
        )


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


            applied = float(
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
                applied,
                tau_raw[
                    i
                ],
            )


            tau_applied[
                i
            ] = applied


    return (
        tau_applied,
        clipped,
    )


# ============================================================
# WRITE TORQUE
# ============================================================

def write_leg_torque(
    model,
    data,
    leg,
    tau,
):

    tau = np.asarray(
        tau,
        dtype=float,
    ).reshape(
        5
    )


    for i, actuator_name in enumerate(
        LEG_JOINT_NAMES[
            leg
        ]
    ):

        actuator_id = (
            get_actuator_id(
                model,
                actuator_name,
            )
        )


        data.ctrl[
            actuator_id
        ] = tau[
            i
        ]


# ============================================================
# STATISTICS
# ============================================================

def create_leg_statistics():

    return {

        "completed_swings":
            0,

        "swing_samples":
            0,

        "sum_cart_error_sq":
            0.0,

        "max_cart_error":
            0.0,

        "peak_tau":
            np.zeros(
                5,
                dtype=float,
            ),

        "clip_count":
            np.zeros(
                5,
                dtype=int,
            ),

        "last_foothold":
            None,
    }


# ============================================================
# PRINT SWING SAMPLE
# ============================================================

def print_swing_sample(
    t,
    leg,
    target,
    tau_applied,
    clipped,
):

    print(
        "-" * 110
    )

    print(
        f"t = {t:.3f} s "
        f"| leg = {leg} "
        f"| swing phase = "
        f"{target['phase']:.3f}"
    )

    print(
        f"T_remaining     : "
        f"{target['T_remaining']:.4f} s"
    )

    print(
        f"Raibert anchor  : "
        f"{target['W_p_anchor']}"
    )

    print(
        f"Raibert Pf      : "
        f"{target['W_p_foothold']}"
    )

    print(
        f"Bezier desired  : "
        f"{target['W_p_des']}"
    )

    print(
        f"Actual ankle    : "
        f"{target['W_p_actual']}"
    )

    print(
        f"Cartesian error : "
        f"{target['cart_error'] * 1000.0:.3f} mm"
    )

    print(
        f"tau applied     : "
        f"{tau_applied}"
    )

    print(
        f"torque clipped  : "
        f"{clipped}"
    )


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(
    model,
    data,
    viewer,
):

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
        f"MuJoCo physics dt       : "
        f"{physics_dt:.6f} s"
    )

    print(
        f"Physics substeps        : "
        f"{physics_substeps}"
    )


    # ========================================================
    # INITIAL STATE
    # ========================================================

    set_initial_state(
        model,
        data,
    )


    # ========================================================
    # PRESCRIBED BASE STATE
    #
    # [
    # p,
    # euler,
    # v,
    # w,
    # 1
    # ]
    # ========================================================

    x_base_ref = np.array(
        [
            0.0,
            0.0,
            Z_DES,

            0.0,
            0.0,
            0.0,

            0.0,
            0.0,
            0.0,

            0.0,
            0.0,
            0.0,

            1.0,
        ],
        dtype=float,
    )


    prescribe_floating_base(
        model,
        data,
        x_base_ref,
    )


    # ========================================================
    # GAIT
    # ========================================================

    gait = GaitScheduler(

        mpc_horizon=
            MPC_HORIZON,

        dsp_durations=
            DSP_DURATIONS,

        ssp_durations=
            SSP_DURATIONS,

        control_dt=
            CONTROL_DT,

        dt_mpc=
            MPC_DT,

        startup_hold_time=
            STARTUP_HOLD_TIME,

        first_swing_leg=
            FIRST_SWING_LEG,
    )


    # ========================================================
    # REFERENCE GENERATOR
    # ========================================================

    reference_generator = ReferenceGenerator(

        horizon_length=
            MPC_HORIZON,

        dt_mpc=
            MPC_DT,

        reference_update_dt=
            REFERENCE_UPDATE_DT,
    )


    # ========================================================
    # INITIAL STANCE CONTACT
    #
    # World ankle-center position.
    # ========================================================

    stance_contact = {

        "L":
            get_ankle_position(
                model,
                data,
                "L",
            ),

        "R":
            get_ankle_position(
                model,
                data,
                "R",
            ),
    }


    print(
        f"Initial LEFT contact    : "
        f"{stance_contact['L']}"
    )

    print(
        f"Initial RIGHT contact   : "
        f"{stance_contact['R']}"
    )


    # ========================================================
    # RAIBERT ANCHOR OFFSET
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
    # CONTEXT
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


    stats = {

        "L":
            create_leg_statistics(),

        "R":
            create_leg_statistics(),
    }


    # ========================================================
    # INTERNAL STATE
    # ========================================================

    ref_output = None

    X_ref = None

    previous_phase_name = None

    previous_active_swing_leg = None


    number_of_steps = int(
        round(
            SIMULATION_TIME
            /
            CONTROL_DT
        )
    )


    # ========================================================
    # MAIN LOOP
    #
    # x_k
    #   ->
    # gait_k
    #   ->
    # reference_k
    #   ->
    # compute u_k
    #   ->
    # apply u_k
    #   ->
    # mj_step()
    #   ->
    # x_(k+1)
    #   ->
    # gait.advance()
    # ========================================================

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


        # ====================================================
        # 1. READ GAIT_k
        # ====================================================

        gait_output = (
            gait.get_output()
        )


        phase_name = (
            gait_output[
                "phase_name"
            ]
        )


        active_swing_leg = (
            gait_output[
                "active_swing_leg"
            ]
        )


        # ====================================================
        # 2. COMMAND
        # ====================================================

        (
            v_des_body,
            yaw_rate_des,

        ) = get_reference_command(
            t
        )


        # ====================================================
        # 3. REFERENCE GENERATOR
        #
        # 200 Hz
        # ====================================================

        update_reference = (

            ref_output is None

            or

            (
                k
                %
                REFERENCE_DECIMATION
                ==
                0
            )
        )


        if update_reference:

            ref_output = (
                reference_generator.update(

                    x_actual=
                        x_base_ref,

                    v_des_body=
                        v_des_body,

                    yaw_rate_des=
                        yaw_rate_des,

                    z_des=
                        Z_DES,

                    roll_des=
                        ROLL_DES,

                    pitch_des=
                        PITCH_DES,
                )
            )


            X_ref = np.asarray(
                ref_output[
                    "X_ref"
                ],
                dtype=float,
            )


            # ------------------------------------------------
            # Reference knot 0 becomes prescribed base state.
            # ------------------------------------------------

            x_base_ref[
                0:6
            ] = X_ref[
                0,
                0:6
            ]


            x_base_ref[
                6:12
            ] = X_ref[
                0,
                6:12
            ]


            x_base_ref[
                12
            ] = 1.0


        # ====================================================
        # 4. PRESCRIBE BASE AT x_k
        # ====================================================

        prescribe_floating_base(
            model,
            data,
            x_base_ref,
        )


        # ====================================================
        # 5. PRINT GAIT TRANSITION
        # ====================================================

        if (
            phase_name
            !=
            previous_phase_name
        ):

            print()

            print(
                "=" * 110
            )

            print(
                f"GAIT TRANSITION @ "
                f"t = {t:.3f} s"
            )

            print(
                "=" * 110
            )

            print(
                f"phase_name      : "
                f"{phase_name}"
            )

            print(
                f"contact_state   : "
                f"{gait_output['contact_state']}"
            )

            print(
                f"swing_state     : "
                f"{gait_output['swing_state']}"
            )

            print(
                f"active swing    : "
                f"{active_swing_leg}"
            )

            print(
                f"base ref p      : "
                f"{x_base_ref[0:3]}"
            )

            print(
                f"v_des_world     : "
                f"{ref_output['v_des_world']}"
            )


            previous_phase_name = (
                phase_name
            )


        # ====================================================
        # 6. TOUCHDOWN EVENT
        #
        # Previous:
        #
        # swing leg = L/R
        #
        # Current:
        #
        # no swing or other leg
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


            if swing_context[
                touchdown_leg
            ][
                "active"
            ]:

                leg_index = (
                    LEG_INDEX[
                        touchdown_leg
                    ]
                )


                T_swing = float(
                    gait_output[
                        "T_swing"
                    ][
                        leg_index
                    ]
                )


                final_target = (
                    compute_swing_target(

                        model=
                            model,

                        data=
                            data,

                        leg=
                            touchdown_leg,

                        swing_phase=
                            1.0,

                        T_swing=
                            T_swing,

                        context=
                            swing_context[
                                touchdown_leg
                            ],

                        B_r_anchor=
                            B_r_anchor[
                                touchdown_leg
                            ],

                        x_base_ref=
                            x_base_ref,

                        ref_output=
                            ref_output,
                    )
                )


                # --------------------------------------------
                # New WORLD stance contact
                # --------------------------------------------

                stance_contact[
                    touchdown_leg
                ] = final_target[
                    "W_p_foothold"
                ].copy()


                swing_context[
                    touchdown_leg
                ][
                    "active"
                ] = False


                stats[
                    touchdown_leg
                ][
                    "completed_swings"
                ] += 1


                stats[
                    touchdown_leg
                ][
                    "last_foothold"
                ] = stance_contact[
                    touchdown_leg
                ].copy()


                reset_joint_reference_memory(
                    joint_memory[
                        touchdown_leg
                    ]
                )


                print(
                    "-" * 110
                )

                print(
                    f"TOUCHDOWN @ "
                    f"t = {t:.3f} s "
                    f"| leg = {touchdown_leg}"
                )

                print(
                    f"Planned contact : "
                    f"{stance_contact[touchdown_leg]}"
                )

                print(
                    f"Actual ankle    : "
                    f"{final_target['W_p_actual']}"
                )

                print(
                    f"Final error     : "
                    f"{final_target['cart_error'] * 1000.0:.3f} mm"
                )


        # ====================================================
        # 7. LIFTOFF EVENT
        # ====================================================

        if (
            active_swing_leg
            is not None

            and

            active_swing_leg
            !=
            previous_active_swing_leg
        ):

            W_p_liftoff = (
                get_ankle_position(
                    model,
                    data,
                    active_swing_leg,
                )
            )


            swing_context[
                active_swing_leg
            ][
                "active"
            ] = True


            swing_context[
                active_swing_leg
            ][
                "W_p_liftoff"
            ] = W_p_liftoff.copy()


            swing_context[
                active_swing_leg
            ][
                "foothold_z"
            ] = float(
                W_p_liftoff[
                    2
                ]
            )


            swing_context[
                active_swing_leg
            ][
                "printed_checkpoints"
            ] = set()


            reset_joint_reference_memory(
                joint_memory[
                    active_swing_leg
                ]
            )


            print()

            print(
                "#" * 110
            )

            print(
                f"LIFTOFF @ "
                f"t = {t:.3f} s "
                f"| leg = {active_swing_leg}"
            )

            print(
                f"W_p_liftoff = "
                f"{W_p_liftoff}"
            )

            print(
                f"Base position = "
                f"{x_base_ref[0:3]}"
            )

            print(
                "#" * 110
            )


        # ====================================================
        # 8. COMPUTE u_k
        # ====================================================

        data.ctrl[:] = 0.0


        for leg in (
            "L",
            "R",
        ):

            # =================================================
            # SWING LEG
            # =================================================

            if leg == active_swing_leg:

                leg_index = (
                    LEG_INDEX[
                        leg
                    ]
                )


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


                target = (
                    compute_swing_target(

                        model=
                            model,

                        data=
                            data,

                        leg=
                            leg,

                        swing_phase=
                            swing_phase,

                        T_swing=
                            T_swing,

                        context=
                            swing_context[
                                leg
                            ],

                        B_r_anchor=
                            B_r_anchor[
                                leg
                            ],

                        x_base_ref=
                            x_base_ref,

                        ref_output=
                            ref_output,
                    )
                )


                # =============================================
                # QDOT DES
                # =============================================

                q_dot_des = (
                    update_q_dot_des(

                        q_des=
                            target[
                                "q_des"
                            ],

                        memory=
                            joint_memory[
                                leg
                            ],
                    )
                )


                # =============================================
                # PD
                # =============================================

                tau_raw = (
                    compute_swing_joint_PD(

                        q_des=
                            target[
                                "q_des"
                            ],

                        q=
                            target[
                                "q_actual"
                            ],

                        q_dot=
                            target[
                                "q_dot_actual"
                            ],

                        Kp=
                            KP_SWING,

                        Kd=
                            KD_SWING,

                        q_dot_des=
                            q_dot_des,
                    )
                )


                # =============================================
                # CLIP
                # =============================================

                (
                    tau_applied,
                    clipped,

                ) = clip_leg_torque(

                    model=
                        model,

                    leg=
                        leg,

                    tau_raw=
                        tau_raw,
                )


                # =============================================
                # WRITE u_k
                # =============================================

                write_leg_torque(

                    model=
                        model,

                    data=
                        data,

                    leg=
                        leg,

                    tau=
                        tau_applied,
                )


                # =============================================
                # STATS
                # =============================================

                stats[
                    leg
                ][
                    "swing_samples"
                ] += 1


                stats[
                    leg
                ][
                    "sum_cart_error_sq"
                ] += (

                    target[
                        "cart_error"
                    ]
                    **
                    2
                )


                stats[
                    leg
                ][
                    "max_cart_error"
                ] = max(

                    stats[
                        leg
                    ][
                        "max_cart_error"
                    ],

                    target[
                        "cart_error"
                    ],
                )


                stats[
                    leg
                ][
                    "peak_tau"
                ] = np.maximum(

                    stats[
                        leg
                    ][
                        "peak_tau"
                    ],

                    np.abs(
                        tau_applied
                    ),
                )


                stats[
                    leg
                ][
                    "clip_count"
                ] += np.asarray(
                    clipped,
                    dtype=int,
                )


                # =============================================
                # PRINT CHECKPOINT
                # =============================================

                printed = swing_context[
                    leg
                ][
                    "printed_checkpoints"
                ]


                for checkpoint in (
                    SWING_PRINT_PHASES
                ):

                    if (
                        checkpoint
                        not in
                        printed

                        and

                        swing_phase
                        >=
                        checkpoint
                        -
                        1e-10
                    ):

                        print_swing_sample(

                            t=
                                t,

                            leg=
                                leg,

                            target=
                                target,

                            tau_applied=
                                tau_applied,

                            clipped=
                                clipped,
                        )


                        printed.add(
                            checkpoint
                        )


            # =================================================
            # STANCE LEG
            #
            # TEMPORARY:
            #
            # keep ankle center fixed in WORLD.
            #
            # Later:
            #
            # SRBD-MPC wrench -> stance joint torque.
            # =================================================

            else:

                stance_target = (
                    solve_leg_cartesian_target(

                        model=
                            model,

                        data=
                            data,

                        leg=
                            leg,

                        W_p_ankle_des=
                            stance_contact[
                                leg
                            ],
                    )
                )


                q_dot_des = (
                    update_q_dot_des(

                        q_des=
                            stance_target[
                                "q_des"
                            ],

                        memory=
                            joint_memory[
                                leg
                            ],
                    )
                )


                tau_raw = (
                    compute_swing_joint_PD(

                        q_des=
                            stance_target[
                                "q_des"
                            ],

                        q=
                            stance_target[
                                "q_actual"
                            ],

                        q_dot=
                            stance_target[
                                "q_dot_actual"
                            ],

                        Kp=
                            KP_SWING,

                        Kd=
                            KD_SWING,

                        q_dot_des=
                            q_dot_des,
                    )
                )


                (
                    tau_applied,
                    clipped,

                ) = clip_leg_torque(

                    model=
                        model,

                    leg=
                        leg,

                    tau_raw=
                        tau_raw,
                )


                write_leg_torque(

                    model=
                        model,

                    data=
                        data,

                    leg=
                        leg,

                    tau=
                        tau_applied,
                )


        # ====================================================
        # 9. VIEWER
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
        # LAST SAMPLE
        # ====================================================

        if k == number_of_steps:

            break


        # ====================================================
        # 10. PHYSICS
        #
        # x_k
        #   ->
        # apply u_k
        #   ->
        # mj_step()
        #   ->
        # physical x_(k+1)
        #
        # Same torque is held for all MuJoCo substeps.
        # ====================================================

        for _ in range(
            physics_substeps
        ):

            mujoco.mj_step(
                model,
                data,
            )


        # ====================================================
        # 11. PROPAGATE PRESCRIBED BASE REFERENCE
        #
        # This becomes base state at the next control step.
        # ====================================================

        x_base_ref = (
            propagate_prescribed_base_state(

                x_base_ref=
                    x_base_ref,

                X_ref=
                    X_ref,
            )
        )


        # ====================================================
        # 12. STORE ACTIVE SWING LEG
        # ====================================================

        previous_active_swing_leg = (
            active_swing_leg
        )


        # ====================================================
        # 13. ADVANCE GAIT
        # ====================================================

        gait.advance()


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "x_base_ref":
            x_base_ref.copy(),

        "stats":
            stats,

        "stance_contact": {

            "L":
                stance_contact[
                    "L"
                ].copy(),

            "R":
                stance_contact[
                    "R"
                ].copy(),
        },
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


    x_base_ref = results[
        "x_base_ref"
    ]


    print()

    print(
        "=" * 110
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 110
    )


    print(
        f"Final prescribed base position : "
        f"{x_base_ref[0:3]}"
    )


    joint_labels = [
        "hip_yaw",
        "hip_roll",
        "hip_pitch",
        "knee",
        "ankle",
    ]


    for leg in (
        "R",
        "L",
    ):

        leg_stats = (
            stats[
                leg
            ]
        )


        samples = int(
            leg_stats[
                "swing_samples"
            ]
        )


        if samples > 0:

            cartesian_rmse = np.sqrt(

                leg_stats[
                    "sum_cart_error_sq"
                ]

                /

                samples
            )

        else:

            cartesian_rmse = np.nan


        print()

        print(
            f"LEG {leg}"
        )

        print(
            "-" * 110
        )


        print(
            f"Completed swings        : "
            f"{leg_stats['completed_swings']}"
        )

        print(
            f"Swing samples           : "
            f"{samples}"
        )

        print(
            f"Cartesian RMSE          : "
            f"{cartesian_rmse * 1000.0:.3f} mm"
        )

        print(
            f"Maximum Cartesian error : "
            f"{leg_stats['max_cart_error'] * 1000.0:.3f} mm"
        )

        print(
            f"Last foothold           : "
            f"{leg_stats['last_foothold']}"
        )


        print()

        print(
            "Peak |tau applied| [N.m]"
        )


        for i, label in enumerate(
            joint_labels
        ):

            print(
                f"  {label:<12}: "
                f"{leg_stats['peak_tau'][i]:.6f}"
            )


        print()

        print(
            "Swing torque clipping count"
        )


        for i, label in enumerate(
            joint_labels
        ):

            print(
                f"  {label:<12}: "
                f"{leg_stats['clip_count'][i]}"
            )


    print()

    print(
        "=" * 110
    )

    print(
        "TEST FINISHED"
    )

    print(
        "=" * 110
    )


    print(
        "Floating base:"
    )

    print(
        "  prescribed directly from Reference Generator."
    )


    print()

    print(
        "Swing leg:"
    )

    print(
        "  Gait -> Raibert -> Bezier -> IK "
        "-> ankle compensation -> PD -> torque."
    )


    print()

    print(
        "Stance leg:"
    )

    print(
        "  temporary WORLD-contact IK + PD."
    )


    print()

    print(
        "SRBD-MPC:"
    )

    print(
        "  NOT active."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 110
    )

    print(
        "TEST: GAIT + REFERENCE GENERATOR "
        "+ SWING CONTROL + PRESCRIBED FLOATING BASE"
    )

    print(
        "=" * 110
    )


    print(
        f"XML                     : "
        f"{XML_PATH}"
    )

    print(
        "Floating base           : YES"
    )

    print(
        "Base motion             : PRESCRIBED FROM REFERENCE"
    )

    print(
        "SRBD-MPC                : NO"
    )

    print(
        "Temporary stance        : WORLD-CONTACT IK + PD"
    )

    print(
        f"Control dt              : "
        f"{CONTROL_DT:.6f} s"
    )

    print(
        f"Control frequency       : "
        f"{1.0 / CONTROL_DT:.1f} Hz"
    )

    print(
        f"Reference update dt     : "
        f"{REFERENCE_UPDATE_DT:.6f} s"
    )

    print(
        f"Reference frequency     : "
        f"{1.0 / REFERENCE_UPDATE_DT:.1f} Hz"
    )

    print(
        f"Reference decimation    : "
        f"{REFERENCE_DECIMATION}"
    )

    print(
        f"MPC prediction dt       : "
        f"{MPC_DT:.6f} s"
    )

    print(
        f"MPC horizon             : "
        f"{MPC_HORIZON}"
    )

    print(
        f"SSP                     : "
        f"{SSP_DURATIONS}"
    )

    print(
        f"DSP                     : "
        f"{DSP_DURATIONS}"
    )

    print(
        f"Startup hold            : "
        f"{STARTUP_HOLD_TIME:.3f} s"
    )

    print(
        f"Velocity target         : "
        f"{V_DES_BODY_TARGET}"
    )

    print(
        f"Velocity starts at      : "
        f"{STARTUP_HOLD_TIME:.3f} s"
    )

    print(
        f"Swing height            : "
        f"{SWING_HEIGHT:.4f} m"
    )

    print(
        f"KP_SWING                : "
        f"{KP_SWING}"
    )

    print(
        f"KD_SWING                : "
        f"{KD_SWING}"
    )

    print()


    # ========================================================
    # XML
    # ========================================================

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"XML not found:\n{XML_PATH}"
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(
            XML_PATH
        )
    )


    data = mujoco.MjData(
        model
    )


    # ========================================================
    # RUN
    # ========================================================

    if USE_GUI:

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            results = run_simulation(

                model=
                    model,

                data=
                    data,

                viewer=
                    viewer,
            )


    else:

        results = run_simulation(

            model=
                model,

            data=
                data,

            viewer=
                None,
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        results
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()