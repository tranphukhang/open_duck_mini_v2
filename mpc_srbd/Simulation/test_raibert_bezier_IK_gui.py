from pathlib import Path
import os
import sys
import time
import tempfile

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


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
# TEST SETTINGS
# ============================================================

SWING_LEG = "R"


# ============================================================
# CONTROL LOOP
#
# Paper-style low-level control:
#
#     dt_control = 0.001 s
#
# XML physics:
#
#     dt_physics = 0.0005 s
#
# Therefore:
#
#     x_k
#       ->
#     calculate u_k
#       ->
#     apply u_k for 2 MuJoCo substeps
#       ->
#     x_(k+1)
# ============================================================

CONTROL_DT = 0.001

T_SWING = 0.24

N_SAMPLES = (
    int(
        round(
            T_SWING
            /
            CONTROL_DT
        )
    )
    +
    1
)


# ============================================================
# GUI
#
# Only wall-clock display is slowed down.
#
# Controller mathematics remains at 1000 Hz.
# ============================================================

DISPLAY_TIME_SCALE = 10.0

GUI_START_HOLD = 2.0

GUI_END_HOLD = 3.0


# ============================================================
# BEZIER
# ============================================================

SWING_HEIGHT = 0.008

CP1_COEF = 0.333

CP2_COEF = 0.666


# ============================================================
# VELOCITY REFERENCE
# ============================================================

W_V_REF = np.array(
    [
        0.08,
        0.00,
        0.00,
    ],
    dtype=float,
)


# ============================================================
# EMULATED STATE-ESTIMATOR VELOCITY
#
# Same signal used in previous integration test.
# ============================================================

VX_FEEDBACK_AMPLITUDE = 0.04

VY_FEEDBACK_AMPLITUDE = 0.02


# ============================================================
# DESIRED JOINT VELOCITY
#
# qdot_des is obtained from finite difference:
#
#     qdot_des_raw =
#         (q_des(k) - q_des(k-1)) / dt
#
# then low-pass filtered.
#
# These values follow the previous swing low-level design.
# ============================================================

Q_DOT_DES_FILTER_TAU = 0.005

Q_DOT_DES_LIMIT = np.array(
    [
        6.0,    # hip yaw
        6.0,    # hip roll
        8.0,    # hip pitch
        10.0,   # knee
        8.0,    # ankle
    ],
    dtype=float,
)


# ============================================================
# FLAT GROUND
# ============================================================

SLOPE_PITCH = 0.0


# ============================================================
# INITIAL CONFIGURATION
#
# Joint order:
#
# [
#     hip yaw,
#     hip roll,
#     hip pitch,
#     knee,
#     ankle
# ]
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
# MUJOCO NAMES
# ============================================================

BASE_BODY_NAME = "base_link"


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

    "L":
        "L5_ankle",

    "R":
        "R5_ankle",
}


# ============================================================
# GUI MARKERS
#
# Transparent + different sizes so desired / actual remain
# visible when overlapping.
# ============================================================

MARKER_RADIUS_FOOTHOLD = 0.0070
MARKER_RADIUS_DESIRED = 0.0050
MARKER_RADIUS_ACTUAL = 0.0030
MARKER_RADIUS_LIFTOFF = 0.0040
MARKER_RADIUS_TRAIL = 0.0018


RGBA_FOOTHOLD = np.array(
    [
        1.0,
        0.45,
        0.05,
        0.16,
    ],
    dtype=np.float32,
)


RGBA_DESIRED = np.array(
    [
        0.10,
        0.85,
        0.20,
        0.22,
    ],
    dtype=np.float32,
)


RGBA_ACTUAL = np.array(
    [
        0.10,
        0.35,
        1.00,
        0.38,
    ],
    dtype=np.float32,
)


RGBA_LIFTOFF = np.array(
    [
        0.70,
        0.70,
        0.70,
        0.14,
    ],
    dtype=np.float32,
)


RGBA_TRAIL_DES = np.array(
    [
        0.10,
        0.85,
        0.20,
        0.07,
    ],
    dtype=np.float32,
)


RGBA_TRAIL_ACT = np.array(
    [
        0.10,
        0.35,
        1.00,
        0.09,
    ],
    dtype=np.float32,
)


TRAIL_DECIMATION = 8

MAX_TRAIL_POINTS = 40


# ============================================================
# TEMPORARY FIXED-BASE MODEL
#
# This test is specifically:
#
#     swing-reference
#       ->
#     joint PD
#       ->
#     MuJoCo joint dynamics
#
# The original XML has a floating base but no walking ground
# scene. Therefore we remove ONLY the freejoint in a temporary
# XML file.
#
# The original XML file is NOT modified.
# ============================================================

def load_fixed_base_model(
    xml_path,
):

    xml_path = Path(
        xml_path
    )

    xml_text = xml_path.read_text(
        encoding="utf-8"
    )


    freejoint_text = (
        '<freejoint name="floating_base"/>'
    )


    if freejoint_text not in xml_text:

        raise RuntimeError(
            "Không tìm thấy floating_base freejoint trong XML."
        )


    fixed_xml_text = xml_text.replace(
        freejoint_text,
        "<!-- floating_base removed for fixed-base PD test -->",
        1,
    )


    fd, temp_name = tempfile.mkstemp(
        prefix="_tmp_fixed_base_",
        suffix=".xml",
        dir=str(
            xml_path.parent
        ),
    )

    os.close(
        fd
    )


    temp_path = Path(
        temp_name
    )


    try:

        temp_path.write_text(
            fixed_xml_text,
            encoding="utf-8",
        )


        model = mujoco.MjModel.from_xml_path(
            str(
                temp_path
            )
        )


    finally:

        if temp_path.exists():

            temp_path.unlink()


    return model


# ============================================================
# NAME HELPERS
# ============================================================

def get_joint_id(
    model,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if idx < 0:

        raise ValueError(
            f"Joint '{name}' not found."
        )

    return int(
        idx
    )


def get_actuator_id(
    model,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if idx < 0:

        raise ValueError(
            f"Actuator '{name}' not found."
        )

    return int(
        idx
    )


def get_site_id(
    model,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    if idx < 0:

        raise ValueError(
            f"Site '{name}' not found."
        )

    return int(
        idx
    )


def get_body_id(
    model,
    name,
):

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if idx < 0:

        raise ValueError(
            f"Body '{name}' not found."
        )

    return int(
        idx
    )


# ============================================================
# JOINT STATE
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


        qpos_adr = int(
            model.jnt_qposadr[
                joint_id
            ]
        )


        dof_adr = int(
            model.jnt_dofadr[
                joint_id
            ]
        )


        q[
            i
        ] = data.qpos[
            qpos_adr
        ]


        q_dot[
            i
        ] = data.qvel[
            dof_adr
        ]


    return (
        q,
        q_dot,
    )


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


        qpos_adr = int(
            model.jnt_qposadr[
                joint_id
            ]
        )


        data.qpos[
            qpos_adr
        ] = q_leg[
            i
        ]


# ============================================================
# INITIAL STATE
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
# BASE / ANKLE POSE
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
# RAIBERT ANCHOR
#
# Equivalent role to paper's get_hip_offset().
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


    # Same structural idea used by paper:
    # vertical foothold is handled separately.
    B_r_anchor[
        2
    ] = 0.0


    return B_r_anchor


# ============================================================
# VELOCITY FEEDBACK
# ============================================================

def compute_velocity_feedback(
    swing_phase,
):

    s = float(
        swing_phase
    )


    W_v_actual = (
        W_V_REF.copy()
    )


    W_v_actual[
        0
    ] += (
        VX_FEEDBACK_AMPLITUDE
        *
        np.sin(
            np.pi
            *
            s
        )
    )


    W_v_actual[
        1
    ] += (
        VY_FEEDBACK_AMPLITUDE
        *
        np.sin(
            2.0
            *
            np.pi
            *
            s
        )
    )


    W_v_actual[
        2
    ] = 0.0


    return W_v_actual


# ============================================================
# ANKLE DESIRED
#
# Paper-style actual-state ankle compensation:
#
#     q_A_des =
#
#     -slope_pitch
#     -q_HP_actual
#     -q_K_actual
# ============================================================

def compute_ankle_desired_from_actual(
    q_actual,
):

    q_actual = np.asarray(
        q_actual,
        dtype=float,
    ).reshape(
        5
    )


    q_hp_actual = q_actual[
        2
    ]


    q_k_actual = q_actual[
        3
    ]


    return float(
        -SLOPE_PITCH
        -
        q_hp_actual
        -
        q_k_actual
    )


# ============================================================
# DESIRED JOINT VELOCITY FILTER
# ============================================================

def update_q_dot_des(
    q_des,
    q_des_previous,
    q_dot_des_filtered_previous,
):

    q_des = np.asarray(
        q_des,
        dtype=float,
    ).reshape(
        5
    )


    if q_des_previous is None:

        return np.zeros(
            5,
            dtype=float,
        )


    q_dot_raw = (
        q_des
        -
        q_des_previous
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


    q_dot_filtered = (
        (
            1.0
            -
            alpha
        )
        *
        q_dot_des_filtered_previous

        +

        alpha
        *
        q_dot_raw
    )


    q_dot_filtered = np.clip(
        q_dot_filtered,
        -Q_DOT_DES_LIMIT,
        +Q_DOT_DES_LIMIT,
    )


    return q_dot_filtered


# ============================================================
# ACTUATOR LIMIT
# ============================================================

def clip_leg_torque(
    model,
    leg,
    tau,
):

    tau = np.asarray(
        tau,
        dtype=float,
    ).reshape(
        5
    )


    tau_clipped = (
        tau.copy()
    )


    clipped_flags = np.zeros(
        5,
        dtype=bool,
    )


    for i, actuator_name in enumerate(
        LEG_JOINT_NAMES[
            leg
        ]
    ):

        actuator_id = get_actuator_id(
            model,
            actuator_name,
        )


        is_limited = bool(
            model.actuator_ctrllimited[
                actuator_id
            ]
        )


        if is_limited:

            ctrl_min = float(
                model.actuator_ctrlrange[
                    actuator_id,
                    0
                ]
            )


            ctrl_max = float(
                model.actuator_ctrlrange[
                    actuator_id,
                    1
                ]
            )


            clipped_value = np.clip(
                tau[
                    i
                ],
                ctrl_min,
                ctrl_max,
            )


            clipped_flags[
                i
            ] = not np.isclose(
                clipped_value,
                tau[
                    i
                ],
            )


            tau_clipped[
                i
            ] = clipped_value


    return (
        tau_clipped,
        clipped_flags,
    )


# ============================================================
# APPLY LEG TORQUE
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

        actuator_id = get_actuator_id(
            model,
            actuator_name,
        )


        data.ctrl[
            actuator_id
        ] = tau[
            i
        ]


# ============================================================
# USER-SCENE SPHERE
# ============================================================

def add_sphere_marker(
    scene,
    position,
    radius,
    rgba,
):

    if scene.ngeom >= scene.maxgeom:
        return


    geom = scene.geoms[
        scene.ngeom
    ]


    mujoco.mjv_initGeom(

        geom,

        mujoco.mjtGeom.mjGEOM_SPHERE,

        np.array(
            [
                radius,
                0.0,
                0.0,
            ],
            dtype=float,
        ),

        np.asarray(
            position,
            dtype=float,
        ),

        np.eye(
            3,
            dtype=float,
        ).reshape(
            -1
        ),

        np.asarray(
            rgba,
            dtype=np.float32,
        ),
    )


    scene.ngeom += 1


def update_gui_markers(
    viewer,
    W_p_liftoff,
    W_p_foothold,
    W_p_des,
    W_p_actual,
    desired_trail,
    actual_trail,
):

    with viewer.lock():

        scene = viewer.user_scn

        scene.ngeom = 0


        add_sphere_marker(
            scene,
            W_p_liftoff,
            MARKER_RADIUS_LIFTOFF,
            RGBA_LIFTOFF,
        )


        add_sphere_marker(
            scene,
            W_p_foothold,
            MARKER_RADIUS_FOOTHOLD,
            RGBA_FOOTHOLD,
        )


        add_sphere_marker(
            scene,
            W_p_des,
            MARKER_RADIUS_DESIRED,
            RGBA_DESIRED,
        )


        add_sphere_marker(
            scene,
            W_p_actual,
            MARKER_RADIUS_ACTUAL,
            RGBA_ACTUAL,
        )


        for point in desired_trail[
            -MAX_TRAIL_POINTS:
        ]:

            add_sphere_marker(
                scene,
                point,
                MARKER_RADIUS_TRAIL,
                RGBA_TRAIL_DES,
            )


        for point in actual_trail[
            -MAX_TRAIL_POINTS:
        ]:

            add_sphere_marker(
                scene,
                point,
                MARKER_RADIUS_TRAIL,
                RGBA_TRAIL_ACT,
            )


# ============================================================
# CAMERA
# ============================================================

def configure_camera(
    viewer,
):

    with viewer.lock():

        viewer.cam.lookat[:] = np.array(
            [
                0.02,
                0.0,
                0.16,
            ],
            dtype=float,
        )


        viewer.cam.distance = 0.50

        viewer.cam.azimuth = 135.0

        viewer.cam.elevation = -15.0


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 92
    )

    print(
        "TEST: RAIBERT -> BEZIER -> IK -> PD -> MUJOCO"
    )

    print(
        "=" * 92
    )


    if SWING_LEG not in (
        "L",
        "R",
    ):

        raise ValueError(
            "SWING_LEG must be 'L' or 'R'."
        )


    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"XML not found:\n{XML_PATH}"
        )


    # ========================================================
    # FIXED-BASE MODEL
    # ========================================================

    model = load_fixed_base_model(
        XML_PATH
    )


    data = mujoco.MjData(
        model
    )


    # ========================================================
    # PHYSICS / CONTROL RATE
    # ========================================================

    physics_dt = float(
        model.opt.timestep
    )


    substep_ratio = (
        CONTROL_DT
        /
        physics_dt
    )


    PHYSICS_SUBSTEPS = int(
        round(
            substep_ratio
        )
    )


    if (
        PHYSICS_SUBSTEPS <= 0
        or
        not np.isclose(
            substep_ratio,
            PHYSICS_SUBSTEPS,
            atol=1e-12,
            rtol=0.0,
        )
    ):

        raise RuntimeError(
            "CONTROL_DT phải là bội nguyên của MuJoCo timestep.\n"
            f"CONTROL_DT = {CONTROL_DT}\n"
            f"physics dt = {physics_dt}"
        )


    print(
        f"XML                    : {XML_PATH}"
    )

    print(
        f"Swing leg              : {SWING_LEG}"
    )

    print(
        f"Fixed-base test        : YES"
    )

    print(
        f"Control dt             : {CONTROL_DT:.6f} s"
    )

    print(
        f"Control frequency      : {1.0 / CONTROL_DT:.1f} Hz"
    )

    print(
        f"MuJoCo physics dt      : {physics_dt:.6f} s"
    )

    print(
        f"Physics substeps       : {PHYSICS_SUBSTEPS}"
    )

    print(
        f"T_swing                : {T_SWING:.3f} s"
    )

    print(
        f"Swing height           : {SWING_HEIGHT:.4f} m"
    )

    print(
        f"KP_SWING               : {KP_SWING}"
    )

    print(
        f"KD_SWING               : {KD_SWING}"
    )

    print()


    # ========================================================
    # INITIAL STATE
    # ========================================================

    set_initial_state(
        model,
        data,
    )


    STANCE_LEG = (
        "R"
        if SWING_LEG == "L"
        else
        "L"
    )


    (
        q_stance_hold,
        _,

    ) = get_leg_joint_state(
        model,
        data,
        STANCE_LEG,
    )


    W_p_liftoff = get_ankle_position(
        model,
        data,
        SWING_LEG,
    )


    (
        W_p_base_initial,
        W_R_B_initial,

    ) = get_base_pose(
        model,
        data,
    )


    B_r_raibert_anchor = (
        compute_raibert_anchor_offset_B(
            model,
            data,
            SWING_LEG,
        )
    )


    virtual_base_position = (
        W_p_base_initial.copy()
    )


    FOOTHOLD_Z = float(
        W_p_liftoff[
            2
        ]
    )


    print(
        f"Initial ankle center   : {W_p_liftoff}"
    )

    print(
        f"Raibert anchor B       : {B_r_raibert_anchor}"
    )

    print(
        f"Touchdown ankle Z      : {FOOTHOLD_Z:.6f} m"
    )

    print()


    # ========================================================
    # ACTUATOR LIMITS
    # ========================================================

    print(
        "Swing actuator ctrl ranges:"
    )


    for actuator_name in LEG_JOINT_NAMES[
        SWING_LEG
    ]:

        actuator_id = get_actuator_id(
            model,
            actuator_name,
        )


        print(
            f"  {actuator_name:<14} "
            f"{model.actuator_ctrlrange[actuator_id]}"
        )


    print()


    # ========================================================
    # LOGS
    # ========================================================

    time_log = []

    phase_log = []

    Trem_log = []

    velocity_log = []

    foothold_log = []

    desired_xyz_log = []

    actual_xyz_log = []

    cart_error_log = []

    q_des_log = []

    q_actual_log = []

    q_dot_des_log = []

    q_dot_actual_log = []

    tau_raw_log = []

    tau_applied_log = []

    tau_clip_log = []

    sigma_log = []

    iteration_log = []


    desired_trail = []

    actual_trail = []


    q_des_previous = None

    q_dot_des_filtered = np.zeros(
        5,
        dtype=float,
    )


    PRINT_INDICES = {
        0,
        int(
            round(
                0.25
                *
                T_SWING
                /
                CONTROL_DT
            )
        ),
        int(
            round(
                0.50
                *
                T_SWING
                /
                CONTROL_DT
            )
        ),
        int(
            round(
                0.75
                *
                T_SWING
                /
                CONTROL_DT
            )
        ),
        N_SAMPLES - 1,
    }


    # ========================================================
    # GUI
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        configure_camera(
            viewer
        )


        # ----------------------------------------------------
        # INITIAL HOLD
        # ----------------------------------------------------

        hold_start = time.time()


        while (
            viewer.is_running()
            and
            time.time()
            -
            hold_start
            <
            GUI_START_HOLD
        ):

            W_p_actual = get_ankle_position(
                model,
                data,
                SWING_LEG,
            )


            update_gui_markers(

                viewer=
                    viewer,

                W_p_liftoff=
                    W_p_liftoff,

                W_p_foothold=
                    W_p_liftoff,

                W_p_des=
                    W_p_liftoff,

                W_p_actual=
                    W_p_actual,

                desired_trail=
                    [],

                actual_trail=
                    [],
            )


            viewer.sync()

            time.sleep(
                0.01
            )


        # ====================================================
        # CONTROL LOOP
        #
        # x_k
        #  ->
        # calculate references
        #  ->
        # calculate tau_k
        #  ->
        # apply tau_k
        #  ->
        # mj_step()
        #  ->
        # x_(k+1)
        # ====================================================

        for k in range(
            N_SAMPLES
        ):

            if not viewer.is_running():
                break


            # =================================================
            # x_k / GAIT TIME
            # =================================================

            t = min(
                k
                *
                CONTROL_DT,
                T_SWING,
            )


            swing_phase = np.clip(
                t
                /
                T_SWING,
                0.0,
                1.0,
            )


            T_remaining = max(
                T_SWING
                -
                t,
                0.0,
            )


            # =================================================
            # VELOCITY FEEDBACK
            # =================================================

            W_v_actual = (
                compute_velocity_feedback(
                    swing_phase
                )
            )


            # =================================================
            # VIRTUAL MOVING BASE FOR RAIBERT PLANNER
            #
            # Physical torso is fixed ONLY to isolate PD.
            #
            # Planner receives a moving virtual body position,
            # exactly like the previous integration test.
            # =================================================

            W_p_raibert_anchor = (
                virtual_base_position

                +

                W_R_B_initial
                @
                B_r_raibert_anchor
            )


            # =================================================
            # RAIBERT
            # =================================================

            raibert_result = (
                compute_raibert_foothold(

                    W_p_hip=
                        W_p_raibert_anchor,

                    W_v_actual=
                        W_v_actual,

                    W_v_ref=
                        W_V_REF,

                    T_remaining=
                        T_remaining,

                    foothold_z=
                        FOOTHOLD_Z,

                    leg=
                        SWING_LEG,
                )
            )


            W_p_foothold = np.asarray(
                raibert_result[
                    "W_p_foothold"
                ],
                dtype=float,
            ).copy()


            # =================================================
            # BEZIER
            # =================================================

            bezier_result = (
                evaluate_bezier_swing_trajectory(

                    W_p_liftoff=
                        W_p_liftoff,

                    W_p_foothold=
                        W_p_foothold,

                    swing_phase=
                        swing_phase,

                    T_swing=
                        T_SWING,

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
            ).copy()


            # =================================================
            # IK
            #
            # IMPORTANT:
            # solve_numerical_IK_hybrid() restores qpos/qvel.
            #
            # It does NOT move the physical robot.
            # =================================================

            (
                q_ik,
                ik_diag,

            ) = solve_numerical_IK_hybrid(

                model=
                    model,

                data=
                    data,

                W_p_ankle_des=
                    W_p_des,

                swing_foot=
                    SWING_LEG,

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
                    f"IK returned None at phase "
                    f"{swing_phase:.4f}"
                )


            # =================================================
            # ACTUAL JOINT STATE x_k
            # =================================================

            (
                q_actual,
                q_dot_actual,

            ) = get_leg_joint_state(
                model,
                data,
                SWING_LEG,
            )


            # =================================================
            # DESIRED ANKLE
            #
            # Follow paper:
            #
            # use ACTUAL HP and K.
            # =================================================

            q_ankle_des = (
                compute_ankle_desired_from_actual(
                    q_actual
                )
            )


            # =================================================
            # COMPLETE q_des
            #
            # [
            #   HY,
            #   HR,
            #   HP,
            #   K,
            #   ankle
            # ]
            # =================================================

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


            # =================================================
            # qdot_des
            # =================================================

            q_dot_des_filtered = (
                update_q_dot_des(

                    q_des=
                        q_des,

                    q_des_previous=
                        q_des_previous,

                    q_dot_des_filtered_previous=
                        q_dot_des_filtered,
                )
            )


            # =================================================
            # PD
            #
            # tau =
            #
            # Kp(q_des-q)
            # +
            # Kd(qdot_des-qdot)
            # =================================================

            tau_raw = (
                compute_swing_joint_PD(

                    q_des=
                        q_des,

                    q=
                        q_actual,

                    q_dot=
                        q_dot_actual,

                    Kp=
                        KP_SWING,

                    Kd=
                        KD_SWING,

                    q_dot_des=
                        q_dot_des_filtered,
                )
            )


            # =================================================
            # ACTUATOR SATURATION
            # =================================================

            (
                tau_applied,
                torque_clipped,

            ) = clip_leg_torque(
                model,
                SWING_LEG,
                tau_raw,
            )


            # =================================================
            # STANCE LEG HOLD PD
            #
            # Only for visualization of this isolated fixed-base
            # actuator test.
            # =================================================

            (
                q_stance,
                q_dot_stance,

            ) = get_leg_joint_state(
                model,
                data,
                STANCE_LEG,
            )


            tau_stance_raw = (
                compute_swing_joint_PD(

                    q_des=
                        q_stance_hold,

                    q=
                        q_stance,

                    q_dot=
                        q_dot_stance,

                    Kp=
                        KP_SWING,

                    Kd=
                        KD_SWING,

                    q_dot_des=
                        np.zeros(
                            5,
                            dtype=float,
                        ),
                )
            )


            (
                tau_stance,
                _,

            ) = clip_leg_torque(
                model,
                STANCE_LEG,
                tau_stance_raw,
            )


            # =================================================
            # CURRENT CARTESIAN STATE
            # =================================================

            W_p_actual = get_ankle_position(
                model,
                data,
                SWING_LEG,
            )


            cartesian_error = float(
                np.linalg.norm(
                    W_p_des
                    -
                    W_p_actual
                )
            )


            # =================================================
            # LOG x_k AND u_k
            # =================================================

            time_log.append(
                t
            )

            phase_log.append(
                swing_phase
            )

            Trem_log.append(
                T_remaining
            )

            velocity_log.append(
                W_v_actual.copy()
            )

            foothold_log.append(
                W_p_foothold.copy()
            )

            desired_xyz_log.append(
                W_p_des.copy()
            )

            actual_xyz_log.append(
                W_p_actual.copy()
            )

            cart_error_log.append(
                cartesian_error
            )

            q_des_log.append(
                q_des.copy()
            )

            q_actual_log.append(
                q_actual.copy()
            )

            q_dot_des_log.append(
                q_dot_des_filtered.copy()
            )

            q_dot_actual_log.append(
                q_dot_actual.copy()
            )

            tau_raw_log.append(
                tau_raw.copy()
            )

            tau_applied_log.append(
                tau_applied.copy()
            )

            tau_clip_log.append(
                torque_clipped.copy()
            )

            sigma_log.append(
                float(
                    ik_diag[
                        "sigma_min"
                    ]
                )
            )

            iteration_log.append(
                int(
                    ik_diag[
                        "iterations"
                    ]
                )
            )


            # =================================================
            # TRAILS
            # =================================================

            if (
                k
                %
                TRAIL_DECIMATION
                ==
                0
            ):

                desired_trail.append(
                    W_p_des.copy()
                )

                actual_trail.append(
                    W_p_actual.copy()
                )


            # =================================================
            # GUI
            # =================================================

            update_gui_markers(

                viewer=
                    viewer,

                W_p_liftoff=
                    W_p_liftoff,

                W_p_foothold=
                    W_p_foothold,

                W_p_des=
                    W_p_des,

                W_p_actual=
                    W_p_actual,

                desired_trail=
                    desired_trail,

                actual_trail=
                    actual_trail,
            )


            viewer.sync()


            # =================================================
            # TERMINAL DEBUG
            # =================================================

            if k in PRINT_INDICES:

                print(
                    "-" * 92
                )

                print(
                    f"phase = {swing_phase:.3f}"
                )

                print(
                    f"T_rem = {T_remaining:.4f} s"
                )

                print(
                    f"Raibert Pf = {W_p_foothold}"
                )

                print(
                    f"Bezier desired = {W_p_des}"
                )

                print(
                    f"Actual ankle = {W_p_actual}"
                )

                print(
                    f"Cartesian error = "
                    f"{cartesian_error * 1000.0:.4f} mm"
                )

                print(
                    f"q_des [deg] = "
                    f"{np.rad2deg(q_des)}"
                )

                print(
                    f"q     [deg] = "
                    f"{np.rad2deg(q_actual)}"
                )

                print(
                    f"tau raw = {tau_raw}"
                )

                print(
                    f"tau applied = {tau_applied}"
                )

                print(
                    f"torque clipped = {torque_clipped}"
                )

                print(
                    f"IK iter = "
                    f"{ik_diag['iterations']}, "
                    f"sigma_min = "
                    f"{ik_diag['sigma_min']:.6e}"
                )


            # =================================================
            # SAVE DESIRED STATE FOR NEXT qdot_des
            # =================================================

            q_des_previous = (
                q_des.copy()
            )


            # =================================================
            # FINAL SAMPLE x_N
            #
            # At t = T_swing we only observe the final state.
            #
            # The previous control u_(N-1) has already produced
            # this x_N.
            # =================================================

            if k == N_SAMPLES - 1:

                break


            # =================================================
            # APPLY u_k
            # =================================================

            data.ctrl[:] = 0.0


            write_leg_torque(
                model,
                data,
                SWING_LEG,
                tau_applied,
            )


            write_leg_torque(
                model,
                data,
                STANCE_LEG,
                tau_stance,
            )


            # =================================================
            # MUJOCO PHYSICS
            #
            # u_k is held constant through all physics substeps.
            #
            # x_k
            #   ->
            # u_k
            #   ->
            # mj_step()
            #   ->
            # x_(k+1)
            # =================================================

            for _ in range(
                PHYSICS_SUBSTEPS
            ):

                mujoco.mj_step(
                    model,
                    data,
                )


            # =================================================
            # VIRTUAL BODY STATE FOR NEXT RAIBERT UPDATE
            # =================================================

            virtual_base_position += (
                W_v_actual
                *
                CONTROL_DT
            )


            # =================================================
            # GUI SLOWDOWN ONLY
            # =================================================

            time.sleep(
                CONTROL_DT
                *
                DISPLAY_TIME_SCALE
            )


        # ====================================================
        # FINAL HOLD
        # ====================================================

        hold_end = time.time()


        while (
            viewer.is_running()
            and
            time.time()
            -
            hold_end
            <
            GUI_END_HOLD
        ):

            viewer.sync()

            time.sleep(
                0.01
            )


    # ========================================================
    # ARRAYS
    # ========================================================

    time_log = np.asarray(
        time_log
    )

    foothold_log = np.asarray(
        foothold_log
    )

    desired_xyz_log = np.asarray(
        desired_xyz_log
    )

    actual_xyz_log = np.asarray(
        actual_xyz_log
    )

    cart_error_log = np.asarray(
        cart_error_log
    )

    q_des_log = np.asarray(
        q_des_log
    )

    q_actual_log = np.asarray(
        q_actual_log
    )

    q_dot_des_log = np.asarray(
        q_dot_des_log
    )

    q_dot_actual_log = np.asarray(
        q_dot_actual_log
    )

    tau_raw_log = np.asarray(
        tau_raw_log
    )

    tau_applied_log = np.asarray(
        tau_applied_log
    )

    tau_clip_log = np.asarray(
        tau_clip_log,
        dtype=bool,
    )

    sigma_log = np.asarray(
        sigma_log
    )

    iteration_log = np.asarray(
        iteration_log
    )


    # ========================================================
    # JOINT METRICS
    # ========================================================

    q_error = (
        q_des_log
        -
        q_actual_log
    )


    joint_rmse_rad = np.sqrt(
        np.mean(
            q_error**2,
            axis=0,
        )
    )


    joint_rmse_deg = np.rad2deg(
        joint_rmse_rad
    )


    peak_tau_raw = np.max(
        np.abs(
            tau_raw_log
        ),
        axis=0,
    )


    peak_tau_applied = np.max(
        np.abs(
            tau_applied_log
        ),
        axis=0,
    )


    clip_count_per_joint = np.sum(
        tau_clip_log,
        axis=0,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 92
    )

    print(
        "PD DYNAMIC INTEGRATION TEST SUMMARY"
    )

    print(
        "=" * 92
    )


    print(
        f"Samples               : {len(time_log)}"
    )


    print(
        f"Final Cartesian error : "
        f"{cart_error_log[-1] * 1000.0:.6f} mm"
    )


    print(
        f"Mean Cartesian error  : "
        f"{np.mean(cart_error_log) * 1000.0:.6f} mm"
    )


    print(
        f"Max Cartesian error   : "
        f"{np.max(cart_error_log) * 1000.0:.6f} mm"
    )


    print(
        f"Minimum sigma         : "
        f"{np.min(sigma_log):.6e}"
    )


    print(
        f"Maximum IK iteration  : "
        f"{np.max(iteration_log)}"
    )


    print()

    print(
        "Joint RMSE [deg]"
    )


    for i, name in enumerate(
        LEG_JOINT_NAMES[
            SWING_LEG
        ]
    ):

        print(
            f"  {name:<14}: "
            f"{joint_rmse_deg[i]:.6f}"
        )


    print()

    print(
        "Peak |tau raw| [N.m]"
    )


    for i, name in enumerate(
        LEG_JOINT_NAMES[
            SWING_LEG
        ]
    ):

        print(
            f"  {name:<14}: "
            f"{peak_tau_raw[i]:.6f}"
        )


    print()

    print(
        "Peak |tau applied| [N.m]"
    )


    for i, name in enumerate(
        LEG_JOINT_NAMES[
            SWING_LEG
        ]
    ):

        print(
            f"  {name:<14}: "
            f"{peak_tau_applied[i]:.6f}"
        )


    print()

    print(
        "Torque clipping count"
    )


    for i, name in enumerate(
        LEG_JOINT_NAMES[
            SWING_LEG
        ]
    ):

        print(
            f"  {name:<14}: "
            f"{int(clip_count_per_joint[i])}"
        )


    print()

    print(
        f"Initial ankle         : {W_p_liftoff}"
    )

    print(
        f"Final foothold        : {foothold_log[-1]}"
    )

    print(
        f"Final desired ankle   : {desired_xyz_log[-1]}"
    )

    print(
        f"Final actual ankle    : {actual_xyz_log[-1]}"
    )


    # ========================================================
    # PLOT 1
    # CARTESIAN TRACKING
    # ========================================================

    xyz_labels = [
        "x",
        "y",
        "z",
    ]


    for axis in range(
        3
    ):

        plt.figure()

        plt.plot(
            time_log,
            desired_xyz_log[
                :,
                axis
            ],
            label="desired",
        )

        plt.plot(
            time_log,
            actual_xyz_log[
                :,
                axis
            ],
            linestyle="--",
            label="actual",
        )

        plt.xlabel(
            "Time [s]"
        )

        plt.ylabel(
            f"{xyz_labels[axis]} [m]"
        )

        plt.title(
            f"PD dynamic ankle tracking - {xyz_labels[axis]}"
        )

        plt.grid(
            True
        )

        plt.legend()


    # ========================================================
    # PLOT 2
    # CARTESIAN ERROR
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        cart_error_log
        *
        1000.0,
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Cartesian error [mm]"
    )

    plt.title(
        "Raibert -> Bezier -> IK -> PD Cartesian error"
    )

    plt.grid(
        True
    )


    # ========================================================
    # PLOT 3
    # JOINT POSITION ERROR
    # ========================================================

    plt.figure()


    for i, name in enumerate(
        LEG_JOINT_NAMES[
            SWING_LEG
        ]
    ):

        plt.plot(
            time_log,
            np.rad2deg(
                q_error[
                    :,
                    i
                ]
            ),
            label=name,
        )


    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Joint position error [deg]"
    )

    plt.title(
        "Swing joint PD tracking error"
    )

    plt.grid(
        True
    )

    plt.legend()


    # ========================================================
    # PLOT 4
    # APPLIED TORQUE
    # ========================================================

    plt.figure()


    for i, name in enumerate(
        LEG_JOINT_NAMES[
            SWING_LEG
        ]
    ):

        plt.plot(
            time_log,
            tau_applied_log[
                :,
                i
            ],
            label=name,
        )


    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Torque [N.m]"
    )

    plt.title(
        "Swing joint applied PD torque"
    )

    plt.grid(
        True
    )

    plt.legend()


    # ========================================================
    # PLOT 5
    # RAIBERT FOOTHOLD
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        foothold_log[
            :,
            0
        ],
        label="Pf x",
    )

    plt.plot(
        time_log,
        foothold_log[
            :,
            1
        ],
        label="Pf y",
    )

    plt.plot(
        time_log,
        foothold_log[
            :,
            2
        ],
        label="Pf z",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Foothold [m]"
    )

    plt.title(
        "Online Raibert foothold"
    )

    plt.grid(
        True
    )

    plt.legend()


    # ========================================================
    # PLOT 6
    # X-Z TRAJECTORY
    # ========================================================

    plt.figure()

    plt.plot(
        desired_xyz_log[
            :,
            0
        ],
        desired_xyz_log[
            :,
            2
        ],
        label="desired",
    )

    plt.plot(
        actual_xyz_log[
            :,
            0
        ],
        actual_xyz_log[
            :,
            2
        ],
        linestyle="--",
        label="actual",
    )

    plt.xlabel(
        "x [m]"
    )

    plt.ylabel(
        "z [m]"
    )

    plt.title(
        "PD dynamic swing path X-Z"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()


    plt.show()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()