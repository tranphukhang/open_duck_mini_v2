from pathlib import Path
import sys
import time

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


# ============================================================
# TEST CONFIGURATION
# ============================================================

SWING_LEG = "L"


# ============================================================
# CONTROL LOOP
#
# Same low-level control period used by the paper:
#
#     dt = 0.001 s
#
# No mj_step() is used in this test.
#
# MuJoCo qpos is directly updated so that we can visually
# inspect the kinematic pipeline.
# ============================================================

CONTROL_DT = 0.001

T_SWING = 0.24

N_CONTROL_STEPS = int(
    round(
        T_SWING
        /
        CONTROL_DT
    )
) + 1


# ============================================================
# GUI
#
# Logical controller still runs using dt = 0.001 s.
#
# DISPLAY_TIME_SCALE only slows wall-clock visualization.
#
# 10 means:
#
#     0.24 s logical swing
#
# appears as approximately:
#
#     2.4 s in the GUI.
# ============================================================

DISPLAY_TIME_SCALE = 10.0

GUI_START_HOLD = 2.0

GUI_END_HOLD = 3.0


# ============================================================
# BEZIER
#
# Start conservatively with 8 mm because our previous test
# showed 12 mm could bring the knee close to its limit.
#
# Later:
#
#     SWING_HEIGHT = 0.012
#
# can be tested again.
# ============================================================

SWING_HEIGHT = 0.008

CP1_COEF = 0.333

CP2_COEF = 0.666


# ============================================================
# TEST VELOCITY COMMAND
#
# Desired walking velocity.
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
# EMULATED VELOCITY FEEDBACK
#
# We do not yet run full dynamics.
#
# Instead, we emulate state-estimator velocity feedback:
#
#     v_actual(k)
#
# and store it in MuJoCo qvel.
#
# A small time-varying error is included so that the Raibert
# foothold actually changes online.
# ============================================================

VX_FEEDBACK_AMPLITUDE = 0.04

VY_FEEDBACK_AMPLITUDE = 0.02


# ============================================================
# INITIAL JOINT CONFIGURATION
#
# [
#     hip_yaw,
#     hip_roll,
#     hip_pitch,
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
# FLAT GROUND
# ============================================================

SLOPE_PITCH = 0.0


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
# VISUAL MARKERS
#
# All markers are intentionally transparent.
#
# Their sizes are also different so that when two points
# overlap, both remain visible.
# ============================================================

MARKER_RADIUS_FOOTHOLD = 0.0070

MARKER_RADIUS_DESIRED = 0.0050

MARKER_RADIUS_ACTUAL = 0.0032

MARKER_RADIUS_LIFTOFF = 0.0040

MARKER_RADIUS_TRAIL = 0.0020


# Semi-transparent RGBA
RGBA_FOOTHOLD = np.array(
    [
        1.0,
        0.45,
        0.05,
        0.18,
    ],
    dtype=np.float32,
)

RGBA_DESIRED = np.array(
    [
        0.10,
        0.85,
        0.20,
        0.23,
    ],
    dtype=np.float32,
)

RGBA_ACTUAL = np.array(
    [
        0.10,
        0.35,
        1.00,
        0.32,
    ],
    dtype=np.float32,
)

RGBA_LIFTOFF = np.array(
    [
        0.70,
        0.70,
        0.70,
        0.16,
    ],
    dtype=np.float32,
)

RGBA_TRAIL_DES = np.array(
    [
        0.10,
        0.85,
        0.20,
        0.08,
    ],
    dtype=np.float32,
)

RGBA_TRAIL_ACT = np.array(
    [
        0.10,
        0.35,
        1.00,
        0.10,
    ],
    dtype=np.float32,
)


# Save one trail marker every N control samples.
TRAIL_DECIMATION = 8

MAX_TRAIL_POINTS = 40


# ============================================================
# JOINT HELPERS
# ============================================================

def get_joint_id(
    model,
    joint_name,
):

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise ValueError(
            f"Joint '{joint_name}' not found."
        )

    return int(
        joint_id
    )


def get_joint_qpos_address(
    model,
    joint_name,
):

    joint_id = get_joint_id(
        model,
        joint_name,
    )

    return int(
        model.jnt_qposadr[
            joint_id
        ]
    )


def get_joint_position(
    model,
    data,
    joint_name,
):

    adr = get_joint_qpos_address(
        model,
        joint_name,
    )

    return float(
        data.qpos[
            adr
        ]
    )


def set_joint_position(
    model,
    data,
    joint_name,
    value,
):

    adr = get_joint_qpos_address(
        model,
        joint_name,
    )

    data.qpos[
        adr
    ] = float(
        value
    )


def set_leg_configuration(
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

    for (
        joint_name,
        q,
    ) in zip(
        LEG_JOINT_NAMES[
            leg
        ],
        q_leg,
    ):

        set_joint_position(
            model,
            data,
            joint_name,
            q,
        )


# ============================================================
# BODY / SITE HELPERS
# ============================================================

def get_body_id(
    model,
    body_name,
):

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise ValueError(
            f"Body '{body_name}' not found."
        )

    return int(
        body_id
    )


def get_site_id(
    model,
    site_name,
):

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise ValueError(
            f"Site '{site_name}' not found."
        )

    return int(
        site_id
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

    # Floating base
    data.qpos[
        0:3
    ] = np.array(
        [
            0.0,
            0.0,
            0.28,
        ],
        dtype=float,
    )

    data.qpos[
        3:7
    ] = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


    set_leg_configuration(
        model,
        data,
        "L",
        INITIAL_Q_LEFT,
    )

    set_leg_configuration(
        model,
        data,
        "R",
        INITIAL_Q_RIGHT,
    )


    data.qvel[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


# ============================================================
# RAIBERT BODY-FRAME ANCHOR
#
# Important:
#
# The paper does NOT use the literal hip joint coordinate.
#
# It uses:
#
#     base position
#       +
#     R_WB * get_hip_offset()
#
# For Open Duck we create an equivalent nominal anchor by
# taking the nominal ankle X/Y relative to the body.
#
# The Z component is intentionally set to zero because Raibert
# touchdown Z is overwritten separately.
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


    W_delta = (
        W_p_ankle
        -
        W_p_base
    )


    B_r_anchor = (
        W_R_B.T
        @
        W_delta
    )


    # Same structural idea as paper:
    # no vertical component in the hip/anchor offset.
    B_r_anchor[
        2
    ] = 0.0


    return B_r_anchor


def compute_raibert_anchor_W(
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
# EMULATED VELOCITY FEEDBACK
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


    # Forward-speed tracking disturbance.
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


    # Small lateral disturbance.
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
# APPLY IK OUTPUT
#
# q_ik:
#
# [
#     hip_yaw,
#     hip_roll,
#     hip_pitch,
#     knee
# ]
# ============================================================

def apply_ik_solution(
    model,
    data,
    leg,
    q_ik,
):

    q_ik = np.asarray(
        q_ik,
        dtype=float,
    ).reshape(
        4
    )


    joint_names = (
        LEG_JOINT_NAMES[
            leg
        ][
            0:4
        ]
    )


    for (
        joint_name,
        value,
    ) in zip(
        joint_names,
        q_ik,
    ):

        set_joint_position(
            model,
            data,
            joint_name,
            value,
        )


# ============================================================
# ANKLE COMPENSATION
# ============================================================

def apply_ankle_compensation(
    model,
    data,
    leg,
):

    if leg == "L":

        hip_pitch_name = (
            "L_hip_pitch"
        )

        knee_name = (
            "L_knee"
        )

        ankle_name = (
            "L_ankle"
        )

    else:

        hip_pitch_name = (
            "R_hip_pitch"
        )

        knee_name = (
            "R_knee"
        )

        ankle_name = (
            "R_ankle"
        )


    q_hp = get_joint_position(
        model,
        data,
        hip_pitch_name,
    )

    q_k = get_joint_position(
        model,
        data,
        knee_name,
    )


    q_ankle_des = (
        -SLOPE_PITCH
        -
        q_hp
        -
        q_k
    )


    set_joint_position(
        model,
        data,
        ankle_name,
        q_ankle_des,
    )


    return float(
        q_ankle_des
    )


# ============================================================
# JOINT-LIMIT MARGIN
# ============================================================

def compute_min_joint_margin(
    model,
    data,
    leg,
):

    min_margin = np.inf

    nearest_joint = None


    for joint_name in LEG_JOINT_NAMES[
        leg
    ]:

        joint_id = get_joint_id(
            model,
            joint_name,
        )


        if not bool(
            model.jnt_limited[
                joint_id
            ]
        ):

            continue


        q = get_joint_position(
            model,
            data,
            joint_name,
        )


        q_min = float(
            model.jnt_range[
                joint_id,
                0
            ]
        )

        q_max = float(
            model.jnt_range[
                joint_id,
                1
            ]
        )


        margin = min(
            q - q_min,
            q_max - q,
        )


        if margin < min_margin:

            min_margin = (
                margin
            )

            nearest_joint = (
                joint_name
            )


    return (
        float(
            min_margin
        ),
        nearest_joint,
    )


# ============================================================
# USER-SCENE MARKER
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


# ============================================================
# UPDATE GUI MARKERS
# ============================================================

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


        # ----------------------------------------------------
        # Liftoff
        # ----------------------------------------------------

        add_sphere_marker(
            scene,
            W_p_liftoff,
            MARKER_RADIUS_LIFTOFF,
            RGBA_LIFTOFF,
        )


        # ----------------------------------------------------
        # Current Raibert foothold
        #
        # Large and very transparent.
        # ----------------------------------------------------

        add_sphere_marker(
            scene,
            W_p_foothold,
            MARKER_RADIUS_FOOTHOLD,
            RGBA_FOOTHOLD,
        )


        # ----------------------------------------------------
        # Current Bezier desired point
        # ----------------------------------------------------

        add_sphere_marker(
            scene,
            W_p_des,
            MARKER_RADIUS_DESIRED,
            RGBA_DESIRED,
        )


        # ----------------------------------------------------
        # Current actual ankle center
        #
        # Smaller radius allows it to remain visible inside
        # the desired point when both overlap.
        # ----------------------------------------------------

        add_sphere_marker(
            scene,
            W_p_actual,
            MARKER_RADIUS_ACTUAL,
            RGBA_ACTUAL,
        )


        # ----------------------------------------------------
        # Faint trajectory trails
        # ----------------------------------------------------

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
        "=" * 84
    )

    print(
        "TEST: RAIBERT -> BEZIER -> IK"
    )

    print(
        "=" * 84
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


    print(
        f"XML                  : {XML_PATH}"
    )

    print(
        f"Swing leg            : {SWING_LEG}"
    )

    print(
        f"Control dt            : {CONTROL_DT:.6f} s"
    )

    print(
        f"Control frequency     : {1.0 / CONTROL_DT:.1f} Hz"
    )

    print(
        f"T_swing               : {T_SWING:.3f} s"
    )

    print(
        f"Swing height          : {SWING_HEIGHT:.4f} m"
    )

    print(
        f"GUI time scale        : {DISPLAY_TIME_SCALE:.1f} x slower"
    )

    print()

    print(
        "Marker legend:"
    )

    print(
        "  orange transparent : current Raibert foothold"
    )

    print(
        "  green  transparent : Bezier desired ankle"
    )

    print(
        "  blue   transparent : actual ankle"
    )

    print(
        "  gray   transparent : liftoff point"
    )

    print()

    print(
        "IMPORTANT: this is a kinematic integration test."
    )

    print(
        "No mj_step() is called."
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
    # INITIAL STATE
    # ========================================================

    set_initial_state(
        model,
        data,
    )


    W_p_liftoff = get_ankle_position(
        model,
        data,
        SWING_LEG,
    )


    # --------------------------------------------------------
    # Equivalent to paper's get_hip_offset().
    # --------------------------------------------------------

    B_r_raibert_anchor = (
        compute_raibert_anchor_offset_B(
            model,
            data,
            SWING_LEG,
        )
    )


    # --------------------------------------------------------
    # Flat-ground touchdown ankle height.
    #
    # Since our planner currently works directly with the
    # ankle center, touchdown ankle Z is kept equal to the
    # initial ankle-center height.
    # --------------------------------------------------------

    FOOTHOLD_Z = float(
        W_p_liftoff[
            2
        ]
    )


    print(
        f"Initial ankle center  : {W_p_liftoff}"
    )

    print(
        f"Raibert anchor B      : {B_r_raibert_anchor}"
    )

    print(
        f"Touchdown ankle Z     : {FOOTHOLD_Z:.6f} m"
    )

    print()


    # ========================================================
    # LOGS
    # ========================================================

    time_log = []

    phase_log = []

    Trem_log = []

    velocity_log = []

    velocity_ref_log = []

    foothold_log = []

    desired_log = []

    actual_log = []

    error_log = []

    sigma_log = []

    iteration_log = []

    joint_margin_log = []

    ankle_command_log = []

    x_clip_log = []

    y_clip_log = []


    desired_trail = []

    actual_trail = []


    # ========================================================
    # LAUNCH GUI
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        configure_camera(
            viewer
        )


        # ====================================================
        # INITIAL HOLD
        # ====================================================

        start_hold_time = (
            time.time()
        )


        while (
            viewer.is_running()
            and
            time.time()
            -
            start_hold_time
            <
            GUI_START_HOLD
        ):

            W_p_actual = get_ankle_position(
                model,
                data,
                SWING_LEG,
            )


            W_p_anchor = compute_raibert_anchor_W(
                model,
                data,
                B_r_raibert_anchor,
            )


            update_gui_markers(

                viewer=
                    viewer,

                W_p_liftoff=
                    W_p_liftoff,

                W_p_foothold=
                    W_p_anchor,

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
        # ====================================================

        for k in range(
            N_CONTROL_STEPS
        ):

            if not viewer.is_running():
                break


            # =================================================
            # x_k
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
            # STATE-ESTIMATOR VELOCITY FEEDBACK
            # =================================================

            W_v_actual = compute_velocity_feedback(
                swing_phase
            )


            # Store the emulated measured velocity in MuJoCo.
            data.qvel[
                0:3
            ] = W_v_actual


            # =================================================
            # PAPER-STYLE RAIBERT ANCHOR
            #
            # Base pose is read from current MuJoCo state.
            # =================================================

            W_p_raibert_anchor = (
                compute_raibert_anchor_W(
                    model,
                    data,
                    B_r_raibert_anchor,
                )
            )


            # =================================================
            # RAIBERT
            #
            # T_remaining is updated every control cycle.
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
            #
            # P0 remains the fixed liftoff point.
            #
            # Pf is updated online by Raibert.
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
            # NUMERICAL IK
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
            # APPLY IK TO MUJOCO STATE
            # =================================================

            apply_ik_solution(
                model,
                data,
                SWING_LEG,
                q_ik,
            )


            mujoco.mj_forward(
                model,
                data,
            )


            # =================================================
            # ANKLE COMPENSATION
            # =================================================

            q_ankle_des = (
                apply_ankle_compensation(
                    model,
                    data,
                    SWING_LEG,
                )
            )


            mujoco.mj_forward(
                model,
                data,
            )


            # =================================================
            # ACTUAL ANKLE POSITION
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
            # JOINT LIMIT
            # =================================================

            (
                min_joint_margin,
                nearest_joint,

            ) = compute_min_joint_margin(
                model,
                data,
                SWING_LEG,
            )


            # =================================================
            # LOG
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

            velocity_ref_log.append(
                W_V_REF.copy()
            )

            foothold_log.append(
                W_p_foothold.copy()
            )

            desired_log.append(
                W_p_des.copy()
            )

            actual_log.append(
                W_p_actual.copy()
            )

            error_log.append(
                cartesian_error
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

            joint_margin_log.append(
                min_joint_margin
            )

            ankle_command_log.append(
                q_ankle_des
            )

            x_clip_log.append(
                bool(
                    raibert_result[
                        "x_was_clipped"
                    ]
                )
            )

            y_clip_log.append(
                bool(
                    raibert_result[
                        "y_was_clipped"
                    ]
                )
            )


            # =================================================
            # TRAIL
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
            # PRINT SELECTED PHASES
            # =================================================

            if (
                k == 0
                or
                abs(
                    swing_phase
                    -
                    0.25
                )
                <
                CONTROL_DT
                /
                T_SWING
                /
                2.0
                or
                abs(
                    swing_phase
                    -
                    0.50
                )
                <
                CONTROL_DT
                /
                T_SWING
                /
                2.0
                or
                abs(
                    swing_phase
                    -
                    0.75
                )
                <
                CONTROL_DT
                /
                T_SWING
                /
                2.0
                or
                k
                ==
                N_CONTROL_STEPS
                -
                1
            ):

                print(
                    "-" * 84
                )

                print(
                    f"phase = {swing_phase:.3f}"
                )

                print(
                    f"T_rem = {T_remaining:.4f} s"
                )

                print(
                    f"v_actual = {W_v_actual}"
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
                    f"error = "
                    f"{cartesian_error * 1000.0:.4f} mm"
                )

                print(
                    f"IK iterations = "
                    f"{ik_diag['iterations']}"
                )

                print(
                    f"sigma_min = "
                    f"{ik_diag['sigma_min']:.6e}"
                )

                print(
                    f"nearest limit = "
                    f"{nearest_joint}, "
                    f"margin = "
                    f"{np.rad2deg(min_joint_margin):.3f} deg"
                )

                print(
                    f"clip X/Y = "
                    f"{raibert_result['x_was_clipped']} / "
                    f"{raibert_result['y_was_clipped']}"
                )


            # =================================================
            # EMULATE BASE MOTION FOR x_(k+1)
            #
            # IMPORTANT:
            #
            # This is NOT mj_step().
            #
            # We manually integrate the floating-base position
            # only so the kinematic test has a moving body,
            # similar to what would happen in actual walking.
            #
            # x_k
            #   ->
            # calculate references
            #   ->
            # apply q_k
            #   ->
            # create next emulated state x_(k+1)
            # =================================================

            if k < N_CONTROL_STEPS - 1:

                data.qpos[
                    0:3
                ] += (
                    W_v_actual
                    *
                    CONTROL_DT
                )


                mujoco.mj_forward(
                    model,
                    data,
                )


            # =================================================
            # SLOW GUI ONLY
            # =================================================

            time.sleep(
                CONTROL_DT
                *
                DISPLAY_TIME_SCALE
            )


        # ====================================================
        # FINAL HOLD
        # ====================================================

        final_hold_time = (
            time.time()
        )


        while (
            viewer.is_running()
            and
            time.time()
            -
            final_hold_time
            <
            GUI_END_HOLD
        ):

            viewer.sync()

            time.sleep(
                0.01
            )


    # ========================================================
    # NUMPY LOGS
    # ========================================================

    time_log = np.asarray(
        time_log
    )

    phase_log = np.asarray(
        phase_log
    )

    Trem_log = np.asarray(
        Trem_log
    )

    velocity_log = np.asarray(
        velocity_log
    )

    velocity_ref_log = np.asarray(
        velocity_ref_log
    )

    foothold_log = np.asarray(
        foothold_log
    )

    desired_log = np.asarray(
        desired_log
    )

    actual_log = np.asarray(
        actual_log
    )

    error_log = np.asarray(
        error_log
    )

    sigma_log = np.asarray(
        sigma_log
    )

    iteration_log = np.asarray(
        iteration_log
    )

    joint_margin_log = np.asarray(
        joint_margin_log
    )

    ankle_command_log = np.asarray(
        ankle_command_log
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 84
    )

    print(
        "INTEGRATION TEST SUMMARY"
    )

    print(
        "=" * 84
    )


    print(
        f"Samples             : {len(time_log)}"
    )

    print(
        f"Final error         : "
        f"{error_log[-1] * 1000.0:.6f} mm"
    )

    print(
        f"Mean error          : "
        f"{np.mean(error_log) * 1000.0:.6f} mm"
    )

    print(
        f"Max error           : "
        f"{np.max(error_log) * 1000.0:.6f} mm"
    )

    print(
        f"Minimum sigma       : "
        f"{np.min(sigma_log):.6e}"
    )

    print(
        f"Maximum IK iteration: "
        f"{np.max(iteration_log)}"
    )

    print(
        f"Minimum joint margin: "
        f"{np.rad2deg(np.min(joint_margin_log)):.4f} deg"
    )

    print(
        f"X clipping count    : "
        f"{sum(x_clip_log)}"
    )

    print(
        f"Y clipping count    : "
        f"{sum(y_clip_log)}"
    )

    print()

    print(
        f"Initial ankle       : "
        f"{W_p_liftoff}"
    )

    print(
        f"Final foothold      : "
        f"{foothold_log[-1]}"
    )

    print(
        f"Final desired ankle : "
        f"{desired_log[-1]}"
    )

    print(
        f"Final actual ankle  : "
        f"{actual_log[-1]}"
    )


    # ========================================================
    # PLOT 1
    # DESIRED VS ACTUAL POSITION
    # ========================================================

    labels = [
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
            desired_log[
                :,
                axis
            ],
            label="desired",
        )

        plt.plot(
            time_log,
            actual_log[
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
            f"{labels[axis]} [m]"
        )

        plt.title(
            f"Swing ankle {labels[axis]} tracking"
        )

        plt.grid(
            True
        )

        plt.legend()


    # ========================================================
    # PLOT 2
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
        "Raibert foothold [m]"
    )

    plt.title(
        "Online Raibert foothold"
    )

    plt.grid(
        True
    )

    plt.legend()


    # ========================================================
    # PLOT 3
    # CARTESIAN ERROR
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        error_log
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
        "Bezier -> IK tracking error"
    )

    plt.grid(
        True
    )


    # ========================================================
    # PLOT 4
    # VELOCITY FEEDBACK
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        velocity_log[
            :,
            0
        ],
        label="vx actual",
    )

    plt.plot(
        time_log,
        velocity_ref_log[
            :,
            0
        ],
        linestyle="--",
        label="vx ref",
    )

    plt.plot(
        time_log,
        velocity_log[
            :,
            1
        ],
        label="vy actual",
    )

    plt.plot(
        time_log,
        velocity_ref_log[
            :,
            1
        ],
        linestyle="--",
        label="vy ref",
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Velocity [m/s]"
    )

    plt.title(
        "Velocity feedback used by Raibert"
    )

    plt.grid(
        True
    )

    plt.legend()


    # ========================================================
    # PLOT 5
    # T REMAINING
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        Trem_log,
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "T_remaining [s]"
    )

    plt.title(
        "Remaining swing time feedback"
    )

    plt.grid(
        True
    )


    # ========================================================
    # PLOT 6
    # X-Z SWING PATH
    # ========================================================

    plt.figure()

    plt.plot(
        desired_log[
            :,
            0
        ],
        desired_log[
            :,
            2
        ],
        label="desired",
    )

    plt.plot(
        actual_log[
            :,
            0
        ],
        actual_log[
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
        "Swing path X-Z"
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