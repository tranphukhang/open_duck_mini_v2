from pathlib import Path
import sys

import mujoco
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

THIS_FILE = Path(__file__).resolve()

FOOTSTEP_PLANNER_DIR = THIS_FILE.parent
MPC_SRBD_DIR = FOOTSTEP_PLANNER_DIR.parent
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
# IMPORT PROJECT MODULES
# ============================================================

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

from Footstep_Planner.raibert_foot_placement import (
    DEFAULT_MAX_CORRECTION_X,
    DEFAULT_MAX_CORRECTION_Y,
)


# ============================================================
# USER SETTINGS
# ============================================================

SWING_LEG = "R"


# ============================================================
# BEZIER SETTINGS
# ============================================================

T_SWING = 0.24


# IMPORTANT:
#
# Previous test with 0.012 m reached knee limit near apex.
#
# For clip identification, start with a slightly smaller
# vertical swing so that horizontal correction is isolated.
#
# Later, repeat with:
#
#     SWING_HEIGHT = 0.012
#
# if that is the final walking value.
#
SWING_HEIGHT = 0.008


CP1_COEF = 0.333
CP2_COEF = 0.666


# ============================================================
# CORRECTION SWEEP
#
# 0 -> 3 cm
# ============================================================

CORRECTION_MAGNITUDES = np.array(
    [
        0.000,
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
    ],
    dtype=float,
)


# ============================================================
# BEZIER PHASE RESOLUTION
#
# 101 points:
#
#     0.00, 0.01, ..., 1.00
#
# No mj_step().
# ============================================================

N_PHASE_SAMPLES = 101


# ============================================================
# FEASIBILITY THRESHOLDS
# ============================================================

# Cartesian error larger than 1 mm is considered FAIL.
MAX_CARTESIAN_ERROR_FAIL = 1.0e-3


# Joint closer than 2 deg to its physical limit is WARNING.
JOINT_LIMIT_MARGIN_WARNING = np.deg2rad(
    2.0
)


# Near-singular Jacobian warning.
SIGMA_WARNING_THRESHOLD = SIGMA_THRESHOLD


# Large numerical effort warning.
ITERATION_WARNING_THRESHOLD = 20


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
#   hip_yaw,
#   hip_roll,
#   hip_pitch,
#   knee,
#   ankle
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
# NOMINAL REACHABLE FOOTHOLD
#
# These joint configurations are used ONLY to generate a
# reachable nominal Cartesian foothold.
#
# They are NOT provided to IK during the sweep.
# ============================================================

TARGET_Q_LEFT = np.array(
    [
        0.0,
        0.08,
        0.55,
        -1.15,
        0.7854,
    ],
    dtype=float,
)


TARGET_Q_RIGHT = np.array(
    [
        0.0,
        -0.08,
        -0.55,
        1.15,
        -0.7854,
    ],
    dtype=float,
)


# ============================================================
# MUJOCO NAMES
# ============================================================

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
# MUJOCO JOINT HELPERS
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
            f"Không tìm thấy joint '{joint_name}'."
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
    q,
):

    adr = get_joint_qpos_address(
        model,
        joint_name,
    )

    data.qpos[
        adr
    ] = float(
        q
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
    ).reshape(-1)

    if q_leg.shape != (5,):
        raise ValueError(
            "q_leg phải có shape (5,)."
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
# SITE HELPERS
# ============================================================

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
            f"Không tìm thấy site '{site_name}'."
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

    # Floating-base position
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

    # Floating-base quaternion
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
# NOMINAL FOOTHOLD
# ============================================================

def generate_nominal_foothold(
    model,
    data,
    leg,
):

    qpos_saved = (
        data.qpos.copy()
    )

    qvel_saved = (
        data.qvel.copy()
    )

    if leg == "L":

        q_target = (
            TARGET_Q_LEFT
        )

    elif leg == "R":

        q_target = (
            TARGET_Q_RIGHT
        )

    else:

        raise ValueError(
            "leg phải là 'L' hoặc 'R'."
        )

    set_leg_configuration(
        model,
        data,
        leg,
        q_target,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    W_p_nominal = (
        get_ankle_position(
            model,
            data,
            leg,
        )
    )

    # Restore original state
    data.qpos[:] = (
        qpos_saved
    )

    data.qvel[:] = (
        qvel_saved
    )

    mujoco.mj_forward(
        model,
        data,
    )

    return W_p_nominal


# ============================================================
# APPLY IK SOLUTION
#
# q_ik:
#
# [
#   HY,
#   HR,
#   HP,
#   K
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
    ).reshape(-1)

    if q_ik.shape != (4,):
        raise ValueError(
            "q_ik phải có shape (4,)."
        )

    joint_names = (
        LEG_JOINT_NAMES[
            leg
        ][0:4]
    )

    for (
        joint_name,
        q,
    ) in zip(
        joint_names,
        q_ik,
    ):

        set_joint_position(
            model,
            data,
            joint_name,
            q,
        )


# ============================================================
# ANKLE COMPENSATION
#
# Flat ground:
#
#     q_A =
#
#     -q_HP_actual
#     -q_K_actual
# ============================================================

def compute_and_apply_ankle(
    model,
    data,
    leg,
):

    if leg == "L":

        hp_name = (
            "L_hip_pitch"
        )

        knee_name = (
            "L_knee"
        )

        ankle_name = (
            "L_ankle"
        )

    elif leg == "R":

        hp_name = (
            "R_hip_pitch"
        )

        knee_name = (
            "R_knee"
        )

        ankle_name = (
            "R_ankle"
        )

    else:

        raise ValueError(
            "leg phải là 'L' hoặc 'R'."
        )

    q_hp_actual = get_joint_position(
        model,
        data,
        hp_name,
    )

    q_k_actual = get_joint_position(
        model,
        data,
        knee_name,
    )

    q_ankle_des = (
        -SLOPE_PITCH
        -
        q_hp_actual
        -
        q_k_actual
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
# JOINT-LIMIT ANALYSIS
# ============================================================

def check_leg_joint_limits(
    model,
    data,
    leg,
):

    minimum_margin = np.inf

    nearest_joint = None

    outside_limit = False

    near_limit = False

    joint_margins = {}


    for joint_name in LEG_JOINT_NAMES[
        leg
    ]:

        joint_id = get_joint_id(
            model,
            joint_name,
        )

        q = get_joint_position(
            model,
            data,
            joint_name,
        )

        # MuJoCo flag:
        #
        # 0 = unlimited
        # 1 = limited
        #
        limited = bool(
            model.jnt_limited[
                joint_id
            ]
        )

        if not limited:

            joint_margins[
                joint_name
            ] = np.inf

            continue


        q_min = float(
            model.jnt_range[
                joint_id,
                0,
            ]
        )

        q_max = float(
            model.jnt_range[
                joint_id,
                1,
            ]
        )


        # ----------------------------------------------------
        # Outside actual physical range
        # ----------------------------------------------------

        if (
            q < q_min - 1e-9
            or
            q > q_max + 1e-9
        ):

            outside_limit = True


        # ----------------------------------------------------
        # Distance to nearest limit
        # ----------------------------------------------------

        margin = min(
            q - q_min,
            q_max - q,
        )

        margin = float(
            margin
        )

        joint_margins[
            joint_name
        ] = margin


        if margin < minimum_margin:

            minimum_margin = (
                margin
            )

            nearest_joint = (
                joint_name
            )


        if margin <= JOINT_LIMIT_MARGIN_WARNING:

            near_limit = True


    return {

        "outside_limit":
            outside_limit,

        "near_limit":
            near_limit,

        "minimum_margin":
            minimum_margin,

        "nearest_joint":
            nearest_joint,

        "joint_margins":
            joint_margins,
    }


# ============================================================
# ONE CORRECTION CASE
# ============================================================

def evaluate_correction_case(
    model,
    data,
    leg,
    W_p_nominal,
    correction,
):

    correction = np.asarray(
        correction,
        dtype=float,
    ).reshape(
        3
    )


    # ========================================================
    # RESET ROBOT
    #
    # Each candidate starts from exactly the same state.
    # ========================================================

    set_initial_state(
        model,
        data,
    )


    W_p_liftoff = (
        get_ankle_position(
            model,
            data,
            leg,
        )
    )


    W_p_foothold = (
        W_p_nominal
        +
        correction
    )


    # ========================================================
    # LOGS FOR THIS CANDIDATE
    # ========================================================

    phases = np.linspace(
        0.0,
        1.0,
        N_PHASE_SAMPLES,
    )

    error_log = []

    sigma_log = []

    iteration_log = []

    converged_log = []

    joint_margin_log = []

    nearest_joint_log = []

    outside_limit_count = 0

    near_limit_count = 0

    exception_message = None


    # ========================================================
    # CONTINUOUS SWING
    #
    # State from phase k becomes initial guess for phase k+1.
    #
    # This mimics online execution:
    #
    #     reference(k)
    #       ->
    #     IK(k)
    #       ->
    #     q(k)
    #       ->
    #     reference(k+1)
    #
    # Still NO mj_step().
    # ========================================================

    try:

        for phase in phases:

            # =================================================
            # 1. BEZIER
            # =================================================

            bezier_result = (
                evaluate_bezier_swing_trajectory(

                    W_p_liftoff=
                        W_p_liftoff,

                    W_p_foothold=
                        W_p_foothold,

                    swing_phase=
                        float(
                            phase
                        ),

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


            W_p_ankle_des = (
                bezier_result[
                    "W_p_swing_des"
                ]
            )


            # =================================================
            # 2. HYBRID NUMERICAL IK
            # =================================================

            (
                q_ik,
                diag,

            ) = solve_numerical_IK_hybrid(

                model=model,

                data=data,

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
                    f"IK returned None at phase "
                    f"{phase:.4f}"
                )


            q_ik = np.asarray(
                q_ik,
                dtype=float,
            )


            if not np.all(
                np.isfinite(
                    q_ik
                )
            ):

                raise RuntimeError(
                    f"IK returned NaN/Inf at phase "
                    f"{phase:.4f}"
                )


            # =================================================
            # 3. APPLY HY, HR, HP, K
            # =================================================

            apply_ik_solution(
                model,
                data,
                leg,
                q_ik,
            )


            mujoco.mj_forward(
                model,
                data,
            )


            # =================================================
            # 4. ACTUAL-STATE ANKLE COMPENSATION
            # =================================================

            compute_and_apply_ankle(
                model,
                data,
                leg,
            )


            mujoco.mj_forward(
                model,
                data,
            )


            # =================================================
            # 5. FORWARD-KINEMATICS ERROR
            # =================================================

            W_p_ankle_actual = (
                get_ankle_position(
                    model,
                    data,
                    leg,
                )
            )


            cartesian_error = float(
                np.linalg.norm(
                    W_p_ankle_des
                    -
                    W_p_ankle_actual
                )
            )


            error_log.append(
                cartesian_error
            )


            # =================================================
            # 6. DIAGNOSTICS
            # =================================================

            sigma_min = float(
                diag.get(
                    "sigma_min",
                    np.nan,
                )
            )


            iterations = int(
                diag.get(
                    "iterations",
                    0,
                )
            )


            position_converged = bool(
                diag.get(
                    "position_converged",
                    diag.get(
                        "converged",
                        False,
                    ),
                )
            )


            sigma_log.append(
                sigma_min
            )

            iteration_log.append(
                iterations
            )

            converged_log.append(
                position_converged
            )


            # =================================================
            # 7. JOINT LIMIT CHECK
            # =================================================

            limit_result = (
                check_leg_joint_limits(
                    model,
                    data,
                    leg,
                )
            )


            minimum_margin = (
                limit_result[
                    "minimum_margin"
                ]
            )


            joint_margin_log.append(
                minimum_margin
            )


            nearest_joint_log.append(
                limit_result[
                    "nearest_joint"
                ]
            )


            if limit_result[
                "outside_limit"
            ]:

                outside_limit_count += 1


            if limit_result[
                "near_limit"
            ]:

                near_limit_count += 1


    except Exception as exc:

        exception_message = str(
            exc
        )


    # ========================================================
    # HANDLE TOTAL FAILURE
    # ========================================================

    if len(
        error_log
    ) == 0:

        return {

            "status":
                "FAIL",

            "correction":
                correction.copy(),

            "W_p_foothold":
                W_p_foothold.copy(),

            "max_error":
                np.inf,

            "mean_error":
                np.inf,

            "min_sigma":
                0.0,

            "max_iterations":
                IK_MAX_ITERATIONS,

            "nonconverged_count":
                N_PHASE_SAMPLES,

            "near_limit_count":
                N_PHASE_SAMPLES,

            "outside_limit_count":
                N_PHASE_SAMPLES,

            "min_joint_margin":
                -np.inf,

            "nearest_joint":
                None,

            "exception":
                exception_message,
        }


    # ========================================================
    # SUMMARY
    # ========================================================

    error_log = np.asarray(
        error_log,
        dtype=float,
    )

    sigma_log = np.asarray(
        sigma_log,
        dtype=float,
    )

    iteration_log = np.asarray(
        iteration_log,
        dtype=int,
    )

    converged_log = np.asarray(
        converged_log,
        dtype=bool,
    )

    joint_margin_log = np.asarray(
        joint_margin_log,
        dtype=float,
    )


    max_error = float(
        np.max(
            error_log
        )
    )


    mean_error = float(
        np.mean(
            error_log
        )
    )


    finite_sigma = sigma_log[
        np.isfinite(
            sigma_log
        )
    ]


    if finite_sigma.size > 0:

        min_sigma = float(
            np.min(
                finite_sigma
            )
        )

    else:

        min_sigma = np.nan


    max_iterations = int(
        np.max(
            iteration_log
        )
    )


    nonconverged_count = int(
        np.sum(
            ~converged_log
        )
    )


    finite_margin = joint_margin_log[
        np.isfinite(
            joint_margin_log
        )
    ]


    if finite_margin.size > 0:

        min_joint_margin = float(
            np.min(
                finite_margin
            )
        )

        min_index = int(
            np.nanargmin(
                joint_margin_log
            )
        )

        nearest_joint = (
            nearest_joint_log[
                min_index
            ]
        )

    else:

        min_joint_margin = np.inf

        nearest_joint = None


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    status = "PASS"


    # --------------------------------------------------------
    # Hard failure
    # --------------------------------------------------------

    if exception_message is not None:

        status = "FAIL"


    if outside_limit_count > 0:

        status = "FAIL"


    if max_error > MAX_CARTESIAN_ERROR_FAIL:

        status = "FAIL"


    # --------------------------------------------------------
    # Warning conditions
    #
    # These are not necessarily impossible configurations,
    # but are not good candidates for a robust walking clip.
    # --------------------------------------------------------

    if status != "FAIL":

        warning = False


        if nonconverged_count > 0:

            warning = True


        if near_limit_count > 0:

            warning = True


        if (
            np.isfinite(
                min_sigma
            )
            and
            min_sigma
            <
            SIGMA_WARNING_THRESHOLD
        ):

            warning = True


        if (
            max_iterations
            >=
            ITERATION_WARNING_THRESHOLD
        ):

            warning = True


        if warning:

            status = "WARNING"


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status":
            status,

        "correction":
            correction.copy(),

        "W_p_foothold":
            W_p_foothold.copy(),

        "max_error":
            max_error,

        "mean_error":
            mean_error,

        "min_sigma":
            min_sigma,

        "max_iterations":
            max_iterations,

        "nonconverged_count":
            nonconverged_count,

        "near_limit_count":
            near_limit_count,

        "outside_limit_count":
            outside_limit_count,

        "min_joint_margin":
            min_joint_margin,

        "nearest_joint":
            nearest_joint,

        "exception":
            exception_message,
    }


# ============================================================
# PRINT ONE RESULT
# ============================================================

def print_result_row(
    label,
    result,
):

    correction = (
        result[
            "correction"
        ]
    )


    magnitude_cm = (
        np.linalg.norm(
            correction[
                0:2
            ]
        )
        *
        100.0
    )


    max_error_mm = (
        result[
            "max_error"
        ]
        *
        1000.0
    )


    min_sigma = (
        result[
            "min_sigma"
        ]
    )


    margin = (
        result[
            "min_joint_margin"
        ]
    )


    if np.isfinite(
        margin
    ):

        margin_deg = np.rad2deg(
            margin
        )

    else:

        margin_deg = np.inf


    print(
        f"{label:>5} | "
        f"{magnitude_cm:7.3f} | "
        f"{result['status']:>7} | "
        f"{max_error_mm:10.4f} | "
        f"{min_sigma:11.4e} | "
        f"{result['max_iterations']:8d} | "
        f"{result['nonconverged_count']:8d} | "
        f"{margin_deg:10.3f} | "
        f"{str(result['nearest_joint']):>14}"
    )


# ============================================================
# RUN ONE DIRECTION
# ============================================================

def run_direction_sweep(
    model,
    data,
    leg,
    W_p_nominal,
    axis,
    sign,
):

    if axis == "x":

        axis_index = 0

        axis_label = "X"

    elif axis == "y":

        axis_index = 1

        axis_label = "Y"

    else:

        raise ValueError(
            "axis phải là 'x' hoặc 'y'."
        )


    if sign > 0:

        sign_label = "+"

    else:

        sign_label = "-"


    print()

    print(
        "=" * 110
    )

    print(
        f"{sign_label}{axis_label} CORRECTION SWEEP"
    )

    print(
        "=" * 110
    )

    print(
        " case | cm corr |  status | max err mm |   min sigma | max iter | nonconv. | margin deg | nearest joint"
    )

    print(
        "-" * 110
    )


    results = []


    for magnitude in CORRECTION_MAGNITUDES:

        correction = np.zeros(
            3,
            dtype=float,
        )


        correction[
            axis_index
        ] = (
            sign
            *
            magnitude
        )


        result = evaluate_correction_case(

            model=
                model,

            data=
                data,

            leg=
                leg,

            W_p_nominal=
                W_p_nominal,

            correction=
                correction,
        )


        results.append(
            result
        )


        label = (
            f"{sign_label}{axis_label}"
        )


        print_result_row(
            label,
            result,
        )


    return results


# ============================================================
# FIND LARGEST PASS
# ============================================================

def find_largest_pass(
    results,
    axis_index,
):

    pass_values = []


    for result in results:

        if result[
            "status"
        ] == "PASS":

            value = abs(
                result[
                    "correction"
                ][
                    axis_index
                ]
            )

            pass_values.append(
                value
            )


    if len(
        pass_values
    ) == 0:

        return None


    return float(
        max(
            pass_values
        )
    )


# ============================================================
# PREPARE SIGNED RESULT ARRAY FOR PLOTTING
# ============================================================

def combine_signed_results(
    positive_results,
    negative_results,
    axis_index,
):

    combined = []


    # Negative branch
    for result in negative_results:

        value = (
            result[
                "correction"
            ][
                axis_index
            ]
        )

        if np.isclose(
            value,
            0.0
        ):
            continue

        combined.append(
            result
        )


    # Zero and positive branch
    combined.extend(
        positive_results
    )


    combined.sort(
        key=lambda item:
            item[
                "correction"
            ][
                axis_index
            ]
    )


    return combined


# ============================================================
# PLOT ONE AXIS
# ============================================================

def plot_axis_results(
    results,
    axis,
):

    if axis == "x":

        axis_index = 0

        title_axis = "X"

        provisional_clip = (
            DEFAULT_MAX_CORRECTION_X
        )

    else:

        axis_index = 1

        title_axis = "Y"

        provisional_clip = (
            DEFAULT_MAX_CORRECTION_Y
        )


    correction_cm = np.array(
        [
            result[
                "correction"
            ][
                axis_index
            ]
            *
            100.0

            for result in results
        ],
        dtype=float,
    )


    max_error_mm = np.array(
        [
            result[
                "max_error"
            ]
            *
            1000.0

            for result in results
        ],
        dtype=float,
    )


    min_sigma = np.array(
        [
            result[
                "min_sigma"
            ]

            for result in results
        ],
        dtype=float,
    )


    max_iterations = np.array(
        [
            result[
                "max_iterations"
            ]

            for result in results
        ],
        dtype=float,
    )


    joint_margin_deg = np.array(
        [
            np.rad2deg(
                result[
                    "min_joint_margin"
                ]
            )

            if np.isfinite(
                result[
                    "min_joint_margin"
                ]
            )

            else np.nan

            for result in results
        ],
        dtype=float,
    )


    provisional_clip_cm = (
        provisional_clip
        *
        100.0
    )


    # ========================================================
    # FIGURE 1
    # MAX CARTESIAN ERROR
    # ========================================================

    plt.figure()

    plt.plot(
        correction_cm,
        max_error_mm,
        marker="o",
    )

    plt.axvline(
        provisional_clip_cm,
        linestyle="--",
    )

    plt.axvline(
        -provisional_clip_cm,
        linestyle="--",
    )

    plt.xlabel(
        f"{title_axis} correction [cm]"
    )

    plt.ylabel(
        "Maximum Cartesian error [mm]"
    )

    plt.title(
        f"Raibert {title_axis} Correction - IK Error"
    )

    plt.grid(
        True
    )


    # ========================================================
    # FIGURE 2
    # MINIMUM SIGMA
    # ========================================================

    plt.figure()

    plt.plot(
        correction_cm,
        min_sigma,
        marker="o",
    )

    plt.axhline(
        SIGMA_WARNING_THRESHOLD,
        linestyle="--",
    )

    plt.axvline(
        provisional_clip_cm,
        linestyle="--",
    )

    plt.axvline(
        -provisional_clip_cm,
        linestyle="--",
    )

    plt.xlabel(
        f"{title_axis} correction [cm]"
    )

    plt.ylabel(
        "Minimum sigma"
    )

    plt.title(
        f"Raibert {title_axis} Correction - Jacobian Conditioning"
    )

    plt.grid(
        True
    )


    # ========================================================
    # FIGURE 3
    # MAX IK ITERATIONS
    # ========================================================

    plt.figure()

    plt.plot(
        correction_cm,
        max_iterations,
        marker="o",
    )

    plt.axhline(
        ITERATION_WARNING_THRESHOLD,
        linestyle="--",
    )

    plt.axvline(
        provisional_clip_cm,
        linestyle="--",
    )

    plt.axvline(
        -provisional_clip_cm,
        linestyle="--",
    )

    plt.xlabel(
        f"{title_axis} correction [cm]"
    )

    plt.ylabel(
        "Maximum IK iterations"
    )

    plt.title(
        f"Raibert {title_axis} Correction - IK Iterations"
    )

    plt.grid(
        True
    )


    # ========================================================
    # FIGURE 4
    # MIN JOINT-LIMIT MARGIN
    # ========================================================

    plt.figure()

    plt.plot(
        correction_cm,
        joint_margin_deg,
        marker="o",
    )

    plt.axhline(
        np.rad2deg(
            JOINT_LIMIT_MARGIN_WARNING
        ),
        linestyle="--",
    )

    plt.axvline(
        provisional_clip_cm,
        linestyle="--",
    )

    plt.axvline(
        -provisional_clip_cm,
        linestyle="--",
    )

    plt.xlabel(
        f"{title_axis} correction [cm]"
    )

    plt.ylabel(
        "Minimum joint-limit margin [deg]"
    )

    plt.title(
        f"Raibert {title_axis} Correction - Joint Limit Margin"
    )

    plt.grid(
        True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "RAIBERT CORRECTION FEASIBILITY TEST"
    )

    print(
        "=" * 90
    )


    if SWING_LEG not in (
        "L",
        "R",
    ):

        raise ValueError(
            "SWING_LEG phải là 'L' hoặc 'R'."
        )


    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"Không tìm thấy XML:\n"
            f"{XML_PATH}"
        )


    print(
        f"XML                  : {XML_PATH}"
    )

    print(
        f"Swing leg            : {SWING_LEG}"
    )

    print(
        f"T_swing              : {T_SWING:.4f} s"
    )

    print(
        f"Swing height         : {SWING_HEIGHT:.4f} m"
    )

    print(
        f"Bezier samples       : {N_PHASE_SAMPLES}"
    )

    print(
        f"Current X clip       : "
        f"{DEFAULT_MAX_CORRECTION_X * 100.0:.2f} cm"
    )

    print(
        f"Current Y clip       : "
        f"{DEFAULT_MAX_CORRECTION_Y * 100.0:.2f} cm"
    )

    print(
        f"Joint warning margin : "
        f"{np.rad2deg(JOINT_LIMIT_MARGIN_WARNING):.2f} deg"
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


    W_p_liftoff = (
        get_ankle_position(
            model,
            data,
            SWING_LEG,
        )
    )


    W_p_nominal = (
        generate_nominal_foothold(
            model,
            data,
            SWING_LEG,
        )
    )


    print()

    print(
        "Initial ankle center:"
    )

    print(
        W_p_liftoff
    )

    print()

    print(
        "Nominal foothold:"
    )

    print(
        W_p_nominal
    )

    print()

    print(
        "Nominal displacement:"
    )

    print(
        W_p_nominal
        -
        W_p_liftoff
    )


    # ========================================================
    # SWEEP
    # ========================================================

    positive_x = run_direction_sweep(

        model=
            model,

        data=
            data,

        leg=
            SWING_LEG,

        W_p_nominal=
            W_p_nominal,

        axis=
            "x",

        sign=
            +1.0,
    )


    negative_x = run_direction_sweep(

        model=
            model,

        data=
            data,

        leg=
            SWING_LEG,

        W_p_nominal=
            W_p_nominal,

        axis=
            "x",

        sign=
            -1.0,
    )


    positive_y = run_direction_sweep(

        model=
            model,

        data=
            data,

        leg=
            SWING_LEG,

        W_p_nominal=
            W_p_nominal,

        axis=
            "y",

        sign=
            +1.0,
    )


    negative_y = run_direction_sweep(

        model=
            model,

        data=
            data,

        leg=
            SWING_LEG,

        W_p_nominal=
            W_p_nominal,

        axis=
            "y",

        sign=
            -1.0,
    )


    # ========================================================
    # LARGEST CLEAN PASS
    # ========================================================

    x_plus_pass = find_largest_pass(
        positive_x,
        0,
    )

    x_minus_pass = find_largest_pass(
        negative_x,
        0,
    )

    y_plus_pass = find_largest_pass(
        positive_y,
        1,
    )

    y_minus_pass = find_largest_pass(
        negative_y,
        1,
    )


    print()

    print(
        "=" * 90
    )

    print(
        "CLEAN PASS SUMMARY"
    )

    print(
        "=" * 90
    )


    def print_pass(
        name,
        value,
    ):

        if value is None:

            print(
                f"{name:<12}: no clean PASS"
            )

        else:

            print(
                f"{name:<12}: "
                f"{value * 100.0:.2f} cm"
            )


    print_pass(
        "+X",
        x_plus_pass,
    )

    print_pass(
        "-X",
        x_minus_pass,
    )

    print_pass(
        "+Y",
        y_plus_pass,
    )

    print_pass(
        "-Y",
        y_minus_pass,
    )


    # ========================================================
    # SYMMETRIC CLIP SUGGESTION
    #
    # Conservative:
    #
    #     max symmetric clip
    #
    #     =
    #
    #     min(positive feasible, negative feasible)
    #
    # ========================================================

    if (
        x_plus_pass is not None
        and
        x_minus_pass is not None
    ):

        suggested_x = min(
            x_plus_pass,
            x_minus_pass,
        )

        print()

        print(
            f"Conservative symmetric X clip candidate: "
            f"{suggested_x * 100.0:.2f} cm"
        )

    else:

        suggested_x = None


    if (
        y_plus_pass is not None
        and
        y_minus_pass is not None
    ):

        suggested_y = min(
            y_plus_pass,
            y_minus_pass,
        )

        print(
            f"Conservative symmetric Y clip candidate: "
            f"{suggested_y * 100.0:.2f} cm"
        )

    else:

        suggested_y = None


    print()

    print(
        "NOTE:"
    )

    print(
        "These values are valid only for the current:"
    )

    print(
        "  initial pose"
    )

    print(
        "  nominal foothold"
    )

    print(
        "  swing height"
    )

    print(
        "  Bezier trajectory"
    )

    print()

    print(
        "Final walking clip should later be validated "
        "over multiple gait states."
    )


    # ========================================================
    # PLOT
    # ========================================================

    x_results = combine_signed_results(
        positive_results=
            positive_x,

        negative_results=
            negative_x,

        axis_index=
            0,
    )


    y_results = combine_signed_results(
        positive_results=
            positive_y,

        negative_results=
            negative_y,

        axis_index=
            1,
    )


    plot_axis_results(
        x_results,
        "x",
    )


    plot_axis_results(
        y_results,
        "y",
    )


    plt.show()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()