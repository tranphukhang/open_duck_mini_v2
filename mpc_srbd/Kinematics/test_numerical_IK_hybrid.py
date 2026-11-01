from pathlib import Path

import mujoco
import numpy as np

from numerical_IK_hybrid import (
    solve_numerical_IK_hybrid
)


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = Path(
    __file__
).resolve().parent


MPC_SRBD_DIR = (
    CURRENT_DIR.parent
)


PROJECT_ROOT = (
    MPC_SRBD_DIR.parent
)


XML_PATH = (
    PROJECT_ROOT
    /
    "xmls"
    /
    "scene_flat_terrian_v3.xml"
)


# ============================================================
# TEST PARAMETERS
# ============================================================

NEWTON_SIGMA_THRESHOLD = 1e-12

GRADIENT_SIGMA_THRESHOLD = 1e6


ALPHA_NEWTON = 1.0

ETA_GRADIENT = 1.0


IK_TOLERANCE = 1e-5

IK_MAX_ITERATIONS = 100


# ============================================================
# LEG INFORMATION
# ============================================================

LEG_CONFIG = {

    "L": {

        "site_name": "L6",

        "joint_names": [
            "L_hip_yaw",
            "L_hip_roll",
            "L_hip_pitch",
            "L_knee",
            "L_ankle"
        ]

    },

    "R": {

        "site_name": "R6",

        "joint_names": [
            "R_hip_yaw",
            "R_hip_roll",
            "R_hip_pitch",
            "R_knee",
            "R_ankle"
        ]

    }

}


# ============================================================
# INITIALIZE MUJOCO
# ============================================================

def initialize_mujoco():

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"Không tìm thấy XML:\n{XML_PATH}"
        )


    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )


    data = mujoco.MjData(
        model
    )


    # --------------------------------------------------------
    # Home keyframe
    # --------------------------------------------------------

    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )


    if home_key_id < 0:

        raise RuntimeError(
            "Không tìm thấy keyframe 'home'."
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


    return model, data


# ============================================================
# GET SITE POSITION
# ============================================================

def get_site_position(
    model,
    data,
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


    return (
        data.site_xpos[
            site_id
        ].copy()
    )


# ============================================================
# GET LEG QPOS ADDRESSES
# ============================================================

def get_leg_qpos_addresses(
    model,
    swing_foot
):

    joint_names = LEG_CONFIG[
        swing_foot
    ][
        "joint_names"
    ]


    qpos_addresses = []


    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )


        if joint_id < 0:

            raise RuntimeError(
                f"Không tìm thấy joint '{joint_name}'."
            )


        qpos_addresses.append(
            int(
                model.jnt_qposadr[
                    joint_id
                ]
            )
        )


    return np.asarray(
        qpos_addresses,
        dtype=int
    )


# ============================================================
# EVALUATE IK SOLUTION
#
# Evaluate q_swing_des in a SEPARATE MuJoCo data object.
#
# Therefore the actual simulation data is not changed.
# ============================================================

def evaluate_IK_solution(
    model,
    data,
    q_swing_des,
    swing_foot
):

    eval_data = mujoco.MjData(
        model
    )


    # --------------------------------------------------------
    # Copy current configuration
    # --------------------------------------------------------

    eval_data.qpos[:] = (
        data.qpos.copy()
    )


    eval_data.qvel[:] = (
        data.qvel.copy()
    )


    # --------------------------------------------------------
    # Get swing-leg joint addresses
    # --------------------------------------------------------

    qpos_addresses = (
        get_leg_qpos_addresses(
            model,
            swing_foot
        )
    )


    # --------------------------------------------------------
    # Apply desired IK configuration
    # --------------------------------------------------------

    eval_data.qpos[
        qpos_addresses
    ] = q_swing_des


    # --------------------------------------------------------
    # Forward kinematics
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        eval_data
    )


    site_name = LEG_CONFIG[
        swing_foot
    ][
        "site_name"
    ]


    W_p_result = get_site_position(
        model,
        eval_data,
        site_name
    )


    return W_p_result


# ============================================================
# TEST 1
#
# DOUBLE SUPPORT
# ============================================================

def test_double_support(
    model,
    data
):

    q_swing_des, diagnostics = (
        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=None,
            swing_foot=None
        )
    )


    assert q_swing_des is None


    assert diagnostics[
        "converged"
    ]


    assert diagnostics[
        "iterations"
    ] == 0


    assert diagnostics[
        "swing_foot"
    ] is None


    print(
        "TEST 1 PASSED: Double Support"
    )


# ============================================================
# TEST 2
#
# RIGHT FOOT - NEWTON IK
#
# Use a small target close to the current R6 position.
# ============================================================

def test_right_foot_newton(
    model,
    data
):

    W_p_R6 = get_site_position(
        model,
        data,
        "R6"
    )


    # --------------------------------------------------------
    # Small reachable displacement
    # --------------------------------------------------------

    target_offset = np.array(
        [
            0.0010,
            0.0000,
            0.0010
        ],
        dtype=float
    )


    W_p_target = (
        W_p_R6
        +
        target_offset
    )


    initial_error_norm = float(
        np.linalg.norm(
            W_p_target
            -
            W_p_R6
        )
    )


    # --------------------------------------------------------
    # Save actual simulation state
    # --------------------------------------------------------

    qpos_before = (
        data.qpos.copy()
    )


    qvel_before = (
        data.qvel.copy()
    )


    # --------------------------------------------------------
    # Solve IK
    # --------------------------------------------------------

    q_swing_des, diagnostics = (
        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=W_p_target,
            swing_foot="R",
            sigma_threshold=NEWTON_SIGMA_THRESHOLD,
            alpha_newton=ALPHA_NEWTON,
            eta_gradient=ETA_GRADIENT,
            tolerance=IK_TOLERANCE,
            max_iterations=IK_MAX_ITERATIONS
        )
    )


    # --------------------------------------------------------
    # Basic output checks
    # --------------------------------------------------------

    assert q_swing_des.shape == (
        5,
    )


    assert np.all(
        np.isfinite(
            q_swing_des
        )
    )


    assert diagnostics[
        "swing_foot"
    ] == "R"


    # --------------------------------------------------------
    # Newton should be used
    # --------------------------------------------------------

    assert diagnostics[
        "newton_iterations"
    ] > 0, (
        "\nNewton không được sử dụng.\n"
        f"Diagnostics = {diagnostics}"
    )


    # --------------------------------------------------------
    # Final Cartesian error should be smaller
    # --------------------------------------------------------

    assert diagnostics[
        "final_error_norm"
    ] < initial_error_norm, (
        "\nIK Newton không làm giảm sai số.\n"
        f"Initial error = {initial_error_norm}\n"
        f"Final error   = "
        f"{diagnostics['final_error_norm']}"
    )


    # --------------------------------------------------------
    # Actual simulation state must be restored
    # --------------------------------------------------------

    assert np.allclose(
        data.qpos,
        qpos_before,
        atol=1e-12
    ), (
        "\ndata.qpos bị thay đổi bởi IK solver."
    )


    assert np.allclose(
        data.qvel,
        qvel_before,
        atol=1e-12
    ), (
        "\ndata.qvel bị thay đổi bởi IK solver."
    )


    # --------------------------------------------------------
    # Evaluate returned solution using fresh MuJoCo data
    # --------------------------------------------------------

    W_p_result = evaluate_IK_solution(
        model=model,
        data=data,
        q_swing_des=q_swing_des,
        swing_foot="R"
    )


    result_error = float(
        np.linalg.norm(
            W_p_target
            -
            W_p_result
        )
    )


    assert result_error < initial_error_norm, (
        "\nNghiệm IK chân phải không cải thiện target.\n"
        f"Initial error = {initial_error_norm}\n"
        f"Result error  = {result_error}"
    )


    print(
        "TEST 2 PASSED: Right Foot Newton IK"
    )


# ============================================================
# TEST 3
#
# LEFT FOOT - NEWTON IK
# ============================================================

def test_left_foot_newton(
    model,
    data
):

    W_p_L6 = get_site_position(
        model,
        data,
        "L6"
    )


    target_offset = np.array(
        [
            0.0010,
            0.0000,
            0.0010
        ],
        dtype=float
    )


    W_p_target = (
        W_p_L6
        +
        target_offset
    )


    initial_error_norm = float(
        np.linalg.norm(
            W_p_target
            -
            W_p_L6
        )
    )


    q_swing_des, diagnostics = (
        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=W_p_target,
            swing_foot="L",
            sigma_threshold=NEWTON_SIGMA_THRESHOLD,
            alpha_newton=ALPHA_NEWTON,
            eta_gradient=ETA_GRADIENT,
            tolerance=IK_TOLERANCE,
            max_iterations=IK_MAX_ITERATIONS
        )
    )


    assert q_swing_des.shape == (
        5,
    )


    assert diagnostics[
        "swing_foot"
    ] == "L"


    assert diagnostics[
        "newton_iterations"
    ] > 0


    assert diagnostics[
        "final_error_norm"
    ] < initial_error_norm


    # --------------------------------------------------------
    # Evaluate solution
    # --------------------------------------------------------

    W_p_result = evaluate_IK_solution(
        model=model,
        data=data,
        q_swing_des=q_swing_des,
        swing_foot="L"
    )


    result_error = float(
        np.linalg.norm(
            W_p_target
            -
            W_p_result
        )
    )


    assert result_error < initial_error_norm


    print(
        "TEST 3 PASSED: Left Foot Newton IK"
    )


# ============================================================
# TEST 4
#
# FORCE GRADIENT DESCENT
#
# Use a very large sigma threshold so:
#
#     sigma_min <= sigma_threshold
#
# always selects Gradient Descent.
#
# We do not require full convergence here.
# We only verify:
#
#     - Gradient is selected
#     - Newton is not selected
#     - Error decreases
# ============================================================

def test_gradient_descent_branch(
    model,
    data
):

    W_p_R6 = get_site_position(
        model,
        data,
        "R6"
    )


    target_offset = np.array(
        [
            0.0010,
            0.0000,
            0.0005
        ],
        dtype=float
    )


    W_p_target = (
        W_p_R6
        +
        target_offset
    )


    initial_error_norm = float(
        np.linalg.norm(
            target_offset
        )
    )


    qpos_before = (
        data.qpos.copy()
    )


    qvel_before = (
        data.qvel.copy()
    )


    q_swing_des, diagnostics = (
        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=W_p_target,
            swing_foot="R",
            sigma_threshold=GRADIENT_SIGMA_THRESHOLD,
            alpha_newton=ALPHA_NEWTON,
            eta_gradient=ETA_GRADIENT,
            tolerance=1e-8,
            max_iterations=50
        )
    )


    # --------------------------------------------------------
    # Must use Gradient Descent
    # --------------------------------------------------------

    assert diagnostics[
        "gradient_iterations"
    ] > 0, (
        "\nGradient Descent không được sử dụng."
    )


    assert diagnostics[
        "newton_iterations"
    ] == 0, (
        "\nNewton không được phép chạy trong test này."
    )


    # --------------------------------------------------------
    # Error should decrease
    # --------------------------------------------------------

    assert diagnostics[
        "final_error_norm"
    ] < initial_error_norm, (
        "\nGradient Descent không làm giảm sai số.\n"
        f"Initial error = {initial_error_norm}\n"
        f"Final error   = "
        f"{diagnostics['final_error_norm']}"
    )


    # --------------------------------------------------------
    # Solver must restore simulation state
    # --------------------------------------------------------

    assert np.allclose(
        data.qpos,
        qpos_before,
        atol=1e-12
    )


    assert np.allclose(
        data.qvel,
        qvel_before,
        atol=1e-12
    )


    assert np.all(
        np.isfinite(
            q_swing_des
        )
    )


    print(
        "TEST 4 PASSED: Gradient Descent Branch"
    )


# ============================================================
# TEST 5
#
# STATE RESTORATION
#
# Explicitly check that solving IK does NOT move the actual
# simulated robot.
# ============================================================

def test_state_restoration(
    model,
    data
):

    W_p_R6_before = get_site_position(
        model,
        data,
        "R6"
    )


    qpos_before = (
        data.qpos.copy()
    )


    qvel_before = (
        data.qvel.copy()
    )


    W_p_target = (
        W_p_R6_before
        +
        np.array(
            [
                0.002,
                0.000,
                0.001
            ],
            dtype=float
        )
    )


    solve_numerical_IK_hybrid(
        model=model,
        data=data,
        W_p_swing_des=W_p_target,
        swing_foot="R",
        sigma_threshold=NEWTON_SIGMA_THRESHOLD,
        alpha_newton=ALPHA_NEWTON,
        eta_gradient=ETA_GRADIENT,
        tolerance=IK_TOLERANCE,
        max_iterations=IK_MAX_ITERATIONS
    )


    # --------------------------------------------------------
    # Check state
    # --------------------------------------------------------

    assert np.allclose(
        data.qpos,
        qpos_before,
        atol=1e-12
    )


    assert np.allclose(
        data.qvel,
        qvel_before,
        atol=1e-12
    )


    # --------------------------------------------------------
    # Check physical site position
    # --------------------------------------------------------

    W_p_R6_after = get_site_position(
        model,
        data,
        "R6"
    )


    assert np.allclose(
        W_p_R6_after,
        W_p_R6_before,
        atol=1e-12
    ), (
        "\nIK solver đã làm thay đổi vị trí thật của R6."
    )


    print(
        "TEST 5 PASSED: Simulation State Restoration"
    )


# ============================================================
# TEST 6
#
# ZERO CARTESIAN ERROR
#
# If desired position equals current physical position:
#
#     error = 0
#
# Solver should immediately converge without iterations.
# ============================================================

def test_zero_error(
    model,
    data
):

    W_p_R6 = get_site_position(
        model,
        data,
        "R6"
    )


    qpos_addresses = (
        get_leg_qpos_addresses(
            model,
            "R"
        )
    )


    q_current = (
        data.qpos[
            qpos_addresses
        ].copy()
    )


    q_swing_des, diagnostics = (
        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=W_p_R6,
            swing_foot="R",
            sigma_threshold=NEWTON_SIGMA_THRESHOLD,
            alpha_newton=ALPHA_NEWTON,
            eta_gradient=ETA_GRADIENT,
            tolerance=IK_TOLERANCE,
            max_iterations=IK_MAX_ITERATIONS
        )
    )


    assert diagnostics[
        "converged"
    ]


    assert diagnostics[
        "iterations"
    ] == 0


    assert np.isclose(
        diagnostics[
            "final_error_norm"
        ],
        0.0,
        atol=1e-12
    )


    assert np.allclose(
        q_swing_des,
        q_current,
        atol=1e-12
    )


    print(
        "TEST 6 PASSED: Zero Cartesian Error"
    )


# ============================================================
# TEST 7
#
# INVALID SWING FOOT
# ============================================================

def test_invalid_swing_foot(
    model,
    data
):

    try:

        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=np.zeros(3),
            swing_foot="X"
        )


        raise AssertionError(
            "Invalid swing_foot phải gây lỗi."
        )


    except ValueError:

        pass


    print(
        "TEST 7 PASSED: Invalid Swing Foot"
    )


# ============================================================
# TEST 8
#
# INVALID TARGET
# ============================================================

def test_invalid_target(
    model,
    data
):

    # --------------------------------------------------------
    # None target during swing
    # --------------------------------------------------------

    try:

        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=None,
            swing_foot="R"
        )


        raise AssertionError(
            "Swing với target=None phải gây lỗi."
        )


    except ValueError:

        pass


    # --------------------------------------------------------
    # Wrong target shape
    # --------------------------------------------------------

    try:

        solve_numerical_IK_hybrid(
            model=model,
            data=data,
            W_p_swing_des=np.zeros(2),
            swing_foot="R"
        )


        raise AssertionError(
            "Target sai shape phải gây lỗi."
        )


    except ValueError:

        pass


    print(
        "TEST 8 PASSED: Invalid Target"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=================================================="
    )

    print(
        " TEST NUMERICAL IK - HYBRID"
    )

    print(
        "==================================================\n"
    )


    print(
        f"XML = {XML_PATH}"
    )


    print(
        f"Newton sigma threshold = "
        f"{NEWTON_SIGMA_THRESHOLD}"
    )


    print(
        f"Gradient sigma threshold = "
        f"{GRADIENT_SIGMA_THRESHOLD}"
    )


    print(
        f"alpha_newton = {ALPHA_NEWTON}"
    )


    print(
        f"eta_gradient = {ETA_GRADIENT}"
    )


    print(
        f"tolerance = {IK_TOLERANCE:.8f} m"
    )


    print(
        f"max_iterations = {IK_MAX_ITERATIONS}\n"
    )


    # ========================================================
    # INITIALIZE MUJOCO
    # ========================================================

    model, data = initialize_mujoco()


    # ========================================================
    # PRINT HOME FOOT POSITIONS
    # ========================================================

    W_p_L6 = get_site_position(
        model,
        data,
        "L6"
    )


    W_p_R6 = get_site_position(
        model,
        data,
        "R6"
    )


    print(
        "Home W_p_L6 ="
    )

    print(
        W_p_L6
    )


    print(
        "\nHome W_p_R6 ="
    )

    print(
        W_p_R6
    )


    print()


    # ========================================================
    # RUN TESTS
    # ========================================================

    test_double_support(
        model,
        data
    )


    test_right_foot_newton(
        model,
        data
    )


    test_left_foot_newton(
        model,
        data
    )


    test_gradient_descent_branch(
        model,
        data
    )


    test_state_restoration(
        model,
        data
    )


    test_zero_error(
        model,
        data
    )


    test_invalid_swing_foot(
        model,
        data
    )


    test_invalid_target(
        model,
        data
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n--------------------------------------------------"
    )

    print(
        "ALL TESTS PASSED"
    )

    print(
        "--------------------------------------------------\n"
    )


if __name__ == "__main__":

    main()