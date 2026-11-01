import mujoco
import numpy as np


# ============================================================
# IMPORT NUMERICAL IK METHODS
# ============================================================

try:

    from .numerical_IK_newton import (
        compute_newton_step
    )

    from .numerical_IK_gradient_descent import (
        compute_gradient_descent_step
    )

except ImportError:

    from numerical_IK_newton import (
        compute_newton_step
    )

    from numerical_IK_gradient_descent import (
        compute_gradient_descent_step
    )


# ============================================================
# LEG CONFIGURATION
#
# Numerical IK uses:
#
#     L6_IK
#     R6_IK
#
# These sites:
#
#     - are located at the same physical positions as L6/R6
#     - are World-aligned at the home configuration
#
# Therefore:
#
#     position task
#         -> L6_IK / R6_IK position
#
#     orientation task
#         -> L6_IK / R6_IK orientation
#
#
# Joint order:
#
#     0. hip yaw
#     1. hip roll
#     2. hip pitch
#     3. knee
#     4. ankle
# ============================================================

LEG_CONFIG = {

    "L": {

        "site_name": "L6_IK",

        "joint_names": [
            "L_hip_yaw",
            "L_hip_roll",
            "L_hip_pitch",
            "L_knee",
            "L_ankle"
        ]

    },

    "R": {

        "site_name": "R6_IK",

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
# VECTOR VALIDATION
# ============================================================

def _validate_vector(
    value,
    name,
    expected_shape
):

    value = np.asarray(
        value,
        dtype=float
    ).reshape(-1)


    if value.shape != expected_shape:

        raise ValueError(
            f"{name} phải có shape {expected_shape}, "
            f"nhưng nhận được {value.shape}."
        )


    if not np.all(
        np.isfinite(
            value
        )
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    return value


# ============================================================
# GET LEG JOINT INFORMATION
# ============================================================

def _get_leg_joint_information(
    model,
    swing_foot
):

    config = LEG_CONFIG[
        swing_foot
    ]


    site_name = config[
        "site_name"
    ]


    joint_names = config[
        "joint_names"
    ]


    # --------------------------------------------------------
    # IK site
    # --------------------------------------------------------

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )


    if site_id < 0:

        raise ValueError(
            f"Không tìm thấy site '{site_name}'."
        )


    # --------------------------------------------------------
    # Swing-leg joint information
    # --------------------------------------------------------

    joint_ids = []

    qpos_addresses = []

    dof_addresses = []


    for joint_name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )


        if joint_id < 0:

            raise ValueError(
                f"Không tìm thấy joint '{joint_name}'."
            )


        joint_ids.append(
            joint_id
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


        qpos_addresses.append(
            qpos_address
        )


        dof_addresses.append(
            dof_address
        )


    return (
        site_id,
        site_name,
        joint_names,

        np.asarray(
            joint_ids,
            dtype=int
        ),

        np.asarray(
            qpos_addresses,
            dtype=int
        ),

        np.asarray(
            dof_addresses,
            dtype=int
        )
    )


# ============================================================
# APPLY JOINT LIMITS
# ============================================================

def _apply_joint_limits(
    model,
    joint_ids,
    q_candidate
):

    q_limited = (
        q_candidate.copy()
    )


    for i, joint_id in enumerate(
        joint_ids
    ):

        if model.jnt_limited[
            joint_id
        ]:

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


            q_limited[
                i
            ] = np.clip(
                q_limited[
                    i
                ],
                q_min,
                q_max
            )


    return q_limited


# ============================================================
# ROTATION MATRIX FROM ROLL-PITCH-YAW
#
# Convention:
#
#     R = Rz(yaw) Ry(pitch) Rx(roll)
#
# This is used only to construct the desired World-frame
# orientation of L6_IK / R6_IK.
#
# Since the IK sites are World-aligned at home:
#
#     roll_des  = 0
#     pitch_des = 0
#
# corresponds naturally to a flat foot.
# ============================================================

def _rotation_matrix_from_rpy(
    roll,
    pitch,
    yaw
):

    roll = float(
        roll
    )

    pitch = float(
        pitch
    )

    yaw = float(
        yaw
    )


    c_r = np.cos(
        roll
    )

    s_r = np.sin(
        roll
    )


    c_p = np.cos(
        pitch
    )

    s_p = np.sin(
        pitch
    )


    c_y = np.cos(
        yaw
    )

    s_y = np.sin(
        yaw
    )


    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_r, -s_r],
            [0.0, s_r, c_r]
        ],
        dtype=float
    )


    R_y = np.array(
        [
            [c_p, 0.0, s_p],
            [0.0, 1.0, 0.0],
            [-s_p, 0.0, c_p]
        ],
        dtype=float
    )


    R_z = np.array(
        [
            [c_y, -s_y, 0.0],
            [s_y, c_y, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=float
    )


    R = (
        R_z
        @
        R_y
        @
        R_x
    )


    return R


# ============================================================
# GET YAW FROM ROTATION MATRIX
#
# Because L6_IK / R6_IK are near the flat-foot configuration,
# the yaw extraction below no longer suffers from the
# pitch = +/- pi/2 singular configuration that existed for
# the original L6/R6 site frames.
#
# ZYX convention:
#
#     yaw = atan2(R[1,0], R[0,0])
# ============================================================

def _get_yaw_from_rotation_matrix(
    R
):

    R = np.asarray(
        R,
        dtype=float
    )


    if R.shape != (
        3,
        3
    ):

        raise ValueError(
            "R phải có shape (3, 3)."
        )


    if not np.all(
        np.isfinite(
            R
        )
    ):

        raise ValueError(
            "R chứa NaN hoặc Inf."
        )


    yaw = np.arctan2(
        R[
            1,
            0
        ],
        R[
            0,
            0
        ]
    )


    return float(
        yaw
    )


# ============================================================
# COMPUTE ROLL-PITCH ORIENTATION ERROR
#
# Only:
#
#     roll
#     pitch
#
# are controlled.
#
# Yaw is deliberately left free.
#
#
# At each IK iteration:
#
#     yaw_des = yaw_current
#
# while:
#
#     roll_des
#     pitch_des
#
# are commanded.
#
#
# Desired orientation:
#
#     R_des
#
# =
#
#     Rz(yaw_current)
#     Ry(pitch_des)
#     Rx(roll_des)
#
#
# Orientation error:
#
#     e_R
#
# =
#
#     1/2 vee(
#
#         R_des R_current^T
#
#         -
#
#         R_current R_des^T
#
#     )
#
#
# e_R is expressed in the World frame.
#
# Only:
#
#     e_R_x
#     e_R_y
#
# are used for the 5D IK task.
# ============================================================

def _compute_roll_pitch_orientation_error(
    R_current,
    roll_des,
    pitch_des
):

    R_current = np.asarray(
        R_current,
        dtype=float
    )


    if R_current.shape != (
        3,
        3
    ):

        raise ValueError(
            "R_current phải có shape (3, 3)."
        )


    if not np.all(
        np.isfinite(
            R_current
        )
    ):

        raise ValueError(
            "R_current chứa NaN hoặc Inf."
        )


    roll_des = float(
        roll_des
    )


    pitch_des = float(
        pitch_des
    )


    if not np.isfinite(
        roll_des
    ):

        raise ValueError(
            "roll_des phải hữu hạn."
        )


    if not np.isfinite(
        pitch_des
    ):

        raise ValueError(
            "pitch_des phải hữu hạn."
        )


    # --------------------------------------------------------
    # Current yaw
    #
    # Yaw is deliberately not controlled.
    # --------------------------------------------------------

    yaw_current = (
        _get_yaw_from_rotation_matrix(
            R_current
        )
    )


    # --------------------------------------------------------
    # Desired orientation
    # --------------------------------------------------------

    R_des = (
        _rotation_matrix_from_rpy(

            roll=roll_des,

            pitch=pitch_des,

            yaw=yaw_current
        )
    )


    # --------------------------------------------------------
    # Rotation error matrix
    # --------------------------------------------------------

    error_matrix = (

        R_des
        @
        R_current.T

        -

        R_current
        @
        R_des.T
    )


    # --------------------------------------------------------
    # vee(.) operator
    #
    #     [  0  -z   y ]
    #     [  z   0  -x ]
    #     [ -y   x   0 ]
    #
    #             |
    #             v
    #
    #         [x y z]^T
    # --------------------------------------------------------

    e_rotation = (
        0.5
        *
        np.array(
            [
                error_matrix[
                    2,
                    1
                ],

                error_matrix[
                    0,
                    2
                ],

                error_matrix[
                    1,
                    0
                ]
            ],
            dtype=float
        )
    )


    # --------------------------------------------------------
    # 2D orientation task:
    #
    #     roll correction
    #     pitch correction
    #
    # Yaw correction is excluded.
    # --------------------------------------------------------

    e_roll_pitch = (
        e_rotation[
            0:2
        ].copy()
    )


    if e_roll_pitch.shape != (
        2,
    ):

        raise RuntimeError(
            "e_roll_pitch phải có shape (2,)."
        )


    if not np.all(
        np.isfinite(
            e_roll_pitch
        )
    ):

        raise RuntimeError(
            "e_roll_pitch chứa NaN hoặc Inf."
        )


    return e_roll_pitch


# ============================================================
# COMPUTE LEG 5D TASK JACOBIAN
#
# Task:
#
#     x
#     y
#     z
#     roll
#     pitch
#
#
# Therefore:
#
#     J_task in R^(5x5)
#
#
# mj_jacSite gives:
#
#     J_position_full
#
#     J_rotation_full
#
# expressed in the World frame.
#
#
# Since L6_IK / R6_IK are rigidly attached to the foot:
#
#     first 3 rows:
#         Cartesian position Jacobian
#
#     last 2 rows:
#         World omega_x
#         World omega_y
# ============================================================

def _compute_leg_task_jacobian(
    model,
    data,
    site_id,
    dof_addresses
):

    J_position_full = np.zeros(
        (
            3,
            model.nv
        ),
        dtype=float
    )


    J_rotation_full = np.zeros(
        (
            3,
            model.nv
        ),
        dtype=float
    )


    mujoco.mj_jacSite(
        model,
        data,
        J_position_full,
        J_rotation_full,
        site_id
    )


    # --------------------------------------------------------
    # Select only the five swing-leg joint columns
    # --------------------------------------------------------

    J_position = (
        J_position_full[
            :,
            dof_addresses
        ].copy()
    )


    J_rotation = (
        J_rotation_full[
            :,
            dof_addresses
        ].copy()
    )


    if J_position.shape != (
        3,
        5
    ):

        raise RuntimeError(
            "J_position phải có shape (3, 5), "
            f"nhưng nhận được {J_position.shape}."
        )


    if J_rotation.shape != (
        3,
        5
    ):

        raise RuntimeError(
            "J_rotation phải có shape (3, 5), "
            f"nhưng nhận được {J_rotation.shape}."
        )


    if not np.all(
        np.isfinite(
            J_position
        )
    ):

        raise RuntimeError(
            "J_position chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            J_rotation
        )
    ):

        raise RuntimeError(
            "J_rotation chứa NaN hoặc Inf."
        )


    # --------------------------------------------------------
    # Only World-frame omega_x and omega_y
    # --------------------------------------------------------

    J_roll_pitch = (
        J_rotation[
            0:2,
            :
        ].copy()
    )


    # --------------------------------------------------------
    # Complete 5D task Jacobian
    # --------------------------------------------------------

    J_task = np.vstack(
        (
            J_position,
            J_roll_pitch
        )
    )


    if J_task.shape != (
        5,
        5
    ):

        raise RuntimeError(
            "J_task phải có shape (5, 5), "
            f"nhưng nhận được {J_task.shape}."
        )


    if not np.all(
        np.isfinite(
            J_task
        )
    ):

        raise RuntimeError(
            "J_task chứa NaN hoặc Inf."
        )


    return J_task


# ============================================================
# COMPUTE CURRENT 5D TASK ERROR
#
#     e
#
# =
#
#     [
#         e_x
#         e_y
#         e_z
#         e_roll
#         e_pitch
#     ]
# ============================================================

def _compute_task_error(
    data,
    site_id,
    W_p_swing_des,
    roll_swing_des,
    pitch_swing_des
):

    # --------------------------------------------------------
    # Current IK-site position
    # --------------------------------------------------------

    W_p_current = (
        data.site_xpos[
            site_id
        ].copy()
    )


    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    position_error = (

        W_p_swing_des

        -

        W_p_current
    )


    # --------------------------------------------------------
    # Current IK-site orientation
    #
    # site_xmat:
    #
    #     local site frame -> World frame
    # --------------------------------------------------------

    R_current = (
        data.site_xmat[
            site_id
        ].reshape(
            3,
            3
        ).copy()
    )


    # --------------------------------------------------------
    # Roll/pitch orientation error
    # --------------------------------------------------------

    orientation_error = (
        _compute_roll_pitch_orientation_error(

            R_current=R_current,

            roll_des=roll_swing_des,

            pitch_des=pitch_swing_des
        )
    )


    # --------------------------------------------------------
    # Complete 5D error
    # --------------------------------------------------------

    task_error = np.concatenate(
        (
            position_error,
            orientation_error
        )
    )


    if position_error.shape != (
        3,
    ):

        raise RuntimeError(
            "position_error phải có shape (3,)."
        )


    if orientation_error.shape != (
        2,
    ):

        raise RuntimeError(
            "orientation_error phải có shape (2,)."
        )


    if task_error.shape != (
        5,
    ):

        raise RuntimeError(
            "task_error phải có shape (5,)."
        )


    if not np.all(
        np.isfinite(
            task_error
        )
    ):

        raise RuntimeError(
            "task_error chứa NaN hoặc Inf."
        )


    return (
        task_error,
        position_error,
        orientation_error
    )


# ============================================================
# HYBRID NUMERICAL IK
# ============================================================

def solve_numerical_IK_hybrid(
    model,
    data,
    W_p_swing_des,
    swing_foot,
    roll_swing_des=0.0,
    pitch_swing_des=0.0,
    sigma_threshold=1e-4,
    alpha_newton=1.0,
    eta_gradient=1.0,
    tolerance=1e-4,
    orientation_tolerance=1e-3,
    max_iterations=50
):
    """
    Solve the 5D swing-leg numerical inverse kinematics.

    Controlled task:

        [
            foot x
            foot y
            foot z
            foot roll
            foot pitch
        ]

    IK sites:

        Left:
            L6_IK

        Right:
            R6_IK

    These sites are World-aligned at the home configuration.

    Therefore for flat-ground walking:

        roll_swing_des  = 0

        pitch_swing_des = 0

    is the natural flat-foot orientation command.


    Yaw is deliberately not controlled.

    At every iteration:

        yaw_des = yaw_current


    Hybrid method:

        sigma_min > sigma_threshold

            -> pseudo-Newton

    otherwise:

            -> Jacobian-transpose Gradient Descent


    Position convergence:

        ||e_position|| <= tolerance


    Orientation convergence:

        ||e_orientation|| <= orientation_tolerance


    Important:

    The function temporarily writes candidate joint positions
    to MuJoCo data in order to evaluate forward kinematics.

    Before returning, the original simulation state is restored.

    Therefore this function does NOT directly move the robot.
    """


    # ========================================================
    # 1. VALIDATE SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R",
        None
    ):

        raise ValueError(
            "swing_foot chỉ được là 'L', 'R' hoặc None."
        )


    # ========================================================
    # 2. DOUBLE SUPPORT
    # ========================================================

    if swing_foot is None:

        diagnostics = {

            "converged": True,

            "iterations": 0,

            "final_error": None,

            "final_error_norm": 0.0,

            "final_position_error": None,

            "final_position_error_norm": 0.0,

            "final_orientation_error": None,

            "final_orientation_error_norm": 0.0,

            "method_used": None,

            "newton_iterations": 0,

            "gradient_iterations": 0,

            "sigma_min": None,

            "swing_foot": None,

            "site_name": None,

            "joint_names": None,

            "roll_swing_des": None,

            "pitch_swing_des": None

        }


        return None, diagnostics


    # ========================================================
    # 3. VALIDATE DESIRED POSITION
    # ========================================================

    if W_p_swing_des is None:

        raise ValueError(
            "W_p_swing_des không được là None "
            "khi đang có swing foot."
        )


    W_p_swing_des = _validate_vector(
        W_p_swing_des,
        "W_p_swing_des",
        (
            3,
        )
    )


    # ========================================================
    # 4. VALIDATE DESIRED ORIENTATION
    # ========================================================

    roll_swing_des = float(
        roll_swing_des
    )


    pitch_swing_des = float(
        pitch_swing_des
    )


    if not np.isfinite(
        roll_swing_des
    ):

        raise ValueError(
            "roll_swing_des phải hữu hạn."
        )


    if not np.isfinite(
        pitch_swing_des
    ):

        raise ValueError(
            "pitch_swing_des phải hữu hạn."
        )


    # ========================================================
    # 5. VALIDATE NUMERICAL PARAMETERS
    # ========================================================

    sigma_threshold = float(
        sigma_threshold
    )


    alpha_newton = float(
        alpha_newton
    )


    eta_gradient = float(
        eta_gradient
    )


    tolerance = float(
        tolerance
    )


    orientation_tolerance = float(
        orientation_tolerance
    )


    if (
        not np.isfinite(
            sigma_threshold
        )
        or
        sigma_threshold < 0.0
    ):

        raise ValueError(
            "sigma_threshold phải hữu hạn và >= 0."
        )


    if (
        not np.isfinite(
            alpha_newton
        )
        or
        alpha_newton <= 0.0
    ):

        raise ValueError(
            "alpha_newton phải hữu hạn và > 0."
        )


    if (
        not np.isfinite(
            eta_gradient
        )
        or
        eta_gradient <= 0.0
    ):

        raise ValueError(
            "eta_gradient phải hữu hạn và > 0."
        )


    if (
        not np.isfinite(
            tolerance
        )
        or
        tolerance <= 0.0
    ):

        raise ValueError(
            "tolerance phải hữu hạn và > 0."
        )


    if (
        not np.isfinite(
            orientation_tolerance
        )
        or
        orientation_tolerance <= 0.0
    ):

        raise ValueError(
            "orientation_tolerance phải hữu hạn và > 0."
        )


    if not isinstance(
        max_iterations,
        (int, np.integer)
    ):

        raise TypeError(
            "max_iterations phải là số nguyên."
        )


    if max_iterations <= 0:

        raise ValueError(
            "max_iterations phải > 0."
        )


    # ========================================================
    # 6. GET SWING-LEG INFORMATION
    # ========================================================

    (
        site_id,
        site_name,
        joint_names,
        joint_ids,
        qpos_addresses,
        dof_addresses

    ) = _get_leg_joint_information(
        model,
        swing_foot
    )


    # ========================================================
    # 7. SAVE ORIGINAL MUJOCO STATE
    # ========================================================

    qpos_original = (
        data.qpos.copy()
    )


    qvel_original = (
        data.qvel.copy()
    )


    # ========================================================
    # 8. INITIAL GUESS
    #
    #     q^(0) = q_current
    # ========================================================

    q_candidate = (
        data.qpos[
            qpos_addresses
        ].copy()
    )


    # ========================================================
    # 9. DIAGNOSTIC INITIALIZATION
    # ========================================================

    converged = False

    iterations = 0

    newton_iterations = 0

    gradient_iterations = 0

    method_used = None

    sigma_min = None


    final_error = None

    final_error_norm = np.inf


    final_position_error = None

    final_position_error_norm = np.inf


    final_orientation_error = None

    final_orientation_error_norm = np.inf


    # ========================================================
    # 10. ITERATIVE IK
    # ========================================================

    try:

        # ----------------------------------------------------
        # Make sure MuJoCo kinematics are up to date
        # ----------------------------------------------------

        mujoco.mj_forward(
            model,
            data
        )


        for _ in range(
            max_iterations
        ):

            # =================================================
            # 10.1 CURRENT 5D TASK ERROR
            # =================================================

            (
                task_error,
                position_error,
                orientation_error

            ) = _compute_task_error(

                data=data,

                site_id=site_id,

                W_p_swing_des=W_p_swing_des,

                roll_swing_des=roll_swing_des,

                pitch_swing_des=pitch_swing_des
            )


            # =================================================
            # 10.2 ERROR NORMS
            # =================================================

            position_error_norm = float(
                np.linalg.norm(
                    position_error
                )
            )


            orientation_error_norm = float(
                np.linalg.norm(
                    orientation_error
                )
            )


            task_error_norm = float(
                np.linalg.norm(
                    task_error
                )
            )


            # =================================================
            # 10.3 CONVERGENCE CHECK
            #
            # Position:
            #
            #     [m]
            #
            # Orientation:
            #
            #     [rad]
            #
            # are checked separately.
            # =================================================

            if (

                position_error_norm
                <=
                tolerance

                and

                orientation_error_norm
                <=
                orientation_tolerance

            ):

                converged = True


                final_error = (
                    task_error.copy()
                )


                final_error_norm = (
                    task_error_norm
                )


                final_position_error = (
                    position_error.copy()
                )


                final_position_error_norm = (
                    position_error_norm
                )


                final_orientation_error = (
                    orientation_error.copy()
                )


                final_orientation_error_norm = (
                    orientation_error_norm
                )


                break


            # =================================================
            # 10.4 COMPLETE 5D TASK JACOBIAN
            # =================================================

            J_task = (
                _compute_leg_task_jacobian(

                    model=model,

                    data=data,

                    site_id=site_id,

                    dof_addresses=dof_addresses
                )
            )


            # =================================================
            # 10.5 MINIMUM SINGULAR VALUE
            # =================================================

            singular_values = (
                np.linalg.svd(

                    J_task,

                    compute_uv=False
                )
            )


            sigma_min = float(
                singular_values[
                    -1
                ]
            )


            # =================================================
            # 10.6 METHOD SELECTION
            # =================================================

            if sigma_min > sigma_threshold:

                # ---------------------------------------------
                # PSEUDO-NEWTON
                # ---------------------------------------------

                delta_q = (
                    compute_newton_step(

                        J=J_task,

                        error=task_error,

                        alpha_newton=alpha_newton
                    )
                )


                method_used = (
                    "newton"
                )


                newton_iterations += 1


            else:

                # ---------------------------------------------
                # GRADIENT DESCENT
                # ---------------------------------------------

                delta_q = (
                    compute_gradient_descent_step(

                        J=J_task,

                        error=task_error,

                        eta_gradient=eta_gradient
                    )
                )


                method_used = (
                    "gradient_descent"
                )


                gradient_iterations += 1


            # =================================================
            # 10.7 UPDATE CANDIDATE JOINT CONFIGURATION
            # =================================================

            q_candidate = (

                q_candidate

                +

                delta_q
            )


            # =================================================
            # 10.8 APPLY JOINT LIMITS
            # =================================================

            q_candidate = (
                _apply_joint_limits(

                    model=model,

                    joint_ids=joint_ids,

                    q_candidate=q_candidate
                )
            )


            # =================================================
            # 10.9 TEMPORARILY WRITE CANDIDATE TO MUJOCO
            # =================================================

            data.qpos[
                qpos_addresses
            ] = q_candidate


            # =================================================
            # 10.10 UPDATE FORWARD KINEMATICS
            # =================================================

            mujoco.mj_forward(
                model,
                data
            )


            iterations += 1


        # ====================================================
        # 11. FINAL ERROR
        # ====================================================

        (
            final_error,
            final_position_error,
            final_orientation_error

        ) = _compute_task_error(

            data=data,

            site_id=site_id,

            W_p_swing_des=W_p_swing_des,

            roll_swing_des=roll_swing_des,

            pitch_swing_des=pitch_swing_des
        )


        final_error_norm = float(
            np.linalg.norm(
                final_error
            )
        )


        final_position_error_norm = float(
            np.linalg.norm(
                final_position_error
            )
        )


        final_orientation_error_norm = float(
            np.linalg.norm(
                final_orientation_error
            )
        )


        if (

            final_position_error_norm
            <=
            tolerance

            and

            final_orientation_error_norm
            <=
            orientation_tolerance

        ):

            converged = True


        # ====================================================
        # 12. FINAL JOINT SOLUTION
        # ====================================================

        q_swing_des = (
            q_candidate.copy()
        )


        if q_swing_des.shape != (
            5,
        ):

            raise RuntimeError(
                "q_swing_des phải có shape (5,), "
                f"nhưng nhận được {q_swing_des.shape}."
            )


        if not np.all(
            np.isfinite(
                q_swing_des
            )
        ):

            raise RuntimeError(
                "q_swing_des chứa NaN hoặc Inf."
            )


    # ========================================================
    # 13. RESTORE REAL MUJOCO SIMULATION STATE
    # ========================================================

    finally:

        data.qpos[
            :
        ] = qpos_original


        data.qvel[
            :
        ] = qvel_original


        mujoco.mj_forward(
            model,
            data
        )


    # ========================================================
    # 14. DIAGNOSTICS
    # ========================================================

    diagnostics = {

        "converged": bool(
            converged
        ),


        "iterations": int(
            iterations
        ),


        # ----------------------------------------------------
        # Complete 5D error
        # ----------------------------------------------------

        "final_error": (
            final_error.copy()
        ),


        "final_error_norm": float(
            final_error_norm
        ),


        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        "final_position_error": (
            final_position_error.copy()
        ),


        "final_position_error_norm": float(
            final_position_error_norm
        ),


        # ----------------------------------------------------
        # Roll/pitch orientation error
        # ----------------------------------------------------

        "final_orientation_error": (
            final_orientation_error.copy()
        ),


        "final_orientation_error_norm": float(
            final_orientation_error_norm
        ),


        # ----------------------------------------------------
        # Numerical method
        # ----------------------------------------------------

        "method_used": (
            method_used
        ),


        "newton_iterations": int(
            newton_iterations
        ),


        "gradient_iterations": int(
            gradient_iterations
        ),


        "sigma_min": (

            None

            if sigma_min is None

            else float(
                sigma_min
            )
        ),


        # ----------------------------------------------------
        # Swing-leg information
        # ----------------------------------------------------

        "swing_foot": (
            swing_foot
        ),


        "site_name": (
            site_name
        ),


        "joint_names": (
            list(
                joint_names
            )
        ),


        # ----------------------------------------------------
        # Desired IK-site orientation
        # ----------------------------------------------------

        "roll_swing_des": float(
            roll_swing_des
        ),


        "pitch_swing_des": float(
            pitch_swing_des
        )

    }


    # ========================================================
    # 15. RETURN
    # ========================================================

    return (
        q_swing_des,
        diagnostics
    )