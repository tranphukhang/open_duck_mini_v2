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
# Controlled IK task:
#
#     x
#     y
#     z
#     pitch
#
# Roll and yaw are deliberately left free.
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
# GET PITCH FROM ROTATION MATRIX
#
# ZYX convention:
#
#     R = Rz(yaw) Ry(pitch) Rx(roll)
#
# For a rotation matrix:
#
#     pitch
#         =
#     atan2(
#         -R[2,0],
#         sqrt(R[0,0]^2 + R[1,0]^2)
#     )
#
# The IK sites are close to a flat-foot configuration, so the
# pitch extraction is well away from the +/- pi/2 singularity
# during normal walking.
# ============================================================

def _get_pitch_from_rotation_matrix(
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


    denominator = np.sqrt(

        R[
            0,
            0
        ] ** 2

        +

        R[
            1,
            0
        ] ** 2
    )


    pitch = np.arctan2(

        -R[
            2,
            0
        ],

        denominator
    )


    return float(
        pitch
    )


# ============================================================
# WRAP ANGLE TO [-pi, pi]
# ============================================================

def _wrap_to_pi(
    angle
):

    angle = float(
        angle
    )


    wrapped = (

        angle
        +
        np.pi

    ) % (

        2.0
        *
        np.pi

    ) - np.pi


    return float(
        wrapped
    )


# ============================================================
# COMPUTE PITCH-ONLY ORIENTATION ERROR
#
# Only pitch is constrained.
#
# Roll:
#     free
#
# Yaw:
#     free
#
#
# Current pitch:
#
#     pitch_current
#
#
# Desired pitch:
#
#     pitch_des
#
#
# Scalar pitch error:
#
#     e_pitch
#
# =
#
#     wrap_to_pi(
#         pitch_des - pitch_current
#     )
#
#
# The error is represented as a 1D vector so that it can be
# concatenated directly with the 3D Cartesian position error:
#
#     [e_x, e_y, e_z, e_pitch]^T
# ============================================================

def _compute_pitch_orientation_error(
    R_current,
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


    pitch_des = float(
        pitch_des
    )


    if not np.isfinite(
        pitch_des
    ):

        raise ValueError(
            "pitch_des phải hữu hạn."
        )


    pitch_current = (
        _get_pitch_from_rotation_matrix(
            R_current
        )
    )


    pitch_error_scalar = (
        _wrap_to_pi(

            pitch_des

            -

            pitch_current
        )
    )


    pitch_error = np.array(
        [
            pitch_error_scalar
        ],
        dtype=float
    )


    if pitch_error.shape != (
        1,
    ):

        raise RuntimeError(
            "pitch_error phải có shape (1,)."
        )


    if not np.all(
        np.isfinite(
            pitch_error
        )
    ):

        raise RuntimeError(
            "pitch_error chứa NaN hoặc Inf."
        )


    return pitch_error


# ============================================================
# COMPUTE LEG 4D TASK JACOBIAN
#
# Task:
#
#     x
#     y
#     z
#     pitch
#
#
# Therefore:
#
#     J_task in R^(4x5)
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
#     final row:
#         World omega_y
#
#
# Roll row:
#     NOT included
#
# Yaw row:
#     NOT included
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
    # Pitch-only angular Jacobian
    #
    # World-frame omega_y is row index 1.
    # Shape:
    #
    #     (1, 5)
    # --------------------------------------------------------

    J_pitch = (
        J_rotation[
            1:2,
            :
        ].copy()
    )


    if J_pitch.shape != (
        1,
        5
    ):

        raise RuntimeError(
            "J_pitch phải có shape (1, 5), "
            f"nhưng nhận được {J_pitch.shape}."
        )


    # --------------------------------------------------------
    # Complete 4D task Jacobian
    # --------------------------------------------------------

    J_task = np.vstack(
        (
            J_position,
            J_pitch
        )
    )


    if J_task.shape != (
        4,
        5
    ):

        raise RuntimeError(
            "J_task phải có shape (4, 5), "
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
# COMPUTE CURRENT 4D TASK ERROR
#
#     e
#
# =
#
#     [
#         e_x
#         e_y
#         e_z
#         e_pitch
#     ]
# ============================================================

def _compute_task_error(
    data,
    site_id,
    W_p_swing_des,
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
    # Pitch-only orientation error
    # --------------------------------------------------------

    orientation_error = (
        _compute_pitch_orientation_error(

            R_current=R_current,

            pitch_des=pitch_swing_des
        )
    )


    # --------------------------------------------------------
    # Complete 4D error
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
        1,
    ):

        raise RuntimeError(
            "orientation_error phải có shape (1,)."
        )


    if task_error.shape != (
        4,
    ):

        raise RuntimeError(
            "task_error phải có shape (4,)."
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
    Solve the 4D swing-leg numerical inverse kinematics.

    Controlled task:

        [
            foot x
            foot y
            foot z
            foot pitch
        ]

    Roll is deliberately NOT controlled.

    Yaw is deliberately NOT controlled.


    IK sites:

        Left:
            L6_IK

        Right:
            R6_IK


    The argument:

        roll_swing_des

    is temporarily retained only for backward compatibility with
    the current swing_low_level_controller.py interface.

    It is NOT used in the IK task.

    It will be removed from the low-level caller in the next step.


    For flat-ground walking:

        pitch_swing_des = 0

    is the desired pitch command.


    Hybrid method:

        sigma_min > sigma_threshold

            -> pseudo-Newton

    otherwise:

            -> Jacobian-transpose Gradient Descent


    Position convergence:

        ||e_position|| <= tolerance


    Pitch convergence:

        ||e_pitch|| <= orientation_tolerance


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

            # Retained only for backward-compatible diagnostics.
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
    #
    # Only pitch is used.
    #
    # roll_swing_des is intentionally ignored for backward
    # compatibility with the current caller.
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
            # 10.1 CURRENT 4D TASK ERROR
            # =================================================

            (
                task_error,
                position_error,
                orientation_error

            ) = _compute_task_error(

                data=data,

                site_id=site_id,

                W_p_swing_des=W_p_swing_des,

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
            # Pitch orientation:
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
            # 10.4 COMPLETE 4D TASK JACOBIAN
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
        # Complete 4D error
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
        # Pitch-only orientation error
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

        # Retained only for backward compatibility.
        # It is NOT part of the IK task.
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
