import mujoco
import numpy as np


# ============================================================
# IMPORT NUMERICAL IK METHODS
# ============================================================

try:
    from .numerical_IK_newton import compute_newton_step
    from .numerical_IK_gradient_descent import (
        compute_gradient_descent_step,
    )
except ImportError:
    from numerical_IK_newton import compute_newton_step
    from numerical_IK_gradient_descent import (
        compute_gradient_descent_step,
    )


# ============================================================
# LEG CONFIGURATION
# ============================================================

LEG_CONFIG = {
    "L": {
        "site_name": "L6_IK",
        "joint_names": [
            "L_hip_yaw",
            "L_hip_roll",
            "L_hip_pitch",
            "L_knee",
            "L_ankle",
        ],
    },
    "R": {
        "site_name": "R6_IK",
        "joint_names": [
            "R_hip_yaw",
            "R_hip_roll",
            "R_hip_pitch",
            "R_knee",
            "R_ankle",
        ],
    },
}


# ============================================================
# HIERARCHICAL IK DEFAULTS
#
# Priority:
#
#   Task 1:
#       Cartesian foot position [x, y, z]
#
#   Task 2:
#       Foot pitch stabilization
#
# There is NO posture task in this version.
#
# Roll is also NOT imposed as a separate Cartesian task.
#
# DEFAULT_K_POSTURE and DEFAULT_K_ROLL are retained only for
# interface compatibility with swing_low_level_controller.py.
# They are not used by the active IK update.
# ============================================================

DEFAULT_K_POSTURE = 0.0
DEFAULT_K_PITCH = 0.25
DEFAULT_K_ROLL = 0.0

DEFAULT_PITCH_TOLERANCE = 2.0e-3


# ============================================================
# PITCH MOTION COST
#
# Joint order:
#
#   0  hip yaw
#   1  hip roll
#   2  hip pitch
#   3  knee
#   4  ankle
#
# Larger number = more expensive to move.
#
# Therefore the pitch task prefers:
#
#   hip pitch + knee
#
# while strongly discouraging:
#
#   hip yaw + hip roll
#
# The ankle is still available to counter-rotate the sole when
# needed, but it is more expensive than hip pitch / knee.
# ============================================================

DEFAULT_PITCH_MOTION_COST = np.array(
    [
        30.0,   # hip yaw
        8.0,    # hip roll
        1.0,    # hip pitch
        1.0,    # knee
        3.0,    # ankle
    ],
    dtype=float,
)


# ============================================================
# POSTURE COMPATIBILITY
#
# These weights are retained only because the current external
# interface still accepts posture_joint_weights.
#
# The active solver does NOT compute a posture correction.
# ============================================================

DEFAULT_POSTURE_JOINT_WEIGHTS = np.zeros(
    5,
    dtype=float,
)


# ============================================================
# VALIDATION
# ============================================================

def _validate_vector(
    value,
    name,
    expected_shape,
):
    value = np.asarray(
        value,
        dtype=float,
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


def _validate_matrix(
    value,
    name,
    expected_shape,
):
    value = np.asarray(
        value,
        dtype=float,
    )

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

    return value.copy()


def _validate_nonnegative_scalar(
    value,
    name,
):
    value = float(
        value
    )

    if (
        not np.isfinite(
            value
        )
        or
        value < 0.0
    ):
        raise ValueError(
            f"{name} phải hữu hạn và >= 0."
        )

    return value


def _validate_positive_scalar(
    value,
    name,
):
    value = float(
        value
    )

    if (
        not np.isfinite(
            value
        )
        or
        value <= 0.0
    ):
        raise ValueError(
            f"{name} phải hữu hạn và > 0."
        )

    return value


def _validate_positive_weights(
    value,
    name,
):
    value = _validate_vector(
        value,
        name,
        (5,),
    )

    if np.any(
        value <= 0.0
    ):
        raise ValueError(
            f"{name} phải chứa các giá trị > 0."
        )

    return value


def _validate_nonnegative_weights(
    value,
    name,
):
    value = _validate_vector(
        value,
        name,
        (5,),
    )

    if np.any(
        value < 0.0
    ):
        raise ValueError(
            f"{name} không được chứa trọng số âm."
        )

    return value


# ============================================================
# LEG INFORMATION
# ============================================================

def _get_leg_joint_information(
    model,
    swing_foot,
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

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise ValueError(
            f"Không tìm thấy site '{site_name}'."
        )

    joint_ids = []
    qpos_addresses = []
    dof_addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise ValueError(
                f"Không tìm thấy joint '{joint_name}'."
            )

        joint_ids.append(
            int(
                joint_id
            )
        )

        qpos_addresses.append(
            int(
                model.jnt_qposadr[
                    joint_id
                ]
            )
        )

        dof_addresses.append(
            int(
                model.jnt_dofadr[
                    joint_id
                ]
            )
        )

    return (
        int(
            site_id
        ),
        str(
            site_name
        ),
        list(
            joint_names
        ),
        np.asarray(
            joint_ids,
            dtype=int,
        ),
        np.asarray(
            qpos_addresses,
            dtype=int,
        ),
        np.asarray(
            dof_addresses,
            dtype=int,
        ),
    )


# ============================================================
# JOINT LIMITS
# ============================================================

def _apply_joint_limits(
    model,
    joint_ids,
    q_candidate,
):
    q_candidate = _validate_vector(
        q_candidate,
        "q_candidate",
        (5,),
    )

    q_limited = q_candidate.copy()

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
                q_max,
            )

    return q_limited


# ============================================================
# SITE KINEMATICS
# ============================================================

def _get_site_rotation_matrix(
    data,
    site_id,
):
    return _validate_matrix(
        np.asarray(
            data.site_xmat[
                site_id
            ],
            dtype=float,
        ).reshape(
            3,
            3,
        ),
        "R_site",
        (
            3,
            3,
        ),
    )


def _compute_leg_jacobians(
    model,
    data,
    site_id,
    dof_addresses,
):
    J_position_full = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    J_rotation_full = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    mujoco.mj_jacSite(
        model,
        data,
        J_position_full,
        J_rotation_full,
        site_id,
    )

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

    J_position = _validate_matrix(
        J_position,
        "J_position",
        (
            3,
            5,
        ),
    )

    J_rotation = _validate_matrix(
        J_rotation,
        "J_rotation",
        (
            3,
            5,
        ),
    )

    return (
        J_position,
        J_rotation,
    )


# ============================================================
# PRIMARY POSITION ERROR
# ============================================================

def _compute_position_error(
    data,
    site_id,
    W_p_swing_des,
):
    W_p_swing_des = _validate_vector(
        W_p_swing_des,
        "W_p_swing_des",
        (3,),
    )

    W_p_current = np.asarray(
        data.site_xpos[
            site_id
        ],
        dtype=float,
    ).reshape(
        3,
    )

    if not np.all(
        np.isfinite(
            W_p_current
        )
    ):
        raise RuntimeError(
            "W_p_current chứa NaN hoặc Inf."
        )

    position_error = (
        W_p_swing_des
        -
        W_p_current
    )

    return _validate_vector(
        position_error,
        "position_error",
        (3,),
    )


# ============================================================
# ORIENTATION ERROR
#
# R_error maps CURRENT site orientation toward REFERENCE:
#
#     R_error = R_ref * R_current^T
#
# Small-angle world-frame orientation error:
#
#     e_R = vee(
#              0.5 * (R_error - R_error^T)
#          )
#
# We only use:
#
#     e_pitch = e_R[y]
#
# Roll and yaw are NOT Cartesian tasks in this hierarchy.
# ============================================================

def compute_world_orientation_error(
    R_current,
    R_reference,
):
    R_current = _validate_matrix(
        R_current,
        "R_current",
        (
            3,
            3,
        ),
    )

    R_reference = _validate_matrix(
        R_reference,
        "R_reference",
        (
            3,
            3,
        ),
    )

    R_error = (
        R_reference
        @
        R_current.T
    )

    e_rotation = 0.5 * np.array(
        [
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1],
        ],
        dtype=float,
    )

    return _validate_vector(
        e_rotation,
        "e_rotation",
        (3,),
    )


def compute_world_pitch_error(
    R_current,
    R_reference,
):
    e_rotation = compute_world_orientation_error(
        R_current=R_current,
        R_reference=R_reference,
    )

    return float(
        e_rotation[
            1
        ]
    )


# ============================================================
# NULL-SPACE PROJECTORS
# ============================================================

def compute_null_space_projector(
    J,
):
    J = _validate_matrix(
        J,
        "J",
        (
            3,
            5,
        ),
    )

    J_pinv = np.linalg.pinv(
        J
    )

    return (
        np.eye(
            5,
            dtype=float,
        )
        -
        J_pinv
        @
        J
    )


def compute_augmented_null_space_projector(
    J_position,
    J_pitch,
):
    J_position = _validate_matrix(
        J_position,
        "J_position",
        (
            3,
            5,
        ),
    )

    J_pitch = np.asarray(
        J_pitch,
        dtype=float,
    ).reshape(
        1,
        5,
    )

    J_pitch = _validate_matrix(
        J_pitch,
        "J_pitch",
        (
            1,
            5,
        ),
    )

    J_augmented = np.vstack(
        (
            J_position,
            J_pitch,
        )
    )

    J_augmented_pinv = np.linalg.pinv(
        J_augmented
    )

    N_augmented = (
        np.eye(
            5,
            dtype=float,
        )
        -
        J_augmented_pinv
        @
        J_augmented
    )

    return (
        N_augmented,
        J_augmented,
    )


# ============================================================
# LEGACY POSTURE-ONLY NULL-SPACE STEP
#
# Retained for compatibility with earlier unit tests.
# ============================================================

def compute_null_space_posture_step(
    J,
    q_candidate,
    q_posture_ref,
    k_posture,
):
    q_candidate = _validate_vector(
        q_candidate,
        "q_candidate",
        (5,),
    )

    q_posture_ref = _validate_vector(
        q_posture_ref,
        "q_posture_ref",
        (5,),
    )

    k_posture = _validate_nonnegative_scalar(
        k_posture,
        "k_posture",
    )

    N = compute_null_space_projector(
        J
    )

    posture_error = (
        q_posture_ref
        -
        q_candidate
    )

    delta_q_posture = (
        k_posture
        *
        (
            N
            @
            posture_error
        )
    )

    return (
        delta_q_posture,
        N,
    )


# ============================================================
# TASK 2:
# WEIGHTED PITCH CORRECTION
#
# We solve:
#
#     minimize
#
#         1/2 * delta_q^T W delta_q
#
#     subject to
#
#         J_position delta_q = 0
#
#         J_pitch delta_q
#             = k_pitch * e_pitch
#
#
# Closed-form weighted minimum-norm solution:
#
#     delta_q
#       =
#     W^-1 C^T
#     (C W^-1 C^T)^+
#     b
#
# where:
#
#     C = [ J_position
#           J_pitch    ]
#
#     b = [0,0,0,k_pitch*e_pitch]^T
#
#
# This gives pitch correction without disturbing the primary
# position task.
# ============================================================

def compute_weighted_pitch_step(
    J_position,
    J_pitch,
    pitch_error,
    k_pitch=DEFAULT_K_PITCH,
    pitch_motion_cost=DEFAULT_PITCH_MOTION_COST,
):
    J_position = _validate_matrix(
        J_position,
        "J_position",
        (
            3,
            5,
        ),
    )

    J_pitch = np.asarray(
        J_pitch,
        dtype=float,
    ).reshape(
        1,
        5,
    )

    J_pitch = _validate_matrix(
        J_pitch,
        "J_pitch",
        (
            1,
            5,
        ),
    )

    pitch_error = float(
        pitch_error
    )

    if not np.isfinite(
        pitch_error
    ):
        raise ValueError(
            "pitch_error phải hữu hạn."
        )

    k_pitch = _validate_nonnegative_scalar(
        k_pitch,
        "k_pitch",
    )

    pitch_motion_cost = _validate_positive_weights(
        pitch_motion_cost,
        "pitch_motion_cost",
    )

    C = np.vstack(
        (
            J_position,
            J_pitch,
        )
    )

    b = np.array(
        [
            0.0,
            0.0,
            0.0,
            k_pitch
            *
            pitch_error,
        ],
        dtype=float,
    )

    W_inverse = np.diag(
        1.0
        /
        pitch_motion_cost
    )

    middle = (
        C
        @
        W_inverse
        @
        C.T
    )

    delta_q_pitch = (
        W_inverse
        @
        C.T
        @
        np.linalg.pinv(
            middle
        )
        @
        b
    )

    delta_q_pitch = _validate_vector(
        delta_q_pitch,
        "delta_q_pitch",
        (5,),
    )

    position_leakage = float(
        np.linalg.norm(
            J_position
            @
            delta_q_pitch
        )
    )

    achieved_pitch_step = float(
        (
            J_pitch
            @
            delta_q_pitch
        )[
            0
        ]
    )

    requested_pitch_step = float(
        k_pitch
        *
        pitch_error
    )

    pitch_residual = float(
        requested_pitch_step
        -
        achieved_pitch_step
    )

    return {
        "delta_q_pitch":
            delta_q_pitch.copy(),

        "requested_pitch_step":
            requested_pitch_step,

        "achieved_pitch_step":
            achieved_pitch_step,

        "pitch_residual":
            pitch_residual,

        "position_leakage":
            position_leakage,

        "constraint_matrix":
            C.copy(),
    }


# ============================================================
# TASK 3:
# POSTURE IN NULL SPACE OF [POSITION + PITCH]
#
#     J_aug = [J_position
#              J_pitch]
#
#     N_aug = I - J_aug^+ J_aug
#
#     delta_q_posture
#       =
#     k_posture
#     N_aug
#     W_posture
#     (q_ref - q)
#
#
# Therefore the posture correction does not disturb either:
#
#     position
#     pitch
# ============================================================

def compute_hierarchical_posture_step(
    J_position,
    J_pitch,
    q_candidate,
    q_posture_ref,
    k_posture=DEFAULT_K_POSTURE,
    # Compatibility only; posture is not an active IK task.
    posture_joint_weights=DEFAULT_POSTURE_JOINT_WEIGHTS,
):
    q_candidate = _validate_vector(
        q_candidate,
        "q_candidate",
        (5,),
    )

    q_posture_ref = _validate_vector(
        q_posture_ref,
        "q_posture_ref",
        (5,),
    )

    k_posture = _validate_nonnegative_scalar(
        k_posture,
        "k_posture",
    )

    posture_joint_weights = _validate_nonnegative_weights(
        posture_joint_weights,
        "posture_joint_weights",
    )

    (
        N_augmented,
        J_augmented,
    ) = compute_augmented_null_space_projector(
        J_position=J_position,
        J_pitch=J_pitch,
    )

    posture_error = (
        q_posture_ref
        -
        q_candidate
    )

    posture_seed = (
        posture_joint_weights
        *
        posture_error
    )

    delta_q_posture = (
        k_posture
        *
        (
            N_augmented
            @
            posture_seed
        )
    )

    delta_q_posture = _validate_vector(
        delta_q_posture,
        "delta_q_posture",
        (5,),
    )

    augmented_leakage = float(
        np.linalg.norm(
            J_augmented
            @
            delta_q_posture
        )
    )

    return {
        "delta_q_posture":
            delta_q_posture.copy(),

        "N_augmented":
            N_augmented.copy(),

        "J_augmented":
            J_augmented.copy(),

        "posture_error":
            posture_error.copy(),

        "posture_seed":
            posture_seed.copy(),

        "augmented_leakage":
            augmented_leakage,
    }


# ============================================================
# HYBRID NUMERICAL IK
# ============================================================

def solve_numerical_IK_hybrid(
    model,
    data,
    W_p_swing_des,
    swing_foot,

    # Legacy compatibility arguments.
    roll_swing_des=0.0,
    pitch_swing_des=0.0,
    orientation_tolerance=1e-3,

    sigma_threshold=1e-4,
    alpha_newton=1.0,
    eta_gradient=1.0,
    tolerance=1e-4,
    max_iterations=50,

    # Compatibility only; posture is not an active IK task.
    q_posture_ref=None,
    k_posture=DEFAULT_K_POSTURE,

    # Latched swing-start orientation.
    R_swing_ref=None,

    # k_roll is accepted for compatibility but not used.
    k_roll=DEFAULT_K_ROLL,

    # Hierarchical pitch task gain.
    k_pitch=DEFAULT_K_PITCH,

    pitch_tolerance=DEFAULT_PITCH_TOLERANCE,

    pitch_motion_cost=DEFAULT_PITCH_MOTION_COST,

    posture_joint_weights=DEFAULT_POSTURE_JOINT_WEIGHTS,

    # Old weighted-null-space arguments accepted but unused.
    roll_joint_weights=None,
    pitch_joint_weights=None,
):
    """
    Hierarchical swing-leg IK.

    Priority 1
    ----------
        Cartesian position [x, y, z].

    Priority 2
    ----------
        Foot pitch stabilization around the latched
        swing-start orientation.

    There is no tertiary posture task.

    Important
    ---------
    - Roll is not a Cartesian task.
    - Joint posture is not regulated by the IK.
    - Pitch correction uses a weighted minimum-norm solution
      that strongly discourages hip-yaw motion and prefers
      hip-pitch + knee.
    - Ankle remains available for sole counter-rotation.
    - No mj_step() is called here.
    """

    # ========================================================
    # 1. SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R",
        None,
    ):
        raise ValueError(
            "swing_foot chỉ được là 'L', 'R' hoặc None."
        )

    if swing_foot is None:
        diagnostics = {
            "converged": True,
            "position_converged": True,
            "pitch_converged": True,
            "tilt_converged": True,
            "iterations": 0,

            "final_error": None,
            "final_error_norm": 0.0,

            "final_position_error": None,
            "final_position_error_norm": 0.0,

            "final_pitch_error": 0.0,
            "final_pitch_error_abs": 0.0,

            "final_tilt_error": None,
            "final_tilt_error_norm": 0.0,

            "final_orientation_error": None,
            "final_orientation_error_norm": 0.0,

            "orientation_task_active": True,
            "pitch_task_active": True,
            "roll_task_active": False,
            "posture_task_active": False,

            "method_used": None,
            "newton_iterations": 0,
            "gradient_iterations": 0,
            "secondary_only_iterations": 0,

            "sigma_min": None,

            "swing_foot": None,
            "site_name": None,
            "joint_names": None,

            "q_posture_ref": None,
            "R_swing_ref": None,

            "k_posture": float(
                k_posture
            ),
            "k_roll": float(
                k_roll
            ),
            "k_pitch": float(
                k_pitch
            ),

            "pitch_motion_cost":
                np.asarray(
                    pitch_motion_cost,
                    dtype=float,
                ).copy(),

            "posture_joint_weights":
                np.asarray(
                    posture_joint_weights,
                    dtype=float,
                ).copy(),

            "primary_step_norm_last": 0.0,
            "pitch_step_norm_last": 0.0,
            "posture_step_norm_last": 0.0,
            "secondary_step_norm_last": 0.0,

            "pitch_position_leakage_last": 0.0,
            "pitch_residual_last": 0.0,
            "posture_augmented_leakage_last": 0.0,

            "null_space_projection_error_last": 0.0,

            "roll_swing_des": None,
            "pitch_swing_des": None,
        }

        return (
            None,
            diagnostics,
        )

    # ========================================================
    # 2. INPUT VALIDATION
    # ========================================================

    if W_p_swing_des is None:
        raise ValueError(
            "W_p_swing_des không được None khi có swing foot."
        )

    W_p_swing_des = _validate_vector(
        W_p_swing_des,
        "W_p_swing_des",
        (3,),
    )

    # Legacy arguments are still validated for compatibility.
    roll_swing_des = float(
        roll_swing_des
    )

    pitch_swing_des = float(
        pitch_swing_des
    )

    orientation_tolerance = _validate_positive_scalar(
        orientation_tolerance,
        "orientation_tolerance",
    )

    sigma_threshold = _validate_nonnegative_scalar(
        sigma_threshold,
        "sigma_threshold",
    )

    alpha_newton = _validate_positive_scalar(
        alpha_newton,
        "alpha_newton",
    )

    eta_gradient = _validate_positive_scalar(
        eta_gradient,
        "eta_gradient",
    )

    tolerance = _validate_positive_scalar(
        tolerance,
        "tolerance",
    )

    k_posture = _validate_nonnegative_scalar(
        k_posture,
        "k_posture",
    )

    k_roll = _validate_nonnegative_scalar(
        k_roll,
        "k_roll",
    )

    k_pitch = _validate_nonnegative_scalar(
        k_pitch,
        "k_pitch",
    )

    pitch_tolerance = _validate_positive_scalar(
        pitch_tolerance,
        "pitch_tolerance",
    )

    pitch_motion_cost = _validate_positive_weights(
        pitch_motion_cost,
        "pitch_motion_cost",
    )

    posture_joint_weights = _validate_nonnegative_weights(
        posture_joint_weights,
        "posture_joint_weights",
    )

    if not isinstance(
        max_iterations,
        (
            int,
            np.integer,
        ),
    ):
        raise TypeError(
            "max_iterations phải là số nguyên."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations phải > 0."
        )

    # ========================================================
    # 3. LEG INFORMATION
    # ========================================================

    (
        site_id,
        site_name,
        joint_names,
        joint_ids,
        qpos_addresses,
        dof_addresses,
    ) = _get_leg_joint_information(
        model,
        swing_foot,
    )

    # ========================================================
    # 4. SAVE REAL MUJOCO STATE
    # ========================================================

    qpos_original = (
        data.qpos.copy()
    )

    qvel_original = (
        data.qvel.copy()
    )

    mujoco.mj_forward(
        model,
        data,
    )

    # ========================================================
    # 5. INITIAL CANDIDATE + REFERENCES
    # ========================================================

    q_candidate = np.asarray(
        data.qpos[
            qpos_addresses
        ],
        dtype=float,
    ).copy()

    q_candidate = _validate_vector(
        q_candidate,
        "q_candidate",
        (5,),
    )

    if q_posture_ref is None:
        q_posture_ref = (
            q_candidate.copy()
        )
    else:
        q_posture_ref = _validate_vector(
            q_posture_ref,
            "q_posture_ref",
            (5,),
        ).copy()

    if R_swing_ref is None:
        R_swing_ref = _get_site_rotation_matrix(
            data,
            site_id,
        )
    else:
        R_swing_ref = _validate_matrix(
            R_swing_ref,
            "R_swing_ref",
            (
                3,
                3,
            ),
        )

    # ========================================================
    # 6. DIAGNOSTIC STATE
    # ========================================================

    converged = False
    position_converged = False
    pitch_converged = False

    iterations = 0
    newton_iterations = 0
    gradient_iterations = 0
    secondary_only_iterations = 0

    method_used = None
    sigma_min = None

    primary_step_norm_last = 0.0
    pitch_step_norm_last = 0.0
    posture_step_norm_last = 0.0
    secondary_step_norm_last = 0.0

    pitch_position_leakage_last = 0.0
    pitch_residual_last = 0.0
    posture_augmented_leakage_last = 0.0
    null_space_projection_error_last = 0.0

    final_position_error = np.zeros(
        3,
        dtype=float,
    )

    final_pitch_error = 0.0

    final_posture_error = (
        q_posture_ref
        -
        q_candidate
    )

    final_posture_error_norm = float(
        np.linalg.norm(
            final_posture_error
        )
    )

    # ========================================================
    # 7. ITERATIVE HIERARCHICAL IK
    # ========================================================

    try:
        for _ in range(
            max_iterations
        ):
            # ------------------------------------------------
            # Current errors
            # ------------------------------------------------

            position_error = _compute_position_error(
                data=data,
                site_id=site_id,
                W_p_swing_des=W_p_swing_des,
            )

            R_current = _get_site_rotation_matrix(
                data,
                site_id,
            )

            pitch_error = compute_world_pitch_error(
                R_current=R_current,
                R_reference=R_swing_ref,
            )

            position_error_norm = float(
                np.linalg.norm(
                    position_error
                )
            )

            pitch_error_abs = float(
                abs(
                    pitch_error
                )
            )

            position_converged = bool(
                position_error_norm
                <=
                tolerance
            )

            pitch_converged = bool(
                pitch_error_abs
                <=
                pitch_tolerance
            )

            if (
                position_converged
                and
                pitch_converged
            ):
                converged = True
                break

            # ------------------------------------------------
            # Current Jacobians
            # ------------------------------------------------

            (
                J_position,
                J_rotation,
            ) = _compute_leg_jacobians(
                model=model,
                data=data,
                site_id=site_id,
                dof_addresses=dof_addresses,
            )

            J_pitch = (
                J_rotation[
                    1:2,
                    :
                ].copy()
            )

            # ------------------------------------------------
            # TASK 1: Cartesian position
            # ------------------------------------------------

            singular_values = np.linalg.svd(
                J_position,
                compute_uv=False,
            )

            sigma_min = float(
                singular_values[
                    -1
                ]
            )

            if position_converged:
                delta_q_primary = np.zeros(
                    5,
                    dtype=float,
                )

                method_used = (
                    "secondary_only"
                )

                secondary_only_iterations += 1

            elif sigma_min > sigma_threshold:
                delta_q_primary = compute_newton_step(
                    J=J_position,
                    error=position_error,
                    alpha_newton=alpha_newton,
                )

                method_used = (
                    "newton"
                )

                newton_iterations += 1

            else:
                delta_q_primary = (
                    compute_gradient_descent_step(
                        J=J_position,
                        error=position_error,
                        eta_gradient=eta_gradient,
                    )
                )

                method_used = (
                    "gradient_descent"
                )

                gradient_iterations += 1

            # ------------------------------------------------
            # TASK 2: pitch, while preserving position
            # ------------------------------------------------

            if pitch_converged:
                pitch_result = {
                    "delta_q_pitch":
                        np.zeros(
                            5,
                            dtype=float,
                        ),

                    "requested_pitch_step":
                        0.0,

                    "achieved_pitch_step":
                        0.0,

                    "pitch_residual":
                        0.0,

                    "position_leakage":
                        0.0,
                }

            else:
                pitch_result = compute_weighted_pitch_step(
                    J_position=J_position,
                    J_pitch=J_pitch,
                    pitch_error=pitch_error,
                    k_pitch=k_pitch,
                    pitch_motion_cost=pitch_motion_cost,
                )

            delta_q_pitch = pitch_result[
                "delta_q_pitch"
            ]

            # ------------------------------------------------
            # NO POSTURE TASK
            #
            # Active hierarchy:
            #
            #     [x, y, z]  >>  pitch
            #
            # The remaining redundant DOF is not driven toward
            # a latched joint posture.
            # ------------------------------------------------

            delta_q_posture = np.zeros(
                5,
                dtype=float,
            )

            # ------------------------------------------------
            # Hierarchical update
            # ------------------------------------------------

            delta_q = (
                delta_q_primary
                +
                delta_q_pitch
            )

            delta_q = _validate_vector(
                delta_q,
                "delta_q",
                (5,),
            )

            primary_step_norm_last = float(
                np.linalg.norm(
                    delta_q_primary
                )
            )

            pitch_step_norm_last = float(
                np.linalg.norm(
                    delta_q_pitch
                )
            )

            posture_step_norm_last = 0.0

            secondary_step_norm_last = float(
                np.linalg.norm(
                    delta_q_pitch
                )
            )

            pitch_position_leakage_last = float(
                pitch_result[
                    "position_leakage"
                ]
            )

            pitch_residual_last = float(
                pitch_result[
                    "pitch_residual"
                ]
            )

            # Compatibility diagnostics: posture is inactive.
            posture_augmented_leakage_last = 0.0
            null_space_projection_error_last = 0.0

            q_candidate = (
                q_candidate
                +
                delta_q
            )

            q_candidate = _apply_joint_limits(
                model=model,
                joint_ids=joint_ids,
                q_candidate=q_candidate,
            )

            data.qpos[
                qpos_addresses
            ] = q_candidate

            mujoco.mj_forward(
                model,
                data,
            )

            iterations += 1

        # ====================================================
        # 8. FINAL ERRORS
        # ====================================================

        final_position_error = _compute_position_error(
            data=data,
            site_id=site_id,
            W_p_swing_des=W_p_swing_des,
        )

        R_current_final = _get_site_rotation_matrix(
            data,
            site_id,
        )

        final_pitch_error = compute_world_pitch_error(
            R_current=R_current_final,
            R_reference=R_swing_ref,
        )

        final_position_error_norm = float(
            np.linalg.norm(
                final_position_error
            )
        )

        final_pitch_error_abs = float(
            abs(
                final_pitch_error
            )
        )

        position_converged = bool(
            final_position_error_norm
            <=
            tolerance
        )

        pitch_converged = bool(
            final_pitch_error_abs
            <=
            pitch_tolerance
        )

        converged = bool(
            position_converged
            and
            pitch_converged
        )

        q_swing_des = (
            q_candidate.copy()
        )

        final_posture_error = (
            q_posture_ref
            -
            q_swing_des
        )

        final_posture_error_norm = float(
            np.linalg.norm(
                final_posture_error
            )
        )

    finally:
        # ====================================================
        # 9. RESTORE REAL MUJOCO STATE
        # ====================================================

        data.qpos[
            :
        ] = qpos_original

        data.qvel[
            :
        ] = qvel_original

        mujoco.mj_forward(
            model,
            data,
        )

    # ========================================================
    # 10. DIAGNOSTICS
    # ========================================================

    final_tilt_error = np.array(
        [
            0.0,
            final_pitch_error,
        ],
        dtype=float,
    )

    diagnostics = {
        "converged":
            bool(
                converged
            ),

        "position_converged":
            bool(
                position_converged
            ),

        "pitch_converged":
            bool(
                pitch_converged
            ),

        # Compatibility with previous tilt-based version.
        "tilt_converged":
            bool(
                pitch_converged
            ),

        "iterations":
            int(
                iterations
            ),

        "final_error":
            final_position_error.copy(),

        "final_error_norm":
            float(
                np.linalg.norm(
                    final_position_error
                )
            ),

        "final_position_error":
            final_position_error.copy(),

        "final_position_error_norm":
            float(
                np.linalg.norm(
                    final_position_error
                )
            ),

        "final_pitch_error":
            float(
                final_pitch_error
            ),

        "final_pitch_error_abs":
            float(
                abs(
                    final_pitch_error
                )
            ),

        # Compatibility fields.
        "final_tilt_error":
            final_tilt_error.copy(),

        "final_tilt_error_norm":
            float(
                abs(
                    final_pitch_error
                )
            ),

        "final_orientation_error":
            final_tilt_error.copy(),

        "final_orientation_error_norm":
            float(
                abs(
                    final_pitch_error
                )
            ),

        "orientation_task_active":
            True,

        "pitch_task_active":
            True,

        "roll_task_active":
            False,

        "posture_task_active":
            False,

        "method_used":
            method_used,

        "newton_iterations":
            int(
                newton_iterations
            ),

        "gradient_iterations":
            int(
                gradient_iterations
            ),

        "secondary_only_iterations":
            int(
                secondary_only_iterations
            ),

        "sigma_min":
            (
                None
                if sigma_min is None
                else float(
                    sigma_min
                )
            ),

        "swing_foot":
            swing_foot,

        "site_name":
            site_name,

        "joint_names":
            list(
                joint_names
            ),

        "q_posture_ref":
            q_posture_ref.copy(),

        "R_swing_ref":
            R_swing_ref.copy(),

        # Compatibility only; inactive in this solver.
        "k_posture":
            float(
                k_posture
            ),

        "k_roll":
            float(
                k_roll
            ),

        "k_pitch":
            float(
                k_pitch
            ),

        "pitch_motion_cost":
            pitch_motion_cost.copy(),

        # Compatibility only; inactive in this solver.
        "posture_joint_weights":
            posture_joint_weights.copy(),

        "final_posture_error":
            final_posture_error.copy(),

        "final_posture_error_norm":
            float(
                final_posture_error_norm
            ),

        "primary_step_norm_last":
            float(
                primary_step_norm_last
            ),

        "pitch_step_norm_last":
            float(
                pitch_step_norm_last
            ),

        "posture_step_norm_last":
            float(
                posture_step_norm_last
            ),

        "secondary_step_norm_last":
            float(
                secondary_step_norm_last
            ),

        "pitch_position_leakage_last":
            float(
                pitch_position_leakage_last
            ),

        "pitch_residual_last":
            float(
                pitch_residual_last
            ),

        "posture_augmented_leakage_last":
            float(
                posture_augmented_leakage_last
            ),

        "null_space_projection_error_last":
            float(
                null_space_projection_error_last
            ),

        # Legacy desired-orientation arguments are no longer
        # used as Cartesian tasks.
        "roll_swing_des":
            float(
                roll_swing_des
            ),

        "pitch_swing_des":
            float(
                pitch_swing_des
            ),
    }

    return (
        q_swing_des,
        diagnostics,
    )
