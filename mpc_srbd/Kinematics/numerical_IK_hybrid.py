import mujoco
import numpy as np

try:
    from .numerical_IK_newton import compute_newton_step
    from .numerical_IK_gradient_descent import compute_gradient_descent_step
    from .numerical_IK_params import HIP_YAW_FIXED
except ImportError:
    from numerical_IK_newton import compute_newton_step
    from numerical_IK_gradient_descent import compute_gradient_descent_step
    from numerical_IK_params import HIP_YAW_FIXED


# ============================================================
# ANKLE-CENTER IK CONFIGURATION
#
# Equality constraint:
#     q_hip_yaw = HIP_YAW_FIXED
#
# Numerically solved variables:
#     [hip_roll, hip_pitch, knee]
#
# Returned joint vector:
#     [hip_yaw, hip_roll, hip_pitch, knee]
#
# The ankle joint is intentionally NOT part of this IK.
# ============================================================

LEG_CONFIG = {
    "L": {
        "site_name": "L5_ankle",
        "hip_yaw_name": "L_hip_yaw",
        "solve_joint_names": [
            "L_hip_roll",
            "L_hip_pitch",
            "L_knee",
        ],
    },
    "R": {
        "site_name": "R5_ankle",
        "hip_yaw_name": "R_hip_yaw",
        "solve_joint_names": [
            "R_hip_roll",
            "R_hip_pitch",
            "R_knee",
        ],
    },
}

# Temporary compatibility names for the current low-level controller.
DEFAULT_K_POSTURE = 0.0
DEFAULT_K_ROLL = 0.0
DEFAULT_K_PITCH = 0.0


def _validate_vector(value, name, expected_shape):
    value = np.asarray(value, dtype=float).reshape(-1)

    if value.shape != expected_shape:
        raise ValueError(
            f"{name} phải có shape {expected_shape}, "
            f"nhưng nhận được {value.shape}."
        )

    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} chứa NaN hoặc Inf.")

    return value


def _validate_nonnegative_scalar(value, name):
    value = float(value)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} phải hữu hạn và >= 0.")

    return value


def _validate_positive_scalar(value, name):
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} phải hữu hạn và > 0.")

    return value


def _get_joint_id(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise ValueError(f"Không tìm thấy joint '{joint_name}'.")

    return int(joint_id)


def _get_leg_information(model, swing_foot):
    config = LEG_CONFIG[swing_foot]

    site_name = config["site_name"]
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id < 0:
        raise ValueError(f"Không tìm thấy site '{site_name}'.")

    hip_yaw_name = config["hip_yaw_name"]
    hip_yaw_id = _get_joint_id(model, hip_yaw_name)
    hip_yaw_qpos_address = int(model.jnt_qposadr[hip_yaw_id])

    solve_joint_names = list(config["solve_joint_names"])
    solve_joint_ids = []
    solve_qpos_addresses = []
    solve_dof_addresses = []

    for joint_name in solve_joint_names:
        joint_id = _get_joint_id(model, joint_name)
        solve_joint_ids.append(joint_id)
        solve_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        solve_dof_addresses.append(int(model.jnt_dofadr[joint_id]))

    solve_joint_ids = np.asarray(solve_joint_ids, dtype=int)
    solve_qpos_addresses = np.asarray(solve_qpos_addresses, dtype=int)
    solve_dof_addresses = np.asarray(solve_dof_addresses, dtype=int)

    if solve_joint_ids.shape != (3,):
        raise RuntimeError("solve_joint_ids phải có shape (3,).")

    return {
        "site_id": int(site_id),
        "site_name": str(site_name),
        "hip_yaw_name": str(hip_yaw_name),
        "hip_yaw_id": int(hip_yaw_id),
        "hip_yaw_qpos_address": hip_yaw_qpos_address,
        "solve_joint_names": solve_joint_names,
        "solve_joint_ids": solve_joint_ids,
        "solve_qpos_addresses": solve_qpos_addresses,
        "solve_dof_addresses": solve_dof_addresses,
    }


def _apply_joint_limits(model, joint_ids, q_candidate):
    q_candidate = _validate_vector(
        q_candidate,
        "q_candidate",
        (3,),
    )

    q_limited = q_candidate.copy()

    for i, joint_id in enumerate(joint_ids):
        if model.jnt_limited[joint_id]:
            q_min = float(model.jnt_range[joint_id, 0])
            q_max = float(model.jnt_range[joint_id, 1])
            q_limited[i] = np.clip(q_limited[i], q_min, q_max)

    return q_limited


def _compute_position_error(data, site_id, W_p_ankle_des):
    W_p_ankle_des = _validate_vector(
        W_p_ankle_des,
        "W_p_ankle_des",
        (3,),
    )

    W_p_ankle = np.asarray(
        data.site_xpos[site_id],
        dtype=float,
    ).reshape(3)

    if not np.all(np.isfinite(W_p_ankle)):
        raise RuntimeError("W_p_ankle chứa NaN hoặc Inf.")

    return W_p_ankle_des - W_p_ankle


def _compute_position_jacobian(
    model,
    data,
    site_id,
    solve_dof_addresses,
):
    J_position_full = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    J_rotation_full = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    mujoco.mj_jacSite(
        model,
        data,
        J_position_full,
        J_rotation_full,
        site_id,
    )

    J_position = J_position_full[:, solve_dof_addresses].copy()

    if J_position.shape != (3, 3):
        raise RuntimeError(
            "J_position phải có shape (3, 3), "
            f"nhưng nhận được {J_position.shape}."
        )

    if not np.all(np.isfinite(J_position)):
        raise RuntimeError("J_position chứa NaN hoặc Inf.")

    return J_position


def solve_numerical_IK_hybrid(
    model,
    data,
    W_p_ankle_des=None,
    swing_foot=None,
    sigma_threshold=1e-4,
    alpha_newton=1.0,
    eta_gradient=1.0,
    tolerance=1e-4,
    max_iterations=50,
    hip_yaw_fixed=HIP_YAW_FIXED,

    # Legacy compatibility arguments. They are intentionally
    # inactive in the new ankle-center IK.
    W_p_swing_des=None,
    roll_swing_des=0.0,
    pitch_swing_des=0.0,
    orientation_tolerance=1e-3,
    q_posture_ref=None,
    k_posture=DEFAULT_K_POSTURE,
    R_swing_ref=None,
    k_roll=DEFAULT_K_ROLL,
    k_pitch=DEFAULT_K_PITCH,
    pitch_tolerance=None,
    pitch_motion_cost=None,
    posture_joint_weights=None,
    roll_joint_weights=None,
    pitch_joint_weights=None,
):
    """
    Hybrid numerical IK for ankle-center Cartesian position.

    Constraint:
        q_hip_yaw = hip_yaw_fixed

    Solved variables:
        [hip_roll, hip_pitch, knee]

    Output:
        [hip_yaw, hip_roll, hip_pitch, knee]

    The floating-base pose is kept at the current MuJoCo state.
    No mj_step() is called. qpos/qvel are restored before return.
    """

    if swing_foot not in ("L", "R", None):
        raise ValueError(
            "swing_foot chỉ được là 'L', 'R' hoặc None."
        )

    if swing_foot is None:
        return None, {
            "converged": True,
            "position_converged": True,
            "iterations": 0,
            "newton_iterations": 0,
            "gradient_iterations": 0,
            "method_used": None,
            "sigma_min": None,
            "final_position_error": None,
            "final_position_error_norm": 0.0,
            "swing_foot": None,
            "site_name": None,
            "solve_joint_names": None,
            "hip_yaw_fixed": float(hip_yaw_fixed),
        }

    # Temporary alias for the old caller.
    if W_p_ankle_des is None:
        W_p_ankle_des = W_p_swing_des

    if W_p_ankle_des is None:
        raise ValueError(
            "W_p_ankle_des không được None khi có swing foot."
        )

    W_p_ankle_des = _validate_vector(
        W_p_ankle_des,
        "W_p_ankle_des",
        (3,),
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

    hip_yaw_fixed = float(hip_yaw_fixed)

    if not np.isfinite(hip_yaw_fixed):
        raise ValueError("hip_yaw_fixed phải hữu hạn.")

    if not isinstance(max_iterations, (int, np.integer)):
        raise TypeError("max_iterations phải là số nguyên.")

    if max_iterations <= 0:
        raise ValueError("max_iterations phải > 0.")

    leg = _get_leg_information(model, swing_foot)

    qpos_original = data.qpos.copy()
    qvel_original = data.qvel.copy()

    converged = False
    iterations = 0
    newton_iterations = 0
    gradient_iterations = 0
    method_used = None
    sigma_min = None

    final_position_error = np.zeros(3, dtype=float)
    final_position_error_norm = np.inf

    q_candidate = None
    q_ik_des = None

    try:
        # Enforce q_hip_yaw = constant before and during IK.
        data.qpos[leg["hip_yaw_qpos_address"]] = hip_yaw_fixed
        mujoco.mj_forward(model, data)

        q_candidate = np.asarray(
            data.qpos[leg["solve_qpos_addresses"]],
            dtype=float,
        ).copy()

        q_candidate = _validate_vector(
            q_candidate,
            "q_candidate",
            (3,),
        )

        q_candidate = _apply_joint_limits(
            model=model,
            joint_ids=leg["solve_joint_ids"],
            q_candidate=q_candidate,
        )

        for _ in range(max_iterations):
            data.qpos[leg["hip_yaw_qpos_address"]] = hip_yaw_fixed
            data.qpos[leg["solve_qpos_addresses"]] = q_candidate
            mujoco.mj_forward(model, data)

            position_error = _compute_position_error(
                data=data,
                site_id=leg["site_id"],
                W_p_ankle_des=W_p_ankle_des,
            )

            position_error_norm = float(
                np.linalg.norm(position_error)
            )

            if position_error_norm <= tolerance:
                converged = True
                break

            J_position = _compute_position_jacobian(
                model=model,
                data=data,
                site_id=leg["site_id"],
                solve_dof_addresses=leg["solve_dof_addresses"],
            )

            singular_values = np.linalg.svd(
                J_position,
                compute_uv=False,
            )
            sigma_min = float(singular_values[-1])

            if sigma_min > sigma_threshold:
                delta_q = compute_newton_step(
                    J=J_position,
                    error=position_error,
                    alpha_newton=alpha_newton,
                )
                method_used = "newton"
                newton_iterations += 1
            else:
                delta_q = compute_gradient_descent_step(
                    J=J_position,
                    error=position_error,
                    eta_gradient=eta_gradient,
                )
                method_used = "gradient_descent"
                gradient_iterations += 1

            q_candidate = q_candidate + delta_q

            q_candidate = _apply_joint_limits(
                model=model,
                joint_ids=leg["solve_joint_ids"],
                q_candidate=q_candidate,
            )

            iterations += 1

        # Final evaluation under the same hip-yaw constraint.
        data.qpos[leg["hip_yaw_qpos_address"]] = hip_yaw_fixed
        data.qpos[leg["solve_qpos_addresses"]] = q_candidate
        mujoco.mj_forward(model, data)

        final_position_error = _compute_position_error(
            data=data,
            site_id=leg["site_id"],
            W_p_ankle_des=W_p_ankle_des,
        )

        final_position_error_norm = float(
            np.linalg.norm(final_position_error)
        )

        converged = bool(
            final_position_error_norm <= tolerance
        )

        J_position_final = _compute_position_jacobian(
            model=model,
            data=data,
            site_id=leg["site_id"],
            solve_dof_addresses=leg["solve_dof_addresses"],
        )

        sigma_min = float(
            np.linalg.svd(
                J_position_final,
                compute_uv=False,
            )[-1]
        )

        q_ik_des = np.array(
            [
                hip_yaw_fixed,
                q_candidate[0],
                q_candidate[1],
                q_candidate[2],
            ],
            dtype=float,
        )

        q_ik_des = _validate_vector(
            q_ik_des,
            "q_ik_des",
            (4,),
        )

    finally:
        data.qpos[:] = qpos_original
        data.qvel[:] = qvel_original
        mujoco.mj_forward(model, data)

    diagnostics = {
        "converged": bool(converged),
        "position_converged": bool(converged),
        "iterations": int(iterations),
        "newton_iterations": int(newton_iterations),
        "gradient_iterations": int(gradient_iterations),
        "method_used": method_used,
        "sigma_min": sigma_min,
        "final_position_error": final_position_error.copy(),
        "final_position_error_norm": float(final_position_error_norm),
        "swing_foot": str(swing_foot),
        "site_name": leg["site_name"],
        "solve_joint_names": list(leg["solve_joint_names"]),
        "hip_yaw_name": leg["hip_yaw_name"],
        "hip_yaw_fixed": float(hip_yaw_fixed),
        "output_joint_order": [
            "hip_yaw",
            "hip_roll",
            "hip_pitch",
            "knee",
        ],
    }

    return q_ik_des, diagnostics
