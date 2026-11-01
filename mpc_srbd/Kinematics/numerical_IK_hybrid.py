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
    # Site ID
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
    # Joint information
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


        # ----------------------------------------------------
        # qpos address
        #
        # Used to read/write joint angles.
        # ----------------------------------------------------

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )


        qpos_addresses.append(
            qpos_address
        )


        # ----------------------------------------------------
        # DOF address
        #
        # Used to extract the corresponding Jacobian column.
        # ----------------------------------------------------

        dof_address = int(
            model.jnt_dofadr[
                joint_id
            ]
        )


        dof_addresses.append(
            dof_address
        )


    return (
        site_id,
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

        # ----------------------------------------------------
        # Only clamp if this MuJoCo joint is limited.
        # ----------------------------------------------------

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
# COMPUTE LEG POSITION JACOBIAN
# ============================================================

def _compute_leg_position_jacobian(
    model,
    data,
    site_id,
    dof_addresses
):

    # --------------------------------------------------------
    # Full site translational Jacobian:
    #
    #     J_full in R^(3 x nv)
    #
    # This contains:
    #
    #     floating-base DOFs
    #     left-leg DOFs
    #     right-leg DOFs
    #
    # We later select only the 5 columns belonging to the
    # current swing leg.
    # --------------------------------------------------------

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
    # Extract only the 5 swing-leg joint columns.
    #
    #     J_leg in R^(3 x 5)
    # --------------------------------------------------------

    J_leg = (
        J_position_full[
            :,
            dof_addresses
        ].copy()
    )


    if J_leg.shape != (
        3,
        5
    ):

        raise RuntimeError(
            "J_leg phải có shape (3, 5), "
            f"nhưng nhận được {J_leg.shape}."
        )


    if not np.all(
        np.isfinite(
            J_leg
        )
    ):

        raise RuntimeError(
            "J_leg chứa NaN hoặc Inf."
        )


    return J_leg


# ============================================================
# HYBRID NUMERICAL IK
# ============================================================

def solve_numerical_IK_hybrid(
    model,
    data,
    W_p_swing_des,
    swing_foot,
    sigma_threshold=1e-4,
    alpha_newton=1.0,
    eta_gradient=1.0,
    tolerance=1e-4,
    max_iterations=50
):
    """
    Solve swing-leg position inverse kinematics using a hybrid
    numerical method.

    Method selection
    ----------------

    At every IK iteration, compute:

        sigma_min = minimum singular value of J

    If:

        sigma_min > sigma_threshold

    use pseudo-Newton:

        delta_q
            =
        alpha_newton
        *
        pinv(J)
        *
        error

    Otherwise use Jacobian-transpose Gradient Descent:

        delta_q
            =
        eta_gradient
        *
        J^T
        *
        error


    Cartesian error
    ---------------

        error
            =
        W_p_swing_des
            -
        W_p_i6


    Position task
    -------------

    Only the 3D physical foot position is controlled:

        W_p_i6 =
        [
            x_i6,
            y_i6,
            z_i6
        ]

    Therefore:

        J_leg in R^(3x5)


    Important
    ---------

    During the iterative solution, this function temporarily
    changes the swing-leg joint values inside MuJoCo data in
    order to evaluate forward kinematics.

    Before returning, the original MuJoCo simulation state is
    restored.

    Therefore this IK function does NOT directly move the
    simulated robot.

    It only returns:

        q_swing_des


    Parameters
    ----------
    model : mujoco.MjModel

        MuJoCo model.


    data : mujoco.MjData

        Current MuJoCo simulation data.


    W_p_swing_des : array_like, shape (3,)

        Desired physical swing-foot position in World frame:

            {}^W p_i6^des


    swing_foot : str or None

        Current swing foot:

            "L"
            "R"

        During Double Support:

            None


    sigma_threshold : float, optional

        Minimum-singular-value threshold used to switch
        between pseudo-Newton and Gradient Descent.

        Default:

            1e-4


    alpha_newton : float, optional

        Step scaling used by pseudo-Newton.

        Default:

            1.0


    eta_gradient : float, optional

        Step size used by Gradient Descent.

        Default:

            1.0


    tolerance : float, optional

        Cartesian position-error tolerance [m].

        Default:

            1e-4 m


    max_iterations : int, optional

        Maximum number of IK iterations.

        Default:

            50


    Returns
    -------
    q_swing_des : np.ndarray, shape (5,), or None

        Desired joint positions for the current swing leg:

            [
                q_hip_yaw,
                q_hip_roll,
                q_hip_pitch,
                q_knee,
                q_ankle
            ]

        Unit:

            [rad]

        Returns None during Double Support.


    diagnostics : dict

        Diagnostic information:

            converged
            iterations
            final_error
            final_error_norm
            method_used
            newton_iterations
            gradient_iterations
            sigma_min
            swing_foot
            joint_names
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
    #
    # No swing-foot IK is required.
    # ========================================================

    if swing_foot is None:

        diagnostics = {

            "converged": True,

            "iterations": 0,

            "final_error": None,

            "final_error_norm": 0.0,

            "method_used": None,

            "newton_iterations": 0,

            "gradient_iterations": 0,

            "sigma_min": None,

            "swing_foot": None,

            "joint_names": None

        }


        return None, diagnostics


    # ========================================================
    # 3. VALIDATE TARGET POSITION
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
    # 4. VALIDATE PARAMETERS
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
    # 5. GET CURRENT SWING-LEG INFORMATION
    # ========================================================

    (
        site_id,
        joint_names,
        joint_ids,
        qpos_addresses,
        dof_addresses

    ) = _get_leg_joint_information(
        model,
        swing_foot
    )


    # ========================================================
    # 6. SAVE ORIGINAL MUJOCO STATE
    #
    # IMPORTANT:
    #
    # IK must NOT directly modify the actual simulation state.
    #
    # Therefore save:
    #
    #     qpos
    #     qvel
    #
    # before starting the iterative numerical solution.
    # ========================================================

    qpos_original = (
        data.qpos.copy()
    )


    qvel_original = (
        data.qvel.copy()
    )


    # ========================================================
    # 7. INITIAL GUESS
    #
    # Use current swing-leg joint configuration:
    #
    #     q^(0) = q_current
    # ========================================================

    q_candidate = (
        data.qpos[
            qpos_addresses
        ].copy()
    )


    # ========================================================
    # 8. DIAGNOSTIC INITIALIZATION
    # ========================================================

    converged = False

    iterations = 0

    newton_iterations = 0

    gradient_iterations = 0

    method_used = None

    sigma_min = None

    final_error = None

    final_error_norm = np.inf


    # ========================================================
    # 9. ITERATIVE IK
    # ========================================================

    try:

        # ----------------------------------------------------
        # Ensure initial MuJoCo kinematics are up to date.
        # ----------------------------------------------------

        mujoco.mj_forward(
            model,
            data
        )


        for _ in range(
            max_iterations
        ):

            # =================================================
            # 9.1 CURRENT PHYSICAL FOOT POSITION
            #
            #     {}^W p_i6
            # =================================================

            W_p_current = (
                data.site_xpos[
                    site_id
                ].copy()
            )


            # =================================================
            # 9.2 CARTESIAN POSITION ERROR
            #
            #     e
            #
            # =
            #
            #     W_p_swing_des
            #
            # -
            #
            #     W_p_current
            # =================================================

            error = (
                W_p_swing_des
                -
                W_p_current
            )


            error_norm = float(
                np.linalg.norm(
                    error
                )
            )


            # =================================================
            # 9.3 CONVERGENCE CHECK
            # =================================================

            if error_norm <= tolerance:

                converged = True

                final_error = (
                    error.copy()
                )

                final_error_norm = (
                    error_norm
                )

                break


            # =================================================
            # 9.4 POSITION JACOBIAN
            #
            #     J_leg in R^(3x5)
            # =================================================

            J_leg = _compute_leg_position_jacobian(
                model=model,
                data=data,
                site_id=site_id,
                dof_addresses=dof_addresses
            )


            # =================================================
            # 9.5 SINGULAR VALUES
            #
            #     J = U Sigma V^T
            #
            # Use:
            #
            #     sigma_min
            #
            # to select numerical IK method.
            # =================================================

            singular_values = np.linalg.svd(
                J_leg,
                compute_uv=False
            )


            sigma_min = float(
                singular_values[
                    -1
                ]
            )


            # =================================================
            # 9.6 METHOD SELECTION
            # =================================================

            if sigma_min > sigma_threshold:

                # ---------------------------------------------
                # PSEUDO-NEWTON
                # ---------------------------------------------

                delta_q = compute_newton_step(
                    J=J_leg,
                    error=error,
                    alpha_newton=alpha_newton
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
                        J=J_leg,
                        error=error,
                        eta_gradient=eta_gradient
                    )
                )


                method_used = (
                    "gradient_descent"
                )


                gradient_iterations += 1


            # =================================================
            # 9.7 UPDATE JOINT CONFIGURATION
            #
            #     q^(k+1)
            #
            # =
            #
            #     q^(k)
            #
            # +
            #
            #     delta_q
            # =================================================

            q_candidate = (
                q_candidate
                +
                delta_q
            )


            # =================================================
            # 9.8 APPLY JOINT LIMITS
            # =================================================

            q_candidate = _apply_joint_limits(
                model=model,
                joint_ids=joint_ids,
                q_candidate=q_candidate
            )


            # =================================================
            # 9.9 TEMPORARILY WRITE CANDIDATE TO MUJOCO
            #
            # This is only used to evaluate forward kinematics.
            # =================================================

            data.qpos[
                qpos_addresses
            ] = q_candidate


            # =================================================
            # 9.10 UPDATE FORWARD KINEMATICS
            # =================================================

            mujoco.mj_forward(
                model,
                data
            )


            iterations += 1


        # ====================================================
        # 10. FINAL ERROR AFTER LAST ITERATION
        #
        # This is also evaluated if max_iterations is reached.
        # ====================================================

        W_p_final = (
            data.site_xpos[
                site_id
            ].copy()
        )


        final_error = (
            W_p_swing_des
            -
            W_p_final
        )


        final_error_norm = float(
            np.linalg.norm(
                final_error
            )
        )


        if final_error_norm <= tolerance:

            converged = True


        # ====================================================
        # 11. FINAL JOINT SOLUTION
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
    # 12. RESTORE REAL MUJOCO SIMULATION STATE
    #
    # This block is ALWAYS executed, even if the IK solver
    # raises an exception.
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
    # 13. DIAGNOSTICS
    # ========================================================

    diagnostics = {

        "converged": bool(
            converged
        ),

        "iterations": int(
            iterations
        ),

        "final_error": (
            final_error.copy()
        ),

        "final_error_norm": float(
            final_error_norm
        ),

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

        "swing_foot": (
            swing_foot
        ),

        "joint_names": (
            list(
                joint_names
            )
        )

    }


    # ========================================================
    # 14. RETURN
    # ========================================================

    return (
        q_swing_des,
        diagnostics
    )