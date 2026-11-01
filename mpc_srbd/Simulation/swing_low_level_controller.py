from dataclasses import dataclass

import mujoco
import numpy as np


# ============================================================
# FOOTHOLD PLANNER
# ============================================================

from Foothold_Planner.raibert_heuristic import (
    compute_raibert_foothold
)


# ============================================================
# KINEMATICS
# ============================================================

from Kinematics.numerical_IK_hybrid import (
    solve_numerical_IK_hybrid
)

from Kinematics.numerical_IK_params import (
    SIGMA_THRESHOLD,
    ALPHA_NEWTON,
    ETA_GRADIENT,
    IK_TOLERANCE,
    IK_MAX_ITERATIONS
)


# ============================================================
# ACTUATOR
# ============================================================

from Actuator.swing_joint_PD import (
    compute_swing_joint_PD
)

from Actuator.swing_joint_PD_params import (
    KP_SWING,
    KD_SWING
)


# ============================================================
# SWING-FOOT ORIENTATION COMMAND
#
# Numerical IK controls the 5D task:
#
#     [
#         x
#         y
#         z
#         roll
#         pitch
#     ]
#
# L6_IK / R6_IK are World-aligned at the home configuration.
#
# Therefore, for flat-ground walking:
#
#     roll_des  = 0
#     pitch_des = 0
#
# Yaw is deliberately not controlled by the numerical IK.
# ============================================================

SWING_ROLL_DES = 0.0

SWING_PITCH_DES = 0.0


# ============================================================
# IK ORIENTATION CONVERGENCE TOLERANCE
# ============================================================

IK_ORIENTATION_TOLERANCE = 1e-3


# ============================================================
# SWING LEG JOINT CONFIGURATION
#
# Joint order:
#
#     0. hip yaw
#     1. hip roll
#     2. hip pitch
#     3. knee
#     4. ankle
# ============================================================

SWING_LEG_JOINT_NAMES = {

    "L": (
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee",
        "L_ankle"
    ),

    "R": (
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee",
        "R_ankle"
    )

}


# ============================================================
# RESULT
# ============================================================

@dataclass
class SwingLowLevelResult:
    """
    Output of one swing-leg low-level control cycle.
    """

    tau_swing: np.ndarray

    W_p_foothold: np.ndarray

    W_p_swing_des: np.ndarray

    q_swing_des: np.ndarray

    q_swing: np.ndarray

    q_dot_swing: np.ndarray

    ik_diagnostics: dict

    swing_foot: str

    phase_next: float


# ============================================================
# GET CURRENT SWING-LEG JOINT STATE
# ============================================================

def _get_swing_joint_state(
    model,
    data,
    swing_foot
):
    """
    Read current joint positions and velocities of one leg.

    Returns
    -------
    q_swing : np.ndarray, shape (5,)

        Current swing-leg joint positions [rad].

    q_dot_swing : np.ndarray, shape (5,)

        Current swing-leg joint velocities [rad/s].
    """


    # ========================================================
    # 1. VALIDATE SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R"
    ):

        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )


    # ========================================================
    # 2. JOINT NAMES
    # ========================================================

    joint_names = (
        SWING_LEG_JOINT_NAMES[
            swing_foot
        ]
    )


    # ========================================================
    # 3. READ JOINT STATE
    # ========================================================

    q_swing = np.zeros(
        5,
        dtype=float
    )


    q_dot_swing = np.zeros(
        5,
        dtype=float
    )


    for i, joint_name in enumerate(
        joint_names
    ):

        # ----------------------------------------------------
        # Joint ID
        # ----------------------------------------------------

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )


        if joint_id < 0:

            raise ValueError(
                f"Không tìm thấy joint "
                f"'{joint_name}' trong MuJoCo model."
            )


        # ----------------------------------------------------
        # qpos address
        # ----------------------------------------------------

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )


        # ----------------------------------------------------
        # qvel / DOF address
        # ----------------------------------------------------

        dof_address = int(
            model.jnt_dofadr[
                joint_id
            ]
        )


        # ----------------------------------------------------
        # Current joint position
        # ----------------------------------------------------

        q_swing[
            i
        ] = data.qpos[
            qpos_address
        ]


        # ----------------------------------------------------
        # Current joint velocity
        # ----------------------------------------------------

        q_dot_swing[
            i
        ] = data.qvel[
            dof_address
        ]


    # ========================================================
    # 4. OUTPUT VALIDATION
    # ========================================================

    if q_swing.shape != (
        5,
    ):

        raise RuntimeError(
            "q_swing phải có shape (5,)."
        )


    if q_dot_swing.shape != (
        5,
    ):

        raise RuntimeError(
            "q_dot_swing phải có shape (5,)."
        )


    if not np.all(
        np.isfinite(
            q_swing
        )
    ):

        raise RuntimeError(
            "q_swing chứa NaN hoặc Inf."
        )


    if not np.all(
        np.isfinite(
            q_dot_swing
        )
    ):

        raise RuntimeError(
            "q_dot_swing chứa NaN hoặc Inf."
        )


    # ========================================================
    # 5. RETURN
    # ========================================================

    return (
        q_swing,
        q_dot_swing
    )


# ============================================================
# SWING LOW-LEVEL CONTROLLER
# ============================================================

def compute_swing_low_level_control(
    model,
    data,
    swing_trajectory,
    swing_foot,
    delta_T,
    W_p_CoM,
    W_R_B,
    B_p_L6_offset,
    B_p_R6_offset,
    W_p_L6,
    W_p_R6,
    W_p_dot_CoM,
    W_p_dot_CoM_ref,
    T_stance,
    k_d,
    ground_height=0.0
):
    """
    Compute the low-level joint torque command for the current
    swing leg.

    Control pipeline
    ----------------

        HECTOR-style Raibert foothold planner

            -> W_p_foothold


        Cubic Bezier trajectory

            -> W_p_swing_des


        5D Hybrid numerical IK

            [
                x
                y
                z
                roll
                pitch
            ]

            -> q_swing_des


        Joint-space PD

            -> tau_swing


    ==========================================================
    HECTOR-STYLE RAIBERT
    ==========================================================

    For swing foot i:

        W_p_foothold,i
            =
        W_p_CoM
            +
        W_R_B @ B_p_offset,i
            +
        W_p_dot_CoM * delta_T
            +
        1/2 * W_p_dot_CoM * T_stance
            +
        k_d
        (
            W_p_dot_CoM
            -
            W_p_dot_CoM_ref
        )


    where:

        delta_T

    is the remaining swing time.


    Only x-y are planned using the Raibert heuristic.

    The foothold z coordinate is assigned separately using:

        ground_height


    ==========================================================
    IMPORTANT
    ==========================================================

    This function is only used when:

        swing_foot = "L"

    or:

        swing_foot = "R"


    Double Support is handled outside this function by the
    simulation run file.


    ==========================================================
    FOOTHOLD PLANNING
    ==========================================================

    The HECTOR-style Raibert planner uses:

        W_p_CoM

        W_R_B

        B_p_L6_offset

        B_p_R6_offset

        W_p_dot_CoM

        W_p_dot_CoM_ref

        delta_T

        T_stance

    to compute the desired touchdown foothold.


    The Bezier trajectory still starts from the actual physical:

        L6
        R6

    positions and ends at the Raibert foothold.


    Since:

        L6_IK
        R6_IK

    are colocated with physical:

        L6
        R6

    the desired Bezier position can still be passed directly
    to the existing 5D numerical IK.


    ==========================================================
    FOOT ORIENTATION
    ==========================================================

    The numerical IK internally uses:

        L6_IK
        R6_IK

    for both position and orientation.


    For flat-ground walking:

        roll_swing_des  = 0

        pitch_swing_des = 0


    Yaw is deliberately not controlled.
    """


    # ========================================================
    # 1. VALIDATE SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R"
    ):

        raise ValueError(
            "compute_swing_low_level_control chỉ được gọi "
            "khi swing_foot là 'L' hoặc 'R'."
        )


    # ========================================================
    # 2. VALIDATE TRAJECTORY GENERATOR
    # ========================================================

    if swing_trajectory is None:

        raise ValueError(
            "swing_trajectory không được là None."
        )


    if not hasattr(
        swing_trajectory,
        "update"
    ):

        raise TypeError(
            "swing_trajectory phải có hàm update()."
        )


    # ========================================================
    # 3. HECTOR-STYLE RAIBERT FOOTHOLD
    #
    # For swing foot i:
    #
    #     W_p_foothold,i
    #
    # =
    #
    #     W_p_CoM
    #
    #     +
    #
    #     W_R_B @ B_p_offset,i
    #
    #     +
    #
    #     W_p_dot_CoM * delta_T
    #
    #     +
    #
    #     1/2 * W_p_dot_CoM * T_stance
    #
    #     +
    #
    #     k_d
    #     (
    #         W_p_dot_CoM
    #         -
    #         W_p_dot_CoM_ref
    #     )
    #
    #
    # Only x-y are planned by the heuristic.
    #
    # z touchdown is assigned separately by:
    #
    #     ground_height
    #
    # Here:
    #
    #     delta_T
    #
    # is the remaining swing time.
    # ========================================================

    W_p_foothold = compute_raibert_foothold(

        W_p_CoM=W_p_CoM,

        W_R_B=W_R_B,

        B_p_L6_offset=B_p_L6_offset,

        B_p_R6_offset=B_p_R6_offset,

        W_p_dot_CoM=W_p_dot_CoM,

        W_p_dot_CoM_ref=(
            W_p_dot_CoM_ref
        ),

        swing_foot=swing_foot,

        delta_T=delta_T,

        T_stance=T_stance,

        k_d=k_d,

        ground_height=ground_height
    )


    if W_p_foothold is None:

        raise RuntimeError(
            "Raibert planner trả về None "
            "trong khi đang có swing foot."
        )


    # ========================================================
    # 4. CUBIC BEZIER SWING TRAJECTORY
    #
    # Input:
    #
    #     actual L6
    #     actual R6
    #     current foothold
    #     swing foot
    #
    #
    # Output:
    #
    #     W_p_swing_des
    #
    # =
    #
    #     desired swing-foot position
    #     expressed in World frame
    #
    #
    # Because:
    #
    #     p_L6_IK = p_L6
    #
    # and:
    #
    #     p_R6_IK = p_R6
    #
    # this position target can be passed directly to the 5D IK.
    # ========================================================

    W_p_swing_des = (
        swing_trajectory.update(

            W_p_L6=W_p_L6,

            W_p_R6=W_p_R6,

            W_p_foothold=W_p_foothold,

            swing_foot=swing_foot
        )
    )


    if W_p_swing_des is None:

        raise RuntimeError(
            "Bezier trajectory trả về None "
            "trong khi đang có swing foot."
        )


    W_p_swing_des = np.asarray(
        W_p_swing_des,
        dtype=float
    ).reshape(-1)


    if W_p_swing_des.shape != (
        3,
    ):

        raise RuntimeError(
            "W_p_swing_des phải có shape (3,), "
            f"nhưng nhận được {W_p_swing_des.shape}."
        )


    if not np.all(
        np.isfinite(
            W_p_swing_des
        )
    ):

        raise RuntimeError(
            "W_p_swing_des chứa NaN hoặc Inf."
        )


    # ========================================================
    # 5. READ CURRENT SWING-LEG JOINT STATE
    #
    # These values are used by the joint-space PD controller:
    #
    #     q_swing
    #
    #     q_dot_swing
    # ========================================================

    (
        q_swing,
        q_dot_swing

    ) = _get_swing_joint_state(
        model,
        data,
        swing_foot
    )


    # ========================================================
    # 6. 5D HYBRID NUMERICAL IK
    #
    # Desired task:
    #
    #     [
    #         W_p_swing_des_x
    #         W_p_swing_des_y
    #         W_p_swing_des_z
    #         SWING_ROLL_DES
    #         SWING_PITCH_DES
    #     ]
    #
    #
    # Flat ground:
    #
    #     SWING_ROLL_DES  = 0
    #
    #     SWING_PITCH_DES = 0
    #
    #
    # The Hybrid solver internally uses:
    #
    #     L6_IK
    #
    # or:
    #
    #     R6_IK
    #
    #
    # Method selection:
    #
    #     sigma_min > SIGMA_THRESHOLD
    #
    #         -> pseudo-Newton
    #
    # otherwise:
    #
    #         -> Gradient Descent
    # ========================================================

    (
        q_swing_des,
        ik_diagnostics

    ) = solve_numerical_IK_hybrid(

        model=model,

        data=data,

        W_p_swing_des=W_p_swing_des,

        swing_foot=swing_foot,

        roll_swing_des=SWING_ROLL_DES,

        pitch_swing_des=SWING_PITCH_DES,

        sigma_threshold=SIGMA_THRESHOLD,

        alpha_newton=ALPHA_NEWTON,

        eta_gradient=ETA_GRADIENT,

        tolerance=IK_TOLERANCE,

        orientation_tolerance=(
            IK_ORIENTATION_TOLERANCE
        ),

        max_iterations=IK_MAX_ITERATIONS
    )


    if q_swing_des is None:

        raise RuntimeError(
            "Hybrid IK trả về q_swing_des = None "
            "trong khi đang có swing foot."
        )


    q_swing_des = np.asarray(
        q_swing_des,
        dtype=float
    ).reshape(-1)


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
    # 7. SWING JOINT PD
    #
    #     tau_swing
    #
    # =
    #
    #     KP_SWING
    #
    #     (
    #         q_swing_des
    #         -
    #         q_swing
    #     )
    #
    #     -
    #
    #     KD_SWING
    #
    #     q_dot_swing
    #
    #
    # Desired joint velocity:
    #
    #     q_dot_des = 0
    # ========================================================

    tau_swing = compute_swing_joint_PD(

        q_des=q_swing_des,

        q=q_swing,

        q_dot=q_dot_swing,

        Kp=KP_SWING,

        Kd=KD_SWING
    )


    # ========================================================
    # 8. OUTPUT VALIDATION
    # ========================================================

    tau_swing = np.asarray(
        tau_swing,
        dtype=float
    ).reshape(-1)


    if tau_swing.shape != (
        5,
    ):

        raise RuntimeError(
            "tau_swing phải có shape (5,), "
            f"nhưng nhận được {tau_swing.shape}."
        )


    if not np.all(
        np.isfinite(
            tau_swing
        )
    ):

        raise RuntimeError(
            "tau_swing chứa NaN hoặc Inf."
        )


    # ========================================================
    # 9. CURRENT TRAJECTORY INTERNAL PHASE
    #
    # CubicBezierSwingTrajectory.update() already advances:
    #
    #     phi_(k+1)
    #
    # Therefore this value represents the phase stored for
    # the NEXT low-level update.
    # ========================================================

    if hasattr(
        swing_trajectory,
        "get_phase"
    ):

        phase_next = (
            swing_trajectory.get_phase()
        )


    else:

        phase_next = np.nan


    # ========================================================
    # 10. RETURN
    #
    # No torque saturation is performed here.
    #
    # MuJoCo actuator ctrlrange defined in the XML is
    # responsible for actuator control limits.
    # ========================================================

    return SwingLowLevelResult(

        tau_swing=(
            tau_swing.copy()
        ),

        W_p_foothold=(
            np.asarray(
                W_p_foothold,
                dtype=float
            ).copy()
        ),

        W_p_swing_des=(
            W_p_swing_des.copy()
        ),

        q_swing_des=(
            q_swing_des.copy()
        ),

        q_swing=(
            q_swing.copy()
        ),

        q_dot_swing=(
            q_dot_swing.copy()
        ),

        ik_diagnostics=(
            ik_diagnostics
        ),

        swing_foot=(
            swing_foot
        ),

        phase_next=float(
            phase_next
        )
    )