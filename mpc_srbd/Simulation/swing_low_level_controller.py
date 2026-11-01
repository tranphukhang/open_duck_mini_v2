from dataclasses import dataclass

import mujoco
import numpy as np


# ============================================================
# FOOTHOLD PLANNER
#
# Keep this import at module level because run.py may replace:
#
#     swing_low_level_controller.compute_raibert_foothold
#
# with the DCM foothold callback.
# ============================================================

from Foothold_Planner.raibert_heuristic import (
    compute_raibert_foothold,
)


# ============================================================
# KINEMATICS
# ============================================================

from Kinematics.numerical_IK_hybrid import (
    solve_numerical_IK_hybrid,
)

from Kinematics.numerical_IK_params import (
    SIGMA_THRESHOLD,
    ALPHA_NEWTON,
    ETA_GRADIENT,
    IK_TOLERANCE,
    IK_MAX_ITERATIONS,
)


# ============================================================
# ACTUATOR
# ============================================================

from Actuator.swing_joint_PD import (
    compute_swing_joint_PD,
)

from Actuator.swing_joint_PD_params import (
    KP_SWING,
    KD_SWING,
)


# ============================================================
# SWING-FOOT ORIENTATION
#
# Current numerical IK task:
#
#     [x, y, z, pitch]
#
# Roll is free.
# Yaw is free.
# Pitch is constrained.
# ============================================================

SWING_PITCH_DES = 0.0


# ============================================================
# DESIRED JOINT VELOCITY ESTIMATOR
#
# q_dot_des is estimated from consecutive IK outputs:
#
#     q_dot_des_raw(k)
#         =
#     [q_des(k) - q_des(k-1)] / dt
#
# Then:
#
#     1) velocity limit
#     2) first-order low-pass filter
#
# Filter:
#
#     alpha = 1 - exp(-dt / tau)
#
#     q_dot_des_f(k)
#         =
#     alpha * q_dot_des_raw(k)
#         +
#     (1-alpha) * q_dot_des_f(k-1)
# ============================================================

Q_DOT_DES_FILTER_TAU = 0.005

Q_DOT_DES_HISTORY_RESET_GAP = 0.020

Q_DOT_DES_LIMIT = np.array(
    [
        6.0,    # hip yaw   [rad/s]
        6.0,    # hip roll  [rad/s]
        8.0,    # hip pitch [rad/s]
        10.0,   # knee      [rad/s]
        8.0,    # ankle     [rad/s]
    ],
    dtype=float,
)


# ============================================================
# JOINT ORDER
# ============================================================

SWING_LEG_JOINT_NAMES = {
    "L": (
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee",
        "L_ankle",
    ),
    "R": (
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee",
        "R_ankle",
    ),
}


# ============================================================
# DESIRED JOINT VELOCITY HISTORY
# ============================================================

def _new_q_dot_des_state():
    return {
        "initialized": False,
        "previous_time": None,
        "previous_q_des": None,
        "q_dot_des_filtered": np.zeros(
            5,
            dtype=float,
        ),
    }


_Q_DOT_DES_STATE = {
    "L": _new_q_dot_des_state(),
    "R": _new_q_dot_des_state(),
}


# ============================================================
# IK ORIENTATION TOLERANCE
# ============================================================

IK_ORIENTATION_TOLERANCE = 1e-3


# ============================================================
# RESULT
# ============================================================

@dataclass
class SwingLowLevelResult:
    tau_swing: np.ndarray

    W_p_foothold: np.ndarray

    W_p_swing_des: np.ndarray

    q_swing_des: np.ndarray

    q_swing: np.ndarray

    q_dot_swing: np.ndarray

    q_dot_swing_des: np.ndarray

    ik_diagnostics: dict

    swing_foot: str

    phase_next: float


# ============================================================
# RESET q_dot_des HISTORY
# ============================================================

def reset_swing_joint_velocity_reference(
    swing_foot=None,
):
    """
    Reset desired-joint-velocity history.

    Parameters
    ----------
    swing_foot : {"L", "R", None}
        None -> reset both legs.
    """

    if swing_foot is None:
        legs = (
            "L",
            "R",
        )

    else:
        if swing_foot not in (
            "L",
            "R",
        ):
            raise ValueError(
                "swing_foot phải là 'L', 'R' hoặc None."
            )

        legs = (
            swing_foot,
        )


    for leg in legs:
        _Q_DOT_DES_STATE[
            leg
        ] = _new_q_dot_des_state()


# ============================================================
# ESTIMATE q_dot_des
# ============================================================

def _estimate_q_dot_swing_des(
    q_swing_des,
    swing_foot,
    current_time,
):
    """
    Estimate desired swing-joint velocity from consecutive IK
    joint references.
    """

    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )


    q_swing_des = np.asarray(
        q_swing_des,
        dtype=float,
    ).reshape(-1)


    if q_swing_des.shape != (
        5,
    ):
        raise ValueError(
            "q_swing_des phải có shape (5,), "
            f"nhưng nhận được {q_swing_des.shape}."
        )


    if not np.all(
        np.isfinite(
            q_swing_des
        )
    ):
        raise ValueError(
            "q_swing_des chứa NaN hoặc Inf."
        )


    current_time = float(
        current_time
    )


    if not np.isfinite(
        current_time
    ):
        raise ValueError(
            "current_time phải hữu hạn."
        )


    state = _Q_DOT_DES_STATE[
        swing_foot
    ]


    # --------------------------------------------------------
    # First sample for this leg
    # --------------------------------------------------------

    if not state[
        "initialized"
    ]:
        q_dot_swing_des = np.zeros(
            5,
            dtype=float,
        )

        state[
            "initialized"
        ] = True

        state[
            "previous_time"
        ] = current_time

        state[
            "previous_q_des"
        ] = q_swing_des.copy()

        state[
            "q_dot_des_filtered"
        ] = q_dot_swing_des.copy()

        return q_dot_swing_des


    previous_time = state[
        "previous_time"
    ]

    previous_q_des = state[
        "previous_q_des"
    ]


    # --------------------------------------------------------
    # Time interval
    # --------------------------------------------------------

    if (
        previous_time is None
        or
        previous_q_des is None
    ):
        reset_swing_joint_velocity_reference(
            swing_foot
        )

        return _estimate_q_dot_swing_des(
            q_swing_des=q_swing_des,
            swing_foot=swing_foot,
            current_time=current_time,
        )


    dt = (
        current_time
        -
        float(
            previous_time
        )
    )


    # --------------------------------------------------------
    # Reset after a long gap or invalid time step
    #
    # This avoids a derivative spike when a leg starts a new
    # swing after being inactive.
    # --------------------------------------------------------

    if (
        not np.isfinite(
            dt
        )
        or
        dt <= 0.0
        or
        dt > Q_DOT_DES_HISTORY_RESET_GAP
    ):
        q_dot_swing_des = np.zeros(
            5,
            dtype=float,
        )

        state[
            "previous_time"
        ] = current_time

        state[
            "previous_q_des"
        ] = q_swing_des.copy()

        state[
            "q_dot_des_filtered"
        ] = q_dot_swing_des.copy()

        return q_dot_swing_des


    # --------------------------------------------------------
    # Finite difference
    # --------------------------------------------------------

    q_dot_des_raw = (
        q_swing_des
        -
        np.asarray(
            previous_q_des,
            dtype=float,
        )
    ) / dt


    # --------------------------------------------------------
    # Limit numerical IK derivative spikes
    # --------------------------------------------------------

    q_dot_des_raw = np.clip(
        q_dot_des_raw,
        -Q_DOT_DES_LIMIT,
        Q_DOT_DES_LIMIT,
    )


    # --------------------------------------------------------
    # Low-pass filter
    # --------------------------------------------------------

    if Q_DOT_DES_FILTER_TAU <= 0.0:
        alpha = 1.0

    else:
        alpha = (
            1.0
            -
            np.exp(
                -dt
                /
                Q_DOT_DES_FILTER_TAU
            )
        )


    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )


    q_dot_des_previous = np.asarray(
        state[
            "q_dot_des_filtered"
        ],
        dtype=float,
    )


    q_dot_swing_des = (
        alpha
        *
        q_dot_des_raw
        +
        (
            1.0
            -
            alpha
        )
        *
        q_dot_des_previous
    )


    q_dot_swing_des = np.clip(
        q_dot_swing_des,
        -Q_DOT_DES_LIMIT,
        Q_DOT_DES_LIMIT,
    )


    if not np.all(
        np.isfinite(
            q_dot_swing_des
        )
    ):
        raise RuntimeError(
            "q_dot_swing_des chứa NaN hoặc Inf."
        )


    # --------------------------------------------------------
    # Update history
    # --------------------------------------------------------

    state[
        "previous_time"
    ] = current_time

    state[
        "previous_q_des"
    ] = q_swing_des.copy()

    state[
        "q_dot_des_filtered"
    ] = q_dot_swing_des.copy()


    return q_dot_swing_des


# ============================================================
# GET CURRENT SWING-LEG JOINT STATE
# ============================================================

def _get_swing_joint_state(
    model,
    data,
    swing_foot,
):
    """
    Read current swing-leg joint positions and velocities.

    Returns
    -------
    q_swing : shape (5,)
    q_dot_swing : shape (5,)
    """

    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )


    joint_names = SWING_LEG_JOINT_NAMES[
        swing_foot
    ]


    q_swing = np.zeros(
        5,
        dtype=float,
    )

    q_dot_swing = np.zeros(
        5,
        dtype=float,
    )


    for i, joint_name in enumerate(
        joint_names
    ):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )


        if joint_id < 0:
            raise ValueError(
                f"Không tìm thấy joint '{joint_name}' "
                "trong MuJoCo model."
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


        q_swing[
            i
        ] = data.qpos[
            qpos_address
        ]

        q_dot_swing[
            i
        ] = data.qvel[
            dof_address
        ]


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


    return (
        q_swing,
        q_dot_swing,
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
    ground_height=0.0,
):
    """
    Compute low-level torque for the current swing leg.

    Pipeline
    --------

        foothold planner
            ->
        Bezier swing trajectory
            ->
        pitch-only 4D numerical IK
            ->
        q_dot_des estimator
            ->
        joint-space PD


    PD law
    ------

        tau
            =
        Kp * (q_des - q)
            +
        Kd * (q_dot_des - q_dot)


    Important
    ---------

    The function signature is kept unchanged so the existing
    run.py can continue calling this function exactly as before.
    """

    # ========================================================
    # 1. VALIDATE INPUT
    # ========================================================

    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "compute_swing_low_level_control chỉ được gọi "
            "khi swing_foot là 'L' hoặc 'R'."
        )


    if swing_trajectory is None:
        raise ValueError(
            "swing_trajectory không được là None."
        )


    if not hasattr(
        swing_trajectory,
        "update",
    ):
        raise TypeError(
            "swing_trajectory phải có hàm update()."
        )


    # ========================================================
    # 2. FOOTHOLD PLANNER
    #
    # IMPORTANT:
    #
    # run.py may monkeypatch the module-level function
    # compute_raibert_foothold with the DCM foothold planner.
    # Keep this call at module level.
    # ========================================================

    W_p_foothold = compute_raibert_foothold(
        W_p_CoM=W_p_CoM,
        W_R_B=W_R_B,
        B_p_L6_offset=B_p_L6_offset,
        B_p_R6_offset=B_p_R6_offset,
        W_p_dot_CoM=W_p_dot_CoM,
        W_p_dot_CoM_ref=W_p_dot_CoM_ref,
        swing_foot=swing_foot,
        delta_T=delta_T,
        T_stance=T_stance,
        k_d=k_d,
        ground_height=ground_height,
    )


    if W_p_foothold is None:
        raise RuntimeError(
            "Foothold planner trả về None "
            "trong khi đang có swing foot."
        )


    W_p_foothold = np.asarray(
        W_p_foothold,
        dtype=float,
    ).reshape(-1)


    if W_p_foothold.shape != (
        3,
    ):
        raise RuntimeError(
            "W_p_foothold phải có shape (3,), "
            f"nhưng nhận được {W_p_foothold.shape}."
        )


    if not np.all(
        np.isfinite(
            W_p_foothold
        )
    ):
        raise RuntimeError(
            "W_p_foothold chứa NaN hoặc Inf."
        )


    # ========================================================
    # 3. SWING-FOOT TRAJECTORY
    # ========================================================

    W_p_swing_des = swing_trajectory.update(
        W_p_L6=W_p_L6,
        W_p_R6=W_p_R6,
        W_p_foothold=W_p_foothold,
        swing_foot=swing_foot,
    )


    if W_p_swing_des is None:
        raise RuntimeError(
            "Swing trajectory trả về None "
            "trong khi đang có swing foot."
        )


    W_p_swing_des = np.asarray(
        W_p_swing_des,
        dtype=float,
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
    # 4. CURRENT SWING JOINT STATE
    # ========================================================

    (
        q_swing,
        q_dot_swing,
    ) = _get_swing_joint_state(
        model=model,
        data=data,
        swing_foot=swing_foot,
    )


    # ========================================================
    # 5. PITCH-ONLY HYBRID NUMERICAL IK
    #
    # Task:
    #
    #     [x, y, z, pitch]
    #
    # Roll free.
    # Yaw free.
    # ========================================================

    (
        q_swing_des,
        ik_diagnostics,
    ) = solve_numerical_IK_hybrid(
        model=model,
        data=data,
        W_p_swing_des=W_p_swing_des,
        swing_foot=swing_foot,
        pitch_swing_des=SWING_PITCH_DES,
        sigma_threshold=SIGMA_THRESHOLD,
        alpha_newton=ALPHA_NEWTON,
        eta_gradient=ETA_GRADIENT,
        tolerance=IK_TOLERANCE,
        orientation_tolerance=IK_ORIENTATION_TOLERANCE,
        max_iterations=IK_MAX_ITERATIONS,
    )


    if q_swing_des is None:
        raise RuntimeError(
            "Hybrid IK trả về q_swing_des = None."
        )


    q_swing_des = np.asarray(
        q_swing_des,
        dtype=float,
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
    # 6. DESIRED JOINT VELOCITY
    # ========================================================

    q_dot_swing_des = _estimate_q_dot_swing_des(
        q_swing_des=q_swing_des,
        swing_foot=swing_foot,
        current_time=data.time,
    )


    # ========================================================
    # 7. JOINT-SPACE PD
    #
    #     tau
    #         =
    #     Kp(q_des-q)
    #         +
    #     Kd(q_dot_des-q_dot)
    # ========================================================

    tau_swing = compute_swing_joint_PD(
        q_des=q_swing_des,
        q=q_swing,
        q_dot=q_dot_swing,
        Kp=KP_SWING,
        Kd=KD_SWING,
        q_dot_des=q_dot_swing_des,
    )


    tau_swing = np.asarray(
        tau_swing,
        dtype=float,
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
    # 8. TRAJECTORY PHASE
    #
    # update() may already advance the internal phase, so this
    # is the phase stored for the next update.
    # ========================================================

    if hasattr(
        swing_trajectory,
        "get_phase",
    ):
        phase_next = float(
            swing_trajectory.get_phase()
        )

    else:
        phase_next = np.nan


    # ========================================================
    # 9. RETURN
    #
    # No torque clipping is performed here.
    # MuJoCo actuator ctrlrange handles actuator limits.
    # ========================================================

    return SwingLowLevelResult(
        tau_swing=tau_swing.copy(),
        W_p_foothold=W_p_foothold.copy(),
        W_p_swing_des=W_p_swing_des.copy(),
        q_swing_des=q_swing_des.copy(),
        q_swing=q_swing.copy(),
        q_dot_swing=q_dot_swing.copy(),
        q_dot_swing_des=q_dot_swing_des.copy(),
        ik_diagnostics=ik_diagnostics,
        swing_foot=swing_foot,
        phase_next=phase_next,
    )