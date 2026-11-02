from dataclasses import dataclass

import mujoco
import numpy as np


# ============================================================
# SWING TRAJECTORY
# ============================================================

from Trajectory_Planner.quintic_swing_trajectory import (
    evaluate_quintic_swing_trajectory,
)


# ============================================================
# KINEMATICS
# ============================================================

from Kinematics.numerical_IK_hybrid import (
    solve_numerical_IK_hybrid,
    DEFAULT_K_POSTURE,
    DEFAULT_K_ROLL,
    DEFAULT_K_PITCH,
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
# SWING IK / SECONDARY TASK PARAMETERS
#
# Primary task:
#     [x, y, z]
#
# Secondary task in primary-task null space:
#     foot roll/pitch stabilization
#     +
#     joint-posture regulation
#
# Pitch seed in numerical_IK_hybrid.py prioritizes:
#     hip pitch + knee
#
# Ankle remains available but receives lower secondary weight.
# ============================================================

SWING_K_POSTURE = DEFAULT_K_POSTURE
SWING_K_ROLL = DEFAULT_K_ROLL
SWING_K_PITCH = DEFAULT_K_PITCH


# ============================================================
# DESIRED JOINT VELOCITY ESTIMATOR
# ============================================================

Q_DOT_DES_FILTER_TAU = 0.005

Q_DOT_DES_HISTORY_RESET_GAP = 0.020

Q_DOT_DES_LIMIT = np.array(
    [
        6.0,    # hip yaw
        6.0,    # hip roll
        8.0,    # hip pitch
        10.0,   # knee
        8.0,    # ankle
    ],
    dtype=float,
)


# ============================================================
# SWING REFERENCE LATCH
#
# A swing reference consists of:
#
#     q_posture_ref
#     R_swing_ref
#
# Both are captured from the real current MuJoCo state once at
# the beginning of a swing command.
#
# IMPORTANT:
#
# We DO NOT use a phase decrease as a new-swing detector.
#
# During startup:
#
#     pre-liftoff phase ~= 0.15
#          ->
#     physical LO
#          ->
#     cyclic swing phase ~= 0
#
# This is still the SAME physical swing.
#
# A new swing of the same leg is recognized by the long period
# during which that leg was not controlled as swing.
# ============================================================

SWING_REFERENCE_RESET_GAP = 0.050

SWING_PHASE_ENDPOINT_TOLERANCE = 1e-10


# ============================================================
# JOINT / SITE ORDER
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

SWING_IK_SITE_NAMES = {
    "L": "L6_IK",
    "R": "R6_IK",
}


# ============================================================
# RESULT
# ============================================================

@dataclass
class SwingLowLevelResult:
    tau_swing: np.ndarray

    W_p_liftoff: np.ndarray
    W_p_foothold: np.ndarray
    W_p_swing_des: np.ndarray
    W_v_swing_des: np.ndarray
    W_a_swing_des: np.ndarray
    W_p_apex: np.ndarray

    q_swing_des: np.ndarray
    q_swing: np.ndarray
    q_dot_swing: np.ndarray
    q_dot_swing_des: np.ndarray

    q_posture_ref: np.ndarray
    R_swing_ref: np.ndarray

    k_posture: float
    k_roll: float
    k_pitch: float

    swing_reference_was_latched: bool
    posture_reference_was_latched: bool
    orientation_reference_was_latched: bool

    ik_diagnostics: dict

    swing_foot: str
    swing_phase: float
    T_swing: float


# ============================================================
# STATE FACTORIES
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


def _new_swing_reference_state():
    return {
        "initialized": False,
        "q_posture_ref": None,
        "R_swing_ref": None,
        "previous_time": None,
        "previous_swing_phase": None,
    }


_Q_DOT_DES_STATE = {
    "L": _new_q_dot_des_state(),
    "R": _new_q_dot_des_state(),
}

_SWING_REFERENCE_STATE = {
    "L": _new_swing_reference_state(),
    "R": _new_swing_reference_state(),
}


# ============================================================
# VALIDATION
# ============================================================

def _validate_vector3(
    value,
    name,
):
    value = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if value.shape != (
        3,
    ):
        raise ValueError(
            f"{name} phải có shape (3,), "
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


def _validate_vector5(
    value,
    name,
):
    value = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if value.shape != (
        5,
    ):
        raise ValueError(
            f"{name} phải có shape (5,), "
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


def _validate_rotation_matrix(
    value,
    name,
):
    value = np.asarray(
        value,
        dtype=float,
    )

    if value.shape != (
        3,
        3,
    ):
        raise ValueError(
            f"{name} phải có shape (3,3), "
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


# ============================================================
# RESET FUNCTIONS
# ============================================================

def reset_swing_joint_velocity_reference(
    swing_foot=None,
):
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


def reset_swing_reference(
    swing_foot=None,
):
    """
    Reset BOTH:
        q_posture_ref
        R_swing_ref
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
        _SWING_REFERENCE_STATE[
            leg
        ] = _new_swing_reference_state()


def reset_swing_posture_reference(
    swing_foot=None,
):
    """
    Backward-compatible name.

    The posture and orientation references belong to one physical
    swing snapshot, so both are reset together.
    """
    reset_swing_reference(
        swing_foot
    )


def reset_swing_low_level_state(
    swing_foot=None,
):
    reset_swing_joint_velocity_reference(
        swing_foot
    )

    reset_swing_reference(
        swing_foot
    )


# ============================================================
# REFERENCE GETTERS
# ============================================================

def get_swing_posture_reference(
    swing_foot,
):
    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    q_ref = _SWING_REFERENCE_STATE[
        swing_foot
    ][
        "q_posture_ref"
    ]

    if q_ref is None:
        return None

    return np.asarray(
        q_ref,
        dtype=float,
    ).copy()


def get_swing_orientation_reference(
    swing_foot,
):
    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    R_ref = _SWING_REFERENCE_STATE[
        swing_foot
    ][
        "R_swing_ref"
    ]

    if R_ref is None:
        return None

    return _validate_rotation_matrix(
        R_ref,
        "R_swing_ref",
    )


# ============================================================
# CURRENT SWING JOINT STATE
# ============================================================

def _get_swing_joint_state(
    model,
    data,
    swing_foot,
):
    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    q_swing = np.zeros(
        5,
        dtype=float,
    )

    q_dot_swing = np.zeros(
        5,
        dtype=float,
    )

    for i, joint_name in enumerate(
        SWING_LEG_JOINT_NAMES[
            swing_foot
        ]
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

    return (
        _validate_vector5(
            q_swing,
            "q_swing",
        ),
        _validate_vector5(
            q_dot_swing,
            "q_dot_swing",
        ),
    )


# ============================================================
# CURRENT SWING-FOOT ORIENTATION
# ============================================================

def _get_swing_foot_rotation_matrix(
    model,
    data,
    swing_foot,
):
    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    site_name = SWING_IK_SITE_NAMES[
        swing_foot
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

    R_swing = np.asarray(
        data.site_xmat[
            site_id
        ],
        dtype=float,
    ).reshape(
        3,
        3,
    )

    return _validate_rotation_matrix(
        R_swing,
        "R_swing",
    )


# ============================================================
# GET / LATCH COMPLETE SWING REFERENCE
# ============================================================

def _get_or_latch_swing_reference(
    q_swing,
    R_swing,
    swing_foot,
    swing_phase,
    current_time,
):
    q_swing = _validate_vector5(
        q_swing,
        "q_swing",
    )

    R_swing = _validate_rotation_matrix(
        R_swing,
        "R_swing",
    )

    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    swing_phase = float(
        swing_phase
    )

    current_time = float(
        current_time
    )

    if (
        not np.isfinite(
            swing_phase
        )
        or
        swing_phase < 0.0
        or
        swing_phase > 1.0
    ):
        raise ValueError(
            "swing_phase phải nằm trong [0,1]."
        )

    if not np.isfinite(
        current_time
    ):
        raise ValueError(
            "current_time phải hữu hạn."
        )

    state = _SWING_REFERENCE_STATE[
        swing_foot
    ]

    should_latch = bool(
        not state[
            "initialized"
        ]
    )

    previous_time = state[
        "previous_time"
    ]

    if (
        state[
            "initialized"
        ]
        and
        previous_time is not None
    ):
        dt = (
            current_time
            -
            float(
                previous_time
            )
        )

        # The same leg has not been used as swing for a long
        # interval -> this is a new physical swing.
        if (
            not np.isfinite(
                dt
            )
            or
            dt < 0.0
            or
            dt
            >
            SWING_REFERENCE_RESET_GAP
        ):
            should_latch = True

    if should_latch:
        state[
            "q_posture_ref"
        ] = q_swing.copy()

        state[
            "R_swing_ref"
        ] = R_swing.copy()

        state[
            "initialized"
        ] = True

    state[
        "previous_time"
    ] = current_time

    state[
        "previous_swing_phase"
    ] = swing_phase

    q_posture_ref = _validate_vector5(
        state[
            "q_posture_ref"
        ],
        "latched q_posture_ref",
    ).copy()

    R_swing_ref = _validate_rotation_matrix(
        state[
            "R_swing_ref"
        ],
        "latched R_swing_ref",
    ).copy()

    return (
        q_posture_ref,
        R_swing_ref,
        bool(
            should_latch
        ),
    )


# ============================================================
# BACKWARD-COMPATIBLE POSTURE LATCH HELPER
# ============================================================

def _get_or_latch_posture_reference(
    q_swing,
    swing_foot,
    swing_phase,
    current_time,
    R_swing=None,
):
    """
    Compatibility wrapper.

    If R_swing is supplied, the full swing snapshot is used.
    Existing external code should prefer
    _get_or_latch_swing_reference().
    """
    if R_swing is None:
        state = _SWING_REFERENCE_STATE[
            swing_foot
        ]

        if state[
            "R_swing_ref"
        ] is None:
            R_swing = np.eye(
                3,
                dtype=float,
            )
        else:
            R_swing = state[
                "R_swing_ref"
            ]

    (
        q_posture_ref,
        _,
        was_latched,
    ) = _get_or_latch_swing_reference(
        q_swing=q_swing,
        R_swing=R_swing,
        swing_foot=swing_foot,
        swing_phase=swing_phase,
        current_time=current_time,
    )

    return (
        q_posture_ref,
        was_latched,
    )


# ============================================================
# ESTIMATE q_dot_des
# ============================================================

def _estimate_q_dot_swing_des(
    q_swing_des,
    swing_foot,
    current_time,
):
    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )

    q_swing_des = _validate_vector5(
        q_swing_des,
        "q_swing_des",
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

    q_dot_des_raw = (
        q_swing_des
        -
        np.asarray(
            previous_q_des,
            dtype=float,
        )
    ) / dt

    q_dot_des_raw = np.clip(
        q_dot_des_raw,
        -Q_DOT_DES_LIMIT,
        Q_DOT_DES_LIMIT,
    )

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
# SWING LOW-LEVEL CONTROLLER
# ============================================================

def compute_swing_low_level_control(
    model,
    data,
    swing_foot,
    W_p_liftoff,
    W_p_foothold,
    swing_phase,
    T_swing,
    swing_height,
):
    """
    Strictly computes u_k from x_k.

    No contact transition is created here.
    No mj_step() is called here.
    """

    if swing_foot not in (
        "L",
        "R",
    ):
        raise ValueError(
            "compute_swing_low_level_control chỉ được gọi "
            "khi swing_foot là 'L' hoặc 'R'."
        )

    W_p_liftoff = _validate_vector3(
        W_p_liftoff,
        "W_p_liftoff",
    )

    W_p_foothold = _validate_vector3(
        W_p_foothold,
        "W_p_foothold",
    )

    swing_phase = float(
        swing_phase
    )

    T_swing = float(
        T_swing
    )

    swing_height = float(
        swing_height
    )

    if (
        not np.isfinite(
            swing_phase
        )
        or
        swing_phase < 0.0
        or
        swing_phase > 1.0
    ):
        raise ValueError(
            "swing_phase phải nằm trong [0,1]."
        )

    if (
        not np.isfinite(
            T_swing
        )
        or
        T_swing <= 0.0
    ):
        raise ValueError(
            "T_swing phải hữu hạn và > 0."
        )

    if (
        not np.isfinite(
            swing_height
        )
        or
        swing_height < 0.0
    ):
        raise ValueError(
            "swing_height phải hữu hạn và >= 0."
        )

    # ========================================================
    # 1. CARTESIAN SWING TRAJECTORY
    # ========================================================

    trajectory = evaluate_quintic_swing_trajectory(
        W_p_liftoff=W_p_liftoff,
        W_p_foothold=W_p_foothold,
        swing_phase=swing_phase,
        T_swing=T_swing,
        swing_height=swing_height,
    )

    W_p_swing_des = _validate_vector3(
        trajectory[
            "W_p_swing_des"
        ],
        "W_p_swing_des",
    )

    W_v_swing_des = _validate_vector3(
        trajectory[
            "W_v_swing_des"
        ],
        "W_v_swing_des",
    )

    W_a_swing_des = _validate_vector3(
        trajectory[
            "W_a_swing_des"
        ],
        "W_a_swing_des",
    )

    W_p_apex = _validate_vector3(
        trajectory[
            "W_p_apex"
        ],
        "W_p_apex",
    )

    # ========================================================
    # 2. REAL CURRENT JOINT / FOOT-ORIENTATION STATE
    # ========================================================

    (
        q_swing,
        q_dot_swing,
    ) = _get_swing_joint_state(
        model=model,
        data=data,
        swing_foot=swing_foot,
    )

    R_swing = _get_swing_foot_rotation_matrix(
        model=model,
        data=data,
        swing_foot=swing_foot,
    )

    # ========================================================
    # 3. LATCH COMPLETE SWING SNAPSHOT
    #
    # q_posture_ref and R_swing_ref are captured together.
    # ========================================================

    (
        q_posture_ref,
        R_swing_ref,
        swing_reference_was_latched,
    ) = _get_or_latch_swing_reference(
        q_swing=q_swing,
        R_swing=R_swing,
        swing_foot=swing_foot,
        swing_phase=swing_phase,
        current_time=float(
            data.time
        ),
    )

    # ========================================================
    # 4. 3D PRIMARY IK + NULL-SPACE TILT/POSTURE SECONDARY
    # ========================================================

    (
        q_swing_des,
        ik_diagnostics,
    ) = solve_numerical_IK_hybrid(
        model=model,
        data=data,
        W_p_swing_des=W_p_swing_des,
        swing_foot=swing_foot,
        sigma_threshold=SIGMA_THRESHOLD,
        alpha_newton=ALPHA_NEWTON,
        eta_gradient=ETA_GRADIENT,
        tolerance=IK_TOLERANCE,
        max_iterations=IK_MAX_ITERATIONS,
        q_posture_ref=q_posture_ref,
        k_posture=SWING_K_POSTURE,
        R_swing_ref=R_swing_ref,
        k_roll=SWING_K_ROLL,
        k_pitch=SWING_K_PITCH,
    )

    if q_swing_des is None:
        raise RuntimeError(
            "Hybrid IK trả về q_swing_des = None."
        )

    q_swing_des = _validate_vector5(
        q_swing_des,
        "q_swing_des",
    )

    # ========================================================
    # 5. DESIRED JOINT VELOCITY
    # ========================================================

    is_liftoff_endpoint = bool(
        swing_phase
        <=
        SWING_PHASE_ENDPOINT_TOLERANCE
    )

    is_touchdown_endpoint = bool(
        swing_phase
        >=
        (
            1.0
            -
            SWING_PHASE_ENDPOINT_TOLERANCE
        )
    )

    if is_liftoff_endpoint:
        reset_swing_joint_velocity_reference(
            swing_foot
        )

        q_dot_swing_des = np.zeros(
            5,
            dtype=float,
        )

        state = _Q_DOT_DES_STATE[
            swing_foot
        ]

        state[
            "initialized"
        ] = True

        state[
            "previous_time"
        ] = float(
            data.time
        )

        state[
            "previous_q_des"
        ] = q_swing_des.copy()

        state[
            "q_dot_des_filtered"
        ] = q_dot_swing_des.copy()

    elif is_touchdown_endpoint:
        q_dot_swing_des = np.zeros(
            5,
            dtype=float,
        )

        reset_swing_joint_velocity_reference(
            swing_foot
        )

    else:
        q_dot_swing_des = _estimate_q_dot_swing_des(
            q_swing_des=q_swing_des,
            swing_foot=swing_foot,
            current_time=float(
                data.time
            ),
        )

    # ========================================================
    # 6. JOINT PD
    # ========================================================

    tau_swing = compute_swing_joint_PD(
        q_des=q_swing_des,
        q=q_swing,
        q_dot=q_dot_swing,
        Kp=KP_SWING,
        Kd=KD_SWING,
        q_dot_des=q_dot_swing_des,
    )

    tau_swing = _validate_vector5(
        tau_swing,
        "tau_swing",
    )

    # ========================================================
    # 7. RETURN
    # ========================================================

    return SwingLowLevelResult(
        tau_swing=tau_swing.copy(),

        W_p_liftoff=W_p_liftoff.copy(),
        W_p_foothold=W_p_foothold.copy(),
        W_p_swing_des=W_p_swing_des.copy(),
        W_v_swing_des=W_v_swing_des.copy(),
        W_a_swing_des=W_a_swing_des.copy(),
        W_p_apex=W_p_apex.copy(),

        q_swing_des=q_swing_des.copy(),
        q_swing=q_swing.copy(),
        q_dot_swing=q_dot_swing.copy(),
        q_dot_swing_des=q_dot_swing_des.copy(),

        q_posture_ref=q_posture_ref.copy(),
        R_swing_ref=R_swing_ref.copy(),

        k_posture=float(
            SWING_K_POSTURE
        ),
        k_roll=float(
            SWING_K_ROLL
        ),
        k_pitch=float(
            SWING_K_PITCH
        ),

        swing_reference_was_latched=bool(
            swing_reference_was_latched
        ),

        # Backward-compatible diagnostics.
        posture_reference_was_latched=bool(
            swing_reference_was_latched
        ),
        orientation_reference_was_latched=bool(
            swing_reference_was_latched
        ),

        ik_diagnostics=ik_diagnostics,

        swing_foot=str(
            swing_foot
        ),
        swing_phase=float(
            swing_phase
        ),
        T_swing=float(
            T_swing
        ),
    )
