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
#
#         =
#
#     [q_des(k) - q_des(k-1)] / dt
#
#
# Then:
#
#     1. velocity limit
#
#     2. first-order low-pass filter
#
#
# Filter:
#
#     alpha
#
#       =
#
#     1 - exp(-dt / tau)
#
#
#     q_dot_des_f(k)
#
#       =
#
#     alpha * q_dot_des_raw(k)
#
#       +
#
#     (1-alpha) * q_dot_des_f(k-1)
#
#
# IMPORTANT:
#
# At:
#
#     swing_phase = 0
#
# and:
#
#     swing_phase = 1
#
# q_dot_des is explicitly forced to zero.
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


# Numerical tolerance used to recognize exact
# liftoff / touchdown trajectory endpoints.
SWING_PHASE_ENDPOINT_TOLERANCE = 1e-10


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
# RESULT
# ============================================================

@dataclass
class SwingLowLevelResult:
    """
    Result of one swing-leg low-level control update.
    """

    # --------------------------------------------------------
    # Final swing-leg joint torque
    # --------------------------------------------------------

    tau_swing: np.ndarray


    # --------------------------------------------------------
    # Swing-foot trajectory
    # --------------------------------------------------------

    W_p_liftoff: np.ndarray

    W_p_foothold: np.ndarray

    W_p_swing_des: np.ndarray

    W_v_swing_des: np.ndarray

    W_a_swing_des: np.ndarray

    W_p_apex: np.ndarray


    # --------------------------------------------------------
    # Joint-space reference / state
    # --------------------------------------------------------

    q_swing_des: np.ndarray

    q_swing: np.ndarray

    q_dot_swing: np.ndarray

    q_dot_swing_des: np.ndarray


    # --------------------------------------------------------
    # IK
    # --------------------------------------------------------

    ik_diagnostics: dict


    # --------------------------------------------------------
    # Gait information
    # --------------------------------------------------------

    swing_foot: str

    swing_phase: float

    T_swing: float


# ============================================================
# DESIRED JOINT VELOCITY HISTORY
# ============================================================

def _new_q_dot_des_state():

    return {

        "initialized":
            False,

        "previous_time":
            None,

        "previous_q_des":
            None,

        "q_dot_des_filtered":
            np.zeros(
                5,
                dtype=float,
            ),
    }


_Q_DOT_DES_STATE = {

    "L":
        _new_q_dot_des_state(),

    "R":
        _new_q_dot_des_state(),
}


# ============================================================
# IK ORIENTATION TOLERANCE
# ============================================================

IK_ORIENTATION_TOLERANCE = 1e-3


# ============================================================
# VECTOR VALIDATION
# ============================================================

def _validate_vector3(
    value,
    name
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

        "L"
            reset left leg.

        "R"
            reset right leg.

        None
            reset both legs.
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

    This is only used inside the swing interval.

    At exact liftoff and touchdown the caller explicitly sets:

        q_dot_des = 0
    """

    # ========================================================
    # 1. VALIDATE FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R",
    ):

        raise ValueError(
            "swing_foot phải là 'L' hoặc 'R'."
        )


    # ========================================================
    # 2. VALIDATE DESIRED JOINT POSITION
    # ========================================================

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


    # ========================================================
    # 3. VALIDATE TIME
    # ========================================================

    current_time = float(
        current_time
    )


    if not np.isfinite(
        current_time
    ):

        raise ValueError(
            "current_time phải hữu hạn."
        )


    # ========================================================
    # 4. CURRENT HISTORY
    # ========================================================

    state = _Q_DOT_DES_STATE[
        swing_foot
    ]


    # ========================================================
    # 5. FIRST SAMPLE
    # ========================================================

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


    # ========================================================
    # 6. PREVIOUS SAMPLE
    # ========================================================

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

            q_swing_des=(
                q_swing_des
            ),

            swing_foot=(
                swing_foot
            ),

            current_time=(
                current_time
            ),
        )


    # ========================================================
    # 7. TIME INTERVAL
    # ========================================================

    dt = (

        current_time

        -

        float(
            previous_time
        )
    )


    # ========================================================
    # 8. INVALID / LONG GAP
    #
    # A long gap normally means that this leg was in stance
    # and has now started another swing.
    #
    # Avoid derivative spikes by restarting with zero desired
    # joint velocity.
    # ========================================================

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


    # ========================================================
    # 9. RAW FINITE-DIFFERENCE VELOCITY
    # ========================================================

    q_dot_des_raw = (

        q_swing_des

        -

        np.asarray(
            previous_q_des,
            dtype=float,
        )

    ) / dt


    # ========================================================
    # 10. LIMIT NUMERICAL DERIVATIVE
    # ========================================================

    q_dot_des_raw = np.clip(

        q_dot_des_raw,

        -Q_DOT_DES_LIMIT,

        Q_DOT_DES_LIMIT,
    )


    # ========================================================
    # 11. LOW-PASS FILTER
    # ========================================================

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


    # ========================================================
    # 12. FINAL VELOCITY LIMIT
    # ========================================================

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


    # ========================================================
    # 13. UPDATE HISTORY
    # ========================================================

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
    q_swing : np.ndarray, shape (5,)

    q_dot_swing : np.ndarray, shape (5,)
    """

    # ========================================================
    # 1. VALIDATE FOOT
    # ========================================================

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


    # ========================================================
    # 2. READ MUJOCO JOINT STATE
    # ========================================================

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


    # ========================================================
    # 3. VALIDATION
    # ========================================================

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
    swing_foot,
    W_p_liftoff,
    W_p_foothold,
    swing_phase,
    T_swing,
    swing_height,
):
    """
    Compute low-level torque for one swing leg.

    Architecture
    ------------

        externally synchronized gait FSM

            ->

        W_p_liftoff
        W_p_foothold
        swing_phase
        T_swing

            ->

        quintic swing-foot trajectory

            ->

        pitch-only 4D numerical IK

            ->

        q_dot_des estimator

            ->

        joint-space PD

            ->

        tau_swing


    Important
    ---------

    This function does NOT:

        - generate footholds

        - run Raibert heuristic

        - modify gait phase

        - detect contact

        - determine liftoff

        - determine touchdown

        - advance MuJoCo

        - call mj_step()


    Those responsibilities belong to the higher-level
    gait / simulation integration layer.


    Parameters
    ----------
    model : mujoco.MjModel

        MuJoCo model.


    data : mujoco.MjData

        Current MuJoCo state.


    swing_foot : {"L", "R"}

        Current swing leg.


    W_p_liftoff : array_like, shape (3,)

        Physical swing-foot World position latched at liftoff.

        IMPORTANT:

            This must remain constant throughout the complete
            swing phase.


    W_p_foothold : array_like, shape (3,)

        Current desired foothold in World frame.

        This target is generated externally by:

            Command/foothold_generator.py

        or a higher-level replanning layer.


    swing_phase : float

        Externally synchronized swing phase:

            0 <= swing_phase <= 1

        Typically:

            swing_phase

                =

            t_phase / T_SS


    T_swing : float

        Total swing duration [s].

        For current walking structure:

            T_swing = T_SS


    swing_height : float

        Desired additional swing-foot clearance [m].


    Returns
    -------
    result : SwingLowLevelResult
    """

    # ========================================================
    # 1. VALIDATE SWING FOOT
    # ========================================================

    if swing_foot not in (
        "L",
        "R",
    ):

        raise ValueError(
            "compute_swing_low_level_control chỉ được gọi "
            "khi swing_foot là 'L' hoặc 'R'."
        )


    # ========================================================
    # 2. VALIDATE TRAJECTORY INPUTS
    # ========================================================

    W_p_liftoff = _validate_vector3(
        W_p_liftoff,
        "W_p_liftoff"
    )


    W_p_foothold = _validate_vector3(
        W_p_foothold,
        "W_p_foothold"
    )


    swing_phase = float(
        swing_phase
    )


    if not np.isfinite(
        swing_phase
    ):

        raise ValueError(
            "swing_phase phải hữu hạn."
        )


    if (
        swing_phase < 0.0
        or
        swing_phase > 1.0
    ):

        raise ValueError(
            "swing_phase phải nằm trong [0, 1]."
        )


    T_swing = float(
        T_swing
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


    swing_height = float(
        swing_height
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
    # 3. QUINTIC SWING-FOOT TRAJECTORY
    #
    # Horizontal:
    #
    #     one quintic over entire swing
    #
    # Vertical:
    #
    #     liftoff -> apex
    #
    #     apex    -> touchdown
    #
    # ========================================================

    trajectory = (
        evaluate_quintic_swing_trajectory(

            W_p_liftoff=(
                W_p_liftoff
            ),

            W_p_foothold=(
                W_p_foothold
            ),

            swing_phase=(
                swing_phase
            ),

            T_swing=(
                T_swing
            ),

            swing_height=(
                swing_height
            )
        )
    )


    W_p_swing_des = np.asarray(

        trajectory[
            "W_p_swing_des"
        ],

        dtype=float,

    ).reshape(-1)


    W_v_swing_des = np.asarray(

        trajectory[
            "W_v_swing_des"
        ],

        dtype=float,

    ).reshape(-1)


    W_a_swing_des = np.asarray(

        trajectory[
            "W_a_swing_des"
        ],

        dtype=float,

    ).reshape(-1)


    W_p_apex = np.asarray(

        trajectory[
            "W_p_apex"
        ],

        dtype=float,

    ).reshape(-1)


    # ========================================================
    # 4. TRAJECTORY OUTPUT VALIDATION
    # ========================================================

    for value, name in (

        (
            W_p_swing_des,
            "W_p_swing_des"
        ),

        (
            W_v_swing_des,
            "W_v_swing_des"
        ),

        (
            W_a_swing_des,
            "W_a_swing_des"
        ),

        (
            W_p_apex,
            "W_p_apex"
        ),
    ):

        if value.shape != (
            3,
        ):

            raise RuntimeError(
                f"{name} phải có shape (3,), "
                f"nhưng nhận được {value.shape}."
            )


        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise RuntimeError(
                f"{name} chứa NaN hoặc Inf."
            )


    # ========================================================
    # 5. CURRENT SWING-JOINT STATE
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
    # 6. PITCH-ONLY HYBRID NUMERICAL IK
    #
    # Task:
    #
    #     [x, y, z, pitch]
    #
    # Roll:
    #
    #     free
    #
    # Yaw:
    #
    #     free
    #
    # Pitch:
    #
    #     constrained to SWING_PITCH_DES
    # ========================================================

    (
        q_swing_des,
        ik_diagnostics,

    ) = solve_numerical_IK_hybrid(

        model=model,

        data=data,

        W_p_swing_des=(
            W_p_swing_des
        ),

        swing_foot=(
            swing_foot
        ),

        pitch_swing_des=(
            SWING_PITCH_DES
        ),

        sigma_threshold=(
            SIGMA_THRESHOLD
        ),

        alpha_newton=(
            ALPHA_NEWTON
        ),

        eta_gradient=(
            ETA_GRADIENT
        ),

        tolerance=(
            IK_TOLERANCE
        ),

        orientation_tolerance=(
            IK_ORIENTATION_TOLERANCE
        ),

        max_iterations=(
            IK_MAX_ITERATIONS
        ),
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
    # 7. DESIRED JOINT VELOCITY
    #
    # IMPORTANT:
    #
    # Quintic Cartesian reference guarantees:
    #
    #     v = 0
    #
    # at:
    #
    #     swing_phase = 0
    #
    # and:
    #
    #     swing_phase = 1
    #
    #
    # Since q_dot_des is currently estimated by finite
    # differences of IK outputs, explicitly enforce:
    #
    #     q_dot_des = 0
    #
    # at both trajectory endpoints.
    # ========================================================

    is_liftoff_endpoint = (

        swing_phase

        <=

        SWING_PHASE_ENDPOINT_TOLERANCE
    )


    is_touchdown_endpoint = (

        swing_phase

        >=

        (
            1.0
            -
            SWING_PHASE_ENDPOINT_TOLERANCE
        )
    )


    if is_liftoff_endpoint:

        # ----------------------------------------------------
        # Start a completely new desired-velocity history.
        # ----------------------------------------------------

        reset_swing_joint_velocity_reference(
            swing_foot
        )


        q_dot_swing_des = np.zeros(
            5,
            dtype=float,
        )


        # ----------------------------------------------------
        # Initialize history using the current desired joint
        # configuration so the next low-level sample has a
        # valid finite-difference reference.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Force zero desired joint velocity at touchdown.
        #
        # Reset history because this leg will become stance.
        # ----------------------------------------------------

        q_dot_swing_des = np.zeros(
            5,
            dtype=float,
        )


        reset_swing_joint_velocity_reference(
            swing_foot
        )


    else:

        q_dot_swing_des = (
            _estimate_q_dot_swing_des(

                q_swing_des=(
                    q_swing_des
                ),

                swing_foot=(
                    swing_foot
                ),

                current_time=(
                    data.time
                ),
            )
        )


    # ========================================================
    # 8. JOINT-SPACE PD
    #
    #     tau
    #
    #       =
    #
    #     Kp (q_des - q)
    #
    #       +
    #
    #     Kd (q_dot_des - q_dot)
    # ========================================================

    tau_swing = (
        compute_swing_joint_PD(

            q_des=(
                q_swing_des
            ),

            q=(
                q_swing
            ),

            q_dot=(
                q_dot_swing
            ),

            Kp=(
                KP_SWING
            ),

            Kd=(
                KD_SWING
            ),

            q_dot_des=(
                q_dot_swing_des
            ),
        )
    )


    tau_swing = np.asarray(
        tau_swing,
        dtype=float,
    ).reshape(-1)


    # ========================================================
    # 9. TORQUE VALIDATION
    # ========================================================

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
    # 10. RETURN
    #
    # No torque clipping is performed here.
    #
    # Final actuator limits remain handled by MuJoCo
    # actuator ctrlrange.
    # ========================================================

    return SwingLowLevelResult(

        tau_swing=(
            tau_swing.copy()
        ),

        W_p_liftoff=(
            W_p_liftoff.copy()
        ),

        W_p_foothold=(
            W_p_foothold.copy()
        ),

        W_p_swing_des=(
            W_p_swing_des.copy()
        ),

        W_v_swing_des=(
            W_v_swing_des.copy()
        ),

        W_a_swing_des=(
            W_a_swing_des.copy()
        ),

        W_p_apex=(
            W_p_apex.copy()
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

        q_dot_swing_des=(
            q_dot_swing_des.copy()
        ),

        ik_diagnostics=(
            ik_diagnostics
        ),

        swing_foot=(
            str(
                swing_foot
            )
        ),

        swing_phase=(
            float(
                swing_phase
            )
        ),

        T_swing=(
            float(
                T_swing
            )
        ),
    )