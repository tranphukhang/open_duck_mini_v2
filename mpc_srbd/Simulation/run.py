from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

SIMULATION_DIR = Path(__file__).resolve().parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent

STATE_DIR = MPC_SRBD_DIR / "State"
SRBD_DIR = MPC_SRBD_DIR / "SRBD"
QP_DIR = MPC_SRBD_DIR / "QP"
KINEMATICS_DIR = MPC_SRBD_DIR / "Kinematics"
ACTUATOR_DIR = MPC_SRBD_DIR / "Actuator"
COMMAND_DIR = MPC_SRBD_DIR / "Command"

for directory in (
    MPC_SRBD_DIR,
    STATE_DIR,
    SRBD_DIR,
    QP_DIR,
    KINEMATICS_DIR,
    ACTUATOR_DIR,
    COMMAND_DIR,
):
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(
            0,
            directory_str
        )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from simulation_setup import (
    initialize_simulation
)

from get_current_state_mpc import (
    get_current_state_mpc
)

from build_SRBD_model import (
    build_SRBD_model
)

from QP_parameters import (
    N,
    dt_mpc,
    dt_mpc_update
)

from build_and_solve_QP import (
    build_and_solve_QP
)

from Command.contact_schedule import (
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT
)

from W_wrench_to_joint_torque import (
    get_joint_torque_from_MPC_wrench
)

from actuator_mapping import (
    apply_joint_torque
)

from swing_joint_PD import (
    compute_swing_joint_PD
)

from swing_joint_PD_params import (
    KP_SWING,
    KD_SWING
)


# ============================================================
# NUMPY DISPLAY
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=200
)


# ============================================================
# TEST PURPOSE
# ============================================================
#
# Mục đích:
#
# So sánh khả năng hiện thực wrench của chân stance trong:
#
#   A. Single support + swing leg giữ yên
#
#   B. Single support + swing leg chuyển động chậm
#
#
# Nếu:
#
#   error_STATIC nhỏ
#
# nhưng:
#
#   error_MOVING tăng mạnh
#
# thì có bằng chứng rằng:
#
#   swing-leg dynamics
#
# ảnh hưởng đáng kể tới:
#
#   stance-wrench realization.
#
#
# Chuỗi test:
#
#   Phase 1:
#
#       Double Support
#       dịch CoM sang trái
#
#   Phase 2:
#
#       Left Single Support
#       nhấc chân phải lên
#
#   Phase 3:
#
#       Left Single Support
#       chân phải giữ yên
#
#   Phase 4:
#
#       Left Single Support
#       chân phải dao động chậm
#
# ============================================================


# ============================================================
# SIMULATION / SOLVER PARAMETERS
# ============================================================

DT_LOW = 0.0005

QP_MAX_ITER = 10000

QP_CONSTRAINT_TOLERANCE = 5e-4

PRINT_EVERY_N_MPC = 10


# ============================================================
# TEST TIMING
# ============================================================

# ------------------------------------------------------------
# Phase 1
#
# Double Support
#
# Shift CoM toward left foot.
# ------------------------------------------------------------

T_SHIFT_END = 2.5


# ------------------------------------------------------------
# Phase 2
#
# Left Single Support
#
# Slowly lift / tuck right leg.
# ------------------------------------------------------------

T_LIFT_END = 3.0


# ------------------------------------------------------------
# Phase 3
#
# Left Single Support
#
# Right leg lifted and held static.
# ------------------------------------------------------------

T_STATIC_END = 4.0


# ------------------------------------------------------------
# Phase 4
#
# Left Single Support
#
# Right leg lifted and moved slowly.
# ------------------------------------------------------------

T_MOVE_END = 5.0


SIM_DURATION = T_MOVE_END


# ============================================================
# ANALYSIS WINDOWS
# ============================================================
#
# Bỏ qua 0.2 s đầu mỗi phase đo
# để tránh transition transient.
# ============================================================

ANALYSIS_SETTLE_TIME = 0.2


STATIC_ANALYSIS_START = (
    T_LIFT_END
    +
    ANALYSIS_SETTLE_TIME
)


STATIC_ANALYSIS_END = (
    T_STATIC_END
)


MOVING_ANALYSIS_START = (
    T_STATIC_END
    +
    ANALYSIS_SETTLE_TIME
)


MOVING_ANALYSIS_END = (
    T_MOVE_END
)


# ============================================================
# COM SHIFT COMMAND
# ============================================================
#
# Home:
#
#     y_CoM ≈ 0
#
#     y_L6 ≈ +0.0907 m
#
#
# Với:
#
#     vy_ref = 0.028 m/s
#
# trong:
#
#     2.5 s
#
# thì reference dịch:
#
#     0.028 * 2.5
#
#       =
#
#     0.070 m
#
# về phía chân trái.
# ============================================================

SHIFT_VY = 0.028


V_COM_SHIFT = np.array(
    [
        0.0,
        SHIFT_VY,
        0.0
    ],
    dtype=float
)


V_COM_HOLD = np.zeros(
    3,
    dtype=float
)


# ============================================================
# RIGHT LEG LIFT CONFIGURATION
# ============================================================
#
# Home right leg:
#
#     [
#         0,
#         0,
#         -30,
#         +60,
#         -30
#     ] deg
#
#
# Lift / tuck target:
#
#     [
#         0,
#         0,
#         -44,
#         +88,
#         -44
#     ] deg
#
#
# Delta:
#
#     [
#         0,
#         0,
#         -14,
#         +28,
#         -14
#     ] deg
#
#
# Tổng:
#
#     hip_pitch + knee + ankle
#
# gần như giữ không đổi,
# nhưng chân được co ngắn lại.
# ============================================================

Q_R_LIFT_DELTA_DEG = np.array(
    [
        0.0,
        0.0,
        -14.0,
        +28.0,
        -14.0
    ],
    dtype=float
)


Q_R_LIFT_DELTA = np.deg2rad(
    Q_R_LIFT_DELTA_DEG
)


# ============================================================
# SLOW SWING-LEG MOTION
# ============================================================
#
# Chỉ dao động nhẹ:
#
#     hip yaw
#     hip roll
#
# để tạo:
#
#     q_dot != 0
#     q_ddot != 0
#
# nhưng không chủ động thay đổi chiều dài chân.
# ============================================================

MOVE_AMPLITUDE_DEG = np.array(
    [
        5.0,
        5.0,
        0.0,
        0.0,
        0.0
    ],
    dtype=float
)


MOVE_AMPLITUDE = np.deg2rad(
    MOVE_AMPLITUDE_DEG
)


MOVE_FREQUENCY_HZ = 0.5


# ============================================================
# XML
# ============================================================

XML_PATH = (
    REPO_ROOT
    /
    "xmls"
    /
    "scene_flat_terrian_v3.xml"
)


# ============================================================
# RIGHT LEG JOINT ORDER
# ============================================================

RIGHT_LEG_JOINT_NAMES = (
    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "R_ankle",
)


# ============================================================
# TIMING HELPER
# ============================================================

def validate_timing_multiple(
    larger_dt,
    smaller_dt,
    name
):
    """
    Kiểm tra larger_dt có phải bội số nguyên
    của smaller_dt hay không.
    """

    steps = int(
        round(
            larger_dt
            /
            smaller_dt
        )
    )


    if steps <= 0:

        raise ValueError(
            f"{name}: số step phải > 0."
        )


    if not np.isclose(
        steps * smaller_dt,
        larger_dt,
        atol=1e-12
    ):

        raise ValueError(
            f"{name}: {larger_dt} phải là bội số nguyên "
            f"của {smaller_dt}."
        )


    return steps


# ============================================================
# TEST PHASE
# ============================================================

def get_test_phase(
    control_time
):
    """
    Return current test phase.
    """

    if control_time < T_SHIFT_END:

        return "DS_SHIFT"


    if control_time < T_LIFT_END:

        return "LIFT_RIGHT"


    if control_time < T_STATIC_END:

        return "STATIC_SINGLE_SUPPORT"


    return "MOVING_SINGLE_SUPPORT"


# ============================================================
# CONTACT SCHEDULE
# ============================================================

def build_constant_contact_schedule(
    contact_state,
    horizon
):
    """
    Build constant contact schedule over MPC horizon.
    """

    contact_state = np.asarray(
        contact_state,
        dtype=int
    ).reshape(
        1,
        2
    )


    contact_schedule = np.tile(
        contact_state,
        (
            int(horizon),
            1
        )
    )


    return contact_schedule


def get_contact_schedule_for_phase(
    phase
):
    """
    Phase 1:
        Double Support

    Phase 2-4:
        Left Single Support
    """

    if phase == "DS_SHIFT":

        return build_constant_contact_schedule(
            DOUBLE_SUPPORT,
            N
        )


    return build_constant_contact_schedule(
        LEFT_SINGLE_SUPPORT,
        N
    )


# ============================================================
# COM VELOCITY COMMAND
# ============================================================

def get_com_velocity_command(
    phase
):
    """
    During DS shift:
        move CoM reference toward left.

    Afterwards:
        hold CoM reference.
    """

    if phase == "DS_SHIFT":

        return V_COM_SHIFT.copy()


    return V_COM_HOLD.copy()


# ============================================================
# RIGHT LEG JOINT STATE
# ============================================================

def get_right_leg_joint_state(
    model,
    data
):
    """
    Get current q and q_dot of right leg.
    """

    q = np.zeros(
        5,
        dtype=float
    )


    q_dot = np.zeros(
        5,
        dtype=float
    )


    for i, joint_name in enumerate(
        RIGHT_LEG_JOINT_NAMES
    ):

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
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


        q[
            i
        ] = data.qpos[
            qpos_address
        ]


        q_dot[
            i
        ] = data.qvel[
            dof_address
        ]


    return (
        q,
        q_dot
    )


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep01(
    s
):
    """
    Cubic smoothstep:

        3s^2 - 2s^3

    for:

        0 <= s <= 1
    """

    s = float(
        np.clip(
            s,
            0.0,
            1.0
        )
    )


    return (
        s * s
        *
        (
            3.0
            -
            2.0 * s
        )
    )


# ============================================================
# RIGHT LEG DESIRED CONFIGURATION
# ============================================================

def get_right_leg_desired_q(
    control_time,
    q_R_home,
    q_R_lift
):
    """
    Generate desired right-leg joint configuration.
    """

    phase = get_test_phase(
        control_time
    )


    # --------------------------------------------------------
    # Phase 1
    #
    # Keep right leg at home.
    # --------------------------------------------------------

    if phase == "DS_SHIFT":

        return q_R_home.copy()


    # --------------------------------------------------------
    # Phase 2
    #
    # Smooth transition:
    #
    #     q_home
    #
    #        ->
    #
    #     q_lift
    # --------------------------------------------------------

    if phase == "LIFT_RIGHT":

        normalized_time = (
            (
                control_time
                -
                T_SHIFT_END
            )
            /
            (
                T_LIFT_END
                -
                T_SHIFT_END
            )
        )


        alpha = smoothstep01(
            normalized_time
        )


        return (
            q_R_home
            +
            alpha
            *
            (
                q_R_lift
                -
                q_R_home
            )
        )


    # --------------------------------------------------------
    # Phase 3
    #
    # Hold right leg static.
    # --------------------------------------------------------

    if phase == "STATIC_SINGLE_SUPPORT":

        return q_R_lift.copy()


    # --------------------------------------------------------
    # Phase 4
    #
    # Slowly move swing leg.
    # --------------------------------------------------------

    motion_time = (
        control_time
        -
        T_STATIC_END
    )


    sine = np.sin(
        2.0
        *
        np.pi
        *
        MOVE_FREQUENCY_HZ
        *
        motion_time
    )


    return (
        q_R_lift
        +
        MOVE_AMPLITUDE
        *
        sine
    )


# ============================================================
# MUJOCO OBJECT HELPER
# ============================================================

def get_required_id(
    model,
    object_type,
    name
):
    """
    Get required MuJoCo object ID.
    """

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name
    )


    if object_id < 0:

        raise ValueError(
            f"Không tìm thấy MuJoCo object '{name}'."
        )


    return int(
        object_id
    )


# ============================================================
# ACTUAL CONTACT-WRENCH READER
# ============================================================

class ContactWrenchReader:
    """
    Reconstruct actual ground-contact wrenches.

    Output convention:

        World frame

    referenced at:

        L6
        R6

    with physical meaning:

        ground -> robot
    """

    def __init__(
        self,
        model
    ):

        self.floor_geom_id = get_required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor"
        )


        self.left_body_id = get_required_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "L5_ankle"
        )


        self.right_body_id = get_required_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "R5_ankle"
        )


        self.L6_site_id = get_required_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "L6"
        )


        self.R6_site_id = get_required_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "R6"
        )


    def read(
        self,
        model,
        data
    ):
        """
        Return:

            W_wrench_L
            W_wrench_R
            n_contact_L
            n_contact_R
        """

        # ----------------------------------------------------
        # Reference points
        # ----------------------------------------------------

        W_p_L6 = np.asarray(
            data.site_xpos[
                self.L6_site_id
            ],
            dtype=float
        ).reshape(
            3
        )


        W_p_R6 = np.asarray(
            data.site_xpos[
                self.R6_site_id
            ],
            dtype=float
        ).reshape(
            3
        )


        # ----------------------------------------------------
        # Accumulators
        # ----------------------------------------------------

        W_F_L = np.zeros(
            3,
            dtype=float
        )


        W_M_L = np.zeros(
            3,
            dtype=float
        )


        W_F_R = np.zeros(
            3,
            dtype=float
        )


        W_M_R = np.zeros(
            3,
            dtype=float
        )


        n_contact_L = 0
        n_contact_R = 0


        # ====================================================
        # LOOP ACTIVE CONTACTS
        # ====================================================

        for contact_id in range(
            int(
                data.ncon
            )
        ):

            contact = data.contact[
                contact_id
            ]


            # ------------------------------------------------
            # Ignore inactive contact constraints
            # ------------------------------------------------

            if int(
                contact.efc_address
            ) < 0:

                continue


            geom1 = int(
                contact.geom1
            )


            geom2 = int(
                contact.geom2
            )


            foot_side = None

            sign = None


            # ------------------------------------------------
            # floor = geom1
            # ------------------------------------------------

            if geom1 == self.floor_geom_id:

                other_body = int(
                    model.geom_bodyid[
                        geom2
                    ]
                )


                if other_body == self.left_body_id:

                    foot_side = "L"

                    sign = +1.0


                elif other_body == self.right_body_id:

                    foot_side = "R"

                    sign = +1.0


            # ------------------------------------------------
            # floor = geom2
            # ------------------------------------------------

            elif geom2 == self.floor_geom_id:

                other_body = int(
                    model.geom_bodyid[
                        geom1
                    ]
                )


                if other_body == self.left_body_id:

                    foot_side = "L"

                    sign = -1.0


                elif other_body == self.right_body_id:

                    foot_side = "R"

                    sign = -1.0


            # ------------------------------------------------
            # Not floor-foot contact
            # ------------------------------------------------

            if foot_side is None:

                continue


            # =================================================
            # CONTACT WRENCH IN CONTACT FRAME
            # =================================================

            contact_wrench = np.zeros(
                6,
                dtype=float
            )


            mujoco.mj_contactForce(
                model,
                data,
                contact_id,
                contact_wrench
            )


            # =================================================
            # CONTACT FRAME -> WORLD
            # =================================================

            contact_R_W = np.asarray(
                contact.frame,
                dtype=float
            ).reshape(
                3,
                3
            )


            W_F_contact = (
                sign
                *
                (
                    contact_R_W.T
                    @
                    contact_wrench[
                        0:3
                    ]
                )
            )


            W_M_contact = (
                sign
                *
                (
                    contact_R_W.T
                    @
                    contact_wrench[
                        3:6
                    ]
                )
            )


            W_p_contact = np.asarray(
                contact.pos,
                dtype=float
            ).reshape(
                3
            )


            # =================================================
            # SHIFT MOMENT TO L6 / R6
            # =================================================

            if foot_side == "L":

                W_M_at_L6 = (
                    W_M_contact
                    +
                    np.cross(
                        W_p_contact
                        -
                        W_p_L6,
                        W_F_contact
                    )
                )


                W_F_L += W_F_contact

                W_M_L += W_M_at_L6


                n_contact_L += 1


            else:

                W_M_at_R6 = (
                    W_M_contact
                    +
                    np.cross(
                        W_p_contact
                        -
                        W_p_R6,
                        W_F_contact
                    )
                )


                W_F_R += W_F_contact

                W_M_R += W_M_at_R6


                n_contact_R += 1


        # ====================================================
        # BUILD WRENCH VECTORS
        # ====================================================

        W_wrench_L = np.concatenate(
            [
                W_F_L,
                W_M_L
            ]
        )


        W_wrench_R = np.concatenate(
            [
                W_F_R,
                W_M_R
            ]
        )


        return (
            W_wrench_L,
            W_wrench_R,
            n_contact_L,
            n_contact_R
        )


# ============================================================
# METRIC BUFFER
# ============================================================

def make_metric_buffer():
    """
    Create metric storage.
    """

    return {

        "force_error_abs": [],

        "force_error_rel": [],

        "moment_error_abs": [],

        "swing_force_norm": [],

        "swing_Fz_abs": [],

        "right_qdot_norm": [],

        "com_position_error_norm": [],

    }


# ============================================================
# APPEND METRICS
# ============================================================

def append_metrics(
    buffer,
    W_left_des,
    W_left_actual,
    W_right_actual,
    right_q_dot,
    com_position_error
):
    """
    Store one diagnostic sample.
    """

    force_des = W_left_des[
        0:3
    ]


    force_error = (
        W_left_actual[
            0:3
        ]
        -
        force_des
    )


    moment_error = (
        W_left_actual[
            3:6
        ]
        -
        W_left_des[
            3:6
        ]
    )


    force_des_norm = float(
        np.linalg.norm(
            force_des
        )
    )


    force_error_abs = float(
        np.linalg.norm(
            force_error
        )
    )


    force_error_rel = (
        force_error_abs
        /
        max(
            force_des_norm,
            1e-9
        )
    )


    buffer[
        "force_error_abs"
    ].append(
        force_error_abs
    )


    buffer[
        "force_error_rel"
    ].append(
        force_error_rel
    )


    buffer[
        "moment_error_abs"
    ].append(
        float(
            np.linalg.norm(
                moment_error
            )
        )
    )


    buffer[
        "swing_force_norm"
    ].append(
        float(
            np.linalg.norm(
                W_right_actual[
                    0:3
                ]
            )
        )
    )


    buffer[
        "swing_Fz_abs"
    ].append(
        abs(
            float(
                W_right_actual[
                    2
                ]
            )
        )
    )


    buffer[
        "right_qdot_norm"
    ].append(
        float(
            np.linalg.norm(
                right_q_dot
            )
        )
    )


    buffer[
        "com_position_error_norm"
    ].append(
        float(
            np.linalg.norm(
                com_position_error
            )
        )
    )


# ============================================================
# METRIC SUMMARY
# ============================================================

def summarize_buffer(
    name,
    buffer
):
    """
    Print mean / median / max for one test phase.
    """

    print("\n")

    print("=" * 70)

    print(
        name
    )

    print("=" * 70)


    sample_count = len(
        buffer[
            "force_error_abs"
        ]
    )


    print(
        "samples =",
        sample_count
    )


    if sample_count == 0:

        print(
            "Không có sample trong cửa sổ phân tích."
        )

        return


    for key in (

        "force_error_abs",

        "force_error_rel",

        "moment_error_abs",

        "swing_force_norm",

        "swing_Fz_abs",

        "right_qdot_norm",

        "com_position_error_norm",

    ):

        values = np.asarray(
            buffer[
                key
            ],
            dtype=float
        )


        print(
            f"\n{key}:"
        )


        print(
            "  mean   =",
            float(
                np.mean(
                    values
                )
            )
        )


        print(
            "  median =",
            float(
                np.median(
                    values
                )
            )
        )


        print(
            "  max    =",
            float(
                np.max(
                    values
                )
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. INITIALIZE SIMULATION
    # ========================================================

    setup = initialize_simulation(

        xml_path=XML_PATH,

        horizon=N,

        dt_mpc=dt_mpc,

        dt_mpc_update=dt_mpc_update,

        qp_max_iter=QP_MAX_ITER,

        qp_constraint_tolerance=(
            QP_CONSTRAINT_TOLERANCE
        )
    )


    model = setup.model

    data = setup.data


    # ========================================================
    # 2. TIMING
    # ========================================================

    DT_SIM = float(
        model.opt.timestep
    )


    LOW_LEVEL_EVERY = (
        validate_timing_multiple(

            larger_dt=DT_LOW,

            smaller_dt=DT_SIM,

            name="DT_LOW / DT_SIM"
        )
    )


    MPC_EVERY = (
        validate_timing_multiple(

            larger_dt=dt_mpc_update,

            smaller_dt=DT_SIM,

            name="dt_mpc_update / DT_SIM"
        )
    )


    if (
        MPC_EVERY
        %
        LOW_LEVEL_EVERY
        !=
        0
    ):

        raise ValueError(
            "MPC tick phải trùng low-level tick."
        )


    # ========================================================
    # 3. INITIAL RIGHT-LEG CONFIGURATION
    # ========================================================

    (
        q_R_home,
        _

    ) = get_right_leg_joint_state(
        model,
        data
    )


    q_R_lift = (
        q_R_home
        +
        Q_R_LIFT_DELTA
    )


    # ========================================================
    # 4. CONTACT WRENCH READER
    # ========================================================

    contact_reader = ContactWrenchReader(
        model
    )


    # ========================================================
    # 5. HELD SIGNALS
    # ========================================================

    held_u0_MPC = np.zeros(
        12,
        dtype=float
    )


    tau_command = np.zeros(
        10,
        dtype=float
    )


    last_solver_info = None

    last_post_diagnostics = None


    # ========================================================
    # 6. METRIC BUFFERS
    # ========================================================

    static_metrics = (
        make_metric_buffer()
    )


    moving_metrics = (
        make_metric_buffer()
    )


    # ========================================================
    # 7. COUNTERS
    # ========================================================

    physics_step = 0

    mpc_iteration = 0

    low_level_iteration = 0


    # ========================================================
    # 8. INITIAL PRINT
    # ========================================================

    initial = get_current_state_mpc(
        model,
        data
    )


    print("\n")

    print("=" * 70)

    print(
        "SINGLE-SUPPORT LEG-DYNAMICS A/B TEST"
    )

    print("=" * 70)


    print(
        "DT_SIM =",
        DT_SIM,
        "s"
    )


    print(
        "DT_LOW =",
        DT_LOW,
        "s"
    )


    print(
        "dt_mpc_update =",
        dt_mpc_update,
        "s"
    )


    print(
        "dt_mpc =",
        dt_mpc,
        "s"
    )


    print(
        "\nInitial CoM =",
        initial.x[
            0:3
        ]
    )


    print(
        "Initial L6 =",
        initial.W_p_L6
    )


    print(
        "Initial R6 =",
        initial.W_p_R6
    )


    print(
        "\nq_R_home [rad] =",
        q_R_home
    )


    print(
        "q_R_home [deg] =",
        np.rad2deg(
            q_R_home
        )
    )


    print(
        "\nq_R_lift [rad] =",
        q_R_lift
    )


    print(
        "q_R_lift [deg] =",
        np.rad2deg(
            q_R_lift
        )
    )


    print(
        "\nPhase timing:"
    )


    print(
        "  0.0 ->",
        T_SHIFT_END,
        "s : DS_SHIFT"
    )


    print(
        " ",
        T_SHIFT_END,
        "->",
        T_LIFT_END,
        "s : LIFT_RIGHT"
    )


    print(
        " ",
        T_LIFT_END,
        "->",
        T_STATIC_END,
        "s : STATIC_SINGLE_SUPPORT"
    )


    print(
        " ",
        T_STATIC_END,
        "->",
        T_MOVE_END,
        "s : MOVING_SINGLE_SUPPORT"
    )


    print(
        "\nStatic analysis window =",
        STATIC_ANALYSIS_START,
        "to",
        STATIC_ANALYSIS_END,
        "s"
    )


    print(
        "Moving analysis window =",
        MOVING_ANALYSIS_START,
        "to",
        MOVING_ANALYSIS_END,
        "s"
    )


    # ========================================================
    # 9. MAIN SIMULATION LOOP
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:


        while (

            viewer.is_running()

            and

            data.time
            <
            SIM_DURATION

        ):


            # =================================================
            # CONTROL TIME
            # =================================================

            control_time = (
                physics_step
                *
                DT_SIM
            )


            phase = get_test_phase(
                control_time
            )


            is_mpc_tick = (

                physics_step
                %
                MPC_EVERY
                ==
                0
            )


            is_low_level_tick = (

                physics_step
                %
                LOW_LEVEL_EVERY
                ==
                0
            )


            current = None


            # =================================================
            # A. MPC UPDATE
            # =================================================

            if is_mpc_tick:


                # =============================================
                # A1. CURRENT STATE
                # =============================================

                current = get_current_state_mpc(
                    model,
                    data
                )


                x_k = (
                    current.x.copy()
                )


                # =============================================
                # A2. REFERENCE COMMAND
                # =============================================

                v_cmd = (
                    get_com_velocity_command(
                        phase
                    )
                )


                X_ref = (
                    setup.reference_generator.generate_reference(
                        v_cmd
                    )
                )


                # =============================================
                # A3. CONTACT SCHEDULE
                # =============================================

                contact_schedule = (
                    get_contact_schedule_for_phase(
                        phase
                    )
                )


                # =============================================
                # A4. SRBD MODEL
                # =============================================

                (
                    Phi,
                    Gamma

                ) = build_SRBD_model(

                    Ew_inverse=(
                        current.Ew_inverse
                    ),

                    mass_CoM=(
                        setup.mass_CoM
                    ),

                    W_inertia_CoM=(
                        setup.W_inertia_CoM_q0
                    ),

                    W_r_L=(
                        current.W_r_L
                    ),

                    W_r_R=(
                        current.W_r_R
                    ),

                    dt_mpc=(
                        dt_mpc
                    ),

                    horizon=N
                )


                # =============================================
                # A5. SOLVE QP
                # =============================================

                (
                    U_opt,
                    u0_opt,
                    solver_info,
                    post_diagnostics

                ) = build_and_solve_QP(

                    qp_solver=(
                        setup.qp_solver
                    ),

                    Phi=Phi,

                    Gamma=Gamma,

                    x_current=x_k,

                    X_ref=X_ref,

                    N=N,

                    L6_R_W=(
                        current.L6_IK_R_W
                    ),

                    R6_R_W=(
                        current.R6_IK_R_W
                    ),

                    contact_schedule=(
                        contact_schedule
                    ),

                    print_diagnostics=False
                )


                if u0_opt is None:

                    raise RuntimeError(
                        "MPC không trả về nghiệm hợp lệ."
                    )


                held_u0_MPC = np.asarray(
                    u0_opt,
                    dtype=float
                ).reshape(
                    12
                )


                last_solver_info = (
                    solver_info
                )


                last_post_diagnostics = (
                    post_diagnostics
                )


            # =================================================
            # B. LOW-LEVEL UPDATE
            # =================================================

            if is_low_level_tick:


                if current is None:

                    current = (
                        get_current_state_mpc(
                            model,
                            data
                        )
                    )


                # =============================================
                # B1. MPC WRENCH -> TORQUE
                # =============================================

                (
                    tau_from_MPC,
                    tau_L_stance,
                    tau_R_from_MPC

                ) = get_joint_torque_from_MPC_wrench(

                    model,

                    data,

                    held_u0_MPC
                )


                # =============================================
                # B2. DOUBLE SUPPORT SHIFT
                #
                # Both legs follow MPC wrench.
                # =============================================

                if phase == "DS_SHIFT":

                    tau_command = (
                        tau_from_MPC.copy()
                    )


                # =============================================
                # B3. LEFT SINGLE SUPPORT
                #
                # Left:
                #
                #     stance torque from MPC
                #
                # Right:
                #
                #     joint-space PD
                # =============================================

                else:

                    tau_L = np.asarray(
                        tau_L_stance,
                        dtype=float
                    ).reshape(
                        5
                    )


                    (
                        q_R,
                        q_dot_R

                    ) = get_right_leg_joint_state(
                        model,
                        data
                    )


                    q_R_des = (
                        get_right_leg_desired_q(

                            control_time,

                            q_R_home,

                            q_R_lift
                        )
                    )


                    tau_R_swing = (
                        compute_swing_joint_PD(

                            q_des=q_R_des,

                            q=q_R,

                            q_dot=q_dot_R,

                            Kp=KP_SWING,

                            Kd=KD_SWING
                        )
                    )


                    tau_command = np.concatenate(
                        [
                            tau_L,
                            tau_R_swing
                        ]
                    )


                # =============================================
                # B4. APPLY TORQUE
                # =============================================

                apply_joint_torque(

                    model,

                    data,

                    tau_command
                )


                low_level_iteration += 1


            # =================================================
            # C. MUJOCO PHYSICS STEP
            # =================================================

            mujoco.mj_step(
                model,
                data
            )


            physics_step += 1


            # =================================================
            # D. POST-STEP CONTACT DIAGNOSTIC
            # =================================================

            if is_mpc_tick:


                (
                    W_left_actual,
                    W_right_actual,
                    n_contact_L,
                    n_contact_R

                ) = contact_reader.read(
                    model,
                    data
                )


                W_left_des = (
                    held_u0_MPC[
                        0:6
                    ].copy()
                )


                W_right_des = (
                    held_u0_MPC[
                        6:12
                    ].copy()
                )


                post_state = (
                    get_current_state_mpc(
                        model,
                        data
                    )
                )


                (
                    q_R_post,
                    q_dot_R_post

                ) = get_right_leg_joint_state(
                    model,
                    data
                )


                com_position_error = (

                    post_state.x[
                        0:3
                    ]

                    -

                    setup.reference_generator.get_position_reference()
                )


                # =============================================
                # D1. STATIC ANALYSIS
                # =============================================

                if (

                    control_time
                    >=
                    STATIC_ANALYSIS_START

                    and

                    control_time
                    <
                    STATIC_ANALYSIS_END

                ):

                    append_metrics(

                        static_metrics,

                        W_left_des,

                        W_left_actual,

                        W_right_actual,

                        q_dot_R_post,

                        com_position_error
                    )


                # =============================================
                # D2. MOVING ANALYSIS
                # =============================================

                if (

                    control_time
                    >=
                    MOVING_ANALYSIS_START

                    and

                    control_time
                    <
                    MOVING_ANALYSIS_END

                ):

                    append_metrics(

                        moving_metrics,

                        W_left_des,

                        W_left_actual,

                        W_right_actual,

                        q_dot_R_post,

                        com_position_error
                    )


                # =============================================
                # D3. PERIODIC PRINT
                # =============================================

                if (

                    mpc_iteration
                    %
                    PRINT_EVERY_N_MPC
                    ==
                    0

                ):


                    force_error = (

                        W_left_actual[
                            0:3
                        ]

                        -

                        W_left_des[
                            0:3
                        ]
                    )


                    force_rel = (

                        np.linalg.norm(
                            force_error
                        )

                        /

                        max(
                            np.linalg.norm(
                                W_left_des[
                                    0:3
                                ]
                            ),
                            1e-9
                        )
                    )


                    print("\n")

                    print("-" * 70)


                    print(
                        "MPC iteration =",
                        mpc_iteration
                    )


                    print(
                        "control_time =",
                        f"{control_time:.4f}",
                        "s"
                    )


                    print(
                        "phase =",
                        phase
                    )


                    print(
                        "contact schedule first =",
                        get_contact_schedule_for_phase(
                            phase
                        )[0]
                    )


                    print(
                        "\nCoM =",
                        post_state.x[
                            0:3
                        ]
                    )


                    print(
                        "CoM ref =",
                        setup.reference_generator.get_position_reference()
                    )


                    print(
                        "Base Euler =",
                        post_state.x[
                            3:6
                        ]
                    )


                    print(
                        "\nLEFT desired [World @ L6] =",
                        W_left_des
                    )


                    print(
                        "LEFT actual  [World @ L6] =",
                        W_left_actual
                    )


                    print(
                        "LEFT force relative error =",
                        float(
                            force_rel
                        )
                    )


                    print(
                        "\nRIGHT desired [World @ R6] =",
                        W_right_des
                    )


                    print(
                        "RIGHT actual  [World @ R6] =",
                        W_right_actual
                    )


                    print(
                        "active contacts L/R =",
                        n_contact_L,
                        "/",
                        n_contact_R
                    )


                    print(
                        "right swing |F| =",
                        float(
                            np.linalg.norm(
                                W_right_actual[
                                    0:3
                                ]
                            )
                        ),
                        "N"
                    )


                    print(
                        "right q [deg] =",
                        np.rad2deg(
                            q_R_post
                        )
                    )


                    print(
                        "right |qdot| =",
                        float(
                            np.linalg.norm(
                                q_dot_R_post
                            )
                        ),
                        "rad/s"
                    )


                    print(
                        "tau_command =",
                        tau_command
                    )


                    if last_solver_info is not None:

                        print(
                            "OSQP status =",
                            last_solver_info[
                                "status"
                            ]
                        )


                        print(
                            "OSQP iterations =",
                            last_solver_info[
                                "iterations"
                            ]
                        )


                    if (
                        last_post_diagnostics
                        is not None
                    ):

                        print(
                            "constraint max violation =",
                            last_post_diagnostics[
                                "max_violation"
                            ]
                        )


                    print("-" * 70)


                mpc_iteration += 1


            # =================================================
            # E. VIEWER
            # =================================================

            if (

                physics_step
                %
                MPC_EVERY
                ==
                0

            ):

                viewer.sync()


            # =================================================
            # F. SAFETY STOP
            # =================================================

            if (

                physics_step
                %
                MPC_EVERY
                ==
                0

            ):

                safety_state = (
                    get_current_state_mpc(
                        model,
                        data
                    )
                )


                if (

                    safety_state.x[
                        2
                    ]
                    <
                    0.10

                    or

                    abs(
                        safety_state.x[
                            3
                        ]
                    )
                    >
                    1.0

                    or

                    abs(
                        safety_state.x[
                            4
                        ]
                    )
                    >
                    1.0

                ):

                    print("\n")

                    print("=" * 70)

                    print(
                        "SAFETY STOP"
                    )

                    print("=" * 70)


                    print(
                        "time =",
                        data.time
                    )


                    print(
                        "state =",
                        safety_state.x
                    )


                    break


    # ========================================================
    # 10. FINAL STATE
    # ========================================================

    final_state = (
        get_current_state_mpc(
            model,
            data
        )
    )


    print("\n")

    print("=" * 70)

    print(
        "TEST FINISHED"
    )

    print("=" * 70)


    print(
        "final simulation time =",
        data.time,
        "s"
    )


    print(
        "final CoM =",
        final_state.x[
            0:3
        ]
    )


    print(
        "final Euler =",
        final_state.x[
            3:6
        ]
    )


    # ========================================================
    # 11. STATIC SUMMARY
    # ========================================================

    summarize_buffer(

        "A. STATIC SINGLE SUPPORT",

        static_metrics
    )


    # ========================================================
    # 12. MOVING SUMMARY
    # ========================================================

    summarize_buffer(

        "B. MOVING SWING LEG",

        moving_metrics
    )


    # ========================================================
    # 13. DIRECT A/B COMPARISON
    # ========================================================

    if (

        len(
            static_metrics[
                "force_error_rel"
            ]
        )
        >
        0

        and

        len(
            moving_metrics[
                "force_error_rel"
            ]
        )
        >
        0

    ):


        static_force_rel_mean = float(
            np.mean(
                static_metrics[
                    "force_error_rel"
                ]
            )
        )


        moving_force_rel_mean = float(
            np.mean(
                moving_metrics[
                    "force_error_rel"
                ]
            )
        )


        static_moment_error_mean = float(
            np.mean(
                static_metrics[
                    "moment_error_abs"
                ]
            )
        )


        moving_moment_error_mean = float(
            np.mean(
                moving_metrics[
                    "moment_error_abs"
                ]
            )
        )


        static_qdot_mean = float(
            np.mean(
                static_metrics[
                    "right_qdot_norm"
                ]
            )
        )


        moving_qdot_mean = float(
            np.mean(
                moving_metrics[
                    "right_qdot_norm"
                ]
            )
        )


        static_swing_force_mean = float(
            np.mean(
                static_metrics[
                    "swing_force_norm"
                ]
            )
        )


        moving_swing_force_mean = float(
            np.mean(
                moving_metrics[
                    "swing_force_norm"
                ]
            )
        )


        print("\n")

        print("=" * 70)

        print(
            "DIRECT A/B COMPARISON"
        )

        print("=" * 70)


        print(
            "Mean LEFT force relative error:"
        )


        print(
            "  STATIC =",
            static_force_rel_mean
        )


        print(
            "  MOVING =",
            moving_force_rel_mean
        )


        print(
            "  ratio MOVING/STATIC =",
            (
                moving_force_rel_mean
                /
                max(
                    static_force_rel_mean,
                    1e-12
                )
            )
        )


        print(
            "\nMean LEFT moment absolute error [N.m]:"
        )


        print(
            "  STATIC =",
            static_moment_error_mean
        )


        print(
            "  MOVING =",
            moving_moment_error_mean
        )


        print(
            "\nMean RIGHT swing-leg |qdot| [rad/s]:"
        )


        print(
            "  STATIC =",
            static_qdot_mean
        )


        print(
            "  MOVING =",
            moving_qdot_mean
        )


        print(
            "\nMean RIGHT swing-foot contact |F| [N]:"
        )


        print(
            "  STATIC =",
            static_swing_force_mean
        )


        print(
            "  MOVING =",
            moving_swing_force_mean
        )


        print("\nInterpretation:")


        # ----------------------------------------------------
        # Swing foot still contacts ground
        # ----------------------------------------------------

        if (

            static_swing_force_mean
            >
            1.0

            or

            moving_swing_force_mean
            >
            1.0

        ):

            print(
                "  WARNING: swing foot vẫn chịu contact force đáng kể."
            )


            print(
                "  Chưa được dùng kết quả này để kết luận leg dynamics."
            )


            print(
                "  Cần tăng độ nhấc chân / kiểm tra swing control trước."
            )


        # ----------------------------------------------------
        # Moving leg clearly increases stance error
        # ----------------------------------------------------

        elif (

            moving_force_rel_mean
            >
            1.5
            *
            static_force_rel_mean

        ):

            print(
                "  Moving swing leg làm stance-wrench tracking xấu đi rõ rệt."
            )


            print(
                "  Đây là bằng chứng ủng hộ giả thuyết neglected leg dynamics."
            )


        # ----------------------------------------------------
        # No strong difference
        # ----------------------------------------------------

        else:

            print(
                "  Moving swing leg chưa làm error tăng đủ mạnh."
            )


            print(
                "  Không nên quy nguyên nhân chính cho swing-leg dynamics."
            )


            print(
                "  Khi đó cần kiểm tra single-support stance realization / "
                "contact mechanics / actuator dynamics tiếp."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()