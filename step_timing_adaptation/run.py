# step_timing_adaptation/run.py

from __future__ import annotations

from pathlib import Path

import time

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

ROOT_DIR = (
    CURRENT_DIR.parent
)

SCENE_XML = (
    ROOT_DIR
    / "xmls"
    / "scene_flat_terrain_torque.xml"
)


# ============================================================
# IMPORTS
# ============================================================

if __package__:

    from .dynamics_model import (
        WholeBodyDynamicsModel,
    )

    from .whole_body_qp import (
        DoubleSupportQPConfig,
        WholeBodyInverseDynamicsQP,
    )

else:

    from dynamics_model import (
        WholeBodyDynamicsModel,
    )

    from whole_body_qp import (
        DoubleSupportQPConfig,
        WholeBodyInverseDynamicsQP,
    )


# ============================================================
# MODEL OBJECT NAMES
# ============================================================

ROBOT_ROOT_BODY = "base"

LEFT_FOOT_SITE = "left_foot"

RIGHT_FOOT_SITE = "right_foot"

LEFT_FOOT_GEOM = "left_foot_bottom_tpu"

RIGHT_FOOT_GEOM = "right_foot_bottom_tpu"

FLOOR_GEOM = "floor"


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

SIMULATION_DURATION = 5.0

# MuJoCo:
#
# dt = 0.0005 s
# simulation frequency = 2000 Hz
#
# WBC:
#
# solve QP at 1000 Hz
# hold torque for 2 MuJoCo simulation steps.

CONTROL_FREQUENCY = 1000.0

SHOW_VIEWER = True

REALTIME_FACTOR = 1.0

VIEWER_REFRESH_FREQUENCY = 60.0

STATUS_PRINT_PERIOD = 0.5


# ============================================================
# CONTACT PARAMETERS
# ============================================================

FRICTION_COEFFICIENT = 0.6


# ============================================================
# COM HEIGHT TASK
# ============================================================

COM_HEIGHT_KP = 200.0

COM_HEIGHT_KD = 25.0


# ============================================================
# POSTURE TASK
# ============================================================

POSTURE_KP = 100.0

POSTURE_KD = 20.0


# ============================================================
# QP WEIGHTS
# ============================================================

POSTURE_WEIGHT = 1.0e4

FORCE_REGULARIZATION_WEIGHT = 1.0

MOMENT_REGULARIZATION_WEIGHT = 10.0

QACC_REGULARIZATION_WEIGHT = 1.0e-6

TORQUE_REGULARIZATION_WEIGHT = 1.0e-8

GLOBAL_REGULARIZATION_WEIGHT = 1.0e-10


# ============================================================
# COM Jdot * v NUMERICAL DERIVATIVE
# ============================================================

COM_JDOT_EPSILON = 1.0e-6


# ============================================================
# PASS / FAIL CRITERIA
# ============================================================

MAX_ALLOWED_COM_HEIGHT_ERROR = 0.010
# 10 mm

MAX_ALLOWED_BASE_TILT_DEG = 15.0

MAX_ALLOWED_FOOT_SLIP = 0.005
# 5 mm

MIN_BOTH_CONTACT_RATIO = 0.95

MAX_ALLOWED_TORQUE_UTILIZATION = 1.001


# ============================================================
# EMERGENCY ABORT
# ============================================================

EMERGENCY_COM_DROP = 0.050
# 50 mm

EMERGENCY_BASE_TILT_DEG = 45.0


# ============================================================
# NUMERICAL TOLERANCE
# ============================================================

TORQUE_BOUND_TOL = 1.0e-7


# ============================================================
# PRINT
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
)


def separator():

    print(
        "=" * 100
    )


# ============================================================
# MUJOCO OBJECT LOOKUP
# ============================================================

def require_object_id(
    model,
    object_type,
    object_name,
):

    object_id = (
        mujoco.mj_name2id(
            model,
            object_type,
            object_name,
        )
    )

    if object_id < 0:

        raise RuntimeError(
            f"MuJoCo object not found: "
            f"'{object_name}'"
        )

    return int(
        object_id
    )


# ============================================================
# ACTUATED QPOS INDICES
# ============================================================

def build_actuated_qpos_indices(
    model,
):

    """
    Map each direct motor actuator to the qpos index
    of its hinge joint.

    Current robot:

        14 motor actuators
        14 hinge joints
    """

    indices = []

    for actuator_id in range(
        model.nactuator
    ):

        joint_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if joint_id < 0:

            raise RuntimeError(
                f"Actuator {actuator_id} "
                "does not target a valid joint."
            )

        joint_type = int(
            model.jnt_type[
                joint_id
            ]
        )

        if (
            joint_type
            !=
            int(
                mujoco.mjtJoint.mjJNT_HINGE
            )
        ):

            raise RuntimeError(
                "Dynamic standing currently assumes "
                "all actuators drive hinge joints."
            )

        qpos_index = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        indices.append(
            qpos_index
        )

    return np.asarray(
        indices,
        dtype=int,
    )


# ============================================================
# BASE TILT
# ============================================================

def get_base_tilt_deg(
    data,
    base_body_id,
):

    """
    Angle between the base local +Z axis and world +Z.

    This gives one scalar measure of total roll/pitch tilt
    without depending on a particular Euler convention.
    """

    rotation = (
        data.xmat[
            base_body_id
        ]
        .reshape(
            3,
            3,
        )
    )

    base_z_world = (
        rotation[
            :,
            2
        ]
    )

    cos_tilt = float(
        np.clip(
            base_z_world[
                2
            ],
            -1.0,
            +1.0,
        )
    )

    tilt_rad = (
        np.arccos(
            cos_tilt
        )
    )

    return float(
        np.degrees(
            tilt_rad
        )
    )


# ============================================================
# ACTUAL MUJOCO CONTACT
# ============================================================

def has_geom_contact(
    data,
    geom_a,
    geom_b,
):

    """
    Check whether MuJoCo currently reports at least one
    collision contact between two geoms.

    IMPORTANT:

    This must be called after mj_step1(), because collision
    detection belongs to the current position-dependent
    pipeline.
    """

    for contact_id in range(
        data.ncon
    ):

        contact = (
            data.contact[
                contact_id
            ]
        )

        geom1 = int(
            contact.geom1
        )

        geom2 = int(
            contact.geom2
        )

        if (
            (
                geom1 == geom_a
                and
                geom2 == geom_b
            )
            or
            (
                geom1 == geom_b
                and
                geom2 == geom_a
            )
        ):

            return True

    return False


# ============================================================
# COM Jdot * v
# ============================================================

def compute_com_jdot_v(
    model,
    data,
    root_body_id,
    scratch_plus,
    scratch_minus,
    epsilon,
):

    """
    Compute:

        Jdot_com(q, v) * v

    using a central directional finite difference.

    Since:

        Jdot = dJ/dq * qdot,

    evaluate the CoM Jacobian at:

        q_plus  = q (+) epsilon * v

        q_minus = q (+) (-epsilon) * v

    where (+) is MuJoCo's manifold-aware integration.

    This is important because the floating base contains a
    quaternion and therefore nq != nv.

    The scratch MjData objects are completely separate from
    the real simulation MjData, so this computation cannot
    corrupt the current simulation pipeline.
    """

    qvel = (
        data.qvel.copy()
    )

    if (
        np.linalg.norm(
            qvel
        )
        <
        1.0e-12
    ):

        return np.zeros(
            3,
            dtype=float,
        )

    # ========================================================
    # PLUS CONFIGURATION
    # ========================================================

    q_plus = (
        data.qpos.copy()
    )

    mujoco.mj_integratePos(
        model,
        q_plus,
        qvel,
        +epsilon,
    )

    scratch_plus.qpos[:] = (
        q_plus
    )

    scratch_plus.qvel[:] = (
        qvel
    )

    scratch_plus.ctrl[:] = (
        0.0
    )

    # Scratch state only.
    #
    # mj_forward does NOT integrate.
    #
    # It makes all derived quantities in scratch_plus
    # consistent with q_plus and qvel.

    mujoco.mj_forward(
        model,
        scratch_plus,
    )

    J_plus = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    mujoco.mj_jacSubtreeCom(
        model,
        scratch_plus,
        J_plus,
        root_body_id,
    )

    # ========================================================
    # MINUS CONFIGURATION
    # ========================================================

    q_minus = (
        data.qpos.copy()
    )

    mujoco.mj_integratePos(
        model,
        q_minus,
        qvel,
        -epsilon,
    )

    scratch_minus.qpos[:] = (
        q_minus
    )

    scratch_minus.qvel[:] = (
        qvel
    )

    scratch_minus.ctrl[:] = (
        0.0
    )

    mujoco.mj_forward(
        model,
        scratch_minus,
    )

    J_minus = np.zeros(
        (
            3,
            model.nv,
        ),
        dtype=float,
    )

    mujoco.mj_jacSubtreeCom(
        model,
        scratch_minus,
        J_minus,
        root_body_id,
    )

    # ========================================================
    # CENTRAL DIRECTIONAL DERIVATIVE
    # ========================================================

    Jdot = (
        J_plus
        -
        J_minus
    ) / (
        2.0
        *
        epsilon
    )

    return (
        Jdot
        @
        qvel
    )


# ============================================================
# DYNAMIC STANDING LOOP
# ============================================================

def run_dynamic_standing(
    model,
    data,
    dynamics,
    controller,
    actuated_qpos_indices,
    posture_reference,
    com_height_reference,
    left_foot_reference,
    right_foot_reference,
    torque_lower,
    torque_upper,
    viewer=None,
):

    # ========================================================
    # TIMING
    # ========================================================

    dt = float(
        model.opt.timestep
    )

    simulation_frequency = (
        1.0
        /
        dt
    )

    control_period = (
        1.0
        /
        CONTROL_FREQUENCY
    )

    control_decimation = int(
        round(
            control_period
            /
            dt
        )
    )

    if control_decimation < 1:

        raise RuntimeError(
            "Requested control frequency is greater than "
            "the MuJoCo simulation frequency."
        )

    actual_control_period = (
        control_decimation
        *
        dt
    )

    actual_control_frequency = (
        1.0
        /
        actual_control_period
    )

    timing_error = abs(
        actual_control_period
        -
        control_period
    )

    if timing_error > 1.0e-12:

        raise RuntimeError(
            "The requested WBC frequency is not an integer "
            "decimation of the MuJoCo simulation frequency.\n"
            f"simulation dt       = {dt}\n"
            f"requested WBC dt    = {control_period}\n"
            f"realizable WBC dt   = {actual_control_period}"
        )

    print()

    print(
        f"simulation frequency = "
        f"{simulation_frequency:.1f} Hz"
    )

    print(
        f"requested WBC        = "
        f"{CONTROL_FREQUENCY:.1f} Hz"
    )

    print(
        f"actual WBC           = "
        f"{actual_control_frequency:.1f} Hz"
    )

    print(
        f"QP every             = "
        f"{control_decimation} MuJoCo steps"
    )

    # ========================================================
    # OBJECT IDS
    # ========================================================

    base_body_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            ROBOT_ROOT_BODY,
        )
    )

    left_site_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            LEFT_FOOT_SITE,
        )
    )

    right_site_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            RIGHT_FOOT_SITE,
        )
    )

    left_geom_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            LEFT_FOOT_GEOM,
        )
    )

    right_geom_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            RIGHT_FOOT_GEOM,
        )
    )

    floor_geom_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            FLOOR_GEOM,
        )
    )

    # ========================================================
    # SCRATCH DATA
    #
    # Used only for numerical Jdot_com * v.
    # They never participate in the actual simulation.
    # ========================================================

    scratch_plus = (
        mujoco.MjData(
            model
        )
    )

    scratch_minus = (
        mujoco.MjData(
            model
        )
    )

    # ========================================================
    # LOW-PRIORITY CONTACT-WRENCH REFERENCES
    # ========================================================

    initial_com = (
        dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )
    )

    robot_weight = (
        initial_com.mass
        *
        np.linalg.norm(
            model.opt.gravity
        )
    )

    nominal_vertical_force = (
        0.5
        *
        robot_weight
    )

    left_wrench_reference = np.array(
        [
            0.0,
            0.0,
            nominal_vertical_force,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    right_wrench_reference = (
        left_wrench_reference.copy()
    )

    # ========================================================
    # METRICS
    # ========================================================

    max_com_height_error = 0.0

    max_base_tilt_deg = 0.0

    max_left_slip = 0.0

    max_right_slip = 0.0

    max_torque_utilization = 0.0

    qp_solve_times = []

    qp_solutions = 0

    fresh_state_samples = 0

    both_contact_samples = 0

    # ========================================================
    # HELD CONTROL
    # ========================================================

    last_tau = np.zeros(
        model.nu,
        dtype=float,
    )

    data.ctrl[:] = (
        last_tau
    )

    # ========================================================
    # STATUS / VIEWER
    # ========================================================

    simulation_start_time = float(
        data.time
    )

    simulation_end_time = (
        simulation_start_time
        +
        SIMULATION_DURATION
    )

    next_status_time = (
        simulation_start_time
        +
        STATUS_PRINT_PERIOD
    )

    next_viewer_time = (
        simulation_start_time
    )

    viewer_period = (
        1.0
        /
        VIEWER_REFRESH_FREQUENCY
    )

    wall_start = (
        time.perf_counter()
    )

    step_index = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================
    #
    # CRITICAL MUJOCO PIPELINE:
    #
    #      state k
    #         |
    #         v
    #     mj_step1()
    #         |
    #         | fresh position/velocity-dependent quantities
    #         |
    #         +--> monitoring
    #         |
    #         +--> WBC / QP
    #         |
    #         v
    #     data.ctrl = tau
    #         |
    #         v
    #     mj_step2()
    #         |
    #         v
    #      state k+1
    #
    #
    # No derived quantities are read after mj_step2().
    # They are refreshed by mj_step1() at the start of
    # the next iteration.
    # ========================================================

    while (
        data.time
        <
        simulation_end_time
        -
        0.5
        *
        dt
    ):

        if (
            viewer is not None
            and
            not viewer.is_running()
        ):

            break

        # ====================================================
        # MUJOCO STEP 1
        # ====================================================
        #
        # Updates quantities depending on the CURRENT:
        #
        #   qpos
        #   qvel
        #
        # including:
        #
        #   kinematics
        #   body/site poses
        #   subtree CoM
        #   inertia
        #   collisions/contact geometry
        #   velocity-dependent quantities
        #   qfrc_bias
        #   qfrc_passive
        #
        # It does NOT integrate the state.
        # ====================================================

        mujoco.mj_step1(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        # ====================================================
        # EVERYTHING READ BELOW BELONGS TO THE SAME
        # FRESH CURRENT STATE.
        # ====================================================

        com_position = (
            data.subtree_com[
                base_body_id
            ]
            .copy()
        )

        left_position = (
            data.site_xpos[
                left_site_id
            ]
            .copy()
        )

        right_position = (
            data.site_xpos[
                right_site_id
            ]
            .copy()
        )

        base_tilt_deg = (
            get_base_tilt_deg(
                data,
                base_body_id,
            )
        )

        left_contact = (
            has_geom_contact(
                data,
                left_geom_id,
                floor_geom_id,
            )
        )

        right_contact = (
            has_geom_contact(
                data,
                right_geom_id,
                floor_geom_id,
            )
        )

        # ====================================================
        # CURRENT-STATE MONITORING
        # ====================================================

        com_height_error_abs = abs(
            float(
                com_position[
                    2
                ]
            )
            -
            com_height_reference
        )

        left_slip = float(
            np.linalg.norm(
                left_position[
                    0:2
                ]
                -
                left_foot_reference[
                    0:2
                ]
            )
        )

        right_slip = float(
            np.linalg.norm(
                right_position[
                    0:2
                ]
                -
                right_foot_reference[
                    0:2
                ]
            )
        )

        torque_limit_abs = np.maximum(
            np.abs(
                torque_lower
            ),
            np.abs(
                torque_upper
            ),
        )

        torque_utilization = float(
            np.max(
                np.abs(
                    last_tau
                )
                /
                torque_limit_abs
            )
        )

        max_com_height_error = max(
            max_com_height_error,
            com_height_error_abs,
        )

        max_base_tilt_deg = max(
            max_base_tilt_deg,
            base_tilt_deg,
        )

        max_left_slip = max(
            max_left_slip,
            left_slip,
        )

        max_right_slip = max(
            max_right_slip,
            right_slip,
        )

        max_torque_utilization = max(
            max_torque_utilization,
            torque_utilization,
        )

        fresh_state_samples += 1

        if (
            left_contact
            and
            right_contact
        ):

            both_contact_samples += 1

        # ====================================================
        # EMERGENCY CHECK
        # ====================================================

        if (
            com_position[
                2
            ]
            <
            com_height_reference
            -
            EMERGENCY_COM_DROP
        ):

            raise RuntimeError(
                "\nEmergency stop:\n"
                f"  t = {current_time:.6f} s\n"
                "  CoM dropped more than "
                f"{1000.0 * EMERGENCY_COM_DROP:.1f} mm."
            )

        if (
            base_tilt_deg
            >
            EMERGENCY_BASE_TILT_DEG
        ):

            raise RuntimeError(
                "\nEmergency stop:\n"
                f"  t = {current_time:.6f} s\n"
                f"  base tilt = "
                f"{base_tilt_deg:.3f} deg."
            )

        if not np.all(
            np.isfinite(
                data.qpos
            )
        ):

            raise RuntimeError(
                f"NaN/Inf in qpos at "
                f"t={current_time:.6f} s."
            )

        if not np.all(
            np.isfinite(
                data.qvel
            )
        ):

            raise RuntimeError(
                f"NaN/Inf in qvel at "
                f"t={current_time:.6f} s."
            )

        # ====================================================
        # 1 kHz WBC UPDATE
        # ====================================================

        if (
            step_index
            %
            control_decimation
            ==
            0
        ):

            # =================================================
            # DYNAMICS FROM CURRENT mj_step1 STATE
            # =================================================
            #
            # IMPORTANT:
            #
            # forward=False.
            #
            # Do NOT call mj_forward() here because mj_step1()
            # has already produced the current position- and
            # velocity-dependent dynamics quantities.
            # =================================================

            terms = (
                dynamics.compute(
                    data,
                    forward=False,
                )
            )

            left = (
                dynamics.get_site_kinematics(
                    data,
                    LEFT_FOOT_SITE,
                )
            )

            right = (
                dynamics.get_site_kinematics(
                    data,
                    RIGHT_FOOT_SITE,
                )
            )

            com = (
                dynamics.get_com_kinematics(
                    data,
                    ROBOT_ROOT_BODY,
                )
            )

            # =================================================
            # COM Jdot * v
            # =================================================

            com_jdot_v = (
                compute_com_jdot_v(
                    model=(
                        model
                    ),

                    data=(
                        data
                    ),

                    root_body_id=(
                        base_body_id
                    ),

                    scratch_plus=(
                        scratch_plus
                    ),

                    scratch_minus=(
                        scratch_minus
                    ),

                    epsilon=(
                        COM_JDOT_EPSILON
                    ),
                )
            )

            # =================================================
            # RANK 2:
            # CONSTANT COM HEIGHT
            # =================================================

            com_height_error = (
                com_height_reference
                -
                com.position[
                    2
                ]
            )

            com_vertical_velocity = (
                com.velocity[
                    2
                ]
            )

            desired_com_acceleration_z = (
                COM_HEIGHT_KP
                *
                com_height_error
                -
                COM_HEIGHT_KD
                *
                com_vertical_velocity
            )

            # =================================================
            # RANK 4:
            # POSTURE
            # =================================================

            q_actuated = (
                data.qpos[
                    actuated_qpos_indices
                ]
            )

            v_actuated = (
                data.qvel[
                    dynamics.actuated_dof_indices
                ]
            )

            posture_error = (
                posture_reference
                -
                q_actuated
            )

            desired_posture_acceleration = (
                POSTURE_KP
                *
                posture_error
                -
                POSTURE_KD
                *
                v_actuated
            )

            # =================================================
            # SOLVE INVERSE-DYNAMICS QP
            # =================================================

            qp_start = (
                time.perf_counter()
            )

            try:

                solution = (
                    controller.solve_double_support(
                        mass_matrix=(
                            terms.mass_matrix
                        ),

                        effective_bias=(
                            terms.effective_bias
                        ),

                        selection_matrix=(
                            terms.selection_matrix
                        ),

                        left_jacobian=(
                            left.jacobian
                        ),

                        left_jdot_v=(
                            left.jacobian_dot_velocity
                        ),

                        right_jacobian=(
                            right.jacobian
                        ),

                        right_jdot_v=(
                            right.jacobian_dot_velocity
                        ),

                        com_jacobian=(
                            com.jacobian
                        ),

                        com_jdot_v_z=(
                            com_jdot_v[
                                2
                            ]
                        ),

                        desired_com_acceleration_z=(
                            desired_com_acceleration_z
                        ),

                        desired_posture_acceleration=(
                            desired_posture_acceleration
                        ),

                        left_wrench_reference=(
                            left_wrench_reference
                        ),

                        right_wrench_reference=(
                            right_wrench_reference
                        ),

                        torque_lower=(
                            torque_lower
                        ),

                        torque_upper=(
                            torque_upper
                        ),
                    )
                )

            except Exception:

                print()

                separator()

                print(
                    "QP FAILURE"
                )

                separator()

                print()

                print(
                    f"time = "
                    f"{current_time:.6f} s"
                )

                print(
                    f"CoM height error = "
                    f"{1000.0 * com_height_error:.3f} mm"
                )

                print(
                    f"base tilt = "
                    f"{base_tilt_deg:.3f} deg"
                )

                print(
                    f"left contact  = "
                    f"{left_contact}"
                )

                print(
                    f"right contact = "
                    f"{right_contact}"
                )

                print()

                raise

            qp_elapsed = (
                time.perf_counter()
                -
                qp_start
            )

            qp_solve_times.append(
                qp_elapsed
            )

            qp_solutions += 1

            # =================================================
            # TORQUE
            # =================================================

            tau = (
                solution.torque.copy()
            )

            if np.any(
                tau
                <
                torque_lower
                -
                TORQUE_BOUND_TOL
            ):

                raise RuntimeError(
                    f"QP violated lower torque bound "
                    f"at t={current_time:.6f} s."
                )

            if np.any(
                tau
                >
                torque_upper
                +
                TORQUE_BOUND_TOL
            ):

                raise RuntimeError(
                    f"QP violated upper torque bound "
                    f"at t={current_time:.6f} s."
                )

            # Numerical guard only.
            #
            # Hard bounds are already inside the QP.

            last_tau = np.clip(
                tau,
                torque_lower,
                torque_upper,
            )

            # =================================================
            # CONTROL FOR CURRENT STATE
            # =================================================

            data.ctrl[:] = (
                last_tau
            )

        # ====================================================
        # STATUS PRINT
        # ====================================================

        if (
            current_time
            >=
            next_status_time
            -
            0.5
            *
            dt
        ):

            if qp_solve_times:

                latest_qp_ms = (
                    1000.0
                    *
                    qp_solve_times[
                        -1
                    ]
                )

            else:

                latest_qp_ms = (
                    0.0
                )

            print(
                f"t={current_time:5.2f} s"
                f" | z_err="
                f"{1000.0 * com_height_error_abs:7.3f} mm"
                f" | tilt="
                f"{base_tilt_deg:6.3f} deg"
                f" | slipL="
                f"{1000.0 * left_slip:7.3f} mm"
                f" | slipR="
                f"{1000.0 * right_slip:7.3f} mm"
                f" | contact="
                f"{int(left_contact)}/{int(right_contact)}"
                f" | tau="
                f"{100.0 * torque_utilization:6.2f}%"
                f" | QP="
                f"{latest_qp_ms:7.3f} ms"
            )

            next_status_time += (
                STATUS_PRINT_PERIOD
            )

        # ====================================================
        # VIEWER
        # ====================================================
        #
        # viewer.sync() is intentionally performed BEFORE
        # mj_step2().
        #
        # Therefore body/site transforms are still synchronized
        # with current qpos/qvel.
        # ====================================================

        if (
            viewer is not None
            and
            current_time
            >=
            next_viewer_time
            -
            0.5
            *
            dt
        ):

            viewer.sync()

            next_viewer_time += (
                viewer_period
            )

            # -----------------------------------------------
            # Optional real-time pacing
            # -----------------------------------------------

            simulation_elapsed = (
                current_time
                -
                simulation_start_time
            )

            target_wall_elapsed = (
                simulation_elapsed
                /
                REALTIME_FACTOR
            )

            actual_wall_elapsed = (
                time.perf_counter()
                -
                wall_start
            )

            sleep_time = (
                target_wall_elapsed
                -
                actual_wall_elapsed
            )

            if sleep_time > 0.0:

                time.sleep(
                    sleep_time
                )

        # ====================================================
        # MUJOCO STEP 2
        # ====================================================
        #
        # Uses current:
        #
        #     qpos_k
        #     qvel_k
        #     ctrl_k
        #
        # then computes:
        #
        #     actuation
        #     acceleration
        #     constraints
        #
        # and integrates:
        #
        #     state k -> state k+1
        #
        # CRITICAL:
        #
        # DO NOT read:
        #
        #     site_xpos
        #     xmat
        #     subtree_com
        #     contact
        #
        # after this call.
        #
        # They will be refreshed by mj_step1() at the beginning
        # of the next loop iteration.
        # ====================================================

        mujoco.mj_step2(
            model,
            data,
        )

        step_index += 1

    # ========================================================
    # FINAL STATE REFRESH
    # ========================================================
    #
    # The final call to mj_step2() has integrated qpos/qvel
    # to the final state.
    #
    # Refresh position/velocity-dependent derived quantities
    # once before inspecting the final state.
    #
    # This does NOT advance simulation time.
    # ========================================================

    mujoco.mj_step1(
        model,
        data,
    )

    final_time = float(
        data.time
    )

    final_com_position = (
        data.subtree_com[
            base_body_id
        ]
        .copy()
    )

    final_left_position = (
        data.site_xpos[
            left_site_id
        ]
        .copy()
    )

    final_right_position = (
        data.site_xpos[
            right_site_id
        ]
        .copy()
    )

    final_base_tilt_deg = (
        get_base_tilt_deg(
            data,
            base_body_id,
        )
    )

    final_left_contact = (
        has_geom_contact(
            data,
            left_geom_id,
            floor_geom_id,
        )
    )

    final_right_contact = (
        has_geom_contact(
            data,
            right_geom_id,
            floor_geom_id,
        )
    )

    final_com_height_error = abs(
        float(
            final_com_position[
                2
            ]
        )
        -
        com_height_reference
    )

    final_left_slip = float(
        np.linalg.norm(
            final_left_position[
                0:2
            ]
            -
            left_foot_reference[
                0:2
            ]
        )
    )

    final_right_slip = float(
        np.linalg.norm(
            final_right_position[
                0:2
            ]
            -
            right_foot_reference[
                0:2
            ]
        )
    )

    # Include final state in extrema.

    max_com_height_error = max(
        max_com_height_error,
        final_com_height_error,
    )

    max_base_tilt_deg = max(
        max_base_tilt_deg,
        final_base_tilt_deg,
    )

    max_left_slip = max(
        max_left_slip,
        final_left_slip,
    )

    max_right_slip = max(
        max_right_slip,
        final_right_slip,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    both_contact_ratio = (
        both_contact_samples
        /
        max(
            fresh_state_samples,
            1,
        )
    )

    if qp_solve_times:

        qp_times = np.asarray(
            qp_solve_times,
            dtype=float,
        )

        average_qp_ms = (
            1000.0
            *
            float(
                np.mean(
                    qp_times
                )
            )
        )

        max_qp_ms = (
            1000.0
            *
            float(
                np.max(
                    qp_times
                )
            )
        )

    else:

        average_qp_ms = (
            0.0
        )

        max_qp_ms = (
            0.0
        )

    return {
        "simulation_time": (
            final_time
            -
            simulation_start_time
        ),

        "qp_solutions": (
            qp_solutions
        ),

        "fresh_state_samples": (
            fresh_state_samples
        ),

        "max_com_height_error": (
            max_com_height_error
        ),

        "max_base_tilt_deg": (
            max_base_tilt_deg
        ),

        "max_left_slip": (
            max_left_slip
        ),

        "max_right_slip": (
            max_right_slip
        ),

        "both_contact_ratio": (
            both_contact_ratio
        ),

        "max_torque_utilization": (
            max_torque_utilization
        ),

        "average_qp_ms": (
            average_qp_ms
        ),

        "max_qp_ms": (
            max_qp_ms
        ),

        "final_com_height_error": (
            final_com_height_error
        ),

        "final_base_tilt_deg": (
            final_base_tilt_deg
        ),

        "final_left_slip": (
            final_left_slip
        ),

        "final_right_slip": (
            final_right_slip
        ),

        "final_left_contact": (
            final_left_contact
        ),

        "final_right_contact": (
            final_right_contact
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    separator()

    print(
        "STEP TIMING ADAPTATION"
    )

    print(
        "STEP 3B - DYNAMIC DOUBLE-SUPPORT STANDING"
    )

    separator()

    print()

    print(
        f"MuJoCo version = "
        f"{mujoco.__version__}"
    )

    print(
        f"CasADi version  = "
        f"{ca.__version__}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    if not SCENE_XML.exists():

        raise FileNotFoundError(
            f"Torque scene not found:\n"
            f"{SCENE_XML}"
        )

    model = (
        mujoco.MjModel.from_xml_path(
            str(
                SCENE_XML
            )
        )
    )

    data = (
        mujoco.MjData(
            model
        )
    )

    print()

    print(
        "Model:"
    )

    print(
        f"  nq = {model.nq}"
    )

    print(
        f"  nv = {model.nv}"
    )

    print(
        f"  nu = {model.nu}"
    )

    print(
        f"  dt = "
        f"{model.opt.timestep:.6f} s"
    )

    print(
        f"  integrator = "
        f"{mujoco.mjtIntegrator(model.opt.integrator).name}"
    )

    # ========================================================
    # INTEGRATOR CHECK
    # ========================================================
    #
    # mj_step1 / mj_step2 is intended for single-step
    # integrators.
    #
    # Our torque XML uses implicitfast.
    # ========================================================

    if (
        model.opt.integrator
        ==
        mujoco.mjtIntegrator.mjINT_RK4
    ):

        raise RuntimeError(
            "mj_step1/mj_step2 must not be used here "
            "with RK4. Use Euler/implicit/implicitfast."
        )

    # ========================================================
    # LOAD HOME
    # ========================================================

    home_id = (
        require_object_id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    # --------------------------------------------------------
    # HOME keyframe contains ctrl values.
    #
    # Dynamic WBC must generate torque itself.
    # --------------------------------------------------------

    data.ctrl[:] = (
        0.0
    )

    data.qvel[:] = (
        0.0
    )

    # --------------------------------------------------------
    # Initial one-time initialization.
    #
    # This does NOT integrate.
    #
    # It is only used to establish initial references before
    # the dynamic loop begins.
    # --------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )

    # ========================================================
    # DYNAMICS BACKEND
    # ========================================================

    dynamics = (
        WholeBodyDynamicsModel(
            model
        )
    )

    # ========================================================
    # REQUIRED OBJECTS
    # ========================================================

    require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ROBOT_ROOT_BODY,
    )

    require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        LEFT_FOOT_SITE,
    )

    require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        RIGHT_FOOT_SITE,
    )

    require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        LEFT_FOOT_GEOM,
    )

    require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        RIGHT_FOOT_GEOM,
    )

    require_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        FLOOR_GEOM,
    )

    # ========================================================
    # ACTUATED POSITION MAPPING
    # ========================================================

    actuated_qpos_indices = (
        build_actuated_qpos_indices(
            model
        )
    )

    if actuated_qpos_indices.shape != (
        model.nu,
    ):

        raise RuntimeError(
            "Unexpected actuated qpos mapping."
        )

    # ========================================================
    # INITIAL REFERENCES
    # ========================================================

    posture_reference = (
        data.qpos[
            actuated_qpos_indices
        ]
        .copy()
    )

    initial_com = (
        dynamics.get_com_kinematics(
            data,
            ROBOT_ROOT_BODY,
        )
    )

    com_height_reference = float(
        initial_com.position[
            2
        ]
    )

    initial_left = (
        dynamics.get_site_kinematics(
            data,
            LEFT_FOOT_SITE,
        )
    )

    initial_right = (
        dynamics.get_site_kinematics(
            data,
            RIGHT_FOOT_SITE,
        )
    )

    left_foot_reference = (
        initial_left.position.copy()
    )

    right_foot_reference = (
        initial_right.position.copy()
    )

    # ========================================================
    # TORQUE LIMITS
    # ========================================================

    torque_lower = (
        model.actuator_ctrlrange[
            :,
            0,
        ]
        .copy()
    )

    torque_upper = (
        model.actuator_ctrlrange[
            :,
            1,
        ]
        .copy()
    )

    # ========================================================
    # QP CONFIG
    # ========================================================

    qp_config = (
        DoubleSupportQPConfig(
            friction_coefficient=(
                FRICTION_COEFFICIENT
            ),

            posture_weight=(
                POSTURE_WEIGHT
            ),

            force_regularization_weight=(
                FORCE_REGULARIZATION_WEIGHT
            ),

            moment_regularization_weight=(
                MOMENT_REGULARIZATION_WEIGHT
            ),

            qacc_regularization_weight=(
                QACC_REGULARIZATION_WEIGHT
            ),

            torque_regularization_weight=(
                TORQUE_REGULARIZATION_WEIGHT
            ),

            global_regularization_weight=(
                GLOBAL_REGULARIZATION_WEIGHT
            ),

            solver_name=(
                "qrqp"
            ),
        )
    )

    controller = (
        WholeBodyInverseDynamicsQP(
            nv=(
                model.nv
            ),

            nu=(
                model.nu
            ),

            actuated_dof_indices=(
                dynamics.actuated_dof_indices
            ),

            config=(
                qp_config
            ),
        )
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    separator()

    print(
        "INITIAL REFERENCES"
    )

    separator()

    print()

    print(
        f"CoM height reference = "
        f"{com_height_reference:.8f} m"
    )

    print(
        "left foot reference  =",
        left_foot_reference,
    )

    print(
        "right foot reference =",
        right_foot_reference,
    )

    print()

    print(
        f"CoM-height PD:"
        f" Kp={COM_HEIGHT_KP:.1f},"
        f" Kd={COM_HEIGHT_KD:.1f}"
    )

    print(
        f"Posture PD:"
        f" Kp={POSTURE_KP:.1f},"
        f" Kd={POSTURE_KD:.1f}"
    )

    print()

    print(
        "MuJoCo dynamic loop:"
    )

    print()

    print(
        "  mj_step1()"
    )

    print(
        "      -> fresh q, v, kinematics, contact, M, h, J"
    )

    print(
        "      -> inverse-dynamics QP"
    )

    print(
        "      -> data.ctrl = tau"
    )

    print(
        "  mj_step2()"
    )

    print(
        "      -> acceleration / constraint solve"
    )

    print(
        "      -> integrate to next state"
    )

    print()

    print(
        "No qpos overwrite."
    )

    print(
        "No qvel overwrite inside loop."
    )

    print(
        "No CoP control."
    )

    print(
        "No ZMP control."
    )

    print(
        "No horizontal CoM tracking."
    )

    separator()

    # ========================================================
    # RUN
    # ========================================================

    if SHOW_VIEWER:

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            result = (
                run_dynamic_standing(
                    model=(
                        model
                    ),

                    data=(
                        data
                    ),

                    dynamics=(
                        dynamics
                    ),

                    controller=(
                        controller
                    ),

                    actuated_qpos_indices=(
                        actuated_qpos_indices
                    ),

                    posture_reference=(
                        posture_reference
                    ),

                    com_height_reference=(
                        com_height_reference
                    ),

                    left_foot_reference=(
                        left_foot_reference
                    ),

                    right_foot_reference=(
                        right_foot_reference
                    ),

                    torque_lower=(
                        torque_lower
                    ),

                    torque_upper=(
                        torque_upper
                    ),

                    viewer=(
                        viewer
                    ),
                )
            )

    else:

        result = (
            run_dynamic_standing(
                model=(
                    model
                ),

                data=(
                    data
                ),

                dynamics=(
                    dynamics
                ),

                controller=(
                    controller
                ),

                actuated_qpos_indices=(
                    actuated_qpos_indices
                ),

                posture_reference=(
                    posture_reference
                ),

                com_height_reference=(
                    com_height_reference
                ),

                left_foot_reference=(
                    left_foot_reference
                ),

                right_foot_reference=(
                    right_foot_reference
                ),

                torque_lower=(
                    torque_lower
                ),

                torque_upper=(
                    torque_upper
                ),

                viewer=None,
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    separator()

    print(
        "DYNAMIC STANDING RESULT"
    )

    separator()

    print()

    print(
        f"simulation time        = "
        f"{result['simulation_time']:.3f} s"
    )

    print(
        f"fresh state samples    = "
        f"{result['fresh_state_samples']}"
    )

    print(
        f"QP solutions           = "
        f"{result['qp_solutions']}"
    )

    print()

    print(
        f"max CoM-height error   = "
        f"{1000.0 * result['max_com_height_error']:.3f} mm"
    )

    print(
        f"max base tilt          = "
        f"{result['max_base_tilt_deg']:.3f} deg"
    )

    print(
        f"max left-foot slip     = "
        f"{1000.0 * result['max_left_slip']:.3f} mm"
    )

    print(
        f"max right-foot slip    = "
        f"{1000.0 * result['max_right_slip']:.3f} mm"
    )

    print(
        f"double-contact ratio   = "
        f"{100.0 * result['both_contact_ratio']:.2f}%"
    )

    print(
        f"max torque utilization = "
        f"{100.0 * result['max_torque_utilization']:.2f}%"
    )

    print()

    print(
        f"mean QP solve time     = "
        f"{result['average_qp_ms']:.3f} ms"
    )

    print(
        f"max QP solve time      = "
        f"{result['max_qp_ms']:.3f} ms"
    )

    print()

    print(
        "Final fresh state:"
    )

    print(
        f"  CoM-height error = "
        f"{1000.0 * result['final_com_height_error']:.3f} mm"
    )

    print(
        f"  base tilt        = "
        f"{result['final_base_tilt_deg']:.3f} deg"
    )

    print(
        f"  left slip        = "
        f"{1000.0 * result['final_left_slip']:.3f} mm"
    )

    print(
        f"  right slip       = "
        f"{1000.0 * result['final_right_slip']:.3f} mm"
    )

    print(
        f"  left contact     = "
        f"{result['final_left_contact']}"
    )

    print(
        f"  right contact    = "
        f"{result['final_right_contact']}"
    )

    # ========================================================
    # PASS / FAIL
    # ========================================================

    failures = []

    if (
        result[
            "max_com_height_error"
        ]
        >
        MAX_ALLOWED_COM_HEIGHT_ERROR
    ):

        failures.append(
            "CoM-height error > 10 mm"
        )

    if (
        result[
            "max_base_tilt_deg"
        ]
        >
        MAX_ALLOWED_BASE_TILT_DEG
    ):

        failures.append(
            "base tilt > 15 deg"
        )

    if (
        result[
            "max_left_slip"
        ]
        >
        MAX_ALLOWED_FOOT_SLIP
    ):

        failures.append(
            "left-foot slip > 5 mm"
        )

    if (
        result[
            "max_right_slip"
        ]
        >
        MAX_ALLOWED_FOOT_SLIP
    ):

        failures.append(
            "right-foot slip > 5 mm"
        )

    if (
        result[
            "both_contact_ratio"
        ]
        <
        MIN_BOTH_CONTACT_RATIO
    ):

        failures.append(
            "double-support contact ratio < 95%"
        )

    if (
        result[
            "max_torque_utilization"
        ]
        >
        MAX_ALLOWED_TORQUE_UTILIZATION
    ):

        failures.append(
            "actuator torque limit violated"
        )

    if not (
        result[
            "final_left_contact"
        ]
        and
        result[
            "final_right_contact"
        ]
    ):

        failures.append(
            "both feet are not in contact at final state"
        )

    print()

    separator()

    if failures:

        print(
            "STEP 3B FAILED"
        )

        separator()

        print()

        for failure in failures:

            print(
                " -",
                failure,
            )

        raise RuntimeError(
            "Dynamic double-support standing "
            "did not satisfy the PASS criteria."
        )

    print(
        "STEP 3B PASSED"
    )

    separator()

    print()

    print(
        "Double-support standing was maintained "
        "using torque control only."
    )

    print()

    print(
        "Verified dynamic chain:"
    )

    print()

    print(
        "  current state"
    )

    print(
        "      -> mj_step1"
    )

    print(
        "      -> inverse-dynamics QP"
    )

    print(
        "      -> tau"
    )

    print(
        "      -> mj_step2"
    )

    print(
        "      -> next state"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()