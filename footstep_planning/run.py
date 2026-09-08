from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin


if __package__:
    from .walking_fsm import (
        WalkingFSM,
        WalkingPhase,
    )

    from .swing_trajectory import (
        compute_swing_trajectory,
    )

    from .com_reference import (
        compute_com_reference,
    )

    from .pinocchio_model import (
        PinocchioModel,
    )

    from .differential_ik import (
        TRUNK_FRAME,
        solve_single_support_ik,
        solve_double_support_ik,
    )

else:
    from walking_fsm import (
        WalkingFSM,
        WalkingPhase,
    )

    from swing_trajectory import (
        compute_swing_trajectory,
    )

    from com_reference import (
        compute_com_reference,
    )

    from pinocchio_model import (
        PinocchioModel,
    )

    from differential_ik import (
        TRUNK_FRAME,
        solve_single_support_ik,
        solve_double_support_ik,
    )


np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(
    __file__
).resolve().parent

ROOT_DIR = (
    CURRENT_DIR.parent
)

ROBOT_XML = (
    ROOT_DIR
    / "xmls"
    / "open_duck_mini_v2.xml"
)

SCENE_XML = (
    ROOT_DIR
    / "xmls"
    / "scene_flat_terrain.xml"
)


# ============================================================
# WALKING PARAMETERS
# ============================================================

DT = 0.0005

STEP_LENGTH = 0.04

FEET_SPACING = 0.16

SWING_HEIGHT = 0.04

COM_HEIGHT = 0.205

SINGLE_SUPPORT_DURATION = 0.18

DOUBLE_SUPPORT_DURATION = 0.09

FIRST_SWING_SIDE = "right"


# ============================================================
# IK / FEEDBACK PARAMETERS
# ============================================================

IK_DAMPING = 1e-8

IK_RCOND = 1e-10

SUPPORT_POSITION_KP = 25.0

SWING_POSITION_KP = 20.0

COM_POSITION_KP = 10.0

TRUNK_ORIENTATION_KP = 10.0


# ============================================================
# SETTLING PARAMETERS
# ============================================================

BASE_LIN_TOL = 5e-4

BASE_ANG_TOL = 1e-3

JOINT_VEL_TOL = 2.5e-3

STABLE_DURATION = 0.10

MIN_SETTLE_TIME = 0.50

MAX_SETTLE_TIME = 5.00


# ============================================================
# VIEWER
# ============================================================

VIEWER_FPS = 60.0

VIEWER_SYNC_STEPS = max(
    1,
    int(
        round(
            (
                1.0
                /
                VIEWER_FPS
            )
            /
            DT
        )
    ),
)


TRUNK_BODY_NAME = (
    "trunk_assembly"
)


# ============================================================
# HELPERS
# ============================================================

def separator():
    print(
        "=" * 90
    )


def rotation_matrix_to_pitch(
    R,
):
    R = np.asarray(
        R,
        dtype=float,
    )

    pitch = np.arctan2(
        -R[2, 0],

        np.sqrt(
            R[0, 0] ** 2
            +
            R[1, 0] ** 2
        ),
    )

    return float(
        pitch
    )


def get_mujoco_body_pitch(
    model,
    data,
    body_name,
):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Body '{body_name}' was not found."
        )

    R = (
        data.xmat[
            body_id
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )

    return rotation_matrix_to_pitch(
        R
    )


def get_trunk_local_y_error(
    robot,
    trunk_rotation_ref,
):
    _, R_actual = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )

    error_local = pin.log3(
        R_actual.T
        @
        trunk_rotation_ref
    )

    return float(
        error_local[1]
    )


# ============================================================
# SETTLE ROBOT
# ============================================================

def settle_robot(
    model,
    data,
):
    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id < 0:
        raise RuntimeError(
            "HOME keyframe was not found."
        )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    stable_time = 0.0

    settled = False

    max_steps = int(
        np.ceil(
            MAX_SETTLE_TIME
            /
            model.opt.timestep
        )
    )

    for _ in range(
        max_steps
    ):

        mujoco.mj_step(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        base_linear_speed = float(
            np.linalg.norm(
                data.qvel[
                    0:3
                ]
            )
        )

        base_angular_speed = float(
            np.linalg.norm(
                data.qvel[
                    3:6
                ]
            )
        )

        max_joint_speed = float(
            np.max(
                np.abs(
                    data.qvel[
                        6:
                    ]
                )
            )
        )

        stable_now = (
            current_time
            >=
            MIN_SETTLE_TIME

            and

            base_linear_speed
            <
            BASE_LIN_TOL

            and

            base_angular_speed
            <
            BASE_ANG_TOL

            and

            max_joint_speed
            <
            JOINT_VEL_TOL
        )

        if stable_now:

            stable_time += (
                model.opt.timestep
            )

        else:

            stable_time = (
                0.0
            )

        if (
            stable_time
            >=
            STABLE_DURATION
        ):

            settled = True

            break

    if not settled:

        raise RuntimeError(
            "Robot did not satisfy "
            "settling criterion."
        )

    return {
        "time":
            float(
                data.time
            ),

        "base_linear_speed":
            float(
                np.linalg.norm(
                    data.qvel[
                        0:3
                    ]
                )
            ),

        "base_angular_speed":
            float(
                np.linalg.norm(
                    data.qvel[
                        3:6
                    ]
                )
            ),

        "max_joint_speed":
            float(
                np.max(
                    np.abs(
                        data.qvel[
                            6:
                        ]
                    )
                )
            ),
    }


# ============================================================
# PINOCCHIO -> MUJOCO
# ============================================================

def update_mujoco_from_pinocchio(
    robot,
    q_pin,
    mj_model,
    mj_data,
):
    q_mj = (
        robot.pin_to_mujoco(
            q_pin
        )
    )

    mj_data.qpos[:] = (
        q_mj
    )

    mj_data.qvel[:] = (
        0.0
    )

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


# ============================================================
# REFERENCES
# ============================================================

def compute_references(
    state,
    com_y,
):
    if (
        state.phase
        ==
        WalkingPhase.SINGLE_SUPPORT
    ):

        (
            p_swing_ref,
            v_swing_ref,
        ) = compute_swing_trajectory(
            start_position=(
                state.swing_start
            ),

            target_position=(
                state.swing_target
            ),

            phase_time=(
                state.phase_time
            ),

            duration=(
                state.phase_duration
            ),

            swing_height=(
                SWING_HEIGHT
            ),
        )


        if (
            state.swing_side
            ==
            "right"
        ):

            p_left_ref = (
                state
                .left_contact_position
                .copy()
            )

            v_left_ref = np.zeros(
                3,
                dtype=float,
            )

            p_right_ref = (
                p_swing_ref.copy()
            )

            v_right_ref = (
                v_swing_ref.copy()
            )


        elif (
            state.swing_side
            ==
            "left"
        ):

            p_left_ref = (
                p_swing_ref.copy()
            )

            v_left_ref = (
                v_swing_ref.copy()
            )

            p_right_ref = (
                state
                .right_contact_position
                .copy()
            )

            v_right_ref = np.zeros(
                3,
                dtype=float,
            )


        else:

            raise RuntimeError(
                f"Invalid swing side: "
                f"{state.swing_side}"
            )


    elif state.phase in (
        WalkingPhase.INITIAL_DOUBLE_SUPPORT,
        WalkingPhase.DOUBLE_SUPPORT,
        WalkingPhase.FINAL_DOUBLE_SUPPORT,
    ):

        p_left_ref = (
            state
            .left_contact_position
            .copy()
        )

        p_right_ref = (
            state
            .right_contact_position
            .copy()
        )

        v_left_ref = np.zeros(
            3,
            dtype=float,
        )

        v_right_ref = np.zeros(
            3,
            dtype=float,
        )

        v_swing_ref = (
            None
        )


    else:

        raise RuntimeError(
            f"Cannot compute references "
            f"for phase "
            f"{state.phase}."
        )


    (
        p_com_ref,
        v_com_ref,
    ) = compute_com_reference(
        left_foot_position=(
            p_left_ref
        ),

        right_foot_position=(
            p_right_ref
        ),

        left_foot_velocity=(
            v_left_ref
        ),

        right_foot_velocity=(
            v_right_ref
        ),

        com_y=(
            com_y
        ),

        com_height=(
            COM_HEIGHT
        ),
    )


    return (
        p_left_ref,
        p_right_ref,

        p_com_ref,
        v_com_ref,

        v_swing_ref,
    )


# ============================================================
# IK DISPATCH
# ============================================================

def solve_current_phase(
    robot,
    q_pin,
    state,

    p_left_ref,
    p_right_ref,

    p_com_ref,
    v_com_ref,

    v_swing_ref,

    trunk_rotation_ref,
):
    if (
        state.phase
        ==
        WalkingPhase.SINGLE_SUPPORT
    ):

        if (
            state.support_side
            ==
            "left"
        ):

            support_position_ref = (
                p_left_ref
            )

            swing_position_ref = (
                p_right_ref
            )

        elif (
            state.support_side
            ==
            "right"
        ):

            support_position_ref = (
                p_right_ref
            )

            swing_position_ref = (
                p_left_ref
            )

        else:

            raise RuntimeError(
                f"Invalid support side: "
                f"{state.support_side}"
            )

        return solve_single_support_ik(
            robot=robot,

            q_pin=q_pin,

            support_side=(
                state.support_side
            ),

            support_position_ref=(
                support_position_ref
            ),

            swing_position_ref=(
                swing_position_ref
            ),

            swing_linear_velocity_ref=(
                v_swing_ref
            ),

            com_position_ref=(
                p_com_ref
            ),

            com_velocity_ref=(
                v_com_ref
            ),

            trunk_rotation_ref=(
                trunk_rotation_ref
            ),

            support_position_gain=(
                SUPPORT_POSITION_KP
            ),

            swing_position_gain=(
                SWING_POSITION_KP
            ),

            com_position_gain=(
                COM_POSITION_KP
            ),

            trunk_orientation_gain=(
                TRUNK_ORIENTATION_KP
            ),

            damping=(
                IK_DAMPING
            ),

            rcond=(
                IK_RCOND
            ),
        )


    if state.phase in (
        WalkingPhase.INITIAL_DOUBLE_SUPPORT,
        WalkingPhase.DOUBLE_SUPPORT,
        WalkingPhase.FINAL_DOUBLE_SUPPORT,
    ):

        return solve_double_support_ik(
            robot=robot,

            q_pin=q_pin,

            left_position_ref=(
                p_left_ref
            ),

            right_position_ref=(
                p_right_ref
            ),

            com_position_ref=(
                p_com_ref
            ),

            com_velocity_ref=(
                v_com_ref
            ),

            trunk_rotation_ref=(
                trunk_rotation_ref
            ),

            foot_position_gain=(
                SUPPORT_POSITION_KP
            ),

            com_position_gain=(
                COM_POSITION_KP
            ),

            trunk_orientation_gain=(
                TRUNK_ORIENTATION_KP
            ),

            damping=(
                IK_DAMPING
            ),

            rcond=(
                IK_RCOND
            ),
        )


    raise RuntimeError(
        f"Cannot solve phase: "
        f"{state.phase}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    separator()

    print(
        "LOAD MUJOCO"
    )

    separator()


    print(
        f"scene = "
        f"{SCENE_XML}"
    )


    mj_model = (
        mujoco.MjModel.from_xml_path(
            str(
                SCENE_XML
            )
        )
    )


    mj_data = mujoco.MjData(
        mj_model
    )


    print(
        f"timestep = "
        f"{mj_model.opt.timestep:.7f} s"
    )

    print(
        f"nq = "
        f"{mj_model.nq}"
    )

    print(
        f"nv = "
        f"{mj_model.nv}"
    )

    print(
        f"nu = "
        f"{mj_model.nu}"
    )


    if abs(
        mj_model.opt.timestep
        -
        DT
    ) > 1e-12:

        raise RuntimeError(
            "MuJoCo timestep "
            "does not match DT."
        )


    # ========================================================
    # SETTLING
    # ========================================================

    separator()

    print(
        "SETTLING"
    )

    separator()


    settle_info = settle_robot(
        mj_model,
        mj_data,
    )


    print(
        f"settled at t = "
        f"{settle_info['time']:.4f} s"
    )

    print(
        f"base linear speed = "
        f"{settle_info['base_linear_speed']:.6e} m/s"
    )

    print(
        f"base angular speed = "
        f"{settle_info['base_angular_speed']:.6e} rad/s"
    )

    print(
        f"max joint speed = "
        f"{settle_info['max_joint_speed']:.6e} rad/s"
    )


    # ========================================================
    # PINOCCHIO
    # ========================================================

    separator()

    print(
        "CREATE PINOCCHIO MODEL"
    )

    separator()


    robot = PinocchioModel(
        mjcf_path=(
            ROBOT_XML
        ),

        mujoco_model=(
            mj_model
        ),
    )


    q_pin = (
        robot.mujoco_to_pin(
            mj_data.qpos.copy()
        )
    )


    robot.update(
        q_pin
    )


    p_left_0, _ = (
        robot.get_left_foot_pose()
    )

    p_right_0, _ = (
        robot.get_right_foot_pose()
    )

    p_com_0 = (
        robot.get_com()
    )


    _, trunk_rotation_ref = (
        robot.get_frame_pose(
            TRUNK_FRAME
        )
    )


    COM_Y0 = float(
        p_com_0[1]
    )


    initial_trunk_pitch = (
        get_mujoco_body_pitch(
            mj_model,
            mj_data,
            TRUNK_BODY_NAME,
        )
    )


    initial_trunk_pitch_deg = float(
        np.degrees(
            initial_trunk_pitch
        )
    )


    print(
        "settled LEFT foot  =",
        p_left_0,
    )

    print(
        "settled RIGHT foot =",
        p_right_0,
    )

    print(
        "settled CoM        =",
        p_com_0,
    )

    print(
        f"CoM y reference    = "
        f"{COM_Y0:+.6f} m"
    )

    print(
        f"settled trunk pitch = "
        f"{initial_trunk_pitch_deg:+.4f} deg"
    )


    # ========================================================
    # FSM
    # ========================================================

    separator()

    print(
        "CREATE WALKING FSM"
    )

    separator()


    fsm = WalkingFSM(
        p_left_initial=(
            p_left_0
        ),

        p_right_initial=(
            p_right_0
        ),

        step_length=(
            STEP_LENGTH
        ),

        feet_spacing=(
            FEET_SPACING
        ),

        single_support_duration=(
            SINGLE_SUPPORT_DURATION
        ),

        double_support_duration=(
            DOUBLE_SUPPORT_DURATION
        ),

        first_swing_side=(
            FIRST_SWING_SIDE
        ),
    )


    print(
        "initial phase =",
        fsm.get_state().phase.value,
    )

    print(
        f"TRUNK_ORIENTATION_KP = "
        f"{TRUNK_ORIENTATION_KP:.1f} 1/s"
    )


    mj_data.qvel[:] = (
        0.0
    )

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


    # ========================================================
    # WALK
    # ========================================================

    separator()

    print(
        "START KINEMATIC WALKING"
    )

    separator()


    print(
        "Foot/CoM position feedback is ACTIVE."
    )

    print(
        "Trunk local-Y task is ACTIVE."
    )

    print(
        "Trunk reference = settled orientation."
    )

    print(
        "mj_step() is NOT called after settling."
    )


    kinematic_time = (
        0.0
    )

    iteration = (
        0
    )

    completed_steps = (
        0
    )

    previous_step_index = (
        -1
    )


    wall_start = (
        time.perf_counter()
    )


    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
    ) as viewer:


        viewer.sync()


        while viewer.is_running():

            state = (
                fsm.get_state()
            )


            if state.finished:

                separator()

                print(
                    "FSM FINISHED"
                )

                separator()

                break


            if (
                state.phase
                ==
                WalkingPhase.SINGLE_SUPPORT

                and

                state.step_index
                !=
                previous_step_index
            ):

                previous_step_index = (
                    state.step_index
                )


                separator()

                print(
                    f"START STEP "
                    f"#{state.step_index}"
                )

                print(
                    f"type    = "
                    f"{state.step_type.value}"
                )

                print(
                    f"support = "
                    f"{state.support_side}"
                )

                print(
                    f"swing   = "
                    f"{state.swing_side}"
                )

                print(
                    "start   =",
                    state.swing_start,
                )

                print(
                    "target  =",
                    state.swing_target,
                )

                print(
                    f"walking time = "
                    f"{kinematic_time:.3f} s"
                )


            (
                p_left_ref,
                p_right_ref,

                p_com_ref,
                v_com_ref,

                v_swing_ref,
            ) = compute_references(
                state=(
                    state
                ),

                com_y=(
                    COM_Y0
                ),
            )


            (
                qdot_full,
                diagnostics,
                Z,
            ) = solve_current_phase(
                robot=(
                    robot
                ),

                q_pin=(
                    q_pin
                ),

                state=(
                    state
                ),

                p_left_ref=(
                    p_left_ref
                ),

                p_right_ref=(
                    p_right_ref
                ),

                p_com_ref=(
                    p_com_ref
                ),

                v_com_ref=(
                    v_com_ref
                ),

                v_swing_ref=(
                    v_swing_ref
                ),

                trunk_rotation_ref=(
                    trunk_rotation_ref
                ),
            )


            if not np.all(
                np.isfinite(
                    qdot_full
                )
            ):

                raise RuntimeError(
                    "Differential IK "
                    "produced NaN/Inf."
                )


            q_pin = robot.integrate(
                q_pin=(
                    q_pin
                ),

                v_pin=(
                    qdot_full
                ),

                dt=(
                    DT
                ),
            )


            if not np.all(
                np.isfinite(
                    q_pin
                )
            ):

                raise RuntimeError(
                    "Pinocchio configuration "
                    "contains NaN/Inf."
                )


            robot.update(
                q_pin
            )


            update_mujoco_from_pinocchio(
                robot=(
                    robot
                ),

                q_pin=(
                    q_pin
                ),

                mj_model=(
                    mj_model
                ),

                mj_data=(
                    mj_data
                ),
            )


            phase_before = (
                state.phase
            )


            state_after = (
                fsm.update(
                    DT
                )
            )


            if (
                phase_before
                ==
                WalkingPhase.SINGLE_SUPPORT

                and

                state_after.phase
                !=
                WalkingPhase.SINGLE_SUPPORT
            ):

                completed_steps += (
                    1
                )


                p_left_actual, _ = (
                    robot.get_left_foot_pose()
                )

                p_right_actual, _ = (
                    robot.get_right_foot_pose()
                )

                p_com_actual = (
                    robot.get_com()
                )


                if (
                    state.swing_side
                    ==
                    "left"
                ):

                    swing_actual = (
                        p_left_actual
                    )

                    support_actual = (
                        p_right_actual
                    )

                    support_target = (
                        state
                        .right_contact_position
                    )

                else:

                    swing_actual = (
                        p_right_actual
                    )

                    support_actual = (
                        p_left_actual
                    )

                    support_target = (
                        state
                        .left_contact_position
                    )


                landing_error = float(
                    np.linalg.norm(
                        state.swing_target
                        -
                        swing_actual
                    )
                )


                support_error = float(
                    np.linalg.norm(
                        support_target
                        -
                        support_actual
                    )
                )


                p_com_target, _ = (
                    compute_com_reference(
                        left_foot_position=(
                            state_after
                            .left_contact_position
                        ),

                        right_foot_position=(
                            state_after
                            .right_contact_position
                        ),

                        left_foot_velocity=(
                            np.zeros(
                                3
                            )
                        ),

                        right_foot_velocity=(
                            np.zeros(
                                3
                            )
                        ),

                        com_y=(
                            COM_Y0
                        ),

                        com_height=(
                            COM_HEIGHT
                        ),
                    )
                )


                com_error = float(
                    np.linalg.norm(
                        p_com_target
                        -
                        p_com_actual
                    )
                )


                trunk_pitch = (
                    get_mujoco_body_pitch(
                        mj_model,
                        mj_data,
                        TRUNK_BODY_NAME,
                    )
                )


                trunk_pitch_deg = float(
                    np.degrees(
                        trunk_pitch
                    )
                )


                trunk_pitch_drift_deg = float(
                    np.degrees(
                        trunk_pitch
                        -
                        initial_trunk_pitch
                    )
                )


                trunk_local_y_error = (
                    get_trunk_local_y_error(
                        robot=(
                            robot
                        ),

                        trunk_rotation_ref=(
                            trunk_rotation_ref
                        ),
                    )
                )


                trunk_local_y_error_deg = float(
                    np.degrees(
                        trunk_local_y_error
                    )
                )


                final_nullity = int(
                    Z.shape[
                        1
                    ]
                )


                separator()

                print(
                    f"LANDING STEP "
                    f"#{state.step_index}"
                )

                print(
                    f"swing side = "
                    f"{state.swing_side}"
                )

                print(
                    f"landing error = "
                    f"{landing_error:.6e} m"
                )

                print(
                    f"support error = "
                    f"{support_error:.6e} m"
                )

                print(
                    f"CoM error = "
                    f"{com_error:.6e} m"
                )

                print(
                    f"trunk pitch = "
                    f"{trunk_pitch_deg:+.4f} deg"
                )

                print(
                    f"trunk pitch drift = "
                    f"{trunk_pitch_drift_deg:+.4f} deg"
                )

                print(
                    f"trunk local-Y error = "
                    f"{trunk_local_y_error_deg:+.4f} deg"
                )

                print(
                    f"final nullity = "
                    f"{final_nullity}"
                )

                print(
                    f"completed steps = "
                    f"{completed_steps}"
                )


            iteration += (
                1
            )


            kinematic_time += (
                DT
            )


            mj_data.time = (
                settle_info[
                    "time"
                ]
                +
                kinematic_time
            )


            if (
                iteration
                %
                VIEWER_SYNC_STEPS
                ==
                0
            ):

                viewer.sync()


                target_wall_time = (
                    wall_start
                    +
                    kinematic_time
                )


                remaining_time = (
                    target_wall_time
                    -
                    time.perf_counter()
                )


                if (
                    remaining_time
                    >
                    0.0
                ):

                    time.sleep(
                        remaining_time
                    )


    separator()

    print(
        "WALKING STOPPED"
    )

    separator()


    robot.update(
        q_pin
    )


    p_left_final, _ = (
        robot.get_left_foot_pose()
    )

    p_right_final, _ = (
        robot.get_right_foot_pose()
    )

    p_com_final = (
        robot.get_com()
    )


    final_trunk_pitch = (
        get_mujoco_body_pitch(
            mj_model,
            mj_data,
            TRUNK_BODY_NAME,
        )
    )


    final_trunk_pitch_deg = float(
        np.degrees(
            final_trunk_pitch
        )
    )


    final_trunk_pitch_drift_deg = float(
        np.degrees(
            final_trunk_pitch
            -
            initial_trunk_pitch
        )
    )


    final_trunk_local_y_error_deg = float(
        np.degrees(
            get_trunk_local_y_error(
                robot=(
                    robot
                ),

                trunk_rotation_ref=(
                    trunk_rotation_ref
                ),
            )
        )
    )


    print(
        f"kinematic walking time = "
        f"{kinematic_time:.3f} s"
    )

    print(
        f"completed steps = "
        f"{completed_steps}"
    )

    print(
        "LEFT foot final  =",
        p_left_final,
    )

    print(
        "RIGHT foot final =",
        p_right_final,
    )

    print(
        "CoM final        =",
        p_com_final,
    )

    print(
        f"initial trunk pitch = "
        f"{initial_trunk_pitch_deg:+.4f} deg"
    )

    print(
        f"final trunk pitch   = "
        f"{final_trunk_pitch_deg:+.4f} deg"
    )

    print(
        f"trunk pitch drift   = "
        f"{final_trunk_pitch_drift_deg:+.4f} deg"
    )

    print(
        f"trunk local-Y error = "
        f"{final_trunk_local_y_error_deg:+.4f} deg"
    )


if __name__ == "__main__":

    main()