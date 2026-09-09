from pathlib import Path
import time
import threading

import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin


# ============================================================
# IMPORTS
# ============================================================

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

    from .walking_visualization import (
        WalkingVisualizer,
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

    from walking_visualization import (
        WalkingVisualizer,
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
# USER STOP REQUEST
# ============================================================

stop_requested = (
    threading.Event()
)


def keyboard_callback(
    keycode,
):
    if keycode in (
        ord("F"),
        ord("f"),
    ):

        if not stop_requested.is_set():

            stop_requested.set()

            print()

            print(
                "F pressed -> graceful stop requested."
            )

            print(
                "The FSM will finish the current swing, "
                "close the stance, then enter final double support."
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
            f"Body '{body_name}' "
            "was not found."
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
# SETTLING
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

            stable_time = 0.0

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

            v_left_ref = (
                np.zeros(
                    3
                )
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

            v_right_ref = (
                np.zeros(
                    3
                )
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

        v_left_ref = (
            np.zeros(
                3
            )
        )

        v_right_ref = (
            np.zeros(
                3
            )
        )

        v_swing_ref = None


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
# IK
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

        else:

            support_position_ref = (
                p_right_ref
            )

            swing_position_ref = (
                p_left_ref
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


# ============================================================
# MAIN
# ============================================================

def main():

    stop_requested.clear()


    # ========================================================
    # MUJOCO
    # ========================================================

    separator()

    print(
        "LOAD MUJOCO"
    )

    separator()


    mj_model = (
        mujoco.MjModel.from_xml_path(
            str(
                SCENE_XML
            )
        )
    )

    mj_data = (
        mujoco.MjData(
            mj_model
        )
    )


    # ========================================================
    # SETTLING
    # ========================================================

    separator()

    print(
        "SETTLING"
    )

    separator()


    settle_info = (
        settle_robot(
            mj_model,
            mj_data,
        )
    )


    print(
        f"settled at t = "
        f"{settle_info['time']:.4f} s"
    )


    # ========================================================
    # PINOCCHIO
    # ========================================================

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


    # ========================================================
    # FSM
    # ========================================================

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


    # ========================================================
    # VISUALIZER
    # ========================================================

    visualizer = (
        WalkingVisualizer(
            mj_model
        )
    )

    visualizer.initialize(
        robot
    )


    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


    # ========================================================
    # INFO
    # ========================================================

    separator()

    print(
        "START KINEMATIC WALKING"
    )

    separator()


    print(
        "Press F -> graceful stop."
    )

    print(
        "Close viewer -> exit immediately."
    )

    print(
        "Visualization:"
    )

    print(
        "  green = left foot trail"
    )

    print(
        "  red   = right foot trail"
    )

    print(
        "  blue  = CoM trail"
    )

    print(
        "  orange/yellow = current support polygon"
    )

    print(
        "  dark red boxes = planned footsteps"
    )


    # ========================================================
    # LOOP DATA
    # ========================================================

    kinematic_time = 0.0

    iteration = 0

    completed_steps = 0

    previous_step_index = -1

    stop_forwarded_to_fsm = False

    finished_announced = False


    # ========================================================
    # VIEWER
    # ========================================================

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        show_right_ui=True,
        key_callback=(
            keyboard_callback
        ),
    ) as viewer:


        visualizer.configure_viewer(
            viewer
        )


        visualizer.update(
            viewer,
            robot,
            fsm.get_state(),
        )


        wall_start = (
            time.perf_counter()
        )


        while viewer.is_running():

            # =================================================
            # FORWARD F EVENT TO FSM
            # =================================================

            if (
                stop_requested.is_set()
                and
                not stop_forwarded_to_fsm
            ):

                fsm.request_stop()

                stop_forwarded_to_fsm = (
                    True
                )

                print()

                print(
                    "Stop request forwarded to WalkingFSM."
                )


            # =================================================
            # CURRENT STATE
            # =================================================

            state = (
                fsm.get_state()
            )


            # =================================================
            # FINISHED -> HOLD STANDING POSE
            # =================================================

            if state.finished:

                if not finished_announced:

                    separator()

                    print(
                        "GRACEFUL STOP COMPLETED"
                    )

                    separator()

                    print(
                        "FSM state = FINISHED"
                    )

                    print(
                        "Robot is holding the final standing pose."
                    )

                    print(
                        "Close the viewer to exit."
                    )

                    finished_announced = (
                        True
                    )


                visualizer.update(
                    viewer,
                    robot,
                    state,
                )

                time.sleep(
                    0.03
                )

                continue


            # =================================================
            # REGISTER CURRENT PLANNED STEP
            # =================================================

            visualizer.register_step(
                state,
                robot,
            )


            # =================================================
            # NEW STEP MESSAGE
            # =================================================

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


            # =================================================
            # REFERENCES
            # =================================================

            (
                p_left_ref,
                p_right_ref,
                p_com_ref,
                v_com_ref,
                v_swing_ref,
            ) = compute_references(
                state,
                COM_Y0,
            )


            # =================================================
            # IK
            # =================================================

            (
                qdot_full,
                diagnostics,
                Z,
            ) = solve_current_phase(
                robot,
                q_pin,
                state,
                p_left_ref,
                p_right_ref,
                p_com_ref,
                v_com_ref,
                v_swing_ref,
                trunk_rotation_ref,
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


            # =================================================
            # INTEGRATE
            # =================================================

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


            robot.update(
                q_pin
            )


            update_mujoco_from_pinocchio(
                robot,
                q_pin,
                mj_model,
                mj_data,
            )


            # =================================================
            # FSM
            # =================================================

            phase_before = (
                state.phase
            )


            state_after = (
                fsm.update(
                    DT
                )
            )


            # =================================================
            # LANDING
            # =================================================

            if (
                phase_before
                ==
                WalkingPhase.SINGLE_SUPPORT
                and
                state_after.phase
                !=
                WalkingPhase.SINGLE_SUPPORT
            ):

                completed_steps += 1


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
                        state.right_contact_position
                    )

                else:

                    swing_actual = (
                        p_right_actual
                    )

                    support_actual = (
                        p_left_actual
                    )

                    support_target = (
                        state.left_contact_position
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
                        robot,
                        trunk_rotation_ref,
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
                    f"type = "
                    f"{state.step_type.value}"
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


            # =================================================
            # TIME
            # =================================================

            iteration += 1

            kinematic_time += DT


            mj_data.time = (
                settle_info[
                    "time"
                ]
                +
                kinematic_time
            )


            # =================================================
            # VISUAL / REAL-TIME
            # =================================================

            if (
                iteration
                %
                VIEWER_SYNC_STEPS
                ==
                0
            ):

                visualizer.update(
                    viewer,
                    robot,
                    state_after,
                )


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


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    separator()

    print(
        "PROGRAM ENDED"
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


    print(
        f"kinematic walking time = "
        f"{kinematic_time:.3f} s"
    )

    print(
        f"completed swing steps = "
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()