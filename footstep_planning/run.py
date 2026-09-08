from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# PACKAGE / DIRECT EXECUTION IMPORTS
# ============================================================

try:
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
        solve_single_support_ik,
        solve_double_support_ik,
    )

except ImportError:

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

ROOT_DIR = CURRENT_DIR.parent

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
# DIFFERENTIAL IK PARAMETERS
# ============================================================

IK_DAMPING = 1e-8

IK_RCOND = 1e-10


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
# VIEWER PARAMETERS
# ============================================================

VIEWER_FPS = 60.0

VIEWER_DT = (
    1.0 / VIEWER_FPS
)

VIEWER_SYNC_STEPS = max(
    1,
    int(
        round(
            VIEWER_DT / DT
        )
    ),
)


# ============================================================
# HELPERS
# ============================================================

def separator():
    print("=" * 90)


# ============================================================
# SETTLE ROBOT
# ============================================================

def settle_robot(
    model,
    data,
):
    """
    Dynamically settle robot from HOME.

    mj_step() is used ONLY here.

    After settling, walking becomes purely kinematic:
        Pinocchio Differential IK
        -> pin.integrate()
        -> MuJoCo qpos
        -> mj_forward()
    """

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

    max_steps = int(
        np.ceil(
            MAX_SETTLE_TIME
            /
            model.opt.timestep
        )
    )

    settled = False

    for _ in range(max_steps):

        mujoco.mj_step(
            model,
            data,
        )

        current_time = float(
            data.time
        )

        base_linear_speed = np.linalg.norm(
            data.qvel[0:3]
        )

        base_angular_speed = np.linalg.norm(
            data.qvel[3:6]
        )

        max_joint_speed = np.max(
            np.abs(
                data.qvel[6:]
            )
        )

        stable_now = (
            current_time >= MIN_SETTLE_TIME
            and
            base_linear_speed < BASE_LIN_TOL
            and
            base_angular_speed < BASE_ANG_TOL
            and
            max_joint_speed < JOINT_VEL_TOL
        )

        if stable_now:

            stable_time += (
                model.opt.timestep
            )

        else:

            stable_time = 0.0

        if stable_time >= STABLE_DURATION:

            settled = True
            break

    if not settled:

        raise RuntimeError(
            "Robot did not satisfy settling criterion."
        )

    return {
        "time":
            float(
                data.time
            ),

        "base_linear_speed":
            float(
                np.linalg.norm(
                    data.qvel[0:3]
                )
            ),

        "base_angular_speed":
            float(
                np.linalg.norm(
                    data.qvel[3:6]
                )
            ),

        "max_joint_speed":
            float(
                np.max(
                    np.abs(
                        data.qvel[6:]
                    )
                )
            ),
    }


# ============================================================
# UPDATE MUJOCO FROM PINOCCHIO
# ============================================================

def update_mujoco_from_pinocchio(
    robot,
    q_pin,
    mj_model,
    mj_data,
):
    """
    Copy Pinocchio configuration into MuJoCo.

    No dynamics integration is performed.
    """

    q_mj = robot.pin_to_mujoco(
        q_pin
    )

    mj_data.qpos[:] = q_mj

    # This is kinematic playback.
    #
    # qvel is deliberately reset because MuJoCo dynamics
    # is not being integrated after settling.
    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


# ============================================================
# BUILD CURRENT TASK REFERENCES
# ============================================================

def compute_references(
    state,
    com_y,
):
    """
    Generate foot + CoM references from current FSM state.

    Returns:
        p_left_ref
        v_left_ref
        p_right_ref
        v_right_ref
        p_com_ref
        v_com_ref
        v_swing_ref
    """

    # ========================================================
    # SINGLE SUPPORT
    # ========================================================

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

        # ----------------------------------------------------
        # LEFT SUPPORT / RIGHT SWING
        # ----------------------------------------------------

        if state.swing_side == "right":

            p_left_ref = (
                state.left_contact_position
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

        # ----------------------------------------------------
        # RIGHT SUPPORT / LEFT SWING
        # ----------------------------------------------------

        elif state.swing_side == "left":

            p_left_ref = (
                p_swing_ref.copy()
            )

            v_left_ref = (
                v_swing_ref.copy()
            )

            p_right_ref = (
                state.right_contact_position
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

    # ========================================================
    # DOUBLE SUPPORT
    # ========================================================

    elif state.phase in (
        WalkingPhase.INITIAL_DOUBLE_SUPPORT,
        WalkingPhase.DOUBLE_SUPPORT,
        WalkingPhase.FINAL_DOUBLE_SUPPORT,
    ):

        p_left_ref = (
            state.left_contact_position
            .copy()
        )

        p_right_ref = (
            state.right_contact_position
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

        v_swing_ref = None

    else:

        raise RuntimeError(
            f"Cannot compute references "
            f"for phase {state.phase}."
        )


    # ========================================================
    # COM REFERENCE
    # ========================================================

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
        v_left_ref,
        p_right_ref,
        v_right_ref,
        p_com_ref,
        v_com_ref,
        v_swing_ref,
    )


# ============================================================
# SOLVE CURRENT PHASE
# ============================================================

def solve_current_phase(
    robot,
    q_pin,
    state,
    v_swing_ref,
    v_com_ref,
):
    """
    Solve current walking phase using hierarchical
    Differential IK.
    """

    # ========================================================
    # SINGLE SUPPORT
    # ========================================================

    if (
        state.phase
        ==
        WalkingPhase.SINGLE_SUPPORT
    ):

        (
            qdot_full,
            diagnostics,
            Z,
        ) = solve_single_support_ik(
            robot=robot,

            q_pin=q_pin,

            support_side=(
                state.support_side
            ),

            swing_linear_velocity_ref=(
                v_swing_ref
            ),

            com_velocity_ref=(
                v_com_ref
            ),

            damping=(
                IK_DAMPING
            ),

            rcond=(
                IK_RCOND
            ),
        )

        return (
            qdot_full,
            diagnostics,
            Z,
        )


    # ========================================================
    # DOUBLE SUPPORT
    # ========================================================

    if state.phase in (
        WalkingPhase.INITIAL_DOUBLE_SUPPORT,
        WalkingPhase.DOUBLE_SUPPORT,
        WalkingPhase.FINAL_DOUBLE_SUPPORT,
    ):

        (
            qdot_full,
            diagnostics,
            Z,
        ) = solve_double_support_ik(
            robot=robot,

            q_pin=q_pin,

            com_velocity_ref=(
                v_com_ref
            ),

            damping=(
                IK_DAMPING
            ),

            rcond=(
                IK_RCOND
            ),
        )

        return (
            qdot_full,
            diagnostics,
            Z,
        )


    raise RuntimeError(
        f"Cannot solve phase: "
        f"{state.phase}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD MUJOCO
    # ========================================================

    separator()
    print("LOAD MUJOCO")
    separator()

    print(
        f"scene = {SCENE_XML}"
    )

    mj_model = mujoco.MjModel.from_xml_path(
        str(SCENE_XML)
    )

    mj_data = mujoco.MjData(
        mj_model
    )

    print(
        f"timestep = "
        f"{mj_model.opt.timestep:.7f} s"
    )

    print(
        f"nq = {mj_model.nq}"
    )

    print(
        f"nv = {mj_model.nv}"
    )

    print(
        f"nu = {mj_model.nu}"
    )


    if abs(
        mj_model.opt.timestep
        -
        DT
    ) > 1e-12:

        raise RuntimeError(
            "MuJoCo timestep does not match DT."
        )


    # ========================================================
    # SETTLE
    # ========================================================

    separator()
    print("SETTLING")
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
    print("CREATE PINOCCHIO MODEL")
    separator()

    robot = PinocchioModel(
        mjcf_path=ROBOT_XML,
        mujoco_model=mj_model,
    )


    # IMPORTANT:
    #
    # Pinocchio starts from the SETTLED MuJoCo configuration.
    q_pin = robot.mujoco_to_pin(
        mj_data.qpos.copy()
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


    COM_Y0 = float(
        p_com_0[1]
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


    # ========================================================
    # FSM
    # ========================================================

    separator()
    print("CREATE WALKING FSM")
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

    print()

    print(
        f"STEP_LENGTH = "
        f"{STEP_LENGTH:.3f} m"
    )

    print(
        f"FEET_SPACING = "
        f"{FEET_SPACING:.3f} m"
    )

    print(
        f"SWING_HEIGHT = "
        f"{SWING_HEIGHT:.3f} m"
    )

    print(
        f"COM_HEIGHT = "
        f"{COM_HEIGHT:.3f} m"
    )

    print(
        f"T_SS = "
        f"{SINGLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"T_DS = "
        f"{DOUBLE_SUPPORT_DURATION:.3f} s"
    )

    print(
        f"IK dt = "
        f"{DT:.4f} s"
    )


    # ========================================================
    # RESET MUJOCO VELOCITIES FOR KINEMATIC EXECUTION
    # ========================================================

    mj_data.qvel[:] = 0.0

    mujoco.mj_forward(
        mj_model,
        mj_data,
    )


    # ========================================================
    # RUN
    # ========================================================

    separator()
    print("START KINEMATIC WALKING")
    separator()

    print(
        "Close the MuJoCo viewer to stop."
    )

    print(
        "Walking dynamics are NOT integrated."
    )

    print(
        "mj_step() is NOT called after settling."
    )

    print()


    kinematic_time = 0.0

    iteration = 0

    completed_steps = 0

    previous_phase = (
        fsm.get_state().phase
    )

    previous_step_index = -1


    # Used only to make viewer playback approximately real-time.
    wall_start = time.perf_counter()


    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
    ) as viewer:


        # ----------------------------------------------------
        # Initial viewer synchronization
        # ----------------------------------------------------

        viewer.sync()


        while viewer.is_running():

            # =================================================
            # CURRENT FSM STATE
            # =================================================

            state = (
                fsm.get_state()
            )


            if state.finished:

                separator()
                print("FSM FINISHED")
                separator()

                break


            # =================================================
            # PRINT NEW STEP
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

                print(
                    f"walking time = "
                    f"{kinematic_time:.3f} s"
                )


            # =================================================
            # REFERENCES
            # =================================================

            (
                p_left_ref,
                v_left_ref,
                p_right_ref,
                v_right_ref,
                p_com_ref,
                v_com_ref,
                v_swing_ref,
            ) = compute_references(
                state=state,
                com_y=COM_Y0,
            )


            # =================================================
            # DIFFERENTIAL IK
            # =================================================

            (
                qdot_full,
                diagnostics,
                Z,
            ) = solve_current_phase(
                robot=robot,
                q_pin=q_pin,
                state=state,
                v_swing_ref=v_swing_ref,
                v_com_ref=v_com_ref,
            )


            # =================================================
            # NUMERICAL SAFETY CHECK
            # =================================================

            if not np.all(
                np.isfinite(
                    qdot_full
                )
            ):

                raise RuntimeError(
                    "Differential IK produced NaN/Inf."
                )


            # =================================================
            # PINOCCHIO INTEGRATION
            # =================================================

            q_pin = robot.integrate(
                q_pin=q_pin,
                v_pin=qdot_full,
                dt=DT,
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


            # =================================================
            # UPDATE PINOCCHIO
            # =================================================

            robot.update(
                q_pin
            )


            # =================================================
            # COPY CONFIGURATION TO MUJOCO
            # =================================================

            update_mujoco_from_pinocchio(
                robot=robot,
                q_pin=q_pin,
                mj_model=mj_model,
                mj_data=mj_data,
            )


            # =================================================
            # ADVANCE WALKING FSM
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
            # LANDING MESSAGE
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


                if state.swing_side == "left":

                    swing_actual = (
                        p_left_actual
                    )

                else:

                    swing_actual = (
                        p_right_actual
                    )


                landing_error = np.linalg.norm(
                    state.swing_target
                    -
                    swing_actual
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
                    "target =",
                    state.swing_target,
                )

                print(
                    "actual =",
                    swing_actual,
                )

                print(
                    f"landing error = "
                    f"{landing_error:.6e} m"
                )

                print(
                    f"completed steps = "
                    f"{completed_steps}"
                )


            previous_phase = (
                state_after.phase
            )


            # =================================================
            # TIME
            # =================================================

            iteration += 1

            kinematic_time += DT


            # We do not call mj_step().
            #
            # This only updates the displayed MuJoCo time.
            mj_data.time = (
                settle_info["time"]
                +
                kinematic_time
            )


            # =================================================
            # VIEWER SYNC
            # =================================================

            if (
                iteration
                %
                VIEWER_SYNC_STEPS
                ==
                0
            ):

                viewer.sync()


                # ---------------------------------------------
                # Approximate real-time playback
                # ---------------------------------------------

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


                if remaining_time > 0.0:

                    time.sleep(
                        remaining_time
                    )


    # ========================================================
    # FINAL STATE
    # ========================================================

    separator()
    print("WALKING STOPPED")
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


    print(
        f"kinematic walking time = "
        f"{kinematic_time:.3f} s"
    )

    print(
        f"completed steps = "
        f"{completed_steps}"
    )

    print()

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()