from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ============================================================
# WALKING PHASE
# ============================================================

class WalkingPhase(Enum):

    INITIAL_DOUBLE_SUPPORT = "INITIAL_DOUBLE_SUPPORT"

    SINGLE_SUPPORT = "SINGLE_SUPPORT"

    DOUBLE_SUPPORT = "DOUBLE_SUPPORT"

    FINAL_DOUBLE_SUPPORT = "FINAL_DOUBLE_SUPPORT"

    FINISHED = "FINISHED"


# ============================================================
# STEP TYPE
# ============================================================

class StepType(Enum):

    START_HALF_STEP = "START_HALF_STEP"

    NORMAL_STEP = "NORMAL_STEP"

    CLOSING_STEP = "CLOSING_STEP"


# ============================================================
# STEP COMMAND
# ============================================================

@dataclass
class StepCommand:

    step_index: int

    step_type: StepType

    support_side: str

    swing_side: str

    start_position: np.ndarray

    target_position: np.ndarray


# ============================================================
# FSM OUTPUT
# ============================================================

@dataclass
class WalkingState:

    phase: WalkingPhase

    phase_time: float

    phase_duration: float

    phase_progress: float

    step_index: int

    step_type: Optional[StepType]

    support_side: Optional[str]

    swing_side: Optional[str]

    swing_start: Optional[np.ndarray]

    swing_target: Optional[np.ndarray]

    left_contact_position: np.ndarray

    right_contact_position: np.ndarray

    stop_requested: bool

    finished: bool


# ============================================================
# WALKING FSM
# ============================================================

class WalkingFSM:

    def __init__(
        self,
        p_left_initial,
        p_right_initial,
        step_length: float,
        feet_spacing: float,
        single_support_duration: float,
        double_support_duration: float,
        first_swing_side: str = "right",
        initial_double_support_duration=None,
    ):

        # ====================================================
        # CHECK INPUT
        # ====================================================

        if first_swing_side not in (
            "left",
            "right",
        ):

            raise ValueError(
                "first_swing_side must be "
                "'left' or 'right'."
            )

        if step_length <= 0.0:

            raise ValueError(
                "step_length must be positive."
            )

        if feet_spacing <= 0.0:

            raise ValueError(
                "feet_spacing must be positive."
            )

        if single_support_duration <= 0.0:

            raise ValueError(
                "single_support_duration "
                "must be positive."
            )

        if double_support_duration <= 0.0:

            raise ValueError(
                "double_support_duration "
                "must be positive."
            )

        # ----------------------------------------------------
        # Initial double support
        #
        # If not explicitly specified, preserve the old
        # behavior:
        #
        # initial DS = normal DS
        # ----------------------------------------------------

        if initial_double_support_duration is None:

            initial_double_support_duration = (
                double_support_duration
            )

        if initial_double_support_duration <= 0.0:

            raise ValueError(
                "initial_double_support_duration "
                "must be positive."
            )

        # ====================================================
        # INITIAL FEET
        # ====================================================

        self.p_left_initial = np.asarray(
            p_left_initial,
            dtype=float,
        ).copy()

        self.p_right_initial = np.asarray(
            p_right_initial,
            dtype=float,
        ).copy()

        if self.p_left_initial.shape != (3,):

            raise ValueError(
                "p_left_initial must have shape (3,)."
            )

        if self.p_right_initial.shape != (3,):

            raise ValueError(
                "p_right_initial must have shape (3,)."
            )

        # ====================================================
        # GAIT PARAMETERS
        # ====================================================

        self.step_length = float(
            step_length
        )

        self.feet_spacing = float(
            feet_spacing
        )

        self.ss_duration = float(
            single_support_duration
        )

        self.ds_duration = float(
            double_support_duration
        )

        self.initial_ds_duration = float(
            initial_double_support_duration
        )

        self.first_swing_side = (
            first_swing_side
        )

        # ====================================================
        # WALKING LANES
        # ====================================================

        self.y_center = 0.5 * (
            self.p_left_initial[1]
            +
            self.p_right_initial[1]
        )

        self.left_lane_y = (
            self.y_center
            +
            0.5 * self.feet_spacing
        )

        self.right_lane_y = (
            self.y_center
            -
            0.5 * self.feet_spacing
        )

        # Average settled foot-site height.
        self.nominal_foot_z = 0.5 * (
            self.p_left_initial[2]
            +
            self.p_right_initial[2]
        )

        # ====================================================
        # CURRENT PLANNED CONTACTS
        # ====================================================

        self.left_position = (
            self.p_left_initial.copy()
        )

        self.right_position = (
            self.p_right_initial.copy()
        )

        # ====================================================
        # FSM STATE
        # ====================================================

        self.phase = (
            WalkingPhase.INITIAL_DOUBLE_SUPPORT
        )

        self.phase_time = 0.0

        self.current_step = None

        self.step_counter = 0

        self.next_swing_side = (
            self.first_swing_side
        )

        # ====================================================
        # STOP LOGIC
        # ====================================================

        self.stop_requested = False

    # ========================================================
    # PUBLIC STOP REQUEST
    # ========================================================

    def request_stop(self):
        """
        Request a graceful stop.

        This does NOT immediately stop the robot.

        The FSM will:
            1. finish the current swing step,
            2. complete double support,
            3. execute a closing step,
            4. enter final double support,
            5. finish.
        """

        if (
            self.phase
            ==
            WalkingPhase.FINISHED
        ):
            return

        if (
            self.phase
            ==
            WalkingPhase.FINAL_DOUBLE_SUPPORT
        ):
            return

        self.stop_requested = True

    # ========================================================
    # UTILITIES
    # ========================================================

    @staticmethod
    def _opposite_side(
        side,
    ):

        if side == "left":
            return "right"

        if side == "right":
            return "left"

        raise ValueError(
            f"Unknown foot side: {side}"
        )

    # ========================================================
    # GET CURRENT FOOT POSITION
    # ========================================================

    def _get_foot_position(
        self,
        side,
    ):

        if side == "left":

            return (
                self.left_position.copy()
            )

        if side == "right":

            return (
                self.right_position.copy()
            )

        raise ValueError(
            f"Unknown foot side: {side}"
        )

    # ========================================================
    # COMPUTE STEP TARGET
    # ========================================================

    def _compute_step_target(
        self,
        step_type: StepType,
    ):
        """
        Compute the next swing-step geometry WITHOUT modifying
        the internal FSM state.

        This function is shared by:
            - the actual footstep planner,
            - the LIPM-MPC preview.

        Returns
        -------
        swing_side
        support_side
        swing_position
        target_position
        """

        swing_side = (
            self.next_swing_side
        )

        support_side = (
            self._opposite_side(
                swing_side
            )
        )

        swing_position = (
            self._get_foot_position(
                swing_side
            )
        )

        support_position = (
            self._get_foot_position(
                support_side
            )
        )

        # ====================================================
        # X TARGET
        # ====================================================

        if (
            step_type
            ==
            StepType.START_HALF_STEP
        ):

            # First transition:
            #
            # x_target =
            # x_support + STEP_LENGTH / 2
            #
            target_x = (
                support_position[0]
                +
                0.5 * self.step_length
            )

        elif (
            step_type
            ==
            StepType.NORMAL_STEP
        ):

            # Normal walking:
            #
            # x_target =
            # x_support + STEP_LENGTH
            #
            target_x = (
                support_position[0]
                +
                self.step_length
            )

        elif (
            step_type
            ==
            StepType.CLOSING_STEP
        ):

            # Graceful stop:
            #
            # Put the rear swing foot at
            # the same longitudinal position
            # as the support foot.
            #
            target_x = (
                support_position[0]
            )

        else:

            raise ValueError(
                f"Unknown step type: {step_type}"
            )

        # ====================================================
        # Y TARGET
        # ====================================================

        if swing_side == "left":

            target_y = (
                self.left_lane_y
            )

        else:

            target_y = (
                self.right_lane_y
            )

        # ====================================================
        # Z TARGET
        # ====================================================

        target_z = (
            self.nominal_foot_z
        )

        target_position = np.array(
            [
                target_x,
                target_y,
                target_z,
            ],
            dtype=float,
        )

        return (
            swing_side,
            support_side,
            swing_position,
            target_position,
        )

    # ========================================================
    # GET NEXT SWING TARGET FOR MPC
    # ========================================================

    def get_next_swing_target(
        self,
    ) -> Optional[np.ndarray]:
        """
        Return the current or next planned swing-foot target.

        This method does NOT modify the FSM.

        It is intended primarily for preview control such as
        LIPM-MPC, which needs the upcoming foothold even while
        the robot is still in double support.

        Returns
        -------
        np.ndarray, shape (3,)
            Current/upcoming swing-foot target.

        None
            No future swing is scheduled.
        """

        # ----------------------------------------------------
        # CURRENT SINGLE SUPPORT
        # ----------------------------------------------------

        if (
            self.phase
            ==
            WalkingPhase.SINGLE_SUPPORT
        ):

            if self.current_step is None:

                raise RuntimeError(
                    "SINGLE_SUPPORT requires "
                    "a current_step."
                )

            return (
                self.current_step
                .target_position
                .copy()
            )

        # ----------------------------------------------------
        # INITIAL DOUBLE SUPPORT
        # ----------------------------------------------------

        if (
            self.phase
            ==
            WalkingPhase.INITIAL_DOUBLE_SUPPORT
        ):

            # Stop requested before walking starts:
            # no future swing will occur.
            if self.stop_requested:

                return None

            (
                _,
                _,
                _,
                target_position,
            ) = self._compute_step_target(
                StepType.START_HALF_STEP
            )

            return (
                target_position.copy()
            )

        # ----------------------------------------------------
        # NORMAL DOUBLE SUPPORT
        # ----------------------------------------------------

        if (
            self.phase
            ==
            WalkingPhase.DOUBLE_SUPPORT
        ):

            # -----------------------------------------------
            # Graceful stop
            # -----------------------------------------------

            if self.stop_requested:

                longitudinal_error = abs(
                    self.left_position[0]
                    -
                    self.right_position[0]
                )

                # Feet already aligned:
                # no closing swing required.
                if (
                    longitudinal_error
                    <
                    1e-9
                ):

                    return None

                next_step_type = (
                    StepType.CLOSING_STEP
                )

            # -----------------------------------------------
            # Continue walking
            # -----------------------------------------------

            else:

                next_step_type = (
                    StepType.NORMAL_STEP
                )

            (
                _,
                _,
                _,
                target_position,
            ) = self._compute_step_target(
                next_step_type
            )

            return (
                target_position.copy()
            )

        # ----------------------------------------------------
        # FINAL DOUBLE SUPPORT / FINISHED
        # ----------------------------------------------------

        if self.phase in (
            WalkingPhase.FINAL_DOUBLE_SUPPORT,
            WalkingPhase.FINISHED,
        ):

            return None

        raise RuntimeError(
            f"Unknown walking phase: {self.phase}"
        )

    # ========================================================
    # CREATE STEP
    # ========================================================

    def _create_step(
        self,
        step_type: StepType,
    ):

        (
            swing_side,
            support_side,
            swing_position,
            target_position,
        ) = self._compute_step_target(
            step_type
        )

        command = StepCommand(
            step_index=self.step_counter,

            step_type=step_type,

            support_side=support_side,

            swing_side=swing_side,

            start_position=(
                swing_position.copy()
            ),

            target_position=(
                target_position.copy()
            ),
        )

        self.step_counter += 1

        return command

    # ========================================================
    # START A SINGLE-SUPPORT STEP
    # ========================================================

    def _start_step(
        self,
        step_type,
    ):

        self.current_step = (
            self._create_step(
                step_type
            )
        )

        self.phase = (
            WalkingPhase.SINGLE_SUPPORT
        )

        self.phase_time = 0.0

    # ========================================================
    # COMMIT LANDING
    # ========================================================

    def _commit_current_step(
        self,
    ):

        if self.current_step is None:

            raise RuntimeError(
                "No current step to commit."
            )

        if (
            self.current_step.swing_side
            ==
            "left"
        ):

            self.left_position = (
                self.current_step
                .target_position
                .copy()
            )

        else:

            self.right_position = (
                self.current_step
                .target_position
                .copy()
            )

        # After landing, the other foot
        # becomes the next swing foot.
        self.next_swing_side = (
            self._opposite_side(
                self.current_step.swing_side
            )
        )

    # ========================================================
    # CURRENT PHASE DURATION
    # ========================================================

    def _phase_duration(
        self,
    ):

        # ----------------------------------------------------
        # SINGLE SUPPORT
        # ----------------------------------------------------

        if (
            self.phase
            ==
            WalkingPhase.SINGLE_SUPPORT
        ):

            return (
                self.ss_duration
            )

        # ----------------------------------------------------
        # INITIAL DOUBLE SUPPORT
        #
        # This duration can be longer than normal DS to allow
        # the LIPM-MPC to perform the initial lateral weight
        # transfer more smoothly.
        # ----------------------------------------------------

        if (
            self.phase
            ==
            WalkingPhase.INITIAL_DOUBLE_SUPPORT
        ):

            return (
                self.initial_ds_duration
            )

        # ----------------------------------------------------
        # NORMAL / FINAL DOUBLE SUPPORT
        # ----------------------------------------------------

        if self.phase in (
            WalkingPhase.DOUBLE_SUPPORT,
            WalkingPhase.FINAL_DOUBLE_SUPPORT,
        ):

            return (
                self.ds_duration
            )

        # ----------------------------------------------------
        # FINISHED
        # ----------------------------------------------------

        if (
            self.phase
            ==
            WalkingPhase.FINISHED
        ):

            return np.inf

        raise RuntimeError(
            f"Unknown phase: {self.phase}"
        )

    # ========================================================
    # PHASE TRANSITION
    # ========================================================

    def _advance_phase(
        self,
    ):

        # ====================================================
        # INITIAL DOUBLE SUPPORT
        # ====================================================

        if (
            self.phase
            ==
            WalkingPhase.INITIAL_DOUBLE_SUPPORT
        ):

            # If stop requested before walking
            # actually starts, remain standing.
            if self.stop_requested:

                self.phase = (
                    WalkingPhase.FINISHED
                )

                self.phase_time = 0.0

                return

            # Begin with a HALF STEP.
            self._start_step(
                StepType.START_HALF_STEP
            )

            return

        # ====================================================
        # SINGLE SUPPORT -> LANDING
        # ====================================================

        if (
            self.phase
            ==
            WalkingPhase.SINGLE_SUPPORT
        ):

            completed_type = (
                self.current_step
                .step_type
            )

            self._commit_current_step()

            self.current_step = None

            self.phase_time = 0.0

            # ------------------------------------------------
            # Closing step completed
            # ------------------------------------------------

            if (
                completed_type
                ==
                StepType.CLOSING_STEP
            ):

                self.phase = (
                    WalkingPhase.FINAL_DOUBLE_SUPPORT
                )

                return

            # ------------------------------------------------
            # Normal / start step completed
            # ------------------------------------------------

            self.phase = (
                WalkingPhase.DOUBLE_SUPPORT
            )

            return

        # ====================================================
        # DOUBLE SUPPORT
        # ====================================================

        if (
            self.phase
            ==
            WalkingPhase.DOUBLE_SUPPORT
        ):

            self.phase_time = 0.0

            # ------------------------------------------------
            # Stop requested
            # ------------------------------------------------

            if self.stop_requested:

                longitudinal_error = abs(
                    self.left_position[0]
                    -
                    self.right_position[0]
                )

                if (
                    longitudinal_error
                    <
                    1e-9
                ):

                    self.phase = (
                        WalkingPhase.FINAL_DOUBLE_SUPPORT
                    )

                    return

                self._start_step(
                    StepType.CLOSING_STEP
                )

                return

            # ------------------------------------------------
            # Continue normal walking
            # ------------------------------------------------

            self._start_step(
                StepType.NORMAL_STEP
            )

            return

        # ====================================================
        # FINAL DOUBLE SUPPORT
        # ====================================================

        if (
            self.phase
            ==
            WalkingPhase.FINAL_DOUBLE_SUPPORT
        ):

            self.phase = (
                WalkingPhase.FINISHED
            )

            self.phase_time = 0.0

            return

        # ====================================================
        # FINISHED
        # ====================================================

        if (
            self.phase
            ==
            WalkingPhase.FINISHED
        ):

            return

        raise RuntimeError(
            f"Unknown phase: {self.phase}"
        )

    # ========================================================
    # UPDATE FSM
    # ========================================================

    def update(
        self,
        dt,
    ) -> WalkingState:
        """
        Advance FSM by dt seconds.

        This FSM is STATEFUL because it must react to
        asynchronous events such as request_stop().
        """

        dt = float(
            dt
        )

        if dt < 0.0:

            raise ValueError(
                "dt cannot be negative."
            )

        remaining_dt = dt

        tolerance = 1e-12

        while (
            remaining_dt
            >
            tolerance
            and
            self.phase
            !=
            WalkingPhase.FINISHED
        ):

            duration = (
                self._phase_duration()
            )

            time_left = (
                duration
                -
                self.phase_time
            )

            # ------------------------------------------------
            # Stay inside current phase
            # ------------------------------------------------

            if (
                remaining_dt
                <
                time_left
            ):

                self.phase_time += (
                    remaining_dt
                )

                remaining_dt = 0.0

            # ------------------------------------------------
            # Reach end of current phase
            # ------------------------------------------------

            else:

                self.phase_time = (
                    duration
                )

                remaining_dt -= (
                    time_left
                )

                self._advance_phase()

        return self.get_state()

    # ========================================================
    # GET FSM STATE
    # ========================================================

    def get_state(
        self,
    ) -> WalkingState:

        duration = (
            self._phase_duration()
        )

        if np.isfinite(
            duration
        ):

            if duration > 0.0:

                progress = (
                    self.phase_time
                    /
                    duration
                )

            else:

                progress = 1.0

            progress = np.clip(
                progress,
                0.0,
                1.0,
            )

        else:

            progress = 1.0

        # ====================================================
        # SINGLE SUPPORT
        # ====================================================

        if (
            self.phase
            ==
            WalkingPhase.SINGLE_SUPPORT
        ):

            step = (
                self.current_step
            )

            if step is None:

                raise RuntimeError(
                    "SINGLE_SUPPORT requires current_step."
                )

            return WalkingState(
                phase=self.phase,

                phase_time=(
                    self.phase_time
                ),

                phase_duration=(
                    duration
                ),

                phase_progress=(
                    progress
                ),

                step_index=(
                    step.step_index
                ),

                step_type=(
                    step.step_type
                ),

                support_side=(
                    step.support_side
                ),

                swing_side=(
                    step.swing_side
                ),

                swing_start=(
                    step.start_position.copy()
                ),

                swing_target=(
                    step.target_position.copy()
                ),

                left_contact_position=(
                    self.left_position.copy()
                ),

                right_contact_position=(
                    self.right_position.copy()
                ),

                stop_requested=(
                    self.stop_requested
                ),

                finished=False,
            )

        # ====================================================
        # DOUBLE SUPPORT / FINISHED
        # ====================================================

        return WalkingState(
            phase=self.phase,

            phase_time=(
                self.phase_time
            ),

            phase_duration=(
                duration
            ),

            phase_progress=(
                progress
            ),

            step_index=-1,

            step_type=None,

            support_side="both",

            swing_side=None,

            swing_start=None,

            swing_target=None,

            left_contact_position=(
                self.left_position.copy()
            ),

            right_contact_position=(
                self.right_position.copy()
            ),

            stop_requested=(
                self.stop_requested
            ),

            finished=(
                self.phase
                ==
                WalkingPhase.FINISHED
            ),
        )