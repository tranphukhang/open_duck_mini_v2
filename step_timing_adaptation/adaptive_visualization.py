# step_timing_adaptation/adaptive_visualization.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# REUSE EXISTING VISUALIZATION
# ============================================================

from footstep_planning.walking_fsm import (
    WalkingPhase,
)

from footstep_planning.walking_visualization import (
    WalkingVisualizer,
    PlannedFootstep,
    MAX_PLANNED_FOOTSTEPS,
    add_sphere,
    add_line,
)

from lipm_mpc.zmp_visualization import (
    ZMPVisualizationConfig,
    LIPMZMPVisualizer,
)


# ============================================================
# VISUAL HEIGHTS
#
# Small positive z values prevent z-fighting with the floor.
# ============================================================

COM_GROUND_Z = 0.012

DCM_GROUND_Z = 0.014

PLANNER_TARGET_Z = 0.016


# ============================================================
# VISUAL COLORS
# ============================================================

COM_GROUND_RGBA = np.array(
    [
        0.05,
        0.40,
        1.00,
        1.00,
    ],
    dtype=np.float32,
)

DCM_RGBA = np.array(
    [
        1.00,
        0.90,
        0.00,
        1.00,
    ],
    dtype=np.float32,
)

PLANNER_TARGET_RGBA = np.array(
    [
        0.00,
        1.00,
        0.85,
        1.00,
    ],
    dtype=np.float32,
)

COM_VERTICAL_RGBA = np.array(
    [
        0.10,
        0.40,
        1.00,
        0.55,
    ],
    dtype=np.float32,
)

COM_DCM_LINE_RGBA = np.array(
    [
        1.00,
        0.80,
        0.05,
        0.70,
    ],
    dtype=np.float32,
)


# ============================================================
# MARKER SIZES
# ============================================================

COM_GROUND_RADIUS = 0.0060

DCM_RADIUS = 0.0070

PLANNER_TARGET_RADIUS = 0.0060

COM_VERTICAL_LINE_WIDTH = 2.5

COM_DCM_LINE_WIDTH = 3.0


# ============================================================
# ZMP VISUALIZATION
#
# Same style used by lipm_mpc.
# ============================================================

ZMP_VISUALIZATION_CONFIG = (
    ZMPVisualizationConfig(
        show_current=True,
        show_trail=True,
        show_preview=True,

        current_z=0.010,
        trail_z=0.008,
        preview_z=0.007,

        current_radius=0.007,
        preview_radius=0.0028,

        trail_width=5.0,
        preview_width=2.5,

        trail_min_distance=5.0e-4,
        trail_max_points=180,

        preview_point_stride=2,

        # Current ZMP: magenta
        current_rgba=np.array(
            [
                1.00,
                0.00,
                1.00,
                1.00,
            ],
            dtype=np.float32,
        ),

        # ZMP history
        trail_rgba=np.array(
            [
                0.90,
                0.10,
                0.95,
                0.80,
            ],
            dtype=np.float32,
        ),

        # Future MPC ZMP points
        preview_rgba=np.array(
            [
                0.72,
                0.28,
                1.00,
                0.72,
            ],
            dtype=np.float32,
        ),

        # Future MPC ZMP line
        preview_line_rgba=np.array(
            [
                0.72,
                0.28,
                1.00,
                0.55,
            ],
            dtype=np.float32,
        ),
    )
)


# ============================================================
# WALKING VISUALIZER STATE ADAPTER
#
# WalkingVisualizer only needs:
#
#     state.phase.value
#     state.support_side
#
# for support-polygon drawing.
# ============================================================

@dataclass
class WalkingVisualizationState:

    phase: WalkingPhase

    support_side: str | None


# ============================================================
# ADAPTIVE WALKING VISUALIZER
# ============================================================

class AdaptiveWalkingVisualizer:
    """
    Visualization wrapper for Step Timing Adaptation.

    Reuses:

        WalkingVisualizer
            - left-foot trail
            - right-foot trail
            - actual CoM trail
            - actual CoM marker
            - support polygon
            - planned-foot footprint geometry

        LIPMZMPVisualizer
            - current LIPM ZMP
            - ZMP trail
            - future MPC ZMP preview

    Adds:

        - current adaptive planner foothold
        - LIPM CoM projection on ground
        - DCM on ground
        - CoM -> ground projection line
        - CoM projection -> DCM line
    """

    def __init__(
        self,
        mj_model,
        *,
        com_height: float,
        gravity: float,
    ) -> None:

        self.walking = (
            WalkingVisualizer(
                mj_model
            )
        )

        self.zmp = (
            LIPMZMPVisualizer(
                config=(
                    ZMP_VISUALIZATION_CONFIG
                ),

                com_height=(
                    com_height
                ),

                gravity=(
                    gravity
                ),
            )
        )

        self.com_height = float(
            com_height
        )

        self.gravity = float(
            gravity
        )

        self.omega = np.sqrt(
            self.gravity
            /
            self.com_height
        )

        self.com_world = None

        self.com_ground = None

        self.dcm_ground = None


    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(
        self,
        robot,
        x_state,
        y_state,
    ) -> None:

        self.walking.initialize(
            robot
        )

        self.zmp.initialize_from_states(
            x_state=(
                x_state
            ),

            y_state=(
                y_state
            ),
        )

        p_com = (
            robot.get_com()
        )

        self.com_world = (
            p_com.copy()
        )

        self.com_ground = np.array(
            [
                x_state[0],
                y_state[0],
                COM_GROUND_Z,
            ],
            dtype=float,
        )

        dcm_x = (
            x_state[0]
            +
            x_state[1]
            /
            self.omega
        )

        dcm_y = (
            y_state[0]
            +
            y_state[1]
            /
            self.omega
        )

        self.dcm_ground = np.array(
            [
                dcm_x,
                dcm_y,
                DCM_GROUND_Z,
            ],
            dtype=float,
        )


    # ========================================================
    # CAMERA
    # ========================================================

    def configure_viewer(
        self,
        viewer,
    ) -> None:

        self.walking.configure_viewer(
            viewer
        )


    # ========================================================
    # CURRENT LIPM STATE
    # ========================================================

    def update_lipm(
        self,
        *,
        com_position,
        com_acceleration,
        dcm_xy,
    ) -> None:

        p_com = np.asarray(
            com_position,
            dtype=float,
        ).reshape(
            3
        )

        dcm_xy = np.asarray(
            dcm_xy,
            dtype=float,
        ).reshape(
            2
        )

        self.com_world = (
            p_com.copy()
        )

        # ----------------------------------------------------
        # LIPM CoM projected onto ground
        # ----------------------------------------------------

        self.com_ground = np.array(
            [
                p_com[0],
                p_com[1],
                COM_GROUND_Z,
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # DCM projected onto ground
        #
        # xi = c + c_dot / omega
        # ----------------------------------------------------

        self.dcm_ground = np.array(
            [
                dcm_xy[0],
                dcm_xy[1],
                DCM_GROUND_Z,
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Existing LIPM ZMP
        # ----------------------------------------------------

        self.zmp.update_current(
            com_position=(
                p_com
            ),

            com_acceleration=(
                com_acceleration
            ),
        )


    # ========================================================
    # MPC ZMP PREVIEW
    # ========================================================

    def update_mpc_preview(
        self,
        *,
        x_result,
        y_result,
    ) -> None:

        self.zmp.update_preview(
            x_result=(
                x_result
            ),

            y_result=(
                y_result
            ),
        )


    # ========================================================
    # COMMIT FINISHED FOOTSTEP TO HISTORY
    # ========================================================

    def commit_footstep(
        self,
        *,
        side,
        position,
        rotation,
    ) -> None:

        footstep = (
            PlannedFootstep(
                side=(
                    side
                ),

                position=np.asarray(
                    position,
                    dtype=float,
                ).reshape(
                    3
                ).copy(),

                rotation=np.asarray(
                    rotation,
                    dtype=float,
                ).reshape(
                    3,
                    3,
                ).copy(),
            )
        )

        self.walking.planned_footsteps.append(
            footstep
        )

        if (
            len(
                self.walking.planned_footsteps
            )
            >
            MAX_PLANNED_FOOTSTEPS
        ):

            remove_count = (
                len(
                    self.walking.planned_footsteps
                )
                -
                MAX_PLANNED_FOOTSTEPS
            )

            del (
                self.walking
                .planned_footsteps[
                    0:remove_count
                ]
            )


    # ========================================================
    # PHASE ADAPTER
    # ========================================================

    @staticmethod
    def _make_state(
        phase,
        support_side,
    ):

        if phase == "INITIAL_DOUBLE_SUPPORT":

            walking_phase = (
                WalkingPhase
                .INITIAL_DOUBLE_SUPPORT
            )

        elif phase == "SINGLE_SUPPORT":

            walking_phase = (
                WalkingPhase
                .SINGLE_SUPPORT
            )

        elif phase == "DOUBLE_SUPPORT":

            walking_phase = (
                WalkingPhase
                .DOUBLE_SUPPORT
            )

        else:

            raise ValueError(
                f"Unknown walking phase: {phase}"
            )

        return (
            WalkingVisualizationState(
                phase=(
                    walking_phase
                ),

                support_side=(
                    support_side
                ),
            )
        )


    # ========================================================
    # DRAW EXTRA ADAPTIVE OVERLAY
    # ========================================================

    def _draw_adaptive_overlay(
        self,
        viewer,
        *,
        planner_target_side=None,
        planner_target_position=None,
        planner_target_rotation=None,
    ) -> None:

        with viewer.lock():

            scene = (
                viewer.user_scn
            )

            # =================================================
            # CURRENT ADAPTIVE FOOTHOLD
            # =================================================

            if (
                planner_target_side
                is not None
                and
                planner_target_position
                is not None
                and
                planner_target_rotation
                is not None
            ):

                current_target = (
                    PlannedFootstep(
                        side=(
                            planner_target_side
                        ),

                        position=np.asarray(
                            planner_target_position,
                            dtype=float,
                        ).reshape(
                            3
                        ),

                        rotation=np.asarray(
                            planner_target_rotation,
                            dtype=float,
                        ).reshape(
                            3,
                            3
                        ),
                    )
                )

                # Reuse the existing planned-foot footprint.
                self.walking.draw_planned_footprint(
                    scene,
                    current_target,
                )

                target_marker = np.array(
                    [
                        planner_target_position[0],
                        planner_target_position[1],
                        PLANNER_TARGET_Z,
                    ],
                    dtype=float,
                )

                add_sphere(
                    scene,
                    target_marker,
                    PLANNER_TARGET_RADIUS,
                    PLANNER_TARGET_RGBA,
                )

            # =================================================
            # COM GROUND PROJECTION
            # =================================================

            if self.com_ground is not None:

                add_sphere(
                    scene,
                    self.com_ground,
                    COM_GROUND_RADIUS,
                    COM_GROUND_RGBA,
                )

            # =================================================
            # COM VERTICAL PROJECTION LINE
            # =================================================

            if (
                self.com_world is not None
                and
                self.com_ground is not None
            ):

                add_line(
                    scene,
                    self.com_world,
                    self.com_ground,
                    COM_VERTICAL_LINE_WIDTH,
                    COM_VERTICAL_RGBA,
                )

            # =================================================
            # DCM
            # =================================================

            if self.dcm_ground is not None:

                add_sphere(
                    scene,
                    self.dcm_ground,
                    DCM_RADIUS,
                    DCM_RGBA,
                )

            # =================================================
            # COM -> DCM RELATION
            # =================================================

            if (
                self.com_ground is not None
                and
                self.dcm_ground is not None
            ):

                add_line(
                    scene,
                    self.com_ground,
                    self.dcm_ground,
                    COM_DCM_LINE_WIDTH,
                    COM_DCM_LINE_RGBA,
                )


    # ========================================================
    # COMPLETE GUI UPDATE
    # ========================================================

    def update_viewer(
        self,
        *,
        viewer,
        robot,
        phase,
        support_side,
        planner_target_side=None,
        planner_target_position=None,
        planner_target_rotation=None,
    ) -> None:

        state = (
            self._make_state(
                phase,
                support_side,
            )
        )

        # ----------------------------------------------------
        # 1. Existing walking visualization
        #
        # This clears and rebuilds user_scn.
        # ----------------------------------------------------

        self.walking.update(
            viewer,
            robot,
            state,
        )

        # ----------------------------------------------------
        # 2. Step Timing Adaptation overlay
        # ----------------------------------------------------

        self._draw_adaptive_overlay(
            viewer,
            planner_target_side=(
                planner_target_side
            ),

            planner_target_position=(
                planner_target_position
            ),

            planner_target_rotation=(
                planner_target_rotation
            ),
        )

        # ----------------------------------------------------
        # 3. Existing LIPM ZMP overlay
        #
        # Must be last because WalkingVisualizer resets ngeom.
        # ----------------------------------------------------

        self.zmp.draw_overlay(
            viewer
        )