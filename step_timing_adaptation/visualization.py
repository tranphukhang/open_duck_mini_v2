"""
step_timing_adaptation/visualization.py
========================================
Vẽ CoM, CoM-desired, DCM, foot markers, support polygon trong MuJoCo GUI.

Style & màu đồng nhất với:
    - footstep_planning/walking_visualization.py
    - lipm_mpc/zmp_visualization.py

Pattern: reset scene.ngeom=0 mỗi frame (giống WalkingVisualizer.update).

Lưu ý: CoP chỉ vẽ point hiện tại (màu đen), KHÔNG vẽ trail.
"""

import numpy as np
import mujoco

from footstep_planning.walking_visualization import (
    # Hàm cơ bản
    add_sphere,
    add_line,
    draw_polyline,
    convex_hull_2d,
    yaw_rotation,
    # Constants
    GROUND_Z,
    CAMERA_DISTANCE,
    CAMERA_AZIMUTH,
    CAMERA_ELEVATION,
    TRAIL_MIN_DISTANCE,
    TRAIL_MAX_POINTS,
    # Màu có sẵn
    LEFT_TRAIL_RGBA,
    RIGHT_TRAIL_RGBA,
    COM_MARKER_RGBA,
    COM_TRAIL_RGBA,
    COM_MARKER_RADIUS,
    COM_TRAIL_WIDTH,
    SUPPORT_HULL_RGBA,
    SUPPORT_VERTEX_RGBA,
    SUPPORT_HULL_WIDTH,
    SUPPORT_VERTEX_RADIUS,
    # Foot geometry
    FOOT_TOE,
    FOOT_HEEL,
    FOOT_HALF_WIDTH,
)


# ============================================================
# MÀU MỚI (đồng bộ style với ZMP = magenta)
# ============================================================

COM_DES_RGBA      = np.array([0.55, 0.70, 1.00, 0.75], dtype=np.float32)
COM_DES_RADIUS    = 0.005

DCM_RGBA          = np.array([1.00, 0.00, 1.00, 1.00], dtype=np.float32)
DCM_TRAIL_RGBA    = np.array([0.90, 0.10, 0.95, 0.85], dtype=np.float32)
DCM_MARKER_RADIUS = 0.007
DCM_TRAIL_WIDTH   = 8.0

# CoP: đen, chỉ point, không trail
COP_RGBA          = np.array([0.00, 0.00, 0.00, 1.00], dtype=np.float32)
COP_MARKER_RADIUS = 0.006

FOOT_MARKER_RADIUS = 0.006


# ============================================================
# ĐỘ CAO MARKER (phân lớp z để dễ nhìn)
# ============================================================

COM_MARKER_Z      = GROUND_Z + 0.014
COM_DES_Z         = GROUND_Z + 0.012
DCM_MARKER_Z      = GROUND_Z + 0.010
COP_MARKER_Z      = GROUND_Z + 0.008
FOOT_MARKER_Z     = GROUND_Z + 0.006

COM_TRAIL_Z       = GROUND_Z + 0.014
DCM_TRAIL_Z       = GROUND_Z + 0.010


# ============================================================
# VISUALIZER
# ============================================================

class WBCVisualizer:
    def __init__(self, mj_model, trail_max_points=TRAIL_MAX_POINTS):
        self.mj_model = mj_model
        self.trail_max_points = trail_max_points
        self.base_body_id = self._find_base_body_id(mj_model)

        # Trails liên tục (CoP không có trail)
        self.com_trail = []
        self.com_des_trail = []
        self.dcm_trail = []

    # --------------------------------------------------------
    @staticmethod
    def _find_base_body_id(mj_model):
        for jid in range(mj_model.njnt):
            if mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(mj_model.jnt_bodyid[jid])
        raise RuntimeError("Không tìm thấy floating-base joint.")

    # --------------------------------------------------------
    def reset(self):
        """Xóa trail — gọi đầu mỗi test."""
        self.com_trail = []
        self.com_des_trail = []
        self.dcm_trail = []

    # --------------------------------------------------------
    def configure_camera(self, viewer):
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.base_body_id
            viewer.cam.distance = CAMERA_DISTANCE
            viewer.cam.azimuth = CAMERA_AZIMUTH
            viewer.cam.elevation = CAMERA_ELEVATION
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.sync()

    # --------------------------------------------------------
    # TRAIL HELPERS
    # --------------------------------------------------------
    @staticmethod
    def _make_world_point(xy, z):
        return np.array([xy[0], xy[1], z], dtype=float)

    def _append_trail(self, trail, xy, z):
        if xy is None:
            return
        p = self._make_world_point(xy, z)
        if len(trail) == 0:
            trail.append(p)
        elif np.linalg.norm(p[:2] - trail[-1][:2]) >= TRAIL_MIN_DISTANCE:
            trail.append(p)
        if len(trail) > self.trail_max_points:
            del trail[0:len(trail) - self.trail_max_points]

    # --------------------------------------------------------
    # SUPPORT POLYGON (convex hull của 2 bàn chân)
    # --------------------------------------------------------
    @staticmethod
    def _sole_corners(position, rotation):
        R_yaw = yaw_rotation(rotation)
        local_corners = np.array([
            [ FOOT_TOE,   FOOT_HALF_WIDTH, 0.0],
            [ FOOT_TOE,  -FOOT_HALF_WIDTH, 0.0],
            [-FOOT_HEEL, -FOOT_HALF_WIDTH, 0.0],
            [-FOOT_HEEL,  FOOT_HALF_WIDTH, 0.0],
        ])
        corners = np.array([
            position + R_yaw @ c for c in local_corners
        ])
        corners[:, 2] = GROUND_Z
        return corners

    def _get_support_polygon_xy(self, foot_L_pose, foot_R_pose):
        p_l, R_l = foot_L_pose
        p_r, R_r = foot_R_pose
        corners_L = self._sole_corners(p_l, R_l)
        corners_R = self._sole_corners(p_r, R_r)
        all_corners = np.vstack([corners_L, corners_R])
        hull = convex_hull_2d(all_corners)   # (N, 3)
        return hull[:, :2]

    # --------------------------------------------------------
    # UPDATE (main API)
    # --------------------------------------------------------
    def update(self, viewer,
               com_xy,
               com_des_xy=None,
               dcm_xy=None,
               cop_xy=None,
               foot_L_pose=None,
               foot_R_pose=None):
        # Trail (không có CoP)
        self._append_trail(self.com_trail, com_xy, COM_TRAIL_Z)
        self._append_trail(self.com_des_trail, com_des_xy, COM_DES_Z)
        self._append_trail(self.dcm_trail, dcm_xy, DCM_TRAIL_Z)

        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0

            # --- 1. Support polygon ---
            if foot_L_pose is not None and foot_R_pose is not None:
                polygon_xy = self._get_support_polygon_xy(
                    foot_L_pose, foot_R_pose)
                hull_world = np.column_stack([
                    polygon_xy[:, 0],
                    polygon_xy[:, 1],
                    np.full(len(polygon_xy), GROUND_Z + 0.004),
                ])
                for p in hull_world:
                    add_sphere(scene, p,
                               SUPPORT_VERTEX_RADIUS, SUPPORT_VERTEX_RGBA)
                if len(hull_world) >= 2:
                    for i in range(len(hull_world)):
                        j = (i + 1) % len(hull_world)
                        add_line(scene,
                                 hull_world[i], hull_world[j],
                                 SUPPORT_HULL_WIDTH, SUPPORT_HULL_RGBA)

            # --- 2. Trails ---
            draw_polyline(scene, self.com_trail,
                          COM_TRAIL_WIDTH, COM_TRAIL_RGBA)
            draw_polyline(scene, self.com_des_trail,
                          COM_TRAIL_WIDTH, COM_DES_RGBA)
            draw_polyline(scene, self.dcm_trail,
                          DCM_TRAIL_WIDTH, DCM_TRAIL_RGBA)

            # --- 3. Markers ---
            if com_xy is not None:
                add_sphere(scene,
                           [com_xy[0], com_xy[1], COM_MARKER_Z],
                           COM_MARKER_RADIUS, COM_MARKER_RGBA)
            if com_des_xy is not None:
                add_sphere(scene,
                           [com_des_xy[0], com_des_xy[1], COM_DES_Z],
                           COM_DES_RADIUS, COM_DES_RGBA)
            if dcm_xy is not None:
                add_sphere(scene,
                           [dcm_xy[0], dcm_xy[1], DCM_MARKER_Z],
                           DCM_MARKER_RADIUS, DCM_RGBA)
            if cop_xy is not None:
                add_sphere(scene,
                           [cop_xy[0], cop_xy[1], COP_MARKER_Z],
                           COP_MARKER_RADIUS, COP_RGBA)
            if foot_L_pose is not None:
                p_l, _ = foot_L_pose
                add_sphere(scene,
                           [p_l[0], p_l[1], FOOT_MARKER_Z],
                           FOOT_MARKER_RADIUS, LEFT_TRAIL_RGBA)
            if foot_R_pose is not None:
                p_r, _ = foot_R_pose
                add_sphere(scene,
                           [p_r[0], p_r[1], FOOT_MARKER_Z],
                           FOOT_MARKER_RADIUS, RIGHT_TRAIL_RGBA)

        viewer.sync()