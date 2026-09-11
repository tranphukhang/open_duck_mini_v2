"""
step_timing_adaptation/visualization.py
========================================
Vẽ CoM, CoM-desired, DCM, ZMP, foot markers + trails, support polygon.

Đặc điểm:
  - Trail giới hạn theo THỜI GIAN (0.5 giây gần nhất) cho tất cả đối tượng.
  - Foot markers: xanh lá (L), đỏ đậm (R) — kèm trail tương ứng.
  - ZMP: tím. DCM: đen. CoM_des: xanh lá mờ.
  - Đơn vị: sphere radius = meters, line width = pixels.
"""

import numpy as np
import mujoco

from footstep_planning.walking_visualization import (
    add_sphere,
    add_line,
    draw_polyline,
    convex_hull_2d,
    yaw_rotation,
    GROUND_Z,
    CAMERA_DISTANCE,
    CAMERA_AZIMUTH,
    CAMERA_ELEVATION,
    TRAIL_MIN_DISTANCE,
    LEFT_TRAIL_RGBA,
    RIGHT_TRAIL_RGBA,
    LEFT_TRAIL_WIDTH,
    RIGHT_TRAIL_WIDTH,
    COM_MARKER_RGBA,
    COM_TRAIL_RGBA,
    COM_MARKER_RADIUS,
    COM_TRAIL_WIDTH,
    SUPPORT_HULL_RGBA,
    SUPPORT_VERTEX_RGBA,
    SUPPORT_HULL_WIDTH,
    SUPPORT_VERTEX_RADIUS,
    FOOT_TOE,
    FOOT_HEEL,
    FOOT_HALF_WIDTH,
)


# ============================================================
# MÀU MARKER
# ============================================================

# CoM desired: xanh lá
COM_DES_RGBA        = np.array([0.10, 0.80, 0.20, 0.85], dtype=np.float32)
COM_DES_TRAIL_RGBA  = np.array([0.10, 0.80, 0.20, 0.65], dtype=np.float32)
COM_DES_RADIUS      = 0.005
COM_DES_TRAIL_WIDTH = 6.0

# DCM: đen
DCM_RGBA          = np.array([0.00, 0.00, 0.00, 1.00], dtype=np.float32)
DCM_TRAIL_RGBA    = np.array([0.00, 0.00, 0.00, 0.65], dtype=np.float32)
DCM_MARKER_RADIUS = 0.007
DCM_TRAIL_WIDTH   = 5.0

# ZMP: tím
ZMP_RGBA          = np.array([0.60, 0.10, 0.90, 1.00], dtype=np.float32)
ZMP_TRAIL_RGBA    = np.array([0.60, 0.10, 0.90, 0.65], dtype=np.float32)
ZMP_MARKER_RADIUS = 0.007
ZMP_TRAIL_WIDTH   = 5.0

FOOT_MARKER_RADIUS = 0.006


# ============================================================
# ĐỘ CAO MARKER (xếp lớp z)
# ============================================================

COM_MARKER_Z  = GROUND_Z + 0.016
COM_DES_Z     = GROUND_Z + 0.014
DCM_MARKER_Z  = GROUND_Z + 0.012
ZMP_MARKER_Z  = GROUND_Z + 0.010
FOOT_MARKER_Z = GROUND_Z + 0.008

COM_TRAIL_Z     = GROUND_Z + 0.016
COM_DES_TRAIL_Z = GROUND_Z + 0.014
DCM_TRAIL_Z     = GROUND_Z + 0.012
ZMP_TRAIL_Z     = GROUND_Z + 0.010
FOOT_TRAIL_Z    = GROUND_Z + 0.008


# ============================================================
# TRAIL DURATION
# ============================================================

TRAIL_DURATION_S = 0.5   # giữ 0.5 giây gần nhất


# ============================================================
# VISUALIZER
# ============================================================

class WBCVisualizer:
    def __init__(self, mj_model, trail_duration_s=TRAIL_DURATION_S):
        self.mj_model = mj_model
        self.trail_duration_s = trail_duration_s
        self.base_body_id = self._find_base_body_id(mj_model)

        # Trail: list[(t, np.array(3))]
        self.com_trail     = []
        self.com_des_trail = []
        self.dcm_trail     = []
        self.zmp_trail     = []
        self.foot_L_trail  = []
        self.foot_R_trail  = []

    @staticmethod
    def _find_base_body_id(mj_model):
        for jid in range(mj_model.njnt):
            if mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(mj_model.jnt_bodyid[jid])
        raise RuntimeError("Không tìm thấy floating-base joint.")

    def reset(self):
        self.com_trail     = []
        self.com_des_trail = []
        self.dcm_trail     = []
        self.zmp_trail     = []
        self.foot_L_trail  = []
        self.foot_R_trail  = []

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

    def _append_trail(self, trail, t, xy, z):
        """Thêm điểm nếu di chuyển đủ xa, rồi cắt bỏ điểm cũ hơn trail_duration."""
        if xy is None:
            return
        p = self._make_world_point(xy, z)
        if len(trail) == 0:
            trail.append([t, p])
        else:
            last_p = trail[-1][1]
            if np.linalg.norm(p[:2] - last_p[:2]) >= TRAIL_MIN_DISTANCE:
                trail.append([t, p])

        # Cắt theo thời gian
        t_cut = t - self.trail_duration_s
        idx = 0
        while idx < len(trail) and trail[idx][0] < t_cut:
            idx += 1
        if idx > 0:
            del trail[0:idx]

    # --------------------------------------------------------
    # SUPPORT POLYGON
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
        hull = convex_hull_2d(all_corners)
        return hull[:, :2]

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------
    def update(self, viewer, t,
               com_xy,
               com_des_xy=None,
               dcm_xy=None,
               zmp_xy=None,
               foot_L_pose=None,
               foot_R_pose=None):
        # --- Cập nhật trail ---
        self._append_trail(self.com_trail,     t, com_xy,     COM_TRAIL_Z)
        self._append_trail(self.com_des_trail, t, com_des_xy, COM_DES_TRAIL_Z)
        self._append_trail(self.dcm_trail,     t, dcm_xy,     DCM_TRAIL_Z)
        self._append_trail(self.zmp_trail,     t, zmp_xy,     ZMP_TRAIL_Z)

        foot_L_xy = None
        foot_R_xy = None
        if foot_L_pose is not None:
            foot_L_xy = foot_L_pose[0][:2]
        if foot_R_pose is not None:
            foot_R_xy = foot_R_pose[0][:2]

        self._append_trail(self.foot_L_trail, t, foot_L_xy, FOOT_TRAIL_Z)
        self._append_trail(self.foot_R_trail, t, foot_R_xy, FOOT_TRAIL_Z)

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
            com_pts     = [p for (_, p) in self.com_trail]
            com_des_pts = [p for (_, p) in self.com_des_trail]
            dcm_pts     = [p for (_, p) in self.dcm_trail]
            zmp_pts     = [p for (_, p) in self.zmp_trail]
            footL_pts   = [p for (_, p) in self.foot_L_trail]
            footR_pts   = [p for (_, p) in self.foot_R_trail]

            draw_polyline(scene, com_pts,
                          COM_TRAIL_WIDTH, COM_TRAIL_RGBA)
            draw_polyline(scene, com_des_pts,
                          COM_DES_TRAIL_WIDTH, COM_DES_TRAIL_RGBA)
            draw_polyline(scene, dcm_pts,
                          DCM_TRAIL_WIDTH, DCM_TRAIL_RGBA)
            draw_polyline(scene, zmp_pts,
                          ZMP_TRAIL_WIDTH, ZMP_TRAIL_RGBA)
            draw_polyline(scene, footL_pts,
                          LEFT_TRAIL_WIDTH, LEFT_TRAIL_RGBA)
            draw_polyline(scene, footR_pts,
                          RIGHT_TRAIL_WIDTH, RIGHT_TRAIL_RGBA)

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
            if zmp_xy is not None:
                add_sphere(scene,
                           [zmp_xy[0], zmp_xy[1], ZMP_MARKER_Z],
                           ZMP_MARKER_RADIUS, ZMP_RGBA)
            if foot_L_xy is not None:
                add_sphere(scene,
                           [foot_L_xy[0], foot_L_xy[1], FOOT_MARKER_Z],
                           FOOT_MARKER_RADIUS, LEFT_TRAIL_RGBA)
            if foot_R_xy is not None:
                add_sphere(scene,
                           [foot_R_xy[0], foot_R_xy[1], FOOT_MARKER_Z],
                           FOOT_MARKER_RADIUS, RIGHT_TRAIL_RGBA)

        viewer.sync()