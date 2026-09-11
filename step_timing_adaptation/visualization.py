"""
step_timing_adaptation/visualization.py
========================================
Vẽ CoM, CoM-desired, DCM, ZMP, support polygon trong MuJoCo GUI.

Không vẽ foot markers.

Đơn vị:
  - Sphere radius: meters.
  - Line width:    pixels.
  - Kích thước tham khảo từ lipm_mpc/run.py và walking_visualization.py.
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
    TRAIL_MAX_POINTS,
    COM_MARKER_RGBA,
    COM_TRAIL_RGBA,
    COM_MARKER_RADIUS,     # 0.007
    COM_TRAIL_WIDTH,       # 10.0
    SUPPORT_HULL_RGBA,
    SUPPORT_VERTEX_RGBA,
    SUPPORT_HULL_WIDTH,    # 7.0
    SUPPORT_VERTEX_RADIUS, # 0.0035
    FOOT_TOE,
    FOOT_HEEL,
    FOOT_HALF_WIDTH,
)


# ============================================================
# MÀU MARKER
# ============================================================

# CoM desired: xanh lá
COM_DES_RGBA      = np.array([0.10, 0.80, 0.20, 0.85], dtype=np.float32)
COM_DES_RADIUS    = 0.005
COM_DES_TRAIL_WIDTH = 6.0

# DCM: đỏ
DCM_RGBA          = np.array([1.00, 0.10, 0.10, 1.00], dtype=np.float32)
DCM_TRAIL_RGBA    = np.array([1.00, 0.10, 0.10, 0.65], dtype=np.float32)
DCM_MARKER_RADIUS = 0.007       # khớp current_radius của reference ZMP
DCM_TRAIL_WIDTH   = 5.0         # khớp trail_width của reference

# ZMP: tím
ZMP_RGBA          = np.array([0.60, 0.10, 0.90, 1.00], dtype=np.float32)
ZMP_TRAIL_RGBA    = np.array([0.60, 0.10, 0.90, 0.65], dtype=np.float32)
ZMP_MARKER_RADIUS = 0.007       # khớp current_radius của reference ZMP
ZMP_TRAIL_WIDTH   = 5.0         # khớp trail_width của reference


# ============================================================
# ĐỘ CAO MARKER (xếp lớp z)
# ============================================================

COM_MARKER_Z = GROUND_Z + 0.016
COM_DES_Z    = GROUND_Z + 0.014
DCM_MARKER_Z = GROUND_Z + 0.012
ZMP_MARKER_Z = GROUND_Z + 0.010

COM_TRAIL_Z  = GROUND_Z + 0.016
DCM_TRAIL_Z  = GROUND_Z + 0.012
ZMP_TRAIL_Z  = GROUND_Z + 0.010


# ============================================================
# VISUALIZER
# ============================================================

class WBCVisualizer:
    def __init__(self, mj_model, trail_max_points=TRAIL_MAX_POINTS):
        self.mj_model = mj_model
        self.trail_max_points = trail_max_points
        self.base_body_id = self._find_base_body_id(mj_model)

        self.com_trail = []
        self.com_des_trail = []
        self.dcm_trail = []
        self.zmp_trail = []

    @staticmethod
    def _find_base_body_id(mj_model):
        for jid in range(mj_model.njnt):
            if mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(mj_model.jnt_bodyid[jid])
        raise RuntimeError("Không tìm thấy floating-base joint.")

    def reset(self):
        self.com_trail = []
        self.com_des_trail = []
        self.dcm_trail = []
        self.zmp_trail = []

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
    def update(self, viewer,
               com_xy,
               com_des_xy=None,
               dcm_xy=None,
               zmp_xy=None,
               foot_L_pose=None,
               foot_R_pose=None):
        self._append_trail(self.com_trail, com_xy, COM_TRAIL_Z)
        self._append_trail(self.com_des_trail, com_des_xy, COM_DES_Z)
        self._append_trail(self.dcm_trail, dcm_xy, DCM_TRAIL_Z)
        self._append_trail(self.zmp_trail, zmp_xy, ZMP_TRAIL_Z)

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
                          COM_DES_TRAIL_WIDTH, COM_DES_RGBA)
            draw_polyline(scene, self.dcm_trail,
                          DCM_TRAIL_WIDTH, DCM_TRAIL_RGBA)
            draw_polyline(scene, self.zmp_trail,
                          ZMP_TRAIL_WIDTH, ZMP_TRAIL_RGBA)

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

        viewer.sync()