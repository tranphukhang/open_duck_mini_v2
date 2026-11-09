"""S1 - Standing DSP + external push + CSV and synchronized MP4 video.

Place this file at:
    open_duck_mini_v2/mpc_srbd/Simulation/run_standing_S1_push_csv.py

Usage:
    python run_standing_S1_push_csv.py
    python run_standing_S1_push_csv.py --duration 10 --output results_s1
    python run_stading_S1_push_video_csv.py --viewer --push-force 5
    python run_stading_S1_push_video_csv.py --no-viewer  # video only
    python run_stading_S1_push_video_csv.py --no-video   # CSV only
    python run_standing_S1_push_csv.py --push-force 0.3 --viewer
    python run_standing_S1_push_csv.py --no-push    # same baseline, without pulse

Outputs (no MATLAB needed while simulating):
    standing_S1_push_<timestamp>_timeseries.csv
    standing_S1_push_<timestamp>_summary.csv
    standing_S1_push_<timestamp>_config.csv
    standing_S1_push_<timestamp>_video.mp4
    standing_S1_push_<timestamp>_video_frames.csv

Sample definition:
    At t_k read x_k + measured contact wrench from MuJoCo (under the
    PREVIOUS held actuator command), compute SRBD-MPC u_k, transform u_k
    into tau_k, and log. Then do exactly one mj_step -> x_(k+1).

The logged MuJoCo contact wrench and newly optimized MPC wrench are NOT
synchronous command/response pairs. The former describes the measured
force at the beginning of the MPC tick (i.e., after previous inputs).

The ACTIVE support polygon uses only load-bearing soles from the measured
MuJoCo contact wrench.  Its sole geometry comes from the low-facing vertices
of compiled collision meshes when available; a declared rectangular model
is a fallback, NEVER silently interpreted as an exact sole contact patch.
The 'planned_dsp' polygon uses both feet independently of measured contacts.
Neither diagnostic changes QP constraints or contact schedule.

Two time-limited, constant-force pushes act at the CURRENT WHOLE-ROBOT
CoM, via mujoco.mj_applyFT() on the floating-base body. They are applied
AFTER computing the current tau_k and BEFORE the ONE mj_step() call.
This is a force pulse, not a Dirac impulse; J = integral F dt.
The MPC never receives the push as a known feedforward signal.

CSV time semantics:
    x_k / measured ground wrench: pre-step measurement at t_k.
    push_force_*: external force that WILL be applied in [t_k,t_k+dt_sim).
    push_impulse_cumulative_*: impulse accumulated through t_k, before step.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import math
import json
import sys
import time
from contextlib import nullcontext

import mujoco
import numpy as np

SIMULATION_DIR = Path(__file__).resolve().parent
MPC_SRBD_DIR = SIMULATION_DIR.parent
REPO_ROOT = MPC_SRBD_DIR.parent
for directory in (
    SIMULATION_DIR,
    MPC_SRBD_DIR,
    MPC_SRBD_DIR / "State",
    MPC_SRBD_DIR / "SRBD",
    MPC_SRBD_DIR / "QP",
    MPC_SRBD_DIR / "Kinematics",
    MPC_SRBD_DIR / "Actuator",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from simulation_setup import initialize_simulation
from get_current_state_mpc import get_current_state_mpc
from get_contact_state import get_contact_state
from build_SRBD_model import build_SRBD_model
from build_and_solve_QP import build_and_solve_QP
from W_wrench_to_joint_torque import get_joint_torque_from_MPC_wrench
from actuator_mapping import ACTUATOR_NAMES, get_actuator_ids, apply_joint_torque
from QP_parameters import Q, R, mu as QP_MU, l_t as FOOT_TOE_M, l_h as FOOT_HEEL_M

# -----------------------------------------------------------------------------
# EXPERIMENT PARAMETERS. Keep unchanged for a reproducible S1 run.
# -----------------------------------------------------------------------------
XML_PATH = REPO_ROOT / "xmls" / "scene_flat_terrian_v3.xml"
N = 10
DT_MPC = 0.05           # Model prediction spacing; NOT the controller rate.
DT_MPC_UPDATE = 0.005   # 200 Hz MPC updates.
SIM_DURATION = 10.0
QP_EPS_ABS = 1e-5
QP_EPS_REL = 1e-5
QP_MAX_ITER = 10000
QP_CONSTRAINT_TOL = 1e-4
MAX_CONSECUTIVE_QP_FALLBACK = 3  # Set --max-fallbacks 0 for no fallback.
PRINT_EVERY_MPC_TICKS = 100

# ------------------------------------------------------------
# Disturbance trial: intentionally conservative starting values.
# Force pulse J = F * duration. Never interpreted as a proven
# no-fall bound: actual safety depends on contact, actuator,
# controller, CoM height and home pose.
# ------------------------------------------------------------
PUSH_FORCE_N = 0.6
PUSH_DURATION_S = 0.12
PUSH1_TIME_S = 3.0             # +World Y: primarily lateral/roll response
PUSH2_TIME_S = 6.0             # +World X: primarily fore-aft/pitch response
PUSH_BODY_NAME = "base_link"   # root floating-base body
FALL_ANGLE_DEG = 35.0
FALL_COM_HEIGHT_RATIO = 0.65  # diagnostic abort relative to initial CoM

# Measured foot width supplied by user: 23 mm TOTAL (not half width).
FOOT_WIDTH_M = 0.023
FOOT_HALF_WIDTH_M = FOOT_WIDTH_M / 2
# This force threshold is a reporting convention, not a physical fall criterion.
CONTACT_LOAD_THRESHOLD_N = 0.05
MIN_NORMAL_FORCE_FOR_COP_N = 0.05
SOLE_BAND_INITIAL_M = 0.0025  # mesh vertices near lowest surface at home pose
SOLE_BAND_CANDIDATES_M = (0.0025, 0.005, 0.010, 0.015)
MIN_SOLE_POLYGON_AREA_M2 = 1e-5
MAX_CSV_POLYGON_VERTICES = 32   # full vertices also serialized as JSON


# -----------------------------------------------------------------------------
# VISUALIZATION / VIDEO
# -----------------------------------------------------------------------------
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
VIDEO_CAMERA_AZIMUTH_DEG = 145.0
VIDEO_CAMERA_ELEVATION_DEG = -18.0
VIDEO_CAMERA_DISTANCE_M = 0.75
VIDEO_ARROW_LENGTH_PER_N_M = 0.025
VIDEO_ARROW_LENGTH_MIN_M = 0.07
VIDEO_ARROW_LENGTH_MAX_M = 0.25
RED_ARROW_RADIUS_M = 0.008
ROBOT_COM_MARKER_RADIUS_M = 0.009


VISUAL_FLAGS = (
    mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
    mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
    mujoco.mjtVisFlag.mjVIS_COM,
)


def enable_visual_flags(opt):
    """Viewer and video use exactly the same MuJoCo visualization flags."""
    for flag in VISUAL_FLAGS:
        opt.flags[flag] = 1


def add_custom_com_and_push_arrow(scene, data, root_body_id,
                                  push_force_world, arrow_scale):
    """Decorations only. NEVER modify forces, model, qpos or any joint state.

    Golden sphere: whole-robot CoM. Red arrow: external force, ONLY while pushed.
    Arrow starts at instantaneous whole-robot CoM, pointing along World force.
    The length is illustrative and not a physical displacement.
    """
    com = np.asarray(data.subtree_com[root_body_id], dtype=float).reshape(3).copy()
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        [ROBOT_COM_MARKER_RADIUS_M, 0.0, 0.0],
        com, np.eye(3).reshape(-1),
        [1.0, 0.85, 0.05, 1.0],
    )
    scene.ngeom += 1

    F = np.asarray(push_force_world, dtype=float).reshape(3)
    magnitude = float(np.linalg.norm(F))
    if magnitude < 1e-10 or scene.ngeom >= scene.maxgeom:
        return
    direction = F / magnitude
    length = float(np.clip(magnitude * arrow_scale,
                           VIDEO_ARROW_LENGTH_MIN_M, VIDEO_ARROW_LENGTH_MAX_M))
    end = com + length * direction
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_ARROW,
        [RED_ARROW_RADIUS_M, RED_ARROW_RADIUS_M, RED_ARROW_RADIUS_M],
        com, np.eye(3).reshape(-1),
        [1.0, 0.0, 0.0, 1.0],
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_ARROW,
        RED_ARROW_RADIUS_M, com, end,
    )
    scene.ngeom += 1


def configure_viewer_visuals(viewer, model, data, root_body_id):
    """Open viewer with CONTACTPOINT / CONTACTFORCE / TRANSPARENT / COM ON."""
    if viewer is None:
        return
    with viewer.lock():
        enable_visual_flags(viewer.opt)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.asarray(data.subtree_com[root_body_id], dtype=float)
        viewer.cam.azimuth = VIDEO_CAMERA_AZIMUTH_DEG
        viewer.cam.elevation = VIDEO_CAMERA_ELEVATION_DEG
        viewer.cam.distance = VIDEO_CAMERA_DISTANCE_M
        viewer.user_scn.ngeom = 0
        add_custom_com_and_push_arrow(
            viewer.user_scn, data, root_body_id, np.zeros(3),
            VIDEO_ARROW_LENGTH_PER_N_M,
        )
    viewer.sync()


def sync_viewer_overlay(viewer, data, root_body_id,
                        push_force_world, arrow_scale):
    if viewer is None:
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        add_custom_com_and_push_arrow(
            viewer.user_scn, data, root_body_id, push_force_world,
            arrow_scale,
        )
    viewer.sync()


class SimulationVideo:
    """Offscreen MP4 capture at fixed simulation-time FPS, not wall-clock FPS.

    The video is rendered from the same live MjData as the MPC simulation,
    at the end of selected physics steps. No second simulation is run.
    """

    def __init__(self, model, data, root_body_id, path, *, fps, width, height,
                 duration, arrow_scale):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                'Missing video dependency. Run: pip install imageio imageio-ffmpeg'
            ) from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root_body_id = int(root_body_id)
        self.fps = int(fps)
        self.duration = float(duration)
        self.arrow_scale = float(arrow_scale)
        self.frames = []
        self.frame_index = 0
        self.renderer = None
        self.writer = None
        self.last_error = ''

        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('Video FPS, width and height must be positive')
        # MuJoCo defaults to a relatively small offscreen framebuffer.
        # Extend it at runtime without modifying the XML or the dynamics.
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))

        try:
            self.renderer = mujoco.Renderer(model, width=int(width), height=int(height))
            self.option = mujoco.MjvOption()
            enable_visual_flags(self.option)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = np.asarray(
                data.subtree_com[root_body_id], dtype=float).reshape(3)
            self.camera.distance = VIDEO_CAMERA_DISTANCE_M
            self.camera.azimuth = VIDEO_CAMERA_AZIMUTH_DEG
            self.camera.elevation = VIDEO_CAMERA_ELEVATION_DEG
            self.writer = imageio.get_writer(
                str(self.path), format='FFMPEG', mode='I', fps=int(fps),
                codec='libx264', pixelformat='yuv420p', quality=8,
                macro_block_size=16, ffmpeg_log_level='error',
            )
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f'Cannot initialize MP4 recording: {exc}. '
                'Install imageio-ffmpeg; check OpenGL drivers / GL context, '
                'or run --no-video to collect CSV only.'
            ) from exc

    def capture_due(self, data, force_world, event_id):
        """Capture each frame exactly once on the simulation-time grid."""
        now = float(data.time)
        dt_tol = 1e-8
        while (self.frame_index / self.fps < self.duration - dt_tol
               and now >= self.frame_index / self.fps - dt_tol):
            target_time = self.frame_index / self.fps
            self.renderer.update_scene(
                data, camera=self.camera, scene_option=self.option,
            )
            add_custom_com_and_push_arrow(
                self.renderer.scene, data, self.root_body_id,
                force_world, self.arrow_scale,
            )
            frame = self.renderer.render()
            self.writer.append_data(frame)
            self.frames.append({
                'frame_index': int(self.frame_index),
                'target_time_s': float(target_time),
                'actual_rendered_sim_time_s': float(now),
                'push_event_id': int(event_id),
                'push_force_x_N': float(force_world[0]),
                'push_force_y_N': float(force_world[1]),
                'push_force_z_N': float(force_world[2]),
            })
            self.frame_index += 1

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self.writer = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self.renderer = None



def create_push_events(args):
    """Fixed perturbations. Direction is in WORLD coordinates."""
    if args.no_push or args.push_force <= 0:
        return []
    events = [
        {"id": 1, "start_s": float(args.push1_at),
         "duration_s": float(args.push_duration),
         "force_world_N": np.array([0.0, float(args.push_force), 0.0])},
    ]
    if not args.one_push:
        events.append(
            {"id": 2, "start_s": float(args.push2_at),
             "duration_s": float(args.push_duration),
             "force_world_N": np.array([float(args.push_force), 0.0, 0.0])}
        )
    events.sort(key=lambda e: e["start_s"])
    for i, event in enumerate(events):
        if event["start_s"] < 0 or event["duration_s"] <= 0:
            raise ValueError("Push time must be >= 0; push duration must be > 0.")
        if event["start_s"] >= args.duration:
            raise ValueError(f"Push {event['id']} starts after the simulation ends.")
        if event["start_s"] + event["duration_s"] > args.duration + 1e-9:
            raise ValueError(f"Push {event['id']} extends past the simulation end.")
        if i and events[i - 1]["start_s"] + events[i - 1]["duration_s"] > event["start_s"] + 1e-12:
            raise ValueError("Push intervals must not overlap.")
    return events


def push_at_time(t, events):
    """Piecewise-constant applied force over physics integration intervals."""
    for event in events:
        start, end = event["start_s"], event["start_s"] + event["duration_s"]
        if t >= start - 1e-10 and t < end - 1e-10:
            return event["force_world_N"].copy(), int(event["id"])
    return np.zeros(3, dtype=float), 0


def apply_force_at_robot_com(model, data, root_body_id, force_world_N):
    """Apply a pure force at the instantaneous CoM of the entire free robot.

    MuJoCo mj_applyFT maps a world-frame force at the supplied world point on
    a chosen body to generalized forces. The root body's subtree is the whole
    robot, so data.subtree_com[root_body_id] is the full-robot CoM.
    qfrc_applied is RESET in the main loop before each application.
    """
    if not np.any(force_world_N):
        return np.full(3, np.nan)
    point_world = np.asarray(data.subtree_com[root_body_id], dtype=float).reshape(3).copy()
    mujoco.mj_applyFT(
        model, data,
        np.asarray(force_world_N, dtype=float).reshape(3),
        np.zeros(3, dtype=float),          # no direct external couple
        point_world, int(root_body_id),
        data.qfrc_applied,
    )
    return point_world


def finite_or_nan(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError, OverflowError):
        return float("nan")


def put_xyz(row, name, vec, unit):
    arr = np.asarray(vec, dtype=float).reshape(3)
    for axis, val in zip("xyz", arr):
        row[f"{name}_{axis}_{unit}"] = finite_or_nan(val)


def get_site_id(model, name):
    idx = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
    if idx < 0:
        raise RuntimeError(f"Cannot locate site '{name}' in robot XML")
    return idx


def get_actuator_joint_addresses(model, actuator_ids):
    # In this model all 10 motors transmit to named hinge joints.
    addresses = []
    for aid in actuator_ids:
        jnt_id = int(model.actuator_trnid[int(aid), 0])
        if jnt_id < 0 or jnt_id >= model.njnt:
            raise RuntimeError("Expected motor actuator with a valid joint ID")
        addresses.append((int(model.jnt_qposadr[jnt_id]), int(model.jnt_dofadr[jnt_id])))
    return addresses


def build_contact_schedule(horizon):
    # Planned DSP is independent from measured contact; no gait phase switching.
    return np.ones((int(horizon), 2), dtype=np.int32)


def foot_geometry_from_sites(data, site_id, contact_site_id):
    """Plan-view reference rectangle from L6/R6 and L6_IK/R6_IK.

    Geometry is a *nominal rectangle for analysis*, not a measured collision
    patch. X dimension: [-FOOT_HEEL_M, +FOOT_TOE_M]; Y: +/- width/2.
    """
    origin = np.asarray(data.site_xpos[site_id], dtype=float).copy()
    world_R_local = np.asarray(data.site_xmat[contact_site_id], dtype=float).reshape(3, 3)
    ex = world_R_local[:2, 0].copy()
    ey = world_R_local[:2, 1].copy()
    # The axes are expected to be approximately horizontal for flat-ground DSP.
    if np.linalg.norm(ex) < 1e-6 or np.linalg.norm(ey) < 1e-6:
        raise RuntimeError("Foot contact-frame XY axis projects to near zero")
    ex /= np.linalg.norm(ex)
    ey /= np.linalg.norm(ey)
    corners_local = np.array([
        [-FOOT_HEEL_M, -FOOT_HALF_WIDTH_M],
        [FOOT_TOE_M, -FOOT_HALF_WIDTH_M],
        [FOOT_TOE_M, FOOT_HALF_WIDTH_M],
        [-FOOT_HEEL_M, FOOT_HALF_WIDTH_M],
    ], dtype=float)
    corners_world = origin[:2] + corners_local[:, 0, None] * ex + corners_local[:, 1, None] * ey
    return {"origin": origin, "ex": ex, "ey": ey, "corners": corners_world}


def convex_hull_xy(points):
    """2D monotone-chain convex hull, counterclockwise, no scipy dependency."""
    pts = sorted(set((float(x), float(y)) for x, y in np.asarray(points).reshape(-1, 2)))
    if len(pts) < 3:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area_xy(points):
    """Polygon area in square meters, valid for a CCW convex hull."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], -1)))) * 0.5


def _mesh_vertices_local(model, geom_id):
    """Compiled MuJoCo mesh_vert are in the corresponding compiled geom frame."""
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return None
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    first = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    if count <= 0:
        return None
    return np.asarray(model.mesh_vert[first:first + count], dtype=float).reshape(-1, 3).copy()


def _geom_to_world(data, geom_id, geom_local_vertices):
    R = np.asarray(data.geom_xmat[int(geom_id)], dtype=float).reshape(3, 3)
    p = np.asarray(data.geom_xpos[int(geom_id)], dtype=float).reshape(3)
    return np.asarray(geom_local_vertices) @ R.T + p


def select_collision_sole_vertices(model, data, contact_map, side):
    """Identify bottom-face collision vertices ONCE in the home pose.

    Convex mesh foot collision can include the ankle's 3D volume. Taking
    all projected vertices would overestimate the sole. We instead find
    a narrow lowest-Z band at the initial nominal upright pose. The chosen
    vertices stay attached to each collision geom during the simulation.
    This is a geometric proxy for the sole, NOT the instantaneous contact patch.
    """
    ids = contact_map[f'{"left" if side == "L" else "right"}_foot_geom_ids']
    meshes = []
    for gid in ids:
        verts = _mesh_vertices_local(model, int(gid))
        if verts is not None:
            world = _geom_to_world(data, int(gid), verts)
            meshes.append((int(gid), verts, world))
    if not meshes:
        return None, 'no_collision_mesh'
    min_z = min(float(np.min(w[:, 2])) for _, _, w in meshes)
    for band in SOLE_BAND_CANDIDATES_M:
        chosen = []
        points = []
        for gid, local, world in meshes:
            mask = world[:, 2] <= min_z + band + 1e-10
            if np.any(mask):
                chosen.append((gid, local[mask].copy()))
                points.append(world[mask, :2])
        if not points:
            continue
        hull = convex_hull_xy(np.concatenate(points, axis=0))
        if polygon_area_xy(hull) >= MIN_SOLE_POLYGON_AREA_M2:
            return chosen, f'collision_mesh_bottom_band_{band:g}m'
    return None, 'insufficient_bottom_mesh_area'


def build_sole_geometries(model, data, contact_map, site_ids, preferred_source):
    """Initialize sole models; explicit source selection and fallback reason."""
    out = {}
    for side, site_name, ik_name in (("L", "L6", "L6_IK"),
                                     ("R", "R6", "R6_IK")):
        candidates, reason = (None, 'rectangle_selected')
        if preferred_source == 'collision':
            candidates, reason = select_collision_sole_vertices(
                model, data, contact_map, side)
        out[side] = {'mesh_vertices': candidates,
                     'source': reason if candidates else
                     f'nominal_rectangle_fallback_{reason}',
                     'site_id': site_ids[site_name],
                     'ik_site_id': site_ids[ik_name]}
    return out


def sole_polygon_current(model, data, sole):
    """Sole area projected into horizontal world XY, expressed as CCW hull."""
    if sole['mesh_vertices']:
        projected = []
        for gid, local_vertices in sole['mesh_vertices']:
            projected.append(_geom_to_world(data, gid, local_vertices)[:, :2])
        pts = np.vstack(projected)
        return convex_hull_xy(pts)
    return convex_hull_xy(foot_geometry_from_sites(
        data, sole['site_id'], sole['ik_site_id'])['corners'])


def active_support_from_load(poly_left, poly_right, contact_left, contact_right):
    """Current measured load-bearing support, not planned contact schedule."""
    if contact_left and contact_right:
        return 'DSP', convex_hull_xy(np.vstack([poly_left, poly_right]))
    if contact_left:
        return 'SSP_L', convex_hull_xy(poly_left)
    if contact_right:
        return 'SSP_R', convex_hull_xy(poly_right)
    return 'NONE', np.empty((0, 2), dtype=float)


def write_polygon_vertices(row, prefix, hull, max_vertices=MAX_CSV_POLYGON_VERTICES):
    arr = np.asarray(hull, dtype=float).reshape(-1, 2)
    row[f'{prefix}_vertex_count'] = int(len(arr))
    row[f'{prefix}_vertices_xy_json'] = json.dumps(arr.tolist(), separators=(',', ':'))
    row[f'{prefix}_vertex_columns_truncated'] = int(len(arr) > max_vertices)
    row[f'{prefix}_area_m2'] = polygon_area_xy(arr)
    for k in range(max_vertices):
        row[f'{prefix}_v{k + 1}_x_m'] = float(arr[k, 0]) if k < len(arr) else np.nan
        row[f'{prefix}_v{k + 1}_y_m'] = float(arr[k, 1]) if k < len(arr) else np.nan


def signed_polygon_margin(point_xy, polygon):
    """Signed minimum Euclidean distance to polygon boundary, inside positive."""
    point = np.asarray(point_xy, dtype=float).reshape(2)
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(point)) or len(polygon) < 3:
        return float("nan")
    inside = True
    best = float("inf")
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        d = b - a
        length2 = float(d @ d)
        if length2 < 1e-20:
            continue
        if d[0] * (point[1] - a[1]) - d[1] * (point[0] - a[0]) < -1e-10:
            inside = False
        t = np.clip(np.dot(point - a, d) / length2, 0.0, 1.0)
        distance = float(np.linalg.norm(point - (a + t * d)))
        best = min(best, distance)
    return best if inside else -best


def extract_contact_wrenches(model, data, contact_state, floor_height):
    """Sum real MuJoCo contact forces on each FOOT, in WORLD coordinates.

    MuJoCo mj_contactForce() uses contact coordinates with the normal as axis 0.
    contact.frame has basis vectors in its ROWS, so WORLD = frame.T @ LOCAL.
    The positive contact normal acts from geom1 toward geom2: the foot gets
    +wrench if geom2=foot, and -wrench if geom1=foot.

    Moments are accumulated about O=(0,0,floor_height) in world coordinates.
    For a horizontal contact plane this gives CoP x=-My/Fz, y=Mx/Fz.
    """
    floor_id = int(contact_state.get("floor_geom_id", -1))
    # floor_geom_id is from contact_map; for compatibility derive via name.
    if floor_id < 0:
        floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    out = {}
    for side, key in (("L", "left_contacts"), ("R", "right_contacts")):
        total_F = np.zeros(3)
        total_M0 = np.zeros(3)
        num_active = 0
        actual_contact_points = []
        for entry in contact_state[key]:
            ci = int(entry["contact_index"])
            con = data.contact[ci]
            if int(con.efc_address) < 0:  # detected but not enforced/active
                continue
            wrench_local = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, ci, wrench_local)
            Rrows = np.asarray(con.frame, dtype=float).reshape(3, 3)
            normal_sign = +1.0 if int(con.geom1) == floor_id else -1.0
            F = normal_sign * (Rrows.T @ wrench_local[:3])
            M = normal_sign * (Rrows.T @ wrench_local[3:])
            p = np.asarray(con.pos, dtype=float).reshape(3)
            arm = p - np.array([0.0, 0.0, floor_height])
            total_F += F
            total_M0 += np.cross(arm, F) + M
            num_active += 1
            if F[2] > CONTACT_LOAD_THRESHOLD_N:
                actual_contact_points.append(p[:2].copy())
        Fz = float(total_F[2])
        if Fz > MIN_NORMAL_FORCE_FOR_COP_N:
            cop = np.array([-total_M0[1] / Fz, total_M0[0] / Fz])
        else:
            cop = np.full(2, np.nan)
        out[side] = {
            "F_world": total_F,
            "M_origin_world": total_M0,
            "cop_world_xy": cop,
            "active_points": num_active,
            "loaded_contact_points_xy": np.asarray(actual_contact_points, dtype=float).reshape(-1, 2),
            "has_load": int(Fz > CONTACT_LOAD_THRESHOLD_N),
        }
    total_F = out["L"]["F_world"] + out["R"]["F_world"]
    total_M0 = out["L"]["M_origin_world"] + out["R"]["M_origin_world"]
    if total_F[2] > MIN_NORMAL_FORCE_FOR_COP_N:
        total_cop = np.array([-total_M0[1] / total_F[2], total_M0[0] / total_F[2]])
    else:
        total_cop = np.full(2, np.nan)
    return out, total_F, total_M0, total_cop


def actual_foot_cop_metrics(cop_xy, geom):
    cop = np.asarray(cop_xy, dtype=float)
    if not np.all(np.isfinite(cop)):
        return float("nan"), float("nan"), float("nan")
    disp = cop - geom["origin"][:2]
    xx = float(np.dot(disp, geom["ex"]))
    yy = float(np.dot(disp, geom["ey"]))
    margin = min(xx + FOOT_HEEL_M, FOOT_TOE_M - xx, FOOT_HALF_WIDTH_M - abs(yy))
    return xx, yy, margin


def mpc_local_wrench(u0, world_to_left, world_to_right):
    out = {}
    for side, offset, rotation in (("L", 0, world_to_left), ("R", 6, world_to_right)):
        F = np.asarray(u0[offset:offset + 3], dtype=float)
        M = np.asarray(u0[offset + 3:offset + 6], dtype=float)
        local_F = rotation @ F
        local_M = rotation @ M
        Fz = float(local_F[2])
        cop_x = -local_M[1] / Fz if Fz > MIN_NORMAL_FORCE_FOR_COP_N else np.nan
        cop_y = local_M[0] / Fz if Fz > MIN_NORMAL_FORCE_FOR_COP_N else np.nan
        out[side] = {"F": F, "M": M, "local_F": local_F, "local_M": local_M,
                     "cop_x": cop_x, "cop_y": cop_y}
    return out


def add_snapshot(row, *, model, data, current, contact_map, u0, tau_raw,
                 tau_limited, actuator_ids, joint_addrs, x_target,
                 solver_info, post_diag, compute_time, step_no, qp_mode,
                 fallback_count, mass, ground_z, site_ids, sole_models,
                 push_force_world, push_event_id, push_point_world,
                 push_impulse_cumulative_world, dt_sim):
    """Add a consistent pre-physics x_k snapshot and current control u_k."""
    state = np.asarray(current.x, dtype=float).reshape(13)
    row["time_s"] = float(data.time)
    row["physics_step"] = int(step_no)
    row["qp_mode"] = qp_mode
    row["qp_fallback_consecutive"] = int(fallback_count)
    row["push_active"] = int(push_event_id != 0)
    row["push_event_id"] = int(push_event_id)
    put_xyz(row, "push_force", push_force_world, "N")
    put_xyz(row, "push_point", push_point_world, "m")
    put_xyz(row, "push_impulse_cumulative", push_impulse_cumulative_world, "Ns")
    put_xyz(row, "push_impulse_this_step", np.asarray(push_force_world) * dt_sim, "Ns")
    row["push_force_magnitude_N"] = float(np.linalg.norm(push_force_world))
    row["push_accel_nominal_m_s2"] = float(
        np.linalg.norm(push_force_world) / mass
    )  # force/mass, not measured acceleration
    for i in range(min(6, len(data.qfrc_applied))):
        row[f"push_qfrc_applied_dof{i}"] = float(data.qfrc_applied[i])
    row["planned_contact_L"] = 1
    row["planned_contact_R"] = 1
    for i, key in enumerate(("com_x_m", "com_y_m", "com_z_m",
                              "roll_rad", "pitch_rad", "yaw_rad",
                              "com_vx_m_s", "com_vy_m_s", "com_vz_m_s",
                              "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s")):
        row[key] = float(state[i])
    for i, key in enumerate(("ref_com_x_m", "ref_com_y_m", "ref_com_z_m",
                              "ref_roll_rad", "ref_pitch_rad", "ref_yaw_rad")):
        row[key] = float(x_target[i])
    row["roll_deg"] = math.degrees(state[3])
    row["pitch_deg"] = math.degrees(state[4])
    row["yaw_deg"] = math.degrees(state[5])
    row["com_xy_drift_from_initial_m"] = float(np.linalg.norm(state[:2] - x_target[:2]))
    row["com_speed_m_s"] = float(np.linalg.norm(state[6:9]))
    row["omega_magnitude_rad_s"] = float(np.linalg.norm(state[9:12]))
    row["robot_mass_kg"] = float(mass)
    row["robot_weight_N"] = float(-mass * model.opt.gravity[2])
    cs = get_contact_state(model, data, contact_map)
    row["contact_geom_L"] = int(cs["c_L"])
    row["contact_geom_R"] = int(cs["c_R"])
    row["contact_geom_count_L"] = int(cs["left_contact_count"])
    row["contact_geom_count_R"] = int(cs["right_contact_count"])
    # Measured wrench is under command from the previous MPC update.
    actual, fsum, msum, cop_total = extract_contact_wrenches(model, data, cs, ground_z)
    row["actual_contact_load_L"] = actual["L"]["has_load"]
    row["actual_contact_load_R"] = actual["R"]["has_load"]
    row["actual_contact_point_count_L"] = actual["L"]["active_points"]
    row["actual_contact_point_count_R"] = actual["R"]["active_points"]
    put_xyz(row, "actual_F_total", fsum, "N")
    put_xyz(row, "actual_M_total_origin", msum, "Nm")
    com_arm_from_ground_origin = state[:3] - np.array([0.0, 0.0, ground_z])
    put_xyz(row, "actual_M_total_about_com", msum - np.cross(com_arm_from_ground_origin, fsum), "Nm")
    row["actual_cop_total_x_m"] = finite_or_nan(cop_total[0])
    row["actual_cop_total_y_m"] = finite_or_nan(cop_total[1])
    row["actual_total_Fz_minus_mg_N"] = float(fsum[2] - row["robot_weight_N"])
    geom_l = foot_geometry_from_sites(data, site_ids["L6"], site_ids["L6_IK"])
    geom_r = foot_geometry_from_sites(data, site_ids["R6"], site_ids["R6_IK"])
    poly_l = sole_polygon_current(model, data, sole_models["L"])
    poly_r = sole_polygon_current(model, data, sole_models["R"])

    # Two different geometric concepts must never be conflated:
    #  1. Active support: load-bearing feet detected from MuJoCo, changes
    #     DSP -> SSP_L/SSP_R -> NONE regardless of planned contact schedule.
    #  2. Planned DSP: both nominal soles, only for comparison/visualization.
    support_mode, hull = active_support_from_load(
        poly_l, poly_r, actual["L"]["has_load"], actual["R"]["has_load"])
    planned_hull = convex_hull_xy(np.vstack([poly_l, poly_r]))
    row["support_mode"] = support_mode
    row["support_loaded_feet"] = int(actual["L"]["has_load"] + actual["R"]["has_load"])
    row["support_polygon_is_measured_load_based"] = 1
    row["support_polygon_source_L"] = sole_models["L"]["source"]
    row["support_polygon_source_R"] = sole_models["R"]["source"]
    row["com_support_margin_m"] = signed_polygon_margin(state[:2], hull)
    row["actual_cop_support_margin_m"] = signed_polygon_margin(cop_total, hull)
    row["com_planned_dsp_margin_m"] = signed_polygon_margin(state[:2], planned_hull)
    row["actual_cop_planned_dsp_margin_m"] = signed_polygon_margin(cop_total, planned_hull)

    # Preserve legacy support_v1...support_v8 columns for MATLAB compatibility:
    # their meaning is now ACTUAL LOADED support, not always DSP.
    row["support_polygon_vertex_count"] = int(len(hull))
    row["support_polygon_area_m2"] = polygon_area_xy(hull)
    for i in range(8):
        row[f"support_v{i+1}_x_m"] = float(hull[i, 0]) if i < min(len(hull), 8) else np.nan
        row[f"support_v{i+1}_y_m"] = float(hull[i, 1]) if i < min(len(hull), 8) else np.nan

    # Full polygon export, including collision-foot contours with >4 vertices.
    write_polygon_vertices(row, 'support_active', hull)
    write_polygon_vertices(row, 'support_planned_dsp', planned_hull)
    write_polygon_vertices(row, 'sole_L', poly_l)
    write_polygon_vertices(row, 'sole_R', poly_r)

    # Secondary, conservative diagnostic: convex hull of contact-point
    # positions carrying normal load in the CURRENT MuJoCo manifold.
    # If 0-2 independent points exist, this hull has no defined 2D area.
    # It must NOT be presented as a fully resolved pressure-contact patch.
    manifold_points = np.vstack([
        actual['L']['loaded_contact_points_xy'],
        actual['R']['loaded_contact_points_xy'],
    ])
    manifold_hull = convex_hull_xy(manifold_points)
    write_polygon_vertices(row, 'contact_point_hull', manifold_hull)
    row['contact_point_hull_is_2d'] = int(
        polygon_area_xy(manifold_hull) >= MIN_SOLE_POLYGON_AREA_M2)
    row['com_contact_point_hull_margin_m'] = signed_polygon_margin(
        state[:2], manifold_hull) if row['contact_point_hull_is_2d'] else np.nan
    row['actual_cop_contact_point_hull_margin_m'] = signed_polygon_margin(
        cop_total, manifold_hull) if row['contact_point_hull_is_2d'] else np.nan
    mpc_foot = mpc_local_wrench(u0, current.L6_IK_R_W, current.R6_IK_R_W)
    for side, geom in (("L", geom_l), ("R", geom_r)):
        phy = actual[side]
        foot = mpc_foot[side]
        put_xyz(row, f"actual_F_{side}", phy["F_world"], "N")
        put_xyz(row, f"actual_M_{side}_origin", phy["M_origin_world"], "Nm")
        row[f"actual_cop_{side}_x_m"] = finite_or_nan(phy["cop_world_xy"][0])
        row[f"actual_cop_{side}_y_m"] = finite_or_nan(phy["cop_world_xy"][1])
        cx, cy, local_margin = actual_foot_cop_metrics(phy["cop_world_xy"], geom)
        row[f"actual_cop_{side}_foot_x_m"] = cx
        row[f"actual_cop_{side}_foot_y_m"] = cy
        row[f"actual_cop_{side}_rectangle_margin_m"] = local_margin
        row[f"actual_cop_{side}_sole_margin_m"] = signed_polygon_margin(
            phy["cop_world_xy"], poly_l if side == "L" else poly_r)
        put_xyz(row, f"foot_site_{side}", geom["origin"], "m")
        for i, corner in enumerate(geom["corners"], 1):
            row[f"foot_{side}_corner{i}_x_m"] = float(corner[0])
            row[f"foot_{side}_corner{i}_y_m"] = float(corner[1])
        put_xyz(row, f"mpc_F_{side}", foot["F"], "N")
        put_xyz(row, f"mpc_M_{side}", foot["M"], "Nm")
        put_xyz(row, f"mpc_contact_F_{side}", foot["local_F"], "N")
        put_xyz(row, f"mpc_contact_M_{side}", foot["local_M"], "Nm")
        row[f"mpc_cop_{side}_foot_x_m"] = finite_or_nan(foot["cop_x"])
        row[f"mpc_cop_{side}_foot_y_m"] = finite_or_nan(foot["cop_y"])
        row[f"mpc_Fz_{side}_world_N"] = float(foot["F"][2])
        Fxy = float(np.linalg.norm(phy["F_world"][:2]))
        Fz = float(phy["F_world"][2])
        row[f"actual_friction_util_{side}"] = Fxy / (QP_MU * Fz) if Fz > MIN_NORMAL_FORCE_FOR_COP_N and QP_MU > 0 else np.nan
        mf = np.asarray(foot["local_F"], dtype=float)
        row[f"mpc_friction_util_{side}"] = float(np.linalg.norm(mf[:2]) / (QP_MU * mf[2])) if mf[2] > MIN_NORMAL_FORCE_FOR_COP_N and QP_MU > 0 else np.nan
    row["mpc_Fz_total_world_N"] = float(u0[2] + u0[8])
    row["mpc_Fz_total_minus_mg_N"] = float(row["mpc_Fz_total_world_N"] - row["robot_weight_N"])
    row["qp_status"] = str(solver_info.get("status", ""))
    row["qp_status_val"] = finite_or_nan(solver_info.get("status_val"))
    row["qp_postcheck_solution_ok"] = int(bool((post_diag or {}).get("solution_ok", False)))
    for key in ("iterations", "objective", "primal_residual", "dual_residual",
                "setup_time", "solve_time", "run_time"):
        row[f"qp_{key}"] = finite_or_nan(solver_info.get(key))
    row["qp_max_violation"] = finite_or_nan((post_diag or {}).get("max_violation"))
    row["mpc_compute_time_s"] = finite_or_nan(compute_time)
    row["mpc_compute_under_update_period"] = int(compute_time <= DT_MPC_UPDATE)
    row["torque_clip_count"] = 0
    for i, (aid, name, (qadr, dadr)) in enumerate(zip(actuator_ids, ACTUATOR_NAMES, joint_addrs)):
        row[f"q_{name}_rad"] = float(data.qpos[qadr])
        row[f"dq_{name}_rad_s"] = float(data.qvel[dadr])
        row[f"tau_{name}_raw_Nm"] = float(tau_raw[i])
        # MuJoCo computed this physical actuator force under PREVIOUS command.
        row[f"actuator_force_previous_{name}_Nm"] = float(data.actuator_force[int(aid)])
        row[f"tau_{name}_limited_cmd_Nm"] = float(tau_limited[i])
        clipped = int(abs(tau_raw[i] - tau_limited[i]) > 1e-9)
        row[f"tau_{name}_clip"] = clipped
        row["torque_clip_count"] += clipped
    return row


def limited_motor_commands(model, tau_raw, actuator_ids):
    tau_limited = np.asarray(tau_raw, dtype=float).copy()
    for i, aid in enumerate(actuator_ids):
        if int(model.actuator_ctrllimited[int(aid)]):
            lower, upper = np.asarray(model.actuator_ctrlrange[int(aid)], dtype=float)
            tau_limited[i] = float(np.clip(tau_limited[i], lower, upper))
    return tau_limited


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return None
    keys = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def make_summary(rows, stop_reason, duration, mass, config):
    s = {
        "scenario": config.get("experiment", "S1_DSP_standing"), "stop_reason": stop_reason,
        "requested_duration_s": duration, "actual_end_time_s": rows[-1]["time_s"] if rows else 0.0,
        "number_of_logged_mpc_ticks": len(rows), "mass_kg": mass,
        "source_commit_is_not_pinned": 1,
        "push_enabled": int(bool(config.get("push_enabled", 0))),
        "push_requested_force_N": config.get("push_force_N", 0.0),
        "push_requested_duration_s": config.get("push_duration_s", 0.0),
    }
    if not rows:
        return s
    def vals(key):
        return np.array([finite_or_nan(r.get(key)) for r in rows], dtype=float)
    def finite_vals(key):
        a = vals(key)
        return a[np.isfinite(a)]
    for name in ("roll_deg", "pitch_deg", "yaw_deg", "com_speed_m_s",
                 "com_xy_drift_from_initial_m", "com_support_margin_m",
                 "actual_cop_support_margin_m", "actual_total_Fz_minus_mg_N",
                 "mpc_compute_time_s", "qp_solve_time"):
        arr = finite_vals(name)
        if arr.size:
            s[f"{name}_mean"] = float(np.mean(arr))
            s[f"{name}_min"] = float(np.min(arr))
            s[f"{name}_max"] = float(np.max(arr))
            s[f"{name}_rms"] = float(np.sqrt(np.mean(arr**2)))
    s["full_DSP_geom_fraction"] = float(np.mean((vals("contact_geom_L") > 0) & (vals("contact_geom_R") > 0)))
    s["full_DSP_loaded_fraction"] = float(np.mean((vals("actual_contact_load_L") > 0) & (vals("actual_contact_load_R") > 0)))
    for mode in ('DSP', 'SSP_L', 'SSP_R', 'NONE'):
        s[f'active_support_mode_{mode}_ticks'] = sum(r.get('support_mode') == mode for r in rows)
        s[f'active_support_mode_{mode}_fraction'] = float(
            s[f'active_support_mode_{mode}_ticks'] / len(rows))
    s['com_active_support_margin_valid_ticks'] = int(np.sum(np.isfinite(vals('com_support_margin_m'))))
    s['cop_active_support_margin_valid_ticks'] = int(np.sum(np.isfinite(vals('actual_cop_support_margin_m'))))
    s["solver_fallback_ticks"] = int(sum(r["qp_mode"] == "fallback" for r in rows))
    s["solver_failed_and_stopped_ticks"] = int(sum(r["qp_mode"] == "failed_stop" for r in rows))
    s["total_actuator_clip_events"] = int(sum(int(r["torque_clip_count"]) for r in rows))
    s["solver_compute_deadline_miss_ticks"] = int(sum(not bool(r["mpc_compute_under_update_period"]) for r in rows))
    s["completed_requested_duration"] = int(stop_reason == "completed")
    s["push_active_log_samples"] = int(np.sum(vals("push_active") == 1))
    for axis in "xyz":
        # The cumulative impulse is updated every physics step; the
        # timeseries samples it at MPC ticks. Final actual impulse is
        # supplied by the caller in the summary after integration.
        a = finite_vals(f"push_impulse_cumulative_{axis}_Ns")
        if len(a):
            s[f"push_impulse_observed_last_{axis}_Ns"] = float(a[-1])
    for eid in (1, 2):
        selections = [r for r in rows if r.get("push_event_id") == eid]
        s[f"push_{eid}_logged_active_mpc_ticks"] = len(selections)
        if selections:
            s[f"push_{eid}_measured_max_roll_abs_deg"] = max(
                abs(float(r["roll_deg"])) for r in selections)
            s[f"push_{eid}_measured_max_pitch_abs_deg"] = max(
                abs(float(r["pitch_deg"])) for r in selections)
    return s


def main():
    parser = argparse.ArgumentParser(description="S1: DSP standing + detailed CSV for MATLAB")
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    viewer_flags = parser.add_mutually_exclusive_group()
    viewer_flags.add_argument('--viewer', dest='viewer', action='store_true',
                              default=True, help='Show MuJoCo GUI (default ON)')
    viewer_flags.add_argument('--no-viewer', dest='viewer', action='store_false',
                              help='Disable GUI; still record MP4 unless --no-video')
    video_flags = parser.add_mutually_exclusive_group()
    video_flags.add_argument('--video', dest='video', action='store_true',
                             default=True, help='Record MP4 (default ON)')
    video_flags.add_argument('--no-video', dest='video', action='store_false',
                             help='Disable MP4 but still export CSV')
    parser.add_argument('--video-fps', type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument('--video-width', type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument('--video-height', type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument('--arrow-scale', type=float,
                        default=VIDEO_ARROW_LENGTH_PER_N_M,
                        help='Arrow length [m/N] (clamped for visibility)')
    parser.add_argument("--realtime", action="store_true", help="Throttle to wall-time (GUI) if possible")
    parser.add_argument("--max-fallbacks", type=int, default=MAX_CONSECUTIVE_QP_FALLBACK)
    parser.add_argument("--output", type=Path, default=SIMULATION_DIR / "results_s1")
    parser.add_argument("--push-force", type=float, default=PUSH_FORCE_N,
                        help="N: common magnitude for pushes 1 and 2 (default 0.6 N)")
    parser.add_argument("--push-duration", type=float, default=PUSH_DURATION_S,
                        help="s: each rectangular pulse duration (default 0.12s)")
    parser.add_argument("--push1-at", type=float, default=PUSH1_TIME_S,
                        help="s: +world Y push time (default 3.0s)")
    parser.add_argument("--push2-at", type=float, default=PUSH2_TIME_S,
                        help="s: +world X push time (default 6.0s)")
    parser.add_argument("--one-push", action="store_true",
                        help="Only apply first (+world Y) push")
    parser.add_argument("--no-push", action="store_true",
                        help="Disable all pushes, for a baseline control experiment")
    parser.add_argument("--sole-source", choices=("collision", "rectangle"),
                        default="collision",
                        help="Support-sole geometry: MuJoCo collision mesh lowest surface "
                             "(preferred) or nominal L6-based rectangle")
    parser.add_argument("--fall-angle-deg", type=float, default=FALL_ANGLE_DEG,
                        help="Abort and save CSV if abs(roll or pitch) exceeds limit")
    parser.add_argument("--fall-height-ratio", type=float, default=FALL_COM_HEIGHT_RATIO,
                        help="Abort if CoM z < ratio * initial CoM z")
    args = parser.parse_args()
    if (args.video_fps <= 0 or args.video_width <= 0 or args.video_height <= 0
            or args.arrow_scale <= 0):
        parser.error('Video fps/dimensions and arrow-scale must be > 0')
    if args.duration <= 0 or args.max_fallbacks < 0:
        parser.error("duration must be > 0 and max-fallbacks must be >= 0")
    if (args.push_force < 0 or args.push_duration <= 0 or
            args.fall_angle_deg <= 0 or not 0 < args.fall_height_ratio < 1):
        parser.error("Invalid push strength/duration or fall diagnostic threshold")
    push_events = create_push_events(args)
    setup = initialize_simulation(
        xml_path=XML_PATH, horizon=N, dt_mpc=DT_MPC,
        dt_mpc_update=DT_MPC_UPDATE, home_key_name="home",
        qp_verbose=False, qp_eps_abs=QP_EPS_ABS,
        qp_eps_rel=QP_EPS_REL, qp_max_iter=QP_MAX_ITER,
        qp_polishing=True, qp_enable_warm_start=False,
        qp_constraint_tolerance=QP_CONSTRAINT_TOL,
        print_initialization=True,
    )
    model, data = setup.model, setup.data
    actuator_ids = get_actuator_ids(model)
    joint_addrs = get_actuator_joint_addresses(model, actuator_ids)
    site_ids = {name: get_site_id(model, name) for name in ("L6", "R6", "L6_IK", "R6_IK")}
    sole_models = build_sole_geometries(
        model, data, setup.contact_map, site_ids, args.sole_source)
    for side in ('L', 'R'):
        poly = sole_polygon_current(model, data, sole_models[side])
        print(f"[SOLE {side}] source={sole_models[side]['source']}, "
              f"area={polygon_area_xy(poly):.8f}m^2, "
              f"vertices={len(poly)}")
        if polygon_area_xy(poly) < MIN_SOLE_POLYGON_AREA_M2:
            raise RuntimeError(f'Sole {side} has degenerate polygon at initial pose')
    root_body_id = int(mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, PUSH_BODY_NAME))
    if root_body_id <= 0:
        raise RuntimeError(f"Cannot find floating-base body {PUSH_BODY_NAME!r}")
    floor_geom_id = int(setup.contact_map["floor_geom_id"])
    ground_z = float(data.geom_xpos[floor_geom_id, 2])
    X_target = np.asarray(setup.x_initial, dtype=float).reshape(13).copy()
    X_ref = np.tile(X_target, N)  # Same as repo run_standing.py: (N*13,)
    contact_schedule = build_contact_schedule(N)
    held_u0 = np.zeros(12)
    last_valid_u0 = None
    fail_streak = 0
    rows = []
    stop_reason = "completed"
    push_impulse_cumulative_world = np.zeros(3, dtype=float)
    push_applied_physics_steps = {e["id"]: 0 for e in push_events}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output if args.output.is_absolute() else (SIMULATION_DIR / args.output)
    prefix = f"standing_S1_push_{timestamp}" if push_events else f"standing_S1_baseline_{timestamp}"
    config = {
        "experiment": ("S1_DSP_standing_with_pulse" if push_events else "S1_DSP_standing_baseline"), "xml": str(XML_PATH),
        "push_enabled": int(bool(push_events)),
        "push_force_N": float(args.push_force) if push_events else 0.0,
        "push_duration_s": float(args.push_duration),
        "push_events": repr([
            {"id": e["id"], "time_s": e["start_s"], "duration_s": e["duration_s"],
             "force_world_N": e["force_world_N"].tolist()}
            for e in push_events]),
        "push_force_coordinate_frame": "MuJoCo world frame",
        "push_apply_point": "current whole-robot CoM via mj_applyFT on base_link",
        "push_unmodeled_feedforward": 1,
        "fall_angle_diagnostic_limit_deg": float(args.fall_angle_deg),
        "fall_com_height_diagnostic_ratio": float(args.fall_height_ratio),
        "duration_s": args.duration, "dt_sim_s": setup.dt_sim,
        "dt_prediction_s": DT_MPC, "dt_mpc_update_s": DT_MPC_UPDATE,
        "prediction_N": N, "mass_kg": setup.mass_CoM,
        "gravity_z_m_s2": float(model.opt.gravity[2]),
        "foot_width_assumed_m": FOOT_WIDTH_M,
        "foot_toe_distance_from_L6_m": FOOT_TOE_M,
        "foot_heel_distance_from_L6_m": FOOT_HEEL_M,
        "footprint_is_approx_rectangle": 1,
        "contact_friction_mu_for_reporting": QP_MU,
        "contact_min_load_for_cop_N": MIN_NORMAL_FORCE_FOR_COP_N,
        "qp_abs_tol": QP_EPS_ABS, "qp_rel_tol": QP_EPS_REL,
        "qp_postcheck_constraint_tol": QP_CONSTRAINT_TOL,
        "qp_max_iter": QP_MAX_ITER,
        "max_consecutive_solver_fallbacks": args.max_fallbacks,
        "Q_diagonal": repr(np.diag(Q).tolist()),
        "R_diagonal": repr(np.diag(R).tolist()),
        "note_actual_wrench": "Measured x_k before new tau_k; depends on preceding held command",
        "note_QP_contact": "Do not interpret CoP diagnostics as complete rectangular CWC enforcement",
        "note_DSP": "Planned [1,1] remains constant for all prediction steps",
        "note_support_polygon": "ACTIVE support uses current measured load flags, DSP/SSP/NONE; also exports PLANNED DSP",
        "sole_source_requested": args.sole_source,
        "sole_source_L_actual": sole_models["L"]["source"],
        "sole_source_R_actual": sole_models["R"]["source"],
        "sole_mesh_bottom_band_candidates_m": repr(SOLE_BAND_CANDIDATES_M),
        "support_polygon_measurement_threshold_N": CONTACT_LOAD_THRESHOLD_N,
        "note_mesh_geometry": "Low-facing compiled collision mesh vertices at initial home pose; attached to moving foot. Approximate sole, not exact manifold.",
        "note_contact_point_hull": "Convex hull of MuJoCo enforced contact-point positions with positive measured Fz; may be point/line, not true finite patch.",
    }
    print("\n=== S1 DSP STANDING: CSV DATA ACQUISITION ===")
    print(f"duration={args.duration}s, dt_sim={setup.dt_sim}s, dt_update={DT_MPC_UPDATE}s")
    print(f"output={output_dir}")
    print(f"Push schedule: {len(push_events)} pulses, force={args.push_force} N, "
          f"duration={args.push_duration} s")
    print("Push 1: +World Y, push 2: +World X (force is applied at whole-robot CoM)")
    for event in push_events:
        print(f"  id={event['id']}  t={event['start_s']:.3f}s "
              f"F_world={event['force_world_N'].tolist()} N  "
              f"J={np.linalg.norm(event['force_world_N']) * event['duration_s']:.5f} N.s")
    if abs(round(DT_MPC_UPDATE / setup.dt_sim) * setup.dt_sim - DT_MPC_UPDATE) > 1e-10:
        raise RuntimeError("dt_mpc_update must be an integer multiple of MuJoCo dt")
    steps_per_mpc = int(setup.steps_per_mpc_update)
    physics_step = 0
    mpc_tick = 0
    video = None
    video_path = output_dir / f'{prefix}_video.mp4'
    video_frames_path = output_dir / f'{prefix}_video_frames.csv'
    config.update({
        'video_requested': int(args.video),
        'video_fps': int(args.video_fps),
        'video_width': int(args.video_width),
        'video_height': int(args.video_height),
        'video_path': str(video_path) if args.video else '',
        'video_camera_azimuth_deg': VIDEO_CAMERA_AZIMUTH_DEG,
        'video_camera_elevation_deg': VIDEO_CAMERA_ELEVATION_DEG,
        'video_camera_distance_m': VIDEO_CAMERA_DISTANCE_M,
        'video_contactpoint': 1, 'video_contactforce': 1,
        'video_transparent': 1, 'video_center_of_mass': 1,
        'video_arrow_world_frame': 1,
        'video_arrow_scale_m_per_N': float(args.arrow_scale),
        'video_time_base': 'simulation time; fixed fps; not screen capture',
    })
    if args.video:
        video = SimulationVideo(
            model, data, root_body_id, video_path,
            fps=args.video_fps, width=args.video_width,
            height=args.video_height, duration=args.duration,
            arrow_scale=args.arrow_scale,
        )
        # Capture initial state t=0 before any mj_step.
        video.capture_due(data, np.zeros(3), 0)
        print(f'[VIDEO] MP4 recording started: {video_path} '
              f'({args.video_width}x{args.video_height}, {args.video_fps} FPS)')
    viewer_context = nullcontext(None)
    if args.viewer:
        # Import the viewer under an alias so 'mujoco' stays the global module
        # throughout main(). 'import mujoco.viewer' here would bind 'mujoco'
        # as a local variable and cause UnboundLocalError earlier in main().
        from mujoco import viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(model, data)

    try:
        with viewer_context as viewer:
            configure_viewer_visuals(viewer, model, data, root_body_id)
            while float(data.time) < float(args.duration) - setup.dt_sim * 0.1:
                if viewer is not None and not viewer.is_running():
                    stop_reason = "viewer_closed"
                    break
                wall_start = time.perf_counter()
                # Clear previous step's external generalized force, even at
                # MPC ticks where get_current_state_mpc() invokes mj_forward.
                data.qfrc_applied[:] = 0.0
                # The disturbance is chosen in world coordinates and is not
                # fed into the SRBD-MPC prediction model.
                push_force_world, push_event_id = push_at_time(
                    float(data.time), push_events)
                is_mpc_tick = (physics_step % steps_per_mpc) == 0
                solver_info = {}
                post_diag = None
                compute_time = np.nan
                qp_mode = "hold"
                stop_after_logging = False
                if is_mpc_tick:
                    clock = time.perf_counter()
                    current = get_current_state_mpc(model, data, setup.B_inertia_CoM)
                    Phi, Gamma = build_SRBD_model(
                        Ew_inverse=current.Ew_inverse,
                        mass_CoM=setup.mass_CoM,
                        W_inertia_CoM=current.W_inertia_CoM,
                        W_r_L=current.W_r_L, W_r_R=current.W_r_R,
                        dt_mpc=DT_MPC, horizon=N,
                    )
                    _, u0_opt, solver_info, post_diag = build_and_solve_QP(
                        qp_solver=setup.qp_solver,
                        Phi=Phi, Gamma=Gamma, x_current=current.x,
                        X_ref=X_ref, N=N,
                        L6_R_W=current.L6_IK_R_W,
                        R6_R_W=current.R6_IK_R_W,
                        contact_schedule=contact_schedule,
                        print_diagnostics=False,
                    )
                    if u0_opt is not None:
                        u_candidate = np.asarray(u0_opt, dtype=float).reshape(-1)
                        if u_candidate.shape == (12,) and np.all(np.isfinite(u_candidate)):
                            held_u0 = u_candidate.copy()
                            last_valid_u0 = u_candidate.copy()
                            fail_streak = 0
                            qp_mode = "solved"
                        else:
                            u0_opt = None
                    if u0_opt is None:
                        fail_streak += 1
                        if last_valid_u0 is not None and fail_streak <= args.max_fallbacks:
                            held_u0 = last_valid_u0.copy()
                            qp_mode = "fallback"
                        else:
                            qp_mode = "failed_stop"
                            stop_after_logging = True
                            stop_reason = "QP_no_valid_solution_or_fallback_limit"
                    compute_time = time.perf_counter() - clock

                # During the disturbance ensure forward kinematics and
                # subtree_com are synchronized with CURRENT x_k at every
                # physics step (not only the 200-Hz MPC ticks).
                if push_event_id and not is_mpc_tick:
                    mujoco.mj_forward(model, data)

                # At every physics tick: Jacobian(q_k)^T * held u_k -> tau_k.
                tau_raw, _, _ = get_joint_torque_from_MPC_wrench(model, data, held_u0)
                tau_raw = np.asarray(tau_raw, dtype=float).reshape(10)
                if not np.all(np.isfinite(tau_raw)):
                    raise RuntimeError("Non-finite joint torque from MPC wrench mapping")
                tau_limited = limited_motor_commands(model, tau_raw, actuator_ids)
                if not stop_after_logging:
                    # Preserve existing actuator_mapping; MuJoCo enforces XML ctrlrange.
                    apply_joint_torque(model, data, tau_raw)
                # Disturbance is applied at the CURRENT whole-robot CoM
                # before integration, after the torque command is set.
                # A force in world frame with zero direct torque at CoM.
                push_point_world = apply_force_at_robot_com(
                    model, data, root_body_id, push_force_world)

                if is_mpc_tick:
                    row = add_snapshot(
                        {}, model=model, data=data, current=current,
                        contact_map=setup.contact_map, u0=held_u0,
                        tau_raw=tau_raw, tau_limited=tau_limited,
                        actuator_ids=actuator_ids, joint_addrs=joint_addrs,
                        x_target=X_target, solver_info=solver_info,
                        post_diag=post_diag, compute_time=compute_time,
                        step_no=physics_step, qp_mode=qp_mode,
                        fallback_count=fail_streak,
                        mass=setup.mass_CoM, ground_z=ground_z,
                        site_ids=site_ids, sole_models=sole_models,
                        push_force_world=push_force_world,
                        push_event_id=push_event_id,
                        push_point_world=push_point_world,
                        push_impulse_cumulative_world=push_impulse_cumulative_world,
                        dt_sim=setup.dt_sim,
                    )
                    rows.append(row)
                    # Conservative diagnostic stop, not a formal balance
                    # criterion. Save CSV before leaving loop.
                    if (abs(row["roll_deg"]) >= args.fall_angle_deg
                            or abs(row["pitch_deg"]) >= args.fall_angle_deg
                            or row["com_z_m"] <=
                               args.fall_height_ratio * X_target[2]):
                        stop_reason = "fall_diagnostic_limit_exceeded"
                        stop_after_logging = True
                    if (mpc_tick % PRINT_EVERY_MPC_TICKS == 0) or qp_mode != "solved":
                        print(
                            f"t={data.time:7.3f}s  QP={str(solver_info.get('status','')):>22s} "
                            f"mode={qp_mode:<11s} "
                            f"roll={row['roll_deg']:+7.3f}deg "
                            f"pitch={row['pitch_deg']:+7.3f}deg "
                            f"contact={row['actual_contact_load_L']}{row['actual_contact_load_R']} "
                            f"Fz={row['actual_F_total_z_N']:.2f}N "
                            f"push={row['push_force_magnitude_N']:.2f}N"
                        )
                    mpc_tick += 1
                if stop_after_logging:
                    break
                # Exactly one physics integration from x_k under tau_k.
                mujoco.mj_step(model, data)
                # True executed impulse integrates at PHYSICS frequency.
                push_impulse_cumulative_world += push_force_world * float(setup.dt_sim)
                if push_event_id:
                    push_applied_physics_steps[push_event_id] += 1
                physics_step += 1
                # VIDEO: capture at fixed simulation-time FPS; no extra mj_step.
                if video is not None:
                    video.capture_due(data, push_force_world, push_event_id)
                # VIEWER: same COM and red push-arrow annotations as MP4.
                if (viewer is not None and
                        physics_step % max(1, int(round(0.02 / setup.dt_sim))) == 0):
                    sync_viewer_overlay(viewer, data, root_body_id,
                                        push_force_world, args.arrow_scale)
                if args.realtime:
                    remaining = float(setup.dt_sim) - (time.perf_counter() - wall_start)
                    if remaining > 0:
                        time.sleep(remaining)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception as exc:
        stop_reason = f"exception_{type(exc).__name__}: {exc}"
        print(f"\n[ERROR] {stop_reason}")
        raise
    finally:
        if video is not None:
            video.close()  # Flush MP4 file even on early termination.
            if video.frames:
                write_csv(video_frames_path, video.frames)
            config['video_frames_written'] = len(video.frames)
            config['video_close_error'] = video.last_error
        # Save partial data even when a QP failure or external exception occurs.
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_path = output_dir / f"{prefix}_timeseries.csv"
        sum_path = output_dir / f"{prefix}_summary.csv"
        cfg_path = output_dir / f"{prefix}_config.csv"
        if rows:
            write_csv(ts_path, rows)
        summary = make_summary(rows, stop_reason, args.duration, setup.mass_CoM, config)
        summary['video_enabled'] = int(args.video)
        summary['video_frames_written'] = int(len(video.frames)) if video is not None else 0
        summary['video_expected_frames'] = int(math.ceil(args.duration * args.video_fps)) if args.video else 0
        summary['video_path'] = str(video_path) if args.video else ''
        for axis, value in zip("xyz", push_impulse_cumulative_world):
            summary[f"push_impulse_applied_total_{axis}_Ns"] = float(value)
        summary["push_impulse_applied_total_norm_Ns"] = float(
            np.linalg.norm(push_impulse_cumulative_world))
        for event in push_events:
            summary[f"push_{event['id']}_actual_physics_steps"] = (
                push_applied_physics_steps[event["id"]])
        write_csv(sum_path, [summary])
        write_csv(cfg_path, [{"parameter": k, "value": v} for k, v in config.items()])
        print("\n" + "=" * 70)
        print("S1 OUTPUT")
        print("=" * 70)
        print("stop_reason:", stop_reason)
        print("Logged MPC samples:", len(rows))
        print("Time-series CSV:", ts_path if rows else "no samples")
        print("Summary CSV:", sum_path)
        print("Config CSV:", cfg_path)
        if video is not None:
            print('Video MP4:', video_path)
            print('Video frame timestamps CSV:', video_frames_path)
            print('Recorded video frames:', len(video.frames))
            if video.last_error:
                print('VIDEO CLOSE WARNING:', video.last_error)
        if rows:
            print("Full-DSP measured load fraction:", f"{summary.get('full_DSP_loaded_fraction',np.nan):.4f}")
            print("QP fallback ticks:", summary.get("solver_fallback_ticks", 0))
            print("Min CoM ACTIVE support margin (m):", summary.get("com_support_margin_m_min", np.nan))
            for mode in ('DSP', 'SSP_L', 'SSP_R', 'NONE'):
                print(f"support mode {mode} samples:", summary.get(f'active_support_mode_{mode}_ticks', 0))
            print("Applied impulse (world, N.s):", push_impulse_cumulative_world)
            print("Each push is a timed force, NOT an infinite instantaneous impulse.")


if __name__ == "__main__":
    main()
