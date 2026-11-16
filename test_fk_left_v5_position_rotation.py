
"""Validate left-foot position and rotation FK against open_duck_mini_v5.xml.

Run from repository root:
    python test_fk_left_v5_position_rotation.py --xml xmls/open_duck_mini_v5.xml

Only evaluates kinematics (mj_forward), never dynamics.
Position and rotation are expressed in the base_link body frame {B}.
The foot frame {E_L} is MuJoCo site 'L6' (not 'L6_IK').
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np


# Geometric lengths from xmls/open_duck_mini_v5.xml (meters)
L2 = 0.08286  # base_link -> L0, z-offset magnitude
L3 = 0.03500  # base_link -> L0, lateral offset
L4 = 0.01606  # L1_hip_yaw -> L2_hip_roll
L5 = 0.05570  # L2_hip_roll -> L3_hip_pitch
L6 = 0.07865  # L3_hip_pitch -> L4_knee
L7 = 0.07865  # L4_knee -> L5_ankle
L8 = 0.03880  # L5_ankle -> site L6

LEFT_JOINTS = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle"
)


def mdh(a: float, alpha_deg: float, d: float, q_rad: float) -> np.ndarray:
    """Same convention as MATLAB MDH(a, alpha_deg, d, q)."""
    ca, sa = np.cos(np.deg2rad(alpha_deg)), np.sin(np.deg2rad(alpha_deg))
    cq, sq = np.cos(q_rad), np.sin(q_rad)
    return np.array([
        [cq,       -sq,       0.,       a],
        [sq * ca,  cq * ca,  -sa,      -d * sa],
        [sq * sa,  cq * sa,   ca,       d * ca],
        [0.,        0.,       0.,       1.],
    ], dtype=float)


def fk_left_mdh_transform(q_rad: np.ndarray) -> np.ndarray:
    """Return the full transform ^B T_{E_L} from the MDH chain."""
    q1, q2, q3, q4, q5 = q_rad
    B_T_0L = np.array([
        [0., -1., 0.,  0.],
        [1.,  0., 0.,  L3],
        [0.,  0., 1., -L2],
        [0.,  0., 0.,  1.],
    ])
    return (B_T_0L
            @ mdh(0, 0, 0, q1)
            @ mdh(0, 90, -L4, q2 - np.pi / 2)
            @ mdh(0, -90, L5, q3)
            @ mdh(L6, 0, 0, q4)
            @ mdh(L7, 0, 0, q5)
            @ mdh(L8, 0, 0, 0))


def fk_left_mdh(q_rad: np.ndarray) -> np.ndarray:
    """Position of {E_L}, expressed in {B}."""
    return fk_left_mdh_transform(q_rad)[:3, 3].copy()


def fk_left_closed_form(q_rad: np.ndarray) -> np.ndarray:
    """Previously simplified left-foot x/y/z position formulas."""
    q1, q2, q3, q4, q5 = q_rad
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    t3, t34, t345 = q3, q3 + q4, q3 + q4 + q5

    x = (-L4*c1 - L5*s1*c2
         - L6*(c1*np.sin(t3) + s1*s2*np.cos(t3))
         - L7*(c1*np.sin(t34) + s1*s2*np.cos(t34))
         - L8*(c1*np.sin(t345) + s1*s2*np.cos(t345)))
    y = (L3 - L4*s1 + L5*c1*c2
         + L6*(c1*s2*np.cos(t3) - s1*np.sin(t3))
         + L7*(c1*s2*np.cos(t34) - s1*np.sin(t34))
         + L8*(c1*s2*np.cos(t345) - s1*np.sin(t345)))
    z = (-L2 + L5*s2
         - c2*(L6*np.cos(t3) + L7*np.cos(t34) + L8*np.cos(t345)))
    return np.array([x, y, z], dtype=float)


def fk_left_closed_form_rotation(q_rad: np.ndarray) -> np.ndarray:
    """All nine simplified terms r_ij,L of ^B R_{E_L}."""
    q1, q2, q3, q4, q5 = q_rad
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    theta = q3 + q4 + q5
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [-c1*st - s1*s2*ct,  -c1*ct + s1*s2*st,  -s1*c2],
        [ c1*s2*ct - s1*st,  -s1*ct - c1*s2*st,   c1*c2],
        [-c2*ct,              c2*st,              s2],
    ], dtype=float)


def rpy_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
    """Quaternion q = q_z(yaw) * q_y(pitch) * q_x(roll), MuJoCo wxyz."""
    r, p, y = 0.5 * np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def set_configuration(model, data, angles_deg, base_rpy_deg=(0., 0., 0.)):
    """Set qpos directly and update kinematics without stepping dynamics."""
    mujoco.mj_resetData(model, data)

    free_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base"
    )
    if free_id == -1:
        raise ValueError("Cannot find floating_base joint")
    free_adr = int(model.jnt_qposadr[free_id])
    data.qpos[free_adr:free_adr+3] = [0.0, 0.0, 0.28]
    data.qpos[free_adr+3:free_adr+7] = rpy_deg_to_quat_wxyz(*base_rpy_deg)

    q_rad = np.deg2rad(angles_deg)
    for name, q in zip(LEFT_JOINTS, q_rad):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1:
            raise ValueError(f"Cannot find joint {name}")
        data.qpos[int(model.jnt_qposadr[j_id])] = q

    mujoco.mj_forward(model, data)
    return q_rad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml", type=Path, default=Path("xmls/open_duck_mini_v5.xml")
    )
    parser.add_argument(
        "--tol", type=float, default=1e-8,
        help="Position tolerance (m)"
    )
    parser.add_argument(
        "--rot-tol", type=float, default=1e-8,
        help="Rotation matrix element tolerance (dimensionless)"
    )
    args = parser.parse_args()

    if not args.xml.is_file():
        parser.error(f"XML not found: {args.xml.resolve()} (run from repo root)")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L6")
    if base_id == -1 or foot_id == -1:
        raise ValueError("Required body base_link or site L6 not found")

    # Case format: (name, 5 joint angles [deg], base [roll, pitch, yaw] [deg])
    cases = [
        ("zero",         [0, 0, 0, 0, 0],          (0, 0, 0)),
        ("home",         [0, -10, 20, -40, 20],    (0, 0, 0)),
        ("yaw10",        [10, -10, 20, -40, 20],  (0, 0, 0)),
        ("mixed_1",      [-10, 10, -15, 30, -10], (0, 0, 0)),
        ("mixed_2",      [15, 5, 25, -20, 5],    (0, 0, 0)),
        ("rotated_base", [10, -10, 20, -40, 20], (0, 0, 30)),
        ("base_3d",      [10, -10, 20, -40, 20], (10, -15, 30)),
    ]

    passed = True
    print(f"Model: {args.xml}")
    print("Comparison frame: {B} (base_link); foot frame: site L6")
    print(f"Tolerances: position {args.tol:.1e} m; rotation {args.rot_tol:.1e}")

    for name, angles_deg, base_rpy_deg in cases:
        q_rad = set_configuration(model, data, angles_deg, base_rpy_deg)

        # MuJoCo reports site position and site orientation in world frame {W}.
        W_p_EL = data.site_xpos[foot_id].copy()
        W_R_EL = data.site_xmat[foot_id].reshape(3, 3).copy()
        W_p_B = data.xpos[base_id].copy()
        W_R_B = data.xmat[base_id].reshape(3, 3).copy()

        # Express BOTH position and orientation in {B}.
        B_p_EL_mj = W_R_B.T @ (W_p_EL - W_p_B)
        B_R_EL_mj = W_R_B.T @ W_R_EL

        # MDH and manually simplified formulas.
        B_T_EL_mdh = fk_left_mdh_transform(q_rad)
        B_p_EL_mdh = B_T_EL_mdh[:3, 3]
        B_R_EL_mdh = B_T_EL_mdh[:3, :3]
        B_p_EL_closed = fk_left_closed_form(q_rad)
        B_R_EL_closed = fk_left_closed_form_rotation(q_rad)

        # Errors: MDH versus MuJoCo, and closed-form versus MDH.
        e_xyz = B_p_EL_mdh - B_p_EL_mj
        e_R = B_R_EL_mdh - B_R_EL_mj
        max_pos_error = np.max(np.abs(e_xyz))
        max_rot_error = np.max(np.abs(e_R))
        max_pos_formula_error = np.max(np.abs(B_p_EL_mdh - B_p_EL_closed))
        max_rot_formula_error = np.max(np.abs(B_R_EL_mdh - B_R_EL_closed))

        # Verify that the derived rotation matrix is a proper rotation.
        orth_error = np.max(np.abs(B_R_EL_closed.T @ B_R_EL_closed - np.eye(3)))
        determinant_error = abs(np.linalg.det(B_R_EL_closed) - 1.0)

        ok_pos = max_pos_error < args.tol and max_pos_formula_error < args.tol
        ok_rot = (max_rot_error < args.rot_tol
                  and max_rot_formula_error < args.rot_tol
                  and orth_error < args.rot_tol
                  and determinant_error < args.rot_tol)
        ok = ok_pos and ok_rot
        passed &= ok

        print(f"\n{name}: q(deg)={angles_deg}; base RPY(deg)={base_rpy_deg}")
        print("  --- Position in {B} [m] ---")
        print(f"  MuJoCo  = {np.array2string(B_p_EL_mj, precision=9)}")
        print(f"  MDH     = {np.array2string(B_p_EL_mdh, precision=9)}")
        print(f"  Closed  = {np.array2string(B_p_EL_closed, precision=9)}")
        print(f"  e_xyz   = {np.array2string(e_xyz, precision=3)}")
        print(f"  max |e_p| = {max_pos_error:.3e} m; "
              f"closed-vs-MDH = {max_pos_formula_error:.3e} m | "
              f"{'PASS' if ok_pos else 'FAIL'}")

        print("  --- Rotation in {B} (3x3) ---")
        print("  MuJoCo:\n", np.array2string(B_R_EL_mj, precision=7, suppress_small=True))
        print("  MDH:\n", np.array2string(B_R_EL_mdh, precision=7, suppress_small=True))
        print("  Closed:\n", np.array2string(B_R_EL_closed, precision=7, suppress_small=True))
        print(f"  max |e_R| = {max_rot_error:.3e}; "
              f"closed-vs-MDH = {max_rot_formula_error:.3e}")
        print(f"  orthogonality = {orth_error:.3e}; "
              f"|det(R)-1| = {determinant_error:.3e} | "
              f"{'PASS' if ok_rot else 'FAIL'}")
        print(f"  CASE: {'PASS' if ok else 'FAIL'}")

    print(f"\nFINAL: {'PASS' if passed else 'FAIL'} ")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
