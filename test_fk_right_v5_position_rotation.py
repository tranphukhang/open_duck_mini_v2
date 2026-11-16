"""Validate right-foot FK position and rotation against MuJoCo V5.

From the repository root:
    python test_fk_right_v5_position_rotation.py --xml xmls/open_duck_mini_v5.xml

Or, if this file is placed in tests/:
    python tests/test_fk_right_v5_position_rotation.py --xml xmls/open_duck_mini_v5.xml

Evaluates only kinematics with mj_forward(): never advances dynamics.
Frame {B}: body 'base_link'.  Foot frame {E_R}: site 'R6'.
Does not use R6_IK or any other _IK site.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np


# Lengths from xmls/open_duck_mini_v5.xml, in meters.
L2 = 0.08286
L3 = 0.03500
L4 = 0.01606
L5 = 0.05570
L6 = 0.07865
L7 = 0.07865
L8 = 0.03880

RIGHT_JOINTS = (
    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "R_ankle",
)


def mdh(a: float, alpha_deg: float, d: float, q_rad: float) -> np.ndarray:
    """Match MATLAB MDH(a, alpha_deg, d, q): alpha in deg, q in rad."""
    ca = np.cos(np.deg2rad(alpha_deg))
    sa = np.sin(np.deg2rad(alpha_deg))
    cq, sq = np.cos(q_rad), np.sin(q_rad)
    return np.array([
        [cq,      -sq,      0.0,        a],
        [sq * ca,  cq * ca, -sa,      -d * sa],
        [sq * sa,  cq * sa,  ca,       d * ca],
        [0.0,      0.0,     0.0,      1.0],
    ], dtype=float)


def fk_right_mdh_transform(q_rad: np.ndarray) -> np.ndarray:
    """Full homogeneous transform ^B T_{E_R} from the right-leg MDH table."""
    q1, q2, q3, q4, q5 = q_rad
    B_T_0R = np.array([
        [0.0,  1.0, 0.0,  0.0],
        [-1.0, 0.0, 0.0, -L3],
        [0.0,  0.0, 1.0, -L2],
        [0.0,  0.0, 0.0,  1.0],
    ], dtype=float)
    return (B_T_0R
            @ mdh(0, 0, 0, q1)
            @ mdh(0, -90, -L4, q2 + np.pi / 2)
            @ mdh(0, 90, L5, q3)
            @ mdh(L6, 0, 0, q4)
            @ mdh(L7, 0, 0, q5)
            @ mdh(L8, 0, 0, 0))


def fk_right_closed_position(q_rad: np.ndarray) -> np.ndarray:
    """Three previously derived right-foot position expressions in {B}."""
    q1, q2, q3, q4, q5 = q_rad
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    t34 = q3 + q4
    t345 = t34 + q5

    x = (-L4 * c1 + L5 * s1 * c2
         + L6 * (c1 * np.sin(q3) - s1 * s2 * np.cos(q3))
         + L7 * (c1 * np.sin(t34) - s1 * s2 * np.cos(t34))
         + L8 * (c1 * np.sin(t345) - s1 * s2 * np.cos(t345)))

    y = (-L3 - L4 * s1 - L5 * c1 * c2
         + L6 * (s1 * np.sin(q3) + c1 * s2 * np.cos(q3))
         + L7 * (s1 * np.sin(t34) + c1 * s2 * np.cos(t34))
         + L8 * (s1 * np.sin(t345) + c1 * s2 * np.cos(t345)))

    z = (-L2 - L5 * s2
         - c2 * (L6 * np.cos(q3)
                 + L7 * np.cos(t34)
                 + L8 * np.cos(t345)))

    return np.array([x, y, z], dtype=float)


def fk_right_closed_rotation(q_rad: np.ndarray) -> np.ndarray:
    """All nine simplified r_ij,R entries of ^B R_{E_R}."""
    q1, q2, q3, q4, q5 = q_rad
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    theta = q3 + q4 + q5
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ c1 * st - s1 * s2 * ct,
          c1 * ct + s1 * s2 * st,
          s1 * c2],
        [ s1 * st + c1 * s2 * ct,
          s1 * ct - c1 * s2 * st,
         -c1 * c2],
        [-c2 * ct,
          c2 * st,
         -s2],
    ], dtype=float)


def rpy_deg_to_quat_wxyz(rpy_deg) -> np.ndarray:
    """Base orientation Rz(yaw)*Ry(pitch)*Rx(roll), MuJoCo w,x,y,z."""
    r, p, y = 0.5 * np.deg2rad(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)


def set_configuration(model, data, angles_deg, base_rpy_deg=(0, 0, 0)):
    """Assign the right-leg angles and floating-base pose; only call mj_forward."""
    mujoco.mj_resetData(model, data)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    if free_id < 0:
        raise ValueError("Missing free joint: floating_base")
    free_adr = int(model.jnt_qposadr[free_id])
    data.qpos[free_adr:free_adr + 3] = [0.0, 0.0, 0.28]
    data.qpos[free_adr + 3:free_adr + 7] = rpy_deg_to_quat_wxyz(base_rpy_deg)

    q_rad = np.deg2rad(angles_deg)
    for name, q in zip(RIGHT_JOINTS, q_rad):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Missing joint: {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = q

    mujoco.mj_forward(model, data)
    return q_rad


def max_abs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path,
                        default=Path("xmls/open_duck_mini_v5.xml"))
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Position error tolerance, meters")
    parser.add_argument("--rot-tol", type=float, default=1e-8,
                        help="Rotation matrix element error tolerance")
    parser.add_argument("--random-cases", type=int, default=0,
                        help="Additional random right-joint configurations (default 0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.xml.is_file():
        parser.error(f"XML not found: {args.xml.resolve()} (run from repository root)")
    if args.tol <= 0 or args.rot_tol <= 0 or args.random_cases < 0:
        parser.error("--tol and --rot-tol must be positive; --random-cases >= 0")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "R6")
    if base_id < 0 or foot_id < 0:
        raise ValueError("Required body 'base_link' or site 'R6' is missing")

    # Joint angle order: right hip yaw, hip roll, hip pitch, knee, ankle [deg].
    # The right-leg home pose in scene_flat_terrian_v5.xml is [0, 10, -20, 40, -20].
    cases = [
        ("zero",          [0, 0, 0, 0, 0],           (0, 0, 0)),
        ("home",          [0, 10, -20, 40, -20],     (0, 0, 0)),
        ("yaw10",         [10, 10, -20, 40, -20],    (0, 0, 0)),
        ("mixed_1",       [-10, -10, 15, -30, 10],  (0, 0, 0)),
        ("mixed_2",       [15, 5, -25, 20, -5],     (0, 0, 0)),
        ("rotated_base",  [10, 10, -20, 40, -20],   (0, 0, 30)),
        ("base_3d",       [10, 10, -20, 40, -20],   (10, -15, 30)),
    ]

    # Optional extra tests, within right-leg ranges in the V5 XML.
    if args.random_cases:
        rng = np.random.default_rng(args.seed)
        low = [-30, -25, -70, -90, -90]
        high = [30, 25, 30, 90, 90]
        for idx in range(args.random_cases):
            q = rng.uniform(low, high).tolist()
            rpy = rng.uniform([-20, -20, -40], [20, 20, 40]).tolist()
            cases.append((f"random_{idx+1:03d}", q, rpy))

    print(f"Model: {args.xml}")
    print("Foot frame: R6; base frame: base_link")
    print("All positions and rotation matrices are expressed in {B}.")
    print(f"Tolerances: position {args.tol:.1e} m, rotation {args.rot_tol:.1e}")

    passed = 0
    largest_pos = 0.0
    largest_rot = 0.0
    for name, angles_deg, base_rpy_deg in cases:
        q_rad = set_configuration(model, data, angles_deg, base_rpy_deg)

        # MuJoCo frame conversions: W -> B.
        W_p_B = data.xpos[base_id].copy()
        W_R_B = data.xmat[base_id].reshape(3, 3).copy()
        W_p_ER = data.site_xpos[foot_id].copy()
        W_R_ER = data.site_xmat[foot_id].reshape(3, 3).copy()
        B_p_ER_mj = W_R_B.T @ (W_p_ER - W_p_B)
        B_R_ER_mj = W_R_B.T @ W_R_ER

        # Direct MDH chain vs. closed-form equations.
        B_T_ER_mdh = fk_right_mdh_transform(q_rad)
        B_p_ER_mdh = B_T_ER_mdh[:3, 3]
        B_R_ER_mdh = B_T_ER_mdh[:3, :3]
        B_p_ER_closed = fk_right_closed_position(q_rad)
        B_R_ER_closed = fk_right_closed_rotation(q_rad)

        e_p = B_p_ER_mdh - B_p_ER_mj
        e_R = B_R_ER_mdh - B_R_ER_mj
        pos_err = max_abs(e_p)
        rot_err = max_abs(e_R)
        pos_formula_err = max_abs(B_p_ER_closed - B_p_ER_mdh)
        rot_formula_err = max_abs(B_R_ER_closed - B_R_ER_mdh)
        orth_err = max_abs(B_R_ER_closed.T @ B_R_ER_closed - np.eye(3))
        det_err = abs(np.linalg.det(B_R_ER_closed) - 1.0)

        ok_p = pos_err < args.tol and pos_formula_err < args.tol
        ok_R = (rot_err < args.rot_tol
                and rot_formula_err < args.rot_tol
                and orth_err < args.rot_tol
                and det_err < args.rot_tol)
        ok = ok_p and ok_R
        passed += int(ok)
        largest_pos = max(largest_pos, pos_err)
        largest_rot = max(largest_rot, rot_err)

        print(f"\n{name}: q(deg)={np.round(angles_deg, 3).tolist()}, "
              f"base RPY(deg)={np.round(base_rpy_deg, 3).tolist()}")
        print("  --- Position [m] ---")
        print("  MuJoCo =", np.array2string(B_p_ER_mj, precision=9))
        print("  MDH    =", np.array2string(B_p_ER_mdh, precision=9))
        print("  Closed =", np.array2string(B_p_ER_closed, precision=9))
        print("  e_xyz  =", np.array2string(e_p, precision=3))
        print(f"  max |e_p|={pos_err:.3e} m | closed-vs-MDH={pos_formula_err:.3e} m "
              f"| {'PASS' if ok_p else 'FAIL'}")

        print("  --- Rotation [3x3] ---")
        print("  MuJoCo:\n", np.array2string(B_R_ER_mj, precision=7, suppress_small=True))
        print("  MDH:\n", np.array2string(B_R_ER_mdh, precision=7, suppress_small=True))
        print("  Closed:\n", np.array2string(B_R_ER_closed, precision=7, suppress_small=True))
        print(f"  max |e_R|={rot_err:.3e} | closed-vs-MDH={rot_formula_err:.3e}")
        print(f"  orthogonality={orth_err:.3e} | |det(R)-1|={det_err:.3e} "
              f"| {'PASS' if ok_R else 'FAIL'}")
        print(f"  CASE: {'PASS' if ok else 'FAIL'}")

    print(f"\nFINAL: {'PASS' if passed == len(cases) else 'FAIL'} "
          f"({passed}/{len(cases)} cases)")
    print(f"Max position error = {largest_pos:.3e} m")
    print(f"Max rotation-entry error = {largest_rot:.3e}")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
