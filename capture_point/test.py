"""Integration tests for RobotKinematics.compute_rotation_jacobian() (MJCF v5).

Place this file next to capture_point/robot_kinematics.py and run from the
repository root:

    python capture_point/test_robot_kinematics_rotation_jacobian.py --verbose
    python capture_point/test_robot_kinematics_rotation_jacobian.py --samples 20 --verbose
    python capture_point/test_robot_kinematics_rotation_jacobian.py --model-only
    python capture_point/test_robot_kinematics_rotation_jacobian.py --device cuda

Requires numpy, torch, mujoco and xmls/assets_v3/*.STL.

Column convention of the (B,3,16) angular Jacobian:
    0:3    base translation, world frame (zero)
    3:6    base infinitesimal rotation, world frame (identity)
    6:16   ten actuated hinges, world-frame angular axes

IMPORTANT: For the base orientation, the finite-difference perturbation is
R_new = Exp([dtheta_world]x) @ R. We deliberately do not compare the base
columns directly to MuJoCo's free-joint qvel columns without converting
its rotational-velocity convention.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Missing mujoco; install with: pip install mujoco") from exc

ROOT_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_HERE))
from robot_kinematics import KinState, RobotKinematics  # noqa: E402

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
SITE_NAMES = ("L6_IK", "R6_IK")
SEED = 20261114
TOL_MUJOCO = 5e-5
TOL_FD = 8e-5
TOL_TORCH_FD = 2e-4
EPS = 1e-6


def locate_xmls() -> Path:
    for parent in (ROOT_HERE, *ROOT_HERE.parents):
        candidate = parent / "xmls" / "open_duck_mini_v5.xml"
        if candidate.exists():
            return candidate.parent
    raise FileNotFoundError(
        "Cannot find xmls/open_duck_mini_v5.xml. Put this test in "
        "<repository>/capture_point/."
    )


def as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def check_close(got, expected, label, tol=TOL_MUJOCO):
    a, b = as_numpy(got), as_numpy(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f"{label}: non-finite values")
    error = float(np.max(np.abs(a - b)))
    if error > tol:
        raise AssertionError(
            f"{label}: max error {error:.6e} > {tol:.2e}; "
            f"got={a!r}, expected={b!r}"
        )


def rotation_matrix_from_quat(quat):
    flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, np.asarray(quat, dtype=np.float64))
    return flat.reshape(3, 3)


def quat_multiply(q1, q2):
    """Hamilton quaternion product; both quaternions use (w,x,y,z)."""
    w1, w2 = float(q1[0]), float(q2[0])
    v1 = np.asarray(q1[1:], dtype=np.float64)
    v2 = np.asarray(q2[1:], dtype=np.float64)
    return np.concatenate((
        [w1 * w2 - np.dot(v1, v2)],
        w1 * v2 + w2 * v1 + np.cross(v1, v2),
    ))


def perturb_quat_world(q, axis, angle):
    """Left-multiply by a small rotation in the planning/world frame."""
    v = np.zeros(3, dtype=np.float64)
    v[axis] = np.sin(angle / 2.0)
    dq = np.concatenate(([np.cos(angle / 2.0)], v))
    out = quat_multiply(dq, q)
    return out / np.linalg.norm(out)


def rotational_velocity(R_plus, R_minus, R_center, step):
    """omega_world = vee(skew((dR/dt) @ R.T))."""
    R_dot = (R_plus - R_minus) / (2.0 * step)
    W = R_dot @ R_center.T
    W = 0.5 * (W - W.T)
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def build_cases(model, n_random):
    joint_ids = [model.joint(name).id for name in JOINT_NAMES]
    limits = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    base_pos = np.array([0.0, 0.0, 0.275], dtype=np.float64)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_home = np.deg2rad(np.array([
        0.0, -10.0, 20.0, -40.0, 20.0,
        0.0, 10.0, -20.0, 40.0, -20.0,
    ], dtype=np.float64))
    free_qpos_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    for key_index in range(model.nkey):
        if model.key(key_index).name == "home":
            qpos_key = np.asarray(model.key_qpos[key_index], dtype=np.float64)
            base_pos = qpos_key[free_qpos_adr:free_qpos_adr + 3].copy()
            base_quat = qpos_key[free_qpos_adr + 3:free_qpos_adr + 7].copy()
            q_home = np.array([
                qpos_key[int(model.jnt_qposadr[jid])] for jid in joint_ids
            ], dtype=np.float64)
            break

    cases = [
        ("home", base_pos, base_quat, q_home),
        ("zero", np.array([0.02, -0.04, 0.32]),
         np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(10)),
    ]
    rng = np.random.default_rng(SEED)
    for i in range(n_random):
        q = limits[:, 0] + (limits[:, 1] - limits[:, 0]) * rng.uniform(
            0.1, 0.9, size=10
        )
        p = rng.uniform([-0.25, -0.20, 0.20], [0.25, 0.20, 0.40])
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        cases.append((f"random_{i:02d}", p, quat, q))
    return cases


def load_inputs(cases, device):
    p = torch.tensor(np.stack([c[1] for c in cases]),
                     dtype=torch.float32, device=device)
    R = torch.tensor(np.stack([rotation_matrix_from_quat(c[2]) for c in cases]),
                     dtype=torch.float32, device=device)
    q = torch.tensor(np.stack([c[3] for c in cases]),
                     dtype=torch.float32, device=device)
    return p, R, q


def mujoco_data_for_case(model, case):
    _name, base_pos, base_quat, q = case
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    data.qpos[free_adr:free_adr + 3] = base_pos
    data.qpos[free_adr + 3:free_adr + 7] = base_quat
    for j, joint_name in enumerate(JOINT_NAMES):
        jid = model.joint(joint_name).id
        data.qpos[int(model.jnt_qposadr[jid])] = q[j]
    mujoco.mj_forward(model, data)
    return data


def mujoco_site_orientation(model, case, site_name):
    data = mujoco_data_for_case(model, case)
    sid = model.site(site_name).id
    return np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()


def finite_difference_orientation(model, case, site_name, col, eps=EPS):
    """Independent MuJoCo FD for columns 0..15, world-frame convention."""
    label, p, quat, q = case
    pp, pm = p.copy(), p.copy()
    qp, qm = q.copy(), q.copy()
    rp, rm = quat.copy(), quat.copy()

    if col < 3:
        pp[col] += eps
        pm[col] -= eps
    elif col < 6:
        rp = perturb_quat_world(quat, col - 3, eps)
        rm = perturb_quat_world(quat, col - 3, -eps)
    else:
        qp[col - 6] += eps
        qm[col - 6] -= eps

    R0 = mujoco_site_orientation(model, case, site_name)
    R_plus = mujoco_site_orientation(
        model, (label + "+", pp, rp, qp), site_name
    )
    R_minus = mujoco_site_orientation(
        model, (label + "-", pm, rm, qm), site_name
    )
    return rotational_velocity(R_plus, R_minus, R0, eps)


class RotationJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = cls.options
        cls.device = torch.device(options.device)
        xml_dir = locate_xmls()
        model_files = ["open_duck_mini_v5.xml"]
        if not options.model_only:
            model_files.append("scene_flat_terrian_v5.xml")
        cls.datasets = []
        for filename in model_files:
            model = mujoco.MjModel.from_xml_path(str(xml_dir / filename))
            kin = RobotKinematics(model, device=options.device)
            cases = build_cases(model, options.samples)
            p, R, q = load_inputs(cases, cls.device)
            ks = kin.forward_kinematics(p, R, q)
            J_L = kin.compute_rotation_jacobian(ks, kin.site_desc[0])
            J_R = kin.compute_rotation_jacobian(ks, kin.site_desc[1])
            J_B = kin.compute_rotation_jacobian(ks, None)
            cls.datasets.append({
                "name": filename, "model": model, "kin": kin,
                "cases": cases, "p": p, "R": R, "q": q,
                "ks": ks, "jac": (J_L, J_R, J_B),
            })

    def test_01_output_shapes_types_and_finite(self):
        for d in self.datasets:
            for name, J in zip(("left", "right", "base"), d["jac"]):
                with self.subTest(model=d["name"], target=name):
                    self.assertEqual(J.shape, (len(d["cases"]), 3, 16))
                    self.assertEqual(J.dtype, torch.float32)
                    self.assertEqual(J.device, self.device)
                    self.assertTrue(torch.isfinite(J).all().item())

    def test_02_base_translation_zeros_and_base_rotation_identity(self):
        for d in self.datasets:
            count = len(d["cases"])
            eye = torch.eye(3, dtype=torch.float32, device=self.device)
            for name, J in zip(("left", "right", "base"), d["jac"]):
                check_close(J[:, :, 0:3], torch.zeros(count, 3, 3),
                            f'{d["name"]}/{name}/translation_zero', 1e-7)
                check_close(J[:, :, 3:6], eye.expand(count, 3, 3),
                            f'{d["name"]}/{name}/rotation_identity', 1e-7)

    def test_03_base_desc_none_has_zero_joint_columns(self):
        for d in self.datasets:
            check_close(d["jac"][2][:, :, 6:16],
                        np.zeros((len(d["cases"]), 3, 10)),
                        d["name"] + "/base_no_hinges", 1e-7)

    def test_04_feet_mask_and_opposite_leg_zero(self):
        for d in self.datasets:
            kin = d["kin"]
            check_close(kin.site_desc[0], np.array([1] * 5 + [0] * 5),
                        d["name"] + "/left_desc", 0)
            check_close(kin.site_desc[1], np.array([0] * 5 + [1] * 5),
                        d["name"] + "/right_desc", 0)
            check_close(d["jac"][0][:, :, 11:16],
                        np.zeros((len(d["cases"]), 3, 5)),
                        d["name"] + "/left_no_right_hinges", 0)
            check_close(d["jac"][1][:, :, 6:11],
                        np.zeros((len(d["cases"]), 3, 5)),
                        d["name"] + "/right_no_left_hinges", 0)

    def test_05_foot_site_rotation_matches_mujoco(self):
        """Check prerequisite that our FK and MuJoCo sites agree."""
        for d in self.datasets:
            for idx, case in enumerate(d["cases"]):
                for foot, site_name in enumerate(SITE_NAMES):
                    expected = mujoco_site_orientation(d["model"], case, site_name)
                    check_close(d["ks"].site_rot[idx, foot], expected,
                                f'{d["name"]}/{case[0]}/{site_name}/R',
                                TOL_MUJOCO)

    def test_06_hinge_columns_match_mujoco_jac_site(self):
        """MuJoCo angular Jacobian is the independent oracle for hinge DOFs."""
        for d in self.datasets:
            model = d["model"]
            dof_ids = [
                int(model.jnt_dofadr[model.joint(name).id])
                for name in JOINT_NAMES
            ]
            for idx, case in enumerate(d["cases"]):
                data = mujoco_data_for_case(model, case)
                for foot, site_name in enumerate(SITE_NAMES):
                    sid = model.site(site_name).id
                    jp = np.zeros((3, model.nv), dtype=np.float64)
                    jr = np.zeros((3, model.nv), dtype=np.float64)
                    mujoco.mj_jacSite(model, data, jp, jr, sid)
                    check_close(
                        d["jac"][foot][idx, :, 6:16], jr[:, dof_ids],
                        f'{d["name"]}/{case[0]}/{site_name}/mj_jacSite',
                        TOL_MUJOCO,
                    )

    def test_07_all_16_columns_finite_difference_mujoco(self):
        """For world-oriented base rotation, use quaternion LEFT perturbation."""
        for d in self.datasets:
            # home, zero, up to two rotated random configurations
            for idx, case in enumerate(d["cases"][:4]):
                for foot, site_name in enumerate(SITE_NAMES):
                    finite_J = np.stack([
                        finite_difference_orientation(
                            d["model"], case, site_name, col
                        ) for col in range(16)
                    ], axis=1)
                    check_close(
                        d["jac"][foot][idx], finite_J,
                        f'{d["name"]}/{case[0]}/{site_name}/FD_all_columns',
                        TOL_FD,
                    )

    def test_08_batched_equals_individual(self):
        for d in self.datasets:
            for idx, case in enumerate(d["cases"]):
                p = d["p"][idx:idx + 1]
                R = d["R"][idx:idx + 1]
                q = d["q"][idx:idx + 1]
                ks = d["kin"].forward_kinematics(p, R, q)
                for target in range(3):
                    desc = None if target == 2 else d["kin"].site_desc[target]
                    j_one = d["kin"].compute_rotation_jacobian(ks, desc)
                    check_close(
                        j_one[0], d["jac"][target][idx],
                        f'{d["name"]}/{case[0]}/batch_{target}',
                        1e-7,
                    )

    def test_09_rotation_of_planning_frame_covariance(self):
        """Under P->P', joint axes rotate, while base orientation block stays I."""
        angle = 0.43
        c, s = np.cos(angle), np.sin(angle)
        U = torch.tensor([
            [c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device).reshape(3, 3)
        offset = torch.tensor([0.15, -0.12, 0.03],
                              dtype=torch.float32, device=self.device)
        for d in self.datasets:
            new_p = (U @ d["p"].T).T + offset
            new_R = U.unsqueeze(0) @ d["R"]
            new_ks = d["kin"].forward_kinematics(new_p, new_R, d["q"])
            for target in range(3):
                desc = None if target == 2 else d["kin"].site_desc[target]
                J_new = d["kin"].compute_rotation_jacobian(new_ks, desc)
                J_old = d["jac"][target]
                check_close(J_new[:, :, 6:16], U @ J_old[:, :, 6:16],
                            f'{d["name"]}/rotated_axes_{target}', TOL_MUJOCO)
                check_close(J_new[:, :, 3:6],
                            np.broadcast_to(np.eye(3), (len(d["cases"]), 3, 3)),
                            f'{d["name"]}/base_eye_{target}', 1e-7)
                check_close(J_new[:, :, :3],
                            np.zeros((len(d["cases"]), 3, 3)),
                            f'{d["name"]}/base_zeros_{target}', 1e-7)

    def test_10_torch_finite_difference_joint_rotations(self):
        """FD on our own FK ensures consistency with differentiable rotation."""
        for d in self.datasets:
            for idx in range(min(3, len(d["cases"]))):
                p = d["p"][idx:idx + 1]
                rot = d["R"][idx:idx + 1]
                q = d["q"][idx:idx + 1]
                for foot in range(2):
                    R0 = as_numpy(d["ks"].site_rot[idx, foot]).astype(np.float64)
                    for col in range(10):
                        dq = torch.zeros_like(q)
                        dq[:, col] = 0.002  # stable for float32 FK
                        kp = d["kin"].forward_kinematics(p, rot, q + dq)
                        km = d["kin"].forward_kinematics(p, rot, q - dq)
                        r_plus = as_numpy(kp.site_rot[0, foot]).astype(np.float64)
                        r_minus = as_numpy(km.site_rot[0, foot]).astype(np.float64)
                        numeric = rotational_velocity(r_plus, r_minus, R0, 0.002)
                        expected = d["jac"][foot][idx, :, 6 + col]
                        check_close(numeric, expected,
                                    f'{d["name"]}/torch_FD_{idx}_{foot}_{col}',
                                    TOL_TORCH_FD)

    def test_11_gradient_through_fk_rotation(self):
        """Autograd jvp -> spatial angular velocity at a sample configuration."""
        for d in self.datasets:
            p = d["p"][:1].detach()
            R_base = d["R"][:1].detach()
            q = d["q"][:1].detach().clone().requires_grad_(True)
            for foot in range(2):
                def site_rot_from_q(qq):
                    return d["kin"].forward_kinematics(p, R_base, qq).site_rot[0, foot]

                R_current = site_rot_from_q(q).detach()
                for joint in range(10):
                    direction = torch.zeros_like(q)
                    direction[0, joint] = 1.0
                    _R_val, dR = torch.autograd.functional.jvp(
                        site_rot_from_q, (q,), (direction,), create_graph=False
                    )
                    W = dR @ R_current.T
                    W = (W - W.T) * 0.5
                    omega = torch.stack([W[2, 1], W[0, 2], W[1, 0]])
                    check_close(
                        omega, d["jac"][foot][0, :, 6 + joint],
                        f'{d["name"]}/autograd_foot{foot}_joint{joint}',
                        4e-5,
                    )

    def test_12_custom_mask_and_does_not_modify_input(self):
        """A custom 10-element mask must multiply exactly the axis columns."""
        for d in self.datasets:
            ks = d["ks"]
            before = ks.axis.detach().clone()
            mask = torch.tensor([1.0, 0.0, 0.25, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.8],
                                device=self.device)
            mask_before = mask.clone()
            J = d["kin"].compute_rotation_jacobian(ks, mask)
            expected = (ks.axis * mask[None, :, None]).transpose(1, 2)
            check_close(J[:, :, 6:16], expected,
                        d["name"] + "/custom_mask", 1e-7)
            check_close(ks.axis, before,
                        d["name"] + "/input_axis_unchanged", 0)
            check_close(mask, mask_before,
                        d["name"] + "/input_mask_unchanged", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6,
                        help="Number of seeded random configurations (default: 6)")
    parser.add_argument("--model-only", action="store_true",
                        help="Test only the standalone v5 MJCF, not the scene")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA not available")
    RotationJacobianTests.options = args
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(RotationJacobianTests)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(tests)
    if result.wasSuccessful():
        print(
            "\n[PASS] compute_rotation_jacobian(): 12 test groups; "
            f"{1 if args.model_only else 2} model(s), "
            f"{args.samples + 2} configurations/model, device={args.device}."
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
