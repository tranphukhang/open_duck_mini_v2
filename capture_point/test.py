"""Tests for RobotKinematics.compute_point_jacobian() on Open Duck Mini v5.

Put this file next to robot_kinematics.py in capture_point/ and run from
repository root:
    python capture_point/test_robot_kinematics_point_jacobian.py --verbose
    python capture_point/test_robot_kinematics_point_jacobian.py --samples 12
    python capture_point/test_robot_kinematics_point_jacobian.py --model-only
    python capture_point/test_robot_kinematics_point_jacobian.py --device cuda

Requirements: mujoco, numpy, torch, xmls/assets_v3/*.STL.
Reference models: xmls/open_duck_mini_v5.xml and scene_flat_terrian_v5.xml.

Jacobian column convention: [3 world-frame base translations,
3 world-frame base rotations (left perturbation), 10 hinge angles].
MuJoCo's hinge columns are compared using DOF IDs, not presumed ordering.
The base rotation columns are independently checked by world-frame finite
rotations of the freejoint quaternion rather than assuming a qvel convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Missing MuJoCo: pip install mujoco") from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robot_kinematics import RobotKinematics

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
SITE_NAMES = ("L6_IK", "R6_IK")
ATOL_ANALYTIC = 4e-5  # float32 vs MuJoCo float64, m/rad
ATOL_FD = 8e-5        # numeric derivative vs float32 analytic result, m/rad
FD_EPS = 1e-6         # perturbation applied to MuJoCo's double qpos


def xml_dir() -> Path:
    for folder in (HERE, *HERE.parents):
        path = folder / "xmls" / "open_duck_mini_v5.xml"
        if path.is_file():
            return path.parent
    raise FileNotFoundError(
        "Cannot find xmls/open_duck_mini_v5.xml. "
        "Place this test under <repository>/capture_point/."
    )


def numpy_value(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def check_close(actual, expected, label, atol=ATOL_ANALYTIC):
    a, b = numpy_value(actual), numpy_value(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label}: NaN or Inf found")
    try:
        np.testing.assert_allclose(a, b, rtol=0.0, atol=atol)
    except AssertionError as exc:
        maxerr = np.max(np.abs(a - b))
        raise AssertionError(
            f"{label}: max |error|={maxerr:.3e}, atol={atol:.1e}\n{exc}"
        ) from exc


def quat_to_mat(quat):
    flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(flat, np.asarray(quat, dtype=np.float64))
    return flat.reshape(3, 3)


def quat_multiply(a, b):
    """Hamilton product wxyz; used for world-frame orientation perturbation."""
    w1, v1 = float(a[0]), np.asarray(a[1:])
    w2, v2 = float(b[0]), np.asarray(b[1:])
    return np.concatenate((
        [w1 * w2 - np.dot(v1, v2)],
        w1 * v2 + w2 * v1 + np.cross(v1, v2),
    ))


def perturb_world_quat(quat, axis, amount):
    v = np.zeros(3, dtype=np.float64)
    v[axis] = np.sin(amount / 2)
    dq = np.concatenate(([np.cos(amount / 2)], v))
    return quat_multiply(dq, quat)


def make_cases(model, nrandom):
    ids = [model.joint(name).id for name in JOINT_NAMES]
    limits = np.asarray(model.jnt_range[ids], dtype=np.float64)
    q_home = np.deg2rad([0, -10, 20, -40, 20, 0, 10, -20, 40, -20])
    pos_home = np.array([0, 0, 0.275], dtype=np.float64)
    quat_home = np.array([1, 0, 0, 0], dtype=np.float64)
    base_qposadr = int(model.jnt_qposadr[model.joint("floating_base").id])
    for key_id in range(model.nkey):
        if model.key(key_id).name == "home":
            v = np.asarray(model.key_qpos[key_id], dtype=np.float64)
            pos_home = v[base_qposadr:base_qposadr + 3].copy()
            quat_home = v[base_qposadr + 3:base_qposadr + 7].copy()
            q_home = np.array([v[int(model.jnt_qposadr[j])] for j in ids])
            break
    cases = [
        ("home", pos_home, quat_home, q_home),
        ("zero", np.array([0.04, -0.03, 0.30]),
         np.array([1., 0., 0., 0.]), np.zeros(10)),
    ]
    rng = np.random.default_rng(20261114)
    for i in range(nrandom):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        q = limits[:, 0] + (limits[:, 1] - limits[:, 0]) * rng.uniform(
            0.15, 0.85, size=10
        )
        pos = rng.uniform([-0.25, -0.22, 0.22], [0.25, 0.22, 0.40])
        cases.append((f"random_{i:02d}", pos, quat, q))
    return cases


def torch_inputs(cases, device):
    base_pos = torch.tensor(
        np.stack([c[1] for c in cases]), device=device, dtype=torch.float32
    )
    base_rot = torch.tensor(
        np.stack([quat_to_mat(c[2]) for c in cases]),
        device=device, dtype=torch.float32,
    )
    q = torch.tensor(
        np.stack([c[3] for c in cases]), device=device, dtype=torch.float32
    )
    return base_pos, base_rot, q


def mujoco_data(model, case):
    """Independent world-frame position and Jacobian oracle."""
    _, position, quat, q = case
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    data.qpos[adr:adr + 3] = position
    data.qpos[adr + 3:adr + 7] = quat
    for j, name in enumerate(JOINT_NAMES):
        data.qpos[int(model.jnt_qposadr[model.joint(name).id])] = q[j]
    mujoco.mj_forward(model, data)
    return data


def site_jacobian(model, data, site_id):
    jp = np.zeros((3, model.nv), dtype=np.float64)
    jr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jp, jr, site_id)
    return jp


def site_position_for_case(model, case, site_id):
    data = mujoco_data(model, case)
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def fd_site_column(model, case, site_id, col, eps=FD_EPS):
    label, pos, quat, q = case
    p_plus, p_minus = pos.copy(), pos.copy()
    q_plus, q_minus = q.copy(), q.copy()
    quat_plus, quat_minus = quat.copy(), quat.copy()
    if col < 3:
        p_plus[col] += eps
        p_minus[col] -= eps
    elif col < 6:
        quat_plus = perturb_world_quat(quat, col - 3, +eps)
        quat_minus = perturb_world_quat(quat, col - 3, -eps)
    else:
        q_plus[col - 6] += eps
        q_minus[col - 6] -= eps
    plus = (label, p_plus, quat_plus, q_plus)
    minus = (label, p_minus, quat_minus, q_minus)
    return (
        site_position_for_case(model, plus, site_id)
        - site_position_for_case(model, minus, site_id)
    ) / (2.0 * eps)


class TestPointJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = cls.options
        cls.device = torch.device(args.device)
        names = ["open_duck_mini_v5.xml"]
        if not args.model_only:
            names.append("scene_flat_terrian_v5.xml")
        cls.models = []
        for name in names:
            path = xml_dir() / name
            model = mujoco.MjModel.from_xml_path(str(path))
            kin = RobotKinematics(model, device=args.device)
            cases = make_cases(model, args.samples)
            base_pos, base_rot, q = torch_inputs(cases, cls.device)
            ks = kin.forward_kinematics(base_pos, base_rot, q)
            jacs = [
                kin.compute_point_jacobian(
                    ks, ks.site_pos[:, s], kin.site_desc[s], base_pos
                )
                for s in range(2)
            ]
            cls.models.append((name, model, kin, cases, base_pos, base_rot, q, ks, jacs))

    def test_01_shape_dtype_device_finite(self):
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            with self.subTest(xml=name):
                for j in jacs:
                    self.assertEqual(j.shape, (len(cases), 3, 16))
                    self.assertEqual(j.dtype, torch.float32)
                    self.assertEqual(j.device, self.device)
                    self.assertTrue(torch.isfinite(j).all().item())

    def test_02_base_translation_identity(self):
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            with self.subTest(xml=name):
                for j in jacs:
                    check_close(
                        j[:, :, :3],
                        torch.eye(3, device=self.device).expand(len(cases), 3, 3),
                        "base translation columns", atol=1e-7,
                    )

    def test_03_opposite_leg_joint_columns_zero(self):
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            with self.subTest(xml=name):
                check_close(jacs[0][:, :, 11:16], np.zeros((len(cases), 3, 5)),
                            "left site / right joints", atol=0.0)
                check_close(jacs[1][:, :, 6:11], np.zeros((len(cases), 3, 5)),
                            "right site / left joints", atol=0.0)

    def test_04_joint_columns_vs_mujoco_site_jacobian(self):
        """Exact MuJoCo jacSite comparison for the 10 hinge joint columns."""
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            dofs = [int(model.jnt_dofadr[model.joint(n).id]) for n in JOINT_NAMES]
            for ci, case in enumerate(cases):
                data = mujoco_data(model, case)
                for leg, site_name in enumerate(SITE_NAMES):
                    sid = model.site(site_name).id
                    reference = site_jacobian(model, data, sid)[:, dofs]
                    check_close(
                        jacs[leg][ci, :, 6:], reference,
                        f"{name}:{case[0]}:{site_name}:hinge columns",
                    )

    def test_05_floating_translation_vs_mujoco_jacobian(self):
        """Translation part is world-aligned identity for this freejoint."""
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            free_id = model.joint("floating_base").id
            dofadr = int(model.jnt_dofadr[free_id])
            for ci, case in enumerate(cases):
                data = mujoco_data(model, case)
                for leg, site_name in enumerate(SITE_NAMES):
                    jac = site_jacobian(model, data, model.site(site_name).id)
                    check_close(
                        jacs[leg][ci, :, :3], jac[:, dofadr:dofadr + 3],
                        f"{name}:{case[0]}:{site_name}:translation",
                    )

    def test_06_base_rotation_world_finite_difference(self):
        """World-frame left quaternion perturbation checks columns 3:6."""
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            for ci, case in enumerate(cases):
                for leg, site_name in enumerate(SITE_NAMES):
                    sid = model.site(site_name).id
                    fd = np.column_stack([
                        fd_site_column(model, case, sid, col)
                        for col in range(3, 6)
                    ])
                    check_close(
                        jacs[leg][ci, :, 3:6], fd,
                        f"{name}:{case[0]}:{site_name}:base rotation FD",
                        atol=ATOL_FD,
                    )

    def test_07_hinge_finite_difference(self):
        """Independent numerical derivatives with respect to joint angles."""
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            for ci, case in enumerate(cases):
                for leg, site_name in enumerate(SITE_NAMES):
                    sid = model.site(site_name).id
                    affected = range(6, 11) if leg == 0 else range(11, 16)
                    fd = np.column_stack([
                        fd_site_column(model, case, sid, col)
                        for col in affected
                    ])
                    check_close(
                        jacs[leg][ci, :, affected.start:affected.stop], fd,
                        f"{name}:{case[0]}:{site_name}:joint FD",
                        atol=ATOL_FD,
                    )

    def test_08_batched_vs_individual(self):
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            for ci in range(len(cases)):
                ks_single = kin.forward_kinematics(
                    pos[ci:ci + 1], rot[ci:ci + 1], q[ci:ci + 1]
                )
                for leg in range(2):
                    single = kin.compute_point_jacobian(
                        ks_single, ks_single.site_pos[:, leg],
                        kin.site_desc[leg], pos[ci:ci + 1],
                    )
                    check_close(
                        jacs[leg][ci:ci + 1], single,
                        f"{name}:{ci}:site{leg}:batch vs single", atol=2e-6,
                    )

    def test_09_generic_point_on_intermediate_body(self):
        """Check API isn't specialized only for foot sites: left knee offset point."""
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            body_id = model.body("L4_knee").id
            bi = kin.bodies.index(body_id)
            offset = torch.tensor([0.015, -0.012, 0.008], device=self.device)
            points = ks.body_pos[:, bi] + (
                ks.body_rot[:, bi] @ offset.expand(len(cases), 3).unsqueeze(-1)
            ).squeeze(-1)
            J = kin.compute_point_jacobian(ks, points, kin.desc[:, bi], pos)
            self.assertEqual(J.shape, (len(cases), 3, 16))
            dofs = [int(model.jnt_dofadr[model.joint(n).id]) for n in JOINT_NAMES]
            for ci, case in enumerate(cases):
                data = mujoco_data(model, case)
                wpoint = np.asarray(data.xpos[body_id], dtype=np.float64) + (
                    np.asarray(data.xmat[body_id]).reshape(3, 3) @ numpy_value(offset)
                )
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, jacr, wpoint, body_id)
                check_close(
                    J[ci, :, 6:], jacp[:, dofs],
                    f"{name}:{case[0]}:generic knee point", atol=ATOL_ANALYTIC,
                )

    def test_10_pytorch_autograd_and_no_input_mutation(self):
        """Autograd of FK site position must reproduce hinge Jacobian columns."""
        for name, model, kin, cases, pos, rot, q, ks, jacs in self.models:
            p = pos[:2].detach().clone().requires_grad_(True)
            r = rot[:2].detach().clone()
            qq = q[:2].detach().clone().requires_grad_(True)
            p_before, q_before = p.detach().clone(), qq.detach().clone()
            state = kin.forward_kinematics(p, r, qq)
            for leg in range(2):
                J = kin.compute_point_jacobian(
                    state, state.site_pos[:, leg], kin.site_desc[leg], p
                )
                for axis in range(3):
                    gq, gp = torch.autograd.grad(
                        state.site_pos[:, leg, axis].sum(), (qq, p),
                        retain_graph=True,
                    )
                    check_close(
                        gq, J[:, axis, 6:],
                        f"{name}:autograd joint/site{leg}/xyz{axis}", atol=ATOL_ANALYTIC,
                    )
                    check_close(
                        gp, J[:, axis, :3],
                        f"{name}:autograd translation/site{leg}/xyz{axis}",
                        atol=ATOL_ANALYTIC,
                    )
            check_close(p, p_before, "base_pos was modified", atol=0)
            check_close(qq, q_before, "q was modified", atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--samples", default=6, type=int,
                        help="Number of deterministic random joint poses per model (default: 6)")
    parser.add_argument("--model-only", action="store_true",
                        help="Test only open_duck_mini_v5.xml, skip the flat scene")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    TestPointJacobian.options = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPointJacobian)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if result.wasSuccessful():
        print(
            "\n[PASS] compute_point_jacobian: 10 test groups; "
            f"{1 if args.model_only else 2} v5 model(s); "
            f"{args.samples + 2} configurations/model; device={args.device}."
        )
        print(
            "Verified 16-column floating-base site Jacobians: MuJoCo hinge "
            "columns, world-rotation finite differences, batch, knee point, autograd."
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
