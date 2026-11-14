"""Integration tests for RobotKinematics.forward_kinematics() using MuJoCo v5.

Place next to capture_point/robot_kinematics.py, with xmls/ at repository root.
Run from the repository root:
    python capture_point/test_robot_kinematics_fk.py --verbose
    python capture_point/test_robot_kinematics_fk.py --samples 20 --verbose
    python capture_point/test_robot_kinematics_fk.py --model-only
    python capture_point/test_robot_kinematics_fk.py --device cuda

Requirements: mujoco, torch, numpy and xmls/assets_v3/*.STL.
This test exercises FK + KinState, not Jacobian, IK, or dynamics.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("MuJoCo is required: pip install mujoco") from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robot_kinematics import KinState, RobotKinematics

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
SITE_NAMES = ("L6_IK", "R6_IK")
FIELD_SHAPES = {
    "body_pos": (11, 3),
    "body_rot": (11, 3, 3),
    "com_body": (11, 3),
    "com": (3,),
    "site_pos": (2, 3),
    "site_rot": (2, 3, 3),
    "axis": (10, 3),
    "anchor": (10, 3),
}
# Float32 calculations compared with MuJoCo float64, in m or dimensionless.
TOL = {
    "body_pos": 5e-5,
    "body_rot": 7e-5,
    "com_body": 5e-5,
    "com": 5e-5,
    "site_pos": 5e-5,
    "site_rot": 7e-5,
    "axis": 7e-5,
    "anchor": 5e-5,
}


def xml_folder() -> Path:
    for parent in (HERE, *HERE.parents):
        folder = parent / "xmls"
        if (folder / "open_duck_mini_v5.xml").is_file():
            return folder
    raise FileNotFoundError(
        "Cannot find xmls/open_duck_mini_v5.xml. "
        "Keep test file under capture_point/ in the repository."
    )


def arr(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def assert_near(actual, expected, label: str, atol: float = 5e-5):
    a, b = arr(actual), arr(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape mismatch {a.shape} != {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label}: NaN or Inf present")
    try:
        np.testing.assert_allclose(a, b, atol=atol, rtol=0)
    except AssertionError as exc:
        difference = float(np.max(np.abs(a - b)))
        raise AssertionError(
            f"{label}: max_abs_error={difference:.6g}; atol={atol:.2g}\n{exc}"
        ) from exc


def quaternion_to_matrix(quat_wxyz):
    out = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(out, np.asarray(quat_wxyz, dtype=np.float64))
    return out.reshape(3, 3)


def build_cases(model, samples: int):
    rng = np.random.default_rng(20261114)
    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    joint_ids = [model.joint(name).id for name in JOINT_NAMES]

    # Scene v5 has keyframe "home"; plain model v5 may not.
    home_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    for i in range(model.nkey):
        if model.key(i).name == "home":
            home_qpos = np.asarray(model.key_qpos[i], dtype=np.float64).copy()
            break

    home = {
        "name": "home_or_qpos0",
        "pos": home_qpos[free_adr:free_adr + 3].copy(),
        "quat": home_qpos[free_adr + 3:free_adr + 7].copy(),
        "q": np.asarray(
            [home_qpos[model.jnt_qposadr[j]] for j in joint_ids],
            dtype=np.float64,
        ),
    }
    zero = {
        "name": "all_zero_joints",
        "pos": np.array([0.04, -0.02, 0.30]),
        "quat": np.array([1., 0., 0., 0.]),
        "q": np.zeros(10),
    }
    result = [home, zero]
    bounds = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    for i in range(samples):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        q = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.uniform(
            0.12, 0.88, size=10
        )
        result.append({
            "name": f"random_{i:02d}",
            "pos": rng.uniform([-0.2, -0.16, 0.20], [0.2, 0.16, 0.38]),
            "quat": quat,
            "q": q,
        })
    return result


def torch_inputs(cases, device):
    pos = torch.tensor(
        np.stack([case["pos"] for case in cases]),
        device=device, dtype=torch.float32,
    )
    rot = torch.tensor(
        np.stack([quaternion_to_matrix(case["quat"]) for case in cases]),
        device=device, dtype=torch.float32,
    )
    q = torch.tensor(
        np.stack([case["q"] for case in cases]),
        device=device, dtype=torch.float32,
    )
    return pos, rot, q


def mujoco_reference(model, kin, case):
    """Independent oracle. MJCF uses world frame; set P = world for comparison."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    data.qpos[free_adr:free_adr + 3] = case["pos"]
    data.qpos[free_adr + 3:free_adr + 7] = case["quat"]
    joint_ids = [model.joint(name).id for name in JOINT_NAMES]
    for j, joint_id in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[joint_id])] = case["q"][j]
    mujoco.mj_forward(model, data)

    site_ids = [model.site(name).id for name in SITE_NAMES]
    base_id = model.body("base_link").id
    return {
        "body_pos": np.asarray(data.xpos[kin.bodies], dtype=float).copy(),
        "body_rot": np.asarray(data.xmat[kin.bodies], dtype=float)
                        .reshape(11, 3, 3).copy(),
        "com_body": np.asarray(data.xipos[kin.bodies], dtype=float).copy(),
        "com": np.asarray(data.subtree_com[base_id], dtype=float).copy(),
        "site_pos": np.asarray(data.site_xpos[site_ids], dtype=float).copy(),
        "site_rot": np.asarray(data.site_xmat[site_ids], dtype=float)
                       .reshape(2, 3, 3).copy(),
        "axis": np.asarray(data.xaxis[joint_ids], dtype=float).copy(),
        "anchor": np.asarray(data.xanchor[joint_ids], dtype=float).copy(),
    }


class TestForwardKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        opts = cls.options
        cls.device = torch.device(opts.device)
        cls.datasets = []
        models = ["open_duck_mini_v5.xml"]
        if not opts.model_only:
            models.append("scene_flat_terrian_v5.xml")
        for filename in models:
            path = xml_folder() / filename
            model = mujoco.MjModel.from_xml_path(str(path))
            kin = RobotKinematics(model, device=cls.device)
            cases = build_cases(model, opts.samples)
            pos, rot, q = torch_inputs(cases, cls.device)
            cls.datasets.append((filename, model, kin, cases, pos, rot, q))

    def test_01_dataclass_and_field_shapes(self):
        self.assertTrue(is_dataclass(KinState))
        self.assertEqual({field.name for field in fields(KinState)}, set(FIELD_SHAPES))
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                ks = kin.forward_kinematics(p, R, q)
                self.assertIsInstance(ks, KinState)
                for key, shape in FIELD_SHAPES.items():
                    value = getattr(ks, key)
                    self.assertIsInstance(value, torch.Tensor)
                    self.assertEqual(tuple(value.shape), (len(cases), *shape), key)
                    self.assertEqual(value.dtype, torch.float32, key)
                    self.assertEqual(value.device, self.device, key)
                    self.assertTrue(torch.isfinite(value).all().item(), key)

    def test_02_every_field_matches_mujoco_v5(self):
        """Independent reference for all 8 KinState outputs."""
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                ks = kin.forward_kinematics(p, R, q)
                refs = [mujoco_reference(model, kin, case) for case in cases]
                for key in FIELD_SHAPES:
                    ref = np.stack([d[key] for d in refs])
                    assert_near(getattr(ks, key), ref, f"{name}: {key}", TOL[key])

    def test_03_matches_three_component_functions(self):
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                ks = kin.forward_kinematics(p, R, q)
                body_pos, body_rot, axis, anchor = kin.compute_body_kinematics(p, R, q)
                com_body, com = kin.compute_center_of_mass(body_pos, body_rot)
                site_pos, site_rot = kin.compute_foot_kinematics(body_pos, body_rot)
                expected = dict(
                    body_pos=body_pos, body_rot=body_rot,
                    com_body=com_body, com=com,
                    site_pos=site_pos, site_rot=site_rot,
                    axis=axis, anchor=anchor,
                )
                for key in FIELD_SHAPES:
                    assert_near(getattr(ks, key), expected[key], key, atol=0)

    def test_04_com_weighted_by_mass(self):
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                ks = kin.forward_kinematics(p, R, q)
                recomputed = (ks.com_body * kin.mass[None, :, None]).sum(1) / kin.total_mass
                assert_near(ks.com, recomputed, f"{name}: weighted COM", 3e-6)

    def test_05_foot_sites_attached_to_correct_bodies(self):
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                ks = kin.forward_kinematics(p, R, q)
                for i in range(2):
                    index = kin.site_body[i]
                    expected_p = (
                        ks.body_pos[:, index]
                        + (ks.body_rot[:, index] @ kin.site_pos0[i, :, None]).squeeze(-1)
                    )
                    expected_R = ks.body_rot[:, index] @ kin.site_rot0[i]
                    assert_near(ks.site_pos[:, i], expected_p, f"{name}: site {i} p", 2e-6)
                    assert_near(ks.site_rot[:, i], expected_R, f"{name}: site {i} R", 2e-6)

    def test_06_rotations_are_valid(self):
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                ks = kin.forward_kinematics(p, R, q)
                for label, rot in (("body", ks.body_rot), ("site", ks.site_rot)):
                    eye = torch.eye(3, device=self.device).expand_as(rot)
                    assert_near(rot.transpose(-1, -2) @ rot, eye,
                                f"{name}: {label} orthogonality", 6e-5)
                    assert_near(torch.linalg.det(rot), torch.ones_like(rot[..., 0, 0]),
                                f"{name}: {label} determinant", 6e-5)

    def test_07_batched_equals_individual(self):
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                batch = kin.forward_kinematics(p, R, q)
                for i in range(len(cases)):
                    single = kin.forward_kinematics(
                        p[i:i+1], R[i:i+1], q[i:i+1]
                    )
                    for key in FIELD_SHAPES:
                        assert_near(
                            getattr(batch, key)[i:i+1], getattr(single, key),
                            f"{name}: batch vs single {i}/{key}", 3e-6,
                        )

    def test_08_translation_of_base_translates_all_points(self):
        delta = torch.tensor([0.12, -0.08, 0.05], device=self.device)
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                a = kin.forward_kinematics(p, R, q)
                b = kin.forward_kinematics(p + delta, R, q)
                for key in ("body_pos", "com_body", "com", "site_pos", "anchor"):
                    assert_near(getattr(b, key), getattr(a, key) + delta,
                                f"{name}: shift {key}", 5e-6)
                for key in ("body_rot", "site_rot", "axis"):
                    assert_near(getattr(b, key), getattr(a, key),
                                f"{name}: invariant {key}", 5e-6)

    def test_09_common_rotation_transforms_all_fields(self):
        phi = 0.43
        c, s = np.cos(phi), np.sin(phi)
        G = torch.tensor(
            [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
            device=self.device, dtype=torch.float32,
        )
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                a = kin.forward_kinematics(p, R, q)
                bp = torch.einsum("ij,bj->bi", G, p)
                bR = torch.einsum("ij,bjk->bik", G, R)
                b = kin.forward_kinematics(bp, bR, q)
                for key in ("body_pos", "com_body", "site_pos", "anchor", "axis"):
                    original = getattr(a, key)
                    expected = torch.einsum("ij,bnj->bni", G, original)
                    assert_near(getattr(b, key), expected, f"{name}: rotate {key}", 5e-6)
                assert_near(b.com, torch.einsum("ij,bj->bi", G, a.com),
                            f"{name}: rotate com", 5e-6)
                for key in ("body_rot", "site_rot"):
                    original = getattr(a, key)
                    expected = torch.einsum("ij,bnjk->bnik", G, original)
                    assert_near(getattr(b, key), expected, f"{name}: rotate {key}", 6e-6)

    def test_10_gradients_and_input_immutability(self):
        for name, model, kin, cases, p, R, q in self.datasets:
            with self.subTest(model=name):
                p = p[:2].clone().detach().requires_grad_(True)
                R = R[:2].clone().detach()
                q = q[:2].clone().detach().requires_grad_(True)
                saved_p, saved_R, saved_q = p.detach().clone(), R.clone(), q.detach().clone()
                ks = kin.forward_kinematics(p, R, q)
                loss = (ks.com.square().sum() + ks.site_pos.square().sum()
                        + 0.1 * ks.body_pos[:, 5].square().sum())
                loss.backward()
                self.assertIsNotNone(p.grad)
                self.assertIsNotNone(q.grad)
                self.assertTrue(torch.isfinite(p.grad).all().item())
                self.assertTrue(torch.isfinite(q.grad).all().item())
                self.assertGreater(float(q.grad.abs().max().item()), 1e-8)
                assert_near(p, saved_p, f"{name}: base_pos unchanged", 0)
                assert_near(R, saved_R, f"{name}: base_rot unchanged", 0)
                assert_near(q, saved_q, f"{name}: q unchanged", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8,
                        help="Random joint configurations per MJCF; default: 8")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-only", action="store_true",
                        help="Test robot MJCF only; skip scene v5")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    TestForwardKinematics.options = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestForwardKinematics)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if result.wasSuccessful():
        count = 1 if args.model_only else 2
        print(
            f"\n[PASS] FK + KinState: 10 test groups, "
            f"{count} v5 MJCF file(s), {args.samples + 2} configurations per model."
        )
        print("Validated all 8 KinState outputs against MuJoCo forward kinematics.")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
