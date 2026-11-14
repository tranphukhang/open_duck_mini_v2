"""Test RobotKinematics.compute_body_kinematics() against MuJoCo FK (v5).

Place this file in capture_point/, next to robot_kinematics.py.

Usage from repository root:
    python capture_point/test_robot_kinematics_body.py
    python capture_point/test_robot_kinematics_body.py --verbose
    python capture_point/test_robot_kinematics_body.py --samples 20
    python capture_point/test_robot_kinematics_body.py --model-only
    python capture_point/test_robot_kinematics_body.py --device cuda

Requirements: mujoco, numpy, torch; original xmls/assets_v3/*.STL.
The test uses only the v5 MJCF files, and does not test CoM/site FK,
Jacobian, DLS, warm-start, or load compensation.
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
    raise SystemExit(
        "Missing 'mujoco'. Install: pip install mujoco numpy torch"
    ) from exc

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
try:
    from robot_kinematics import RobotKinematics
except ImportError as exc:
    raise SystemExit(
        "Could not import robot_kinematics.py. Put this file in capture_point/."
    ) from exc


JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
BODY_NAMES = (
    "base_link",
    "L1_hip_yaw", "L2_hip_roll", "L3_hip_pitch", "L4_knee", "L5_ankle",
    "R1_hip_yaw", "R2_hip_roll", "R3_hip_pitch", "R4_knee", "R5_ankle",
)
ATOL_POS = 3e-5
ATOL_ROT = 5e-5
ATOL_AXIS = 5e-5


def find_v5_dir() -> Path:
    for folder in (TEST_DIR, *TEST_DIR.parents):
        path = folder / "xmls" / "open_duck_mini_v5.xml"
        if path.is_file():
            return path.parent
    raise FileNotFoundError(
        "Could not find xmls/open_duck_mini_v5.xml. "
        "Preserve capture_point/ and xmls/ at repository root."
    )


def np_result(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def check_close(actual, reference, name: str, *, atol: float) -> float:
    a = np_result(actual) if isinstance(actual, torch.Tensor) else np.asarray(actual)
    r = np.asarray(reference)
    if a.shape != r.shape:
        raise AssertionError(f"{name}: shape {a.shape} != {r.shape}")
    err = float(np.max(np.abs(a - r))) if a.size else 0.0
    try:
        np.testing.assert_allclose(a, r, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{name}: max_abs_error={err:.3e}\n{exc}") from exc
    return err


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat_wxyz)
    return mat.reshape(3, 3)


def random_quaternion(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q


def build_cases(model: mujoco.MjModel, count: int) -> list[dict]:
    rng = np.random.default_rng(20261114)
    joint_ids = [model.joint(n).id for n in JOINT_NAMES]
    bounds = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)

    # Reproduce the scene v5 "home" keyframe when present.
    home_base = np.array([0.0, 0.0, 0.275], dtype=np.float64)
    home_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    home_joints = np.deg2rad(
        [0., -10., 20., -40., 20.,
         0., 10., -20., 40., -20.]
    )
    for key_id in range(model.nkey):
        if model.key(key_id).name == "home":
            key_qpos = np.asarray(model.key_qpos[key_id], dtype=np.float64)
            free_addr = int(model.jnt_qposadr[model.joint("floating_base").id])
            home_base = key_qpos[free_addr:free_addr + 3].copy()
            home_quat = key_qpos[free_addr + 3:free_addr + 7].copy()
            home_joints = np.array(
                [key_qpos[int(model.jnt_qposadr[j])] for j in joint_ids]
            )
            break

    cases = [
        {"name": "home", "pos": home_base, "quat": home_quat, "q": home_joints},
        {"name": "zero", "pos": np.array([0.03, -0.02, 0.31]),
         "quat": np.array([1., 0., 0., 0.]), "q": np.zeros(10)},
    ]
    for i in range(count):
        # Stay within joint limits; deliberately vary translation and orientation.
        lo, hi = bounds[:, 0], bounds[:, 1]
        q = lo + (hi - lo) * rng.uniform(0.12, 0.88, size=10)
        cases.append({
            "name": f"random_{i:02d}",
            "pos": np.array([0.0, 0.0, 0.28]) +
                   rng.uniform([-0.3, -0.2, -0.06], [0.3, 0.2, 0.08]),
            "quat": random_quaternion(rng),
            "q": q,
        })
    return cases


def mujoco_reference(model: mujoco.MjModel, kin: RobotKinematics, case: dict):
    """Independent oracle: MuJoCo forward kinematics, not copied PyTorch formulas."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    floating_id = model.joint("floating_base").id
    free_addr = int(model.jnt_qposadr[floating_id])
    data.qpos[free_addr:free_addr + 3] = case["pos"]
    data.qpos[free_addr + 3:free_addr + 7] = case["quat"]

    joint_ids = [model.joint(name).id for name in JOINT_NAMES]
    for j, jid in enumerate(joint_ids):
        data.qpos[int(model.jnt_qposadr[jid])] = case["q"][j]

    mujoco.mj_forward(model, data)

    # data.xmat is flattened or (3,3), depending on binding/version.
    body_pos = np.asarray(data.xpos[kin.bodies], dtype=np.float64).copy()
    body_rot = np.asarray(data.xmat[kin.bodies], dtype=np.float64).reshape(11, 3, 3).copy()

    # xaxis / xanchor are MuJoCo's WORLD-frame joint quantities.
    axis = np.asarray(data.xaxis[joint_ids], dtype=np.float64).copy()
    anchor = np.asarray(data.xanchor[joint_ids], dtype=np.float64).copy()

    return body_pos, body_rot, axis, anchor


def get_batched_inputs(cases: list[dict], device: torch.device):
    base_pos = torch.tensor(
        np.stack([c["pos"] for c in cases]), device=device, dtype=torch.float32
    )
    base_rot = torch.tensor(
        np.stack([quat_to_matrix(c["quat"]) for c in cases]),
        device=device, dtype=torch.float32,
    )
    q = torch.tensor(
        np.stack([c["q"] for c in cases]), device=device, dtype=torch.float32
    )
    return base_pos, base_rot, q


class TestComputeBodyKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        opts = getattr(
            cls, "options",
            argparse.Namespace(device="cpu", samples=8, model_only=False),
        )
        cls.device = torch.device(opts.device)
        cls.models = []
        xml_dir = find_v5_dir()
        filenames = ["open_duck_mini_v5.xml"]
        if not opts.model_only:
            filenames.append("scene_flat_terrian_v5.xml")

        for filename in filenames:
            path = xml_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing v5 model: {path}")
            try:
                model = mujoco.MjModel.from_xml_path(str(path))
                kin = RobotKinematics(model, device=opts.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load/initialize {path}. "
                    "Check MJCF and xmls/assets_v3/*.STL."
                ) from exc
            cases = build_cases(model, opts.samples)
            cls.models.append((filename, model, kin, cases))

    def test_01_batch_shapes_finite_and_dtype(self):
        """Batched shapes and finite results for all four outputs."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                out = kin.compute_body_kinematics(pos, rot, q)
                self.assertEqual(len(out), 4)
                expected_shapes = (
                    (len(cases), 11, 3),
                    (len(cases), 11, 3, 3),
                    (len(cases), 10, 3),
                    (len(cases), 10, 3),
                )
                for value, shape in zip(out, expected_shapes):
                    self.assertEqual(tuple(value.shape), shape)
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertEqual(value.device, self.device)
                    self.assertTrue(torch.isfinite(value).all().item())

    def test_02_all_cases_match_mujoco_forward(self):
        """The main test: positions, rotations, joint axes and anchors."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                actual = kin.compute_body_kinematics(pos, rot, q)
                reference = [
                    np.stack([mujoco_reference(model, kin, c)[j] for c in cases])
                    for j in range(4)
                ]
                for key, got, want, tol in zip(
                    ("body_pos", "body_rot", "axis", "anchor"),
                    actual, reference,
                    (ATOL_POS, ATOL_ROT, ATOL_AXIS, ATOL_POS),
                ):
                    check_close(got, want, f"{name}: {key}", atol=tol)

    def test_03_base_pose_is_preserved(self):
        """Output for base_link equals given base pose (frame P=W in oracle)."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                bp, br, axis, anchor = kin.compute_body_kinematics(pos, rot, q)
                check_close(bp[:, 0], pos, "base position", atol=1e-7)
                check_close(br[:, 0], rot, "base rotation", atol=1e-7)

    def test_04_joint_anchor_equals_joint_body_origin_in_v5(self):
        """The current v5 declares hinge joint positions at body origin."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                bp, br, axis, anchor = kin.compute_body_kinematics(pos, rot, q)
                joint_origins = bp[:, kin.joint_body]
                check_close(anchor, joint_origins, "joint origin = anchor", atol=1e-7)
                check_close(
                    torch.linalg.vector_norm(axis, dim=-1),
                    torch.ones_like(axis[..., 0]),
                    "joint axes unit length", atol=1e-5,
                )

    def test_05_every_rotation_is_orthonormal(self):
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                _, body_rot, _, _ = kin.compute_body_kinematics(pos, rot, q)
                identity = torch.eye(3, dtype=body_rot.dtype, device=self.device)
                check_close(
                    body_rot.transpose(-1, -2) @ body_rot,
                    identity.expand_as(body_rot),
                    "R^T R == I", atol=ATOL_ROT,
                )
                check_close(
                    torch.linalg.det(body_rot),
                    torch.ones_like(body_rot[..., 0, 0]),
                    "det(R) == 1", atol=ATOL_ROT,
                )

    def test_06_batch_is_equivalent_to_individual_execution(self):
        """Catch incorrect broadcasting, batch-to-batch mixing, and reordering."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                batched = kin.compute_body_kinematics(pos, rot, q)
                for i in range(len(cases)):
                    single = kin.compute_body_kinematics(
                        pos[i:i + 1], rot[i:i + 1], q[i:i + 1]
                    )
                    for j, key in enumerate(("body_pos", "body_rot", "axis", "anchor")):
                        check_close(
                            batched[j][i:i + 1], single[j],
                            f"{key} batch vs single ({i})", atol=2e-6,
                        )

    def test_07_translation_equivariance(self):
        """Translating the base translates positions/anchors, not orientations/axes."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                shift = torch.tensor([0.12, -0.08, 0.04], device=self.device)
                p1, r1, a1, o1 = kin.compute_body_kinematics(pos, rot, q)
                p2, r2, a2, o2 = kin.compute_body_kinematics(pos + shift, rot, q)
                check_close(p2, p1 + shift, "shifted body positions", atol=2e-6)
                check_close(o2, o1 + shift, "shifted anchors", atol=2e-6)
                check_close(r2, r1, "unchanged body rotations", atol=2e-6)
                check_close(a2, a1, "unchanged joint axes", atol=2e-6)

    def test_08_world_rotation_equivariance(self):
        """A common frame rotation must rotate all positions, rotations, and axes."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases, self.device)
                phi = 0.53
                c, s = np.cos(phi), np.sin(phi)
                G = torch.tensor(
                    [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
                    device=self.device, dtype=torch.float32,
                )
                old = kin.compute_body_kinematics(pos, rot, q)
                rot_pos = torch.einsum("ij,bj->bi", G, pos)
                rot_rot = torch.einsum("ij,bjk->bik", G, rot)
                new = kin.compute_body_kinematics(rot_pos, rot_rot, q)
                check_close(
                    new[0], torch.einsum("ij,bnj->bni", G, old[0]),
                    "rotated positions", atol=ATOL_POS,
                )
                check_close(
                    new[1], torch.einsum("ij,bnjk->bnik", G, old[1]),
                    "rotated body matrices", atol=ATOL_ROT,
                )
                check_close(
                    new[2], torch.einsum("ij,bnj->bni", G, old[2]),
                    "rotated axes", atol=ATOL_AXIS,
                )
                check_close(
                    new[3], torch.einsum("ij,bnj->bni", G, old[3]),
                    "rotated anchors", atol=ATOL_POS,
                )

    def test_09_left_joint_changes_do_not_move_right_leg(self):
        """With identical base pose, left-leg q cannot alter right-leg bodies."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos = torch.tensor([[0., 0., 0.275]], device=self.device).repeat(2, 1)
                rot = torch.eye(3, device=self.device)[None].repeat(2, 1, 1)
                q = torch.zeros(2, 10, device=self.device)
                q[1, 2] = 0.45
                q[1, 3] = -0.50
                bp, br, axis, anchor = kin.compute_body_kinematics(pos, rot, q)
                check_close(bp[0, 6:], bp[1, 6:], "right leg positions", atol=1e-7)
                check_close(br[0, 6:], br[1, 6:], "right leg rotations", atol=1e-7)
                check_close(axis[0, 5:], axis[1, 5:], "right leg axes", atol=1e-7)
                check_close(anchor[0, 5:], anchor[1, 5:], "right leg anchors", atol=1e-7)
                self.assertGreater(
                    float(torch.max(torch.abs(bp[0, 4:6] - bp[1, 4:6]))), 1e-3,
                    "Changing left hip/knee must affect left ankle geometry.",
                )

    def test_10_inputs_unchanged_and_gradients_finite(self):
        """FK remains a differentiable PyTorch operation without input mutation."""
        for name, model, kin, cases in self.models:
            with self.subTest(xml=name):
                pos, rot, q = get_batched_inputs(cases[:2], self.device)
                pos.requires_grad_(True)
                q.requires_grad_(True)
                orig_pos = pos.detach().clone()
                orig_q = q.detach().clone()
                bp, br, axis, anchor = kin.compute_body_kinematics(pos, rot, q)
                check_close(pos, orig_pos, "base_pos unchanged", atol=0.0)
                check_close(q, orig_q, "q unchanged", atol=0.0)
                loss = bp[:, 5, :].square().sum() + bp[:, 10, :].square().sum()
                loss = loss + 0.01 * br[:, 5].square().sum()
                loss.backward()
                self.assertIsNotNone(pos.grad)
                self.assertIsNotNone(q.grad)
                self.assertTrue(torch.isfinite(pos.grad).all().item())
                self.assertTrue(torch.isfinite(q.grad).all().item())
                self.assertGreater(float(q.grad.abs().max()), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--samples", type=int, default=8,
                        help="Number of random configurations per MJCF (default: 8)")
    parser.add_argument("--model-only", action="store_true",
                        help="Test bare open_duck_mini_v5.xml only, skip scene")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    TestComputeBodyKinematics.options = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestComputeBodyKinematics
    )
    result = unittest.TextTestRunner(
        verbosity=2 if args.verbose else 1
    ).run(suite)
    if result.wasSuccessful():
        checked = 1 if args.model_only else 2
        print(
            f"\n[PASS] compute_body_kinematics: 10 test groups; "
            f"{checked} v5 MJCF file(s); "
            f"{args.samples + 2} configurations per file; "
            f"device={args.device}"
        )
        print(
            "MuJoCo oracle: body_xpos/xmat + joint_xaxis/xanchor; "
            "also batch, equivariance, orthogonality, gradients."
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
