"""Kiem thu RobotKinematics.compute_foot_kinematics() voi MuJoCo v5.

Dat file nay trong capture_point/ (canh robot_kinematics.py), giu thu muc
xmls/ va xmls/assets_v3/ tai thu muc goc repository.

Chay tu thu muc goc:
    python capture_point/test_robot_kinematics_foot.py --verbose
    python capture_point/test_robot_kinematics_foot.py --samples 20 --verbose
    python capture_point/test_robot_kinematics_foot.py --model-only
    python capture_point/test_robot_kinematics_foot.py --device cuda

File nay chi test FK tai 2 site L6_IK / R6_IK; chua test Jacobian/WBIK.
"""

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Chua cai mujoco. Hay cai: pip install mujoco") from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robot_kinematics import RobotKinematics

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
SITE_NAMES = ("L6_IK", "R6_IK")
POS_TOL = 5e-5   # met, torch.float32
ROT_TOL = 6e-5   # matrix entries, torch.float32
INVAR_TOL = 5e-6


def find_xml_dir():
    for parent in (HERE, *HERE.parents):
        directory = parent / "xmls"
        if (directory / "open_duck_mini_v5.xml").exists():
            return directory
    raise FileNotFoundError(
        "Khong tim thay xmls/open_duck_mini_v5.xml. "
        "Hay dat file test vao capture_point/ trong repository."
    )


def as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def check_close(actual, expected, name, tol):
    a = as_numpy(actual)
    b = as_numpy(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape {a.shape} != {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise AssertionError(f"{name}: NaN/Inf in values")
    try:
        np.testing.assert_allclose(a, b, rtol=0.0, atol=tol)
    except AssertionError as exc:
        max_err = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{name}: max error={max_err:.3e}, tol={tol:.1e}\n{exc}") from exc


def quat_to_rot(q_wxyz):
    out = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(out, q_wxyz)
    return out.reshape(3, 3)


def case_list(model, samples):
    rng = np.random.default_rng(20261114)
    joint_ids = [model.joint(n).id for n in JOINT_NAMES]
    limits = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    home_q = np.deg2rad(
        [0., -10., 20., -40., 20., 0., 10., -20., 40., -20.]
    )
    home_pos = np.array([0., 0., 0.275])
    home_quat = np.array([1., 0., 0., 0.])
    for key in range(model.nkey):
        if model.key(key).name == "home":
            state = np.asarray(model.key_qpos[key], dtype=np.float64)
            free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
            home_pos = state[free_adr:free_adr + 3].copy()
            home_quat = state[free_adr + 3:free_adr + 7].copy()
            home_q = np.asarray(
                [state[int(model.jnt_qposadr[jid])] for jid in joint_ids],
                dtype=np.float64,
            )
            break

    cases = [
        {"name": "home", "pos": home_pos, "quat": home_quat, "q": home_q},
        {"name": "all_zero", "pos": np.array([0., 0., 0.28]),
         "quat": np.array([1., 0., 0., 0.]), "q": np.zeros(10)},
    ]
    for i in range(samples):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        q = limits[:, 0] + (limits[:, 1] - limits[:, 0]) * rng.uniform(
            0.12, 0.88, size=10
        )
        cases.append({
            "name": f"random_{i:02d}",
            "pos": rng.uniform([-0.15, -0.12, 0.21], [0.20, 0.13, 0.35]),
            "quat": quat,
            "q": q,
        })
    return cases


def inputs(cases, device):
    pos = torch.tensor(
        np.stack([c["pos"] for c in cases]),
        dtype=torch.float32, device=device,
    )
    rot = torch.tensor(
        np.stack([quat_to_rot(c["quat"]) for c in cases]),
        dtype=torch.float32, device=device,
    )
    q = torch.tensor(
        np.stack([c["q"] for c in cases]),
        dtype=torch.float32, device=device,
    )
    return pos, rot, q


def mujoco_foot_reference(model, case):
    """Oracle doc lap: MuJoCo mj_forward, data.site_xpos va site_xmat."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    data.qpos[free_adr:free_adr + 3] = case["pos"]
    data.qpos[free_adr + 3:free_adr + 7] = case["quat"]
    for idx, name in enumerate(JOINT_NAMES):
        jid = model.joint(name).id
        data.qpos[int(model.jnt_qposadr[jid])] = case["q"][idx]
    mujoco.mj_forward(model, data)
    site_ids = [model.site(name).id for name in SITE_NAMES]
    p = np.array(data.site_xpos[site_ids], dtype=np.float64, copy=True)
    R = np.array(data.site_xmat[site_ids], dtype=np.float64, copy=True)
    return p, R.reshape(2, 3, 3)


class TestFootKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        opts = cls.options
        cls.device = torch.device(opts.device)
        cls.items = []
        models = ["open_duck_mini_v5.xml"]
        if not opts.model_only:
            models.append("scene_flat_terrian_v5.xml")

        for name in models:
            path = find_xml_dir() / name
            model = mujoco.MjModel.from_xml_path(str(path))
            kin = RobotKinematics(model, device=opts.device)
            cases = case_list(model, opts.samples)
            bp, br, q = inputs(cases, cls.device)
            cls.items.append((name, model, kin, cases, bp, br, q))

    @staticmethod
    def evaluate(kin, base_pos, base_rot, q):
        body_pos, body_rot, _, _ = kin.compute_body_kinematics(
            base_pos, base_rot, q
        )
        site_pos, site_rot = kin.compute_foot_kinematics(body_pos, body_rot)
        return body_pos, body_rot, site_pos, site_rot

    def test_01_shapes_dtypes_devices_and_finite(self):
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, p, R = self.evaluate(kin, bp, br, q)
                self.assertEqual(tuple(p.shape), (len(cases), 2, 3))
                self.assertEqual(tuple(R.shape), (len(cases), 2, 3, 3))
                for x in (p, R):
                    self.assertEqual(x.dtype, torch.float32)
                    self.assertEqual(x.device, self.device)
                    self.assertTrue(torch.isfinite(x).all().item())

    def test_02_foot_positions_match_mujoco_v5(self):
        """Test chinh: pos L6_IK/R6_IK match independent MuJoCo FK."""
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, p, _ = self.evaluate(kin, bp, br, q)
                expected = np.stack([mujoco_foot_reference(model, c)[0]
                                     for c in cases])
                check_close(p, expected, f"{name}: site_xpos", POS_TOL)

    def test_03_foot_orientations_match_mujoco_v5(self):
        """Test chinh: rot L6_IK/R6_IK match independent MuJoCo FK."""
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, _, R = self.evaluate(kin, bp, br, q)
                expected = np.stack([mujoco_foot_reference(model, c)[1]
                                     for c in cases])
                check_close(R, expected, f"{name}: site_xmat", ROT_TOL)

    def test_04_site_names_and_local_frames_are_correct(self):
        """Dam bao khong nham site IK voi L6/R6 hay ankle origin."""
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                ids = [model.site(n).id for n in SITE_NAMES]
                expected_bodies = [
                    model.body("L5_ankle").id,
                    model.body("R5_ankle").id,
                ]
                self.assertEqual([kin.bodies[i] for i in kin.site_body],
                                 expected_bodies)
                check_close(kin.site_pos0, model.site_pos[ids],
                            f"{name}: site offsets", 1e-7)
                expected_R0 = np.stack([
                    quat_to_rot(model.site_quat[site_id]) for site_id in ids
                ])
                check_close(kin.site_rot0, expected_R0,
                            f"{name}: local site orientation", 1e-6)
                self.assertGreater(float(np.linalg.norm(expected_R0[0] -
                                                        expected_R0[1])), 0.1)

    def test_05_foot_rotation_matrices_are_valid(self):
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, _, R = self.evaluate(kin, bp, br, q)
                eye = torch.eye(3, dtype=R.dtype, device=self.device)
                check_close(R.transpose(-1, -2) @ R,
                            eye.expand_as(R), f"{name}: R^TR=I", ROT_TOL)
                check_close(torch.linalg.det(R),
                            torch.ones_like(R[..., 0, 0]),
                            f"{name}: det(R)=1", ROT_TOL)

    def test_06_batch_matches_individual_calls(self):
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, p_batch, R_batch = self.evaluate(kin, bp, br, q)
                for i in range(len(cases)):
                    _, _, p, R = self.evaluate(kin, bp[i:i+1],
                                                br[i:i+1], q[i:i+1])
                    check_close(p_batch[i:i+1], p,
                                f"{name}: position env {i}", INVAR_TOL)
                    check_close(R_batch[i:i+1], R,
                                f"{name}: rotation env {i}", INVAR_TOL)

    def test_07_translation_of_base(self):
        """Dich base +d -> feet pos +d, feet rotation unchanged."""
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, p1, R1 = self.evaluate(kin, bp, br, q)
                shift = torch.tensor([0.08, -0.11, 0.05],
                                     device=self.device, dtype=bp.dtype)
                _, _, p2, R2 = self.evaluate(kin, bp + shift, br, q)
                check_close(p2, p1 + shift, f"{name}: translation", INVAR_TOL)
                check_close(R2, R1, f"{name}: orientation unchanged", INVAR_TOL)

    def test_08_rotation_of_planning_frame(self):
        """Quay cung mot G vao base pose -> feet pose quay theo G."""
        ang = 0.47
        c, s = np.cos(ang), np.sin(ang)
        G = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
                         device=self.device, dtype=torch.float32)
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                _, _, p1, R1 = self.evaluate(kin, bp, br, q)
                new_bp = torch.einsum("ij,bj->bi", G, bp)
                new_br = torch.einsum("ij,bjk->bik", G, br)
                _, _, p2, R2 = self.evaluate(kin, new_bp, new_br, q)
                expected_p = torch.einsum("ij,bfj->bfi", G, p1)
                expected_R = torch.einsum("ij,bfjk->bfik", G, R1)
                check_close(p2, expected_p, f"{name}: rotated position", POS_TOL)
                check_close(R2, expected_R, f"{name}: rotated orientation", ROT_TOL)

    def test_09_left_and_right_joint_independence(self):
        """Thay q 1 chan khong duoc lam chuyen dong site chan con lai."""
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                bp = torch.tensor([[0., 0., 0.275]], device=self.device).repeat(3, 1)
                br = torch.eye(3, device=self.device)[None].repeat(3, 1, 1)
                qq = torch.zeros(3, 10, device=self.device)
                qq[1, 2], qq[1, 3] = 0.35, -0.45  # only left leg
                qq[2, 7], qq[2, 8] = -0.35, 0.45  # only right leg
                _, _, p, R = self.evaluate(kin, bp, br, qq)
                check_close(p[1, 1], p[0, 1],
                            f"{name}: right position unchanged", INVAR_TOL)
                check_close(R[1, 1], R[0, 1],
                            f"{name}: right rotation unchanged", INVAR_TOL)
                check_close(p[2, 0], p[0, 0],
                            f"{name}: left position unchanged", INVAR_TOL)
                check_close(R[2, 0], R[0, 0],
                            f"{name}: left rotation unchanged", INVAR_TOL)
                self.assertGreater(float(torch.linalg.norm(p[1, 0] - p[0, 0])), 1e-3)
                self.assertGreater(float(torch.linalg.norm(p[2, 1] - p[0, 1])), 1e-3)

    def test_10_site_transform_gradients_and_input_immutability(self):
        """Foot FK differentiable wrt q and does not modify inputs."""
        for name, model, kin, cases, bp, br, q in self.items:
            with self.subTest(xml=name):
                bp = bp[:2].detach().clone().requires_grad_(True)
                q = q[:2].detach().clone().requires_grad_(True)
                br = br[:2].detach().clone()
                bp0, q0, br0 = bp.detach().clone(), q.detach().clone(), br.clone()
                _, _, p, R = self.evaluate(kin, bp, br, q)
                # Site position and one orientation column affect the loss.
                loss = p.square().sum() + 0.07 * R[..., 0, 2].sum()
                loss.backward()
                self.assertIsNotNone(bp.grad)
                self.assertIsNotNone(q.grad)
                self.assertTrue(torch.isfinite(bp.grad).all().item())
                self.assertTrue(torch.isfinite(q.grad).all().item())
                self.assertGreater(float(q.grad.abs().max()), 1e-7)
                check_close(bp, bp0, f"{name}: base_pos not modified", 0.)
                check_close(q, q0, f"{name}: joint q not modified", 0.)
                check_close(br, br0, f"{name}: base_rot not modified", 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8,
                        help="Number of random configurations per model (default 8)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-only", action="store_true",
                        help="Only open_duck_mini_v5.xml, skip scene")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA not available; use --device cpu")

    TestFootKinematics.options = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFootKinematics)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if result.wasSuccessful():
        num = 1 if args.model_only else 2
        print(f"\n[PASS] compute_foot_kinematics: 10 test groups, "
              f"{num} v5 MJCF model(s), {args.samples + 2} poses/model, "
              f"device={args.device}.")
        print("Independent MuJoCo check: L6_IK/R6_IK site_xpos and site_xmat.")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
