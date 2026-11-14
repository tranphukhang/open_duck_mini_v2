"""Kiem thu compute_center_of_mass() voi MuJoCo v5.

Dat file vao capture_point/ (cung cap voi robot_kinematics.py).
Chay tu thu muc goc repository:
    python capture_point/test_robot_kinematics_com.py --verbose
    python capture_point/test_robot_kinematics_com.py --samples 20 --verbose
    python capture_point/test_robot_kinematics_com.py --model-only
    python capture_point/test_robot_kinematics_com.py --device cuda

Yeu cau: mujoco, numpy, torch; thu muc xmls/assets_v3.
Chi test compute_center_of_mass() va luong FK -> CoM, chua test WBIK.
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
    raise SystemExit("Chua cai mujoco. Chay: pip install mujoco") from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robot_kinematics import RobotKinematics


JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
ATOL_COM = 5e-5  # float32 FK + batched matmul; units = metres
ATOL_INVARIANT = 5e-6


def xml_directory():
    for candidate in (HERE, *HERE.parents):
        xml_file = candidate / "xmls" / "open_duck_mini_v5.xml"
        if xml_file.is_file():
            return xml_file.parent
    raise FileNotFoundError(
        "Khong tim thay xmls/open_duck_mini_v5.xml. "
        "Hay de file test trong thu muc capture_point/ cua repository."
    )


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def assert_close(a, b, label, atol=ATOL_COM):
    x, y = as_numpy(a), as_numpy(b)
    if x.shape != y.shape:
        raise AssertionError(f"{label}: shape {x.shape} != {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise AssertionError(f"{label}: co gia tri NaN/Inf")
    try:
        np.testing.assert_allclose(x, y, atol=atol, rtol=0.0)
    except AssertionError as exc:
        err = np.max(np.abs(x - y))
        raise AssertionError(
            f"{label}: max_abs_error = {err:.3e}, tolerance = {atol:.1e}\n{exc}"
        ) from exc


def make_cases(model, count):
    rng = np.random.default_rng(20261114)
    joint_ids = [model.joint(n).id for n in JOINT_NAMES]
    low, high = np.asarray(model.jnt_range[joint_ids], dtype=np.float64).T
    free_id = model.joint("floating_base").id
    free_adr = int(model.jnt_qposadr[free_id])

    # Home tu keyframe scene v5; neu test model rieng thi dung home v5 tuong ung.
    home_qpos = np.array(model.qpos0, dtype=np.float64, copy=True)
    for k in range(model.nkey):
        if model.key(k).name == "home":
            home_qpos = np.asarray(model.key_qpos[k], dtype=np.float64).copy()
            break

    home = {
        "label": "home",
        "base_pos": home_qpos[free_adr:free_adr + 3].copy(),
        "base_quat": home_qpos[free_adr + 3:free_adr + 7].copy(),
        "q": np.array([
            home_qpos[int(model.jnt_qposadr[j])] for j in joint_ids
        ], dtype=np.float64),
    }

    zero = {
        "label": "zero",
        "base_pos": np.array([0.01, -0.025, 0.29]),
        "base_quat": np.array([1., 0., 0., 0.]),
        "q": np.zeros(10, dtype=np.float64),
    }

    cases = [home, zero]
    for i in range(count):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        cases.append({
            "label": f"random_{i:02d}",
            "base_pos": rng.uniform(
                [-0.20, -0.15, 0.21], [0.20, 0.15, 0.38]
            ),
            "base_quat": quat,
            "q": low + (high - low) * rng.uniform(0.12, 0.88, size=10),
        })
    return cases


def quat_to_mat(quat):
    result = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


def batch_inputs(cases, device):
    pos = torch.as_tensor(
        np.stack([c["base_pos"] for c in cases]),
        dtype=torch.float32, device=device,
    )
    rot = torch.as_tensor(
        np.stack([quat_to_mat(c["base_quat"]) for c in cases]),
        dtype=torch.float32, device=device,
    )
    q = torch.as_tensor(
        np.stack([c["q"] for c in cases]),
        dtype=torch.float32, device=device,
    )
    return pos, rot, q


def mujoco_com_reference(model, kin, case):
    """MuJoCo la oracle doc lap cho CoM va CoM tung khau."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    data.qpos[free_adr:free_adr + 3] = case["base_pos"]
    data.qpos[free_adr + 3:free_adr + 7] = case["base_quat"]
    for index, name in enumerate(JOINT_NAMES):
        adr = int(model.jnt_qposadr[model.joint(name).id])
        data.qpos[adr] = case["q"][index]
    mujoco.mj_forward(model, data)

    com_body = np.asarray(data.xipos[kin.bodies], dtype=np.float64).copy()
    total = np.asarray(
        data.subtree_com[model.body("base_link").id], dtype=np.float64
    ).copy()
    return com_body, total


class TestCenterOfMass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = cls.options
        cls.device = torch.device(args.device)
        cls.records = []
        models = ["open_duck_mini_v5.xml"]
        if not args.model_only:
            models.append("scene_flat_terrian_v5.xml")

        for name in models:
            path = xml_directory() / name
            model = mujoco.MjModel.from_xml_path(str(path))
            kin = RobotKinematics(model, device=args.device)
            cases = make_cases(model, args.samples)
            pos, rot, q = batch_inputs(cases, cls.device)
            cls.records.append((name, model, kin, cases, pos, rot, q))

    @staticmethod
    def fk_and_com(kin, base_pos, base_rot, q):
        body_pos, body_rot, _, _ = kin.compute_body_kinematics(
            base_pos, base_rot, q
        )
        com_body, com = kin.compute_center_of_mass(body_pos, body_rot)
        return body_pos, body_rot, com_body, com

    def test_01_shapes_dtype_and_finite(self):
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                bp, br, cb, com = self.fk_and_com(kin, pos, rot, q)
                self.assertEqual(tuple(cb.shape), (len(cases), 11, 3))
                self.assertEqual(tuple(com.shape), (len(cases), 3))
                self.assertEqual(cb.dtype, torch.float32)
                self.assertEqual(com.dtype, torch.float32)
                self.assertEqual(cb.device, self.device)
                self.assertEqual(com.device, self.device)
                self.assertTrue(torch.isfinite(cb).all().item())
                self.assertTrue(torch.isfinite(com).all().item())

    def test_02_each_body_com_matches_mujoco_xipos(self):
        """So sanh vi tri CoM cua tung khau voi MuJoCo."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                _, _, cb, _ = self.fk_and_com(kin, pos, rot, q)
                expected = np.stack([
                    mujoco_com_reference(model, kin, c)[0] for c in cases
                ])
                assert_close(cb, expected, f"{name}: 11 body COMs")

    def test_03_whole_robot_com_matches_mujoco_subtree(self):
        """So sanh CoM toan robot voi data.subtree_com(base_link)."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                _, _, _, com = self.fk_and_com(kin, pos, rot, q)
                expected = np.stack([
                    mujoco_com_reference(model, kin, c)[1] for c in cases
                ])
                assert_close(com, expected, f"{name}: subtree COM")

    def test_04_weighted_average_formula(self):
        """Kiem tra cong thuc tong m_i * p_Ci / sum m_i."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                _, _, cb, com = self.fk_and_com(kin, pos, rot, q)
                weights = kin.mass / kin.mass.sum()
                direct = (cb * weights[None, :, None]).sum(dim=1)
                assert_close(com, direct, f"{name}: weighted average",
                             atol=ATOL_INVARIANT)
                self.assertGreater(float(kin.total_mass), 0.0)
                self.assertTrue(torch.all(kin.mass > 0).item())

    def test_05_identity_rotation_known_local_com(self):
        """Voi body_pos=0 va R_i=I, com_body phai chinh la ipos."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                n = 4
                bp = torch.zeros(n, 11, 3, device=self.device)
                br = torch.eye(3, device=self.device).expand(n, 11, 3, 3)
                cb, com = kin.compute_center_of_mass(bp, br)
                assert_close(
                    cb, kin.ipos[None].expand(n, -1, -1),
                    f"{name}: local inertial CoM", atol=ATOL_INVARIANT,
                )
                known = (
                    kin.ipos * kin.mass[:, None]
                ).sum(dim=0) / kin.mass.sum()
                assert_close(
                    com, known[None].expand(n, -1),
                    f"{name}: synthetic total CoM", atol=ATOL_INVARIANT,
                )

    def test_06_translation_invariance(self):
        """Dich moi body +delta => CoM moi body va toan robot +delta."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                bp, br, cb1, com1 = self.fk_and_com(kin, pos, rot, q)
                delta = torch.tensor(
                    [0.12, -0.075, 0.04], dtype=torch.float32,
                    device=self.device,
                )
                cb2, com2 = kin.compute_center_of_mass(bp + delta, br)
                assert_close(cb2, cb1 + delta, f"{name}: body translations",
                             atol=ATOL_INVARIANT)
                assert_close(com2, com1 + delta, f"{name}: total translation",
                             atol=ATOL_INVARIANT)

    def test_07_rotation_equivariance(self):
        """Quay dong thoi body_pos va body_rot quanh goc P => CoM quay theo."""
        theta = 0.72
        cs, sn = np.cos(theta), np.sin(theta)
        G = torch.tensor(
            [[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]],
            dtype=torch.float32, device=self.device,
        )
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                bp, br, cb1, com1 = self.fk_and_com(kin, pos, rot, q)
                bp2 = torch.einsum("ij,bnj->bni", G, bp)
                br2 = torch.einsum("ij,bnjk->bnik", G, br)
                cb2, com2 = kin.compute_center_of_mass(bp2, br2)
                assert_close(
                    cb2, torch.einsum("ij,bnj->bni", G, cb1),
                    f"{name}: rotated body CoM", atol=ATOL_INVARIANT,
                )
                assert_close(
                    com2, torch.einsum("ij,bj->bi", G, com1),
                    f"{name}: rotated robot CoM", atol=ATOL_INVARIANT,
                )

    def test_08_joint_motion_changes_com_as_mujoco(self):
        """Giu base co dinh, thay doi 1 khop, kiem tra CoM khop voi MuJoCo."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                bp = pos[:1].repeat(2, 1)
                br = rot[:1].repeat(2, 1, 1)
                qq = torch.zeros(2, 10, device=self.device)
                qq[1, 2] = 0.4  # L hip pitch
                qq[1, 3] = -0.6  # L knee
                _, _, cb, com = self.fk_and_com(kin, bp, br, qq)
                self.assertGreater(
                    float(torch.linalg.vector_norm(com[1] - com[0])),
                    1e-5,
                    "CoM should respond to changes in left leg configuration",
                )
                sample = []
                for i in range(2):
                    sample.append({
                        "base_pos": as_numpy(bp[i]),
                        "base_quat": cases[0]["base_quat"],
                        "q": as_numpy(qq[i]),
                    })
                expected = np.stack([
                    mujoco_com_reference(model, kin, c)[1] for c in sample
                ])
                assert_close(com, expected, f"{name}: changed joint COM")

    def test_09_batch_independence(self):
        """Ket qua batched va tinh tung environment phai tuong duong."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                bp, br, cb, com = self.fk_and_com(kin, pos, rot, q)
                for index in range(len(cases)):
                    cb_one, com_one = kin.compute_center_of_mass(
                        bp[index:index + 1], br[index:index + 1]
                    )
                    assert_close(
                        cb[index:index + 1], cb_one,
                        f"{name}: batch vs single (body {index})",
                        atol=ATOL_INVARIANT,
                    )
                    assert_close(
                        com[index:index + 1], com_one,
                        f"{name}: batch vs single (robot {index})",
                        atol=ATOL_INVARIANT,
                    )

    def test_10_mass_weighting_gradient_no_mutation(self):
        """Kiem tra gradient va trong so khoi luong cho 1 khau dich chuyen."""
        for name, model, kin, cases, pos, rot, q in self.records:
            with self.subTest(model=name):
                bp, br, _, _ = self.fk_and_com(kin, pos[:2], rot[:2], q[:2])
                bp = bp.detach().clone().requires_grad_(True)
                br = br.detach().clone().requires_grad_(True)
                bp_before, br_before = bp.detach().clone(), br.detach().clone()
                cb0, com0 = kin.compute_center_of_mass(bp, br)

                selected = 4
                delta = torch.tensor(
                    [0.03, -0.012, 0.021],
                    dtype=torch.float32, device=self.device,
                )
                add = torch.zeros_like(bp)
                add[:, selected, :] = delta
                cb1, com1 = kin.compute_center_of_mass(bp + add, br)
                expected = delta * kin.mass[selected] / kin.total_mass
                assert_close(
                    com1 - com0,
                    expected[None].expand_as(com0),
                    f"{name}: mass weighted shift",
                    atol=ATOL_INVARIANT,
                )
                assert_close(
                    cb1[:, selected] - cb0[:, selected],
                    delta[None].expand_as(cb0[:, selected]),
                    f"{name}: moved body CoM",
                    atol=ATOL_INVARIANT,
                )
                loss = com0.square().sum() + cb0[:, selected, :].square().sum()
                loss.backward()
                self.assertIsNotNone(bp.grad)
                self.assertIsNotNone(br.grad)
                self.assertTrue(torch.isfinite(bp.grad).all().item())
                self.assertTrue(torch.isfinite(br.grad).all().item())
                assert_close(bp, bp_before, "body_pos mutated", atol=0)
                assert_close(br, br_before, "body_rot mutated", atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8,
                        help="So cau hinh ngau nhien (mac dinh: 8)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-only", action="store_true",
                        help="Chi test open_duck_mini_v5.xml, bo qua scene")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    TestCenterOfMass.options = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCenterOfMass)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)

    if result.wasSuccessful():
        print(
            "\n[PASS] compute_center_of_mass: 10 test groups; "
            f"{1 if args.model_only else 2} v5 model(s); "
            f"{args.samples + 2} configurations per model."
        )
        print(
            "Checked: MuJoCo data.xipos + subtree_com, CoM weighted formula, "
            "translations, rotations, joint changes, batch and gradients."
        )

    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
