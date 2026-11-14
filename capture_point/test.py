"""Kiểm thử RobotKinematics.__init__() với mô hình Open Duck Mini V2 v5.

Đặt file này ở capture_point/test_robot_kinematics_init.py rồi chạy từ bất kỳ thư mục nào:
    python capture_point/test_robot_kinematics_init.py
    python capture_point/test_robot_kinematics_init.py --model-only
    python capture_point/test_robot_kinematics_init.py --verbose

Yêu cầu: pip install mujoco numpy torch
Cần giữ nguyên xmls/assets_v3/*.STL của repository.

Chỉ kiểm tra __init__; chưa kiểm tra FK, Jacobian số học hay WBIK.
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
        "Không tìm thấy 'mujoco'. Cài bằng: pip install mujoco\n"
        "Cần cài cả numpy và torch để chạy bài test."
    ) from exc

# Nhập robot_kinematics.py ở cùng thư mục với file test.
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
try:
    from robot_kinematics import RobotKinematics
except ImportError as exc:
    raise SystemExit(
        "Không tìm thấy robot_kinematics.py. "
        "Hãy đặt test_robot_kinematics_init.py trong cùng thư mục capture_point/."
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
FOOT_SITES = ("L6_IK", "R6_IK")
JOINT_RANGE_DEGREES = np.array([
    [-30, 30], [-25, 25], [-30, 70], [-90, 90], [-90, 90],
    [-30, 30], [-25, 25], [-70, 30], [-90, 90], [-90, 90],
], dtype=np.float64)


def locate_xml_dir() -> Path:
    # Tìm repository root, kể cả khi script được gọi từ thư mục khác.
    for directory in (TEST_DIR, *TEST_DIR.parents):
        candidate = directory / "xmls" / "open_duck_mini_v5.xml"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError(
        "Không tìm thấy xmls/open_duck_mini_v5.xml. "
        "Hãy giữ đúng cấu trúc repository: xmls/ và capture_point/."
    )


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def tensor_close(actual: torch.Tensor, expected, atol=1e-6):
    np.testing.assert_allclose(to_numpy(actual), np.asarray(expected), atol=atol, rtol=1e-6)


def ancestor_of(model: mujoco.MjModel, ancestor: int, body: int) -> bool:
    """Kiểm tra quan hệ hậu duệ trực tiếp từ body_parentid của MuJoCo."""
    b = int(body)
    while b > 0:
        if b == ancestor:
            return True
        b = int(model.body_parentid[b])
    return False


def depth_for_local_body(kin: RobotKinematics, i: int) -> int:
    d, k = 0, i
    while kin.parent[k] != -1:
        k = kin.parent[k]
        d += 1
    return d


class TestRobotKinematicsInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml_dir = locate_xml_dir()
        filenames = ["open_duck_mini_v5.xml"]
        if not getattr(cls, "model_only", False):
            filenames.append("scene_flat_terrian_v5.xml")

        cls.models = []
        for name in filenames:
            path = xml_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy file v5: {path}")
            try:
                model = mujoco.MjModel.from_xml_path(str(path))
            except Exception as exc:
                raise RuntimeError(
                    f"Không thể biên dịch {path}. Kiểm tra thư mục xmls/assets_v3/ "
                    "có đầy đủ các STL của mô hình v5."
                ) from exc
            kin = RobotKinematics(model, device="cpu")
            cls.models.append((name, model, kin))

    def test_01_tree_structure(self):
        """Chỉ đọc 11 khâu robot, không lấy world/floor từ scene."""
        for name, model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(len(kin.bodies), 11)
                self.assertEqual([model.body(b).name for b in kin.bodies], list(BODY_NAMES))
                self.assertEqual(len(kin.parent), 11)
                self.assertEqual(kin.parent[0], -1)
                self.assertEqual(kin.parent[1:6], [0, 1, 2, 3, 4])
                self.assertEqual(kin.parent[6:], [0, 6, 7, 8, 9])
                self.assertEqual(len(set(kin.bodies)), 11)
                if name.startswith("scene"):
                    self.assertGreater(model.nbody, 12)  # world + robot + floor

    def test_02_joint_order_and_model_anchors(self):
        for name, model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(len(kin.joint_of_body), 11)
                self.assertEqual(len(kin.joint_body), 10)
                self.assertEqual(kin.joint_of_body[0], -1)
                self.assertEqual(sorted(kin.joint_body), list(range(1, 11)))
                for j, joint_name in enumerate(JOINT_NAMES):
                    body_local = kin.joint_body[j]
                    body_global = kin.bodies[body_local]
                    joint_id = model.joint(joint_name).id
                    self.assertEqual(model.jnt_bodyid[joint_id], body_global)
                    self.assertEqual(kin.joint_of_body[body_local], j)
                    self.assertEqual(model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_HINGE)
                    self.assertAlmostEqual(float(np.linalg.norm(model.jnt_pos[joint_id])), 0., places=8)

    def test_03_geometry_axes_and_mass(self):
        for name, model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(tuple(kin.body_pos0.shape), (11, 3))
                self.assertEqual(tuple(kin.body_rot0.shape), (11, 3, 3))
                self.assertEqual(tuple(kin.joint_axis.shape), (10, 3))
                self.assertEqual(tuple(kin.mass.shape), (11,))
                self.assertEqual(tuple(kin.ipos.shape), (11, 3))

                tensor_close(kin.body_pos0, model.body_pos[kin.bodies])
                tensor_close(kin.mass, model.body_mass[kin.bodies])
                tensor_close(kin.ipos, model.body_ipos[kin.bodies])
                self.assertAlmostEqual(kin.total_mass, float(model.body_mass[kin.bodies].sum()), places=5)
                self.assertTrue(np.isfinite(kin.total_mass))
                self.assertGreater(kin.total_mass, 0.)
                self.assertTrue(np.all(to_numpy(kin.mass) > 0))

                for j, joint_name in enumerate(JOINT_NAMES):
                    tensor_close(kin.joint_axis[j], model.jnt_axis[model.joint(joint_name).id])
                for i, body_id in enumerate(kin.bodies):
                    # Kiểm tra phép quay từ quaternion của body trong MjModel.
                    rm = np.zeros(9)
                    mujoco.mju_quat2Mat(rm, model.body_quat[body_id])
                    tensor_close(kin.body_rot0[i], rm.reshape(3, 3))
                    rm_kin = to_numpy(kin.body_rot0[i])
                    np.testing.assert_allclose(rm_kin.T @ rm_kin, np.eye(3), atol=1e-5)
                    self.assertAlmostEqual(np.linalg.det(rm_kin), 1., places=5)

    def test_04_foot_site_definition(self):
        for name, model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(tuple(kin.site_pos0.shape), (2, 3))
                self.assertEqual(tuple(kin.site_rot0.shape), (2, 3, 3))
                self.assertEqual(len(kin.site_body), 2)
                self.assertEqual(kin.site_body, [5, 10])  # L5_ankle / R5_ankle
                for i, site_name in enumerate(FOOT_SITES):
                    site_id = model.site(site_name).id
                    self.assertEqual(kin.bodies[kin.site_body[i]], model.site_bodyid[site_id])
                    tensor_close(kin.site_pos0[i], model.site_pos[site_id])
                    rm = np.zeros(9)
                    mujoco.mju_quat2Mat(rm, model.site_quat[site_id])
                    tensor_close(kin.site_rot0[i], rm.reshape(3, 3))
                # Trong v5 hai site có hướng cục bộ khác nhau.
                self.assertFalse(np.allclose(to_numpy(kin.site_rot0[0]), to_numpy(kin.site_rot0[1])))

    def test_05_joint_limits_in_radians(self):
        for name, model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(tuple(kin.qmin.shape), (10,))
                self.assertEqual(tuple(kin.qmax.shape), (10,))
                tensor_close(kin.qmin, np.deg2rad(JOINT_RANGE_DEGREES[:, 0]))
                tensor_close(kin.qmax, np.deg2rad(JOINT_RANGE_DEGREES[:, 1]))
                self.assertTrue(bool(torch.all(kin.qmin < kin.qmax)))
                for j, joint_name in enumerate(JOINT_NAMES):
                    jid = model.joint(joint_name).id
                    tensor_close(kin.qmin[j], model.jnt_range[jid, 0])
                    tensor_close(kin.qmax[j], model.jnt_range[jid, 1])

    def test_06_desc_independent_ancestry(self):
        """Đối chiếu ma trận desc bằng cây parent của MuJoCo, không dùng desc để tự xác nhận."""
        for name, model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(tuple(kin.desc.shape), (10, 11))
                expected = np.zeros((10, 11), dtype=np.float32)
                for ji, joint_name in enumerate(JOINT_NAMES):
                    joint_body = int(model.jnt_bodyid[model.joint(joint_name).id])
                    for bi, body_id in enumerate(kin.bodies):
                        expected[ji, bi] = float(ancestor_of(model, joint_body, body_id))
                tensor_close(kin.desc, expected, atol=0)
                np.testing.assert_array_equal(to_numpy(kin.desc[:, 0]), np.zeros(10))
                np.testing.assert_array_equal(expected.sum(axis=1),
                                              np.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1]))

    def test_07_site_desc_left_right(self):
        for name, _model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(tuple(kin.site_desc.shape), (2, 10))
                expected = np.array([[1] * 5 + [0] * 5, [0] * 5 + [1] * 5], dtype=np.float32)
                tensor_close(kin.site_desc, expected, atol=0)
                tensor_close(kin.site_desc, to_numpy(kin.desc[:, kin.site_body].T), atol=0)
                tensor_close(kin.legs[0], np.arange(5), atol=0)
                tensor_close(kin.legs[1], np.arange(5, 10), atol=0)

    def test_08_levels_data_and_parent_mapping(self):
        """Đối chiếu từng level với cây robot và kiểm tra skew/k2."""
        for name, _model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(len(kin.levels), 5)
                depth = [depth_for_local_body(kin, i) for i in range(11)]
                self.assertEqual(depth, [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
                for d, lv in enumerate(kin.levels, start=1):
                    ids = [i for i in range(11) if depth[i] == d]
                    prev = [i for i in range(11) if depth[i] == d - 1]
                    self.assertEqual(len(ids), 2)
                    tensor_close(lv["joint"], [kin.joint_of_body[i] for i in ids], atol=0)
                    tensor_close(lv["parent"], [prev.index(kin.parent[i]) for i in ids], atol=0)
                    tensor_close(lv["pos0"], to_numpy(kin.body_pos0[ids]))
                    tensor_close(lv["rot0"], to_numpy(kin.body_rot0[ids]))
                    tensor_close(lv["axis"], to_numpy(kin.joint_axis[lv["joint"]]))
                    self.assertEqual(tuple(lv["k"].shape), (2, 3, 3))
                    tensor_close(lv["k2"], to_numpy(lv["k"] @ lv["k"]))
                    for k, axis in zip(to_numpy(lv["k"]), to_numpy(lv["axis"])):
                        ax, ay, az = axis
                        expected_skew = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
                        np.testing.assert_allclose(k, expected_skew, atol=1e-7)

    def test_09_level_permutation_roundtrip(self):
        """Kiểm tra hoán vị level -> thứ tự khâu/khớp ban đầu."""
        for name, _model, kin in self.models:
            with self.subTest(xml=name):
                body_order = [0]
                joint_order = []
                depth = [depth_for_local_body(kin, i) for i in range(11)]
                for d, level in enumerate(kin.levels, start=1):
                    body_order.extend(i for i in range(11) if depth[i] == d)
                    joint_order.extend(level["joint"].tolist())
                self.assertEqual(sorted(body_order), list(range(11)))
                self.assertEqual(sorted(joint_order), list(range(10)))
                tensor_close(kin.body_from_level, [body_order.index(i) for i in range(11)], atol=0)
                tensor_close(kin.joint_from_level, [joint_order.index(j) for j in range(10)], atol=0)
                self.assertEqual(to_numpy(kin.body_from_level).dtype.kind, "i")
                self.assertEqual(to_numpy(kin.joint_from_level).dtype.kind, "i")

    def test_10_device_dtype_and_finite(self):
        for name, _model, kin in self.models:
            with self.subTest(xml=name):
                self.assertEqual(kin.device.type, "cpu")
                tensor_fields = (
                    "body_pos0", "body_rot0", "joint_axis", "mass", "ipos",
                    "site_pos0", "site_rot0", "qmin", "qmax", "desc", "site_desc",
                )
                for field in tensor_fields:
                    v = getattr(kin, field)
                    self.assertEqual(v.device.type, "cpu", field)
                    self.assertEqual(v.dtype, torch.float32, field)
                    self.assertTrue(bool(torch.isfinite(v).all()), field)
                for field in ("body_from_level", "joint_from_level"):
                    v = getattr(kin, field)
                    self.assertEqual(v.dtype, torch.long, field)
                for level in kin.levels:
                    for key in ("parent", "joint"):
                        self.assertEqual(level[key].dtype, torch.long)
                    for key in ("pos0", "rot0", "axis", "k", "k2"):
                        self.assertEqual(level[key].dtype, torch.float32)
                        self.assertTrue(bool(torch.isfinite(level[key]).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "Không có CUDA; chỉ chạy CPU")
    def test_11_optional_cuda(self):
        model = self.models[0][1]
        kin = RobotKinematics(model, device="cuda")
        self.assertEqual(kin.joint_axis.device.type, "cuda")
        self.assertEqual(kin.site_desc.device.type, "cuda")
        self.assertEqual(kin.levels[0]["k"].device.type, "cuda")
        tensor_close(kin.site_desc, np.array([[1] * 5 + [0] * 5, [0] * 5 + [1] * 5]))


def main():
    parser = argparse.ArgumentParser(description="Test RobotKinematics.__init__() - v5")
    parser.add_argument("--model-only", action="store_true", help="Chỉ kiểm tra open_duck_mini_v5.xml")
    parser.add_argument("--verbose", action="store_true", help="Hiển thị chi tiết từng bài test")
    args = parser.parse_args()
    TestRobotKinematicsInit.model_only = args.model_only
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRobotKinematicsInit)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if result.wasSuccessful():
        model_count = 1 if args.model_only else 2
        print(f"\nPASS: __init__() v5 trên {model_count} mô hình (robot-only / scene).")
        print("Chưa kiểm tra FK/Jacobian số học; sẽ kiểm tra ở các hàm tiếp theo.")
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
