"""Tests for state_estimate.py, with real MuJoCo v5 integration when installed.

Place BOTH files in capture_point/, then run:
    python .\\test_state_estimate.py -v
    # If you renamed this test to test.py:
    python .\\test.py -v

Tests 01-10 use a minimal deterministic MuJoCo stand-in.
Tests 11-14 load xmls/scene_flat_terrian_v5.xml and exercise real MuJoCo.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    mujoco = types.ModuleType("mujoco")
    mujoco.MjModel = type("MjModel", (), {})
    mujoco.mj_subtreeVel = lambda model, data: None
    sys.modules["mujoco"] = mujoco

from state_estimate import GroundTruthState, GroundTruthStateEstimator


class FakeBody:
    id = 2


class FakeModel:
    def body(self, name):
        if name != "base_link":
            raise KeyError(name)
        return FakeBody()


class FakeData:
    def __init__(self, pos=(1., 2., 3.), vel=(0.1, 0.2, 0.3)):
        self.subtree_com = np.zeros((4, 3), dtype=np.float64)
        self.subtree_linvel = np.zeros((4, 3), dtype=np.float64)
        self.subtree_com[2] = pos
        self.subtree_linvel[2] = vel


class TestGroundTruthUnit(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.est = GroundTruthStateEstimator(self.model)
        self.data = FakeData()

        # MuJoCo's native binding only accepts real MjModel/MjData.
        # Unit tests use FakeModel/FakeData with preset Ground Truth buffers,
        # so bypass ONLY the native update call in these unit tests.
        # The real-MuJoCo integration tests below are never patched.
        self._subtree_vel_patcher = patch.object(mujoco, "mj_subtreeVel")
        self.mock_subtree_vel = self._subtree_vel_patcher.start()
        self.addCleanup(self._subtree_vel_patcher.stop)

    def test_01_base_body(self):
        self.assertEqual(self.est.base_id, 2)
        with self.assertRaises(ValueError):
            GroundTruthStateEstimator(self.model, base_body="missing")

    def test_02_world_com_position(self):
        value = self.est.get_com_position(self.data)
        self.assertEqual(value.shape, (1, 3))
        torch.testing.assert_close(value, torch.tensor([[1., 2., 3.]]))

    def test_03_world_com_velocity_calls_mujoco(self):
        with patch.object(mujoco, "mj_subtreeVel") as fn:
            value = self.est.get_com_velocity(self.data)
            fn.assert_called_once_with(self.model, self.data)
        torch.testing.assert_close(value, torch.tensor([[0.1, 0.2, 0.3]]))

    def test_04_default_planning_frame_equals_world(self):
        result = self.est.update(self.data)
        self.assertIsInstance(result, GroundTruthState)
        torch.testing.assert_close(result.com_pos, torch.tensor([[1., 2., 3.]]))
        torch.testing.assert_close(result.com_vel, torch.tensor([[0.1, 0.2, 0.3]]))
        self.assertEqual(result.valid.tolist(), [True])

    def test_05_translation_changes_position_only(self):
        result = self.est.update(self.data, origin_world=[0.3, 1.2, -0.5])
        torch.testing.assert_close(result.com_pos, torch.tensor([[0.7, 0.8, 3.5]]))
        torch.testing.assert_close(result.com_vel, torch.tensor([[0.1, 0.2, 0.3]]))

    def test_06_yaw_rotation_changes_position_and_velocity(self):
        # R_WP is a +90-degree rotation about world Z.
        R_wp = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        result = self.est.update(self.data, rotation_world_from_plan=R_wp)
        torch.testing.assert_close(result.com_pos, torch.tensor([[2., -1., 3.]]))
        torch.testing.assert_close(result.com_vel, torch.tensor([[0.2, -0.1, 0.3]]))

    def test_07_batches(self):
        d2 = FakeData(pos=(4., 5., 6.), vel=(-1., -2., -3.))
        origin = [[1., 0., 0.], [0., 1., 0.]]
        result = self.est.update([self.data, d2], origin_world=origin)
        torch.testing.assert_close(result.com_pos, torch.tensor([[0., 2., 3.], [4., 4., 6.]]))
        torch.testing.assert_close(result.com_vel, torch.tensor([[0.1, 0.2, 0.3], [-1., -2., -3.]]))
        self.assertEqual(result.valid.tolist(), [True, True])

    def test_08_no_shared_numpy_memory(self):
        result = self.est.update(self.data)
        self.data.subtree_com[2] = [100, 100, 100]
        self.data.subtree_linvel[2] = [100, 100, 100]
        torch.testing.assert_close(result.com_pos, torch.tensor([[1., 2., 3.]]))
        torch.testing.assert_close(result.com_vel, torch.tensor([[0.1, 0.2, 0.3]]))

    def test_09_bad_shapes_and_empty_data(self):
        with self.assertRaises(ValueError):
            self.est.update([])
        with self.assertRaises(ValueError):
            self.est.update(self.data, origin_world=[1, 2])
        with self.assertRaises(ValueError):
            self.est.update(self.data, rotation_world_from_plan=torch.eye(4))

    def test_10_nonfinite_detection(self):
        self.data.subtree_linvel[2, 0] = float("nan")
        result = self.est.update(self.data)
        self.assertFalse(bool(result.valid[0]))
        self.assertTrue(torch.isnan(result.com_vel[0, 0]))


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
class TestRealMuJoCoV5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test file is in repository/capture_point/; XML is in repository/xmls/.
        cls.xml_path = Path(__file__).resolve().parent.parent / "xmls" / "scene_flat_terrian_v5.xml"
        if not cls.xml_path.is_file():
            raise unittest.SkipTest(f"MJCF v5 not found: {cls.xml_path}")
        cls.model = mujoco.MjModel.from_xml_path(str(cls.xml_path))
        cls.est = GroundTruthStateEstimator(cls.model)
        cls.data = mujoco.MjData(cls.model)
        key_id = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(cls.model, cls.data, key_id)
        mujoco.mj_forward(cls.model, cls.data)

    def test_11_position_matches_mass_weighted_body_com(self):
        result = self.est.update(self.data)
        root = self.est.base_id
        bodies = []
        for bid in range(self.model.nbody):
            parent = bid
            while parent > 0 and parent != root:
                parent = int(self.model.body_parentid[parent])
            if parent == root:
                bodies.append(bid)
        masses = np.asarray(self.model.body_mass[bodies])
        expected = np.sum(self.data.xipos[bodies] * masses[:, None], axis=0) / masses.sum()
        np.testing.assert_allclose(result.com_pos[0].numpy(), expected, atol=2e-6)
        self.assertEqual(result.com_pos.shape, (1, 3))
        self.assertTrue(bool(result.valid[0]))
        print(f"\n[MuJoCo v5] robot mass = {masses.sum():.4f} kg")
        print(f"[MuJoCo v5] CoM position = {result.com_pos[0].tolist()}")

    def test_12_velocity_matches_com_finite_difference(self):
        # Arbitrary consistent generalized velocity. No physics stepping.
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0
        d.qvel[:3] = [0.13, -0.09, 0.02]
        d.qvel[6:] = np.linspace(-0.35, 0.4, len(d.qvel) - 6)
        mujoco.mj_forward(self.model, d)
        got = self.est.update(d).com_vel[0].numpy()

        eps = 1e-6
        qpos_forward = d.qpos.copy()
        mujoco.mj_integratePos(self.model, qpos_forward, d.qvel, eps)
        d_next = mujoco.MjData(self.model)
        d_next.qpos[:] = qpos_forward
        d_next.qvel[:] = d.qvel
        mujoco.mj_forward(self.model, d_next)
        expected = (d_next.subtree_com[self.est.base_id] - d.subtree_com[self.est.base_id]) / eps
        np.testing.assert_allclose(got, expected, atol=5e-4, rtol=5e-4)
        print(f"[MuJoCo v5] CoM vel GT = {got.tolist()}, finite diff = {expected.tolist()}")

    def test_13_transformed_world_com(self):
        p0 = np.array([0.1, -0.05, 0.02])
        result = self.est.update(self.data, origin_world=p0)
        expected = self.data.subtree_com[self.est.base_id] - p0
        np.testing.assert_allclose(result.com_pos[0].numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(result.com_vel[0].numpy(), self.data.subtree_linvel[self.est.base_id], atol=1e-6)

    def test_14_capture_point_interface(self):
        from capture_point import CapturePoint

        gt = self.est.update(self.data)
        h = max(0.05, float(gt.com_pos[0, 2] - self.data.site_xpos[self.model.site("L6_IK").id, 2]))
        cp = CapturePoint(com_height=h)
        com_plan = torch.zeros((1, 2, 3), dtype=torch.float32)
        com_plan[:, :, 0] = gt.com_pos[:, :2]
        com_plan[:, :, 1] = gt.com_vel[:, :2]
        dcm_meas, dcm_plan, error = cp.compute_dcm_error(gt.com_pos, gt.com_vel, com_plan)
        torch.testing.assert_close(dcm_meas, dcm_plan, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(error, torch.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
