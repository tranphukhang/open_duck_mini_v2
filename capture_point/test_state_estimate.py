"""Tests for Ground Truth CoM, feet, and FootstepPlanner.reanchor_x.

Place alongside state_estimate.py and footstep_planner.py in capture_point/.
Run:  python .\\test_state_estimate.py -v

Unit tests mock ONLY mj_subtreeVel for fake MuJoCo objects.
Integration tests use the actual xmls/scene_flat_terrian_v5.xml model.
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
from footstep_planner import FootstepPlanner


class FakeID:
    def __init__(self, value):
        self.id = value


class FakeModel:
    def body(self, name):
        if name != "base_link":
            raise KeyError(name)
        return FakeID(2)

    def site(self, name):
        if name == "L6_IK":
            return FakeID(1)
        if name == "R6_IK":
            return FakeID(3)
        raise KeyError(name)


class FakeData:
    def __init__(self,
                 pos=(1.0, 2.0, 3.0), vel=(0.1, 0.2, 0.3),
                 left=(0.5, 1.8, 0.3), right=(1.2, 1.7, -0.1)):
        self.subtree_com = np.zeros((4, 3), dtype=np.float64)
        self.subtree_linvel = np.zeros((4, 3), dtype=np.float64)
        self.site_xpos = np.zeros((4, 3), dtype=np.float64)
        self.subtree_com[2] = pos
        self.subtree_linvel[2] = vel
        self.site_xpos[1] = left
        self.site_xpos[3] = right


class TestGroundTruthUnit(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.est = GroundTruthStateEstimator(self.model)
        self.data = FakeData()
        # Native MuJoCo only accepts real MjModel/MjData. Mock it for fakes.
        self._patcher = patch.object(mujoco, "mj_subtreeVel")
        self.mock_subtree_vel = self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_01_model_and_sites(self):
        self.assertEqual(self.est.base_id, 2)
        self.assertEqual(self.est.foot_site_ids, (1, 3))
        with self.assertRaises(ValueError):
            GroundTruthStateEstimator(self.model, base_body="missing")
        with self.assertRaises(ValueError):
            GroundTruthStateEstimator(self.model, foot_sites=("L6_IK", "missing"))
        with self.assertRaises(ValueError):
            GroundTruthStateEstimator(self.model, foot_sites=("L6_IK", "L6_IK"))

    def test_02_com_position_world(self):
        torch.testing.assert_close(self.est.get_com_position(self.data), torch.tensor([[1., 2., 3.]]))

    def test_03_com_velocity_refresh(self):
        out = self.est.get_com_velocity(self.data)
        self.mock_subtree_vel.assert_called_once_with(self.model, self.data)
        torch.testing.assert_close(out, torch.tensor([[0.1, 0.2, 0.3]]))

    def test_04_foot_position_world_left_right(self):
        feet = self.est.get_foot_position(self.data)
        self.assertEqual(tuple(feet.shape), (1, 2, 3))
        torch.testing.assert_close(feet[0], torch.tensor([[0.5, 1.8, 0.3], [1.2, 1.7, -0.1]]))

    def test_05_default_frame_and_relative_vectors(self):
        s = self.est.update(self.data)
        self.assertIsInstance(s, GroundTruthState)
        self.assertEqual(s.valid.tolist(), [True])
        torch.testing.assert_close(s.com_pos, torch.tensor([[1., 2., 3.]]))
        torch.testing.assert_close(s.com_vel, torch.tensor([[0.1, 0.2, 0.3]]))
        torch.testing.assert_close(s.com_relative_to_feet[0], torch.tensor([
            [0.5, 0.2, 2.7], [-0.2, 0.3, 3.1]
        ]))
        self.assertEqual(tuple(s.feet_pos.shape), (1, 2, 3))
        self.assertEqual(tuple(s.com_relative_to_feet.shape), (1, 2, 3))

    def test_06_translation_changes_position_not_relative_or_velocity(self):
        s0 = self.est.update(self.data)
        s = self.est.update(self.data, origin_world=[0.3, 1.2, -0.5])
        torch.testing.assert_close(s.com_pos, torch.tensor([[0.7, 0.8, 3.5]]))
        torch.testing.assert_close(s.com_vel, s0.com_vel)
        torch.testing.assert_close(s.feet_pos, s0.feet_pos - torch.tensor([0.3, 1.2, -0.5]))
        torch.testing.assert_close(s.com_relative_to_feet, s0.com_relative_to_feet)

    def test_07_yaw_rotation_applies_to_com_and_feet(self):
        R_wp = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        s = self.est.update(self.data, rotation_world_from_plan=R_wp)
        torch.testing.assert_close(s.com_pos, torch.tensor([[2., -1., 3.]]))
        torch.testing.assert_close(s.com_vel, torch.tensor([[0.2, -0.1, 0.3]]))
        torch.testing.assert_close(s.feet_pos[0], torch.tensor([
            [1.8, -0.5, 0.3], [1.7, -1.2, -0.1]
        ]))
        torch.testing.assert_close(s.com_relative_to_feet, s.com_pos[:, None] - s.feet_pos)

    def test_08_batches_with_distinct_origins(self):
        d2 = FakeData(pos=(4, 5, 6), vel=(-1, -2, -3), left=(2, 3, 0), right=(5, 2, 0))
        s = self.est.update([self.data, d2], origin_world=[[1, 0, 0], [0, 1, 0]])
        self.assertEqual(tuple(s.com_pos.shape), (2, 3))
        self.assertEqual(tuple(s.feet_pos.shape), (2, 2, 3))
        self.assertEqual(s.valid.tolist(), [True, True])
        torch.testing.assert_close(s.com_pos, torch.tensor([[0., 2., 3.], [4., 4., 6.]]))
        torch.testing.assert_close(s.com_relative_to_feet[1], torch.tensor([[2., 2., 6.], [-1., 3., 6.]]))
        self.assertEqual(self.mock_subtree_vel.call_count, 2)

    def test_09_no_shared_numpy_memory(self):
        s = self.est.update(self.data)
        self.data.subtree_com[2] = 100
        self.data.subtree_linvel[2] = 100
        self.data.site_xpos[1] = 100
        self.data.site_xpos[3] = 100
        torch.testing.assert_close(s.com_pos, torch.tensor([[1., 2., 3.]]))
        torch.testing.assert_close(s.com_vel, torch.tensor([[0.1, 0.2, 0.3]]))
        torch.testing.assert_close(s.feet_pos[0, 0], torch.tensor([0.5, 1.8, 0.3]))
        torch.testing.assert_close(s.com_relative_to_feet[0, 1], torch.tensor([-0.2, 0.3, 3.1]))

    def test_10_bad_shapes_and_empty_batch(self):
        with self.assertRaises(ValueError):
            self.est.update([])
        with self.assertRaises(ValueError):
            self.est.update(self.data, origin_world=[1, 2])
        with self.assertRaises(ValueError):
            self.est.update(self.data, rotation_world_from_plan=torch.eye(4))

    def test_11_nonfinite_velocity_and_foot(self):
        self.data.subtree_linvel[2, 0] = np.nan
        s = self.est.update(self.data)
        self.assertFalse(bool(s.valid[0]))
        self.data.subtree_linvel[2, 0] = 0.1
        self.data.site_xpos[3, 1] = np.inf
        s = self.est.update(self.data)
        self.assertFalse(bool(s.valid[0]))

    def test_12_relative_position_equal_foot_displacement(self):
        s = self.est.update(self.data, origin_world=[0.8, 0.7, 0.6])
        # (CoM-foot_left) - (CoM-foot_right) = foot_right - foot_left.
        torch.testing.assert_close(
            s.com_relative_to_feet[:, 0] - s.com_relative_to_feet[:, 1],
            s.feet_pos[:, 1] - s.feet_pos[:, 0],
        )

    def test_13_reanchor_x_uses_ground_truth_relative_feet(self):
        # End of right-foot swing: planned right x = 0.10, actual x = 0.12.
        self.data = FakeData(pos=(0.06, 0, 0.18), vel=(0, 0, 0),
                             left=(0.00, 0.065, 0), right=(0.12, -0.065, 0))
        gt = self.est.update(self.data)
        planner = FootstepPlanner(num_envs=1)
        feet_initial = torch.tensor([[[0., 0.065, 0.], [0., -0.065, 0.]]])
        planner.reset(torch.tensor([0]), feet_initial)
        planner.updated_landing[0, 0] = 0.10
        j_now = torch.tensor([1])
        new_step = torch.tensor([True])
        stance = torch.tensor([1])  # New stance is right foot
        planner.update_new_step(j_now, new_step, stance, torch.zeros(1, 2))
        com_state = torch.zeros(1, 2, 3)
        com_state[0, 0, 0] = 0.07
        delta = planner.reanchor_x(j_now, new_step, stance,
                                   gt.com_relative_to_feet, gt.valid, com_state)
        torch.testing.assert_close(delta, torch.tensor([0.02]), atol=1e-6, rtol=0)
        torch.testing.assert_close(planner.feet_start[0, 1, 0], torch.tensor(0.12), atol=1e-6, rtol=0)
        torch.testing.assert_close(com_state[0, 0, 0], torch.tensor(0.09), atol=1e-6, rtol=0)

    def test_14_reanchor_x_skips_invalid_ground_truth(self):
        self.data = FakeData(pos=(0.06, 0, 0.18), vel=(0, 0, 0),
                             left=(0.0, 0.065, 0), right=(0.12, -0.065, 0))
        self.data.site_xpos[3, 0] = np.nan
        gt = self.est.update(self.data)
        self.assertFalse(bool(gt.valid[0]))
        planner = FootstepPlanner(num_envs=1)
        planner.reset(torch.tensor([0]), torch.zeros((1, 2, 3)))
        com_state = torch.zeros((1, 2, 3))
        delta = planner.reanchor_x(
            torch.tensor([1]), torch.tensor([True]), torch.tensor([1]),
            gt.com_relative_to_feet, gt.valid, com_state,
        )
        torch.testing.assert_close(delta, torch.zeros(1))
        self.assertTrue(torch.isfinite(planner.feet_start).all())
        self.assertTrue(torch.isfinite(com_state).all())


@unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
class TestRealMuJoCoV5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_15_com_position_matches_mass_weighted_body_com(self):
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
        self.assertTrue(bool(result.valid[0]))
        print(f"\n[MuJoCo v5] robot mass = {masses.sum():.4f} kg")

    def test_16_com_velocity_matches_finite_difference(self):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0
        d.qvel[:3] = [0.13, -0.09, 0.02]
        d.qvel[6:] = np.linspace(-0.35, 0.4, len(d.qvel) - 6)
        mujoco.mj_forward(self.model, d)
        got = self.est.update(d).com_vel[0].numpy()
        eps = 1e-6
        qpos_next = d.qpos.copy()
        mujoco.mj_integratePos(self.model, qpos_next, d.qvel, eps)
        d_next = mujoco.MjData(self.model)
        d_next.qpos[:] = qpos_next
        d_next.qvel[:] = d.qvel
        mujoco.mj_forward(self.model, d_next)
        expected = (d_next.subtree_com[self.est.base_id] - d.subtree_com[self.est.base_id]) / eps
        np.testing.assert_allclose(got, expected, atol=5e-4, rtol=5e-4)

    def test_17_foot_sites_match_mujoco_and_relative_vectors(self):
        gt = self.est.update(self.data)
        site_ids = [self.model.site("L6_IK").id, self.model.site("R6_IK").id]
        expected_feet = self.data.site_xpos[site_ids]
        np.testing.assert_allclose(gt.feet_pos[0].numpy(), expected_feet, atol=1e-6)
        np.testing.assert_allclose(
            gt.com_relative_to_feet[0].numpy(),
            self.data.subtree_com[self.est.base_id][None, :] - expected_feet,
            atol=1e-6,
        )
        self.assertEqual(gt.feet_pos.shape, (1, 2, 3))
        self.assertEqual(gt.com_relative_to_feet.shape, (1, 2, 3))

    def test_18_frame_transform_com_and_feet(self):
        origin = np.array([0.1, -0.05, 0.02])
        R_wp = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        gt = self.est.update(self.data, origin_world=origin,
                             rotation_world_from_plan=R_wp)
        p_gt = R_wp.T @ (self.data.subtree_com[self.est.base_id] - origin)
        f_gt = (R_wp.T @ (self.data.site_xpos[list(self.est.foot_site_ids)] - origin).T).T
        np.testing.assert_allclose(gt.com_pos[0].numpy(), p_gt, atol=1e-6)
        np.testing.assert_allclose(gt.feet_pos[0].numpy(), f_gt, atol=1e-6)
        np.testing.assert_allclose(gt.com_relative_to_feet[0].numpy(), p_gt[None, :] - f_gt, atol=1e-6)

    def test_19_capture_point_interface(self):
        from capture_point import CapturePoint
        gt = self.est.update(self.data)
        h = max(0.05, float(gt.com_pos[0, 2] - gt.feet_pos[0, 0, 2]))
        cp = CapturePoint(com_height=h)
        com_plan = torch.zeros((1, 2, 3), dtype=torch.float32)
        com_plan[:, :, 0] = gt.com_pos[:, :2]
        com_plan[:, :, 1] = gt.com_vel[:, :2]
        _, _, error = cp.compute_dcm_error(gt.com_pos, gt.com_vel, com_plan)
        torch.testing.assert_close(error, torch.zeros((1, 2)), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
