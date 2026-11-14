r"""Validation tests for load_compensation.py (Open Duck Mini V2).

Place this file and load_compensation.py in capture_point/, then run:

    python .\test_load_compensation.py -v
    python .\test_load_compensation.py --skip-real -v
    python .\test_load_compensation.py --model ..\xmls\scene_flat_terrian_v5.xml -v

The first tests use synthetic FK/kinematic data, so they need only torch.
If MuJoCo and the robot's XML/meshes are available, integration tests also
run against the real MJCF v5 and RobotKinematics from the project.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_compensation import LoadCompensation, LoadCompensationParams


MODEL_PATH = Path(__file__).resolve().parent.parent / "xmls" / "scene_flat_terrian_v5.xml"
SKIP_REAL = False


def fake_robot(batch: int = 3, dtype: torch.dtype = torch.float64):
    """Create deterministic kinematics with a plausible 2-leg descendant map."""
    gen = torch.Generator(device="cpu").manual_seed(2405)
    masses = torch.tensor([0.7, 0.08, 0.10, 0.12, 0.15, 0.2,
                           0.08, 0.10, 0.12, 0.15, 0.2], dtype=dtype)
    desc = torch.zeros(10, 11, dtype=dtype)
    for j in range(5):
        desc[j, j + 1:6] = 1.0
        desc[j + 5, j + 6:11] = 1.0
    kin = SimpleNamespace(mass=masses, desc=desc, total_mass=float(masses.sum()))
    com_body = torch.randn(batch, 11, 3, generator=gen, dtype=dtype) * 0.06
    com_body[..., 2] += 0.18
    anchor = torch.randn(batch, 10, 3, generator=gen, dtype=dtype) * 0.04
    anchor[..., 2] += 0.15
    axis = torch.randn(batch, 10, 3, generator=gen, dtype=dtype)
    axis = torch.nn.functional.normalize(axis, dim=-1)
    sites = torch.zeros(batch, 2, 3, dtype=dtype)
    sites[:, 0, :] = torch.tensor([0.04, 0.060, 0.0], dtype=dtype)
    sites[:, 1, :] = torch.tensor([0.04, -0.060, 0.0], dtype=dtype)
    ks = SimpleNamespace(com_body=com_body, anchor=anchor,
                         axis=axis, site_pos=sites)
    return kin, ks


def reference_joint_torques_loop(kin, ks, forces, moments=None, g=9.81):
    """Independent scalar-loop oracle for author's projection formula."""
    bsz = ks.axis.shape[0]
    out = torch.zeros(bsz, 10, dtype=ks.axis.dtype)
    upward = np.array([0.0, 0.0, g])
    for b in range(bsz):
        for j in range(10):
            p = ks.anchor[b, j].detach().numpy()
            a = ks.axis[b, j].detach().numpy()
            mg = np.zeros(3)
            for body in range(kin.mass.numel()):
                if kin.desc[j, body].item():
                    c = ks.com_body[b, body].detach().numpy()
                    mg += np.cross(c - p, kin.mass[body].item() * upward)
            i = 0 if j < 5 else 1
            foot = ks.site_pos[b, i].detach().numpy()
            f = forces[b, i].detach().numpy()
            lever = np.cross(foot - p, f)
            if moments is not None:
                lever += moments[b, i].detach().numpy()
            out[b, j] = np.dot(a, mg - lever)
    return out


class TestLoadCompensation(unittest.TestCase):
    def setUp(self):
        self.kin, self.ks = fake_robot(batch=3)
        self.lc = LoadCompensation(self.kin)
        self.feet = self.ks.site_pos.clone()
        self.zmp = torch.tensor([[0.04, 0.00], [0.03, 0.035],
                                 [0.09, -0.020]], dtype=torch.float64)
        self.contact = torch.tensor([[True, True], [True, False],
                                     [False, True]])
        self.acc = torch.tensor([[0.0, 0.0], [0.2, -0.3],
                                 [-0.1, 0.05]], dtype=torch.float64)
        self.q_ik = torch.randn(3, 10, dtype=torch.float64) * 0.1

    def test_01_default_parameters(self):
        p = self.lc.params
        self.assertAlmostEqual(p.gravity, 9.81)
        self.assertAlmostEqual(p.servo_kp, 13.37)
        self.assertAlmostEqual(p.zmp_box_back, 0.022)
        self.assertAlmostEqual(p.zmp_box_front, 0.045)
        self.assertAlmostEqual(p.zmp_offset_out, 0.007)
        self.assertAlmostEqual(p.zmp_box_half_width, 0.003)

    def test_02_total_force_at_rest(self):
        f = self.lc.compute_total_force(torch.zeros(3, 2, dtype=torch.float64))
        torch.testing.assert_close(f[:, :2], torch.zeros_like(f[:, :2]))
        torch.testing.assert_close(f[:, 2],
                                   torch.full((3,), self.kin.total_mass * 9.81, dtype=f.dtype))

    def test_03_total_force_with_acceleration(self):
        f = self.lc.compute_total_force(self.acc)
        torch.testing.assert_close(f[:, :2], self.kin.total_mass * self.acc)
        torch.testing.assert_close(f[:, 2],
                                   self.acc[:, 0] * 0 + self.kin.total_mass * 9.81)

    def test_04_double_support_even_split(self):
        f = self.lc.compute_total_force(self.acc)
        out = self.lc.distribute_contact_forces(f, torch.ones_like(self.contact),
                                                self.feet, self.zmp * 0)
        # Left y=+0.06 and right y=-0.06; y_zmp=0 => lambda=0.5.
        torch.testing.assert_close(out[:, 0], 0.5 * f)
        torch.testing.assert_close(out[:, 1], 0.5 * f)
        torch.testing.assert_close(out.sum(dim=1), f)

    def test_05_double_support_fraction_and_clamp(self):
        f = self.lc.compute_total_force(self.acc)
        z = self.zmp.clone()
        z[:, 1] = torch.tensor([0.03, 0.50, -0.50], dtype=z.dtype)
        out = self.lc.distribute_contact_forces(f, torch.ones_like(self.contact), self.feet, z)
        fractions = torch.tensor([0.75, 1.0, 0.0], dtype=f.dtype)
        torch.testing.assert_close(out[:, 0], fractions[:, None] * f)
        torch.testing.assert_close(out[:, 1], (1.0 - fractions)[:, None] * f)

    def test_06_single_support(self):
        f = self.lc.compute_total_force(self.acc)
        out = self.lc.distribute_contact_forces(f, self.contact, self.feet, self.zmp)
        torch.testing.assert_close(out[1, 0], f[1])
        torch.testing.assert_close(out[1, 1], torch.zeros(3, dtype=f.dtype))
        torch.testing.assert_close(out[2, 0], torch.zeros(3, dtype=f.dtype))
        torch.testing.assert_close(out[2, 1], f[2])

    def test_07_no_contact_zero_forces(self):
        f = self.lc.compute_total_force(self.acc)
        zero_contact = torch.zeros_like(self.contact)
        out = self.lc.distribute_contact_forces(f, zero_contact, self.feet, self.zmp)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_08_zero_foot_lateral_separation(self):
        f = self.lc.compute_total_force(self.acc)
        feet = self.feet.clone()
        feet[:, 0, 1] = feet[:, 1, 1]  # matches author's denominator guard
        out = self.lc.distribute_contact_forces(f, torch.ones_like(self.contact), feet, self.zmp)
        self.assertTrue(torch.isfinite(out).all())

    def test_09_double_support_cop_outward(self):
        f = self.lc.compute_total_force(self.acc)
        contact = torch.ones_like(self.contact)
        forces = self.lc.distribute_contact_forces(f, contact, self.feet, self.zmp)
        moments = self.lc.compute_contact_moments(forces, contact, self.feet, self.zmp)
        self.assertEqual(tuple(moments.shape), (3, 2, 3))
        # For left/right CoP y = +/- 0.007, independent of ZMP y in DS.
        x = (self.zmp[:, None, 0] - self.feet[:, :, 0]).clamp(-0.022, 0.045)
        r = torch.stack((x, x * 0 + torch.tensor([0.007, -0.007]),
                         torch.zeros_like(x)), dim=-1)
        torch.testing.assert_close(moments, torch.cross(r, forces, dim=-1))

    def test_10_single_support_cop_bounds(self):
        force = torch.tensor([[[1.0, 2.0, 12.0], [3.0, -1.0, 16.0]]], dtype=torch.float64)
        feet = self.feet[:1]
        # y=+0.40 far outside left strip; x +0.50 far beyond front bound.
        zmp = torch.tensor([[0.54, 0.40]], dtype=torch.float64)
        contact = torch.tensor([[True, False]])
        moments = self.lc.compute_contact_moments(force, contact, feet, zmp)
        r_left = torch.tensor([0.045, 0.010, 0.0], dtype=torch.float64)
        torch.testing.assert_close(moments[0, 0], torch.linalg.cross(r_left, force[0, 0]))
        torch.testing.assert_close(moments[0, 1], torch.zeros(3, dtype=torch.float64))
        # x = -0.50, y far below right strip: clamp x to -0.022, y to -0.010.
        zmp2 = torch.tensor([[-0.46, -0.40]], dtype=torch.float64)
        contact2 = torch.tensor([[False, True]])
        moments2 = self.lc.compute_contact_moments(force, contact2, feet, zmp2)
        r_right = torch.tensor([-0.022, -0.010, 0.0], dtype=torch.float64)
        torch.testing.assert_close(moments2[0, 1], torch.linalg.cross(r_right, force[0, 1]))

    def test_11_disable_cop_moment(self):
        lc = LoadCompensation(self.kin, LoadCompensationParams(cop_moment=False))
        f = lc.compute_total_force(self.acc)
        forces = lc.distribute_contact_forces(f, self.contact, self.feet, self.zmp)
        self.assertIsNone(lc.compute_contact_moments(forces, self.contact, self.feet, self.zmp))
        result = lc.compute(self.q_ik, self.ks, self.contact, self.feet, self.acc, self.zmp)
        self.assertIsNone(result.contact_moments)

    def test_12_joint_torques_vs_independent_loop(self):
        f = self.lc.compute_total_force(self.acc)
        forces = self.lc.distribute_contact_forces(f, self.contact, self.feet, self.zmp)
        moments = self.lc.compute_contact_moments(forces, self.contact, self.feet, self.zmp)
        tau = self.lc.compute_joint_torques(self.ks, forces, moments)
        oracle = reference_joint_torques_loop(self.kin, self.ks, forces, moments)
        torch.testing.assert_close(tau, oracle, atol=1e-12, rtol=1e-12)

    def test_13_joint_torques_without_moments(self):
        f = self.lc.compute_total_force(self.acc)
        forces = self.lc.distribute_contact_forces(f, self.contact, self.feet, self.zmp)
        tau = self.lc.compute_joint_torques(self.ks, forces, None)
        oracle = reference_joint_torques_loop(self.kin, self.ks, forces, None)
        torch.testing.assert_close(tau, oracle, atol=1e-12, rtol=1e-12)

    def test_14_scalar_equilibrium_sign(self):
        # One descendant body of 2 kg at x=0.2; upward contact fz=10 at x=0.5;
        # joint axis +Y. tau_g = -2*9.81*0.2; contact moment_y = -5;
        # tau = tau_g - (-5) = 1.076 N*m.
        kin, ks = fake_robot(batch=1)
        kin.mass[:] = 0.0
        kin.mass[1] = 2.0
        kin.total_mass = 2.0
        kin.desc[:] = 0.0
        kin.desc[0, 1] = 1.0
        ks.anchor[:] = 0.0
        ks.com_body[:] = 0.0
        ks.com_body[0, 1, 0] = 0.2
        ks.axis[:] = 0.0
        ks.axis[:, :, 1] = 1.0
        ks.site_pos[:] = 0.0
        ks.site_pos[0, 0, 0] = 0.5
        forces = torch.zeros(1, 2, 3, dtype=torch.float64)
        forces[0, 0, 2] = 10.0
        comp = LoadCompensation(kin)
        actual = comp.compute_joint_torques(ks, forces)
        self.assertAlmostEqual(actual[0, 0].item(), -2 * 9.81 * 0.2 + 5.0, places=12)
        # All left-leg joints see the same foot force; only joint 0 has the
        # artificial descendant mass in this isolated sign-convention test.
        torch.testing.assert_close(actual[0, 1:5], torch.full((4,), 5.0, dtype=torch.float64))
        torch.testing.assert_close(actual[0, 5:], torch.zeros(5, dtype=torch.float64))

    def test_15_apply_gain_alpha(self):
        tau = torch.full((3, 10), 1.337, dtype=torch.float64)
        lc = LoadCompensation(self.kin, LoadCompensationParams(load=0.5))
        cmd, dq = lc.apply_compensation(self.q_ik, tau)
        torch.testing.assert_close(dq, torch.full_like(dq, 0.05))
        torch.testing.assert_close(cmd, self.q_ik + 0.05)

    def test_16_load_zero_unmodified_target(self):
        lc = LoadCompensation(self.kin, LoadCompensationParams(load=0.0))
        res = lc.compute(self.q_ik, self.ks, self.contact, self.feet, self.acc, self.zmp)
        torch.testing.assert_close(res.q_cmd, self.q_ik)
        torch.testing.assert_close(res.delta_q, torch.zeros_like(self.q_ik))

    def test_17_full_compute_vs_separate_calls(self):
        res = self.lc.compute(self.q_ik, self.ks, self.contact, self.feet, self.acc, self.zmp)
        total = self.lc.compute_total_force(self.acc)
        forces = self.lc.distribute_contact_forces(total, self.contact, self.feet, self.zmp)
        moments = self.lc.compute_contact_moments(forces, self.contact, self.feet, self.zmp)
        tau = self.lc.compute_joint_torques(self.ks, forces, moments)
        cmd, dq = self.lc.apply_compensation(self.q_ik, tau)
        for name, expected in (("total_force", total), ("contact_forces", forces),
                               ("contact_moments", moments), ("joint_torques", tau),
                               ("q_cmd", cmd), ("delta_q", dq)):
            torch.testing.assert_close(getattr(res, name), expected)
        self.assertEqual(tuple(res.q_cmd.shape), (3, 10))
        self.assertTrue(torch.isfinite(res.q_cmd).all())

    def test_18_gradients_finite(self):
        # Important when used later in differentiable batched training.
        q = self.q_ik.clone().requires_grad_(True)
        acc = self.acc.clone().requires_grad_(True)
        zmp = self.zmp.clone().requires_grad_(True)
        res = self.lc.compute(q, self.ks, self.contact, self.feet, acc, zmp)
        loss = (res.q_cmd ** 2).sum()
        dq, da, dz = torch.autograd.grad(loss, (q, acc, zmp))
        self.assertTrue(torch.isfinite(dq).all())
        self.assertTrue(torch.isfinite(da).all())
        self.assertTrue(torch.isfinite(dz).all())

    def test_19_bad_parameters(self):
        with self.assertRaises(ValueError):
            LoadCompensation(self.kin, LoadCompensationParams(servo_kp=0.0))
        with self.assertRaises(ValueError):
            LoadCompensation(self.kin, LoadCompensationParams(load=1.1))
        with self.assertRaises(ValueError):
            self.lc.compute_total_force(torch.zeros(3, 3))


class TestRealMuJoCoIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SKIP_REAL:
            raise unittest.SkipTest("--skip-real specified")
        try:
            import mujoco
            from robot_kinematics import RobotKinematics
        except ImportError as exc:
            raise unittest.SkipTest(f"Requires MuJoCo / RobotKinematics: {exc}")
        if not MODEL_PATH.exists():
            raise unittest.SkipTest(f"Robot MJCF file not found: {MODEL_PATH}")

        cls.mujoco = mujoco
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        cls.data = mujoco.MjData(cls.model)
        key_id = cls.model.key("home").id
        mujoco.mj_resetDataKeyframe(cls.model, cls.data, key_id)
        mujoco.mj_forward(cls.model, cls.data)
        cls.kin = RobotKinematics(cls.model, device="cpu")
        cls.comp = LoadCompensation(cls.kin)
        cls.joint_names = [
            "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
            "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
        ]

    def pose_fk(self):
        pos = torch.as_tensor(np.array(self.data.qpos[:3]), dtype=torch.float32)[None, :]
        R_np = np.empty(9)
        self.mujoco.mju_quat2Mat(R_np, self.data.qpos[3:7])
        rot = torch.as_tensor(R_np.reshape(3, 3), dtype=torch.float32)[None, :, :]
        q = torch.tensor([[self.data.qpos[self.model.joint(n).qposadr[0]]
                            for n in self.joint_names]], dtype=torch.float32)
        return q, self.kin.forward_kinematics(pos, rot, q)

    def test_20_real_mjcf_fk_and_load(self):
        q, ks = self.pose_fk()
        sites = [self.model.site(n).id for n in ("L6_IK", "R6_IK")]
        expected_sites = torch.as_tensor(self.data.site_xpos[sites].copy(), dtype=torch.float32)
        torch.testing.assert_close(ks.site_pos[0], expected_sites, atol=3e-4, rtol=0)
        contact = torch.tensor([[True, True]])
        acc = torch.zeros(1, 2)
        zmp = ks.com[:, :2].clone()
        result = self.comp.compute(q, ks, contact, ks.site_pos, acc, zmp)
        self.assertEqual(result.q_cmd.shape, (1, 10))
        self.assertTrue(torch.isfinite(result.q_cmd).all())
        self.assertTrue(torch.isfinite(result.joint_torques).all())
        expected = self.kin.total_mass * 9.81
        self.assertAlmostEqual(result.contact_forces[0, :, 2].sum().item(),
                               expected, delta=1e-5)
        print("\n[MuJoCo v5] robot mass = %.4f kg" % self.kin.total_mass)
        print("[MuJoCo v5] max |tau_load| = %.5f N*m" % result.joint_torques.abs().max().item())
        print("[MuJoCo v5] max |delta_q| = %.5f deg" %
              math.degrees(result.delta_q.abs().max().item()))

    def test_21_real_mjcf_single_support(self):
        q, ks = self.pose_fk()
        acc = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
        zmp = ks.site_pos[:, 0, :2].clone()
        result = self.comp.compute(q, ks, torch.tensor([[True, False]]),
                                   ks.site_pos, acc, zmp)
        torch.testing.assert_close(result.contact_forces[:, 1],
                                   torch.zeros_like(result.contact_forces[:, 1]))
        torch.testing.assert_close(result.contact_forces[:, 0], result.total_force)
        self.assertTrue(torch.isfinite(result.q_cmd).all())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test load compensation, including optional MJCF v5")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="Path to scene_flat_terrian_v5.xml")
    parser.add_argument("--skip-real", action="store_true", help="Only run torch-based synthetic tests")
    args, unittest_args = parser.parse_known_args()
    MODEL_PATH = args.model.resolve()
    SKIP_REAL = args.skip_real
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
