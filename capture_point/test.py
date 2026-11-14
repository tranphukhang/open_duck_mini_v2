"""Integration tests for Open Duck Mini V2 WholeBodyIK (MJCF v5).

Place this file at <repo>/capture_point/test_whole_body_ik.py.
The repository currently names its module ``whole_body_IK.py`` (capital IK).

Run from repository root:
    python capture_point/test_whole_body_ik.py --verbose
    python capture_point/test_whole_body_ik.py --samples 8 --verbose
    python capture_point/test_whole_body_ik.py --model-only
    python capture_point/test_whole_body_ik.py --check-grad --verbose

Dependencies: numpy, torch, mujoco and xmls/assets_v3/ (MJCF mesh assets).

Conventions: the 16 generalized IK increments consist of world-frame base
translation (3), world-frame LEFT rotation increment (3), 10 hinge joints.
The target base orientation used by the author's initial WBIK is identity.

The optional --check-grad test specifically probes the exact zero rotation
vector's autograd. It may expose a NaN-gradient bug in the current Rodrigues
implementation, without conflating it with standard numerical IK correctness.
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
    raise SystemExit("Missing mujoco. Install with: pip install mujoco") from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Exact current GitHub filename (case matters on Linux).
try:
    from whole_body_IK import IkWeights, WholeBodyIK
except ImportError as exc:
    raise SystemExit(
        "Cannot import capture_point/whole_body_IK.py (note capital IK). "
        "Check the filename and place this test in capture_point/."
    ) from exc

from robot_kinematics import RobotKinematics

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)
HOME_Q = np.array([
    0.0, -0.174532925, 0.349065850, -0.698131701, 0.349065850,
    0.0, 0.174532925, -0.349065850, 0.698131701, -0.349065850,
], dtype=np.float64)
TOL = 5e-5
SEED = 20261114


def assert_near(actual, expected, label, atol=TOL, rtol=0.0):
    a = actual.detach().cpu().numpy() if torch.is_tensor(actual) else np.asarray(actual)
    b = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shapes differ: {a.shape} vs {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f"{label}: NaN/Inf detected")
    try:
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    except AssertionError as exc:
        difference = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{label}: max error={difference:.4e}\n{exc}") from exc


def skew_np(v):
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def so3_exp_np(v):
    """Independent SO(3) exponential, double precision, for test oracle."""
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v)
    K = skew_np(v)
    if theta < 1e-9:
        a = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        b = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def rotation_error_np(R_des, R_cur):
    R = np.asarray(R_des) @ np.asarray(R_cur).T
    return np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) * 0.5


def find_xml_dir():
    for root in (HERE, *HERE.parents):
        path = root / "xmls" / "open_duck_mini_v5.xml"
        if path.is_file():
            return path.parent
    raise FileNotFoundError(
        "Could not locate xmls/open_duck_mini_v5.xml. "
        "Place this test inside the repository's capture_point/ directory."
    )


def home_pose(model):
    p = np.array([0., 0., 0.275], dtype=np.float64)
    q = HOME_Q.copy()
    if model.nkey:
        for idx in range(model.nkey):
            if model.key(idx).name == "home":
                qpos = np.asarray(model.key_qpos[idx], dtype=np.float64)
                free_id = model.joint("floating_base").id
                adr = int(model.jnt_qposadr[free_id])
                p = qpos[adr:adr + 3].copy()
                q = np.array([
                    qpos[int(model.jnt_qposadr[model.joint(n).id])]
                    for n in JOINT_NAMES
                ], dtype=np.float64)
                break
    return p, q


def model_data_from_pose(model, p, R, q):
    """Set pose in MuJoCo independently from Python FK, then mj_forward."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    free_id = model.joint("floating_base").id
    adr = int(model.jnt_qposadr[free_id])
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).reshape(9))
    data.qpos[adr:adr + 3] = np.asarray(p, dtype=np.float64)
    data.qpos[adr + 3:adr + 7] = quat
    for j, name in enumerate(JOINT_NAMES):
        data.qpos[int(model.jnt_qposadr[model.joint(name).id])] = float(q[j])
    mujoco.mj_forward(model, data)
    return data


def weighted_cost(ik, J_error):
    w = ik.weight_vector
    return (J_error.square() * w.unsqueeze(0)).sum(-1)


class TestWholeBodyIK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        opt = cls.options
        cls.device = torch.device(opt.device)
        xml_dir = find_xml_dir()
        names = ["open_duck_mini_v5.xml"]
        if not opt.model_only:
            names.append("scene_flat_terrian_v5.xml")
        cls.records = []
        for name in names:
            xml_path = xml_dir / name
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            kin = RobotKinematics(model, device=opt.device)
            ik = WholeBodyIK(kin)
            p0, q0 = home_pose(model)
            device = cls.device
            base_pos = torch.tensor(p0, device=device, dtype=torch.float32)[None]
            base_rot = torch.eye(3, device=device)[None]
            q_home = torch.tensor(q0, device=device, dtype=torch.float32)[None]
            ks0 = kin.forward_kinematics(base_pos, base_rot, q_home)
            cls.records.append({
                "name": name, "model": model, "kin": kin, "ik": ik,
                "p0": base_pos, "R0": base_rot, "q0": q_home, "ks0": ks0,
            })

    def test_01_default_weights_and_custom_weights(self):
        expected = ([100.] * 3 + [100.] * 3 + [30.] * 3
                    + [100.] * 3 + [30.] * 3
                    + [0.05, 5., 5.] + [1e-3] * 10)
        for s in self.records:
            with self.subTest(xml=s["name"]):
                ik = s["ik"]
                self.assertIsInstance(ik.weights, IkWeights)
                self.assertIs(ik.kin, s["kin"])
                self.assertEqual(tuple(ik.weight_vector.shape), (28,))
                self.assertEqual(ik.weight_vector.device, self.device)
                assert_near(ik.weight_vector, expected, "default weights", atol=1e-6)
                custom = IkWeights(com=200., foot_pos=125., foot_rot=19.,
                                   base_rot=(0.1, 2., 3.), posture=0.02,
                                   damping=2e-3, iterations=3)
                alt = WholeBodyIK(s["kin"], weights=custom)
                ref = ([200.] * 3 + [125.] * 3 + [19.] * 3
                       + [125.] * 3 + [19.] * 3 + [0.1, 2., 3.]
                       + [0.02] * 10)
                assert_near(alt.weight_vector, ref, "custom weights", atol=1e-5)
                self.assertIs(alt.weights, custom)

    def test_02_rotation_error_identity_sign_and_batch(self):
        for s in self.records:
            ik = s["ik"]
            I = torch.eye(3, device=self.device)
            assert_near(ik.compute_rotation_error(I, I),
                        np.zeros(3), "same rotation", atol=1e-7)
            angle = 0.17
            Rz = torch.tensor(so3_exp_np([0, 0, angle]),
                              device=self.device, dtype=torch.float32)
            assert_near(ik.compute_rotation_error(Rz, I),
                        [0, 0, np.sin(angle)], "positive Z", atol=1e-6)
            assert_near(ik.compute_rotation_error(I, Rz),
                        [0, 0, -np.sin(angle)], "negative Z", atol=1e-6)
            two = ik.compute_rotation_error(
                torch.stack([I, Rz]).unsqueeze(0).repeat(2, 1, 1, 1),
                I.expand(2, 2, 3, 3),
            )
            self.assertEqual(two.shape, (2, 2, 3))
            assert_near(two[:, 0], torch.zeros(2, 3, device=self.device),
                        "batched identity", atol=1e-7)
            assert_near(two[:, 1, 2],
                        [np.sin(angle)] * 2, "batched positive Z", atol=1e-6)

    def test_03_rodrigues_zero_random_and_so3(self):
        rng = np.random.default_rng(SEED)
        vecs = np.vstack([np.zeros(3),
                          np.array([1e-7, -2e-7, 3e-7]),
                          np.array([0.15, -0.25, 0.12]),
                          rng.uniform(-1, 1, size=(8, 3))])
        for s in self.records:
            mats = s["ik"].rotation_vector_to_matrix(
                torch.tensor(vecs, device=self.device, dtype=torch.float32)
            )
            self.assertEqual(mats.shape, (len(vecs), 3, 3))
            assert_near(mats, np.stack([so3_exp_np(v) for v in vecs]),
                        "Rodrigues vs numpy oracle", atol=4e-6)
            R = mats.detach().cpu().numpy().astype(np.float64)
            assert_near(R @ np.transpose(R, (0, 2, 1)),
                        np.tile(np.eye(3), (len(R), 1, 1)),
                        "R R^T = I", atol=4e-6)
            assert_near(np.linalg.det(R), np.ones(len(R)),
                        "det(R)=1", atol=4e-6)

    def test_04_build_tasks_zero_reference_and_jacobian_blocks(self):
        for s in self.records:
            kin, ik = s["kin"], s["ik"]
            p, R, q, ks = s["p0"], s["R0"], s["q0"], s["ks0"]
            J, e = ik.build_tasks(ks, p, R, q, ks.com.clone(),
                                  ks.site_pos.clone(), ks.site_rot.clone(), q.clone())
            self.assertEqual(J.shape, (1, 28, 16))
            self.assertEqual(e.shape, (1, 28))
            assert_near(e, np.zeros((1, 28)), f'{s["name"]}:zero error', atol=1e-7)
            assert_near(J[:, 0:3], kin.compute_com_jacobian(ks, p),
                        "COM Jacobian", atol=1e-7)
            for foot in range(2):
                start = 3 if foot == 0 else 9
                assert_near(J[:, start:start + 3],
                            kin.compute_point_jacobian(
                                ks, ks.site_pos[:, foot], kin.site_desc[foot], p
                            ), f"foot {foot} pos Jacobian", atol=1e-7)
                assert_near(J[:, start + 3:start + 6],
                            kin.compute_rotation_jacobian(ks, kin.site_desc[foot]),
                            f"foot {foot} rot Jacobian", atol=1e-7)
            assert_near(J[:, 15:18], kin.compute_rotation_jacobian(ks, None),
                        "base orientation Jacobian", atol=1e-7)
            posture = np.zeros((10, 16))
            posture[:, 6:] = np.eye(10)
            assert_near(J[:, 18:28], posture[None],
                        "posture Jacobian", atol=1e-7)

    def test_05_build_tasks_residual_order(self):
        for s in self.records:
            ik, ks = s["ik"], s["ks0"]
            p, q = s["p0"], s["q0"]
            R = ik.rotation_vector_to_matrix(torch.tensor(
                [[0.02, -0.015, 0.025]], device=self.device
            ))
            state = s["kin"].forward_kinematics(p, R, q)
            com_delta = torch.tensor([[0.009, -0.004, 0.003]], device=self.device)
            foot_delta = torch.tensor([[[0.002, 0., 0.001],
                                        [-0.001, 0.002, 0.]]], device=self.device)
            q_ref = q + 0.015
            foot_r_des = state.site_rot.clone()
            foot_r_des[:, 0] = (
                ik.rotation_vector_to_matrix(torch.tensor(
                    [[0., 0.03, 0.]], device=self.device
                )) @ foot_r_des[:, 0]
            )
            J, e = ik.build_tasks(state, p, R, q, state.com + com_delta,
                                  state.site_pos + foot_delta, foot_r_des, q_ref)
            self.assertEqual(J.shape, (1, 28, 16))
            assert_near(e[:, 0:3], com_delta, "COM error", atol=1e-7)
            assert_near(e[:, 3:6], foot_delta[:, 0], "left pos error", atol=1e-7)
            assert_near(e[:, 9:12], foot_delta[:, 1], "right pos error", atol=1e-7)
            assert_near(e[:, 6:9], np.array([0., np.sin(0.03), 0.])[None],
                        "left orientation error", atol=2e-6)
            assert_near(e[:, 12:15], np.zeros((1, 3)),
                        "right orientation error", atol=1e-6)
            assert_near(e[:, 15:18], rotation_error_np(np.eye(3),
                                                       R[0].detach().cpu().numpy())[None],
                        "base orientation error", atol=1e-6)
            assert_near(e[:, 18:28], np.full((1, 10), 0.015),
                        "posture error", atol=1e-6)

    def test_06_solve_dls_against_independent_numpy_oracle(self):
        rng = np.random.default_rng(SEED + 1)
        J_np = rng.normal(size=(3, 28, 16)).astype(np.float32) * 0.2
        e_np = rng.normal(size=(3, 28)).astype(np.float32) * 0.04
        for s in self.records:
            ik = s["ik"]
            J = torch.tensor(J_np, device=self.device)
            e = torch.tensor(e_np, device=self.device)
            dx = ik.solve_dls(J, e)
            self.assertEqual(dx.shape, (3, 16))
            w = ik.weight_vector.detach().cpu().numpy().astype(np.float64)
            refs = []
            for n in range(3):
                j = J_np[n].astype(np.float64)
                err = e_np[n].astype(np.float64)
                h = j.T @ (w[:, None] * j) + ik.weights.damping * np.eye(16)
                b = j.T @ (w * err)
                refs.append(np.linalg.solve(h, b))
            assert_near(dx, np.stack(refs),
                        f'{s["name"]}:WDLS independent numpy solve',
                        atol=2e-4, rtol=2e-3)
            cost_zero = weighted_cost(ik, e)
            residual = e - (J @ dx.unsqueeze(-1)).squeeze(-1)
            cost_solution = weighted_cost(ik, residual) + (
                ik.weights.damping * dx.square().sum(-1)
            )
            self.assertTrue(torch.all(cost_solution < cost_zero).item(),
                            f'{s["name"]}:WDLS did not improve cost')

    def test_07_zero_error_and_invalid_solve_fallback(self):
        for s in self.records:
            ik = s["ik"]
            J = torch.randn(2, 28, 16, device=self.device)
            dx = ik.solve_dls(J, torch.zeros(2, 28, device=self.device))
            assert_near(dx, np.zeros((2, 16)), "zero error -> zero dx", atol=1e-7)
            e = torch.randn(2, 28, device=self.device)
            e[1, 0] = float('nan')
            result = ik.solve_dls(J, e)
            self.assertTrue(torch.isfinite(result).all().item())
            assert_near(result[1], np.zeros(16), "bad environment fallback", atol=0.)
            self.assertTrue(torch.linalg.norm(result[0]).item() > 1e-9)
            degenerate = WholeBodyIK(s["kin"], IkWeights(damping=0.0))
            z = torch.zeros(2, 28, 16, device=self.device)
            dx_singular = degenerate.solve_dls(
                z, torch.ones(2, 28, device=self.device)
            )
            assert_near(dx_singular, np.zeros((2, 16)),
                        "singular solve fallback", atol=0.)

    def test_08_update_configuration_left_rotation_and_clamp(self):
        delta = torch.zeros(1, 16, device=self.device)
        delta[:, 0:3] = torch.tensor([[0.01, -0.015, 0.005]], device=self.device)
        delta[:, 3:6] = torch.tensor([[0.05, -0.08, 0.03]], device=self.device)
        delta[:, 6:16] = torch.tensor([[100.] * 5 + [-100.] * 5], device=self.device)
        for s in self.records:
            ik, kin = s["ik"], s["kin"]
            p, q = s["p0"].clone(), s["q0"].clone()
            R0 = ik.rotation_vector_to_matrix(torch.tensor(
                [[0.07, 0.02, -0.15]], device=self.device
            ))
            p_old, R_old, q_old = p.clone(), R0.clone(), q.clone()
            new_p, new_R, new_q = ik.update_configuration(p, R0, q, delta)
            assert_near(new_p, p + delta[:, :3], "update position", atol=1e-7)
            R_np = (so3_exp_np(delta[0, 3:6].detach().cpu().numpy())
                    @ R0[0].detach().cpu().numpy())
            assert_near(new_R[0], R_np, "left-world rotation update", atol=2e-6)
            assert_near(new_q, torch.clamp(q + delta[:, 6:], kin.qmin, kin.qmax),
                        "joint clamping", atol=1e-7)
            self.assertTrue(torch.all(new_q <= kin.qmax + 1e-7).item())
            self.assertTrue(torch.all(new_q >= kin.qmin - 1e-7).item())
            assert_near(p, p_old, "p input unchanged", atol=0.)
            assert_near(R0, R_old, "R input unchanged", atol=0.)
            assert_near(q, q_old, "q input unchanged", atol=0.)

    def test_09_solve_fixed_point_zero_reference(self):
        for s in self.records:
            p, R, q, ks = s["p0"], s["R0"], s["q0"], s["ks0"]
            solution = s["ik"].solve(
                p, R, q, ks.com, ks.site_pos, ks.site_rot, q,
                iterations=None, ks0=ks,
            )
            new_p, new_R, new_q, new_ks = solution
            assert_near(new_p, p, "fixed point base position", atol=2e-6)
            assert_near(new_R, R, "fixed point base orientation", atol=2e-6)
            assert_near(new_q, q, "fixed point 10 joints", atol=2e-6)
            assert_near(new_ks.com, ks.com, "fixed point CoM", atol=2e-6)
            assert_near(new_ks.site_pos, ks.site_pos, "fixed point foot pos", atol=2e-6)
            assert_near(new_ks.site_rot, ks.site_rot, "fixed point foot rot", atol=2e-6)

    def test_10_small_reachable_target_reduces_weighted_error(self):
        """Reference is exactly a real FK configuration, so it is reachable."""
        for s in self.records:
            ik, kin = s["ik"], s["kin"]
            p_des, R_des, q_des = s["p0"], s["R0"], s["q0"]
            target = s["ks0"]
            p = p_des + torch.tensor([[0.008, -0.004, 0.003]], device=self.device)
            R = ik.rotation_vector_to_matrix(torch.tensor(
                [[0.015, -0.02, 0.01]], device=self.device
            )) @ R_des
            q = torch.clamp(
                q_des + torch.tensor([[0.0, 0.012, -0.018, 0.023, -0.009,
                                       0.0, -0.014, 0.016, -0.022, 0.01]],
                                     device=self.device),
                kin.qmin, kin.qmax
            )
            before = kin.forward_kinematics(p, R, q)
            _, e0 = ik.build_tasks(before, p, R, q,
                                   target.com, target.site_pos,
                                   target.site_rot, q_des)
            cost0 = weighted_cost(ik, e0).item()
            self.assertGreater(cost0, 1e-7)
            p1, R1, q1, state1 = ik.solve(
                p, R, q, target.com, target.site_pos, target.site_rot,
                q_des, iterations=8,
            )
            _, e1 = ik.build_tasks(state1, p1, R1, q1,
                                   target.com, target.site_pos,
                                   target.site_rot, q_des)
            cost1 = weighted_cost(ik, e1).item()
            print(f'\n  [{s["name"]}] reachable-goal WDLS cost: '
                  f'{cost0:.6g} -> {cost1:.6g}', flush=True)
            self.assertTrue(np.isfinite(cost1))
            self.assertLess(
                cost1, 0.80 * cost0,
                f'{s["name"]}: reachable reference did not converge sufficiently',
            )
            self.assertTrue(torch.all(q1 <= kin.qmax + 1e-6).item())
            self.assertTrue(torch.all(q1 >= kin.qmin - 1e-6).item())
            assert_near(state1.com,
                        kin.forward_kinematics(p1, R1, q1).com,
                        "final KinState matches final q", atol=2e-6)

    def test_11_warm_start_matches_two_iterations(self):
        for s in self.records:
            ik, kin = s["ik"], s["kin"]
            p, R, q = s["p0"], s["R0"], s["q0"]
            target = s["ks0"]
            com_des = target.com + torch.tensor(
                [[0.004, 0.002, 0.0]], device=self.device
            )
            p1, R1, q1, ks1 = ik.solve(
                p, R, q, com_des, target.site_pos, target.site_rot, q,
                iterations=1, ks0=target,
            )
            p2, R2, q2, ks2 = ik.solve(
                p1, R1, q1, com_des, target.site_pos, target.site_rot, q,
                iterations=1, ks0=ks1,
            )
            p_ref, R_ref, q_ref, ks_ref = ik.solve(
                p, R, q, com_des, target.site_pos, target.site_rot, q,
                iterations=2,
            )
            for label, actual, expected in (
                ("p", p2, p_ref), ("R", R2, R_ref),
                ("q", q2, q_ref), ("com", ks2.com, ks_ref.com),
                ("foot", ks2.site_pos, ks_ref.site_pos),
            ):
                assert_near(actual, expected, f'{s["name"]}:warmstart:{label}',
                            atol=2e-5)
            p_no, R_no, q_no, _ = ik.solve(
                p1, R1, q1, com_des, target.site_pos, target.site_rot, q,
                iterations=1, ks0=None,
            )
            assert_near(p_no, p2, "warmstart vs freshly computed FK p", atol=2e-5)
            assert_near(R_no, R2, "warmstart vs freshly computed FK R", atol=2e-5)
            assert_near(q_no, q2, "warmstart vs freshly computed FK q", atol=2e-5)

    def test_12_batch_vs_single_solve(self):
        rng = np.random.default_rng(SEED + 11)
        for s in self.records:
            ik, kin = s["ik"], s["kin"]
            count = 2 + self.options.samples
            q_lo = kin.qmin.detach().cpu().numpy()
            q_hi = kin.qmax.detach().cpu().numpy()
            q0 = s["q0"].detach().cpu().numpy()[0]
            jitter = rng.uniform(-0.025, 0.025, size=(count, 10))
            q_np = np.clip(q0 + jitter, q_lo, q_hi)
            p = s["p0"].expand(count, 3).clone() + torch.tensor(
                rng.uniform(-0.01, 0.01, size=(count, 3)),
                device=self.device, dtype=torch.float32
            )
            q = torch.tensor(q_np, device=self.device, dtype=torch.float32)
            R = ik.rotation_vector_to_matrix(torch.tensor(
                rng.uniform(-0.025, 0.025, size=(count, 3)),
                device=self.device, dtype=torch.float32
            ))
            com = s["ks0"].com.expand(count, 3).clone()
            feet = s["ks0"].site_pos.expand(count, 2, 3).clone()
            feet_rot = s["ks0"].site_rot.expand(count, 2, 3, 3).clone()
            q_home = s["q0"].expand(count, 10).clone()
            result = ik.solve(p, R, q, com, feet, feet_rot, q_home, iterations=2)
            self.assertEqual(result[0].shape, (count, 3))
            self.assertEqual(result[1].shape, (count, 3, 3))
            self.assertEqual(result[2].shape, (count, 10))
            self.assertEqual(result[3].com.shape, (count, 3))
            for i in range(count):
                one = ik.solve(
                    p[i:i + 1], R[i:i + 1], q[i:i + 1],
                    com[i:i + 1], feet[i:i + 1], feet_rot[i:i + 1],
                    q_home[i:i + 1], iterations=2,
                )
                for label, actual, single in (
                    ("p", result[0][i], one[0][0]),
                    ("R", result[1][i], one[1][0]),
                    ("q", result[2][i], one[2][0]),
                    ("com", result[3].com[i], one[3].com[0]),
                    ("feet", result[3].site_pos[i], one[3].site_pos[0]),
                ):
                    assert_near(actual, single,
                                f'{s["name"]}:env={i}:batch {label}', atol=3e-5)

    def test_13_final_solution_fk_matches_mujoco_v5(self):
        for s in self.records:
            ik, model = s["ik"], s["model"]
            p, R, q = s["p0"], s["R0"], s["q0"]
            target = s["ks0"]
            com_des = target.com + torch.tensor(
                [[0.003, -0.002, 0.0]], device=self.device
            )
            p1, R1, q1, ks1 = ik.solve(
                p, R, q, com_des, target.site_pos, target.site_rot, q,
                iterations=5,
            )
            data = model_data_from_pose(
                model, p1[0].detach().cpu().numpy(),
                R1[0].detach().cpu().numpy(),
                q1[0].detach().cpu().numpy(),
            )
            com_mj = np.asarray(data.subtree_com[model.body("base_link").id]).copy()
            feet_mj = np.stack([
                np.asarray(data.site(name).xpos).copy()
                for name in ("L6_IK", "R6_IK")
            ])
            feet_R_mj = np.stack([
                np.asarray(data.site(name).xmat).copy().reshape(3, 3)
                for name in ("L6_IK", "R6_IK")
            ])
            assert_near(ks1.com[0], com_mj,
                        f'{s["name"]}:final MuJoCo CoM', atol=8e-5)
            assert_near(ks1.site_pos[0], feet_mj,
                        f'{s["name"]}:final MuJoCo feet', atol=8e-5)
            assert_near(ks1.site_rot[0], feet_R_mj,
                        f'{s["name"]}:final MuJoCo foot orientation', atol=8e-5)

    def test_14_reject_invalid_iteration_counts(self):
        for s in self.records:
            ik, ks = s["ik"], s["ks0"]
            args = (s["p0"], s["R0"], s["q0"],
                    ks.com, ks.site_pos, ks.site_rot, s["q0"])
            for number in (0, -1, 2.2, "3"):
                with self.subTest(xml=s["name"], iterations=number):
                    with self.assertRaises(ValueError):
                        ik.solve(*args, iterations=number)

    def test_15_optional_autograd_at_exact_zero_rotvec(self):
        if not self.options.check_grad:
            self.skipTest("Optional: run with --check-grad to probe zero-angle autograd")
        for s in self.records:
            v = torch.zeros(1, 3, device=self.device, requires_grad=True)
            R = s["ik"].rotation_vector_to_matrix(v)
            loss = (R * torch.tensor(
                [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                dtype=R.dtype, device=self.device
            )).sum()
            grad = torch.autograd.grad(loss, v)[0]
            self.assertTrue(torch.isfinite(grad).all().item(),
                            "Exact zero rotvec produces NaN/Inf gradient. "
                            "If training requires gradients through WBIK, "
                            "Rodrigues needs a safe zero-angle derivative.")
            assert_near(R[0], np.eye(3), "zero angle rotation", atol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--samples", type=int, default=3,
                        help="Extra random batch environments for comparison (default: 3)")
    parser.add_argument("--model-only", action="store_true",
                        help="Test only open_duck_mini_v5.xml, skip scene")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-grad", action="store_true",
                        help="Also check autograd of SO(3) Rodrigues at exactly zero")
    opt = parser.parse_args()
    if opt.samples < 1:
        parser.error("--samples must be >= 1")
    if opt.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    torch.manual_seed(SEED)
    TestWholeBodyIK.options = opt
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestWholeBodyIK)
    result = unittest.TextTestRunner(verbosity=2 if opt.verbose else 1).run(suite)
    if result.wasSuccessful():
        print(f"\n[PASS] WholeBodyIK integration: "
              f"{2 if not opt.model_only else 1} MJCF model(s), "
              f"batch={2 + opt.samples}, device={opt.device}.")
    else:
        print("\n[FAIL] See the failing test group and its numerical error above.")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
