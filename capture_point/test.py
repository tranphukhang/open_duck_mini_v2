"""
Kiem thu RobotKinematics.compute_com_jacobian() - Open Duck Mini V2 (MJCF v5).

Dat file nay tai capture_point/ (cung cap voi robot_kinematics.py).
Chay tu thu muc goc repo:

    python capture_point/test_robot_kinematics_com_jacobian.py --verbose
    python capture_point/test_robot_kinematics_com_jacobian.py --samples 20 --verbose
    python capture_point/test_robot_kinematics_com_jacobian.py --model-only
    python capture_point/test_robot_kinematics_com_jacobian.py --device cuda

Yeu cau: pip install mujoco numpy torch
Mo hinh: xmls/open_duck_mini_v5.xml, xmls/scene_flat_terrian_v5.xml
         va thu muc mesh xmls/assets_v3/.

QUY UOC 16 COT:
  0:3   = Tinh tien base trong he world/{P}
  3:6   = Quay base gia so nho trong he world/{P} (left perturbation)
  6:16  = 10 khop quay theo thu tu JOINT_NAMES.

Quan trong: 3 cot quay base KHONG so sanh truc tiep voi qvel free joint
MuJoCo vi co the khac quy uoc he truc. Chung duoc kiem tra bang
sai phan huu han theo quaternion quay ben trai trong he world/{P}.
Cac cot khop duoc so sanh voi mujoco.mj_jacSubtreeCom().
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
    raise SystemExit("Thieu mujoco. Cai bang: pip install mujoco") from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robot_kinematics import RobotKinematics

JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)

ATOL_MJ = 5e-5
ATOL_FD = 8e-5
ATOL_GRAD = 6e-5
FD_EPS = 1e-6
SEED = 20261114


def find_xml_dir() -> Path:
    for folder in (HERE, *HERE.parents):
        candidate = folder / "xmls" / "open_duck_mini_v5.xml"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError(
        "Khong tim thay xmls/open_duck_mini_v5.xml. "
        "Hay dat file test trong <repo>/capture_point/."
    )


def as_np(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def assert_close(actual, expected, label, atol=ATOL_MJ):
    a = as_np(actual)
    b = as_np(expected)
    if a.shape != b.shape:
        raise AssertionError(
            f"{label}: sai kich thuoc actual={a.shape}, expected={b.shape}"
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{label}: chua NaN/Inf")
    try:
        np.testing.assert_allclose(a, b, rtol=0.0, atol=atol)
    except AssertionError as exc:
        error = float(np.max(np.abs(a - b)))
        raise AssertionError(
            f"{label}: max_error={error:.6e} (tol={atol:.2e})\n{exc}"
        ) from exc


def quat_to_mat(quat):
    out = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=np.float64))
    return out.reshape(3, 3)


def quat_mul(a, b):
    """Hamilton product quaternion wxyz."""
    w1, w2 = float(a[0]), float(b[0])
    v1 = np.asarray(a[1:4], dtype=np.float64)
    v2 = np.asarray(b[1:4], dtype=np.float64)
    return np.concatenate((
        [w1 * w2 - float(np.dot(v1, v2))],
        w1 * v2 + w2 * v1 + np.cross(v1, v2),
    ))


def perturb_quat_world(quat, axis, delta):
    """R_new = Exp(delta*[axis]_x) @ R; quaternion left multiplication."""
    v = np.zeros(3, dtype=np.float64)
    v[axis] = np.sin(delta / 2.0)
    dq = np.concatenate(([np.cos(delta / 2.0)], v))
    result = quat_mul(dq, np.asarray(quat, dtype=np.float64))
    return result / np.linalg.norm(result)


def make_cases(model, samples):
    joint_ids = [model.joint(n).id for n in JOINT_NAMES]
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    home_q = np.deg2rad(
        [0, -10, 20, -40, 20, 0, 10, -20, 40, -20]
    )
    home_pos = np.array([0.0, 0.0, 0.275], dtype=np.float64)
    home_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    for key_id in range(model.nkey):
        if model.key(key_id).name == "home":
            key_qpos = np.asarray(model.key_qpos[key_id], dtype=np.float64)
            home_pos = key_qpos[free_adr:free_adr + 3].copy()
            home_quat = key_qpos[free_adr + 3:free_adr + 7].copy()
            home_q = np.asarray([
                key_qpos[int(model.jnt_qposadr[j])]
                for j in joint_ids
            ], dtype=np.float64)
            break

    cases = [
        ("home", home_pos, home_quat, home_q),
        ("zero", np.array([0.04, -0.03, 0.30], dtype=np.float64),
         np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
         np.zeros(10, dtype=np.float64)),
    ]

    rng = np.random.default_rng(SEED)
    for i in range(samples):
        quat = rng.normal(size=4)
        quat = quat / np.linalg.norm(quat)
        q = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * rng.uniform(
            0.15, 0.85, size=10
        )
        pos = rng.uniform([-0.24, -0.22, 0.22], [0.24, 0.22, 0.42])
        cases.append((f"random_{i:02d}", pos, quat, q))
    return cases


def tensor_inputs(cases, device):
    p = torch.tensor(
        np.stack([c[1] for c in cases]),
        device=device, dtype=torch.float32,
    )
    r = torch.tensor(
        np.stack([quat_to_mat(c[2]) for c in cases]),
        device=device, dtype=torch.float32,
    )
    q = torch.tensor(
        np.stack([c[3] for c in cases]),
        device=device, dtype=torch.float32,
    )
    return p, r, q


def make_mj_data(model, case):
    """Set qpos -> run independent MuJoCo forward kinematics."""
    _, base_pos, base_quat, q = case
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    free_adr = int(model.jnt_qposadr[model.joint("floating_base").id])
    data.qpos[free_adr:free_adr + 3] = base_pos
    data.qpos[free_adr + 3:free_adr + 7] = base_quat
    for j, name in enumerate(JOINT_NAMES):
        jid = model.joint(name).id
        data.qpos[int(model.jnt_qposadr[jid])] = q[j]
    mujoco.mj_forward(model, data)
    return data


def subtree_jacobian(model, data):
    jac = np.zeros((3, model.nv), dtype=np.float64)
    base_id = model.body("base_link").id
    mujoco.mj_jacSubtreeCom(model, data, jac, base_id)
    return jac


def subtree_com(model, case):
    data = make_mj_data(model, case)
    return np.asarray(
        data.subtree_com[model.body("base_link").id], dtype=np.float64
    ).copy()


def finite_difference_column(model, case, col, eps=FD_EPS):
    """
    Central FD of MuJoCo subtree COM.
    d(base_quat) = quat_exp_world(dtheta) * base_quat for cols 3..5.
    """
    label, pos, quat, q = case
    pp, pm = pos.copy(), pos.copy()
    qp, qm = q.copy(), q.copy()
    rp, rm = quat.copy(), quat.copy()

    if col < 3:
        pp[col] += eps
        pm[col] -= eps
    elif col < 6:
        rp = perturb_quat_world(quat, col - 3, +eps)
        rm = perturb_quat_world(quat, col - 3, -eps)
    else:
        qp[col - 6] += eps
        qm[col - 6] -= eps

    plus = (label + "/plus", pp, rp, qp)
    minus = (label + "/minus", pm, rm, qm)
    return (subtree_com(model, plus) - subtree_com(model, minus)) / (2 * eps)


def skew_from_vec(v):
    """Numpy skew matrix for independent analytic check."""
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


class TestComJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        opt = cls.options
        cls.device = torch.device(opt.device)
        names = ["open_duck_mini_v5.xml"]
        if not opt.model_only:
            names.append("scene_flat_terrian_v5.xml")

        cls.records = []
        xmls = find_xml_dir()
        for name in names:
            path = xmls / name
            model = mujoco.MjModel.from_xml_path(str(path))
            kin = RobotKinematics(model, device=opt.device)
            cases = make_cases(model, opt.samples)
            p, r, q = tensor_inputs(cases, cls.device)
            ks = kin.forward_kinematics(p, r, q)
            J = kin.compute_com_jacobian(ks, p)
            cls.records.append({
                "name": name, "model": model, "kin": kin,
                "cases": cases, "pos": p, "rot": r, "q": q,
                "ks": ks, "J": J,
            })

    def test_01_shape_dtype_device_and_finite(self):
        for s in self.records:
            with self.subTest(xml=s["name"]):
                J = s["J"]
                self.assertEqual(J.shape, (len(s["cases"]), 3, 16))
                self.assertEqual(J.dtype, torch.float32)
                self.assertEqual(J.device, self.device)
                self.assertTrue(torch.isfinite(J).all().item())
                self.assertEqual(s["ks"].com.shape, (len(s["cases"]), 3))

    def test_02_com_position_matches_mujoco(self):
        """Prerequisite: FK COM and MuJoCo are aligned in the same frame."""
        for s in self.records:
            for i, case in enumerate(s["cases"]):
                expected = subtree_com(s["model"], case)
                assert_close(
                    s["ks"].com[i], expected,
                    f'{s["name"]}:{case[0]}:CoM', atol=ATOL_MJ,
                )

    def test_03_three_base_translation_columns_identity(self):
        for s in self.records:
            J = s["J"]
            I = torch.eye(3, device=self.device).unsqueeze(0).expand(
                len(s["cases"]), 3, 3
            )
            assert_close(
                J[:, :, :3], I,
                f'{s["name"]}:translation identity', atol=1e-7,
            )

    def test_04_base_rotation_skew_formula(self):
        """3 world-frame infinitesimal rotation columns = -skew(com-base)."""
        for s in self.records:
            for i, case in enumerate(s["cases"]):
                r = as_np(s["ks"].com[i] - s["pos"][i])
                expected = -skew_from_vec(r)
                assert_close(
                    s["J"][i, :, 3:6], expected,
                    f'{s["name"]}:{case[0]}:rotation skew', atol=1e-7,
                )

    def test_05_hinge_columns_vs_mujoco_subtree_com_jacobian(self):
        """Compare the 10 joint columns using MuJoCo DOF IDs."""
        for s in self.records:
            model = s["model"]
            hinge_dofs = [
                int(model.jnt_dofadr[model.joint(n).id]) for n in JOINT_NAMES
            ]
            for i, case in enumerate(s["cases"]):
                data = make_mj_data(model, case)
                oracle = subtree_jacobian(model, data)
                assert_close(
                    s["J"][i, :, 6:16], oracle[:, hinge_dofs],
                    f'{s["name"]}:{case[0]}:MuJoCo hinge columns',
                    atol=ATOL_MJ,
                )

    def test_06_mass_weighted_body_jacobians_vs_subtree_jacobian(self):
        """Independent MuJoCo oracle: sum of body COM Jacobians by mass."""
        for s in self.records:
            model, kin = s["model"], s["kin"]
            for case in s["cases"][:min(3, len(s["cases"]))]:
                data = make_mj_data(model, case)
                result = np.zeros((3, model.nv), dtype=np.float64)
                masses = np.asarray(model.body_mass[kin.bodies], dtype=np.float64)
                for body_id, mass in zip(kin.bodies, masses):
                    jp = np.zeros((3, model.nv), dtype=np.float64)
                    jr = np.zeros((3, model.nv), dtype=np.float64)
                    mujoco.mj_jacBodyCom(model, data, jp, jr, body_id)
                    result += mass * jp
                result /= masses.sum()
                oracle = subtree_jacobian(model, data)
                assert_close(
                    result, oracle,
                    f'{s["name"]}:{case[0]}:MuJoCo weighted body Jacobian',
                    atol=1e-9,
                )

    def test_07_all_16_columns_vs_mujoco_finite_difference(self):
        """World-frame base rotations via left quaternion perturbation."""
        for s in self.records:
            model = s["model"]
            # Home + zero + first two random configurations (if supplied).
            for i in range(min(len(s["cases"]), 4)):
                case = s["cases"][i]
                fd = np.stack([
                    finite_difference_column(model, case, col)
                    for col in range(16)
                ], axis=1)
                assert_close(
                    s["J"][i], fd,
                    f'{s["name"]}:{case[0]}:FD 16 columns',
                    atol=ATOL_FD,
                )

    def test_08_torch_autograd_joint_and_base_translation(self):
        """Autograd of CoM FK is independent of analytic Jacobian routine."""
        for s in self.records:
            pos = s["pos"][:2].detach().clone().requires_grad_(True)
            rot = s["rot"][:2].detach().clone()
            q = s["q"][:2].detach().clone().requires_grad_(True)
            pos_before = pos.detach().clone()
            q_before = q.detach().clone()

            kin = s["kin"]
            ks = kin.forward_kinematics(pos, rot, q)
            J = kin.compute_com_jacobian(ks, pos)
            for row in range(3):
                grad_p, grad_q = torch.autograd.grad(
                    ks.com[:, row].sum(), (pos, q),
                    retain_graph=True,
                )
                assert_close(
                    grad_p, J[:, row, :3],
                    f'{s["name"]}:autograd base translation row={row}',
                    atol=ATOL_GRAD,
                )
                assert_close(
                    grad_q, J[:, row, 6:16],
                    f'{s["name"]}:autograd joints row={row}',
                    atol=ATOL_GRAD,
                )
            assert_close(pos, pos_before, "base_pos mutated", atol=0)
            assert_close(q, q_before, "q mutated", atol=0)

    def test_09_torch_autograd_world_base_rotation(self):
        """Differentiate left-multiplied SO(3) rotations at zero angle."""
        generators = torch.tensor([
            [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],  # X
            [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],  # Y
            [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],  # Z
        ], dtype=torch.float32, device=self.device)
        # Generator above: X has [0,0,0; 0,0,-1; 0,1,0].
        for s in self.records:
            theta = torch.zeros(
                3, device=self.device, dtype=torch.float32, requires_grad=True
            )
            R_delta = torch.eye(3, device=self.device, dtype=torch.float32)
            I = torch.eye(3, device=self.device, dtype=torch.float32)
            for a in range(3):
                K = generators[a]
                R_axis = I + torch.sin(theta[a]) * K + (
                    1.0 - torch.cos(theta[a])
                ) * (K @ K)
                R_delta = R_axis @ R_delta

            rotated_base = R_delta.unsqueeze(0) @ s["rot"][:1]
            state = s["kin"].forward_kinematics(
                s["pos"][:1], rotated_base, s["q"][:1]
            )
            J = s["kin"].compute_com_jacobian(state, s["pos"][:1])
            for xyz in range(3):
                grad = torch.autograd.grad(
                    state.com[0, xyz], theta, retain_graph=True
                )[0]
                assert_close(
                    grad, J[0, xyz, 3:6],
                    f'{s["name"]}:autograd world base rotation xyz={xyz}',
                    atol=ATOL_GRAD,
                )

    def test_10_batched_vs_individual_environments(self):
        for s in self.records:
            kin = s["kin"]
            for i in range(len(s["cases"])):
                p = s["pos"][i:i + 1]
                rot = s["rot"][i:i + 1]
                q = s["q"][i:i + 1]
                one = kin.forward_kinematics(p, rot, q)
                J_one = kin.compute_com_jacobian(one, p)
                assert_close(
                    J_one[0], s["J"][i],
                    f'{s["name"]}:{s["cases"][i][0]}:single vs batch',
                    atol=1e-7,
                )

    def test_11_world_frame_translation_and_rotation_covariance(self):
        """J_rot' = U J_rot U^T; J_joint' = U J_joint."""
        angle = 0.48
        c, sn = np.cos(angle), np.sin(angle)
        U_np = np.array([
            [c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        U = torch.tensor(U_np, device=self.device, dtype=torch.float32)
        t = torch.tensor(
            [0.11, -0.09, 0.04], device=self.device, dtype=torch.float32
        )
        for s in self.records:
            kin = s["kin"]
            p2 = (U @ s["pos"].T).T + t
            r2 = U.unsqueeze(0) @ s["rot"]
            state = kin.forward_kinematics(p2, r2, s["q"])
            J2 = kin.compute_com_jacobian(state, p2)
            expected_com = (U @ s["ks"].com.T).T + t
            assert_close(
                state.com, expected_com,
                f'{s["name"]}:CoM under SE3 transform',
                atol=ATOL_MJ,
            )
            expected_rot = U @ s["J"][:, :, 3:6] @ U.T
            expected_joint = U @ s["J"][:, :, 6:16]
            assert_close(
                J2[:, :, 3:6], expected_rot,
                f'{s["name"]}:rotation covariance',
                atol=ATOL_MJ,
            )
            assert_close(
                J2[:, :, 6:16], expected_joint,
                f'{s["name"]}:joint covariance',
                atol=ATOL_MJ,
            )
            I = torch.eye(3, device=self.device).expand(len(s["cases"]), 3, 3)
            assert_close(
                J2[:, :, :3], I,
                f'{s["name"]}:translation Jacobian unchanged',
                atol=1e-7,
            )

    def test_12_reconstruction_from_all_body_point_jacobians(self):
        """CoM J equals mass-weighted point Jacobians of body COMs."""
        for s in self.records:
            kin = s["kin"]
            weighted = torch.zeros_like(s["J"])
            for body_index in range(len(kin.bodies)):
                J_body = kin.compute_point_jacobian(
                    s["ks"], s["ks"].com_body[:, body_index],
                    kin.desc[:, body_index], s["pos"],
                )
                weighted = weighted + kin.mass[body_index] * J_body
            weighted = weighted / kin.total_mass
            assert_close(
                s["J"], weighted,
                f'{s["name"]}:weighted 11 point Jacobians',
                atol=2e-6,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="PyTorch execution device (default: cpu)",
    )
    parser.add_argument(
        "--samples", type=int, default=6,
        help="Number of deterministic random configurations per model (default: 6)",
    )
    parser.add_argument(
        "--model-only", action="store_true",
        help="Only test open_duck_mini_v5.xml (skip scene_flat_terrian_v5.xml)",
    )
    parser.add_argument("--verbose", action="store_true")
    options = parser.parse_args()
    if options.samples < 1:
        parser.error("--samples must be >= 1")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    TestComJacobian.options = options
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(TestComJacobian)
    result = unittest.TextTestRunner(
        verbosity=2 if options.verbose else 1
    ).run(tests)

    if result.wasSuccessful():
        print(
            "\n[PASS] compute_com_jacobian(): 12 test groups; "
            f"{1 if options.model_only else 2} model(s); "
            f"{options.samples + 2} configurations/model; "
            f"device={options.device}."
        )
        print(
            "Checked: MuJoCo mj_jacSubtreeCom, mass-weighted body Jacobians, "
            "16-column finite differences, PyTorch autograd, batch, SE(3)."
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
