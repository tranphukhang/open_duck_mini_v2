import numpy as np
import mujoco
from pathlib import Path

# ============================================================
# 1. CAU HINH
# ============================================================
ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "xmls" / "open_duck_mini_v5.xml"
if not XML_PATH.exists():
    XML_PATH = ROOT / "open_duck_mini_v5.xml"

N_TESTS = 107
TOL = 1e-8
rng = np.random.default_rng(2026)

# Thong so MDH, don vi m
l4 = 0.01606
l5 = 0.05570
l6 = 0.07865
l7 = 0.07865
l8 = 0.03880

JOINT_NAMES = {
    "L": ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle"],
    "R": ["R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"],
}
SITE_NAMES = {"L": "L6", "R": "R6"}

# ============================================================
# 2. JACOBIAN TINH TIEN (GIAI TICH MDH) - GIU NGUYEN
# ============================================================
def trig_terms(q):
    q1, q2, q3, q4, q5 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    q34 = q3 + q4
    q345 = q3 + q4 + q5

    C3 = l6*np.cos(q3) + l7*np.cos(q34) + l8*np.cos(q345)
    S3 = l6*np.sin(q3) + l7*np.sin(q34) + l8*np.sin(q345)
    C4 = l7*np.cos(q34) + l8*np.cos(q345)
    S4 = l7*np.sin(q34) + l8*np.sin(q345)
    C5 = l8*np.cos(q345)
    S5 = l8*np.sin(q345)
    return c1, s1, c2, s2, C3, S3, C4, S4, C5, S5


def jacobian_left_mdh(q):
    """^B J_v,L, kich thuoc 3x5 [m/rad]."""
    c1, s1, c2, s2, C3, S3, C4, S4, C5, S5 = trig_terms(q)
    return np.array([
        [l4*s1 - l5*c1*c2 + s1*S3 - c1*s2*C3,
         s1*(l5*s2 - c2*C3),
         -c1*C3 + s1*s2*S3,
         -c1*C4 + s1*s2*S4,
         -c1*C5 + s1*s2*S5],
        [-l4*c1 - l5*s1*c2 - c1*S3 - s1*s2*C3,
         -c1*(l5*s2 - c2*C3),
         -s1*C3 - c1*s2*S3,
         -s1*C4 - c1*s2*S4,
         -s1*C5 - c1*s2*S5],
        [0.0,
         l5*c2 + s2*C3,
         c2*S3, c2*S4, c2*S5],
    ], dtype=float)


def jacobian_right_mdh(q):
    """^B J_v,R, kich thuoc 3x5 [m/rad]."""
    c1, s1, c2, s2, C3, S3, C4, S4, C5, S5 = trig_terms(q)
    return np.array([
        [l4*s1 + l5*c1*c2 - s1*S3 - c1*s2*C3,
         -s1*(l5*s2 + c2*C3),
         c1*C3 + s1*s2*S3,
         c1*C4 + s1*s2*S4,
         c1*C5 + s1*s2*S5],
        [-l4*c1 + l5*s1*c2 + c1*S3 - s1*s2*C3,
         c1*(l5*s2 + c2*C3),
         s1*C3 - c1*s2*S3,
         s1*C4 - c1*s2*S4,
         s1*C5 - c1*s2*S5],
        [0.0,
         -l5*c2 + s2*C3,
         c2*S3, c2*S4, c2*S5],
    ], dtype=float)

# ============================================================
# 3. JACOBIAN QUAY (GIAI TICH MDH) - MOI BO SUNG
# ============================================================
def jacobian_angular_left_mdh(q):
    """^B J_omega,L, kich thuoc 3x5."""
    q1, q2 = q[0], q[1]
    return np.array([
        [0.0, np.cos(q1), -np.cos(q2)*np.sin(q1),
         -np.cos(q2)*np.sin(q1), -np.cos(q2)*np.sin(q1)],
        [0.0, np.sin(q1), np.cos(q1)*np.cos(q2),
         np.cos(q1)*np.cos(q2), np.cos(q1)*np.cos(q2)],
        [1.0, 0.0, np.sin(q2), np.sin(q2), np.sin(q2)],
    ], dtype=float)


def jacobian_angular_right_mdh(q):
    """^B J_omega,R, kich thuoc 3x5."""
    q1, q2 = q[0], q[1]
    return np.array([
        [0.0, np.cos(q1), np.cos(q2)*np.sin(q1),
         np.cos(q2)*np.sin(q1), np.cos(q2)*np.sin(q1)],
        [0.0, np.sin(q1), -np.cos(q1)*np.cos(q2),
         -np.cos(q1)*np.cos(q2), -np.cos(q1)*np.cos(q2)],
        [1.0, 0.0, -np.sin(q2), -np.sin(q2), -np.sin(q2)],
    ], dtype=float)

# ============================================================
# 4. QUATERNION EULER ZYX -> MUJOCO [w, x, y, z]
# ============================================================
def rpy_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


# ============================================================
# 4A. JACOBIAN FLOATING-BASE CHAN TRAI VA CHAN PHAI
# ============================================================
# Giu nguyen 4 Jacobian MDH 3x5 o muc 2 va 3.
# Su dung truc tiep 9 bieu thuc rut gon W_J_EL va W_J_ER (1:3,4:6)
# tu MATLAB. Khong tinh lai khoi nay bang tich -skew(r)*E.
# Cot 1:3 = tinh tien base, 4:6 = Euler [phi,theta,psi].
# Cot 7:11 = khop trai, 12:16 = khop phai.

l2 = 0.08286
l3 = 0.03500


def foot_position_left_B(q):
    """^B p_E_L (3,) tu dong hoc thuan MDH da xac dinh."""
    c1, s1, c2, s2, C3, S3, *_ = trig_terms(q)
    return np.array([
        -l4*c1 - l5*s1*c2 - c1*S3 - s1*s2*C3,
        l3 - l4*s1 + l5*c1*c2 - s1*S3 + c1*s2*C3,
        -l2 + l5*s2 - c2*C3,
    ], dtype=float)


def foot_position_right_B(q):
    """^B p_E_R (3,) tu dong hoc thuan MDH chan phai da xac dinh."""
    c1, s1, c2, s2, C3, S3, *_ = trig_terms(q)
    return np.array([
        -l4*c1 + l5*s1*c2 + c1*S3 - s1*s2*C3,
        -l3 - l4*s1 - l5*c1*c2 + s1*S3 + c1*s2*C3,
        -l2 - l5*s2 - c2*C3,
    ], dtype=float)


def rotation_zyx(rpy):
    """^W R_B = Rz(psi) Ry(theta) Rx(phi)."""
    phi, theta, psi = rpy
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy*ct, cy*st*sp - sy*cp, cy*st*cp + sy*sp],
        [sy*ct, sy*st*sp + cy*cp, sy*st*cp - cy*sp],
        [-st,   ct*sp,              ct*cp],
    ], dtype=float)


def euler_E_world(rpy):
    """Phuong trinh (1.12): ^W omega_B = E * [phi_dot,theta_dot,psi_dot]."""
    _, theta, psi = rpy
    return np.array([
        [np.cos(psi)*np.cos(theta), -np.sin(psi), 0.0],
        [np.sin(psi)*np.cos(theta),  np.cos(psi), 0.0],
        [-np.sin(theta),            0.0,         1.0],
    ], dtype=float)


def jacobian_floating_left(q, rpy, Jv_B, Jw_B):
    """^W J_E_L (6x16) theo cong thuc (1.10), dung W_J_EL da rut gon.

    W_J_EL[:, 0:3] la cac cot tinh tien than [x_B, y_B, z_B].
    W_J_EL[:, 3:6] la cac cot [phi, theta, psi] (Euler ZYX).
    W_J_EL[:, 6:11] la 5 cot khop trai da tinh o muc 2, 3.
    W_J_EL[:, 11:16] bang 0 vi ban chan trai khong phu thuoc khop phai.
    """
    px, py, pz = foot_position_left_B(q)
    phi, theta, psi = rpy
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)

    # CHINH LA 9 PHAN TU W_J_EL(1:3, 4:6) DA RUT GON:
    # Cac dong tuong ung van toc x_W, y_W, z_W.
    # Cac cot tuong ung phi_dot, theta_dot, psi_dot.
    W_J_EL_146 = np.array([
        [
            (cy*st*cf + sy*sf)*py + (sy*cf - cy*st*sf)*pz,  # (1,4)
            cy*(-st*px + ct*sf*py + ct*cf*pz),               # (1,5)
            -sy*ct*px - (cy*cf + sy*st*sf)*py
            + (cy*sf - sy*st*cf)*pz,                         # (1,6)
        ],
        [
            (-cy*sf + sy*st*cf)*py
            - (cy*cf + sy*st*sf)*pz,                         # (2,4)
            sy*(-st*px + ct*sf*py + ct*cf*pz),               # (2,5)
            cy*ct*px + (cy*st*sf - sy*cf)*py
            + (cy*st*cf + sy*sf)*pz,                         # (2,6)
        ],
        [
            ct*(cf*py - sf*pz),                             # (3,4)
            -ct*px - st*sf*py - st*cf*pz,                   # (3,5)
            0.0,                                            # (3,6)
        ],
    ], dtype=float)

    R = rotation_zyx(rpy)
    E = euler_E_world(rpy)
    W_J_EL = np.zeros((6, 16), dtype=float)
    W_J_EL[0:3, 0:3] = np.eye(3)
    W_J_EL[0:3, 3:6] = W_J_EL_146
    W_J_EL[3:6, 3:6] = E
    W_J_EL[0:3, 6:11] = R @ Jv_B
    W_J_EL[3:6, 6:11] = R @ Jw_B
    return W_J_EL


def jacobian_floating_right(q, rpy, Jv_B, Jw_B):
    """^W J_E_R (6x16): 9 bieu thuc W_J_ER da rut gon tu MATLAB.

    Giu nguyen ^B J_v,R va ^B J_omega,R da kiem chung truoc do.
    Cot 6:11 bang 0 vi chan phai khong phu thuoc khop chan trai.
    """
    px, py, pz = foot_position_right_B(q)
    phi, theta, psi = rpy
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)

    # 9 BIEU THUC RUT GON W_J_ER(1:3, 4:6) TU MATLAB:
    # Row: x_W, y_W, z_W; column: phi_dot, theta_dot, psi_dot.
    # Khong dung -skew(r)@E de tao khoi Jacobian nay.
    W_J_ER_146 = np.array([
        [
            (cy*st*cf + sy*sf)*py + (sy*cf - cy*st*sf)*pz,  # (1,4)
            cy*(-st*px + ct*sf*py + ct*cf*pz),              # (1,5)
            -sy*ct*px - (cy*cf + sy*st*sf)*py
            + (cy*sf - sy*st*cf)*pz,                        # (1,6)
        ],
        [
            (-cy*sf + sy*st*cf)*py
            - (cy*cf + sy*st*sf)*pz,                        # (2,4)
            sy*(-st*px + ct*sf*py + ct*cf*pz),              # (2,5)
            cy*ct*px + (cy*st*sf - sy*cf)*py
            + (cy*st*cf + sy*sf)*pz,                        # (2,6)
        ],
        [
            ct*(cf*py - sf*pz),                            # (3,4)
            -ct*px - st*sf*py - st*cf*pz,                  # (3,5)
            0.0,                                           # (3,6)
        ],
    ], dtype=float)

    R = rotation_zyx(rpy)
    E = euler_E_world(rpy)
    W_J_ER = np.zeros((6, 16), dtype=float)
    W_J_ER[0:3, 0:3] = np.eye(3)
    W_J_ER[0:3, 3:6] = W_J_ER_146
    W_J_ER[3:6, 3:6] = E
    W_J_ER[0:3, 11:16] = R @ Jv_B
    W_J_ER[3:6, 11:16] = R @ Jw_B
    return W_J_ER


def mujoco_velocity_map(model, qpos, rpy, free_qpos_adr, free_dof_adr,
                        leg_info, eps=1e-5):
    """qvel_MJ = T * [p_dot_W; Euler_ZYX_dot; qdot_L; qdot_R].

    Dung mj_differentiatePos de doi quy uoc Euler/quaternion o MUJOCO.
    Khong dung E() tu (1.12) de tao mau so tham chieu.
    """
    T = np.zeros((model.nv, 16), dtype=float)
    T[free_dof_adr:free_dof_adr+3, :3] = np.eye(3)
    for i in range(3):
        rpy_plus = rpy.copy()
        rpy_minus = rpy.copy()
        rpy_plus[i] += eps
        rpy_minus[i] -= eps
        qp = qpos.copy()
        qm = qpos.copy()
        qp[free_qpos_adr+3:free_qpos_adr+7] = rpy_to_quat(*rpy_plus)
        qm[free_qpos_adr+3:free_qpos_adr+7] = rpy_to_quat(*rpy_minus)
        vp = np.zeros(model.nv)
        vm = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, vp, eps, qpos, qp)
        mujoco.mj_differentiatePos(model, vm, eps, qm, qpos)
        T[:, 3+i] = (vp + vm) / 2.0
    for side, offset in (("L", 6), ("R", 11)):
        for j, dof in enumerate(leg_info[side]["dof_ids"]):
            T[dof, offset+j] = 1.0
    return T


FLOAT_BLOCKS = {
    "L": (
        ("base_xyz", slice(0, 3)),
        ("base_euler", slice(3, 6)),
        ("left_joints", slice(6, 11)),
        ("right_joints_zero", slice(11, 16)),
    ),
    "R": (
        ("base_xyz", slice(0, 3)),
        ("base_euler", slice(3, 6)),
        ("left_joints_zero", slice(6, 11)),
        ("right_joints", slice(11, 16)),
    ),
}


def new_result(shape):
    return {"passed": 0, "max_error": 0.0, "worst_case": 0,
            "element_errors": np.zeros(shape)}

# ============================================================
# 5. NAP MUJOCO VA XAC DINH CAC COT KHOP
# ============================================================
if not XML_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay XML: {XML_PATH}")
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)


def get_id(obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Khong tim thay: {name}")
    return idx

base_id = get_id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
free_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
free_qpos_adr = int(model.jnt_qposadr[free_id])
free_dof_adr = int(model.jnt_dofadr[free_id])
assert model.jnt_type[free_id] == mujoco.mjtJoint.mjJNT_FREE
if model.nv != 16:
    raise ValueError(f"Mo hinh can nv=16, nhung dang la {model.nv}")

leg_info = {}
for side in ("L", "R"):
    site_id = get_id(mujoco.mjtObj.mjOBJ_SITE, SITE_NAMES[side])
    qpos_ids, dof_ids, joint_ranges = [], [], []
    for name in JOINT_NAMES[side]:
        jid = get_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))
        joint_ranges.append(model.jnt_range[jid].copy())
    leg_info[side] = {
        "site_id": site_id,
        "qpos_ids": qpos_ids,
        "dof_ids": dof_ids,
        "ranges": np.array(joint_ranges),
    }
    print(f"\n{side} LEG")
    print("Joint order:", JOINT_NAMES[side])
    print("DoF indices:", dof_ids)

# ============================================================
# 6. TAO 107 CAU HINH CHO MOI CHAN
# ============================================================
samples = {
    "L": [np.zeros(5), np.deg2rad([10, -10, 15, 25, -10])],
    "R": [np.zeros(5), np.deg2rad([-10, 10, -15, 25, -10])],
}
for _ in range(N_TESTS - 2):
    for side in ("L", "R"):
        ranges = leg_info[side]["ranges"]
        samples[side].append(rng.uniform(ranges[:, 0], ranges[:, 1]))

# Moi chan co 2 phep kiem chung: v (tinh tien), w (quay)
results = {
    side: {
        kind: {
            "passed": 0,
            "max_error": 0.0,
            "worst_case": 0,
            "element_errors": np.zeros((3, 5)),
        } for kind in ("v", "w")
    } for side in ("L", "R")
}

# Thong ke floating-base 6x16 cho CA HAI CHAN, rieng tung nhom cot.
results_float = {
    side: {
        kind: {name: new_result((3, cols.stop-cols.start))
               for name, cols in FLOAT_BLOCKS[side]}
        for kind in ("v", "w")
    } for side in ("L", "R")
}
results_full = {side: new_result((6, 16)) for side in ("L", "R")}
max_base_rotation_error = 0.0
max_site_position_error = {"L": 0.0, "R": 0.0}

# ============================================================
# 7. SO SANH 2 JACOBIAN CHO CA HAI CHAN
# ============================================================
np.set_printoptions(precision=12, suppress=False)

for k in range(N_TESTS):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    if k == 0:
        xyz = np.array([0.0, 0.0, 0.28])
        rpy = np.zeros(3)
    else:
        xyz = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.26, 0.32),
        ])
        rpy = np.deg2rad([
            rng.uniform(-20, 20),
            rng.uniform(-15, 15),
            rng.uniform(-30, 30),
        ])

    data.qpos[free_qpos_adr:free_qpos_adr+3] = xyz
    data.qpos[free_qpos_adr+3:free_qpos_adr+7] = rpy_to_quat(*rpy)

    for side in ("L", "R"):
        for j, adr in enumerate(leg_info[side]["qpos_ids"]):
            data.qpos[adr] = samples[side][k][j]

    mujoco.mj_forward(model, data)
    R_WB = data.xmat[base_id].reshape(3, 3)

    # MuJoCo dung quaternion, con bao cao dung Euler ZYX.
    # T la chuyen doi toa do van toc. Tinh 1 lan/cau hinh.
    T = mujoco_velocity_map(model, data.qpos, rpy, free_qpos_adr,
                            free_dof_adr, leg_info)
    max_base_rotation_error = max(
        max_base_rotation_error,
        float(np.max(np.abs(rotation_zyx(rpy) - R_WB)))
    )

    for side in ("L", "R"):
        q = samples[side][k]
        info = leg_info[side]

        if side == "L":
            Jv_mdh = jacobian_left_mdh(q)
            Jw_mdh = jacobian_angular_left_mdh(q)
        else:
            Jv_mdh = jacobian_right_mdh(q)
            Jw_mdh = jacobian_angular_right_mdh(q)

        # MuJoCo: jacp, jacr deu bieu dien trong {W}
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, info["site_id"])

        # Lay dung 5 cot khop va chuyen {W} -> {B}
        Jv_mujoco = R_WB.T @ jacp[:, info["dof_ids"]]
        Jw_mujoco = R_WB.T @ jacr[:, info["dof_ids"]]

        # ====================================================
        # FLOATING-BASE 6x16: CA HAI CHAN, THEO BIEU THUC RUT GON
        # ====================================================
        if side == "L":
            J_analytic = jacobian_floating_left(q, rpy, Jv_mdh, Jw_mdh)
            p_B_mdh = foot_position_left_B(q)
        else:
            J_analytic = jacobian_floating_right(q, rpy, Jv_mdh, Jw_mdh)
            p_B_mdh = foot_position_right_B(q)

        # MuJoCo tra ve Jacobian theo qvel; doi ve Euler ZYX cua bao cao.
        J_mj_all = np.vstack((jacp, jacr)) @ T
        full_diff = J_analytic - J_mj_all
        full_err = float(np.max(np.abs(full_diff)))
        rf = results_full[side]
        rf["element_errors"] = np.maximum(rf["element_errors"],
                                          np.abs(full_diff))
        if full_err > rf["max_error"]:
            rf["max_error"] = full_err
            rf["worst_case"] = k + 1
        if full_err < TOL:
            rf["passed"] += 1

        # Doi chieu vi tri FK doc lap tu MuJoCo cho moi chan.
        p_B_from_mj = R_WB.T @ (
            data.site_xpos[info["site_id"]] - data.xpos[base_id]
        )
        max_site_position_error[side] = max(
            max_site_position_error[side],
            float(np.max(np.abs(p_B_mdh - p_B_from_mj)))
        )

        for kind, rows in (("v", slice(0, 3)), ("w", slice(3, 6))):
            for name, columns in FLOAT_BLOCKS[side]:
                diff_f = (J_analytic[rows, columns]
                          - J_mj_all[rows, columns])
                error_f = float(np.max(np.abs(diff_f)))
                r = results_float[side][kind][name]
                r["element_errors"] = np.maximum(
                    r["element_errors"], np.abs(diff_f))
                if error_f > r["max_error"]:
                    r["max_error"] = error_f
                    r["worst_case"] = k + 1
                if error_f < TOL:
                    r["passed"] += 1
                else:
                    print(f"\nFAIL FLOATING {side} {kind} {name} CASE {k+1}")
                    print("rpy (deg):", np.rad2deg(rpy))
                    print(f"q {side} (deg):", np.rad2deg(q))
                    print("MDH floating:\n", J_analytic[rows, columns])
                    print("MuJoCo:\n", J_mj_all[rows, columns])
                    print("Difference:\n", diff_f)

        for kind, J_mdh, J_mujoco in (
            ("v", Jv_mdh, Jv_mujoco),
            ("w", Jw_mdh, Jw_mujoco),
        ):
            diff = J_mdh - J_mujoco
            error = float(np.max(np.abs(diff)))
            result = results[side][kind]
            result["element_errors"] = np.maximum(
                result["element_errors"], np.abs(diff)
            )
            if error > result["max_error"]:
                result["max_error"] = error
                result["worst_case"] = k + 1

            if error < TOL:
                result["passed"] += 1
            else:
                print(f"\nFAIL - {side} LEG - {'Jv' if kind == 'v' else 'Jw'} - CASE {k+1}")
                print("q (deg):", np.rad2deg(q))
                print("Jacobian MDH:\n", J_mdh)
                print("Jacobian MuJoCo:\n", J_mujoco)
                print("Difference:\n", diff)

            if k == 0:
                print(f"\n===== {side} LEG - {'Jv' if kind == 'v' else 'Jw'} - ZERO CONFIG =====")
                print("Jacobian MDH:\n", J_mdh)
                print("Jacobian MuJoCo:\n", J_mujoco)
                print("Difference:\n", diff)

# ============================================================
# 8. BAO CAO KET QUA
# ============================================================
print("\n==========================================")
print("TRANSLATIONAL + ANGULAR JACOBIAN CHECK")
print("==========================================")

all_pass = True
passed_total = 0
for side in ("L", "R"):
    for kind in ("v", "w"):
        result = results[side][kind]
        name = "TRANSLATIONAL" if kind == "v" else "ANGULAR"
        status = "PASS" if result["passed"] == N_TESTS else "FAIL"
        print(f"\n{side} LEG - {name} JACOBIAN")
        print(f"Tests          : {N_TESTS}")
        print(f"Passed         : {result['passed']}/{N_TESTS}")
        print(f"Maximum error  : {result['max_error']:.12e}")
        print(f"Worst case     : {result['worst_case']}")
        print(f"Tolerance      : {TOL:.1e}")
        print("Maximum error for each element:")
        print(result["element_errors"])
        print(f"RESULT         : {status}")
        passed_total += result["passed"]
        all_pass &= status == "PASS"

print("\n==========================================")
print("FINAL VERIFICATION")
print("==========================================")
print(f"TOTAL : {passed_total}/{4*N_TESTS}")
if all_pass:
    print("FINAL : PASS")
else:
    print("FINAL : FAIL")
    raise SystemExit(1)


# ============================================================
# 9. BAO CAO FLOATING-BASE CA HAI CHAN (6x16)
# ============================================================
float_all_pass = max_base_rotation_error < TOL
float_full_pass_total = 0
float_group_pass_total = 0
float_group_total = 0

for side in ("L", "R"):
    leg_name = "LEFT" if side == "L" else "RIGHT"
    J_name = "W_J_EL" if side == "L" else "W_J_ER"
    print("\n" + "=" * 62)
    print(f"{leg_name} FLOATING-BASE {J_name} (6x16): MDH vs MuJoCo")
    print("=" * 62)
    print(f"Euler ZYX vs base_link xmat max error: {max_base_rotation_error:.12e}")
    print(f"{leg_name} foot FK vs MuJoCo max error: "
          f"{max_site_position_error[side]:.12e}")

    group_passed = 0
    side_all_pass = (max_base_rotation_error < TOL and
                     max_site_position_error[side] < TOL)
    for kind in ("v", "w"):
        print(f"\n{side} LEG - FLOATING {'Jv' if kind == 'v' else 'Jw'} (3x16)")
        for name, _ in FLOAT_BLOCKS[side]:
            r = results_float[side][kind][name]
            passed = r["passed"]
            group_passed += passed
            side_all_pass &= (passed == N_TESTS)
            print(f"  {name:<19}: {passed:3d}/{N_TESTS}  "
                  f"max = {r['max_error']:.12e}  worst = {r['worst_case']}")
            if passed != N_TESTS:
                print("  Maximum error per element:\n", r["element_errors"])

    rf = results_full[side]
    side_all_pass &= (rf["passed"] == N_TESTS)
    float_full_pass_total += rf["passed"]
    float_group_pass_total += group_passed
    float_group_total += 2 * len(FLOAT_BLOCKS[side]) * N_TESTS
    print("\n" + "=" * 62)
    print(f"FULL {leg_name} {J_name} 6x16: {rf['passed']}/{N_TESTS}  "
          f"max error = {rf['max_error']:.12e}  worst = {rf['worst_case']}")
    if rf["passed"] != N_TESTS:
        print("Maximum error for each element:\n", rf["element_errors"])
    print(f"FLOATING GROUP TOTAL: {group_passed}/"
          f"{2*len(FLOAT_BLOCKS[side])*N_TESTS}")
    print(f"TOLERANCE     : {TOL:.1e}")
    print(f"FINAL {leg_name} FLOATING:", "PASS" if side_all_pass else "FAIL")
    float_all_pass &= side_all_pass

print("\n" + "=" * 62)
print("FLOATING-BASE VERIFICATION - BOTH LEGS")
print("=" * 62)
print(f"FULL 6x16 TOTAL: {float_full_pass_total}/{2*N_TESTS}")
print(f"GROUP TOTAL: {float_group_pass_total}/{float_group_total}")
print("FINAL BOTH FLOATING:", "PASS" if float_all_pass else "FAIL")
if not float_all_pass:
    raise SystemExit(1)
