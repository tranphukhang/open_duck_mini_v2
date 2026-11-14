
import mujoco
import numpy as np
import torch


class RobotKinematics:
    def __init__(self, model: mujoco.MjModel, device="cpu"):
        self.device = torch.device(device)

        # Tên và thứ tự 10 khớp chân trong mô hình v5
        joint_names = ("L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",)
        foot_sites = ("L6_IK", "R6_IK")
        n_joints = len(joint_names)

        # Chuyển dữ liệu từ MuJoCo sang PyTorch
        def to_tensor(data):
            return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=self.device)

        # --------------------------------------------------
        # 1. Xác định cây động học của robot
        # --------------------------------------------------
        base_id = model.body("base_link").id

        # Lấy base và toàn bộ các khâu hậu duệ
        bodies = [base_id]

        for b in range(base_id + 1, model.nbody):
            if int(model.body_parentid[b]) in bodies:
                bodies.append(b)

        self.bodies = bodies
        n_bodies = len(bodies)

        if n_bodies != 11:
            raise ValueError(f"Expected 11 robot bodies, got {n_bodies}")

        # Ánh xạ chỉ số body của MuJoCo sang cây robot
        body_index = {b: i for i, b in enumerate(bodies)}

        self.parent = [body_index.get(int(model.body_parentid[b]), -1) for b in bodies]
        self.parent[0] = -1

        # --------------------------------------------------
        # 2. Xác định các khớp và trục quay
        # --------------------------------------------------
        joint_index = {name: i for i, name in enumerate(joint_names)}

        self.joint_of_body = [-1] * n_bodies
        self.joint_body = [-1] * n_joints

        joint_axis = np.zeros((n_joints, 3))

        for i, b in enumerate(bodies[1:], start=1):
            if model.body_jntnum[b] != 1:
                raise ValueError(f"Body {model.body(b).name} must have one joint")

            jid = int(model.body_jntadr[b])

            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Expected hinge joint")

            if np.linalg.norm(model.jnt_pos[jid]) > 1e-9:
                raise ValueError("Joint anchor must be at body origin")

            name = model.joint(jid).name
            ji = joint_index[name]

            self.joint_of_body[i] = ji
            self.joint_body[ji] = i
            joint_axis[ji] = model.jnt_axis[jid]

        if any(i < 0 for i in self.joint_body):
            raise ValueError("Missing leg joints in MJCF")

        self.joint_axis = to_tensor(joint_axis)

        # --------------------------------------------------
        # 3. Đọc thông số hình học và khối lượng
        # --------------------------------------------------
        self.body_pos0 = to_tensor(model.body_pos[bodies])

        body_rot0 = np.zeros((n_bodies, 3, 3))

        for i, b in enumerate(bodies):
            mujoco.mju_quat2Mat(body_rot0[i].reshape(9), model.body_quat[b])

        self.body_rot0 = to_tensor(body_rot0)

        self.mass = to_tensor(model.body_mass[bodies])
        self.total_mass = float(self.mass.sum())
        self.ipos = to_tensor(model.body_ipos[bodies])

        # --------------------------------------------------
        # 4. Đọc vị trí và hướng hai site bàn chân
        # --------------------------------------------------
        site_ids = [model.site(name).id for name in foot_sites]

        self.site_body = [body_index[int(model.site_bodyid[s])] for s in site_ids]

        self.site_pos0 = to_tensor(model.site_pos[site_ids])

        site_rot0 = np.zeros((2, 3, 3))

        for i, s in enumerate(site_ids):
            mujoco.mju_quat2Mat(site_rot0[i].reshape(9), model.site_quat[s])

        self.site_rot0 = to_tensor(site_rot0)

        # --------------------------------------------------
        # 5. Đọc giới hạn khớp
        # --------------------------------------------------
        joint_ids = [model.joint(name).id for name in joint_names]

        self.qmin = to_tensor(model.jnt_range[joint_ids, 0])
        self.qmax = to_tensor(model.jnt_range[joint_ids, 1])

        # --------------------------------------------------
        # 6. Xây dựng ma trận desc
        # --------------------------------------------------
        desc = torch.zeros(n_joints, n_bodies, dtype=torch.float32, device=self.device)

        for ji, joint_body in enumerate(self.joint_body):
            for bi in range(n_bodies):
                k = bi

                while k != -1:
                    if k == joint_body:
                        desc[ji, bi] = 1.0
                        break

                    k = self.parent[k]

        self.desc = desc

        # Quan hệ giữa 10 khớp và hai site bàn chân
        self.site_desc = self.desc[:, self.site_body].T.contiguous()

        # Chỉ số khớp của hai chân
        self.legs = (torch.arange(0, 5, device=self.device), torch.arange(5, 10, device=self.device))

        # --------------------------------------------------
        # 7. Nhóm các khâu theo độ sâu cây động học
        # --------------------------------------------------
        depth = [0] * n_bodies

        for i in range(1, n_bodies):
            depth[i] = depth[self.parent[i]] + 1

        self.levels = []
        body_order = [0]

        for d in range(1, max(depth) + 1):
            ids = [i for i in range(n_bodies) if depth[i] == d]

            prev = [i for i in range(n_bodies) if depth[i] == d - 1]

            js = [self.joint_of_body[i] for i in ids]

            joint_ids_level = torch.tensor(js, dtype=torch.long, device=self.device)

            axis = self.joint_axis[joint_ids_level]

            # Ma trận skew của trục quay
            x, y, z = axis.unbind(-1)
            o = torch.zeros_like(x)

            k = torch.stack([
                o, -z, y,
                z, o, -x,
                -y, x, o
            ], dim=-1).reshape(-1, 3, 3)

            self.levels.append({
                "parent": torch.tensor([prev.index(self.parent[i]) for i in ids], dtype=torch.long, device=self.device),
                "joint": joint_ids_level,
                "pos0": self.body_pos0[ids],
                "rot0": self.body_rot0[ids],
                "axis": axis,
                "k": k,
                "k2": k @ k
            })

            body_order.extend(ids)

        # --------------------------------------------------
        # 8. Ánh xạ thứ tự khâu/khớp sau khi nhóm theo độ sâu
        # --------------------------------------------------
        self.body_from_level = torch.tensor([body_order.index(i) for i in range(n_bodies)], dtype=torch.long, device=self.device)

        joint_order = [j for level in self.levels for j in level["joint"].tolist()]

        self.joint_from_level = torch.tensor([joint_order.index(j) for j in range(n_joints)], dtype=torch.long, device=self.device)
