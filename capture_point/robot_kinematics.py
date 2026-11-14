from dataclasses import dataclass
import mujoco
import numpy as np
import torch


@dataclass
class KinState:
    """Trạng thái động học của robot trong hệ {P}."""

    body_pos: torch.Tensor   # (B, 11, 3)
    body_rot: torch.Tensor   # (B, 11, 3, 3)

    com_body: torch.Tensor   # (B, 11, 3)
    com: torch.Tensor        # (B, 3)

    site_pos: torch.Tensor   # (B, 2, 3)
    site_rot: torch.Tensor   # (B, 2, 3, 3)

    axis: torch.Tensor       # (B, 10, 3)
    anchor: torch.Tensor     # (B, 10, 3)


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

    
    def compute_body_kinematics(self, base_pos, base_rot, q):
        """
        Tính động học các khâu và khớp của robot.

        Args:
            base_pos: (B, 3)       Vị trí base trong {P}
            base_rot: (B, 3, 3)    Hướng base trong {P}
            q:        (B, 10)      Góc 10 khớp (rad)

        Returns:
            body_pos: (B, 11, 3)      Vị trí các khâu
            body_rot: (B, 11, 3, 3)   Hướng các khâu
            axis:     (B, 10, 3)      Trục khớp trong {P}
            anchor:   (B, 10, 3)      Gốc khớp trong {P}
        """

        # --------------------------------------------------
        # 1. Khởi tạo trạng thái base
        # --------------------------------------------------
        body_pos_levels = [base_pos.unsqueeze(1)]
        body_rot_levels = [base_rot.unsqueeze(1)]

        axis_levels = []
        anchor_levels = []

        I = torch.eye(3, dtype=base_rot.dtype, device=base_rot.device)

        # --------------------------------------------------
        # 2. Tính FK theo từng độ sâu
        # --------------------------------------------------
        for level in self.levels:

            # Khâu cha thuộc mức trước
            parent = level["parent"]

            p_parent = body_pos_levels[-1][:, parent]
            R_parent = body_rot_levels[-1][:, parent]

            # Góc khớp tại mức đang xét
            q_level = q[:, level["joint"]]

            # Vị trí gốc khâu
            p_body = p_parent + (R_parent @ level["pos0"].unsqueeze(-1)).squeeze(-1)

            # Hướng cố định trước phép quay khớp
            R_fixed = R_parent @ level["rot0"]

            # Ma trận quay khớp theo Rodrigues
            angle = q_level[..., None, None]

            R_joint = (I + torch.sin(angle) * level["k"] + (1.0 - torch.cos(angle)) * level["k2"])

            # Hướng khâu sau khi quay khớp
            R_body = R_fixed @ R_joint

            # Trục quay khớp trong hệ {P}
            axis = (R_fixed @ level["axis"].unsqueeze(-1)).squeeze(-1)

            # Gốc khớp trùng gốc khâu (MJCF v5)
            anchor = p_body

            # Lưu kết quả của mức hiện tại
            body_pos_levels.append(p_body)
            body_rot_levels.append(R_body)

            axis_levels.append(axis)
            anchor_levels.append(anchor)

        # --------------------------------------------------
        # 3. Ghép kết quả và khôi phục thứ tự khâu
        # --------------------------------------------------
        body_pos = torch.cat(body_pos_levels, dim=1)[:, self.body_from_level]

        body_rot = torch.cat(body_rot_levels, dim=1)[:, self.body_from_level]

        # --------------------------------------------------
        # 4. Khôi phục thứ tự 10 khớp
        # --------------------------------------------------
        axis = torch.cat(axis_levels, dim=1)[:, self.joint_from_level]

        anchor = torch.cat(anchor_levels, dim=1)[:, self.joint_from_level]

        return body_pos, body_rot, axis, anchor


    def compute_center_of_mass(self, body_pos, body_rot):
        """
        Tính vị trí CoM từng khâu và CoM toàn robot.

        Args:
            body_pos: (B, 11, 3)     Vị trí các khâu trong {P}
            body_rot: (B, 11, 3, 3)  Hướng các khâu trong {P}

        Returns:
            com_body: (B, 11, 3)     CoM từng khâu trong {P}
            com:      (B, 3)         CoM toàn robot trong {P}
        """

        # 1. Chuyển CoM cục bộ từng khâu sang hệ {P}
        com_body = body_pos + (body_rot @ self.ipos[None, :, :, None]).squeeze(-1)

        # 2. Tính CoM toàn robot theo trọng số khối lượng
        com = (com_body * self.mass[None, :, None]).sum(dim=1) / self.total_mass

        return com_body, com


    def compute_foot_kinematics(self, body_pos, body_rot):
        """
        Tính vị trí và hướng hai bàn chân trong hệ {P}.

        Args:
            body_pos: (B, 11, 3)     Vị trí các khâu trong {P}
            body_rot: (B, 11, 3, 3)  Hướng các khâu trong {P}

        Returns:
            site_pos: (B, 2, 3)      Vị trí hai bàn chân
            site_rot: (B, 2, 3, 3)   Hướng hai bàn chân

        Thứ tự site:
            0: L6_IK
            1: R6_IK
        """

        # 1. Lấy vị trí và hướng của các khâu chứa site IK
        foot_body_pos = body_pos[:, self.site_body]
        foot_body_rot = body_rot[:, self.site_body]

        # 2. Tính vị trí hai site trong hệ {P}
        site_pos = foot_body_pos + (foot_body_rot @ self.site_pos0[None, :, :, None]).squeeze(-1)

        # 3. Tính hướng hai site trong hệ {P}
        site_rot = foot_body_rot @ self.site_rot0[None, :, :, :]

        return site_pos, site_rot

    
    def forward_kinematics(self, base_pos, base_rot, q):
        """
        Tính động học thuận toàn robot.

        Args:
            base_pos: (B, 3)
                Vị trí base trong {P}.

            base_rot: (B, 3, 3)
                Hướng base trong {P}.

            q: (B, 10)
                Góc 10 khớp chân (rad).

        Returns:
            KinState:
                Chứa trạng thái động học toàn robot.
        """

        # 1. Tính vị trí, hướng các khâu và thông số khớp
        body_pos, body_rot, axis, anchor = (self.compute_body_kinematics(base_pos, base_rot, q))

        # 2. Tính CoM từng khâu và CoM toàn robot
        com_body, com = self.compute_center_of_mass(body_pos, body_rot)

        # 3. Tính vị trí và hướng hai bàn chân
        site_pos, site_rot = self.compute_foot_kinematics(body_pos, body_rot)

        # 4. Đóng gói trạng thái động học
        return KinState(
            body_pos=body_pos,
            body_rot=body_rot,
            com_body=com_body,
            com=com,
            site_pos=site_pos,
            site_rot=site_rot,
            axis=axis,
            anchor=anchor,
        )


