
from dataclasses import dataclass

import torch

from robot_kinematics import RobotKinematics


@dataclass
class IkWeights:
    """Trọng số và tham số của bộ giải WholeBodyIK."""

    com: float = 100.0
    foot_pos: float = 100.0
    foot_rot: float = 30.0
    base_rot: tuple = (0.05, 5.0, 5.0)
    posture: float = 1e-3

    damping: float = 1e-4
    iterations: int = 8


class WholeBodyIK:
    def __init__(self, kin: RobotKinematics, weights=None):
        """
        Khởi tạo bộ giải Whole-Body Inverse Kinematics.

        Args:
            kin: RobotKinematics
                Đối tượng động học của robot.

            weights: IkWeights hoặc None
                Trọng số và tham số WDLS.
                Nếu None, sử dụng giá trị mặc định.
        """

        # 1. Lưu đối tượng động học
        self.kin = kin
        self.device = kin.device

        # 2. Khởi tạo tham số WDLS
        self.weights = (
            weights if weights is not None
            else IkWeights()
        )

        w = self.weights

        # 3. Xây dựng vector trọng số của 28 task
        self.weight_vector = torch.tensor(
            [w.com] * 3
            + [w.foot_pos] * 3
            + [w.foot_rot] * 3
            + [w.foot_pos] * 3
            + [w.foot_rot] * 3
            + list(w.base_rot)
            + [w.posture] * 10,
            dtype=kin.mass.dtype,
            device=self.device
        )


    @staticmethod
    def compute_rotation_error(R_des, R_cur):
        """
        Tính vector sai số hướng trong hệ {P}.

        Args:
            R_des: (..., 3, 3)
                Ma trận quay mong muốn.

            R_cur: (..., 3, 3)
                Ma trận quay hiện tại.

        Returns:
            e_rot: (..., 3)
                Vector sai số hướng trong hệ {P}.
        """

        # 1. Tính ma trận quay sai lệch
        R_err = R_des @ R_cur.transpose(-1, -2)

        # 2. Trích xuất vector sai số hướng
        e_rot = 0.5 * torch.stack([
            R_err[..., 2, 1] - R_err[..., 1, 2],
            R_err[..., 0, 2] - R_err[..., 2, 0],
            R_err[..., 1, 0] - R_err[..., 0, 1]
        ], dim=-1)

        return e_rot


    @staticmethod
    def rotation_vector_to_matrix(rotvec):
        """
        Chuyển vector quay thành ma trận quay bằng Rodrigues.

        Args:
            rotvec: (..., 3)
                Vector quay trong hệ {P}, đơn vị rad.

        Returns:
            R: (..., 3, 3)
                Ma trận quay tương ứng.
        """

        # 1. Tính bình phương góc quay
        theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta2)

        # 2. Tính hệ số Rodrigues
        # Khai triển Taylor khi góc quay rất nhỏ
        small = theta2 < 1e-6

        theta_safe = torch.where(small, torch.ones_like(theta), theta)

        theta2_safe = torch.where(small, torch.ones_like(theta2), theta2)

        A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta_safe)

        B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / theta2_safe)

        # 3. Xây dựng ma trận skew
        x, y, z = rotvec.unbind(dim=-1)

        zero = torch.zeros_like(x)

        K = torch.stack([
            zero, -z,    y,
            z,     zero, -x,
            -y,    x,     zero
        ], dim=-1).reshape(*rotvec.shape[:-1], 3, 3)

        # 4. Tính ma trận quay
        I = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)

        R = (I + A[..., None] * K + B[..., None] * (K @ K))

        return R

    
    def build_tasks(self, ks, base_pos, base_rot, q, com_des, feet_des, feet_rot_des, q_home):
        """
        Xây dựng Jacobian và vector sai số cho WholeBodyIK.

        Args:
            ks: KinState
                Trạng thái động học hiện tại.

            base_pos: (B, 3)
                Vị trí base hiện tại.

            base_rot: (B, 3, 3)
                Hướng base hiện tại.

            q: (B, 10)
                Góc khớp hiện tại.

            com_des: (B, 3)
                Vị trí CoM mong muốn.

            feet_des: (B, 2, 3)
                Vị trí hai bàn chân mong muốn.

            feet_rot_des: (B, 2, 3, 3)
                Hướng hai bàn chân mong muốn.

            q_home: (B, 10)
                Cấu hình khớp chuẩn.

        Returns:
            J: (B, 28, 16)
                Jacobian tổng hợp.

            e: (B, 28)
                Vector sai số tổng hợp.
        """

        B = q.shape[0]

        J_list = []
        e_list = []

        # --------------------------------------------------
        # 1. Task vị trí CoM
        # --------------------------------------------------
        J_com = self.kin.compute_com_jacobian(ks, base_pos)

        e_com = com_des - ks.com

        J_list.append(J_com)
        e_list.append(e_com)

        # --------------------------------------------------
        # 2. Task vị trí và hướng hai bàn chân
        # --------------------------------------------------
        for i in range(2):

            # Mặt nạ các khớp của chân đang xét
            desc = self.kin.site_desc[i]

            # Task vị trí bàn chân
            J_foot_pos = self.kin.compute_point_jacobian(ks, ks.site_pos[:, i], desc, base_pos)

            e_foot_pos = (feet_des[:, i] - ks.site_pos[:, i])

            J_list.append(J_foot_pos)
            e_list.append(e_foot_pos)

            # Task hướng bàn chân
            J_foot_rot = self.kin.compute_rotation_jacobian(ks, desc)

            e_foot_rot = self.compute_rotation_error(feet_rot_des[:, i], ks.site_rot[:, i])

            J_list.append(J_foot_rot)
            e_list.append(e_foot_rot)

        # --------------------------------------------------
        # 3. Task hướng base
        # --------------------------------------------------
        R_base_des = torch.eye(3, dtype=base_rot.dtype, device=base_rot.device).unsqueeze(0).expand(B, -1, -1)

        J_base_rot = self.kin.compute_rotation_jacobian(ks, None)

        e_base_rot = self.compute_rotation_error(R_base_des, base_rot)

        J_list.append(J_base_rot)
        e_list.append(e_base_rot)

        # --------------------------------------------------
        # 4. Task posture
        # --------------------------------------------------
        J_posture = torch.cat([
            torch.zeros(
                B, 10, 6,
                dtype=q.dtype,
                device=q.device
            ),
            torch.eye(
                10,
                dtype=q.dtype,
                device=q.device
            ).unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)

        e_posture = q_home - q

        J_list.append(J_posture)
        e_list.append(e_posture)

        # --------------------------------------------------
        # 5. Ghép các task
        # --------------------------------------------------
        J = torch.cat(J_list, dim=1)
        e = torch.cat(e_list, dim=1)

        return J, e

