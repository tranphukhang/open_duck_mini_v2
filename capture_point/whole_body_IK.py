
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
