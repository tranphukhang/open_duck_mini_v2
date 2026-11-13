import torch

class ZMPGenerator:
    def __init__(self, zmp_offset_x=0.005, zmp_offset_y_out=0.007):
        # Độ lệch ZMP tham chiếu so với feet site theo phương x
        self.zmp_offset_x = zmp_offset_x

        # Độ lệch ZMP về phía mép ngoài bàn chân so với feet site theo phương y
        self.zmp_offset_y_out = zmp_offset_y_out

    # Hàm nội suy trơn ZMP trong giai đoạn Shift và DS
    @staticmethod
    def smoothstep(s):
        # Giới hạn hệ số nội suy trong khoảng [0, 1]
        s = s.clamp(0.0, 1.0)

        # Hàm nội suy bậc ba
        return 3.0 * s**2 - 2.0 * s**3

    # Xác định ZMP tham chiếu từ vị trí bàn chân
    def zmp_point(self, foot, side):
        # Xác định dấu theo chân trái/phải
        sign = torch.where(side == 0, 1.0, -1.0)

        # Tính ZMP tham chiếu trong mặt phẳng XY
        zmp = foot[..., :2] + torch.stack([torch.full_like(sign, self.zmp_offset_x), sign * self.zmp_offset_y_out], dim=-1)

        return zmp