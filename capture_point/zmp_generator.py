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

    # Tạo ZMP tham chiếu trong giai đoạn Shift ban đầu
    def zmp_shift(self, feet0, local_future, shift_duration):
        # Chỉ số chân trái và chân phải
        left = torch.zeros(feet0.shape[0], dtype=torch.long, device=feet0.device)
        right = torch.ones_like(left)

        # ZMP tham chiếu tại hai chân ban đầu
        zmp_left = self.zmp_point(feet0[:, 0], left)
        zmp_right = self.zmp_point(feet0[:, 1], right)

        # Trung điểm ZMP của hai chân
        zmp_mid = 0.5 * (zmp_left + zmp_right)

        # Hệ số nội suy theo thời gian Shift
        s = self.smoothstep(local_future / shift_duration)[..., None]

        # Nội suy ZMP từ trung điểm sang chân trái
        zmp_shift = ((1.0 - s) * zmp_mid[:, None, :]+ s * zmp_left[:, None, :])

        return zmp_shift
