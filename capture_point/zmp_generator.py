
import torch


class ZMPGenerator:
    # Hàm nội suy trơn ZMP trong giai đoạn Shift và DS
    @staticmethod
    def smoothstep(s):
        # Giới hạn hệ số nội suy trong khoảng [0, 1]
        s = s.clamp(0.0, 1.0)

        # Hàm nội suy bậc ba
        return 3.0 * s**2 - 2.0 * s**3
