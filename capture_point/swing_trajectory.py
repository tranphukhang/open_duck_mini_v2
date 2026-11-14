import torch

class SwingTrajectoryGenerator:
    def __init__(self, period_phase=0.4, double_support=0.1, swing_height=0.02):
        # Chu kỳ gồm hai bước chân
        self.period_phase = period_phase

        # Tỷ lệ thời gian Double Support
        self.double_support = double_support

        # Độ cao nâng chân tối đa
        self.swing_height = swing_height

        # Thời gian của một bước
        self.step_duration = self.period_phase / 2.0

        # Thời gian Double Support
        self.ds_duration = self.double_support * self.period_phase

        # Thời gian thực hiện Swing
        self.swing_duration = self.step_duration - self.ds_duration

        # Hàm nội suy trơn bậc ba cho quỹ đạo chân swing
    @staticmethod
    def smoothstep(s):
        # Giới hạn thời gian chuẩn hóa trong khoảng [0, 1]
        s = s.clamp(0.0, 1.0)

        # Tính hệ số nội suy bậc ba
        sigma = 3.0 * s**2 - 2.0 * s**3

        return sigma

    
    # Tính vị trí mục tiêu chân swing theo thời gian pha
    def compute_swing_position(self, start, end, local_next):
        # start:      (B, 3) - Vị trí bắt đầu swing
        # end:        (B, 3) - Vị trí tiếp đất mục tiêu
        # local_next: (B,)   - Thời gian cục bộ tại t(k+1)

        # Chuẩn hóa thời gian trong pha swing
        s = ((local_next - self.ds_duration) / self.swing_duration).clamp(0.0, 1.0)

        # Hệ số nội suy trơn theo X/Y
        sigma = self.smoothstep(s)

        # Nội suy vị trí chân swing
        sw_pos = start + (end - start) * sigma[:, None]

        # Tính độ nâng chân theo Z
        sw_pos[:, 2] = (start[:, 2] + self.swing_height * 64.0 * s**3 * (1.0 - s)**3)

        return sw_pos

    
    # Tạo vị trí mục tiêu hai bàn chân tại thời điểm t(k+1)
    def generate(self, footsteps, feet0, j_base, j_next, local_next, contact_next):
        # footsteps:    (B, 14, 3) - Chuỗi vị trí đặt chân
        # feet0:        (B, 2, 3)  - Vị trí hai chân ban đầu
        # j_base:       (B,)       - Chỉ số bước cơ sở
        # j_next:       (B,)       - Chỉ số bước tại t(k+1)
        # local_next:   (B,)       - Thời gian cục bộ tại t(k+1)
        # contact_next: (B, 2)    - Trạng thái tiếp xúc tại t(k+1)

        # Chỉ số môi trường
        b = torch.arange(footsteps.shape[0], device=footsteps.device)

        # Xác định chỉ số bước tương đối trong chuỗi footstep
        i = (j_next - j_base).clamp(0, footsteps.shape[1] - 3)

        # Xác định chân stance và swing
        stance = torch.remainder(j_next.clamp_min(0), 2)
        swing = 1 - stance

        # Lấy vị trí bắt đầu swing, chân stance và vị trí tiếp đất
        start = footsteps[b, i]
        stance_pos = footsteps[b, i + 1]
        end = footsteps[b, i + 2]

        # Tính vị trí mục tiêu chân swing
        sw_pos = self.compute_swing_position(start, end,local_next)

        # Kiểm tra chân swing đã rời mặt đất hay chưa
        swinging = (j_next >= 0) & (~contact_next[b, swing])

        # Trong Double Support, giữ chân swing tại vị trí ban đầu
        sw_pos = torch.where(swinging[:, None], sw_pos, start)

        # Ghép vị trí mục tiêu của hai bàn chân
        feet_target = torch.zeros_like(feet0)
        feet_target[b, stance] = stance_pos
        feet_target[b, swing] = sw_pos

        # Trong giai đoạn Shift, giữ nguyên vị trí hai chân ban đầu
        feet_target = torch.where((j_next < 0)[:, None, None], feet0, feet_target)

        return feet_target, contact_next

