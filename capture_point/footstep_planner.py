import torch

class FootstepPlanner:
    def __init__(self, num_envs=1, period_phase=0.4, num_footsteps=14, device="cpu"):
        # Chu kì pha
        self.period_phase = period_phase

        # Chuỗi vị trí đặt chân dự kiến
        self.num_footsteps = num_footsteps

        # Vị trí ban đầu của hai chân
        self.feet0 = torch.zeros(num_envs, 2, 3, dtype=torch.float32, device=device)

        # Độ lệch hình học ban đầu giữa hai chân (dy, dz)
        self.feet_offset0 = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)

        # Vị trí hai chân tại đầu bước hiện tại (Chu kì cập nhật 0.2s)
        self.feet_start = torch.zeros(num_envs, 2, 3, dtype=torch.float32, device=device)

        # Vị trí đặt chân danh định tại một thời điểm
        self.nominal_landing = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)

        # Vị trí đặt chân sau khi cập nhật (Chu kì cập nhật 0.02s)
        self.updated_landing = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        

    # Khởi tạo vị trí đặt chân từ FK trong hệ tọa độ {P}
    def reset(self, env_ids, feet_pos):
        # Vị trí ban đầu của hai chân
        self.feet0[env_ids] = feet_pos

        # Độ lệch hình học ban đầu giữa hai chân (dy, dz)
        self.feet_offset0[env_ids] = (feet_pos[:, 1, 1:3] - feet_pos[:, 0, 1:3])

        # Vị trí hai chân tại đầu bước hiện tại
        self.feet_start[env_ids] = feet_pos

        # Khởi tạo vị trí đặt chân bằng vị trí chân phải (Ban đầu trọng tâm được đưa sang trái do đó chân phải sẽ là chân swing)
        self.nominal_landing[env_ids] = feet_pos[:, 1, :]
        self.updated_landing[env_ids] = feet_pos[:, 1, :]

    # Tính vector độ dịch chuyển đặt chân danh định
    def step_offset(self, swing, cmd):
        # Thời gian của một bước
        half_period_phase = self.period_phase / 2.0

        # Xác định dấu theo chân swing
        sign = torch.where(swing == 1, 1.0, -1.0)

        # Vector độ dịch chuyển đặt chân danh định thông qua vận tốc đặt từ người dùng
        offset = torch.zeros(cmd.shape[0], 3,dtype=cmd.dtype, device=cmd.device)
        offset[:, 0] = cmd[:, 0] * half_period_phase
        offset[:, 1] = sign * self.feet_offset0[:, 0] + cmd[:, 1] * half_period_phase
        offset[:, 2] = sign * self.feet_offset0[:, 1]

        return offset

    # Cập nhật vị trí đặt chân khi bắt đầu bước mới
    def update_new_step(self, j_now, new_step, stance, cmd):
        # Xác định chân swing
        swing = 1 - stance

        # Xác định bước đầu tiên và bước tiếp theo
        first = new_step & (j_now == 0) # Bước đầu tiên sau shift, j = 0
        later = new_step & (j_now > 0) # Các bước tiếp theo, j = 1,...

        # Lấy vị trí hai chân tại đầu bước
        fs = self.feet_start.clone()
        b = torch.arange(fs.shape[0], device=fs.device)

        # Khi j > 0: Chân swing trước đó trở thành chân stance
        fs[b, stance] = torch.where(later[:, None], self.updated_landing, fs[b, stance])

        # Khi j = 0: Sử dụng vị trí hai chân ban đầu
        fs = torch.where(first[:, None, None], self.feet0, fs)

        # Tính vị trí đặt chân danh định cho bước mới
        land = fs[b, stance] + self.step_offset(swing, cmd)

        # Chỉ cập nhật thì có sự kiện new step
        self.feet_start = torch.where(new_step[:, None, None], fs, self.feet_start)
        self.nominal_landing = torch.where(new_step[:, None], land, self.nominal_landing)
        self.updated_landing = torch.where(new_step[:, None], land, self.updated_landing)

    # Hiệu chỉnh vị trí lập kế hoạch theo phương x sau khi hoàn thành một bước dựa trên FK (Ảnh hưởng đến từ WBIK và low-level dẫn đến vị trí đặt chân thực tế sai lệch so với kế hoạch)
    def reanchor_x(self, j_now, new_step, stance, com_relative_to_feet, fk_valid, com_state):
        # Xác định chân stance trước thời điểm hoàn thành bước (chân này sẽ trở thành swing trong bước tiếp theo)
        swing = 1 - stance

        b = torch.arange(self.feet_start.shape[0], device=self.feet_start.device)

        # Ước lượng vị trí đặt chân của chân swing trước thời điểm hoàn thành bước (chân này hiện đã trở thành stance)
        landed_x = self.feet_start[b, swing, 0] + com_relative_to_feet[b, swing, 0] - com_relative_to_feet[b, stance, 0]

        # Chỉ hiệu chỉnh khi đã hoàn thành ít nhất một bước và dữ liệu FK hợp lệ
        condition = new_step & (j_now > 0) & fk_valid

        # Sai lệch giữa vị trí tiếp đất ước lượng từ FK và vị trí đã lập kế hoạch
        delta_x = torch.where(condition, landed_x - self.feet_start[b, stance, 0], torch.zeros_like(landed_x))

        # Hiệu chỉnh đồng bộ các vị trí lập kế hoạch theo phương x
        self.feet_start[:, :, 0] += delta_x[:, None]
        self.nominal_landing[:, 0] += delta_x
        self.updated_landing[:, 0] += delta_x
        com_state[:, 0, 0] += delta_x # Đồng bộ lại CoM được dự đoán từ MPC

        return delta_x

    # Cập nhật vị trí đặt chân sau khi được hiệu chỉnh bởi Capture Point
    def set_landing(self, corrected_landing, allow_adjust):
        self.updated_landing = torch.where(allow_adjust[:, None], corrected_landing, self.updated_landing)

    # Tạo chuỗi 14 vị trí đặt chân dự kiến trong hệ tọa độ {P}
    def footstep_sequence(self, j_now, cmd):
        num_footsteps = self.num_footsteps

        # Khi j = -1: Dự đoán từ bước j = 0
        j_base = j_now.clamp_min(0)

        # Xác định chân stance và swing của bước cơ sở
        stance = torch.remainder(j_base, 2)
        swing = 1 - stance

        b = torch.arange(self.feet_start.shape[0], device=self.feet_start.device)

        # Vị trí chân swing tại đầu bước
        p_minus_1 = self.feet_start[b, swing]

        # Vị trí chân stance tại đầu bước
        p0 = self.feet_start[b, stance]

        # Vị trí tiếp đất của chân swing hiện tại
        # Khi Shift: tính danh định từ step_offset()
        # Khi Walking: sử dụng updated_landing (có thể đã hiệu chỉnh CP)
        p1 = torch.where((j_now >= 0)[:, None], self.updated_landing, p0 + self.step_offset(swing, cmd))

        # Vector dịch chuyển đặt chân cho các bước tiếp theo
        # Chân stance hiện tại sẽ swing ở bước kế tiếp
        offset = torch.stack([self.step_offset(stance, cmd), self.step_offset(swing, cmd)], dim=1)

        # Tạo chỉ số luân phiên chân: 0, 1, 0, 1, ...
        alternate = torch.arange(num_footsteps - 3, device=self.feet_start.device) % 2

        # Tính các vị trí tiếp đất từ p2 đến p12
        later = p1[:, None, :] + torch.cumsum(offset[:, alternate, :], dim=1)

        # Ghép thành chuỗi 14 vị trí từ p-1 đến p12
        footsteps = torch.cat([p_minus_1[:, None, :], p0[:, None, :], p1[:, None, :], later], dim=1)

        return footsteps, j_base





        

    