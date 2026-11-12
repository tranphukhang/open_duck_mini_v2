import torch

class FootstepPlanner:
    def __init__(self, num_envs=1, period_phase=0.4, device="cpu"):
        # Chu kì pha
        self.period_phase = period_phase

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

        # Vector độ dịch chuyển đặt chân danh định
        offset = torch.zeros(cmd.shape[0], 3,dtype=cmd.dtype, device=cmd.device)
        offset[:, 0] = cmd[:, 0] * half_period_phase
        offset[:, 1] = sign * self.feet_offset0[:, 0] + cmd[:, 1] * half_period_phase
        offset[:, 2] = sign * self.feet_offset0[:, 1]

        return offset


        

    