import torch

class ContactScheduler:
    def __init__(self, dt=0.02, period_phase=0.4, shift=0.4, double_support=0.1, num_envs=1, device="cpu"):
        
        # Thời gian
        self.dt = dt
        self.tick = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Thời gian đã trôi qua kể từ khi bắt đầu (tick * dt)
        self.t = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Chỉ số bước
        self.period_phase = period_phase
        self.shift = shift

        # Trạng thái của pha
        self.double_support = double_support # Phần trăm thời gian DS trong một pha

    # Reset thời gian đã trôi về 0
    def reset(self, env_ids=None):
        if env_ids is None:
            self.tick.zero_()
            self.t.zero_()
        else:
            self.tick[env_ids] = 0
            self.t[env_ids] = 0.0

    # Đếm thời gian
    def count(self):
        self.tick = self.tick + 1
        self.t = self.tick.float() * self.dt

    # Xác định chỉ số bước theo thời gian
    def step_index(self, t):
        half_period_phase = self.period_phase / 2.0
        eps = 1e-3 * self.dt

        # Chỉ số bước
        j = torch.floor((t - self.shift + eps) / half_period_phase).long()

        # Nếu chưa hoàn thành giai đoạn chuyển tâm khối lượng thì chỉ số bước là -1
        shift_condition = (t - self.shift) < -eps
        j = torch.where(shift_condition, torch.full_like(j, -1), j)

        return j

    # Xác định thời gian cục bộ trong bước hiện tại
    def local_time_phase(self, t, j):
        half_period_phase = self.period_phase / 2.0

        # Thời gian cục bộ trong bước hiện tại (Chuyển tâm khối lượng thì thời gian cục bộ là t)
        local_time = ((t - self.shift) - j * half_period_phase).clamp(min=0.0)
        local_time = torch.where(j == -1, t, local_time)

        return local_time

    def phase_state(self, j, local_time):
        ds = self.double_support * self.period_phase

        # Trạng thái của pha
        shift_condition = j < 0
        ds_condition = (j >= 0) & (local_time < ds)
        ss_condition = (j >= 0) & (local_time >= ds)

        return shift_condition, ds_condition, ss_condition
        