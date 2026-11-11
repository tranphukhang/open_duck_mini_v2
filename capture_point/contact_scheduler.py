import torch

class ContactScheduler:
    def __init__(self, dt=0.02, period_phase=0.4, shift=0.4, num_envs=1, device="cpu"):
        
        # Thời gian
        self.dt = dt
        self.tick = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Thời gian đã trôi qua kể từ khi bắt đầu (tick * dt)
        self.t = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Chỉ số bước
        self.period_phase = period_phase
        self.shift = shift

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
        shift = (t - self.shift) < -eps
        j = torch.where(shift, torch.full_like(j, -1), j)

        return j

        