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

    # Xác định trạng thái của pha (Shift, Double Support, Swing)
    def phase_state(self, j, local_time):
        ds = self.double_support * self.period_phase
        eps = 1e-3 * self.dt

        # Trạng thái của pha
        shift_condition = j < 0
        ss_condition = (j >= 0) & (local_time + eps >= ds)
        ds_condition = (j >= 0) & ~ss_condition
        
        return shift_condition, ds_condition, ss_condition

    # Xác định chân stance theo chỉ số bước
    def stance_feet(self, j):
        stance = torch.remainder(j.clamp_min(0), 2) # stance = 0 (chân trái) nếu j chẵn, stance = 1 (chân phải) nếu j lẻ

        return stance

    # Xác định trạng thái tiếp xúc của hai chân
    def contact_state(self, stance, ss_condition):
        # Xác định chân Swing
        swing = 1 - stance # 0 là chân trái swing, 1 là chân phải swing

        contact_condition = torch.ones(stance.shape[0], 2, dtype=torch.bool, device=stance.device)
        b = torch.arange(stance.shape[0], device=stance.device)
        contact_condition[b, swing] = ~ss_condition

        return contact_condition # ["chân trái", "chân phải"]

    # Phát hiện sự kiện bắt đầu bước mới (Chạy theo chu kì điều khiển (hiện tại là 0.02s))
    def new_step_event(self, j_now, last_step):
        new_step = (j_now >= 0) & (j_now != last_step)

        return new_step