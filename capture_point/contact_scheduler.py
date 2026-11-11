import torch

class ContactScheduler:
    def __init__(self, dt=0.02, num_envs=1, device="cpu"):
        self.dt = dt
        self.tick = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Thời gian đã trôi qua kể từ khi bắt đầu (tick * dt)
        self.t = torch.zeros(num_envs, dtype=torch.float32, device=device)

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

        