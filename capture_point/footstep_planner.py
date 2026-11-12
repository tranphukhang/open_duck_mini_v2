import torch

class FootstepPlanner:
    def __init__(self, num_envs=1, device="cpu"):
        # Vị trí ban đầu của hai chân
        self.feet0 = torch.zeros(num_envs, 2, 3, dtype=torch.float32, device=device)

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

        # Vị trí hai chân tại đầu bước hiện tại
        self.feet_start[env_ids] = feet_pos

        # Khởi tạo vị trí đặt chân bằng vị trí chân phải (Ban đầu trọng tâm được đưa sang trái do đó chân phải sẽ là chân swing)
        self.nominal_landing[env_ids] = feet_pos[:, 1, :]
        self.updated_landing[env_ids] = feet_pos[:, 1, :]