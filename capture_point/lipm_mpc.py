import torch

class LIPMMPC:
    def __init__(self, com_height, dt=0.02, horizon=80, q_zmp=1.0, q_vel=1e-3, r_jerk=1e-6, device="cpu"):
        # Chu kỳ điều khiển
        self.dt = dt

        # Số bước dự đoán MPC
        self.horizon = horizon

        # Chiều cao CoM danh định so với mặt phẳng tiếp xúc
        self.com_height = com_height

        # Gia tốc trọng trường
        self.g = 9.81

        # Trọng số của hàm chi phí MPC
        self.q_zmp = q_zmp
        self.q_vel = q_vel
        self.r_jerk = r_jerk

        # Thiết bị tính toán
        self.device = torch.device(device)

    
    # Xây dựng mô hình trạng thái LIPM rời rạc
    def build_model(self):
        # Chu kỳ điều khiển và chiều cao CoM danh định
        dt = self.dt
        h = self.com_height
        g = self.g

        # Ma trận trạng thái
        self.A = torch.tensor([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=self.device)

        # Vector đầu vào jerk
        self.B = torch.tensor([
            dt**3 / 6.0,
            0.5 * dt**2,
            dt
        ], dtype=torch.float32, device=self.device)

        # Vector đầu ra ZMP
        self.C = torch.tensor([
            1.0,
            0.0,
            -h / g
        ], dtype=torch.float32, device=self.device)

