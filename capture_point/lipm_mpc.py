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


    # Xây dựng các ma trận dự đoán trạng thái CoM trong horizon MPC
    def build_state_prediction(self):
        # Số bước dự đoán
        N = self.horizon

        # Ma trận trạng thái và vector đầu vào
        A = self.A
        B = self.B

        # Tính trước các lũy thừa A^0, A^1, ..., A^N
        powers = [torch.eye(3, dtype=A.dtype, device=self.device)]

        for _ in range(N):
            powers.append(A @ powers[-1])

        # Ma trận dự đoán từ trạng thái ban đầu
        # Phi[i] = A^(i+1)
        self.Phi = torch.stack(powers[1:], dim=0)

        # Ma trận dự đoán từ chuỗi jerk
        self.Gamma = torch.zeros(N, 3, N, dtype=A.dtype, device=self.device)

        # Gamma[i, :, j] = A^(i-j) @ B khi j <= i
        for i in range(N):
            for j in range(i + 1):
                self.Gamma[i, :, j] = powers[i - j] @ B


    # Xây dựng các ma trận dự đoán ZMP và vận tốc CoM
    def build_output_prediction(self):
        # Ma trận đầu ra ZMP
        C = self.C

        # Ma trận dự đoán trạng thái CoM
        Phi = self.Phi
        Gamma = self.Gamma

        # Ma trận dự đoán ZMP
        self.Z_x = torch.einsum("i,kij->kj", C, Phi)
        self.Z_u = torch.einsum("i,kij->kj", C, Gamma)

        # Ma trận dự đoán vận tốc CoM
        self.V_x = Phi[:, 1, :].clone()
        self.V_u = Gamma[:, 1, :].clone()
