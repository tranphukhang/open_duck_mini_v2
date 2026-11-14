import torch


class CapturePoint:
    def __init__(self, com_height, step_gain=(0.4, 0.3), step_limit=(0.04, 0.03), min_step_width=0.12, device="cpu"):
        # Thiết bị tính toán
        self.device = torch.device(device)

        # Chiều cao CoM danh định
        self.com_height = com_height

        # Gia tốc trọng trường
        self.g = 9.81

        # Tần số tự nhiên của mô hình LIPM
        self.omega = (self.g / self.com_height) ** 0.5

        # Gain hiệu chỉnh vị trí đặt chân theo X/Y
        self.step_gain = torch.tensor(step_gain, dtype=torch.float32, device=self.device)

        # Giới hạn hiệu chỉnh vị trí đặt chân theo X/Y
        self.step_limit = torch.tensor(step_limit, dtype=torch.float32, device=self.device)

        # Bề rộng bước chân tối thiểu
        self.min_step_width = min_step_width

    
    # Tính sai số DCM giữa Ground Truth và LIPM-MPC
    def compute_dcm_error(self, com_meas, vcom_meas, com_state):
        # com_meas:   (B, 3)    - Vị trí CoM Ground Truth
        # vcom_meas:  (B, 3)    - Vận tốc CoM Ground Truth
        # com_state:  (B, 2, 3) - Trạng thái CoM kế hoạch từ MPC

        # Tần số tự nhiên của mô hình LIPM
        omega = self.omega

        # DCM thực tế từ Ground Truth
        dcm_meas = (com_meas[:, :2] + vcom_meas[:, :2] / omega)

        # DCM kế hoạch từ trạng thái LIPM-MPC
        dcm_plan = (com_state[:, :, 0] + com_state[:, :, 1] / omega)

        # Sai số DCM (thực tế - kế hoạch)
        dcm_error = dcm_meas - dcm_plan

        return dcm_meas, dcm_plan, dcm_error

