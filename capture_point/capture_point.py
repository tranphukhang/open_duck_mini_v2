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


    # Tính độ hiệu chỉnh vị trí đặt chân từ sai số DCM
    def compute_correction(self, dcm_error):
        # dcm_error: (B, 2) - Sai số DCM theo hai trục X/Y

        # Tính độ hiệu chỉnh dựa trên gain Capture Point
        correction_raw = self.step_gain * dcm_error

        # Giới hạn độ hiệu chỉnh theo từng trục X/Y
        correction = torch.clamp(correction_raw, min=-self.step_limit, max=self.step_limit)

        return correction

    
    # Hiệu chỉnh vị trí tiếp đất và giới hạn bề rộng bước chân
    def correct_landing(self, nominal_landing, correction, stance, stance_y):
        # nominal_landing: (B, 3) - Vị trí tiếp đất danh định
        # correction:      (B, 2) - Độ hiệu chỉnh CP theo X/Y
        # stance:          (B,)   - Chân stance (0: trái, 1: phải)
        # stance_y:        (B,)   - Tọa độ Y của chân stance

        # Tạo bản sao để không thay đổi nominal_landing
        corrected_landing = nominal_landing.clone()

        # Cộng độ hiệu chỉnh CP theo X/Y
        corrected_landing[:, :2] += correction

        # Giới hạn bề rộng bước chân theo từng chân stance
        landing_y = torch.where(stance == 0,
            torch.minimum(
                corrected_landing[:, 1],
                stance_y - self.min_step_width
            ),
            torch.maximum(
                corrected_landing[:, 1],
                stance_y + self.min_step_width
            )
        )

        # Cập nhật tọa độ Y sau khi giới hạn
        corrected_landing[:, 1] = landing_y

        return corrected_landing



