class SwingTrajectoryGenerator:
    def __init__(self, period_phase=0.4, double_support=0.1, swing_height=0.02):
        # Chu kỳ gồm hai bước chân
        self.period_phase = period_phase

        # Tỷ lệ thời gian Double Support
        self.double_support = double_support

        # Độ cao nâng chân tối đa
        self.swing_height = swing_height

        # Thời gian của một bước
        self.step_duration = self.period_phase / 2.0

        # Thời gian Double Support
        self.ds_duration = self.double_support * self.period_phase

        # Thời gian thực hiện Swing
        self.swing_duration = self.step_duration - self.ds_duration

        # Hàm nội suy trơn bậc ba cho quỹ đạo chân swing
    @staticmethod
    def smoothstep(s):
        # Giới hạn thời gian chuẩn hóa trong khoảng [0, 1]
        s = s.clamp(0.0, 1.0)

        # Tính hệ số nội suy bậc ba
        sigma = 3.0 * s**2 - 2.0 * s**3

        return sigma