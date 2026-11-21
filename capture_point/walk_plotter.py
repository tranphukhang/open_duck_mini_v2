"""Ghi dữ liệu trong RAM và hiển thị đồ thị đi bộ Open Duck Mini V2.

Sử dụng cho cả run_walk_load_on.py và run_walk_load_off.py.
KHÔNG ghi CSV, PNG hay bất kỳ tệp kết quả nào.

Đồng bộ thời điểm:
    out = controller.step(data, cmd)  # tham chiếu ở t[k+1]
    data.ctrl[...] = out.q_cmd[0]
    for ...: mujoco.mj_step(model, data)
    plotter.record(out, data, cmd)     # ground truth ở t[k+1]

Định nghĩa các đại lượng:
- CoM plan: out.com_des (tọa độ kế hoạch {P}).
- CoM actual: Ground Truth MuJoCo được quy đổi sang cùng tọa độ {P},
  dùng vị trí tương đối so với bàn chân đang hỗ trợ và vị trí chân kế hoạch;
  phương pháp này nhất quán với phép so sánh DCM của WalkController sau reanchor_x.
- z_CoM_plan: tọa độ z cố định dùng làm mục tiêu WBIK; chiều cao LIPM
  h = z_CoM ban đầu - độ cao trung bình của hai foot site ban đầu.
- velocity_x/y_ref: vận tốc lệnh của người dùng (vx, vy), không phải vận tốc MPC.
- velocity_x/y_actual: vận tốc CoM Ground Truth từ MuJoCo (không lọc).
- joint_ref_IK: q_ik trước bù tải, KHÔNG phải q_cmd đã bù tải.
- joint_actual: qpos của khớp trong MuJoCo sau bước mô phỏng.

Cần numpy, torch (qua controller), matplotlib chỉ khi hiển thị đồ thị.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


JOINT_NAMES = (
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
)


@dataclass
class WalkPlotData:
    """Dữ liệu lấy mẫu ở cùng thời điểm vật lý t[k+1]."""

    time: np.ndarray                 # (T,), s
    com_plan: np.ndarray             # (T,3), m trong {P}
    com_actual: np.ndarray           # (T,3), m trong {P}
    velocity_ref: np.ndarray         # (T,2), m/s: user command
    velocity_actual: np.ndarray      # (T,2), m/s: GT CoM
    joint_ref_ik: np.ndarray         # (T,10), rad
    joint_actual: np.ndarray         # (T,10), rad


class WalkPlotter:
    """Thu thập và vẽ dữ liệu một môi trường trong batch controller.

    Khởi tạo ngay sau controller.reset(). Trong vòng lặp gọi record()
    SAU các bước mj_step, không gọi trước; dữ liệu thực và tham chiếu phải
    được lấy tại cùng thời điểm t[k+1].
    """

    def __init__(self, controller: Any, env_index: int = 0, case_name: str = "LOAD ON"):
        if not controller.initialized:
            raise ValueError("Phải gọi controller.reset(...) trước WalkPlotter(...)" )
        if not 0 <= env_index < controller.n:
            raise ValueError("env_index vượt khỏi batch của controller")

        self.controller = controller
        self.env_index = int(env_index)
        self.case_name = str(case_name)
        model = controller.model
        self.qpos_addresses = np.array([
            int(model.jnt_qposadr[model.joint(j).id]) for j in JOINT_NAMES
        ], dtype=int)
        self.lipm_height = float(controller.lipm_height)
        self._samples: list[dict[str, np.ndarray | float]] = []
        self.figures = []

    @staticmethod
    def _to_numpy(value):
        """Sao chép Torch/NumPy trước khi controller thay đổi dữ liệu."""
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().copy()
        return np.asarray(value).copy()

    def record(self, output: Any, data: Any, velocity_cmd: Any) -> None:
        """Ghi một mẫu sau khi MuJoCo đã tiến đến t[k+1].

        output: WalkOutput vừa tạo từ WalkController.step().
        data: MjData; hoặc list MjData nếu controller chạy batch.
        velocity_cmd: [vx,vy] hoặc (B,2) [m/s].
        """
        ds = list(data) if isinstance(data, (tuple, list)) else [data]
        if len(ds) != self.controller.n:
            raise ValueError("Số lượng MjData khác num_envs của controller")
        i = self.env_index
        d = ds[i]
        cmd = np.asarray(self._to_numpy(velocity_cmd), dtype=np.float64)
        if cmd.shape == (2,):
            cmd_i = cmd
        elif cmd.shape == (self.controller.n, 2):
            cmd_i = cmd[i]
        else:
            raise ValueError("velocity_cmd phải có kích thước (2,) hoặc (B,2)")

        # Đọc Ground Truth sau physics: CoM, vận tốc CoM, hai foot sites.
        # estimator.update() gọi mj_subtreeVel để vận tốc CoM không bị stale.
        gt = self.controller.estimator.update(ds)
        if not bool(gt.valid[i].item()):
            raise FloatingPointError("Ground Truth không hữu hạn khi ghi đồ thị")

        # Đồng nhất hệ {P}: CoM thực được tính tương đối với chân hỗ trợ.
        # Không so sánh trực tiếp raw World x với plan x sau reanchor_x().
        # Cùng công thức với WalkController bước 3, nhưng dùng GT sau mj_step().
        j_now = int(output.j_now[i].item())
        planner = self.controller.planner
        feet_plan = planner.feet_start[i] if j_now >= 0 else planner.feet0[i]
        support = output.contact[i].to(dtype=feet_plan.dtype)
        weight_sum = float(support.sum().item())
        if weight_sum < 1:
            raise ValueError("Không có chân hỗ trợ theo Contact Scheduler")
        actual_in_plan = (
            ((feet_plan + gt.com_relative_to_feet[i]) * support[:, None]).sum(dim=0)
            / support.sum()
        )

        com_plan = np.asarray(self._to_numpy(output.com_des[i]), dtype=np.float64)
        com_actual = np.asarray(self._to_numpy(actual_in_plan), dtype=np.float64)
        vel_actual = np.asarray(self._to_numpy(gt.com_vel[i, :2]), dtype=np.float64)
        q_ik = np.asarray(self._to_numpy(output.q_ik[i]), dtype=np.float64)
        q_actual = np.asarray(d.qpos[self.qpos_addresses], dtype=np.float64).copy()

        if not all(np.isfinite(a).all() for a in (
            cmd_i, com_plan, com_actual, vel_actual, q_ik, q_actual,
        )):
            raise FloatingPointError("Giá trị không hữu hạn khi ghi dữ liệu đồ thị")

        self._samples.append({
            "time": float(d.time),
            "com_plan": com_plan,
            "com_actual": com_actual,
            "velocity_ref": cmd_i.copy(),
            "velocity_actual": vel_actual,
            "joint_ref_ik": q_ik,
            "joint_actual": q_actual,
        })

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def get_data(self) -> WalkPlotData:
        """Xuất mảng NumPy từ dữ liệu RAM; không ghi tệp."""
        if not self._samples:
            raise RuntimeError("Chưa có dữ liệu; hãy gọi record() sau mj_step().")
        s = self._samples
        return WalkPlotData(
            time=np.asarray([x["time"] for x in s], dtype=np.float64),
            com_plan=np.stack([x["com_plan"] for x in s]),
            com_actual=np.stack([x["com_actual"] for x in s]),
            velocity_ref=np.stack([x["velocity_ref"] for x in s]),
            velocity_actual=np.stack([x["velocity_actual"] for x in s]),
            joint_ref_ik=np.stack([x["joint_ref_ik"] for x in s]),
            joint_actual=np.stack([x["joint_actual"] for x in s]),
        )

    @staticmethod
    def _plot_pair(ax, t, ref, actual, title, unit, ref_label, actual_label):
        ax.plot(t, ref, linewidth=1.8, label=ref_label)
        ax.plot(t, actual, linewidth=1.3, label=actual_label, alpha=0.88)
        ax.set(title=title, xlabel="Thời gian (s)", ylabel=unit)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    def make_figures(self):
        """Tạo hai figure, không show và không lưu ảnh.

        Figure 1: 5 trục: x/y/z CoM, vx/vy.
        Figure 2: 10 khớp trên 10 ô riêng, mỗi ô gồm IK và actual.
        """
        import matplotlib.pyplot as plt

        a = self.get_data()
        t = a.time

        fig1, axes1 = plt.subplots(3, 2, figsize=(13.5, 10), sharex=True, constrained_layout=True)
        fig1.suptitle(
            f"{self.case_name} | CoM và vận tốc | h_LIPM = {self.lipm_height:.4f} m",
            fontweight="bold",
        )
        self._plot_pair(axes1[0, 0], t, a.com_plan[:, 0], a.com_actual[:, 0],
                        "x_CoM: kế hoạch / thực tế", "x (m)", "x_CoM_plan", "x_CoM_actual")
        self._plot_pair(axes1[0, 1], t, a.com_plan[:, 1], a.com_actual[:, 1],
                        "y_CoM: kế hoạch / thực tế", "y (m)", "y_CoM_plan", "y_CoM_actual")
        self._plot_pair(axes1[1, 0], t, a.com_plan[:, 2], a.com_actual[:, 2],
                        "z_CoM: cao độ đặt / thực tế", "z (m)", "z_CoM_plan", "z_CoM_actual")
        self._plot_pair(axes1[1, 1], t, a.velocity_ref[:, 0], a.velocity_actual[:, 0],
                        "Velocity X: lệnh / CoM thực tế", "vx (m/s)", "velocity_x_ref", "velocity_x_actual")
        self._plot_pair(axes1[2, 0], t, a.velocity_ref[:, 1], a.velocity_actual[:, 1],
                        "Velocity Y: lệnh / CoM thực tế", "vy (m/s)", "velocity_y_ref", "velocity_y_actual")
        axes1[2, 1].axis("off")

        fig2, axes2 = plt.subplots(5, 2, figsize=(14, 13.5), sharex=True, constrained_layout=True)
        fig2.suptitle(
            f"{self.case_name} | Góc khớp: IK reference / MuJoCo actual\n"
            "Lưu ý: Load ON gửi q_cmd = q_IK + delta_q_load, không phải q_IK",
            fontweight="bold",
        )
        q_ik_deg = np.rad2deg(a.joint_ref_ik)
        q_actual_deg = np.rad2deg(a.joint_actual)
        for j, name in enumerate(JOINT_NAMES):
            ax = axes2.flat[j]
            self._plot_pair(ax, t, q_ik_deg[:, j], q_actual_deg[:, j],
                            name, "Góc (deg)", "joint_ref_IK", "joint_actual")

        self.figures = [fig1, fig2]
        return self.figures

    def show(self, block: bool = True):
        """Hiện hai cửa sổ Matplotlib. block=False nếu viewer vẫn đang mở."""
        import matplotlib.pyplot as plt

        self.make_figures()
        plt.show(block=block)
        if not block:
            # Khởi động event loop để cửa sổ đồ thị xuất hiện và render.
            plt.pause(0.1)
        return self.figures
