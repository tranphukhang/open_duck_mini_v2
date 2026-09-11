"""
step_timing_adaptation/dcm.py
==============================
Công thức DCM / ZMP / CoP cho LIPM.

- DCM:           xi = x_com + v_com / omega
- ZMP (LIPM):    p_zmp = x_com - (h/g) * a_com
- CoP (contact): p_cop từ wrench của contact foot-ground (chỉ geom bàn chân)

Không phụ thuộc WBC / planner — test standalone được.
"""

import numpy as np
import mujoco


class DCMCalculator:
    def __init__(self, com_height: float, gravity: float = 9.81):
        self.com_height = float(com_height)
        self.gravity = float(gravity)
        self.omega = float(np.sqrt(self.gravity / self.com_height))

    # ========================================================
    # DCM
    # ========================================================
    def compute_dcm_xy(self, com_xy, com_vel_xy):
        """xi = x + v/omega. Nhận/trả np.array shape (2,)."""
        com_xy = np.asarray(com_xy, dtype=float)
        com_vel_xy = np.asarray(com_vel_xy, dtype=float)
        return com_xy + com_vel_xy / self.omega

    # ========================================================
    # ZMP từ LIPM (dùng khi chưa có contact force)
    # ========================================================
    def compute_zmp_from_lipm(self, com_xy, com_acc_xy):
        """p_zmp = x_com - (h/g) * a_com."""
        com_xy = np.asarray(com_xy, dtype=float)
        com_acc_xy = np.asarray(com_acc_xy, dtype=float)
        return com_xy - (self.com_height / self.gravity) * com_acc_xy

    # ========================================================
    # CoP từ MuJoCo contacts (chỉ foot-ground)
    # ========================================================
    def compute_cop_xy(self, mj_model, mj_data,
                       foot_geom_ids, floor_geom_id):
        """
        Tính CoP từ các contact giữa chân và sàn.

        Parameters
        ----------
        foot_geom_ids : list[int]  — IDs của các geom bàn chân.
        floor_geom_id : int        — ID của geom sàn.

        Returns
        -------
        np.array (2,) hoặc None nếu không có contact foot-floor.
        """
        foot_set = {int(g) for g in foot_geom_ids}
        floor_id = int(floor_geom_id)

        F_total = np.zeros(3)
        M_o_total = np.zeros(3)
        n_contacts = 0

        wrench_local = np.zeros(6)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            is_foot_floor = (
                (g1 == floor_id and g2 in foot_set) or
                (g2 == floor_id and g1 in foot_set)
            )
            if not is_foot_floor:
                continue

            n_contacts += 1

            mujoco.mj_contactForce(mj_model, mj_data, i, wrench_local)
            frame = np.array(c.frame, dtype=float).reshape(3, 3)
            # Trong MuJoCo, contact.frame có các HÀNG là trục contact
            # biểu diễn trong world (row 0 = normal, row 1 = tangent1, row 2 = tangent2).
            # Do đó vector viết trong contact frame chuyển sang world bằng frame.T @ v.
            f_world = frame.T @ wrench_local[:3]
            m_world = frame.T @ wrench_local[3:]

            p = np.array(c.pos, dtype=float)

            F_total += f_world
            M_o_total += m_world + np.cross(p, f_world)

        if n_contacts == 0:
            return None
        if abs(F_total[2]) < 1e-6:
            return None

        # CoP là điểm trên mặt sàn (z = 0) mà moment ngang = 0:
        #   M_x + p_y * F_z = 0   -> p_y = -M_x / F_z
        #   M_y - p_x * F_z = 0   -> p_x =  M_y / F_z
        p_cop_x = M_o_total[1] / F_total[2]
        p_cop_y = -M_o_total[0] / F_total[2]

        return np.array([p_cop_x, p_cop_y], dtype=float)