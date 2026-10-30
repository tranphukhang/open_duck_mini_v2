import mujoco
import numpy as np

from scipy.spatial.transform import Rotation


def get_W_euler_base(
    model,
    data,
    site_name="base"
):
    """
    Lấy Euler angles của base site so với World frame.

    Euler convention:
        [roll, pitch, yaw]
        = [phi, theta, psi]

    Đơn vị:
        rad

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data hiện tại.

    site_name : str, optional
        Site đại diện cho base frame.
        Mặc định: "base".

    Returns
    -------
    W_euler_base : np.ndarray, shape (3,)
        Euler angles của base so với World frame:

        [roll, pitch, yaw]

        Đơn vị: [rad]
    """

    # ---------------------------------------------------------
    # 1. Tìm site ID
    # ---------------------------------------------------------
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    if site_id == -1:
        raise ValueError(
            f"Không tìm thấy site '{site_name}' trong MuJoCo model."
        )

    # ---------------------------------------------------------
    # 2. Rotation matrix:
    #
    #       ^W R_base
    #
    # data.site_xmat lưu orientation của site
    # trong World frame.
    # ---------------------------------------------------------
    W_R_base = (
        data.site_xmat[site_id]
        .reshape(3, 3)
        .copy()
    )

    # ---------------------------------------------------------
    # 3. Rotation matrix -> Euler angles
    #
    # Convention:
    #
    #       R = Rz(yaw) Ry(pitch) Rx(roll)
    #
    # Output:
    #
    #       [roll, pitch, yaw]
    #
    # Đơn vị: rad
    # ---------------------------------------------------------
    W_euler_base = (
        Rotation
        .from_matrix(W_R_base)
        .as_euler("xyz", degrees=False)
    )

    return np.asarray(
        W_euler_base,
        dtype=float
    )


def get_Ew_inverse(
    W_euler_base,
    singularity_tolerance=1e-6
):
    """
    Tính ma trận Ew^(-1) dùng để ánh xạ angular velocity
    biểu diễn trong World frame sang Euler angle rates.

    Quan hệ:

        euler_dot = Ew^(-1) @ W_angular_velocity_base

    với:

        euler =
            [roll, pitch, yaw]
            [phi, theta, psi]

    và:

        W_angular_velocity_base =
            [wx, wy, wz]

    Parameters
    ----------
    W_euler_base : np.ndarray, shape (3,)
        Euler angles:

        [roll, pitch, yaw]

        Đơn vị: [rad]

    singularity_tolerance : float, optional
        Ngưỡng kiểm tra singularity tại:

            cos(pitch) = 0

        Mặc định: 1e-6.

    Returns
    -------
    Ew_inverse : np.ndarray, shape (3, 3)
        Ma trận biến đổi từ angular velocity trong World frame
        sang Euler angle rates.
    """

    # ---------------------------------------------------------
    # 1. Lấy Euler angles
    # ---------------------------------------------------------
    roll, pitch, yaw = W_euler_base

    # ---------------------------------------------------------
    # 2. Các giá trị lượng giác cần thiết
    # ---------------------------------------------------------
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # ---------------------------------------------------------
    # 3. Kiểm tra Euler singularity
    #
    # pitch = +/- pi/2
    # -> cos(pitch) = 0
    # ---------------------------------------------------------
    if abs(cos_pitch) < singularity_tolerance:
        raise ValueError(
            "Euler angle singularity: "
            "pitch đang quá gần +/- 90 deg, "
            "không thể tính Ew_inverse ổn định."
        )

    # ---------------------------------------------------------
    # 4. Tính Ew^(-1)
    #
    # euler_dot =
    #
    # [ phi_dot   ]
    # [ theta_dot ] = Ew_inverse * W_omega
    # [ psi_dot   ]
    #
    # ---------------------------------------------------------
    Ew_inverse = np.array([
        [
            cos_yaw / cos_pitch,
            sin_yaw / cos_pitch,
            0.0
        ],
        [
            -sin_yaw,
            cos_yaw,
            0.0
        ],
        [
            cos_yaw * sin_pitch / cos_pitch,
            sin_yaw * sin_pitch / cos_pitch,
            1.0
        ]
    ], dtype=float)

    return Ew_inverse