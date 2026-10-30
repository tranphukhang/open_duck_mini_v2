import mujoco
import numpy as np


def get_site_transform(
    model,
    data,
    leg_name,
    q_leg,
    target_site,
    reference_site,
):
    """
    Tính ma trận biến đổi thuần nhất của target_site
    biểu diễn trong hệ tọa độ reference_site bằng MuJoCo.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo data.

    leg_name : str
        Chân cần thiết lập góc khớp:
        "left", "right", "L", hoặc "R".

    q_leg : array-like, shape (5,)
        Góc 5 khớp của chân theo thứ tự:

        [hip_yaw,
         hip_roll,
         hip_pitch,
         knee,
         ankle]

        Đơn vị: rad.

    target_site : str
        Site cần tính pose.
        Ví dụ: "L6", "L5_ankle", "R6".

    reference_site : str
        Site dùng làm hệ tọa độ tham chiếu.
        Ví dụ: "base_link", "L1_hip_yaw", "R1_hip_yaw".

    Returns
    -------
    T_ref_target : np.ndarray, shape (4,4)
        Ma trận biến đổi thuần nhất:

            ^ref T_target

        với dạng:

            [ R  p ]
            [ 0  1 ]
    """

    # ---------------------------------------------------------
    # 1. Xác định các joint thuộc chân được chọn
    # ---------------------------------------------------------
    leg = leg_name.lower()

    if leg in ["left", "l"]:
        joint_names = [
            "L_hip_yaw",
            "L_hip_roll",
            "L_hip_pitch",
            "L_knee",
            "L_ankle",
        ]

    elif leg in ["right", "r"]:
        joint_names = [
            "R_hip_yaw",
            "R_hip_roll",
            "R_hip_pitch",
            "R_knee",
            "R_ankle",
        ]

    else:
        raise ValueError(
            "leg_name phải là 'left', 'right', 'L' hoặc 'R'."
        )

    # ---------------------------------------------------------
    # 2. Kiểm tra vector góc khớp
    # ---------------------------------------------------------
    q_leg = np.asarray(q_leg, dtype=float).reshape(-1)

    if q_leg.size != 5:
        raise ValueError(
            f"q_leg phải chứa đúng 5 góc khớp, hiện tại nhận {q_leg.size}."
        )

    # ---------------------------------------------------------
    # 3. Gán góc khớp vào qpos
    # ---------------------------------------------------------
    for joint_name, q in zip(joint_names, q_leg):

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id == -1:
            raise ValueError(
                f"Không tìm thấy joint '{joint_name}' trong MuJoCo model."
            )

        # Địa chỉ của joint trong qpos
        qpos_adr = model.jnt_qposadr[joint_id]

        data.qpos[qpos_adr] = q

    # ---------------------------------------------------------
    # 4. Chạy forward kinematics của MuJoCo
    # ---------------------------------------------------------
    mujoco.mj_forward(model, data)

    # ---------------------------------------------------------
    # 5. Tìm ID của target site và reference site
    # ---------------------------------------------------------
    target_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        target_site,
    )

    reference_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        reference_site,
    )

    if target_id == -1:
        raise ValueError(
            f"Không tìm thấy target site '{target_site}'."
        )

    if reference_id == -1:
        raise ValueError(
            f"Không tìm thấy reference site '{reference_site}'."
        )

    # ---------------------------------------------------------
    # 6. Pose của hai site trong World frame
    # ---------------------------------------------------------

    # Position
    p_W_target = data.site_xpos[target_id].copy()
    p_W_ref = data.site_xpos[reference_id].copy()

    # Rotation matrix
    R_W_target = data.site_xmat[target_id].reshape(3, 3).copy()
    R_W_ref = data.site_xmat[reference_id].reshape(3, 3).copy()

    # ---------------------------------------------------------
    # 7. Xây dựng homogeneous transformation
    # ---------------------------------------------------------
    T_W_target = np.eye(4)
    T_W_target[:3, :3] = R_W_target
    T_W_target[:3, 3] = p_W_target

    T_W_ref = np.eye(4)
    T_W_ref[:3, :3] = R_W_ref
    T_W_ref[:3, 3] = p_W_ref

    # ---------------------------------------------------------
    # 8. Tính:
    #
    #       ^ref T_target
    #
    # =     (^W T_ref)^(-1)  *  ^W T_target
    #
    # ---------------------------------------------------------
    T_ref_target = np.linalg.inv(T_W_ref) @ T_W_target

    return T_ref_target