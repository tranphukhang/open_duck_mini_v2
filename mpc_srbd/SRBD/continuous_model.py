import numpy as np


def skew_symmetric(vector):
    """
    Tạo ma trận skew-symmetric [r]_x sao cho:

        [r]_x @ F = r x F

    Parameters
    ----------
    vector : array_like, shape (3,)
        Vector 3D.

    Returns
    -------
    skew : np.ndarray, shape (3, 3)
        Ma trận skew-symmetric.
    """

    vector = np.asarray(
        vector,
        dtype=float
    ).reshape(3)

    x, y, z = vector

    skew = np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0]
    ], dtype=float)

    return skew


def build_continuous_model(
    Ew_inverse,
    mass_CoM,
    W_inertia_CoM,
    W_r_L,
    W_r_R,
    gravity=9.81
):
    """
    Xây dựng mô hình SRBD liên tục:

        x_dot = Ac @ x + Bc @ u

    State:

        x = [
            W_position_CoM,         # 3
            W_euler_base,           # 3
            W_linear_velocity_CoM,  # 3
            W_angular_velocity_base,# 3
            1                       # 1
        ]

    Input:

        u = [
            W_F_L,  # 3
            W_M_L,  # 3
            W_F_R,  # 3
            W_M_R   # 3
        ]

    Parameters
    ----------
    Ew_inverse : np.ndarray, shape (3, 3)
        Ma trận biến đổi:

            euler_dot = Ew_inverse @ W_omega_base

    mass_CoM : float
        Tổng khối lượng robot [kg].

    W_inertia_CoM : np.ndarray, shape (3, 3)
        Whole-body inertia tensor tại CoM,
        biểu diễn trong World frame [kg.m^2].

        Trong MPC hiện tại, giá trị này được tính tại
        tư thế khởi tạo rồi freeze.

    W_r_L : np.ndarray, shape (3,)
        Moment arm:

            W_r_L = W_p_L6 - W_p_CoM

    W_r_R : np.ndarray, shape (3,)
        Moment arm:

            W_r_R = W_p_R6 - W_p_CoM

    gravity : float, optional
        Gia tốc trọng trường [m/s^2].
        Mặc định: 9.81.

    Returns
    -------
    Ac : np.ndarray, shape (13, 13)
        Ma trận trạng thái liên tục.

    Bc : np.ndarray, shape (13, 12)
        Ma trận đầu vào liên tục.
    """

    # ========================================================
    # Input conversion / validation
    # ========================================================

    Ew_inverse = np.asarray(
        Ew_inverse,
        dtype=float
    ).reshape(3, 3)

    W_inertia_CoM = np.asarray(
        W_inertia_CoM,
        dtype=float
    ).reshape(3, 3)

    W_r_L = np.asarray(
        W_r_L,
        dtype=float
    ).reshape(3)

    W_r_R = np.asarray(
        W_r_R,
        dtype=float
    ).reshape(3)

    mass_CoM = float(mass_CoM)

    if mass_CoM <= 0.0:
        raise ValueError(
            "mass_CoM phải lớn hơn 0."
        )

    # ========================================================
    # Constants
    # ========================================================

    I3 = np.eye(3)

    W_gravity = np.array([
        0.0,
        0.0,
        -gravity
    ], dtype=float)

    # Inertia inverse
    W_inertia_CoM_inverse = np.linalg.inv(
        W_inertia_CoM
    )

    # Moment-arm skew matrices
    W_r_L_skew = skew_symmetric(
        W_r_L
    )

    W_r_R_skew = skew_symmetric(
        W_r_R
    )

    # ========================================================
    # Continuous A matrix
    #
    # State order:
    #
    # 0:3   -> position CoM
    # 3:6   -> Euler base
    # 6:9   -> linear velocity CoM
    # 9:12  -> angular velocity base
    # 12    -> constant 1
    # ========================================================

    Ac = np.zeros(
        (13, 13),
        dtype=float
    )

    # p_dot = v
    Ac[0:3, 6:9] = I3

    # euler_dot = Ew_inverse * omega
    Ac[3:6, 9:12] = Ew_inverse

    # v_dot = ... + gravity * 1
    Ac[6:9, 12] = W_gravity

    # Last augmented state:
    #
    # d(1)/dt = 0
    #
    # Ac[12, :] đã bằng 0.


    # ========================================================
    # Continuous B matrix
    #
    # Input order:
    #
    # 0:3   -> W_F_L
    # 3:6   -> W_M_L
    # 6:9   -> W_F_R
    # 9:12  -> W_M_R
    # ========================================================

    Bc = np.zeros(
        (13, 12),
        dtype=float
    )

    # --------------------------------------------------------
    # Linear dynamics:
    #
    # v_dot =
    #     (1/m) F_L
    #   + (1/m) F_R
    #   + gravity
    # --------------------------------------------------------

    Bc[6:9, 0:3] = I3 / mass_CoM

    Bc[6:9, 6:9] = I3 / mass_CoM


    # --------------------------------------------------------
    # Angular dynamics:
    #
    # omega_dot =
    #
    # I_CoM^-1 (
    #       r_L x F_L
    #     + M_L
    #     + r_R x F_R
    #     + M_R
    # )
    #
    # with:
    #
    # r x F = [r]_x F
    # --------------------------------------------------------

    # Left force
    Bc[9:12, 0:3] = (
        W_inertia_CoM_inverse
        @ W_r_L_skew
    )

    # Left moment
    Bc[9:12, 3:6] = (
        W_inertia_CoM_inverse
    )

    # Right force
    Bc[9:12, 6:9] = (
        W_inertia_CoM_inverse
        @ W_r_R_skew
    )

    # Right moment
    Bc[9:12, 9:12] = (
        W_inertia_CoM_inverse
    )

    return Ac, Bc