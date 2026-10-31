import numpy as np


# ============================================================
# QP PARAMETERS
# ============================================================


# ============================================================
# STATE COST WEIGHT
#
# State:
#
# x =
# [
#     px, py, pz,
#     roll, pitch, yaw,
#     vx, vy, vz,
#     wx, wy, wz,
#     1
# ]
#
# Q được lấy theo bộ tham số đã dùng trong mô hình
# SRBD-MPC trước đó.
#
# Phần tử thứ 13 tương ứng với augmented constant state.
# Đây không phải trạng thái vật lý cần tracking,
# do đó đặt trọng số bằng 0.
# ============================================================

Q = np.diag([
    150.0,   # px
    150.0,   # py
    250.0,   # pz

    100.0,   # roll
    100.0,   # pitch
    250.0,   # yaw

    1.0,     # vx
    1.0,     # vy
    1.0,     # vz

    10.0,    # wx
    10.0,    # wy
    1.0,     # wz

    0.0      # augmented constant
])


# ============================================================
# CONTROL COST WEIGHT
#
# Input:
#
# u =
# [
#     F_Lx, F_Ly, F_Lz,
#     M_Lx, M_Ly, M_Lz,
#     F_Rx, F_Ry, F_Rz,
#     M_Rx, M_Ry, M_Rz
# ]
#
# Force weights  = 1e-4
# Moment weights = 1e-3
# ============================================================

R = np.diag([
    1e-4, 1e-4, 1e-4,   # Left foot force
    1e-3, 1e-3, 1e-3,   # Left foot moment

    1e-4, 1e-4, 1e-4,   # Right foot force
    1e-3, 1e-3, 1e-3    # Right foot moment
])


# ============================================================
# CONTACT PARAMETERS
# ============================================================

# Friction coefficient
mu = 0.6


# Maximum vertical ground reaction force per foot [N]
#
# Robot mass:
#
#     m ≈ 1.5 kg
#
# Robot weight:
#
#     mg ≈ 14.7 N
#
# Fmax = 30 N gives approximately:
#
#     Fmax ≈ 2 * mg
#
# which provides sufficient margin for the initial
# standing and locomotion tests.
# ============================================================

Fmax = 30.0


# ============================================================
# FOOT GEOMETRY
#
# Wrench reference point:
#
#     L6 / R6
#
# l_t:
# distance from point 6 to toe
#
# l_h:
# distance from point 6 to heel
# ============================================================

# Point 6 -> toe [m]
l_t = 0.056


# Point 6 -> heel [m]
l_h = 0.027


# ============================================================
# MPC PARAMETERS
# ============================================================

# Prediction horizon
N = 10


# MPC discretization timestep [s]
dt_mpc = 0.025