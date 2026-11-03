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
# The first 12 weights are mapped from the HECTOR
# SRBD-MPC paper-code ordering into the state ordering
# used in this project.
#
# The 13th state is the augmented constant state.
# It is not a physical tracking state, therefore its
# tracking weight is zero.
# ============================================================

Q = np.diag([
    100.0,   # px
    100.0,   # py
    500.0,   # pz

    150.0,   # roll
    150.0,   # pitch
    250.0,   # yaw

    10.0,    # vx
    10.0,    # vy
    1.0,     # vz

    1.0,     # wx
    1.0,     # wy
    5.0,     # wz

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
# Force weights:
#
#     1e-5
#
# Moment weights:
#
#     1e-4
#
# These values are mapped from the HECTOR paper-code
# input ordering into the input ordering of this project.
# ============================================================

R = np.diag([
    1e-5, 1e-5, 1e-5,   # Left foot force
    1e-4, 1e-4, 1e-4,   # Left foot moment

    1e-5, 1e-5, 1e-5,   # Right foot force
    1e-4, 1e-4, 1e-4    # Right foot moment
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
#     distance from point 6 to toe
#
# l_h:
#     distance from point 6 to heel
# ============================================================

# Point 6 -> toe [m]
l_t = 0.056


# Point 6 -> heel [m]
l_h = 0.027


# ============================================================
# LINE-CONTACT ROLL-MOMENT RELAXATION
#
# Ideal line-contact assumption:
#
#     Mx_foot = 0
#
# The real foot has a finite width.
#
# The L6 / R6 contact line is approximately centered in
# the foot, with about 0.02 m available on each lateral side.
#
# Instead of imposing the exact equality Mx = 0, use a
# small symmetric relaxation:
#
#     -Mx_tolerance <= Mx_foot <= Mx_tolerance
#
# This still preserves the line-contact approximation while
# avoiding an unnecessarily hard numerical equality.
# ============================================================

Mx_tolerance = 0.01   # [N.m]


# ============================================================
# MPC PARAMETERS
# ============================================================

N = 10


# Prediction-model discretization step
dt_mpc = 0.05   # [s]


# MPC controller update period
#
# 0.005 s -> 200 Hz MPC update rate
dt_mpc_update = 0.005   # [s]