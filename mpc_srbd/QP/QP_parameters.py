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
# ============================================================

Q = np.diag([
    1,# 100.0,   # px
    1,# 100.0,   # py
    5,# 500.0,   # pz

    1,# 150.0,   # roll
    1,# 150.0,   # pitch
    2,# 250.0,   # yaw

    1,# 10.0,    # vx
    1,# 10.0,    # vy
    1,# 10.0,     # vz

    1,# 1.0,     # wx
    1,# 1.0,     # wy
    2,# 5.0,     # wz

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
#
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
#     m ~= 1.5 kg
#
# Robot weight:
#
#     mg ~= 14.7 N
#
# Fmax = 30 N gives sufficient margin for the current
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
l_t = 0.064


# Point 6 -> heel [m]
l_h = 0.038


# ============================================================
# LINE-CONTACT ROLL-MOMENT RELAXATION
#
# Ideal line-contact model:
#
#     Mx_foot = 0
#
# Practical implementation:
#
#     -Mx_tolerance
#     <=
#     Mx_foot
#     <=
#     Mx_tolerance
#
# The physical foot has approximately 0.02 m lateral
# distance from the L6/R6 contact line to each side.
#
# 0.01 N.m is therefore used as a small symmetric
# relaxation while preserving the line-contact approximation.
# ============================================================

Mx_tolerance = 0.17   # [N.m]


# ============================================================
# OPTIONAL TOE / HEEL LATERAL-FRICTION CONSTRAINTS
#
# False:
#
#     Baseline mode.
#
#     Keep only:
#
#         1. relaxed Mx line-contact constraint
#         2. toe CoP constraint
#         3. heel CoP constraint
#
#     Together with force_constraint.py:
#
#         5 force constraints / foot
#         3 moment constraints / foot
#
#     -> 8 constraints / foot
#
#     This is closer to the HECTOR paper-code formulation.
#
#
# True:
#
#     Extended mode.
#
#     Additionally enforce separated toe/heel lateral
#     friction constraints:
#
#         |f_ty| <= mu * f_tz
#
#         |f_hy| <= mu * f_hz
#
#     This adds 4 constraints / foot and produces a more
#     restrictive admissible contact-wrench set.
#
#     The formulas remain implemented in moment_constraint.py
#     and can therefore be enabled again without rewriting code.
# ============================================================

enable_toe_heel_lateral_friction = False


# ============================================================
# MPC PARAMETERS
# ============================================================

N = 10


# Prediction-model discretization step
dt_mpc = 0.04   # [s]


# MPC controller update period
#
# 0.005 s -> 200 Hz
dt_mpc_update = 0.005   # [s]