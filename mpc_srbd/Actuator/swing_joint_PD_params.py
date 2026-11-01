import numpy as np


# ============================================================
# SWING LEG JOINT PD PARAMETERS
#
# Joint order:
#
#     0. hip yaw
#     1. hip roll
#     2. hip pitch
#     3. knee
#     4. ankle
# ============================================================


# ============================================================
# PROPORTIONAL GAINS
# ============================================================

KP_SWING = np.array(
    [
        8.0,    # hip yaw
        8.0,    # hip roll
        12.0,   # hip pitch
        12.0,   # knee
        8.0     # ankle
    ],
    dtype=float
)


# ============================================================
# DERIVATIVE GAINS
# ============================================================

KD_SWING = np.array(
    [
        0.15,   # hip yaw
        0.15,   # hip roll
        0.25,   # hip pitch
        0.25,   # knee
        0.15    # ankle
    ],
    dtype=float
)