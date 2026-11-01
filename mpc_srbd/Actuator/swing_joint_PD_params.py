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
#
# Tuned using isolated MuJoCo swing-joint tracking tests,
# including a 2.5 Hz coupled moving-reference test.
# ============================================================


# ============================================================
# PROPORTIONAL GAINS
# ============================================================

KP_SWING = np.array(
    [
        14.0,   # hip yaw
        14.0,   # hip roll
        16.0,   # hip pitch
        16.0,   # knee
        14.0,   # ankle
    ],
    dtype=float,
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
        0.15,   # ankle
    ],
    dtype=float,
)
