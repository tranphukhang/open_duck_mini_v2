import numpy as np


# ============================================================
# SWING LEG JOINT PD PARAMETERS
#
# Joint order:
#
#   0. hip yaw
#   1. hip roll
#   2. hip pitch
#   3. knee
#   4. ankle
#
# Final gains selected by automatic L/R dynamic sweep.
# ============================================================


KP_SWING = np.array(
    [
        14.0,
        14.0,
        40.0,
        38.0,
        20.0,
    ],
    dtype=float,
)


KD_SWING = np.array(
    [
        0.15,
        0.15,
        0.65,
        0.65,
        0.30,
    ],
    dtype=float,
)