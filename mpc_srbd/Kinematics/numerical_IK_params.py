# ============================================================
# NUMERICAL IK PARAMETERS
#
# Ankle-center IK:
#   site: L5_ankle / R5_ankle
#   constraint: q_hip_yaw = HIP_YAW_FIXED
#   solved variables: [hip_roll, hip_pitch, knee]
#   solver output: [hip_yaw, hip_roll, hip_pitch, knee]
# ============================================================

HIP_YAW_FIXED = 0.0

# Hybrid switching threshold.
SIGMA_THRESHOLD = 1e-3

# Pseudo-Newton step scale.
ALPHA_NEWTON = 0.5

# Jacobian-transpose gradient-descent step size.
ETA_GRADIENT = 1.0

# Cartesian ankle-position tolerance [m].
IK_TOLERANCE = 1e-4

# Maximum IK iterations.
IK_MAX_ITERATIONS = 100
