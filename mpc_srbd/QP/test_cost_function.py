from pathlib import Path
import sys

import mujoco
import numpy as np


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_DIR = PROJECT_ROOT / "mpc_srbd" / "State"
SRBD_DIR = PROJECT_ROOT / "mpc_srbd" / "SRBD"

if str(STATE_DIR) not in sys.path:
    sys.path.append(str(STATE_DIR))

if str(SRBD_DIR) not in sys.path:
    sys.path.append(str(SRBD_DIR))


# ============================================================
# Imports
# ============================================================

from W_position_CoM import get_W_position_CoM

from W_linear_velocity_CoM import (
    get_W_linear_velocity_CoM
)

from W_euler_base import (
    get_W_euler_base,
    get_Ew_inverse
)

from W_angular_velocity_base import (
    get_W_angular_velocity_base
)

from mass_CoM import get_mass_CoM

from W_inertia_CoM import get_W_inertia_CoM

from W_moment_arm import get_W_moment_arm

from continuous_model import build_continuous_model

from discretization import discretize_model

from prediction_model import build_prediction_model

from cost_function import build_cost_function


# ============================================================
# XML path
# ============================================================

XML_PATH = (
    PROJECT_ROOT
    / "xmls"
    / "scene_flat_terrian_v3.xml"
)

if not XML_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy MuJoCo XML:\n{XML_PATH}"
    )


print("Loading MuJoCo model:")
print(XML_PATH)


# ============================================================
# Load MuJoCo
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)


# ============================================================
# Update MuJoCo data
# ============================================================

mujoco.mj_forward(
    model,
    data
)

mujoco.mj_subtreeVel(
    model,
    data
)


# ============================================================
# Read current states
# ============================================================

W_position_CoM = get_W_position_CoM(
    model=model,
    data=data,
    root_body_name="base_link"
)

W_linear_velocity_CoM = (
    get_W_linear_velocity_CoM(
        model=model,
        data=data,
        root_body_name="base_link"
    )
)

W_euler_base = get_W_euler_base(
    model=model,
    data=data,
    site_name="base"
)

W_angular_velocity_base = (
    get_W_angular_velocity_base(
        model=model,
        data=data,
        site_name="base"
    )
)


# ============================================================
# Build current augmented state
#
# x =
# [
#   p_CoM
#   euler_base
#   v_CoM
#   omega_base
#   1
# ]
# ============================================================

x_current = np.concatenate([
    W_position_CoM,
    W_euler_base,
    W_linear_velocity_CoM,
    W_angular_velocity_base,
    np.array([1.0])
])


# ============================================================
# Read SRBD parameters
# ============================================================

Ew_inverse = get_Ew_inverse(
    W_euler_base
)

mass_CoM = get_mass_CoM(
    model=model,
    root_body_name="base_link"
)

W_inertia_CoM = get_W_inertia_CoM(
    model=model,
    data=data,
    root_body_name="base_link"
)

W_r_L, W_r_R = get_W_moment_arm(
    model=model,
    data=data,
    left_site_name="L6",
    right_site_name="R6",
    root_body_name="base_link"
)


# ============================================================
# Continuous model
# ============================================================

Ac, Bc = build_continuous_model(
    Ew_inverse=Ew_inverse,
    mass_CoM=mass_CoM,
    W_inertia_CoM=W_inertia_CoM,
    W_r_L=W_r_L,
    W_r_R=W_r_R
)


# ============================================================
# Discrete model
# ============================================================

Ts = 0.01

Ad, Bd = discretize_model(
    Ac=Ac,
    Bc=Bc,
    Ts=Ts
)


# ============================================================
# Prediction model
# ============================================================

N = 3

Phi, Gamma = build_prediction_model(
    Ad=Ad,
    Bd=Bd,
    N=N
)


# ============================================================
# Define Q
#
# State order:
#
# 0:3   position CoM
# 3:6   Euler base
# 6:9   linear velocity CoM
# 9:12  angular velocity base
# 12    augmented constant
#
# Weight cuối bắt buộc = 0
# ============================================================

Q = np.diag([
    100.0, 100.0, 300.0,   # position
    50.0,  50.0,  20.0,    # Euler
    5.0,   5.0,   10.0,    # linear velocity
    2.0,   2.0,   2.0,     # angular velocity
    0.0                    # augmented state
])


# ============================================================
# Define R
#
# Input order:
#
# F_Lx F_Ly F_Lz
# M_Lx M_Ly M_Lz
# F_Rx F_Ry F_Rz
# M_Rx M_Ry M_Rz
# ============================================================

R = np.diag([
    0.01, 0.01, 0.01,
    0.01, 0.01, 0.01,
    0.01, 0.01, 0.01,
    0.01, 0.01, 0.01
])


# ============================================================
# Reference state
#
# Test reference:
# giữ x, y hiện tại
# tăng z lên 0.01 m
# orientation = 0
# velocity = 0
# omega = 0
# augmented state = 1
# ============================================================

x_ref = np.zeros(13)

x_ref[0] = W_position_CoM[0]
x_ref[1] = W_position_CoM[1]
x_ref[2] = W_position_CoM[2] + 0.01

x_ref[12] = 1.0


# ============================================================
# Build cost function
# ============================================================

H, f, Q_bar, R_bar, X_ref_stack = (
    build_cost_function(
        Phi=Phi,
        Gamma=Gamma,
        x_current=x_current,
        X_ref=x_ref,
        Q=Q,
        R=R
    )
)


# ============================================================
# Display
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=200
)


print("\n========================================")
print("QP cost function")
print("========================================")

print(f"N = {N}")

print(f"\nx_current shape = {x_current.shape}")
print(f"Phi shape        = {Phi.shape}")
print(f"Gamma shape      = {Gamma.shape}")

print(f"\nQ_bar shape = {Q_bar.shape}")
print(f"R_bar shape = {R_bar.shape}")

print(f"\nH shape = {H.shape}")
print(f"f shape = {f.shape}")


# ============================================================
# Basic checks
# ============================================================

print("\n========================================")
print("Basic checks")
print("========================================")


# ------------------------------------------------------------
# 1. Dimensions
# ------------------------------------------------------------

nx = 13
nu = 12

assert H.shape == (
    N * nu,
    N * nu
)

assert f.shape == (
    N * nu,
)

assert Q_bar.shape == (
    N * nx,
    N * nx
)

assert R_bar.shape == (
    N * nu,
    N * nu
)

assert X_ref_stack.shape == (
    N * nx,
)

print("Matrix dimensions: OK")


# ------------------------------------------------------------
# 2. H symmetry
# ------------------------------------------------------------

assert np.allclose(
    H,
    H.T
)

print("H symmetry: OK")


# ------------------------------------------------------------
# 3. Eigenvalues of H
#
# Vì R > 0 nên H phải positive definite.
# ------------------------------------------------------------

H_eigenvalues = np.linalg.eigvalsh(
    H
)

print("\nH minimum eigenvalue =")
print(np.min(H_eigenvalues))

assert np.all(
    H_eigenvalues > 0.0
)

print("H positive definite: OK")


# ------------------------------------------------------------
# 4. Q augmented-state weight
# ------------------------------------------------------------

assert np.isclose(
    Q[12, 12],
    0.0
)

print("Augmented-state Q weight = 0: OK")


# ------------------------------------------------------------
# 5. Reference stacking
# ------------------------------------------------------------

for k in range(N):

    block = X_ref_stack[
        k * nx:
        (k + 1) * nx
    ]

    assert np.allclose(
        block,
        x_ref
    )

print("Reference stacking: OK")


# ============================================================
# Cost equivalence test
#
# Kiểm tra:
#
# J_original =
#   (X-Xref)' Qbar (X-Xref)
#   + U' Rbar U
#
# và:
#
# J_QP =
#   0.5 U' H U
#   + f' U
#
# Hai cost khác nhau duy nhất bởi constant term:
#
# c =
#   (Phi*x - Xref)' Qbar
#   (Phi*x - Xref)
# ============================================================

print("\n========================================")
print("Cost equivalence test")
print("========================================")


# ------------------------------------------------------------
# Arbitrary test input sequence
# ------------------------------------------------------------

rng = np.random.default_rng(
    seed=1
)

U_test = rng.normal(
    loc=0.0,
    scale=0.1,
    size=N * nu
)


# ------------------------------------------------------------
# Predicted states
# ------------------------------------------------------------

X_test = (
    Phi @ x_current
    + Gamma @ U_test
)


# ------------------------------------------------------------
# Original MPC cost
# ------------------------------------------------------------

tracking_error = (
    X_test
    - X_ref_stack
)

J_original = (
    tracking_error.T
    @ Q_bar
    @ tracking_error

    +

    U_test.T
    @ R_bar
    @ U_test
)


# ------------------------------------------------------------
# QP cost
# ------------------------------------------------------------

J_QP = (
    0.5
    * U_test.T
    @ H
    @ U_test

    +

    f.T
    @ U_test
)


# ------------------------------------------------------------
# Constant term
# ------------------------------------------------------------

prediction_error_0 = (
    Phi @ x_current
    - X_ref_stack
)

J_constant = (
    prediction_error_0.T
    @ Q_bar
    @ prediction_error_0
)


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

print(f"J_original       = {J_original:.12f}")
print(f"J_QP             = {J_QP:.12f}")
print(f"J_constant       = {J_constant:.12f}")

print(
    "J_QP + constant = "
    f"{J_QP + J_constant:.12f}"
)


assert np.allclose(
    J_original,
    J_QP + J_constant,
    rtol=1e-9,
    atol=1e-9
)

print(
    "Original MPC cost = "
    "QP cost + constant: OK"
)


# ============================================================
# Test stacked reference input
# ============================================================

print("\n========================================")
print("Stacked reference test")
print("========================================")

X_ref_manual = np.tile(
    x_ref,
    N
)

H_2, f_2, _, _, X_ref_stack_2 = (
    build_cost_function(
        Phi=Phi,
        Gamma=Gamma,
        x_current=x_current,
        X_ref=X_ref_manual,
        Q=Q,
        R=R
    )
)

assert np.allclose(
    H,
    H_2
)

assert np.allclose(
    f,
    f_2
)

assert np.allclose(
    X_ref_stack,
    X_ref_stack_2
)

print(
    "Single reference and stacked reference "
    "give identical H and f: OK"
)


# ============================================================
# Final
# ============================================================

print("\n========================================")
print("All cost-function tests passed.")
print("========================================")