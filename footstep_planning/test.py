from pathlib import Path

import numpy as np
import mujoco
import pinocchio as pin

from pinocchio_model import (
    PinocchioModel,
    LEFT_FOOT_FRAME,
    RIGHT_FOOT_FRAME,
)


np.set_printoptions(
    precision=10,
    suppress=True,
)


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(
    __file__
).resolve().parent

ROOT_DIR = (
    CURRENT_DIR.parent
)

ROBOT_XML = (
    ROOT_DIR
    /
    "xmls"
    /
    "open_duck_mini_v2.xml"
)

SCENE_XML = (
    ROOT_DIR
    /
    "xmls"
    /
    "scene_flat_terrain.xml"
)


# ============================================================
# HELPERS
# ============================================================

def separator():

    print("=" * 90)


def assert_close(
    actual,
    expected,
    tol,
    message,
):

    if not np.allclose(
        actual,
        expected,
        atol=tol,
        rtol=0.0,
    ):

        raise AssertionError(
            f"{message}\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual}"
        )


def rotation_error_angle(
    R1,
    R2,
):

    R_error = (
        R1.T
        @
        R2
    )

    value = (
        np.trace(
            R_error
        )
        -
        1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return float(
        np.arccos(
            value
        )
    )


# ============================================================
# LOAD MUJOCO
# ============================================================

separator()
print("LOAD MUJOCO")
separator()

mj_model = (
    mujoco.MjModel.from_xml_path(
        str(
            SCENE_XML
        )
    )
)

mj_data = mujoco.MjData(
    mj_model
)


# ============================================================
# LOAD HOME KEYFRAME
# ============================================================

home_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)

if home_id < 0:

    raise RuntimeError(
        "HOME keyframe was not found."
    )


mujoco.mj_resetDataKeyframe(
    mj_model,
    mj_data,
    home_id,
)

mujoco.mj_forward(
    mj_model,
    mj_data,
)


# ============================================================
# LOAD PINOCCHIO
# ============================================================

separator()
print("LOAD PINOCCHIO")
separator()

print(
    "Pinocchio version =",
    pin.__version__,
)

robot = PinocchioModel(
    mjcf_path=ROBOT_XML,
    mujoco_model=mj_model,
)


print(
    f"MuJoCo:    nq={mj_model.nq}, "
    f"nv={mj_model.nv}"
)

print(
    f"Pinocchio: nq={robot.model.nq}, "
    f"nv={robot.model.nv}"
)


if robot.model.nq != mj_model.nq:

    raise AssertionError(
        "Pinocchio and MuJoCo nq differ."
    )


if robot.model.nv != mj_model.nv:

    raise AssertionError(
        "Pinocchio and MuJoCo nv differ."
    )


print("[OK]")


# ============================================================
# TEST 1
# CONFIGURATION MAPPING
# ============================================================

separator()
print("TEST 1 - MUJOCO <-> PINOCCHIO CONFIGURATION")
separator()


q_mj = (
    mj_data.qpos.copy()
)

q_pin = (
    robot.mujoco_to_pin(
        q_mj
    )
)

q_mj_roundtrip = (
    robot.pin_to_mujoco(
        q_pin
    )
)


print(
    "MuJoCo q:",
    q_mj,
)

print()

print(
    "Pinocchio q:",
    q_pin,
)


max_roundtrip_error = np.max(
    np.abs(
        q_mj_roundtrip
        -
        q_mj
    )
)


print(
    f"\nRound-trip max error = "
    f"{max_roundtrip_error:.3e}"
)


assert_close(
    q_mj_roundtrip,
    q_mj,
    tol=1e-12,
    message=(
        "MuJoCo -> Pinocchio -> "
        "MuJoCo mapping failed."
    ),
)


print("[OK]")


# ============================================================
# UPDATE PINOCCHIO
# ============================================================

robot.update(
    q_pin
)


# ============================================================
# TEST 2
# ACTIVE WALKING DOFS
# ============================================================

separator()
print("TEST 2 - WALKING DOFS")
separator()


active = (
    robot.walking_velocity_indices
)


print(
    "active velocity indices =",
    active,
)

print(
    "number of walking DoFs  =",
    len(active),
)


if len(active) != 16:

    raise AssertionError(
        "Expected 16 active walking DoFs."
    )


print("[OK]")


# ============================================================
# TEST 3
# LEFT FOOT POSITION
# ============================================================

separator()
print("TEST 3 - LEFT FOOT POSITION")
separator()


mj_left_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    LEFT_FOOT_FRAME,
)


p_left_mj = (
    mj_data.site_xpos[
        mj_left_id
    ].copy()
)


R_left_mj = (
    mj_data.site_xmat[
        mj_left_id
    ]
    .reshape(3, 3)
    .copy()
)


p_left_pin, R_left_pin = (
    robot.get_left_foot_pose()
)


print(
    "MuJoCo    =",
    p_left_mj,
)

print(
    "Pinocchio =",
    p_left_pin,
)


left_position_error = np.linalg.norm(
    p_left_pin
    -
    p_left_mj
)


left_rotation_error = (
    rotation_error_angle(
        R_left_pin,
        R_left_mj,
    )
)


print(
    f"position error = "
    f"{left_position_error:.3e} m"
)

print(
    f"rotation error = "
    f"{left_rotation_error:.3e} rad"
)


if left_position_error > 1e-7:

    raise AssertionError(
        "LEFT foot position mismatch."
    )


if left_rotation_error > 1e-6:

    raise AssertionError(
        "LEFT foot orientation mismatch."
    )


print("[OK]")


# ============================================================
# TEST 4
# RIGHT FOOT POSITION
# ============================================================

separator()
print("TEST 4 - RIGHT FOOT POSITION")
separator()


mj_right_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    RIGHT_FOOT_FRAME,
)


p_right_mj = (
    mj_data.site_xpos[
        mj_right_id
    ].copy()
)


R_right_mj = (
    mj_data.site_xmat[
        mj_right_id
    ]
    .reshape(3, 3)
    .copy()
)


p_right_pin, R_right_pin = (
    robot.get_right_foot_pose()
)


print(
    "MuJoCo    =",
    p_right_mj,
)

print(
    "Pinocchio =",
    p_right_pin,
)


right_position_error = np.linalg.norm(
    p_right_pin
    -
    p_right_mj
)


right_rotation_error = (
    rotation_error_angle(
        R_right_pin,
        R_right_mj,
    )
)


print(
    f"position error = "
    f"{right_position_error:.3e} m"
)

print(
    f"rotation error = "
    f"{right_rotation_error:.3e} rad"
)


if right_position_error > 1e-7:

    raise AssertionError(
        "RIGHT foot position mismatch."
    )


if right_rotation_error > 1e-6:

    raise AssertionError(
        "RIGHT foot orientation mismatch."
    )


print("[OK]")


# ============================================================
# TEST 5
# CENTER OF MASS
# ============================================================

separator()
print("TEST 5 - CENTER OF MASS")
separator()


base_body_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base",
)


p_com_mj = (
    mj_data.subtree_com[
        base_body_id
    ].copy()
)


p_com_pin = (
    robot.get_com()
)


print(
    "MuJoCo    =",
    p_com_mj,
)

print(
    "Pinocchio =",
    p_com_pin,
)


com_error = np.linalg.norm(
    p_com_pin
    -
    p_com_mj
)


print(
    f"CoM error = "
    f"{com_error:.3e} m"
)


if com_error > 1e-7:

    raise AssertionError(
        "Center of mass mismatch."
    )


print("[OK]")


# ============================================================
# TEST 6
# FOOT JACOBIAN SHAPES
# ============================================================

separator()
print("TEST 6 - FOOT TASK JACOBIANS")
separator()


J_left = (
    robot.get_foot_task_jacobian(
        LEFT_FOOT_FRAME,
        active_only=True,
    )
)


J_right = (
    robot.get_foot_task_jacobian(
        RIGHT_FOOT_FRAME,
        active_only=True,
    )
)


print(
    "J_left shape  =",
    J_left.shape,
)

print(
    "J_right shape =",
    J_right.shape,
)


if J_left.shape != (
    5,
    16,
):

    raise AssertionError(
        "LEFT foot task Jacobian "
        "must be 5x16."
    )


if J_right.shape != (
    5,
    16,
):

    raise AssertionError(
        "RIGHT foot task Jacobian "
        "must be 5x16."
    )


print(
    "rank(J_left)  =",
    np.linalg.matrix_rank(
        J_left
    ),
)

print(
    "rank(J_right) =",
    np.linalg.matrix_rank(
        J_right
    ),
)


print("[OK]")


# ============================================================
# TEST 7
# COM JACOBIAN
# ============================================================

separator()
print("TEST 7 - COM JACOBIAN")
separator()


J_com = (
    robot.get_com_jacobian(
        active_only=True
    )
)


print(
    "J_com shape =",
    J_com.shape,
)

print(
    "rank(J_com) =",
    np.linalg.matrix_rank(
        J_com
    ),
)


if J_com.shape != (
    3,
    16,
):

    raise AssertionError(
        "CoM Jacobian must be 3x16."
    )


print("[OK]")


# ============================================================
# TEST 8
# PINOCCHIO FRAME JACOBIAN FINITE DIFFERENCE
# ============================================================

separator()
print("TEST 8 - FRAME JACOBIAN FINITE DIFFERENCE")
separator()


rng = np.random.default_rng(
    1234
)


v_test = np.zeros(
    robot.model.nv
)


v_test[
    robot.walking_velocity_indices
] = rng.normal(
    size=16
)


# Normalize perturbation direction.
v_test /= np.linalg.norm(
    v_test
)


eps = 1e-7


# ------------------------------------------------------------
# Original frame
# ------------------------------------------------------------

robot.update(
    q_pin
)


p0, R0 = (
    robot.get_left_foot_pose()
)


J6 = (
    robot.get_frame_jacobian_lwa(
        LEFT_FOOT_FRAME
    )
)


predicted_twist = (
    J6
    @
    v_test
)


# ------------------------------------------------------------
# Perturbed configuration
# ------------------------------------------------------------

q_plus = pin.integrate(
    robot.model,
    q_pin,
    eps * v_test,
)


robot.update(
    q_plus
)


p1, R1 = (
    robot.get_left_foot_pose()
)


linear_fd = (
    p1
    -
    p0
) / eps


angular_fd = (
    pin.log3(
        R1
        @
        R0.T
    )
    /
    eps
)


linear_error = np.max(
    np.abs(
        predicted_twist[
            0:3
        ]
        -
        linear_fd
    )
)


angular_error = np.max(
    np.abs(
        predicted_twist[
            3:6
        ]
        -
        angular_fd
    )
)


print(
    f"linear Jacobian error  = "
    f"{linear_error:.3e}"
)

print(
    f"angular Jacobian error = "
    f"{angular_error:.3e}"
)


if linear_error > 1e-5:

    raise AssertionError(
        "Pinocchio linear frame "
        "Jacobian finite-difference test failed."
    )


if angular_error > 1e-5:

    raise AssertionError(
        "Pinocchio angular frame "
        "Jacobian finite-difference test failed."
    )


print("[OK]")


# ============================================================
# TEST 9
# COM JACOBIAN FINITE DIFFERENCE
# ============================================================

separator()
print("TEST 9 - COM JACOBIAN FINITE DIFFERENCE")
separator()


robot.update(
    q_pin
)


com0 = (
    robot.get_com()
)


J_com_full = (
    robot.get_com_jacobian(
        active_only=False
    )
)


predicted_com_velocity = (
    J_com_full
    @
    v_test
)


robot.update(
    q_plus
)


com1 = (
    robot.get_com()
)


com_velocity_fd = (
    com1
    -
    com0
) / eps


com_jac_error = np.max(
    np.abs(
        predicted_com_velocity
        -
        com_velocity_fd
    )
)


print(
    f"CoM Jacobian error = "
    f"{com_jac_error:.3e}"
)


if com_jac_error > 1e-5:

    raise AssertionError(
        "Pinocchio CoM Jacobian "
        "finite-difference test failed."
    )


print("[OK]")


# ============================================================
# FINAL
# ============================================================

separator()
print(
    "ALL PINOCCHIO MODEL TESTS PASSED"
)
separator()


print("""
Verified:

  1. Pinocchio loads the same Open Duck MJCF model
  2. nq and nv agree with MuJoCo
  3. MuJoCo <-> Pinocchio configuration mapping is reversible
  4. Free-flyer quaternion conversion is correct
  5. 16 walking DoFs are selected
  6. Left foot pose agrees between Pinocchio and MuJoCo
  7. Right foot pose agrees between Pinocchio and MuJoCo
  8. Center of mass agrees between Pinocchio and MuJoCo
  9. Foot task Jacobians have shape 5 x 16
 10. CoM Jacobian has shape 3 x 16
 11. Pinocchio frame Jacobian passes finite difference
 12. Pinocchio CoM Jacobian passes finite difference

No differential IK has been solved yet.
""")