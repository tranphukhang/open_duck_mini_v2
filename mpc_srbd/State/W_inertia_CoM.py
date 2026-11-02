import mujoco
import numpy as np


def _validate_inertia(I, name):
    I = np.asarray(I, dtype=float)

    if I.shape != (3, 3):
        raise ValueError(f"{name} phải có shape (3, 3).")

    if not np.all(np.isfinite(I)):
        raise ValueError(f"{name} chứa NaN hoặc Inf.")

    if not np.allclose(I, I.T, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} phải đối xứng.")

    return I


def _get_root_body_id(model, root_body_name):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        root_body_name,
    )

    if body_id < 0:
        raise ValueError(
            f"Không tìm thấy body '{root_body_name}' trong MuJoCo model."
        )

    return body_id


def get_W_R_B(
    model,
    data,
    root_body_name="base_link",
):
    """
    Return current body orientation:

        W_R_B

    mapping a vector expressed in body frame B into world frame W.
    """

    body_id = _get_root_body_id(
        model,
        root_body_name,
    )

    W_R_B = np.asarray(
        data.xmat[body_id],
        dtype=float,
    ).reshape(3, 3)

    if not np.all(np.isfinite(W_R_B)):
        raise RuntimeError("W_R_B chứa NaN hoặc Inf.")

    return W_R_B.copy()


def get_W_inertia_CoM(
    model,
    data,
    root_body_name="base_link",
):
    """
    Read the CURRENT MuJoCo composite whole-body inertia at the
    subtree CoM, expressed in MuJoCo's global/world-oriented frame.

    This function is used at initialization to identify the fixed
    SRBD body-frame inertia.

    It should NOT be called every MPC update to change the SRBD
    rigid-body inertia with joint configuration.
    """

    body_id = _get_root_body_id(
        model,
        root_body_name,
    )

    crb = np.asarray(
        data.crb[body_id],
        dtype=float,
    ).reshape(-1)

    W_inertia_CoM = np.array(
        [
            [crb[0], crb[3], crb[4]],
            [crb[3], crb[1], crb[5]],
            [crb[4], crb[5], crb[2]],
        ],
        dtype=float,
    )

    return _validate_inertia(
        W_inertia_CoM,
        "W_inertia_CoM",
    ).copy()


def get_B_inertia_CoM(
    model,
    data,
    root_body_name="base_link",
):
    """
    Identify the constant SRBD inertia in body coordinates.

    At the initialization configuration q0:

        B_I_CoM
            =
        W_R_B(q0).T
        @ W_I_CoM(q0)
        @ W_R_B(q0)

    B_I_CoM is then treated as CONSTANT by the SRBD model.
    """

    W_inertia_CoM_q0 = get_W_inertia_CoM(
        model=model,
        data=data,
        root_body_name=root_body_name,
    )

    W_R_B_q0 = get_W_R_B(
        model=model,
        data=data,
        root_body_name=root_body_name,
    )

    B_inertia_CoM = (
        W_R_B_q0.T
        @ W_inertia_CoM_q0
        @ W_R_B_q0
    )

    return _validate_inertia(
        B_inertia_CoM,
        "B_inertia_CoM",
    ).copy()


def get_W_inertia_CoM_from_feedback(
    model,
    data,
    B_inertia_CoM,
    root_body_name="base_link",
):
    """
    Re-express the CONSTANT SRBD body-frame inertia in the world
    frame using the CURRENT body orientation feedback:

        W_I_CoM(k)
            =
        W_R_B(k)
        @ B_I_CoM
        @ W_R_B(k).T

    Important
    ---------
    B_I_CoM is constant.

    W_I_CoM changes only because the body orientation changes.
    It is frozen over one MPC prediction horizon and recomputed
    at the next MPC update.
    """

    B_inertia_CoM = _validate_inertia(
        B_inertia_CoM,
        "B_inertia_CoM",
    )

    W_R_B = get_W_R_B(
        model=model,
        data=data,
        root_body_name=root_body_name,
    )

    W_inertia_CoM = (
        W_R_B
        @ B_inertia_CoM
        @ W_R_B.T
    )

    return _validate_inertia(
        W_inertia_CoM,
        "W_inertia_CoM",
    ).copy()
