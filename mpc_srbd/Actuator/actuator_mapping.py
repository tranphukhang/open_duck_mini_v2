import numpy as np
import mujoco


# ============================================================
# JOINT TORQUE ORDER
#
# This order must be consistent with:
#
#     W_wrench_to_joint_torque.py
#
# tau =
#
# [
#     tau_L_hip_yaw,
#     tau_L_hip_roll,
#     tau_L_hip_pitch,
#     tau_L_knee,
#     tau_L_ankle,
#
#     tau_R_hip_yaw,
#     tau_R_hip_roll,
#     tau_R_hip_pitch,
#     tau_R_knee,
#     tau_R_ankle
# ]
# ============================================================

ACTUATOR_NAMES = (
    "L_hip_yaw",
    "L_hip_roll",
    "L_hip_pitch",
    "L_knee",
    "L_ankle",

    "R_hip_yaw",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "R_ankle",
)


# ============================================================
# GET ACTUATOR IDS
# ============================================================

def get_actuator_ids(model):
    """
    Get MuJoCo actuator IDs corresponding to the joint torque
    ordering used by the MPC controller.

    Returns
    -------
    actuator_ids : np.ndarray
        MuJoCo actuator IDs.

        Shape:
            (10,)
    """

    actuator_ids = []


    for actuator_name in ACTUATOR_NAMES:

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name
        )


        if actuator_id == -1:

            raise ValueError(
                f"Không tìm thấy actuator '{actuator_name}' "
                "trong MuJoCo model."
            )


        actuator_ids.append(
            actuator_id
        )


    actuator_ids = np.asarray(
        actuator_ids,
        dtype=int
    )


    if actuator_ids.shape != (10,):

        raise RuntimeError(
            "actuator_ids phải có shape (10,), "
            f"nhưng nhận được {actuator_ids.shape}."
        )


    if len(
        np.unique(
            actuator_ids
        )
    ) != 10:

        raise RuntimeError(
            "Actuator mapping chứa actuator ID bị trùng."
        )


    return actuator_ids


# ============================================================
# APPLY JOINT TORQUE TO MUJOCO ACTUATORS
# ============================================================

def apply_joint_torque(
    model,
    data,
    tau
):
    """
    Map the 10-joint torque vector to MuJoCo data.ctrl.

    Input torque ordering:

        tau =
        [
            tau_L_hip_yaw,
            tau_L_hip_roll,
            tau_L_hip_pitch,
            tau_L_knee,
            tau_L_ankle,

            tau_R_hip_yaw,
            tau_R_hip_roll,
            tau_R_hip_pitch,
            tau_R_knee,
            tau_R_ankle
        ]

    Mapping is performed using actuator names instead of
    assuming that actuator IDs are 0...9.

    No torque saturation is performed here.

    MuJoCo handles actuator control limits according to the
    actuator settings defined in the XML model.

    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model.

    data : mujoco.MjData
        MuJoCo simulation data.

    tau : array_like
        Joint torque command.

        Shape:
            (10,)

    Returns
    -------
    actuator_ids : np.ndarray
        MuJoCo actuator IDs used for the mapping.

        Shape:
            (10,)
    """

    # ========================================================
    # 1. Convert torque input
    # ========================================================

    tau = np.asarray(
        tau,
        dtype=float
    ).reshape(-1)


    # ========================================================
    # 2. Dimension check
    # ========================================================

    if tau.shape != (10,):

        raise ValueError(
            "tau phải có shape (10,), "
            f"nhưng nhận được {tau.shape}."
        )


    # ========================================================
    # 3. Finite check
    # ========================================================

    if not np.all(
        np.isfinite(
            tau
        )
    ):

        raise ValueError(
            "tau chứa NaN hoặc Inf."
        )


    # ========================================================
    # 4. Get actuator IDs
    # ========================================================

    actuator_ids = get_actuator_ids(
        model
    )


    # ========================================================
    # 5. Write joint torque commands to data.ctrl
    #
    # Mapping:
    #
    # tau[i]
    #
    #     ->
    #
    # ACTUATOR_NAMES[i]
    #
    #     ->
    #
    # actuator_id
    #
    #     ->
    #
    # data.ctrl[actuator_id]
    #
    # ========================================================

    for i, actuator_id in enumerate(
        actuator_ids
    ):

        data.ctrl[
            actuator_id
        ] = tau[
            i
        ]


    # ========================================================
    # 6. Final finite check
    # ========================================================

    mapped_ctrl = data.ctrl[
        actuator_ids
    ]


    if not np.all(
        np.isfinite(
            mapped_ctrl
        )
    ):

        raise RuntimeError(
            "data.ctrl chứa NaN hoặc Inf sau khi mapping."
        )


    return actuator_ids