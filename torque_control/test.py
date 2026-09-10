"""
Test MuJoCo inverse dynamics
for Open Duck Mini V2.

Check:

1. Load XML
2. Floating base dimension
3. Joint list
4. Gravity torque
5. Torque split
"""


from pathlib import Path

import numpy as np
import mujoco


from inverse_dynamics import (
    MuJoCoInverseDynamics
)



# ==================================================
# Find XML automatically
# ==================================================


ROOT = Path(
    __file__
).resolve().parent.parent



XML_PATH = None


for p in ROOT.rglob(
    "open_duck_mini_v2.xml"
):

    XML_PATH = p
    break



if XML_PATH is None:

    raise FileNotFoundError(
        "Cannot find open_duck_mini_v2.xml"
    )




def main():


    print("="*70)
    print(
        "OPEN DUCK MINI V2 "
        "INVERSE DYNAMICS TEST"
    )
    print("="*70)



    print(
        "XML:"
    )

    print(
        XML_PATH
    )



    # ==============================================
    # Load model
    # ==============================================


    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )


    data = mujoco.MjData(
        model
    )


    mujoco.mj_forward(
        model,
        data
    )



    print()
    print("[PASS] MuJoCo model loaded")



    # ==============================================
    # Dimension
    # ==============================================


    print()
    print("MODEL DIMENSION")
    print("----------------")

    print(
        "nq =",
        model.nq
    )

    print(
        "nv =",
        model.nv
    )


    assert model.nq > model.nv


    print(
        "[PASS] Floating-base model detected"
    )



    # ==============================================
    # Joint information
    # ==============================================


    print()
    print("JOINT LIST")
    print("----------------")


    for i in range(
        model.njnt
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )

        print(
            i,
            name
        )



    # ==============================================
    # Actuator
    # ==============================================


    print()
    print("ACTUATOR LIST")
    print("----------------")


    for i in range(
        model.nu
    ):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i
        )


        print(
            i,
            name
        )



    # ==============================================
    # Gravity compensation
    # ==============================================


    invdyn = MuJoCoInverseDynamics(
        model
    )


    qpos = data.qpos.copy()


    tau = invdyn.gravity_compensation(
        qpos
    )


    print()
    print("GRAVITY TORQUE")
    print("----------------")


    print(
        tau
    )



    assert tau.shape[0] == model.nv


    print(
        "[PASS] Gravity torque dimension"
    )



    # ==============================================
    # Split
    # ==============================================


    base_tau, joint_tau = (
        invdyn.split_base_joint_torque(
            tau
        )
    )


    print()
    print("BASE WRENCH")
    print("----------------")

    print(
        base_tau
    )


    print()
    print("JOINT TORQUE")
    print("----------------")

    print(
        joint_tau
    )


    print()
    print(
        "[PASS] Torque split"
    )



    print()
    print("="*70)
    print(
        "INVERSE DYNAMICS TEST PASSED"
    )
    print("="*70)





if __name__ == "__main__":

    main()