"""
MuJoCo inverse dynamics module
for floating-base humanoid robot.

Compute:

tau = M(q) qdd + h(q,qdot)

For gravity compensation:

qdot = 0
qddot = 0

tau = gravity torque
"""


import numpy as np
import mujoco



class MuJoCoInverseDynamics:


    def __init__(
        self,
        model: mujoco.MjModel
    ):

        self.model = model

        self.data = mujoco.MjData(
            model
        )



    def compute(
        self,
        qpos,
        qvel,
        qacc
    ):
        """
        General inverse dynamics.

        Parameters
        ----------
        qpos:
            generalized position

        qvel:
            generalized velocity

        qacc:
            generalized acceleration


        Returns
        -------
        tau:
            generalized force
        """


        self.data.qpos[:] = qpos

        self.data.qvel[:] = qvel

        self.data.qacc[:] = qacc


        mujoco.mj_inverse(
            self.model,
            self.data
        )


        return self.data.qfrc_inverse.copy()



    def gravity_compensation(
        self,
        qpos
    ):
        """
        Compute static gravity torque.

        qdot = 0
        qddot = 0
        """


        qvel = np.zeros(
            self.model.nv
        )


        qacc = np.zeros(
            self.model.nv
        )


        return self.compute(
            qpos,
            qvel,
            qacc
        )



    def split_base_joint_torque(
        self,
        tau
    ):
        """
        Separate floating base force
        and actuated joint torque.

        MuJoCo generalized force:

        [base wrench]
        [joint torque]

        """


        nv_base = 6


        base = tau[:nv_base]

        joints = tau[nv_base:]


        return base, joints