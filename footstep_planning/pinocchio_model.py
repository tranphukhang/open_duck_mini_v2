from pathlib import Path

import numpy as np
import mujoco
import pinocchio as pin


# ============================================================
# MODEL NAMES
# ============================================================

FLOATING_BASE_JOINT = "floating_base"

LEFT_FOOT_FRAME = "left_foot"
RIGHT_FOOT_FRAME = "right_foot"

HEAD_JOINT_NAMES = (
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
)


# ============================================================
# PINOCCHIO KINEMATIC MODEL
# ============================================================

class PinocchioModel:

    def __init__(
        self,
        mjcf_path,
        mujoco_model,
    ):

        self.mjcf_path = Path(
            mjcf_path
        ).resolve()

        self.mj_model = (
            mujoco_model
        )

        if not self.mjcf_path.exists():

            raise FileNotFoundError(
                f"MJCF file not found:\n"
                f"{self.mjcf_path}"
            )

        # ====================================================
        # LOAD MJCF DIRECTLY INTO PINOCCHIO
        # ====================================================

        self.model = self._load_mjcf_model(
            str(self.mjcf_path)
        )

        self.data = (
            self.model.createData()
        )

        # ====================================================
        # CHECK MAIN FRAMES
        # ====================================================

        self.left_foot_frame_id = (
            self._get_frame_id(
                LEFT_FOOT_FRAME
            )
        )

        self.right_foot_frame_id = (
            self._get_frame_id(
                RIGHT_FOOT_FRAME
            )
        )

        # ====================================================
        # ACTIVE WALKING VELOCITY INDICES
        # ====================================================

        self.walking_velocity_indices = (
            self._build_walking_velocity_indices()
        )

    # ========================================================
    # LOAD MJCF
    # ========================================================

    @staticmethod
    def _load_mjcf_model(
        filename,
    ):
        """
        Load MJCF into Pinocchio.

        Pinocchio 3.x/4.x compatibility.
        """

        if hasattr(
            pin,
            "buildModelFromMJCF",
        ):

            return pin.buildModelFromMJCF(
                filename
            )

        if hasattr(
            pin,
            "buildModelAndConstraintsFromMJCF",
        ):

            result = (
                pin.buildModelAndConstraintsFromMJCF(
                    filename
                )
            )

            return result[0]

        raise RuntimeError(
            "This Pinocchio installation "
            "does not provide an MJCF parser."
        )

    # ========================================================
    # FRAME LOOKUP
    # ========================================================

    def _get_frame_id(
        self,
        frame_name,
    ):

        frame_id = int(
            self.model.getFrameId(
                frame_name
            )
        )

        if frame_id >= len(
            self.model.frames
        ):

            raise ValueError(
                f"Pinocchio frame "
                f"'{frame_name}' was not found."
            )

        return frame_id

    # ========================================================
    # JOINT LOOKUP
    # ========================================================

    def _get_pin_joint_id(
        self,
        joint_name,
    ):

        joint_id = int(
            self.model.getJointId(
                joint_name
            )
        )

        if (
            joint_id
            >=
            self.model.njoints
        ):

            raise ValueError(
                f"Pinocchio joint "
                f"'{joint_name}' was not found."
            )

        if (
            self.model.names[
                joint_id
            ]
            !=
            joint_name
        ):

            raise ValueError(
                f"Pinocchio joint lookup failed "
                f"for '{joint_name}'."
            )

        return joint_id

    # ========================================================
    # MUJOCO -> PINOCCHIO CONFIGURATION
    # ========================================================

    def mujoco_to_pin(
        self,
        q_mj,
    ):
        """
        Convert MuJoCo qpos to Pinocchio configuration.

        Mapping is done by JOINT NAME, not by assuming
        identical array ordering.

        Important quaternion convention:

        MuJoCo:
            [qw, qx, qy, qz]

        Pinocchio:
            [qx, qy, qz, qw]
        """

        q_mj = np.asarray(
            q_mj,
            dtype=float,
        )

        if q_mj.shape != (
            self.mj_model.nq,
        ):

            raise ValueError(
                "MuJoCo qpos has wrong shape."
            )

        q_pin = pin.neutral(
            self.model
        ).copy()

        # Iterate through all MuJoCo joints.
        for mj_joint_id in range(
            self.mj_model.njnt
        ):

            joint_name = mujoco.mj_id2name(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                mj_joint_id,
            )

            if joint_name is None:
                continue

            pin_joint_id = (
                self._get_pin_joint_id(
                    joint_name
                )
            )

            mj_qadr = int(
                self.mj_model.jnt_qposadr[
                    mj_joint_id
                ]
            )

            pin_qadr = int(
                self.model.idx_qs[
                    pin_joint_id
                ]
            )

            pin_nq = int(
                self.model.nqs[
                    pin_joint_id
                ]
            )

            mj_joint_type = int(
                self.mj_model.jnt_type[
                    mj_joint_id
                ]
            )

            # =================================================
            # FREE JOINT
            # =================================================

            if (
                mj_joint_type
                ==
                int(
                    mujoco.mjtJoint.mjJNT_FREE
                )
            ):

                if pin_nq != 7:

                    raise ValueError(
                        f"Free joint '{joint_name}' "
                        f"does not have nq=7 in Pinocchio."
                    )

                # Position
                q_pin[
                    pin_qadr:
                    pin_qadr + 3
                ] = q_mj[
                    mj_qadr:
                    mj_qadr + 3
                ]

                # MuJoCo quaternion:
                #
                # qw qx qy qz
                qw = q_mj[
                    mj_qadr + 3
                ]

                qx = q_mj[
                    mj_qadr + 4
                ]

                qy = q_mj[
                    mj_qadr + 5
                ]

                qz = q_mj[
                    mj_qadr + 6
                ]

                # Pinocchio quaternion:
                #
                # qx qy qz qw
                q_pin[
                    pin_qadr + 3:
                    pin_qadr + 7
                ] = np.array(
                    [
                        qx,
                        qy,
                        qz,
                        qw,
                    ],
                    dtype=float,
                )

            # =================================================
            # HINGE / SLIDE JOINT
            # =================================================

            elif (
                mj_joint_type
                in (
                    int(
                        mujoco.mjtJoint.mjJNT_HINGE
                    ),
                    int(
                        mujoco.mjtJoint.mjJNT_SLIDE
                    ),
                )
            ):

                if pin_nq != 1:

                    raise ValueError(
                        f"Joint '{joint_name}' "
                        f"does not have nq=1 "
                        f"in Pinocchio."
                    )

                q_pin[
                    pin_qadr
                ] = q_mj[
                    mj_qadr
                ]

            else:

                raise NotImplementedError(
                    f"Unsupported MuJoCo joint type "
                    f"for '{joint_name}'."
                )

        return q_pin

    # ========================================================
    # PINOCCHIO -> MUJOCO CONFIGURATION
    # ========================================================

    def pin_to_mujoco(
        self,
        q_pin,
    ):
        """
        Convert Pinocchio configuration back to MuJoCo qpos.
        """

        q_pin = np.asarray(
            q_pin,
            dtype=float,
        )

        if q_pin.shape != (
            self.model.nq,
        ):

            raise ValueError(
                "Pinocchio configuration "
                "has wrong shape."
            )

        q_mj = np.zeros(
            self.mj_model.nq,
            dtype=float,
        )

        for mj_joint_id in range(
            self.mj_model.njnt
        ):

            joint_name = mujoco.mj_id2name(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                mj_joint_id,
            )

            if joint_name is None:
                continue

            pin_joint_id = (
                self._get_pin_joint_id(
                    joint_name
                )
            )

            mj_qadr = int(
                self.mj_model.jnt_qposadr[
                    mj_joint_id
                ]
            )

            pin_qadr = int(
                self.model.idx_qs[
                    pin_joint_id
                ]
            )

            mj_joint_type = int(
                self.mj_model.jnt_type[
                    mj_joint_id
                ]
            )

            # =================================================
            # FREE JOINT
            # =================================================

            if (
                mj_joint_type
                ==
                int(
                    mujoco.mjtJoint.mjJNT_FREE
                )
            ):

                q_mj[
                    mj_qadr:
                    mj_qadr + 3
                ] = q_pin[
                    pin_qadr:
                    pin_qadr + 3
                ]

                # Pinocchio:
                #
                # qx qy qz qw
                qx = q_pin[
                    pin_qadr + 3
                ]

                qy = q_pin[
                    pin_qadr + 4
                ]

                qz = q_pin[
                    pin_qadr + 5
                ]

                qw = q_pin[
                    pin_qadr + 6
                ]

                # MuJoCo:
                #
                # qw qx qy qz
                q_mj[
                    mj_qadr + 3:
                    mj_qadr + 7
                ] = np.array(
                    [
                        qw,
                        qx,
                        qy,
                        qz,
                    ],
                    dtype=float,
                )

            # =================================================
            # 1-DOF JOINT
            # =================================================

            else:

                q_mj[
                    mj_qadr
                ] = q_pin[
                    pin_qadr
                ]

        return q_mj

    # ========================================================
    # WALKING VELOCITY INDICES
    # ========================================================

    def _build_walking_velocity_indices(
        self,
    ):
        """
        Return Pinocchio velocity indices used for walking.

        All DoFs are enabled except:

            neck_pitch
            head_pitch
            head_yaw
            head_roll
        """

        excluded = set()

        for joint_name in HEAD_JOINT_NAMES:

            joint_id = (
                self._get_pin_joint_id(
                    joint_name
                )
            )

            idx_v = int(
                self.model.idx_vs[
                    joint_id
                ]
            )

            nv_joint = int(
                self.model.nvs[
                    joint_id
                ]
            )

            for k in range(
                nv_joint
            ):

                excluded.add(
                    idx_v + k
                )

        active = [
            i
            for i in range(
                self.model.nv
            )
            if i not in excluded
        ]

        return np.asarray(
            active,
            dtype=int,
        )

    # ========================================================
    # UPDATE KINEMATICS
    # ========================================================

    def update(
        self,
        q_pin,
    ):
        """
        Update Pinocchio kinematic quantities.
        """

        q_pin = np.asarray(
            q_pin,
            dtype=float,
        )

        pin.forwardKinematics(
            self.model,
            self.data,
            q_pin,
        )

        pin.computeJointJacobians(
            self.model,
            self.data,
            q_pin,
        )

        pin.updateFramePlacements(
            self.model,
            self.data,
        )

        pin.centerOfMass(
            self.model,
            self.data,
            q_pin,
        )

        self.q = q_pin.copy()

    # ========================================================
    # FRAME POSE
    # ========================================================

    def get_frame_pose(
        self,
        frame_name,
    ):
        """
        Return:

            position : (3,)
            rotation : (3,3)

        in world coordinates.
        """

        frame_id = (
            self._get_frame_id(
                frame_name
            )
        )

        placement = (
            self.data.oMf[
                frame_id
            ]
        )

        return (
            placement.translation.copy(),
            placement.rotation.copy(),
        )

    # ========================================================
    # LEFT FOOT POSE
    # ========================================================

    def get_left_foot_pose(
        self,
    ):

        return self.get_frame_pose(
            LEFT_FOOT_FRAME
        )

    # ========================================================
    # RIGHT FOOT POSE
    # ========================================================

    def get_right_foot_pose(
        self,
    ):

        return self.get_frame_pose(
            RIGHT_FOOT_FRAME
        )

    # ========================================================
    # FRAME JACOBIAN
    # ========================================================

    def get_frame_jacobian_lwa(
        self,
        frame_name,
    ):
        """
        Frame Jacobian in LOCAL_WORLD_ALIGNED frame.

        Row convention in Pinocchio:

            rows 0:3 -> linear velocity
            rows 3:6 -> angular velocity

        Both are expressed in world-aligned axes.
        """

        frame_id = (
            self._get_frame_id(
                frame_name
            )
        )

        return pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).copy()

    # ========================================================
    # FRAME JACOBIAN - LOCAL
    # ========================================================

    def get_frame_jacobian_local(
        self,
        frame_name,
    ):
        """
        Full 6D Jacobian expressed in the local frame.
        """

        frame_id = (
            self._get_frame_id(
                frame_name
            )
        )

        return pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL,
        ).copy()

    # ========================================================
    # FOOT 5D TASK JACOBIAN
    # ========================================================

    def get_foot_task_jacobian(
        self,
        frame_name,
        active_only=False,
    ):
        """
        Foot task:

            3D position in world frame
            +
            local angular Y/Z

        J_foot =
            [ J_position_world ]
            [ J_angular_local_y ]
            [ J_angular_local_z ]

        Shape:

            5 x nv

        or:

            5 x 16

        if active_only=True.
        """

        J_lwa = (
            self.get_frame_jacobian_lwa(
                frame_name
            )
        )

        J_local = (
            self.get_frame_jacobian_local(
                frame_name
            )
        )

        J_task = np.vstack(
            [
                # Position XYZ in world
                J_lwa[
                    0:3,
                    :
                ],

                # Angular Y/Z in local foot frame
                J_local[
                    4:6,
                    :
                ],
            ]
        )

        if active_only:

            J_task = J_task[
                :,
                self.walking_velocity_indices
            ]

        return J_task

    # ========================================================
    # COM POSITION
    # ========================================================

    def get_com(
        self,
    ):

        return (
            self.data.com[
                0
            ].copy()
        )

    # ========================================================
    # COM JACOBIAN
    # ========================================================

    def get_com_jacobian(
        self,
        active_only=False,
    ):
        """
        CoM Jacobian:

            J_com in R^(3 x nv)
        """

        J_com = (
            pin.jacobianCenterOfMass(
                self.model,
                self.data,
                self.q,
            )
            .copy()
        )

        if active_only:

            J_com = J_com[
                :,
                self.walking_velocity_indices
            ]

        return J_com

    # ========================================================
    # CONFIGURATION INTEGRATION
    # ========================================================

    def integrate(
        self,
        q_pin,
        v_pin,
        dt,
    ):
        """
        Integrate a Pinocchio generalized velocity:

            q_next = integrate(q, v * dt)

        This correctly handles the free-flyer quaternion.
        """

        q_pin = np.asarray(
            q_pin,
            dtype=float,
        )

        v_pin = np.asarray(
            v_pin,
            dtype=float,
        )

        if q_pin.shape != (
            self.model.nq,
        ):

            raise ValueError(
                "q_pin has wrong shape."
            )

        if v_pin.shape != (
            self.model.nv,
        ):

            raise ValueError(
                "v_pin has wrong shape."
            )

        if dt < 0.0:

            raise ValueError(
                "dt cannot be negative."
            )

        return pin.integrate(
            self.model,
            q_pin,
            v_pin * float(dt),
        )