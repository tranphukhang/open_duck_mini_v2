# step_timing_adaptation/dynamics_model.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


# ============================================================
# DATA
# ============================================================

@dataclass
class DynamicsTerms:

    qpos: np.ndarray
    qvel: np.ndarray

    mass_matrix: np.ndarray

    bias_force: np.ndarray
    passive_force: np.ndarray
    effective_bias: np.ndarray

    selection_matrix: np.ndarray


@dataclass
class SiteKinematics:

    site_name: str

    site_id: int
    body_id: int

    position: np.ndarray
    rotation: np.ndarray

    jacobian_linear: np.ndarray
    jacobian_angular: np.ndarray

    jacobian: np.ndarray

    jacobian_dot_linear: np.ndarray
    jacobian_dot_angular: np.ndarray

    jacobian_dot: np.ndarray

    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    jacobian_dot_velocity: np.ndarray


@dataclass
class CoMKinematics:

    body_name: str
    body_id: int

    mass: float

    position: np.ndarray

    jacobian: np.ndarray

    velocity: np.ndarray

    subtree_velocity: np.ndarray


# ============================================================
# WHOLE-BODY DYNAMICS MODEL
# ============================================================

class WholeBodyDynamicsModel:
    """
    MuJoCo 3.12 whole-body dynamics / kinematics interface.

    Controller equation used later:

        M(q) vdot + h_eff(q, v)
            = S.T tau + Jc.T lambda

    where:

        h_eff
            =
        qfrc_bias - qfrc_passive

    The module also provides:

        - foot/site Jacobians
        - foot/site Jacobian derivatives
        - Jdot * v
        - whole-robot CoM
        - whole-robot CoM Jacobian

    No QP is implemented here.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
    ) -> None:

        self.model = model

        # ====================================================
        # DIMENSIONS
        # ====================================================

        self.nq = int(
            model.nq
        )

        self.nv = int(
            model.nv
        )

        self.nu = int(
            model.nu
        )

        self.nactuator = int(
            model.nactuator
        )

        self.nout = int(
            model.nout
        )

        # ====================================================
        # CURRENT MODEL: SISO MOTORS
        # ====================================================

        if not (
            self.nactuator
            ==
            self.nu
            ==
            self.nout
        ):

            raise RuntimeError(
                "Current WBC assumes SISO motor actuators.\n"
                f"nactuator = {self.nactuator}\n"
                f"nu        = {self.nu}\n"
                f"nout      = {self.nout}"
            )

        # ====================================================
        # ACTUATED DOF MAPPING
        # ====================================================

        (
            self.actuated_dof_indices,
            self.actuator_names,
            self.actuated_joint_names,
        ) = (
            self._build_actuated_dof_mapping()
        )

        actuated_set = set(
            self.actuated_dof_indices.tolist()
        )

        self.unactuated_dof_indices = np.asarray(
            [
                i
                for i in range(
                    self.nv
                )
                if i not in actuated_set
            ],
            dtype=int,
        )

        if (
            len(
                np.unique(
                    self.actuated_dof_indices
                )
            )
            !=
            self.nactuator
        ):

            raise RuntimeError(
                "Actuated DOF mapping is not unique."
            )


    # ========================================================
    # OBJECT LOOKUP
    # ========================================================

    def get_site_id(
        self,
        site_name: str,
    ) -> int:

        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id < 0:

            raise ValueError(
                f"Site '{site_name}' was not found."
            )

        return int(
            site_id
        )


    def get_body_id(
        self,
        body_name: str,
    ) -> int:

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id < 0:

            raise ValueError(
                f"Body '{body_name}' was not found."
            )

        return int(
            body_id
        )


    # ========================================================
    # ACTUATED DOF MAPPING
    # ========================================================

    def _build_actuated_dof_mapping(
        self,
    ):

        dof_indices = []

        actuator_names = []

        joint_names = []

        for actuator_id in range(
            self.nactuator
        ):

            actuator_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_id,
                )
            )

            if actuator_name is None:

                raise RuntimeError(
                    f"Actuator {actuator_id} has no name."
                )

            joint_id = int(
                self.model.actuator_trnid[
                    actuator_id,
                    0,
                ]
            )

            if joint_id < 0:

                raise RuntimeError(
                    f"Actuator '{actuator_name}' "
                    "does not target a valid joint."
                )

            joint_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id,
                )
            )

            if joint_name is None:

                raise RuntimeError(
                    f"Joint for actuator "
                    f"'{actuator_name}' has no name."
                )

            joint_type = int(
                self.model.jnt_type[
                    joint_id
                ]
            )

            if (
                joint_type
                !=
                int(
                    mujoco.mjtJoint.mjJNT_HINGE
                )
            ):

                raise RuntimeError(
                    f"Actuator '{actuator_name}' "
                    f"targets non-hinge joint "
                    f"'{joint_name}'."
                )

            dof_adr = int(
                self.model.jnt_dofadr[
                    joint_id
                ]
            )

            dof_indices.append(
                dof_adr
            )

            actuator_names.append(
                actuator_name
            )

            joint_names.append(
                joint_name
            )

        return (
            np.asarray(
                dof_indices,
                dtype=int,
            ),

            tuple(
                actuator_names
            ),

            tuple(
                joint_names
            ),
        )


    # ========================================================
    # MASS MATRIX
    # ========================================================

    def get_mass_matrix(
        self,
        data: mujoco.MjData,
    ) -> np.ndarray:

        M = np.zeros(
            (
                self.nv,
                self.nv,
            ),
            dtype=float,
        )

        # MuJoCo 3.12 API
        mujoco.mj_fullM(
            self.model,
            data,
            M,
        )

        return M


    # ========================================================
    # BIAS
    # ========================================================

    @staticmethod
    def get_bias_force(
        data: mujoco.MjData,
    ) -> np.ndarray:

        return np.asarray(
            data.qfrc_bias,
            dtype=float,
        ).copy()


    # ========================================================
    # PASSIVE FORCE
    # ========================================================

    @staticmethod
    def get_passive_force(
        data: mujoco.MjData,
    ) -> np.ndarray:

        return np.asarray(
            data.qfrc_passive,
            dtype=float,
        ).copy()


    # ========================================================
    # EFFECTIVE BIAS
    # ========================================================

    @staticmethod
    def get_effective_bias(
        data: mujoco.MjData,
    ) -> np.ndarray:

        return (
            np.asarray(
                data.qfrc_bias,
                dtype=float,
            )
            -
            np.asarray(
                data.qfrc_passive,
                dtype=float,
            )
        ).copy()


    # ========================================================
    # ACTUATOR MOMENT / SELECTION MATRIX
    # ========================================================

    def get_selection_matrix(
        self,
        data: mujoco.MjData,
    ) -> np.ndarray:

        S = np.zeros(
            (
                self.nout,
                self.nv,
            ),
            dtype=float,
        )

        # MuJoCo 3.12:
        # actuator_moment is sparse CSR.
        mujoco.mju_sparse2dense(
            S,
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )

        return S


    # ========================================================
    # WHOLE-BODY DYNAMICS
    # ========================================================

    def compute(
        self,
        data: mujoco.MjData,
        forward: bool = True,
    ) -> DynamicsTerms:

        if forward:

            mujoco.mj_forward(
                self.model,
                data,
            )

        M = self.get_mass_matrix(
            data
        )

        bias = self.get_bias_force(
            data
        )

        passive = self.get_passive_force(
            data
        )

        h_eff = self.get_effective_bias(
            data
        )

        S = self.get_selection_matrix(
            data
        )

        if M.shape != (
            self.nv,
            self.nv,
        ):

            raise RuntimeError(
                "Mass matrix has invalid shape."
            )

        if S.shape != (
            self.nout,
            self.nv,
        ):

            raise RuntimeError(
                "Selection matrix has invalid shape."
            )

        for name, value in (
            (
                "M",
                M,
            ),
            (
                "bias",
                bias,
            ),
            (
                "passive",
                passive,
            ),
            (
                "h_eff",
                h_eff,
            ),
            (
                "S",
                S,
            ),
        ):

            if not np.all(
                np.isfinite(
                    value
                )
            ):

                raise RuntimeError(
                    f"{name} contains NaN/Inf."
                )

        return DynamicsTerms(
            qpos=(
                data.qpos.copy()
            ),

            qvel=(
                data.qvel.copy()
            ),

            mass_matrix=(
                M
            ),

            bias_force=(
                bias
            ),

            passive_force=(
                passive
            ),

            effective_bias=(
                h_eff
            ),

            selection_matrix=(
                S
            ),
        )


    # ========================================================
    # GENERALIZED ACTUATION
    # ========================================================

    def generalized_actuation(
        self,
        terms: DynamicsTerms,
        torque,
    ) -> np.ndarray:

        torque = np.asarray(
            torque,
            dtype=float,
        )

        if torque.shape != (
            self.nu,
        ):

            raise ValueError(
                f"torque must have shape ({self.nu},)."
            )

        if self.nu != self.nout:

            raise RuntimeError(
                "Current torque mapping requires nu == nout."
            )

        return (
            terms.selection_matrix.T
            @
            torque
        )


    # ========================================================
    # SITE JACOBIAN
    # ========================================================

    def get_site_jacobian(
        self,
        data: mujoco.MjData,
        site_name: str,
    ) -> np.ndarray:
        """
        Return full 6D site Jacobian:

            J =
            [ J_linear  ]
            [ J_angular ]

        Shape:

            6 x nv.

        Both components are expressed in world coordinates.
        """

        site_id = self.get_site_id(
            site_name
        )

        jacp = np.zeros(
            (
                3,
                self.nv,
            ),
            dtype=float,
        )

        jacr = np.zeros(
            (
                3,
                self.nv,
            ),
            dtype=float,
        )

        mujoco.mj_jacSite(
            self.model,
            data,
            jacp,
            jacr,
            site_id,
        )

        return np.vstack(
            (
                jacp,
                jacr,
            )
        )


    # ========================================================
    # SITE JACOBIAN DOT
    # ========================================================

    def get_site_jacobian_dot(
        self,
        data: mujoco.MjData,
        site_name: str,
    ) -> np.ndarray:
        """
        Compute time derivative of site Jacobian:

            Jdot in R^(6 x nv).

        mj_jacDot expects:
            - current world position of the point
            - body to which the point is attached.

        The site origin is rigidly attached to its parent body.
        """

        site_id = self.get_site_id(
            site_name
        )

        body_id = int(
            self.model.site_bodyid[
                site_id
            ]
        )

        point = (
            data.site_xpos[
                site_id
            ]
            .copy()
        )

        jacp_dot = np.zeros(
            (
                3,
                self.nv,
            ),
            dtype=float,
        )

        jacr_dot = np.zeros(
            (
                3,
                self.nv,
            ),
            dtype=float,
        )

        mujoco.mj_jacDot(
            self.model,
            data,
            jacp_dot,
            jacr_dot,
            point,
            body_id,
        )

        return np.vstack(
            (
                jacp_dot,
                jacr_dot,
            )
        )


    # ========================================================
    # SITE KINEMATICS
    # ========================================================

    def get_site_kinematics(
        self,
        data: mujoco.MjData,
        site_name: str,
    ) -> SiteKinematics:

        site_id = self.get_site_id(
            site_name
        )

        body_id = int(
            self.model.site_bodyid[
                site_id
            ]
        )

        position = (
            data.site_xpos[
                site_id
            ]
            .copy()
        )

        rotation = (
            data.site_xmat[
                site_id
            ]
            .reshape(
                3,
                3,
            )
            .copy()
        )

        J = self.get_site_jacobian(
            data,
            site_name,
        )

        Jdot = (
            self.get_site_jacobian_dot(
                data,
                site_name,
            )
        )

        Jp = (
            J[
                0:3,
                :
            ]
            .copy()
        )

        Jr = (
            J[
                3:6,
                :
            ]
            .copy()
        )

        Jp_dot = (
            Jdot[
                0:3,
                :
            ]
            .copy()
        )

        Jr_dot = (
            Jdot[
                3:6,
                :
            ]
            .copy()
        )

        qvel = (
            data.qvel.copy()
        )

        linear_velocity = (
            Jp
            @
            qvel
        )

        angular_velocity = (
            Jr
            @
            qvel
        )

        Jdot_v = (
            Jdot
            @
            qvel
        )

        return SiteKinematics(
            site_name=(
                site_name
            ),

            site_id=(
                site_id
            ),

            body_id=(
                body_id
            ),

            position=(
                position
            ),

            rotation=(
                rotation
            ),

            jacobian_linear=(
                Jp
            ),

            jacobian_angular=(
                Jr
            ),

            jacobian=(
                J
            ),

            jacobian_dot_linear=(
                Jp_dot
            ),

            jacobian_dot_angular=(
                Jr_dot
            ),

            jacobian_dot=(
                Jdot
            ),

            linear_velocity=(
                linear_velocity
            ),

            angular_velocity=(
                angular_velocity
            ),

            jacobian_dot_velocity=(
                Jdot_v
            ),
        )


    # ========================================================
    # CENTER OF MASS
    # ========================================================

    def get_com_kinematics(
        self,
        data: mujoco.MjData,
        root_body_name: str,
    ) -> CoMKinematics:
        """
        Return center of mass and translational Jacobian for the
        entire subtree rooted at root_body_name.

        For this robot:

            root_body_name = "base"

        therefore the subtree is the complete robot.
        """

        body_id = self.get_body_id(
            root_body_name
        )

        position = (
            data.subtree_com[
                body_id
            ]
            .copy()
        )

        mass = float(
            self.model.body_subtreemass[
                body_id
            ]
        )

        J_com = np.zeros(
            (
                3,
                self.nv,
            ),
            dtype=float,
        )

        mujoco.mj_jacSubtreeCom(
            self.model,
            data,
            J_com,
            body_id,
        )

        velocity = (
            J_com
            @
            data.qvel
        )

        # Computes data.subtree_linvel.
        mujoco.mj_subtreeVel(
            self.model,
            data,
        )

        subtree_velocity = (
            data.subtree_linvel[
                body_id
            ]
            .copy()
        )

        return CoMKinematics(
            body_name=(
                root_body_name
            ),

            body_id=(
                body_id
            ),

            mass=(
                mass
            ),

            position=(
                position
            ),

            jacobian=(
                J_com
            ),

            velocity=(
                velocity
            ),

            subtree_velocity=(
                subtree_velocity
            ),
        )


    # ========================================================
    # PRINT MODEL STRUCTURE
    # ========================================================

    def print_dof_structure(
        self,
    ) -> None:

        print()

        print(
            "MuJoCo 3.12 actuator structure:"
        )

        print(
            f"  nactuator = {self.nactuator}"
        )

        print(
            f"  nu        = {self.nu}"
        )

        print(
            f"  nout      = {self.nout}"
        )

        print()

        print(
            "Generalized velocity structure:"
        )

        print(
            "  unactuated DOFs =",
            self.unactuated_dof_indices,
        )

        print(
            "  actuated DOFs   =",
            self.actuated_dof_indices,
        )

        print()

        print(
            f"{'ACT':>3}  "
            f"{'ACTUATOR':<22}  "
            f"{'JOINT':<22}  "
            f"{'DOF':>4}"
        )

        print(
            "-" * 60
        )

        for actuator_id in range(
            self.nactuator
        ):

            print(
                f"{actuator_id:>3d}  "
                f"{self.actuator_names[actuator_id]:<22}  "
                f"{self.actuated_joint_names[actuator_id]:<22}  "
                f"{self.actuated_dof_indices[actuator_id]:>4d}"
            )