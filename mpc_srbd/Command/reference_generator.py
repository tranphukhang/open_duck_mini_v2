import numpy as np


# ============================================================
# REFERENCE GENERATOR
# ============================================================

class ReferenceGenerator:
    """
    Generate MPC reference trajectory from desired CoM
    linear velocity.

    State definition:

        x =
        [
            p_CoM,          # 3
            euler_base,     # 3
            v_CoM,          # 3
            omega_base,     # 3
            1               # 1
        ]

    State dimension:

        nx = 13


    Main idea
    ---------

    The reference generator maintains its own CoM position
    reference internally.

    Therefore, the position reference does NOT follow the
    measured robot position x_k.

    At control cycle k:

        p_ref,k

    is stored internally.

    From the desired CoM linear velocity:

        v_CoM_cmd

    the prediction reference is generated as:

        p_ref(k+i|k)
            =
        p_ref,k
            +
        i * dt_prediction * v_CoM_cmd

        i = 1, ..., N


    After generating the current MPC horizon, the internal
    reference position is advanced according to the actual
    MPC update period:

        p_ref,k+1
            =
        p_ref,k
            +
        dt_update * v_CoM_cmd


    This distinction is important because:

        dt_prediction

    defines the prediction grid inside MPC, while:

        dt_update

    defines how frequently the controller is actually updated.
    """


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        x_initial,
        horizon,
        dt_prediction,
        dt_update
    ):

        # ----------------------------------------------------
        # Convert inputs
        # ----------------------------------------------------

        x_initial = np.asarray(
            x_initial,
            dtype=float
        ).reshape(-1)


        # ----------------------------------------------------
        # Validate x_initial
        # ----------------------------------------------------

        if x_initial.shape != (13,):
            raise ValueError(
                "x_initial phải có shape (13,), "
                f"nhưng nhận được {x_initial.shape}."
            )


        if not np.all(
            np.isfinite(
                x_initial
            )
        ):
            raise ValueError(
                "x_initial chứa NaN hoặc Inf."
            )


        # ----------------------------------------------------
        # Validate horizon
        # ----------------------------------------------------

        if not isinstance(
            horizon,
            (int, np.integer)
        ):
            raise TypeError(
                "horizon phải là số nguyên."
            )


        if horizon <= 0:
            raise ValueError(
                "horizon phải > 0."
            )


        # ----------------------------------------------------
        # Validate dt_prediction
        # ----------------------------------------------------

        dt_prediction = float(
            dt_prediction
        )


        if (
            not np.isfinite(dt_prediction)
            or
            dt_prediction <= 0.0
        ):
            raise ValueError(
                "dt_prediction phải là số hữu hạn và > 0."
            )


        # ----------------------------------------------------
        # Validate dt_update
        # ----------------------------------------------------

        dt_update = float(
            dt_update
        )


        if (
            not np.isfinite(dt_update)
            or
            dt_update <= 0.0
        ):
            raise ValueError(
                "dt_update phải là số hữu hạn và > 0."
            )


        # ====================================================
        # STORE PARAMETERS
        # ====================================================

        self.nx = 13

        self.horizon = int(
            horizon
        )

        self.dt_prediction = (
            dt_prediction
        )

        self.dt_update = (
            dt_update
        )


        # ====================================================
        # INITIAL REFERENCE POSITION
        #
        # The initial CoM position becomes the reference anchor.
        #
        # IMPORTANT:
        #
        # This reference is independent of future measured
        # robot positions.
        # ====================================================

        self.p_ref = x_initial[
            0:3
        ].copy()


    # ========================================================
    # VELOCITY COMMAND VALIDATION
    # ========================================================

    @staticmethod
    def _validate_velocity_command(
        v_CoM_cmd
    ):

        v_CoM_cmd = np.asarray(
            v_CoM_cmd,
            dtype=float
        ).reshape(-1)


        if v_CoM_cmd.shape != (3,):
            raise ValueError(
                "v_CoM_cmd phải có shape (3,), "
                f"nhưng nhận được {v_CoM_cmd.shape}."
            )


        if not np.all(
            np.isfinite(
                v_CoM_cmd
            )
        ):
            raise ValueError(
                "v_CoM_cmd chứa NaN hoặc Inf."
            )


        return v_CoM_cmd


    # ========================================================
    # GENERATE MPC REFERENCE
    # ========================================================

    def generate_reference(
        self,
        v_CoM_cmd
    ):
        """
        Generate the MPC reference trajectory.

        Parameters
        ----------
        v_CoM_cmd : array_like, shape (3,)

            Desired CoM linear velocity in World frame:

                [
                    vx_cmd,
                    vy_cmd,
                    vz_cmd
                ]


        Returns
        -------
        X_ref : ndarray, shape (13 * horizon,)

            Stacked MPC reference trajectory:

                X_ref =
                [
                    x_ref(k+1|k),
                    x_ref(k+2|k),
                    ...
                    x_ref(k+N|k)
                ]
        """


        # ====================================================
        # 1. VALIDATE COMMAND
        # ====================================================

        v_CoM_cmd = (
            self._validate_velocity_command(
                v_CoM_cmd
            )
        )


        # ====================================================
        # 2. ALLOCATE REFERENCE TRAJECTORY
        # ====================================================

        X_ref_matrix = np.zeros(
            (
                self.horizon,
                self.nx
            ),
            dtype=float
        )


        # ====================================================
        # 3. BUILD REFERENCE OVER MPC HORIZON
        # ====================================================

        for i in range(
            self.horizon
        ):

            # ------------------------------------------------
            # Prediction step
            #
            # i = 0       -> k+1
            # i = 1       -> k+2
            # ...
            # i = N - 1   -> k+N
            # ------------------------------------------------

            prediction_step = (
                i
                +
                1
            )


            # ------------------------------------------------
            # Prediction time relative to current MPC cycle
            # ------------------------------------------------

            prediction_time = (
                prediction_step
                *
                self.dt_prediction
            )


            # ------------------------------------------------
            # CoM position reference
            #
            # p_ref(k+i|k)
            #
            # =
            #
            # p_ref,k
            #
            # +
            #
            # i * dt_prediction * v_CoM_cmd
            # ------------------------------------------------

            p_ref_prediction = (
                self.p_ref
                +
                prediction_time
                *
                v_CoM_cmd
            )


            X_ref_matrix[
                i,
                0:3
            ] = p_ref_prediction


            # ------------------------------------------------
            # Base orientation reference
            #
            # roll_ref  = 0
            # pitch_ref = 0
            # yaw_ref   = 0
            # ------------------------------------------------

            X_ref_matrix[
                i,
                3:6
            ] = 0.0


            # ------------------------------------------------
            # CoM linear velocity reference
            # ------------------------------------------------

            X_ref_matrix[
                i,
                6:9
            ] = v_CoM_cmd


            # ------------------------------------------------
            # Base angular velocity reference
            #
            # wx_ref = 0
            # wy_ref = 0
            # wz_ref = 0
            # ------------------------------------------------

            X_ref_matrix[
                i,
                9:12
            ] = 0.0


            # ------------------------------------------------
            # Augmented constant
            # ------------------------------------------------

            X_ref_matrix[
                i,
                12
            ] = 1.0


        # ====================================================
        # 4. STACK INTO MPC FORMAT
        #
        # X_ref =
        #
        # [
        #     x_ref(k+1|k)
        #     x_ref(k+2|k)
        #     ...
        #     x_ref(k+N|k)
        # ]
        # ====================================================

        X_ref = X_ref_matrix.reshape(
            self.horizon
            *
            self.nx
        )


        # ====================================================
        # 5. FINAL VALIDATION
        # ====================================================

        expected_shape = (
            self.nx
            *
            self.horizon,
        )


        if X_ref.shape != expected_shape:
            raise RuntimeError(
                "X_ref sai kích thước. "
                f"Expected {expected_shape}, "
                f"received {X_ref.shape}."
            )


        if not np.all(
            np.isfinite(
                X_ref
            )
        ):
            raise RuntimeError(
                "X_ref chứa NaN hoặc Inf."
            )


        # ====================================================
        # 6. ADVANCE INTERNAL REFERENCE
        #
        # IMPORTANT:
        #
        # The reference advances according to the controller
        # update period, NOT the MPC prediction step.
        #
        # p_ref,k+1
        #
        # =
        #
        # p_ref,k
        #
        # +
        #
        # dt_update * v_CoM_cmd
        # ====================================================

        self.p_ref = (
            self.p_ref
            +
            self.dt_update
            *
            v_CoM_cmd
        )


        return X_ref


    # ========================================================
    # GET CURRENT REFERENCE POSITION
    # ========================================================

    def get_position_reference(
        self
    ):
        """
        Return the current internally stored CoM position
        reference.
        """

        return self.p_ref.copy()