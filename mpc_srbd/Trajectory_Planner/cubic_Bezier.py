import numpy as np


# ============================================================
# CUBIC BEZIER SWING-FOOT TRAJECTORY
# ============================================================

class CubicBezierSwingTrajectory:
    """
    Online cubic-Bezier swing-foot trajectory generator.

    The trajectory is generated online at the low-level
    controller frequency.

    Main idea
    ---------

    When a foot changes from stance to swing:

        1. Save its actual current World position as:

               p0

        2. Reset swing phase:

               phi = 0

    During the swing phase:

        p0

    remains constant.

    The Raibert foothold:

        p3 = W_p_foothold

    may be updated at every low-level control cycle.

    The intermediate Bezier control points are:

        p1 = p0 + (1/3) * (p3 - p0)

        p2 = p0 + (2/3) * (p3 - p0)

    The cubic Bezier trajectory is:

        B(phi)
            =
        (1-phi)^3 p0
            +
        3(1-phi)^2 phi p1
            +
        3(1-phi) phi^2 p2
            +
        phi^3 p3

    where:

        0 <= phi <= 1

    The swing phase is updated according to:

        phi_(k+1)
            =
        phi_k
            +
        dt_low / T_s

    A vertical foot-clearance profile is added:

        z_clearance
            =
        4 * h_swing * phi * (1 - phi)

    Therefore:

        z_clearance = 0

    at:

        phi = 0

    and:

        phi = 1

    while:

        z_clearance = h_swing

    at:

        phi = 0.5
    """


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        dt_low,
        T_s,
        h_swing
    ):

        # ----------------------------------------------------
        # Convert parameters
        # ----------------------------------------------------

        dt_low = float(
            dt_low
        )

        T_s = float(
            T_s
        )

        h_swing = float(
            h_swing
        )


        # ----------------------------------------------------
        # Validate dt_low
        # ----------------------------------------------------

        if (
            not np.isfinite(
                dt_low
            )
            or
            dt_low <= 0.0
        ):

            raise ValueError(
                "dt_low phải là số hữu hạn và > 0."
            )


        # ----------------------------------------------------
        # Validate swing duration
        # ----------------------------------------------------

        if (
            not np.isfinite(
                T_s
            )
            or
            T_s <= 0.0
        ):

            raise ValueError(
                "T_s phải là số hữu hạn và > 0."
            )


        # ----------------------------------------------------
        # Validate swing height
        # ----------------------------------------------------

        if (
            not np.isfinite(
                h_swing
            )
            or
            h_swing < 0.0
        ):

            raise ValueError(
                "h_swing phải là số hữu hạn và >= 0."
            )


        # ====================================================
        # STORE PARAMETERS
        # ====================================================

        self.dt_low = dt_low

        self.T_s = T_s

        self.h_swing = h_swing


        # ----------------------------------------------------
        # Swing-phase increment
        #
        #     delta_phi = dt_low / T_s
        # ----------------------------------------------------

        self.delta_phi = (
            self.dt_low
            /
            self.T_s
        )


        # ====================================================
        # INTERNAL STATE
        # ====================================================

        # Current swing phase:
        #
        #     0 <= phi <= 1

        self.phi = 0.0


        # Saved liftoff position:
        #
        #     p0 = W_p_i6(t_liftoff)

        self.p0 = None


        # Currently active swing foot:
        #
        #     "L"
        #     "R"
        #     None

        self.previous_swing_foot = None


    # ========================================================
    # VECTOR VALIDATION
    # ========================================================

    @staticmethod
    def _validate_vector(
        value,
        name
    ):

        value = np.asarray(
            value,
            dtype=float
        ).reshape(-1)


        if value.shape != (
            3,
        ):

            raise ValueError(
                f"{name} phải có shape (3,), "
                f"nhưng nhận được {value.shape}."
            )


        if not np.all(
            np.isfinite(
                value
            )
        ):

            raise ValueError(
                f"{name} chứa NaN hoặc Inf."
            )


        return value


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self
    ):
        """
        Reset the internal swing-trajectory state.
        """

        self.phi = 0.0

        self.p0 = None

        self.previous_swing_foot = None


    # ========================================================
    # UPDATE TRAJECTORY
    # ========================================================

    def update(
        self,
        W_p_L6,
        W_p_R6,
        W_p_foothold,
        swing_foot
    ):
        """
        Compute the current desired swing-foot position.

        Parameters
        ----------
        W_p_L6 : array_like, shape (3,)

            Current physical left-foot position expressed
            in the World frame:

                {}^W p_L6


        W_p_R6 : array_like, shape (3,)

            Current physical right-foot position expressed
            in the World frame:

                {}^W p_R6


        W_p_foothold : array_like, shape (3,), or None

            Current desired foothold generated by the
            Raibert heuristic:

                {}^W p_foothold

            During Double Support:

                W_p_foothold = None


        swing_foot : str or None

            Current swing foot:

                "L"  -> left foot is swinging

                "R"  -> right foot is swinging

                None -> Double Support


        Returns
        -------
        W_p_swing_des : np.ndarray, shape (3,), or None

            Current desired physical swing-foot position
            expressed in the World frame:

                {}^W p_i6^des

            Returns None during Double Support.
        """


        # ====================================================
        # 1. VALIDATE PHYSICAL FOOT POSITIONS
        # ====================================================

        W_p_L6 = self._validate_vector(
            W_p_L6,
            "W_p_L6"
        )


        W_p_R6 = self._validate_vector(
            W_p_R6,
            "W_p_R6"
        )


        # ====================================================
        # 2. VALIDATE SWING FOOT
        # ====================================================

        if swing_foot not in (
            "L",
            "R",
            None
        ):

            raise ValueError(
                "swing_foot chỉ được là 'L', 'R' hoặc None."
            )


        # ====================================================
        # 3. DOUBLE SUPPORT
        #
        # During DS:
        #
        #     swing_foot = None
        #
        # There is no swing trajectory.
        #
        # Reset the active swing state so that the next
        # stance -> swing transition can be detected.
        # ====================================================

        if swing_foot is None:

            self.phi = 0.0

            self.p0 = None

            self.previous_swing_foot = None

            return None


        # ====================================================
        # 4. VALIDATE FOOTHOLD
        # ====================================================

        if W_p_foothold is None:

            raise ValueError(
                "W_p_foothold không được là None "
                "khi đang có swing foot."
            )


        W_p_foothold = self._validate_vector(
            W_p_foothold,
            "W_p_foothold"
        )


        # ====================================================
        # 5. DETECT NEW SWING PHASE
        #
        # A new swing phase occurs when:
        #
        #     previous_swing_foot != swing_foot
        #
        # Example:
        #
        #     DS -> LSS
        #
        # gives:
        #
        #     previous_swing_foot = None
        #
        #     swing_foot = "R"
        #
        #
        # At that instant:
        #
        #     p0 = current physical position of R6
        #
        # and:
        #
        #     phi = 0
        # ====================================================

        new_swing_phase = (
            self.previous_swing_foot
            !=
            swing_foot
        )


        if new_swing_phase:

            # ------------------------------------------------
            # Left foot starts swing
            # ------------------------------------------------

            if swing_foot == "L":

                self.p0 = (
                    W_p_L6.copy()
                )


            # ------------------------------------------------
            # Right foot starts swing
            # ------------------------------------------------

            else:

                self.p0 = (
                    W_p_R6.copy()
                )


            # ------------------------------------------------
            # Reset phase
            # ------------------------------------------------

            self.phi = 0.0


        # ====================================================
        # 6. INTERNAL STATE VALIDATION
        # ====================================================

        if self.p0 is None:

            raise RuntimeError(
                "p0 chưa được khởi tạo cho swing phase."
            )


        # ====================================================
        # 7. CURRENT SWING PHASE
        # ====================================================

        phi = float(
            np.clip(
                self.phi,
                0.0,
                1.0
            )
        )


        # ====================================================
        # 8. BEZIER CONTROL POINTS
        #
        # p0:
        #
        #     saved physical foot position at liftoff
        #
        #
        # p3:
        #
        #     current Raibert foothold
        #
        #
        # p1:
        #
        #     p0 + 1/3 (p3 - p0)
        #
        #
        # p2:
        #
        #     p0 + 2/3 (p3 - p0)
        #
        # ====================================================

        p0 = (
            self.p0.copy()
        )


        p3 = (
            W_p_foothold.copy()
        )


        p1 = (
            p0
            +
            (1.0 / 3.0)
            *
            (
                p3
                -
                p0
            )
        )


        p2 = (
            p0
            +
            (2.0 / 3.0)
            *
            (
                p3
                -
                p0
            )
        )


        # ====================================================
        # 9. CUBIC BEZIER
        #
        # B(phi)
        #
        # =
        #
        # (1-phi)^3 p0
        #
        # +
        #
        # 3(1-phi)^2 phi p1
        #
        # +
        #
        # 3(1-phi) phi^2 p2
        #
        # +
        #
        # phi^3 p3
        # ====================================================

        one_minus_phi = (
            1.0
            -
            phi
        )


        W_p_swing_des = (

            one_minus_phi**3
            *
            p0

            +

            3.0
            *
            one_minus_phi**2
            *
            phi
            *
            p1

            +

            3.0
            *
            one_minus_phi
            *
            phi**2
            *
            p2

            +

            phi**3
            *
            p3

        )


        # ====================================================
        # 10. VERTICAL SWING CLEARANCE
        #
        # Add:
        #
        #     z_clearance
        #
        # =
        #
        #     4 h_swing phi (1-phi)
        #
        #
        # Properties:
        #
        # phi = 0:
        #
        #     z_clearance = 0
        #
        #
        # phi = 0.5:
        #
        #     z_clearance = h_swing
        #
        #
        # phi = 1:
        #
        #     z_clearance = 0
        # ====================================================

        z_clearance = (
            4.0
            *
            self.h_swing
            *
            phi
            *
            (
                1.0
                -
                phi
            )
        )


        W_p_swing_des[
            2
        ] += z_clearance


        # ====================================================
        # 11. OUTPUT VALIDATION
        # ====================================================

        if W_p_swing_des.shape != (
            3,
        ):

            raise RuntimeError(
                "W_p_swing_des phải có shape (3,), "
                f"nhưng nhận được "
                f"{W_p_swing_des.shape}."
            )


        if not np.all(
            np.isfinite(
                W_p_swing_des
            )
        ):

            raise RuntimeError(
                "W_p_swing_des chứa NaN hoặc Inf."
            )


        # ====================================================
        # 12. UPDATE SWING PHASE
        #
        # According to:
        #
        #     phi_(k+1)
        #
        # =
        #
        #     phi_k
        #
        # +
        #
        #     dt_low / T_s
        #
        # ====================================================

        self.phi = min(
            1.0,
            self.phi
            +
            self.delta_phi
        )


        # ====================================================
        # 13. STORE ACTIVE SWING FOOT
        # ====================================================

        self.previous_swing_foot = (
            swing_foot
        )


        # ====================================================
        # 14. RETURN CURRENT DESIRED SWING POSITION
        # ====================================================

        return W_p_swing_des


    # ========================================================
    # GET CURRENT PHASE
    # ========================================================

    def get_phase(
        self
    ):
        """
        Return the internally stored swing phase.
        """

        return float(
            self.phi
        )


    # ========================================================
    # GET LIFTOFF POSITION
    # ========================================================

    def get_liftoff_position(
        self
    ):
        """
        Return the saved physical foot position at liftoff.

        Returns
        -------
        p0 : np.ndarray, shape (3,), or None
        """

        if self.p0 is None:

            return None


        return self.p0.copy()


    # ========================================================
    # GET ACTIVE SWING FOOT
    # ========================================================

    def get_active_swing_foot(
        self
    ):
        """
        Return the currently active swing-foot identity.
        """

        return self.previous_swing_foot