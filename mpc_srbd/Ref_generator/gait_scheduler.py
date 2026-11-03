import numpy as np


LEFT = 0
RIGHT = 1


class GaitScheduler:
    """
    Biped gait scheduler based on the HECTOR gait logic.

    Canonical gait cycle:

        SSP_L -> DSP1 -> SSP_R -> DSP2 -> repeat

    Contact interpretation:

        SSP_L:
            [L, R] = [1, 0]
            Left stance
            Right swing

        DSP1:
            [L, R] = [1, 1]

        SSP_R:
            [L, R] = [0, 1]
            Left swing
            Right stance

        DSP2:
            [L, R] = [1, 1]

    Open Duck startup:

        Robot starts in double support.

        Default:
            first_swing_leg = "R"

        Therefore scheduler starts from DSP2:

            DSP2
              ->
            SSP_L
              ->
            RIGHT swing
              ->
            DSP1
              ->
            SSP_R
              ->
            LEFT swing
              ->
            DSP2
              ->
            repeat

    IMPORTANT:

        This class does NOT contain any MuJoCo mj_step().

        It only generates:

            gait phase
            contact phase
            swing phase
            contact state
            swing state
            active swing leg
            swing duration
            stance duration
            MPC contact schedule
    """

    def __init__(
        self,
        mpc_horizon=10,
        dsp_durations=(3, 3),
        ssp_durations=(6, 6),
        control_dt=0.001,
        dt_mpc=0.04,
        startup_hold_time=0.25,
        first_swing_leg="R",
    ):

        # ====================================================
        # BASIC PARAMETERS
        # ====================================================

        self.mpc_horizon = int(
            mpc_horizon
        )

        self.dsp_durations = np.asarray(
            dsp_durations,
            dtype=int,
        ).reshape(
            2
        )

        self.ssp_durations = np.asarray(
            ssp_durations,
            dtype=int,
        ).reshape(
            2
        )

        self.control_dt = float(
            control_dt
        )

        self.dt_mpc = float(
            dt_mpc
        )

        self.startup_hold_time = float(
            startup_hold_time
        )

        self.first_swing_leg = (
            str(
                first_swing_leg
            )
            .strip()
            .upper()
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if self.first_swing_leg not in (
            "L",
            "R",
        ):

            raise ValueError(
                "first_swing_leg must be 'L' or 'R'."
            )


        if self.control_dt <= 0.0:

            raise ValueError(
                "control_dt must be positive."
            )


        if self.dt_mpc <= 0.0:

            raise ValueError(
                "dt_mpc must be positive."
            )


        if self.mpc_horizon <= 0:

            raise ValueError(
                "mpc_horizon must be positive."
            )


        if np.any(
            self.dsp_durations < 0
        ):

            raise ValueError(
                "dsp_durations must be non-negative."
            )


        if np.any(
            self.ssp_durations < 0
        ):

            raise ValueError(
                "ssp_durations must be non-negative."
            )


        if self.startup_hold_time < 0.0:

            raise ValueError(
                "startup_hold_time must be non-negative."
            )


        # ====================================================
        # PRECOMPUTE GAIT PARAMETERS
        # ====================================================

        self._recompute_parameters()


        # ====================================================
        # RESET INTERNAL STATE
        # ====================================================

        self.reset()


    # ========================================================
    # RECOMPUTE GAIT PARAMETERS
    # ========================================================

    def _recompute_parameters(self):

        # ----------------------------------------------------
        # Complete gait cycle:
        #
        # SSP_L + DSP1 + SSP_R + DSP2
        # ----------------------------------------------------

        self.gait_cycle_length = int(

            np.sum(
                self.ssp_durations
            )

            +

            np.sum(
                self.dsp_durations
            )
        )


        if self.gait_cycle_length <= 0:

            raise ValueError(
                "Total gait cycle length must be > 0."
            )


        # ----------------------------------------------------
        # Swing duration convention from paper:
        #
        # Left foot swings during SSP_R.
        # Right foot swings during SSP_L.
        # ----------------------------------------------------

        self.swing_durations = np.array(
            [
                self.ssp_durations[1],
                self.ssp_durations[0],
            ],
            dtype=int,
        )


        # ----------------------------------------------------
        # Stance duration:
        #
        # Left stance:
        #     SSP_L + DSP1 + DSP2
        #
        # Right stance:
        #     SSP_R + DSP1 + DSP2
        # ----------------------------------------------------

        dsp_total = int(
            np.sum(
                self.dsp_durations
            )
        )


        self.stance_durations = np.array(
            [
                self.ssp_durations[0]
                +
                dsp_total,

                self.ssp_durations[1]
                +
                dsp_total,
            ],
            dtype=int,
        )


        # ----------------------------------------------------
        # Durations in seconds
        # ----------------------------------------------------

        self.swing_durations_sec = (

            self.swing_durations.astype(
                float
            )

            *

            self.dt_mpc
        )


        self.stance_durations_sec = (

            self.stance_durations.astype(
                float
            )

            *

            self.dt_mpc
        )


        self.gait_duration_sec = (

            self.gait_cycle_length

            *

            self.dt_mpc
        )


        # ----------------------------------------------------
        # Gait boundaries in MPC-segment coordinates
        #
        # 0
        # |
        # | SSP_L
        # |
        # b1
        # |
        # | DSP1
        # |
        # b2
        # |
        # | SSP_R
        # |
        # b3
        # |
        # | DSP2
        # |
        # gait_cycle_length
        # ----------------------------------------------------

        self._b1 = int(
            self.ssp_durations[0]
        )


        self._b2 = int(

            self._b1

            +

            self.dsp_durations[0]
        )


        self._b3 = int(

            self._b2

            +

            self.ssp_durations[1]
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        # ----------------------------------------------------
        # Startup DS timer
        # ----------------------------------------------------

        self._startup_elapsed = 0.0


        # ----------------------------------------------------
        # Initial gait phase
        #
        # RIGHT swings first:
        #
        #     start at DSP2
        #     DSP2 -> SSP_L -> R swing
        #
        # LEFT swings first:
        #
        #     start at DSP1
        #     DSP1 -> SSP_R -> L swing
        # ----------------------------------------------------

        if self.first_swing_leg == "R":

            start_step = self._b3

        else:

            start_step = self._b1


        self._gait_phase = (

            float(
                start_step
            )

            /

            float(
                self.gait_cycle_length
            )
        )


    # ========================================================
    # STARTUP HOLD
    # ========================================================

    @property
    def in_startup_hold(self):

        return (

            self._startup_elapsed

            <

            self.startup_hold_time
        )


    # ========================================================
    # GAIT PHASE
    # ========================================================

    @property
    def gait_phase(self):

        return float(
            self._gait_phase
        )


    # ========================================================
    # CONTINUOUS GAIT STEP
    # ========================================================

    def _continuous_gait_step(self):
        """
        Continuous position inside the gait cycle.

        Unit:
            MPC segment

        Range:
            [0, gait_cycle_length)

        Numerical boundary handling is applied here so that
        values extremely close to an integer gait boundary are
        snapped exactly to that boundary.

        Example:

            5.999999999999999
                ->
            6.0

        This prevents a phase transition from being delayed by
        one control timestep due to floating-point accumulation.
        """

        gait_step = (

            self._gait_phase

            *

            float(
                self.gait_cycle_length
            )
        )


        nearest_integer = round(
            gait_step
        )


        if np.isclose(
            gait_step,
            nearest_integer,
            atol=1e-10,
            rtol=0.0,
        ):

            gait_step = float(
                nearest_integer
            )


        # ----------------------------------------------------
        # Protect against numerical value equal to cycle length
        # after rounding.
        # ----------------------------------------------------

        if gait_step >= self.gait_cycle_length:

            gait_step -= float(
                self.gait_cycle_length
            )


        return float(
            gait_step
        )


    # ========================================================
    # PHASE NAME
    # ========================================================

    def get_phase_name(self):

        if self.in_startup_hold:

            return "STARTUP_DS"


        s = self._continuous_gait_step()


        if s < self._b1:

            return "SSP_L"


        elif s < self._b2:

            return "DSP1"


        elif s < self._b3:

            return "SSP_R"


        else:

            return "DSP2"


    # ========================================================
    # CONTACT PHASE
    # ========================================================

    def get_contact_phase(self):
        """
        Return:

            [contact_phase_L,
             contact_phase_R]

        Convention:

            contact_phase >= 0
                foot is in stance/contact

            contact_phase = -1
                foot is in swing
        """

        # ----------------------------------------------------
        # Startup:
        # both feet are in contact.
        # ----------------------------------------------------

        if self.in_startup_hold:

            return np.array(
                [
                    0.0,
                    0.0,
                ],
                dtype=float,
            )


        s = self._continuous_gait_step()


        ssp_L = float(
            self.ssp_durations[0]
        )

        ssp_R = float(
            self.ssp_durations[1]
        )

        dsp_1 = float(
            self.dsp_durations[0]
        )

        dsp_2 = float(
            self.dsp_durations[1]
        )


        contact_phase = np.array(
            [
                -1.0,
                -1.0,
            ],
            dtype=float,
        )


        # ====================================================
        # LEFT FOOT
        #
        # Contact:
        #     SSP_L + DSP1
        #
        # Swing:
        #     SSP_R
        #
        # Contact:
        #     DSP2
        # ====================================================

        first_left_contact_duration = (

            ssp_L

            +

            dsp_1
        )


        if s < first_left_contact_duration:

            if first_left_contact_duration > 0.0:

                contact_phase[LEFT] = (

                    s

                    /

                    first_left_contact_duration
                )

            else:

                contact_phase[LEFT] = 0.0


        elif s < (

            first_left_contact_duration

            +

            ssp_R
        ):

            contact_phase[LEFT] = -1.0


        else:

            if dsp_2 > 0.0:

                contact_phase[LEFT] = (

                    s

                    -

                    (
                        first_left_contact_duration

                        +

                        ssp_R
                    )

                ) / dsp_2

            else:

                contact_phase[LEFT] = 0.0


        # ====================================================
        # RIGHT FOOT
        #
        # Swing:
        #     SSP_L
        #
        # Contact:
        #     DSP1 + SSP_R + DSP2
        # ====================================================

        if s < ssp_L:

            contact_phase[RIGHT] = -1.0


        else:

            right_contact_duration = (

                dsp_1

                +

                ssp_R

                +

                dsp_2
            )


            if right_contact_duration > 0.0:

                contact_phase[RIGHT] = (

                    s

                    -

                    ssp_L

                ) / right_contact_duration

            else:

                contact_phase[RIGHT] = 0.0


        return contact_phase


    # ========================================================
    # SWING PHASE
    # ========================================================

    def get_swing_phase(self):
        """
        Return:

            [swing_phase_L,
             swing_phase_R]

        Convention:

            0 <= swing_phase <= 1
                foot is swinging

            swing_phase = -1
                foot is not swinging
        """

        # ----------------------------------------------------
        # Startup DS:
        # no swing foot.
        # ----------------------------------------------------

        if self.in_startup_hold:

            return np.array(
                [
                    -1.0,
                    -1.0,
                ],
                dtype=float,
            )


        s = self._continuous_gait_step()


        ssp_L = float(
            self.ssp_durations[0]
        )

        ssp_R = float(
            self.ssp_durations[1]
        )

        dsp_1 = float(
            self.dsp_durations[0]
        )


        swing_phase = np.array(
            [
                -1.0,
                -1.0,
            ],
            dtype=float,
        )


        # ====================================================
        # RIGHT FOOT SWING
        #
        # Occurs during SSP_L
        # ====================================================

        if (
            ssp_L > 0.0
            and
            0.0 <= s < ssp_L
        ):

            swing_phase[RIGHT] = (

                s

                /

                ssp_L
            )


        # ====================================================
        # LEFT FOOT SWING
        #
        # Occurs during SSP_R
        # ====================================================

        left_swing_start = (

            ssp_L

            +

            dsp_1
        )


        left_swing_end = (

            left_swing_start

            +

            ssp_R
        )


        if (
            ssp_R > 0.0
            and
            left_swing_start <= s < left_swing_end
        ):

            swing_phase[LEFT] = (

                s

                -

                left_swing_start

            ) / ssp_R


        return swing_phase


    # ========================================================
    # BINARY CONTACT STATE
    # ========================================================

    def get_contact_state(self):

        contact_phase = (
            self.get_contact_phase()
        )


        return (

            contact_phase

            >=

            0.0

        ).astype(
            int
        )


    # ========================================================
    # BINARY SWING STATE
    # ========================================================

    def get_swing_state(self):

        swing_phase = (
            self.get_swing_phase()
        )


        return (

            swing_phase

            >=

            0.0

        ).astype(
            int
        )


    # ========================================================
    # ACTIVE SWING LEG
    # ========================================================

    def get_active_swing_leg(self):

        swing_state = (
            self.get_swing_state()
        )


        if (
            swing_state[LEFT] == 1
            and
            swing_state[RIGHT] == 0
        ):

            return "L"


        if (
            swing_state[LEFT] == 0
            and
            swing_state[RIGHT] == 1
        ):

            return "R"


        return None


    # ========================================================
    # DISCRETE CONTACT STATE FROM GAIT STEP
    # ========================================================

    def _contact_state_from_gait_step(
        self,
        gait_step,
    ):

        gait_step = int(
            gait_step
        ) % self.gait_cycle_length


        # ----------------------------------------------------
        # SSP_L
        #
        # Left stance
        # Right swing
        # ----------------------------------------------------

        if gait_step < self._b1:

            return np.array(
                [
                    1,
                    0,
                ],
                dtype=int,
            )


        # ----------------------------------------------------
        # DSP1
        # ----------------------------------------------------

        elif gait_step < self._b2:

            return np.array(
                [
                    1,
                    1,
                ],
                dtype=int,
            )


        # ----------------------------------------------------
        # SSP_R
        #
        # Left swing
        # Right stance
        # ----------------------------------------------------

        elif gait_step < self._b3:

            return np.array(
                [
                    0,
                    1,
                ],
                dtype=int,
            )


        # ----------------------------------------------------
        # DSP2
        # ----------------------------------------------------

        else:

            return np.array(
                [
                    1,
                    1,
                ],
                dtype=int,
            )


    # ========================================================
    # MPC CONTACT SCHEDULE
    # ========================================================

    def get_mpc_contact_schedule(self):
        """
        Return:

            schedule.shape = (mpc_horizon, 2)

        Columns:

            column 0 -> LEFT
            column 1 -> RIGHT

        Values:

            1 -> contact / stance
            0 -> swing

        The schedule follows the paper's mpc_gait() logic.

        IMPORTANT:

            This function uses the same corrected continuous
            gait-step calculation as the low-level phase logic.

            Therefore:

                phase_name
                contact_state
                swing_state
                mpc_contact_schedule

            all switch at the same gait boundary.
        """

        # ----------------------------------------------------
        # Startup hold:
        #
        # Keep future schedule in DS.
        #
        # This is sufficient for the current scheduler/swing
        # tests.
        #
        # Later, when MPC is integrated, we can optionally
        # account exactly for the remaining startup-hold time.
        # ----------------------------------------------------

        if self.in_startup_hold:

            return np.ones(
                (
                    self.mpc_horizon,
                    2,
                ),
                dtype=int,
            )


        # ----------------------------------------------------
        # Current discrete gait index
        # ----------------------------------------------------

        continuous_step = (
            self._continuous_gait_step()
        )


        current_gait_step = int(
            np.floor(
                continuous_step
            )
        )


        current_gait_step %= (
            self.gait_cycle_length
        )


        # ----------------------------------------------------
        # Allocate schedule
        # ----------------------------------------------------

        schedule = np.zeros(
            (
                self.mpc_horizon,
                2,
            ),
            dtype=int,
        )


        # ----------------------------------------------------
        # Predict future contact state
        # ----------------------------------------------------

        for k in range(
            self.mpc_horizon
        ):

            future_step = (

                current_gait_step

                +

                k

            ) % self.gait_cycle_length


            schedule[
                k,
                :
            ] = self._contact_state_from_gait_step(
                future_step
            )


        return schedule


    # ========================================================
    # COMPLETE OUTPUT
    # ========================================================

    def get_output(self):

        contact_phase = (
            self.get_contact_phase()
        )


        swing_phase = (
            self.get_swing_phase()
        )


        contact_state = (

            contact_phase

            >=

            0.0

        ).astype(
            int
        )


        swing_state = (

            swing_phase

            >=

            0.0

        ).astype(
            int
        )


        return {

            "phase_name":
                self.get_phase_name(),

            "gait_phase":
                self.gait_phase,

            "gait_step":
                self._continuous_gait_step(),

            "in_startup_hold":
                self.in_startup_hold,

            "contact_phase":
                contact_phase,

            "swing_phase":
                swing_phase,

            "contact_state":
                contact_state,

            "swing_state":
                swing_state,

            "active_swing_leg":
                self.get_active_swing_leg(),

            "T_swing":
                self.swing_durations_sec.copy(),

            "T_stance":
                self.stance_durations_sec.copy(),

            "gait_duration":
                float(
                    self.gait_duration_sec
                ),

            "mpc_contact_schedule":
                self.get_mpc_contact_schedule(),
        }


    # ========================================================
    # ADVANCE SCHEDULER
    # ========================================================

    def advance(self):
        """
        Advance the scheduler by ONE control timestep.

        Intended simulation order:

            x_k

                ->

            gait_output_k

                ->

            compute control u_k

                ->

            apply u_k

                ->

            mj_step()

                ->

            gait.advance()

                ->

            x_(k+1)

        IMPORTANT:

            No mj_step() is called inside this class.
        """

        dt_remaining = float(
            self.control_dt
        )


        # ====================================================
        # STARTUP HOLD
        # ====================================================

        if self.in_startup_hold:

            hold_remaining = (

                self.startup_hold_time

                -

                self._startup_elapsed
            )


            hold_dt = min(
                dt_remaining,
                hold_remaining,
            )


            self._startup_elapsed += (
                hold_dt
            )


            dt_remaining -= (
                hold_dt
            )


            # ------------------------------------------------
            # Numerical clamp
            # ------------------------------------------------

            if np.isclose(
                self._startup_elapsed,
                self.startup_hold_time,
                atol=1e-12,
                rtol=0.0,
            ):

                self._startup_elapsed = (
                    self.startup_hold_time
                )


        # ====================================================
        # GAIT PHASE
        # ====================================================

        if dt_remaining > 0.0:

            phase_increment = (

                dt_remaining

                /

                self.gait_duration_sec
            )


            self._gait_phase += (
                phase_increment
            )


            self._gait_phase %= 1.0


            # ------------------------------------------------
            # Numerical cleanup near 0 / 1
            # ------------------------------------------------

            if np.isclose(
                self._gait_phase,
                1.0,
                atol=1e-12,
                rtol=0.0,
            ):

                self._gait_phase = 0.0


            elif np.isclose(
                self._gait_phase,
                0.0,
                atol=1e-12,
                rtol=0.0,
            ):

                self._gait_phase = 0.0


    # ========================================================
    # UPDATE MPC SAMPLING TIME
    # ========================================================

    def update_mpc_sampling_time(
        self,
        dt_mpc,
    ):

        dt_mpc = float(
            dt_mpc
        )


        if dt_mpc <= 0.0:

            raise ValueError(
                "dt_mpc must be positive."
            )


        self.dt_mpc = (
            dt_mpc
        )


        self._recompute_parameters()


    # ========================================================
    # UPDATE GAIT PARAMETERS
    # ========================================================

    def update_gait_parameters(
        self,
        dsp_durations,
        ssp_durations,
        reset=True,
    ):

        self.dsp_durations = np.asarray(
            dsp_durations,
            dtype=int,
        ).reshape(
            2
        )


        self.ssp_durations = np.asarray(
            ssp_durations,
            dtype=int,
        ).reshape(
            2
        )


        if np.any(
            self.dsp_durations < 0
        ):

            raise ValueError(
                "dsp_durations must be non-negative."
            )


        if np.any(
            self.ssp_durations < 0
        ):

            raise ValueError(
                "ssp_durations must be non-negative."
            )


        self._recompute_parameters()


        if reset:

            self.reset()