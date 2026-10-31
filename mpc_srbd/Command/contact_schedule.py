import numpy as np


# ============================================================
# CONTACT STATES
# ============================================================

DOUBLE_SUPPORT = np.array(
    [
        1,
        1
    ],
    dtype=int
)

LEFT_SINGLE_SUPPORT = np.array(
    [
        1,
        0
    ],
    dtype=int
)

RIGHT_SINGLE_SUPPORT = np.array(
    [
        0,
        1
    ],
    dtype=int
)


# ============================================================
# GET CONTACT SCHEDULE
# ============================================================

def get_contact_schedule(
    current_time,
    N,
    dt_mpc
):
    """
    Generate the open-loop contact schedule over the current
    MPC prediction horizon.

    Gait sequence:

        Double Support
            ↓
        Left Single Support
            ↓
        Double Support
            ↓
        Right Single Support
            ↓
        repeat


    Contact convention
    ------------------

        [c_L, c_R]

    Double Support:

        [1, 1]

    Left Single Support:

        [1, 0]

    Right Single Support:

        [0, 1]


    Contact durations
    -----------------

        T_DS = 2 * dt_mpc

        T_SS = 8 * dt_mpc


    Complete gait cycle
    -------------------

        DS
        +
        LSS
        +
        DS
        +
        RSS

    Therefore:

        T_cycle = 20 * dt_mpc


    Parameters
    ----------
    current_time : float

        Current simulation / MPC time.


    N : int

        MPC prediction horizon.


    dt_mpc : float

        MPC prediction discretization period.


    Returns
    -------
    contact_schedule : np.ndarray

        Shape:

            (N, 2)

        Columns:

            [:, 0] = left foot contact

            [:, 1] = right foot contact
    """


    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    current_time = float(
        current_time
    )


    if not np.isfinite(
        current_time
    ):

        raise ValueError(
            "current_time phải finite."
        )


    if current_time < 0.0:

        raise ValueError(
            "current_time không được nhỏ hơn 0."
        )


    if not isinstance(
        N,
        (int, np.integer)
    ):

        raise TypeError(
            "N phải là số nguyên."
        )


    if N <= 0:

        raise ValueError(
            "N phải lớn hơn 0."
        )


    dt_mpc = float(
        dt_mpc
    )


    if not np.isfinite(
        dt_mpc
    ):

        raise ValueError(
            "dt_mpc phải finite."
        )


    if dt_mpc <= 0.0:

        raise ValueError(
            "dt_mpc phải lớn hơn 0."
        )


    # ========================================================
    # 2. CONTACT PHASE LENGTHS
    #
    # Express the gait directly in units of dt_mpc.
    #
    # DS  = 2 prediction intervals
    # SS  = 8 prediction intervals
    # ========================================================

    N_DS = 2

    N_SS = 8


    N_cycle = (
        2 * N_DS
        +
        2 * N_SS
    )


    # ========================================================
    # 3. PHASE BOUNDARIES
    #
    # Dimensionless gait phase:
    #
    #     0 <= phase < 20
    #
    # Phase definition:
    #
    #     [0, 2)    -> DS
    #
    #     [2, 10)   -> LSS
    #
    #     [10, 12)  -> DS
    #
    #     [12, 20)  -> RSS
    # ========================================================

    phase_DS_1_end = (
        N_DS
    )


    phase_LSS_end = (
        N_DS
        +
        N_SS
    )


    phase_DS_2_end = (
        2 * N_DS
        +
        N_SS
    )


    # ========================================================
    # 4. INITIALIZE CONTACT SCHEDULE
    # ========================================================

    contact_schedule = np.zeros(
        (
            N,
            2
        ),
        dtype=int
    )


    # ========================================================
    # 5. NUMERICAL TOLERANCE
    #
    # Used only to remove floating-point errors very close
    # to exact gait-phase boundaries.
    # ========================================================

    phase_tolerance = 1e-10


    # ========================================================
    # 6. BUILD CONTACT SCHEDULE
    # ========================================================

    for k in range(
        N
    ):

        # ----------------------------------------------------
        # Prediction time
        #
        #     t_k = current_time + k * dt_mpc
        # ----------------------------------------------------

        prediction_time = (
            current_time
            +
            k
            *
            dt_mpc
        )


        # ----------------------------------------------------
        # Convert time into dimensionless gait phase.
        #
        # Example:
        #
        #     prediction_time = 0.30 s
        #     dt_mpc = 0.025 s
        #
        #     phase = 12
        #
        # One full gait cycle:
        #
        #     phase in [0, 20)
        # ----------------------------------------------------

        phase = np.mod(
            prediction_time
            /
            dt_mpc,
            N_cycle
        )


        # ----------------------------------------------------
        # Correct floating-point error near exact integer
        # phase boundaries.
        #
        # Example:
        #
        #     11.999999999999998
        #
        # becomes:
        #
        #     12.0
        # ----------------------------------------------------

        nearest_integer = round(
            phase
        )


        if abs(
            phase
            -
            nearest_integer
        ) < phase_tolerance:

            phase = float(
                nearest_integer
                %
                N_cycle
            )


        # ====================================================
        # PHASE 1:
        #
        # DOUBLE SUPPORT
        #
        # 0 <= phase < 2
        # ====================================================

        if phase < phase_DS_1_end:

            contact_schedule[
                k,
                :
            ] = DOUBLE_SUPPORT


        # ====================================================
        # PHASE 2:
        #
        # LEFT SINGLE SUPPORT
        #
        # 2 <= phase < 10
        # ====================================================

        elif phase < phase_LSS_end:

            contact_schedule[
                k,
                :
            ] = LEFT_SINGLE_SUPPORT


        # ====================================================
        # PHASE 3:
        #
        # DOUBLE SUPPORT
        #
        # 10 <= phase < 12
        # ====================================================

        elif phase < phase_DS_2_end:

            contact_schedule[
                k,
                :
            ] = DOUBLE_SUPPORT


        # ====================================================
        # PHASE 4:
        #
        # RIGHT SINGLE SUPPORT
        #
        # 12 <= phase < 20
        # ====================================================

        else:

            contact_schedule[
                k,
                :
            ] = RIGHT_SINGLE_SUPPORT


    # ========================================================
    # 7. FINAL VALIDATION
    # ========================================================

    if contact_schedule.shape != (
        N,
        2
    ):

        raise RuntimeError(
            "contact_schedule phải có shape (N, 2)."
        )


    if not np.all(
        np.isin(
            contact_schedule,
            [
                0,
                1
            ]
        )
    ):

        raise RuntimeError(
            "contact_schedule chỉ được chứa 0 hoặc 1."
        )


    # ========================================================
    # 8. RETURN
    # ========================================================

    return contact_schedule


# ============================================================
# GET CURRENT CONTACT INFO
# ============================================================

def get_current_contact_info(
    current_time,
    dt_mpc
):
    """
    Get the current contact state and swing-foot information
    required by the foothold planner.

    Contact convention
    ------------------

        [c_L, c_R]

    where:

        c_i = 1  -> stance / contact

        c_i = 0  -> swing / no contact


    Gait sequence
    -------------

        DS
        ->
        LSS
        ->
        DS
        ->
        RSS
        ->
        repeat


    Swing-foot convention
    ---------------------

        [1, 0] -> right foot is swing

        [0, 1] -> left foot is swing

        [1, 1] -> no swing foot


    Remaining swing time
    --------------------

    delta_T is the remaining time until the current
    single-support phase ends.

    During Double Support:

        delta_T = None


    Parameters
    ----------
    current_time : float

        Current simulation / control time [s].


    dt_mpc : float

        MPC prediction discretization period [s].


    Returns
    -------
    current_contact : np.ndarray, shape (2,)

        Current contact state:

            [c_L, c_R]


    swing_foot : str or None

        "L"  -> left foot is swing

        "R"  -> right foot is swing

        None -> Double Support


    delta_T : float or None

        Remaining time of the current swing phase [s].

        None during Double Support.
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    current_time = float(
        current_time
    )

    if not np.isfinite(
        current_time
    ):
        raise ValueError(
            "current_time phải finite."
        )

    if current_time < 0.0:
        raise ValueError(
            "current_time không được nhỏ hơn 0."
        )


    dt_mpc = float(
        dt_mpc
    )

    if not np.isfinite(
        dt_mpc
    ):
        raise ValueError(
            "dt_mpc phải finite."
        )

    if dt_mpc <= 0.0:
        raise ValueError(
            "dt_mpc phải lớn hơn 0."
        )


    # ========================================================
    # 2. GET CURRENT CONTACT STATE
    #
    # Reuse the already tested contact-schedule function.
    #
    # N = 1:
    #
    #     schedule[0]
    #
    # is exactly the contact state at current_time.
    # ========================================================

    current_contact = get_contact_schedule(
        current_time=current_time,
        N=1,
        dt_mpc=dt_mpc
    )[0].copy()


    # ========================================================
    # 3. GAIT PHASE PARAMETERS
    # ========================================================

    N_DS = 2

    N_SS = 8

    N_cycle = (
        2 * N_DS
        +
        2 * N_SS
    )


    phase_LSS_end = (
        N_DS
        +
        N_SS
    )

    phase_RSS_end = (
        N_cycle
    )


    # ========================================================
    # 4. CURRENT DIMENSIONLESS GAIT PHASE
    #
    #     0 <= phase < 20
    #
    #     [0, 2)   -> DS
    #
    #     [2, 10)  -> LSS
    #
    #     [10, 12) -> DS
    #
    #     [12, 20) -> RSS
    # ========================================================

    phase = np.mod(
        current_time
        /
        dt_mpc,
        N_cycle
    )


    # --------------------------------------------------------
    # Floating-point correction near exact boundaries
    # --------------------------------------------------------

    phase_tolerance = 1e-10

    nearest_integer = round(
        phase
    )

    if abs(
        phase
        -
        nearest_integer
    ) < phase_tolerance:

        phase = float(
            nearest_integer
            %
            N_cycle
        )


    # ========================================================
    # 5. DETERMINE SWING FOOT AND REMAINING SWING TIME
    # ========================================================

    # --------------------------------------------------------
    # DOUBLE SUPPORT
    #
    # [1, 1]
    # --------------------------------------------------------

    if np.array_equal(
        current_contact,
        DOUBLE_SUPPORT
    ):

        swing_foot = None

        delta_T = None


    # --------------------------------------------------------
    # LEFT SINGLE SUPPORT
    #
    # Left foot  = stance
    # Right foot = swing
    #
    # [1, 0]
    #
    # Remaining time:
    #
    #     delta_T =
    #         (phase_LSS_end - phase) * dt_mpc
    # --------------------------------------------------------

    elif np.array_equal(
        current_contact,
        LEFT_SINGLE_SUPPORT
    ):

        swing_foot = "R"

        delta_T = (
            phase_LSS_end
            -
            phase
        ) * dt_mpc


    # --------------------------------------------------------
    # RIGHT SINGLE SUPPORT
    #
    # Right foot = stance
    # Left foot  = swing
    #
    # [0, 1]
    #
    # Remaining time:
    #
    #     delta_T =
    #         (phase_RSS_end - phase) * dt_mpc
    # --------------------------------------------------------

    elif np.array_equal(
        current_contact,
        RIGHT_SINGLE_SUPPORT
    ):

        swing_foot = "L"

        delta_T = (
            phase_RSS_end
            -
            phase
        ) * dt_mpc


    # --------------------------------------------------------
    # Invalid contact state
    # --------------------------------------------------------

    else:

        raise RuntimeError(
            "Current contact state không hợp lệ: "
            f"{current_contact}."
        )


    # ========================================================
    # 6. OUTPUT VALIDATION
    # ========================================================

    if current_contact.shape != (
        2,
    ):

        raise RuntimeError(
            "current_contact phải có shape (2,)."
        )


    if swing_foot is not None:

        if swing_foot not in (
            "L",
            "R"
        ):

            raise RuntimeError(
                "swing_foot chỉ được là 'L', 'R' hoặc None."
            )


    if delta_T is not None:

        delta_T = float(
            delta_T
        )

        if not np.isfinite(
            delta_T
        ):

            raise RuntimeError(
                "delta_T phải finite."
            )

        if delta_T <= 0.0:

            raise RuntimeError(
                "delta_T phải lớn hơn 0 khi đang ở Single Support."
            )


    # ========================================================
    # 7. RETURN
    # ========================================================

    return (
        current_contact,
        swing_foot,
        delta_T
    )