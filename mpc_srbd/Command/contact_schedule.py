import numpy as np


# ============================================================
# IMPORT GAIT TIMING
#
# Support both:
#
#   python contact_schedule.py
#
# and:
#
#   from Command.contact_schedule import ...
# ============================================================

try:

    from .gait_timing import (
        get_nominal_phase_duration
    )

except ImportError:

    from gait_timing import (
        get_nominal_phase_duration
    )


# ============================================================
# GAIT PHASE DEFINITIONS
# ============================================================

VALID_GAIT_PHASES = (
    "LSS",
    "DS_L_TO_R",
    "RSS",
    "DS_R_TO_L"
)


# ============================================================
# PHYSICAL CONTACT STATES
#
# [c_L, c_R]
# ============================================================

PHASE_CONTACT_STATE = {

    "LSS": np.array(
        [1, 0],
        dtype=int
    ),

    "DS_L_TO_R": np.array(
        [1, 1],
        dtype=int
    ),

    "RSS": np.array(
        [0, 1],
        dtype=int
    ),

    "DS_R_TO_L": np.array(
        [1, 1],
        dtype=int
    ),
}


# ============================================================
# NOMINAL PHASE ORDER
# ============================================================

NEXT_GAIT_PHASE = {

    "LSS":
        "DS_L_TO_R",

    "DS_L_TO_R":
        "RSS",

    "RSS":
        "DS_R_TO_L",

    "DS_R_TO_L":
        "LSS",
}


# ============================================================
# EXPECTED PHYSICAL CONTACT EVENT
#
# Actual phase transitions are triggered by these events.
# ============================================================

EXPECTED_TRANSITION_EVENT = {

    "LSS":
        "TD_R",

    "DS_L_TO_R":
        "LO_L",

    "RSS":
        "TD_L",

    "DS_R_TO_L":
        "LO_R",
}


# ============================================================
# NORMALIZE GAIT PHASE
# ============================================================

def normalize_gait_phase(
    phase
):
    """
    Normalize and validate a gait-phase name.

    Valid phases
    ------------

        LSS

        DS_L_TO_R

        RSS

        DS_R_TO_L


    Returns
    -------
    phase : str
    """


    phase = str(
        phase
    ).upper()


    if phase not in VALID_GAIT_PHASES:

        raise ValueError(
            "Unknown gait phase: "
            f"{phase}"
        )


    return phase


# ============================================================
# GET PHYSICAL CONTACT STATE FROM PHASE
# ============================================================

def get_phase_contact_state(
    phase
):
    """
    Return physical contact state:

        [c_L, c_R]

    for the requested gait phase.


    Mapping
    -------

        LSS
            -> [1, 0]

        DS_L_TO_R
            -> [1, 1]

        RSS
            -> [0, 1]

        DS_R_TO_L
            -> [1, 1]
    """


    phase = normalize_gait_phase(
        phase
    )


    return PHASE_CONTACT_STATE[
        phase
    ].copy()


# ============================================================
# GET NEXT NOMINAL GAIT PHASE
# ============================================================

def get_next_gait_phase(
    phase
):
    """
    Return the next phase in the nominal cyclic gait:

        LSS
          ->
        DS_L_TO_R
          ->
        RSS
          ->
        DS_R_TO_L
          ->
        LSS
    """


    phase = normalize_gait_phase(
        phase
    )


    return NEXT_GAIT_PHASE[
        phase
    ]


# ============================================================
# GET EXPECTED CONTACT EVENT
# ============================================================

def get_expected_transition_event(
    phase
):
    """
    Return the physical contact event required to leave
    the current actual gait phase.


    FSM
    ---

        LSS
            -- TD_R -->
        DS_L_TO_R


        DS_L_TO_R
            -- LO_L -->
        RSS


        RSS
            -- TD_L -->
        DS_R_TO_L


        DS_R_TO_L
            -- LO_R -->
        LSS
    """


    phase = normalize_gait_phase(
        phase
    )


    return EXPECTED_TRANSITION_EVENT[
        phase
    ]


# ============================================================
# VALIDATE PHASE TIMER
# ============================================================

def validate_phase_time(
    t_phase
):
    """
    Validate elapsed time in the current actual gait phase.
    """


    t_phase = float(
        t_phase
    )


    if not np.isfinite(
        t_phase
    ):

        raise ValueError(
            "t_phase chứa NaN hoặc Inf."
        )


    if t_phase < 0.0:

        raise ValueError(
            "t_phase không được âm."
        )


    return t_phase


# ============================================================
# ADVANCE PHASE TIMER
# ============================================================

def advance_phase_timer(
    t_phase,
    dt_elapsed
):
    """
    Advance the actual gait-phase timer.

    This function ONLY updates elapsed time.

    It does NOT change gait phase.

    Actual gait-phase transitions are handled by:

        synchronize_gait_phase(...)


    Parameters
    ----------
    t_phase : float

        Current phase elapsed time [s].


    dt_elapsed : float

        Elapsed simulation/control time [s].


    Returns
    -------
    t_phase_new : float
    """


    t_phase = validate_phase_time(
        t_phase
    )


    dt_elapsed = float(
        dt_elapsed
    )


    if not np.isfinite(
        dt_elapsed
    ):

        raise ValueError(
            "dt_elapsed chứa NaN hoặc Inf."
        )


    if dt_elapsed < 0.0:

        raise ValueError(
            "dt_elapsed không được âm."
        )


    return (
        t_phase
        +
        dt_elapsed
    )


# ============================================================
# VALIDATE CONTACT EVENTS
# ============================================================

def validate_contact_events(
    events
):
    """
    Validate event dictionary generated by:

        contact_switch.detect_contact_events(...)

    Required keys:

        TD_L

        TD_R

        LO_L

        LO_R
    """


    if not isinstance(
        events,
        dict
    ):

        raise TypeError(
            "events phải là dictionary."
        )


    required_keys = (
        "TD_L",
        "TD_R",
        "LO_L",
        "LO_R"
    )


    for key in required_keys:

        if key not in events:

            raise KeyError(
                f"events thiếu key '{key}'."
            )


        if not isinstance(
            events[key],
            (
                bool,
                np.bool_
            )
        ):

            raise TypeError(
                f"events['{key}'] phải là bool."
            )


    return events


# ============================================================
# SYNCHRONIZE ACTUAL GAIT PHASE
# ============================================================

def synchronize_gait_phase(
    current_phase,
    t_phase,
    events
):
    """
    Synchronize the ACTUAL gait phase using contact-switch
    events.

    Important
    ---------

    The actual gait phase is NOT changed by nominal timing.

    It changes ONLY when the expected physical contact
    event occurs.


    FSM
    ---

        LSS
            -- TD_R -->
        DS_L_TO_R


        DS_L_TO_R
            -- LO_L -->
        RSS


        RSS
            -- TD_L -->
        DS_R_TO_L


        DS_R_TO_L
            -- LO_R -->
        LSS


    Parameters
    ----------
    current_phase : str

        Current synchronized gait phase.


    t_phase : float

        Elapsed time in current phase [s].


    events : dict

        Output of:

            detect_contact_events(...)


    Returns
    -------
    phase_new : str

        Updated actual gait phase.


    t_phase_new : float

        Updated phase timer.

        If transition occurs:

            t_phase_new = 0


        Otherwise:

            t_phase_new = t_phase


    transitioned : bool

        True if the expected contact event caused a
        gait-phase transition.


    transition_event : str or None

        Event responsible for transition.

        Example:

            "TD_R"

        or:

            None
    """


    current_phase = normalize_gait_phase(
        current_phase
    )


    t_phase = validate_phase_time(
        t_phase
    )


    events = validate_contact_events(
        events
    )


    # ========================================================
    # EXPECTED EVENT FOR CURRENT PHASE
    # ========================================================

    expected_event = (
        get_expected_transition_event(
            current_phase
        )
    )


    # ========================================================
    # ACTUAL CONTACT EVENT OCCURRED
    # ========================================================

    if events[
        expected_event
    ]:

        phase_new = (
            get_next_gait_phase(
                current_phase
            )
        )


        t_phase_new = 0.0


        transitioned = True


        transition_event = (
            expected_event
        )


        return (
            phase_new,
            t_phase_new,
            transitioned,
            transition_event
        )


    # ========================================================
    # NO VALID TRANSITION
    #
    # Unexpected events do NOT advance the FSM.
    # ========================================================

    return (
        current_phase,
        t_phase,
        False,
        None
    )


# ============================================================
# GET NOMINAL REMAINING PHASE TIME
# ============================================================

def get_nominal_remaining_phase_time(
    phase,
    t_phase,
    timing
):
    """
    Return remaining nominal time in the current phase.

    If the actual phase has already exceeded its nominal
    duration because the expected contact event has not
    occurred, this function returns 0.

    This does NOT force an actual phase transition.
    """


    phase = normalize_gait_phase(
        phase
    )


    t_phase = validate_phase_time(
        t_phase
    )


    T_phase = (
        get_nominal_phase_duration(
            phase,
            timing
        )
    )


    remaining_time = max(
        T_phase - t_phase,
        0.0
    )


    return remaining_time


# ============================================================
# BUILD NOMINAL PHASE PREVIEW
# ============================================================

def build_phase_preview(
    current_phase,
    t_phase,
    timing,
    horizon=None,
    dt_mpc=None
):
    """
    Build the nominal future gait-phase sequence used by MPC.

    Important distinction
    ---------------------

    Index j = 0 always uses the ACTUAL synchronized phase.

    Future indices:

        j = 1, ..., N-1

    are rolled forward using nominal timing.


    Therefore:

        current phase
            =
        physical / contact-synchronized


    while:

        future phases
            =
        nominal prediction


    This is consistent with receding-horizon MPC:

        measure actual state/contact now

            ->

        predict nominal future

            ->

        solve QP

            ->

        apply only first control

            ->

        measure again


    Parameters
    ----------
    current_phase : str

        Current synchronized actual gait phase.


    t_phase : float

        Elapsed time in current phase [s].


    timing : dict

        Output of:

            build_gait_timing(...)


    horizon : int, optional

        Prediction horizon.

        If None:

            timing["horizon"]


    dt_mpc : float, optional

        Prediction interval.

        If None:

            timing["dt_mpc"]


    Returns
    -------
    preview : dict

        Contains:

            preview["phase_sequence"]

            preview["phase_time_sequence"]

            preview["current_phase_overdue"]
    """


    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    current_phase = normalize_gait_phase(
        current_phase
    )


    t_phase = validate_phase_time(
        t_phase
    )


    if horizon is None:

        horizon = int(
            timing["horizon"]
        )

    else:

        horizon = int(
            horizon
        )


    if dt_mpc is None:

        dt_mpc = float(
            timing["dt_mpc"]
        )

    else:

        dt_mpc = float(
            dt_mpc
        )


    if horizon <= 0:

        raise ValueError(
            "horizon phải lớn hơn 0."
        )


    if (
        not np.isfinite(
            dt_mpc
        )
        or
        dt_mpc <= 0.0
    ):

        raise ValueError(
            "dt_mpc phải finite và lớn hơn 0."
        )


    # ========================================================
    # 2. CURRENT NOMINAL PHASE DURATION
    # ========================================================

    current_phase_duration = (
        get_nominal_phase_duration(
            current_phase,
            timing
        )
    )


    current_phase_overdue = (
        t_phase
        >
        current_phase_duration
    )


    # ========================================================
    # 3. PREVIEW STATE
    #
    # If actual phase is overdue, do NOT propagate the
    # accumulated overdue time through multiple future phases.
    #
    # Clamp it to the nominal phase duration.
    #
    # Actual phase at j=0 is still unchanged.
    # ========================================================

    preview_phase = (
        current_phase
    )


    preview_phase_time = min(
        t_phase,
        current_phase_duration
    )


    phase_sequence = []


    phase_time_sequence = np.zeros(
        horizon,
        dtype=float
    )


    # ========================================================
    # 4. BUILD PREVIEW
    # ========================================================

    for j in range(
        horizon
    ):

        # ----------------------------------------------------
        # j = 0:
        #
        # use ACTUAL synchronized current phase.
        # ----------------------------------------------------

        if j == 0:

            phase_sequence.append(
                preview_phase
            )


            phase_time_sequence[j] = (
                preview_phase_time
            )


            continue


        # ----------------------------------------------------
        # Move one MPC prediction interval forward.
        # ----------------------------------------------------

        preview_phase_time += (
            dt_mpc
        )


        # ----------------------------------------------------
        # Advance nominal phases if a boundary is crossed.
        #
        # The while loop also supports cases where dt_mpc
        # is larger than one phase duration.
        # ----------------------------------------------------

        while True:

            T_phase = (
                get_nominal_phase_duration(
                    preview_phase,
                    timing
                )
            )


            if (
                preview_phase_time
                <
                T_phase
                -
                1e-12
            ):

                break


            preview_phase_time -= (
                T_phase
            )


            # Avoid tiny negative floating-point residue.
            if (
                preview_phase_time
                <
                0.0
                and
                abs(
                    preview_phase_time
                )
                <=
                1e-12
            ):

                preview_phase_time = 0.0


            preview_phase = (
                get_next_gait_phase(
                    preview_phase
                )
            )


        # ----------------------------------------------------
        # Store preview state
        # ----------------------------------------------------

        phase_sequence.append(
            preview_phase
        )


        phase_time_sequence[j] = (
            preview_phase_time
        )


    # ========================================================
    # 5. OUTPUT
    # ========================================================

    preview = {

        "phase_sequence":
            phase_sequence,

        "phase_time_sequence":
            phase_time_sequence,

        "current_phase_overdue":
            current_phase_overdue,
    }


    return preview


# ============================================================
# BUILD MPC CONTACT SCHEDULE
# ============================================================

def build_contact_schedule(
    current_phase,
    t_phase,
    timing,
    horizon=None,
    dt_mpc=None
):
    """
    Build MPC contact schedule over the prediction horizon.

    Output convention
    -----------------

        contact_schedule[j, 0]
            =
        left-foot contact


        contact_schedule[j, 1]
            =
        right-foot contact


    Contact states
    --------------

        LSS
            -> [1, 0]


        DS_L_TO_R
            -> [1, 1]


        RSS
            -> [0, 1]


        DS_R_TO_L
            -> [1, 1]


    Parameters
    ----------
    current_phase : str

        Current synchronized actual gait phase.


    t_phase : float

        Current actual phase elapsed time [s].


    timing : dict

        Output of:

            build_gait_timing(...)


    horizon : int, optional

        MPC prediction horizon.


    dt_mpc : float, optional

        MPC prediction interval.


    Returns
    -------
    contact_schedule : np.ndarray, shape (N, 2)

        MPC contact schedule.
    """


    preview = build_phase_preview(

        current_phase=current_phase,

        t_phase=t_phase,

        timing=timing,

        horizon=horizon,

        dt_mpc=dt_mpc
    )


    phase_sequence = (
        preview[
            "phase_sequence"
        ]
    )


    N = len(
        phase_sequence
    )


    contact_schedule = np.zeros(
        (
            N,
            2
        ),
        dtype=int
    )


    for j, phase in enumerate(
        phase_sequence
    ):

        contact_schedule[
            j,
            :
        ] = get_phase_contact_state(
            phase
        )


    return contact_schedule