from dataclasses import dataclass

import numpy as np


# ============================================================
# STARTUP STAGES
# ============================================================

STAGE_SETTLE = "SETTLE"

STAGE_INITIAL_DS = "INITIAL_DS"

STAGE_COMPLETE = "COMPLETE"

STAGE_FAULT = "FAULT"


VALID_STARTUP_STAGES = (
    STAGE_SETTLE,
    STAGE_INITIAL_DS,
    STAGE_COMPLETE,
    STAGE_FAULT,
)


# ============================================================
# FIRST SWING FOOT
# ============================================================

VALID_SWING_FEET = (
    "L",
    "R",
)


# ============================================================
# STARTUP MAPPING
#
# IMPORTANT
# ---------
#
# First swing RIGHT:
#
#     home DS
#         ->
#     DS_R_TO_L
#         ->
#     LO_R
#         ->
#     LSS
#
#
# First swing LEFT:
#
#     home DS
#         ->
#     DS_L_TO_R
#         ->
#     LO_L
#         ->
#     RSS
#
# ============================================================

_STARTUP_MAPPING = {

    "R": {

        "initial_ds_phase":
            "DS_R_TO_L",

        "expected_event":
            "LO_R",

        "handoff_phase":
            "LSS",

        "support_foot":
            "L",
    },


    "L": {

        "initial_ds_phase":
            "DS_L_TO_R",

        "expected_event":
            "LO_L",

        "handoff_phase":
            "RSS",

        "support_foot":
            "R",
    },
}


# ============================================================
# STATE
# ============================================================

@dataclass
class WalkingStartupState:
    """
    Persistent state of walking startup.

    Startup sequence
    ----------------

        SETTLE
            |
            v
        INITIAL_DS
            |
            | confirmed first liftoff
            v
        COMPLETE
            |
            v
        cyclic gait_manager


    This module does NOT generate the physical swing motion.

    During INITIAL_DS the higher-level walking controller
    receives:

        first_swing_foot

        initial_ds_phase

        should_command_first_swing = True

    and is responsible for commanding the first physical
    liftoff.

    This module only confirms that the expected contact
    transition really occurred.
    """

    stage: str

    first_swing_foot: str

    support_foot: str

    initial_ds_phase: str

    expected_event: str

    handoff_phase: str


    # --------------------------------------------------------
    # Startup timing
    # --------------------------------------------------------

    settle_time: float

    liftoff_timeout: float

    t_settle: float

    t_initial_ds: float


    # --------------------------------------------------------
    # Previous contact state
    #
    # Required for LO edge detection.
    # --------------------------------------------------------

    previous_c_L: int

    previous_c_R: int


    # --------------------------------------------------------
    # Previous physical foot positions
    #
    # At confirmed liftoff we latch the previous contact
    # location as the realized liftoff position.
    # --------------------------------------------------------

    previous_W_p_L6: np.ndarray

    previous_W_p_R6: np.ndarray


    # --------------------------------------------------------
    # Latched first liftoff position
    # --------------------------------------------------------

    W_p_liftoff: object


    # --------------------------------------------------------
    # Fault
    # --------------------------------------------------------

    fault_reason: object


# ============================================================
# OUTPUT
# ============================================================

@dataclass
class WalkingStartupOutput:
    """
    Output consumed by run_walking.py.
    """

    stage: str

    active: bool

    completed: bool

    faulted: bool


    first_swing_foot: str

    support_foot: str


    # --------------------------------------------------------
    # Directional initial DS
    # --------------------------------------------------------

    initial_ds_phase: str

    expected_event: str


    # --------------------------------------------------------
    # Handoff to cyclic gait manager
    # --------------------------------------------------------

    handoff_ready: bool

    handoff_phase: object


    # --------------------------------------------------------
    # Startup control signals
    #
    # During INITIAL_DS:
    #
    #     should_command_first_swing = True
    #
    # run_walking.py will later use this signal to command
    # the first swing foot.
    # --------------------------------------------------------

    should_command_first_swing: bool

    ready_for_liftoff: bool


    # --------------------------------------------------------
    # Current contact
    # --------------------------------------------------------

    c_L: int

    c_R: int

    contact_mode: str


    # --------------------------------------------------------
    # Current detected event
    # --------------------------------------------------------

    event: object


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    settle_complete: bool

    t_settle: float

    t_initial_ds: float


    # --------------------------------------------------------
    # Realized first liftoff
    # --------------------------------------------------------

    W_p_liftoff: object


    # --------------------------------------------------------
    # Fault
    # --------------------------------------------------------

    fault_reason: object


# ============================================================
# VALIDATE CONTACT
# ============================================================

def _normalize_binary_contact(
    value,
    name,
):

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):

        return int(
            value
        )


    value_float = float(
        value
    )


    if not np.isfinite(
        value_float
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    if value_float not in (
        0.0,
        1.0,
    ):

        raise ValueError(
            f"{name} phải bằng 0 hoặc 1."
        )


    return int(
        value_float
    )


# ============================================================
# VALIDATE FLOAT
# ============================================================

def _validate_nonnegative_float(
    value,
    name,
):

    value = float(
        value
    )


    if not np.isfinite(
        value
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    if value < 0.0:

        raise ValueError(
            f"{name} phải >= 0."
        )


    return value


def _validate_positive_float(
    value,
    name,
):

    value = float(
        value
    )


    if not np.isfinite(
        value
    ):

        raise ValueError(
            f"{name} chứa NaN hoặc Inf."
        )


    if value <= 0.0:

        raise ValueError(
            f"{name} phải > 0."
        )


    return value


# ============================================================
# VALIDATE POSITION
# ============================================================

def _validate_position(
    value,
    name,
):

    value = np.asarray(
        value,
        dtype=float,
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


    return value.copy()


# ============================================================
# FIRST SWING FOOT NORMALIZATION
# ============================================================

def normalize_first_swing_foot(
    first_swing_foot,
):

    first_swing_foot = str(
        first_swing_foot
    ).strip().upper()


    if first_swing_foot not in (
        VALID_SWING_FEET
    ):

        raise ValueError(
            "first_swing_foot phải là 'L' hoặc 'R'."
        )


    return first_swing_foot


# ============================================================
# STARTUP MAPPING
# ============================================================

def get_startup_mapping(
    first_swing_foot,
):

    first_swing_foot = (
        normalize_first_swing_foot(
            first_swing_foot
        )
    )


    return dict(
        _STARTUP_MAPPING[
            first_swing_foot
        ]
    )


# ============================================================
# CONTACT MODE
# ============================================================

def get_startup_contact_mode(
    c_L,
    c_R,
):

    c_L = (
        _normalize_binary_contact(
            c_L,
            "c_L",
        )
    )


    c_R = (
        _normalize_binary_contact(
            c_R,
            "c_R",
        )
    )


    if (
        c_L == 1
        and
        c_R == 1
    ):

        return "DS"


    if (
        c_L == 1
        and
        c_R == 0
    ):

        return "LSS"


    if (
        c_L == 0
        and
        c_R == 1
    ):

        return "RSS"


    return "FLIGHT"


# ============================================================
# LIFTOFF EDGE DETECTION
#
# Only LO events are required by startup.
#
# We deliberately keep startup independent from the cyclic
# contact FSM.
# ============================================================

def detect_startup_liftoff_event(
    previous_c_L,
    previous_c_R,
    current_c_L,
    current_c_R,
):

    previous_c_L = (
        _normalize_binary_contact(
            previous_c_L,
            "previous_c_L",
        )
    )


    previous_c_R = (
        _normalize_binary_contact(
            previous_c_R,
            "previous_c_R",
        )
    )


    current_c_L = (
        _normalize_binary_contact(
            current_c_L,
            "current_c_L",
        )
    )


    current_c_R = (
        _normalize_binary_contact(
            current_c_R,
            "current_c_R",
        )
    )


    LO_L = (

        previous_c_L == 1

        and

        current_c_L == 0
    )


    LO_R = (

        previous_c_R == 1

        and

        current_c_R == 0
    )


    if (
        LO_L
        and
        LO_R
    ):

        return "LO_BOTH"


    if LO_L:

        return "LO_L"


    if LO_R:

        return "LO_R"


    return None


# ============================================================
# BUILD OUTPUT
# ============================================================

def _build_startup_output(
    state,
    c_L,
    c_R,
    event=None,
):

    completed = (

        state.stage

        ==

        STAGE_COMPLETE
    )


    faulted = (

        state.stage

        ==

        STAGE_FAULT
    )


    active = (

        not completed

        and

        not faulted
    )


    settle_complete = (

        state.t_settle

        >=

        state.settle_time
        -
        1e-12
    )


    ready_for_liftoff = (

        state.stage
        ==
        STAGE_INITIAL_DS

        and

        c_L == 1

        and

        c_R == 1
    )


    should_command_first_swing = (

        state.stage

        ==

        STAGE_INITIAL_DS
    )


    if completed:

        handoff_phase = (
            state.handoff_phase
        )

    else:

        handoff_phase = None


    if state.W_p_liftoff is None:

        W_p_liftoff = None

    else:

        W_p_liftoff = (
            state.W_p_liftoff.copy()
        )


    return WalkingStartupOutput(

        stage=(
            state.stage
        ),

        active=(
            active
        ),

        completed=(
            completed
        ),

        faulted=(
            faulted
        ),

        first_swing_foot=(
            state.first_swing_foot
        ),

        support_foot=(
            state.support_foot
        ),

        initial_ds_phase=(
            state.initial_ds_phase
        ),

        expected_event=(
            state.expected_event
        ),

        handoff_ready=(
            completed
        ),

        handoff_phase=(
            handoff_phase
        ),

        should_command_first_swing=(
            should_command_first_swing
        ),

        ready_for_liftoff=(
            ready_for_liftoff
        ),

        c_L=(
            int(
                c_L
            )
        ),

        c_R=(
            int(
                c_R
            )
        ),

        contact_mode=(
            get_startup_contact_mode(
                c_L,
                c_R,
            )
        ),

        event=(
            event
        ),

        settle_complete=(
            settle_complete
        ),

        t_settle=(
            float(
                state.t_settle
            )
        ),

        t_initial_ds=(
            float(
                state.t_initial_ds
            )
        ),

        W_p_liftoff=(
            W_p_liftoff
        ),

        fault_reason=(
            state.fault_reason
        ),
    )


# ============================================================
# SET FAULT
# ============================================================

def _set_startup_fault(
    state,
    reason,
):

    state.stage = (
        STAGE_FAULT
    )


    state.fault_reason = str(
        reason
    )


# ============================================================
# INITIALIZE WALKING STARTUP
# ============================================================

def initialize_walking_startup(
    first_swing_foot,
    settle_time,
    liftoff_timeout,
    c_L,
    c_R,
    W_p_L6,
    W_p_R6,
):
    """
    Initialize walking startup from the physical HOME state.

    Required initial physical condition
    -----------------------------------

        c_L = 1
        c_R = 1

    because the robot starts from double support.


    Parameters
    ----------
    first_swing_foot : {"L", "R"}

        First foot that will leave the ground.


    settle_time : float

        Time for which both feet must remain in contact before
        the first directional DS starts.


    liftoff_timeout : float

        Maximum time allowed in INITIAL_DS while waiting for
        the expected physical liftoff.


    c_L, c_R : int

        Current physical contacts.


    W_p_L6, W_p_R6 : ndarray, shape (3,)

        Current physical foot positions.


    Returns
    -------
    state, output
    """

    first_swing_foot = (
        normalize_first_swing_foot(
            first_swing_foot
        )
    )


    settle_time = (
        _validate_nonnegative_float(
            settle_time,
            "settle_time",
        )
    )


    liftoff_timeout = (
        _validate_positive_float(
            liftoff_timeout,
            "liftoff_timeout",
        )
    )


    c_L = (
        _normalize_binary_contact(
            c_L,
            "c_L",
        )
    )


    c_R = (
        _normalize_binary_contact(
            c_R,
            "c_R",
        )
    )


    W_p_L6 = (
        _validate_position(
            W_p_L6,
            "W_p_L6",
        )
    )


    W_p_R6 = (
        _validate_position(
            W_p_R6,
            "W_p_R6",
        )
    )


    # ========================================================
    # STARTUP MUST BEGIN FROM PHYSICAL DOUBLE SUPPORT
    # ========================================================

    if (
        c_L != 1

        or

        c_R != 1
    ):

        raise ValueError(
            "Walking startup phải bắt đầu từ physical "
            "double support [c_L, c_R] = [1, 1]."
        )


    mapping = (
        get_startup_mapping(
            first_swing_foot
        )
    )


    # ========================================================
    # INITIAL STAGE
    #
    # settle_time = 0:
    #
    #     enter INITIAL_DS immediately.
    # ========================================================

    if settle_time <= 0.0:

        initial_stage = (
            STAGE_INITIAL_DS
        )

    else:

        initial_stage = (
            STAGE_SETTLE
        )


    state = WalkingStartupState(

        stage=(
            initial_stage
        ),

        first_swing_foot=(
            first_swing_foot
        ),

        support_foot=(
            mapping[
                "support_foot"
            ]
        ),

        initial_ds_phase=(
            mapping[
                "initial_ds_phase"
            ]
        ),

        expected_event=(
            mapping[
                "expected_event"
            ]
        ),

        handoff_phase=(
            mapping[
                "handoff_phase"
            ]
        ),

        settle_time=(
            settle_time
        ),

        liftoff_timeout=(
            liftoff_timeout
        ),

        t_settle=0.0,

        t_initial_ds=0.0,

        previous_c_L=(
            c_L
        ),

        previous_c_R=(
            c_R
        ),

        previous_W_p_L6=(
            W_p_L6.copy()
        ),

        previous_W_p_R6=(
            W_p_R6.copy()
        ),

        W_p_liftoff=None,

        fault_reason=None,
    )


    output = (
        _build_startup_output(

            state=(
                state
            ),

            c_L=(
                c_L
            ),

            c_R=(
                c_R
            ),

            event=None,
        )
    )


    return (
        state,
        output,
    )


# ============================================================
# UPDATE WALKING STARTUP
# ============================================================

def update_walking_startup(
    state,
    dt_elapsed,
    c_L,
    c_R,
    W_p_L6,
    W_p_R6,
):
    """
    Advance the walking-startup state machine.

    IMPORTANT
    ---------

    This function does NOT:

        - call MuJoCo
        - call mj_step()
        - solve MPC
        - generate footholds
        - generate swing trajectories
        - command actuators


    Control responsibility
    ----------------------

    During:

        STAGE_INITIAL_DS

    output:

        should_command_first_swing = True

    tells run_walking.py that the higher-level controller may
    begin commanding the selected first swing foot.

    Startup waits until the corresponding REAL contact edge is
    detected.

    Example
    -------

    First swing foot = R:

        [1,1]
          |
          | command first right swing
          |
          v
        physical LO_R
          |
          v
        [1,0]
          |
          v
        startup COMPLETE
          |
          v
        handoff_phase = LSS
    """

    if not isinstance(
        state,
        WalkingStartupState,
    ):

        raise TypeError(
            "state phải là WalkingStartupState."
        )


    dt_elapsed = (
        _validate_nonnegative_float(
            dt_elapsed,
            "dt_elapsed",
        )
    )


    c_L = (
        _normalize_binary_contact(
            c_L,
            "c_L",
        )
    )


    c_R = (
        _normalize_binary_contact(
            c_R,
            "c_R",
        )
    )


    W_p_L6 = (
        _validate_position(
            W_p_L6,
            "W_p_L6",
        )
    )


    W_p_R6 = (
        _validate_position(
            W_p_R6,
            "W_p_R6",
        )
    )


    # ========================================================
    # TERMINAL STATES
    #
    # Once COMPLETE or FAULT is reached, startup no longer
    # changes phase.
    # ========================================================

    if state.stage in (
        STAGE_COMPLETE,
        STAGE_FAULT,
    ):

        state.previous_c_L = (
            c_L
        )

        state.previous_c_R = (
            c_R
        )

        state.previous_W_p_L6 = (
            W_p_L6.copy()
        )

        state.previous_W_p_R6 = (
            W_p_R6.copy()
        )


        output = (
            _build_startup_output(

                state=(
                    state
                ),

                c_L=(
                    c_L
                ),

                c_R=(
                    c_R
                ),

                event=None,
            )
        )


        return (
            state,
            output,
        )


    # ========================================================
    # DETECT CURRENT CONTACT EDGE
    # ========================================================

    event = (
        detect_startup_liftoff_event(

            previous_c_L=(
                state.previous_c_L
            ),

            previous_c_R=(
                state.previous_c_R
            ),

            current_c_L=(
                c_L
            ),

            current_c_R=(
                c_R
            ),
        )
    )


    # ========================================================
    # STAGE 1: SETTLE
    # ========================================================

    if state.stage == (
        STAGE_SETTLE
    ):

        # ----------------------------------------------------
        # During settle, both contacts must remain physically
        # present.
        # ----------------------------------------------------

        if (
            c_L != 1

            or

            c_R != 1
        ):

            _set_startup_fault(

                state,

                (
                    "Contact lost during SETTLE: "
                    f"c_L={c_L}, c_R={c_R}."
                ),
            )


        else:

            state.t_settle = min(

                state.settle_time,

                state.t_settle
                +
                dt_elapsed,
            )


            # ------------------------------------------------
            # SETTLE finished.
            #
            # Enter directional INITIAL_DS.
            # ------------------------------------------------

            if (

                state.t_settle

                >=

                state.settle_time
                -
                1e-12
            ):

                state.stage = (
                    STAGE_INITIAL_DS
                )


                state.t_initial_ds = 0.0


    # ========================================================
    # STAGE 2: INITIAL DIRECTIONAL DOUBLE SUPPORT
    # ========================================================

    elif state.stage == (
        STAGE_INITIAL_DS
    ):

        # ----------------------------------------------------
        # Both feet leaving simultaneously is invalid.
        # ----------------------------------------------------

        if event == "LO_BOTH":

            _set_startup_fault(

                state,

                "Both feet lifted off during INITIAL_DS.",
            )


        # ----------------------------------------------------
        # Wrong foot lifted first.
        # ----------------------------------------------------

        elif (

            event is not None

            and

            event
            !=
            state.expected_event
        ):

            _set_startup_fault(

                state,

                (
                    "Unexpected liftoff event "
                    f"{event}; expected "
                    f"{state.expected_event}."
                ),
            )


        # ----------------------------------------------------
        # EXPECTED FIRST LIFTOFF CONFIRMED
        # ----------------------------------------------------

        elif event == (
            state.expected_event
        ):

            current_contact_mode = (
                get_startup_contact_mode(
                    c_L,
                    c_R,
                )
            )


            # ------------------------------------------------
            # After expected LO, physical contact mode must
            # match the intended first cyclic SS phase.
            #
            # R lift:
            #
            #     [1,0] -> LSS
            #
            # L lift:
            #
            #     [0,1] -> RSS
            # ------------------------------------------------

            if current_contact_mode != (
                state.handoff_phase
            ):

                _set_startup_fault(

                    state,

                    (
                        f"{event} detected but physical "
                        f"contact mode is "
                        f"{current_contact_mode}; expected "
                        f"{state.handoff_phase}."
                    ),
                )


            else:

                # =============================================
                # LATCH REALIZED LIFTOFF POSITION
                #
                # Use the PREVIOUS sample because it represents
                # the final known contacted foot position.
                # =============================================

                if state.first_swing_foot == "R":

                    state.W_p_liftoff = (
                        state.previous_W_p_R6.copy()
                    )


                else:

                    state.W_p_liftoff = (
                        state.previous_W_p_L6.copy()
                    )


                # =============================================
                # STARTUP COMPLETE
                # =============================================

                state.stage = (
                    STAGE_COMPLETE
                )


        # ----------------------------------------------------
        # No liftoff edge.
        # ----------------------------------------------------

        else:

            current_contact_mode = (
                get_startup_contact_mode(
                    c_L,
                    c_R,
                )
            )


            # ------------------------------------------------
            # INITIAL_DS must physically remain DS until a
            # valid LO edge occurs.
            # ------------------------------------------------

            if current_contact_mode != "DS":

                _set_startup_fault(

                    state,

                    (
                        "INITIAL_DS lost physical double "
                        "support without a valid expected "
                        "liftoff edge. "
                        f"Current mode = "
                        f"{current_contact_mode}."
                    ),
                )


            else:

                state.t_initial_ds += (
                    dt_elapsed
                )


                # =============================================
                # TIMEOUT
                #
                # Do NOT fake the liftoff transition.
                # =============================================

                if (

                    state.t_initial_ds

                    >

                    state.liftoff_timeout
                    +
                    1e-12
                ):

                    _set_startup_fault(

                        state,

                        (
                            "First liftoff timeout after "
                            f"{state.t_initial_ds:.6f} s; "
                            f"expected "
                            f"{state.expected_event}."
                        ),
                    )


    else:

        raise RuntimeError(
            f"Unknown walking startup stage: {state.stage}"
        )


    # ========================================================
    # SAVE CURRENT SAMPLE FOR NEXT UPDATE
    # ========================================================

    state.previous_c_L = (
        c_L
    )


    state.previous_c_R = (
        c_R
    )


    state.previous_W_p_L6 = (
        W_p_L6.copy()
    )


    state.previous_W_p_R6 = (
        W_p_R6.copy()
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    output = (
        _build_startup_output(

            state=(
                state
            ),

            c_L=(
                c_L
            ),

            c_R=(
                c_R
            ),

            event=(
                event
            ),
        )
    )


    return (
        state,
        output,
    )